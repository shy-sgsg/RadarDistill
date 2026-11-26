#!/usr/bin/env python3
"""Run the frozen R7-B DPCA component and CPU/CUDA parity audit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d2a_r7_teacher_diagnostics import (
    dpca_component_diagnostics,
    summarize_component_diagnostics,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch, synthesize_scatterer_echoes
from radardistill.teachers.d2a_k2 import k2_response
from radardistill.teachers.d2a_k2_cuda import k2_response_cuda


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _static_scatterer_echo(config: dict[str, Any], count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    platform = np.asarray(config["platform"]["initial_position_xyz_m"], dtype=np.float64)
    height = float(platform[2])
    range_start = float(config["range_grid"]["axis_start_m"])
    range_step = float(config["waveform"]["range_cell_m"])
    cells = np.linspace(8.25, 24.25, count)
    azimuths = np.linspace(88.9, 91.1, count)
    positions = []
    for cell, azimuth in zip(cells, azimuths):
        slant = range_start + cell * range_step
        horizontal = np.sqrt(slant**2 - height**2)
        angle = np.deg2rad(azimuth)
        positions.append(
            platform + [horizontal * np.cos(angle), horizontal * np.sin(angle), -height]
        )
    reflectivity = (
        rng.standard_normal(count) + 1j * rng.standard_normal(count)
    ) / np.sqrt(2.0)
    pulse_times = np.arange(int(config["radar"]["pulse_count"]), dtype=np.float64) / float(
        config["radar"]["prf_hz"]
    )
    return synthesize_scatterer_echoes(
        np.asarray(positions), reflectivity.astype(np.complex128), pulse_times, config
    ).iq


def _combine_parts(parts: list[dict[str, np.ndarray | float]]) -> dict[str, np.ndarray | float]:
    result: dict[str, np.ndarray | float] = {
        "shift_samples": float(parts[0]["shift_samples"])
    }
    for key in parts[0]:
        if key == "shift_samples":
            continue
        result[key] = np.concatenate(
            [np.asarray(part[key]).reshape(-1) for part in parts]
        )
    return result


def _summarize_all(parts: dict[str, list[dict[str, np.ndarray | float]]], seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, (component, component_parts) in enumerate(parts.items()):
        result[component] = summarize_component_diagnostics(
            _combine_parts(component_parts), bootstrap_replicates=2000, seed=seed + index
        )
    return result


def _cuda_parity(iq: np.ndarray, config: dict[str, Any], device: str) -> dict[str, Any]:
    cpu_map = k2_response(iq, config)
    cuda_map = k2_response_cuda(iq, config, device=device, oversampling_factor=1)
    difference = np.abs(cuda_map.astype(np.float64) - cpu_map.astype(np.float64))
    return {
        "shape": list(cuda_map.shape),
        "cpu_dtype": str(cpu_map.dtype),
        "cuda_dtype": str(cuda_map.dtype),
        "max_absolute_difference": float(np.max(difference)),
        "mean_absolute_difference": float(np.mean(difference)),
        "max_total_mass_difference": float(
            np.max(np.abs(cuda_map.sum(axis=(1, 2)) - cpu_map.sum(axis=(1, 2))))
        ),
    }


def run_audit(config_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R7-B requires a visible CUDA GPU; CPU fallback is forbidden")
    torch.cuda.set_device(torch.device(device))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment_config = yaml.safe_load(
        (ROOT / "experiments/D2-A-R7/R7-B/config.yaml").read_text(encoding="utf-8")
    )
    started = time.perf_counter()
    seed_start = int(experiment_config["seed_start"])
    batch_size = int(experiment_config["batch_size"])
    component_names = ("static_clutter", "noise_only", "target_only", "mixed_h0", "mixed_h1")
    condition_parts: dict[str, dict[str, list[dict[str, np.ndarray | float]]]] = {}
    pooled_parts = {name: [] for name in component_names}
    parity_by_condition: dict[str, Any] = {}
    generated_total = 0
    generation_seconds = 0.0
    cpu_cuda_seconds = 0.0
    cuda_seen = False
    torch.cuda.reset_peak_memory_stats(torch.device(device))

    for condition_index, condition in enumerate(experiment_config["diagnostic_conditions"]):
        key = f"snr_{condition['snr_db']:g}_cnr_{condition['cnr_db']:g}"
        condition_parts[key] = {name: [] for name in component_names}
        count_total = int(condition["scenes"])
        for offset in range(0, count_total, batch_size):
            count = min(batch_size, count_total - offset)
            seed = seed_start + int(condition["seed_offset"]) + offset
            generation_started = time.perf_counter()
            batch = generate_d2a_paired_batch(
                count,
                seed,
                float(condition["snr_db"]),
                float(condition["cnr_db"]),
                config,
                scenario_id=str(condition["scenario_id"]),
            )
            generation_seconds += time.perf_counter() - generation_started
            generated_total += count

            inputs = {
                "static_clutter": batch.clutter_iq,
                "noise_only": batch.noise_iq,
                "target_only": batch.target_signal_iq,
                "mixed_h0": batch.iq_h0,
                "mixed_h1": batch.iq_h1,
            }
            for component, iq in inputs.items():
                diagnostic = dpca_component_diagnostics(iq, config)
                condition_parts[key][component].append(diagnostic)
                pooled_parts[component].append(diagnostic)

            if offset == 0:
                cuda_seen = True
                current: dict[str, Any] = {}
                for component in ("static_clutter", "mixed_h0", "mixed_h1"):
                    parity_started = time.perf_counter()
                    current[component] = _cuda_parity(inputs[component], config, device)
                    torch.cuda.synchronize(torch.device(device))
                    cpu_cuda_seconds += time.perf_counter() - parity_started
                parity_by_condition[key] = current
        del batch

    single_echo = _static_scatterer_echo(config, 1, seed_start + 900_001)
    multi_echo = _static_scatterer_echo(config, 4, seed_start + 900_004)
    analytic_static = {
        "one_scatterer": summarize_component_diagnostics(
            dpca_component_diagnostics(single_echo, config), bootstrap_replicates=100, seed=31
        ),
        "four_scatterers": summarize_component_diagnostics(
            dpca_component_diagnostics(multi_echo, config), bootstrap_replicates=100, seed=41
        ),
    }

    by_condition_summary = {
        condition: _summarize_all(parts, int(experiment_config["bootstrap"]["seed"]) + index * 10)
        for index, (condition, parts) in enumerate(condition_parts.items())
    }
    pooled_summary = _summarize_all(
        pooled_parts, int(experiment_config["bootstrap"]["seed"])
    )
    old_metrics = json.loads(
        (ROOT / "experiments/D2-A-E001/metrics.json").read_text(encoding="utf-8")
    )
    old_dpca = old_metrics["dpca_diagnostics"]["B3"]
    new_clutter = pooled_summary["static_clutter"][
        "suppression_db_relative_to_channel_a_median"
    ]
    new_target = pooled_summary["target_only"][
        "aligned_residual_power_ratio_median"
    ]
    torch.cuda.synchronize(torch.device(device))
    free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device(device))
    output = {
        "experiment": "D2-A-R7-B",
        "status": "completed_diagnostic_only",
        "scientific_performance_claim": "NOT_EVALUATED",
        "config_path": "experiments/D2-A-E001/config.yaml",
        "new_scene_count": generated_total,
        "seed_start": seed_start,
        "conditions": by_condition_summary,
        "pooled_components": pooled_summary,
        "analytic_finite_cpi": analytic_static,
        "cpu_cuda_parity": {
            "device": device,
            "by_condition": parity_by_condition,
            "max_absolute_difference": max(
                row[component]["max_absolute_difference"]
                for row in parity_by_condition.values()
                for component in row
            ),
            "tolerance": float(experiment_config["cuda_parity"]["maximum_absolute_difference_tolerance"]),
            "passed": max(
                row[component]["max_absolute_difference"]
                for row in parity_by_condition.values()
                for component in row
            )
            <= float(experiment_config["cuda_parity"]["maximum_absolute_difference_tolerance"]),
        },
        "historical_e001_recomputation": {
            "old_scene_count": int(old_dpca["diagnostic_scenes"]),
            "old_median_clutter_suppression_db": float(
                old_dpca["median_separated_clutter_suppression_db"]
            ),
            "new_median_clutter_suppression_db": float(new_clutter),
            "new_minus_old_suppression_db": float(
                new_clutter - float(old_dpca["median_separated_clutter_suppression_db"])
            ),
            "old_median_unscaled_target_power_ratio": float(
                old_dpca["median_separated_target_retention"]
            ),
            "new_median_unscaled_target_power_ratio": float(new_target),
            "target_ratio_definition": "mean(|DPCA(target)|^2)/mean(|target channel 0|^2); may exceed 1",
        },
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "platform": platform.platform(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(torch.device(device)),
            "cuda_free_bytes_before_end": int(free_bytes),
            "cuda_total_bytes": int(total_bytes),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device(device))),
            "cpu_fallback": False,
        },
        "runtime_seconds": {
            "scene_generation": float(generation_seconds),
            "cpu_cuda_parity_total": float(cpu_cuda_seconds),
            "total": float(time.perf_counter() - started),
        },
        "storage": {"iq_written": False, "rd_maps_written": False},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(output), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/D2-A-E001/config.yaml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/D2-A-R7/R7-B/metrics.json"
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run_audit(args.config, args.output, args.device)
    print(
        json.dumps(
            {
                "status": result["status"],
                "new_scenes": result["new_scene_count"],
                "clutter_ccr_db": result["historical_e001_recomputation"][
                    "new_median_clutter_suppression_db"
                ],
                "cuda_parity": result["cpu_cuda_parity"]["max_absolute_difference"],
                "gpu": result["environment"]["gpu_name"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
