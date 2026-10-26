#!/usr/bin/env python3
"""Staged, reproducible D2-A-E003 execution on the authorized CUDA device."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_e002 import (
    calibrate_student_threshold,
    evaluate_paired_predictions,
    paired_difference_interval,
    summarize_h0_verification,
)
from radardistill.distillation.task_losses import decode_task_outputs
from radardistill.evaluation.physical_consistency import (
    select_scene_ids_by_stratum,
    summarize_prediction_intervention,
    wrap_periodic_difference,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.training.d2a_e002_data import materialize_identifiability_boundary
from radardistill.training.d2a_e002 import (
    _cpu_state_dict,
    _paired_validation,
    make_paired_examples,
    scale_d2a_e002_microbatch_loss,
)
from radardistill.students.physical_coherence import FixedCoherenceFeatures
from radardistill.students.physics_guided_student import PhysicsGuidedTaskStudent
from radardistill.students.task_student import TaskStudent
from radardistill.training.d2a_e003 import (
    build_e003_pair_plan,
    e003_seed_manifest,
    validate_e003_config,
)
from radardistill.training.d2a_e003_data import (
    materialize_e003_h0_split,
    materialize_e003_paired_split,
    validate_e003_split_disjointness,
)
from run_d2a_r4 import _b3_t1_cuda


CONFIG_PATH = ROOT / "experiments/D2-A-E003/config.yaml"
OUT = ROOT / "experiments/D2-A-E003"
PHYSICS_PATH = ROOT / "experiments/D2-A-E001/config.yaml"
R3_PATH = ROOT / "experiments/D2-A-R3/config.yaml"
FROZEN_INPUTS = (
    "docs/task_alignments/D2-A-E003_alignment.md",
    "experiments/D2-A-E003/protocol.md",
    "experiments/D2-A-E003/config.yaml",
    "experiments/D2-A-E003/feature_design.md",
    "experiments/D2-A-E001/config.yaml",
    "experiments/D2-A-R3/config.yaml",
    "src/radardistill/evaluation/physical_consistency.py",
    "src/radardistill/evaluation/d2a_e001.py",
    "src/radardistill/evaluation/d2a_e002.py",
    "src/radardistill/distillation/task_losses.py",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/students/physical_coherence.py",
    "src/radardistill/students/physics_guided_student.py",
    "src/radardistill/students/task_student.py",
    "src/radardistill/training/d2a_e002.py",
    "src/radardistill/training/d2a_e002_data.py",
    "src/radardistill/training/d2a_e003.py",
    "src/radardistill/training/d2a_e003_data.py",
    "scripts/run_d2a_e003.py",
    "scripts/run_d2a_r4.py",
    "tests/test_d2a_e003_protocol.py",
    "tests/test_d2a_e003_runner.py",
    "tests/test_d2a_r4.py",
    "tests/test_physical_consistency.py",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _require_frozen_main() -> str:
    branch = _git("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"E003 requires the authorized main branch, found {branch!r}")
    if subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *FROZEN_INPUTS], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("E003 protocol/code has tracked uncommitted changes")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError("E003 requires the frozen code and protocol to be pushed to origin/main")
    return head


def _nvidia_smi() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    return {
        "returncode": int(query.returncode),
        "query": query.stdout.strip(),
        "stderr": query.stderr.strip(),
        "full": subprocess.run(["nvidia-smi"], text=True, capture_output=True).stdout.strip()
        if query.returncode == 0 else None,
    }


def _environment_record(commit: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "git_sha": commit,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_memory_free_bytes": int(free),
        "cuda_memory_total_bytes": int(total),
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "nvidia_smi": _nvidia_smi(),
    }


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("E003 requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    expected = "NVIDIA GeForce RTX 3050 Laptop GPU"
    actual = torch.cuda.get_device_name(0)
    if actual != expected:
        raise RuntimeError(f"E003 requires {expected}, found {actual}")
    return torch.device("cuda:0")


def _estimated_cache_bytes(config: dict[str, Any]) -> int:
    paired_scenes = (
        int(config["data"]["train"]["paired_base_scenes"])
        + int(config["data"]["validation"]["paired_base_scenes"])
        + int(config["data"]["development"]["paired_base_scenes"])
        + int(config["data"]["exact_overlap_boundary"]["paired_base_scenes"])
    )
    paired_iq = paired_scenes * 2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
    h0_count = (
        int(config["data"]["h0_calibration"]["scenes"])
        + int(config["data"]["h0_verification"]["scenes"])
        + int(config["data"]["c3_h0_verification"]["scenes"])
    )
    h0_iq = h0_count * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
    return int(paired_iq + h0_iq + 3 * 1024**3)  # labels, memmap overhead and working reserve


def run_preflight() -> dict[str, Any]:
    config = _read_yaml(CONFIG_PATH)
    validate_e003_config(config)
    commit = _require_frozen_main()
    device = _require_cuda()
    estimated = _estimated_cache_bytes(config)
    free = shutil.disk_usage(ROOT).free
    reserve_gib = 5.0
    need = estimated + int(reserve_gib * 1024**3)
    if free < need:
        raise RuntimeError(f"insufficient free disk: need {need / 1024**3:.2f} GiB, have {free / 1024**3:.2f} GiB")
    manifest = e003_seed_manifest(config)
    _write_json(OUT / "seed_manifest.json", {
        "git_sha": commit,
        "streams": manifest,
        "unique_chunk_seed_count": sum(len(seeds) for seeds in manifest.values()),
        "non_scene_random_seeds": {
            "training": int(config["training"]["training_seed"]),
            "training_pair_plan": int(config["training"]["pair_plan_seed"]),
            "train_assignment": int(config["data"]["train"]["assignment_seed"]),
            "validation_assignment": int(config["data"]["validation"]["assignment_seed"]),
            "h0_calibration_assignment": int(config["data"]["h0_calibration"]["mixture_assignment_seed"]),
            "h0_verification_assignment": int(config["data"]["h0_verification"]["mixture_assignment_seed"]),
            "c3_h0_assignment": int(config["data"]["c3_h0_verification"]["assignment_seed"]),
            "development_assignment": int(config["data"]["development"]["assignment_seed"]),
            "bootstrap": int(config["statistics"]["bootstrap_seed"]),
            "physical_diagnostics": int(config["statistics"]["physical_diagnostic_seed"]),
        },
        "final_blind_seed_range_untouched": config["data"]["final_blind_reserved_seed_range"],
        "r2_reused_seeds_only_within_matched_intervention_family": True,
    })
    environment = _environment_record(commit)
    _write_json(OUT / "environment.json", environment)
    result = {
        "git_sha": commit,
        "branch": "main",
        "pushed_to_origin_main": True,
        "device": str(device),
        "gpu": environment["gpu_name"],
        "disk_free_bytes_before_generation": int(free),
        "estimated_cache_bytes": estimated,
        "required_reserve_gib": reserve_gib,
        "passed": True,
        "final_blind_generated": False,
        "training_started": False,
    }
    _write_json(OUT / "preflight.json", result)
    return result


def _require_preflight() -> dict[str, Any]:
    path = OUT / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run --stage preflight before E003 diagnostics or data generation")
    result = json.loads(path.read_text(encoding="utf-8"))
    current = _require_frozen_main()
    if not result.get("passed"):
        raise RuntimeError("E003 preflight did not pass")
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", str(result.get("git_sha")), current], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("preflight source commit is not an ancestor of current main")
    if subprocess.run(
        ["git", "diff", "--quiet", str(result["git_sha"]), current, "--", *FROZEN_INPUTS], cwd=ROOT
    ).returncode != 0:
        raise RuntimeError("E003 frozen protocol/code changed after preflight")
    _require_cuda()
    return result


def _make_tone(frequency_hz: float, phases: tuple[float, float, float] = (0.0, 0.0, 0.0), *, dtype: torch.dtype) -> torch.Tensor:
    pulse = torch.arange(130, dtype=torch.float64)
    channels = [
        torch.exp(1j * (2.0 * torch.pi * frequency_hz * pulse / 1300.0 + phase))
        for phase in phases
    ]
    return torch.stack(channels).to(dtype=dtype)[None, :, :, None].expand(1, 3, 130, 32).contiguous()


def _relative_max_error(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    absolute = float((candidate.double() - reference.double()).abs().max().item())
    relative = absolute / max(1.0, float(reference.abs().max().item()))
    return absolute, relative


def _cuda_timing(function, *, warmup: int = 10, repeats: int = 50) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    timings: list[float] = []
    for _ in range(repeats):
        start.record()
        function()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return {
        "mean_ms": float(np.mean(timings)),
        "median_ms": float(np.median(timings)),
        "p95_ms": float(np.quantile(timings, 0.95)),
        "repeats": repeats,
    }


def run_r0_r1() -> dict[str, Any]:
    preflight = _require_preflight()
    config = _read_yaml(CONFIG_PATH)
    device = _require_cuda()
    frontend = FixedCoherenceFeatures().to(device).eval()
    rng = np.random.default_rng(int(config["statistics"]["physical_diagnostic_seed"]))
    real = rng.standard_normal((2, 3, 130, 32))
    imag = rng.standard_normal((2, 3, 130, 32))
    iq64 = torch.from_numpy((real + 1j * imag).astype(np.complex64)).to(device)
    iq128 = iq64.to(torch.complex128)
    with torch.inference_mode():
        reference = frontend(iq128)
        candidate = frontend(iq64)
        absolute_error, relative_error = _relative_max_error(candidate, reference)
        phase = 2.173
        rotated = frontend(iq64 * torch.exp(torch.tensor(1j * phase, dtype=torch.complex64, device=device)))
        phase_abs_error, phase_rel_error = _relative_max_error(rotated, candidate)

    mapping_errors: list[float] = []
    for frequency in np.linspace(-650.0, 650.0, 129, endpoint=False):
        tone = frontend(_make_tone(float(frequency), dtype=torch.complex128).to(device))
        lag1 = torch.complex(tone[:, 0, 0, 0], tone[:, 12, 0, 0])
        estimated = torch.angle(lag1) * (1300.0 / (2.0 * torch.pi))
        mapping_errors.append(abs(float(wrap_periodic_difference(
            float(estimated.item()) - float(frequency), period=1300.0
        ))))

    base = frontend(_make_tone(87.5, dtype=torch.complex128).to(device))
    lag4_alias = frontend(_make_tone(87.5 + 325.0, dtype=torch.complex128).to(device))
    lag16_alias = frontend(_make_tone(87.5 + 81.25, dtype=torch.complex128).to(device))
    lag4_error = float((base[:, 1] - lag4_alias[:, 1]).abs().max().item())
    lag16_error = float((base[:, 2] - lag16_alias[:, 2]).abs().max().item())
    lag1_alias_discriminator = float((base[:, 0] - lag4_alias[:, 0]).abs().max().item())

    phase_pair = (0.2, -0.7, 1.1)
    spatial = frontend(_make_tone(100.0, phase_pair, dtype=torch.complex128).to(device))
    cross01 = torch.complex(spatial[:, 9, 0, 0], spatial[:, 21, 0, 0])
    cross01_expected = torch.exp(torch.tensor(1j * (phase_pair[0] - phase_pair[1]), dtype=torch.complex128, device=device))
    cross_phase_error = float((cross01 - cross01_expected).abs().max().item())

    localized_iq = torch.zeros((1, 3, 130, 32), dtype=torch.complex128, device=device)
    localized_iq[:, :, :, 11] = _make_tone(73.0, dtype=torch.complex128).to(device)[0, :, :, 0]
    localized = frontend(localized_iq)
    locality_leakage = int(torch.count_nonzero(localized[:, :27, :, :11]).item() + torch.count_nonzero(localized[:, :27, :, 12:]).item())

    numeric_extremes: dict[str, Any] = {}
    finite_extremes = True
    for name, amplitude in (("zero", 0.0), ("tiny", 1e-25), ("large", 1e30)):
        sample = _make_tone(41.0, dtype=torch.complex64).to(device) * amplitude
        result = frontend(sample)
        is_finite = bool(torch.isfinite(result).all().item())
        finite_extremes = finite_extremes and is_finite
        numeric_extremes[name] = {
            "finite": is_finite,
            "max_abs_feature": float(result.abs().max().item()),
            "log_scene_power": float(result[0, 27, 0, 0].item()),
        }

    parameter_model = PhysicsGuidedTaskStudent().to(device).eval()
    raw_parameters = sum(parameter.numel() for parameter in parameter_model.student.parameters())
    projection_parameters = parameter_model.projection_parameter_count
    total_parameters = parameter_model.total_parameter_count
    parameter_fraction = (total_parameters - raw_parameters) / raw_parameters
    output_feature_bytes = candidate.numel() * candidate.element_size()

    feature_benchmarks: dict[str, Any] = {}
    for batch_size in (1, 16):
        sample = torch.randn(batch_size, 3, 130, 32, device=device) + 1j * torch.randn(
            batch_size, 3, 130, 32, device=device
        )
        sample = sample.to(torch.complex64)
        torch.cuda.reset_peak_memory_stats(device)
        feature_benchmarks[str(batch_size)] = _cuda_timing(lambda: frontend(sample))
        feature_benchmarks[str(batch_size)]["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    del parameter_model
    torch.cuda.empty_cache()

    phase_invariance_passed = relative_error <= 1e-4 and phase_rel_error <= 1e-4
    gate_passed = bool(
        phase_invariance_passed
        and max(mapping_errors) <= 1e-8
        and lag4_error <= 1e-8
        and lag16_error <= 1e-8
        and lag1_alias_discriminator > 1e-2
        and cross_phase_error <= 1e-8
        and locality_leakage == 0
        and finite_extremes
        and parameter_fraction <= 0.05
    )
    result = {
        "git_sha": preflight["git_sha"],
        "r0": {
            "stable_scene_id_sampler_tests": "covered by tests/test_physical_consistency.py",
            "seed_streams_disjoint": True,
            "historical_R4_results_modified": False,
        },
        "r1": {
            "feature_shape": list(candidate.shape),
            "feature_channel_count": int(candidate.shape[1]),
            "complex64_vs_complex128_reference": {
                "max_absolute_error": absolute_error,
                "relative_max_error": relative_error,
                "pre_registered_relative_limit": 1e-4,
                "passed": relative_error <= 1e-4,
            },
            "common_phase_feature_error": {
                "max_absolute_error": phase_abs_error,
                "relative_max_error": phase_rel_error,
                "passed": phase_rel_error <= 1e-4,
            },
            "lag1_frequency_map_max_circular_error_hz": float(max(mapping_errors)),
            "lag4_alias_max_abs_error": lag4_error,
            "lag16_alias_max_abs_error": lag16_error,
            "lag1_separates_lag4_alias": lag1_alias_discriminator > 1e-2,
            "cross_channel_phase_max_abs_error": cross_phase_error,
            "range_locality_nonzero_leakage_count": locality_leakage,
            "zero_small_large_input": numeric_extremes,
            "parameters": {
                "S0_shared_taskstudent": int(raw_parameters),
                "S2_projection": int(projection_parameters),
                "S2_total": int(total_parameters),
                "fractional_increase": float(parameter_fraction),
            },
            "feature_output_tensor_bytes_batch2": int(output_feature_bytes),
            "cuda_feature_timing": feature_benchmarks,
            "public_phase_mathematical_property_passed": bool(phase_invariance_passed),
        },
        "G1_frontend_passed": gate_passed,
        "training_authorized_by_prerequisite": gate_passed,
        "final_blind_generated": False,
    }
    _write_json(OUT / "r0_r1_metrics.json", result)
    return result


def _generate_in_chunks(
    count: int,
    seed_base: int,
    snr_db: float | np.ndarray,
    cnr_db: float | np.ndarray,
    physics: dict[str, Any],
    *,
    scenario_id: str,
    chunk_size: int,
    seed_stride: int,
    target_velocity_mps: float | np.ndarray | None = None,
) -> SimpleNamespace:
    parts: list[Any] = []
    snr_values = np.full(count, float(snr_db)) if np.isscalar(snr_db) else np.asarray(snr_db)
    cnr_values = np.full(count, float(cnr_db)) if np.isscalar(cnr_db) else np.asarray(cnr_db)
    velocity_values = None if target_velocity_mps is None else (
        np.full(count, float(target_velocity_mps))
        if np.isscalar(target_velocity_mps) else np.asarray(target_velocity_mps)
    )
    for chunk_index, start in enumerate(range(0, count, chunk_size)):
        stop = min(start + chunk_size, count)
        kwargs: dict[str, Any] = {}
        if velocity_values is not None:
            kwargs["target_radial_velocity_mps"] = velocity_values[start:stop]
        parts.append(generate_d2a_paired_batch(
            stop - start,
            int(seed_base) + chunk_index * int(seed_stride),
            snr_values[start:stop],
            cnr_values[start:stop],
            physics,
            scenario_id=scenario_id,
            **kwargs,
        ))
    values = {
        field.name: np.concatenate([np.asarray(getattr(part, field.name)) for part in parts], axis=0)
        for field in fields(parts[0])
    }
    return SimpleNamespace(**values)


def _feature_batch(frontend: FixedCoherenceFeatures, iq: np.ndarray, device: torch.device, batch_size: int = 16) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(iq), batch_size):
            batch = torch.from_numpy(np.asarray(iq[start:start + batch_size])).to(device)
            outputs.append(frontend(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def _feature_separation(h0_features: np.ndarray, h1_features: np.ndarray) -> dict[str, float]:
    delta = h1_features.astype(np.float64) - h0_features.astype(np.float64)
    per_scene = np.sqrt(np.mean(delta**2, axis=(1, 2, 3)))
    h0_flat = h0_features.astype(np.float64).reshape(len(h0_features), -1)
    h1_flat = h1_features.astype(np.float64).reshape(len(h1_features), -1)
    paired_mean_delta = np.mean(h1_flat - h0_flat, axis=0)
    within_variance = 0.5 * (np.var(h0_flat, axis=0).mean() + np.var(h1_flat, axis=0).mean())
    dprime = float(np.linalg.norm(paired_mean_delta) / max(math.sqrt(float(within_variance) * h0_flat.shape[1]), 1e-12))
    norm_h0 = np.linalg.norm(h0_flat, axis=1)
    norm_h1 = np.linalg.norm(h1_flat, axis=1)
    return {
        "n": int(len(per_scene)),
        "paired_feature_increment_rms_mean": float(np.mean(per_scene)),
        "paired_feature_increment_rms_median": float(np.median(per_scene)),
        "paired_feature_increment_rms_p95": float(np.quantile(per_scene, 0.95)),
        "standardized_paired_separation": dprime,
        "feature_norm_auc_diagnostic": float(empirical_auc(norm_h0, norm_h1)),
    }


def _teacher_diagnostic(
    batch: Any,
    physics: dict[str, Any],
    t1_options: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Run frozen B3/T1 reference formulas through the audited CUDA kernel."""
    results: dict[str, Any] = {}
    scored: dict[str, dict[str, dict[str, np.ndarray]]] = {
        method: {split: {name: [] for name in ("score", "range_cell", "doppler_hz")} for split in ("h0", "h1")}
        for method in ("B3", "T1")
    }
    for split in ("h0", "h1"):
        iq = np.asarray(getattr(batch, f"iq_{split}"))
        for start in range(0, len(iq), batch_size):
            outputs = _b3_t1_cuda(iq[start:start + batch_size], physics, t1_options, device)
            for method in ("B3", "T1"):
                for name in ("score", "range_cell", "doppler_hz"):
                    scored[method][split][name].append(
                        np.atleast_1d(np.asarray(outputs[method][name], dtype=np.float64))
                    )
    for method in ("B3", "T1"):
        h0 = {name: np.concatenate(parts) for name, parts in scored[method]["h0"].items()}
        h1 = {name: np.concatenate(parts) for name, parts in scored[method]["h1"].items()}
        range_error = h1["range_cell"] - batch.truth_target_range_cell
        doppler_error = wrap_periodic_difference(h1["doppler_hz"] - batch.truth_target_doppler_hz, period=1300.0)
        results[method] = {
            "auc": float(empirical_auc(h0["score"], h1["score"])),
            "range_rmse_cells": float(np.sqrt(np.mean(range_error**2))),
            "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error**2))),
            "joint_localization_rate": float(np.mean((np.abs(range_error) <= 1.0) & (np.abs(doppler_error) <= 20.0))),
            "thresholded_detection_not_evaluated_here": True,
            "teacher_status": "research_reference_only; not an approved teacher",
        }
    return results


def _metric_row(family: str, scenario: str, condition: str, summary: dict[str, Any]) -> dict[str, Any]:
    return {"family": family, "scenario": scenario, "condition": condition, **summary}


def run_r2() -> dict[str, Any]:
    preflight = _require_preflight()
    r1_path = OUT / "r0_r1_metrics.json"
    if not r1_path.is_file():
        raise RuntimeError("complete and record R0/R1 before running controlled R2 scenes")
    r1 = json.loads(r1_path.read_text(encoding="utf-8"))
    if not r1.get("G1_frontend_passed"):
        raise RuntimeError("R1 front-end correctness gate failed; R2 model/training path is stopped")
    config = _read_yaml(CONFIG_PATH)
    physics = _read_yaml(PHYSICS_PATH)
    r3 = _read_yaml(R3_PATH)
    t1_options = r3["detection"]["t1_global_robust_background"]
    device = _require_cuda()
    frontend = FixedCoherenceFeatures().to(device).eval()
    r2_config = config["data"]["controlled_r2"]
    family_bases = [int(value) for value in r2_config["seed_bases_by_family"]]
    family_names = ("target_scale", "clutter_scale", "doppler_interventions", "regenerated_geometry")
    family_seed_bases = dict(zip(family_names, family_bases, strict=True))
    output_rows: list[dict[str, Any]] = []
    scene_ids: dict[str, dict[str, list[str]]] = {}
    baseline_by_scenario: dict[str, Any] = {}
    g2_by_scenario: dict[str, Any] = {}
    scenario_ids = tuple(r2_config["scenario_ids"])

    for scenario_index, scenario in enumerate(scenario_ids):
        seed_base = family_seed_bases["target_scale"] + scenario_index * 100_000
        batch = _generate_in_chunks(
            int(r2_config["scenes_per_geometry"]), seed_base, r2_config["fixed_snr_db"],
            r2_config["fixed_cnr_db"], physics, scenario_id=scenario,
            chunk_size=int(r2_config["generation_chunk_size"]), seed_stride=int(config["data"]["scene_seed_stride"]),
        )
        scene_ids.setdefault("target_scale", {})[scenario] = batch.base_scene_id.astype(str).tolist()
        f0 = _feature_batch(frontend, batch.iq_h0, device)
        f1 = _feature_batch(frontend, batch.iq_h1, device)
        base_summary = _feature_separation(f0, f1)
        base_summary["mean_target_component_power"] = float(np.mean(np.abs(batch.target_signal_iq.astype(np.complex128)) ** 2))
        base_summary["mean_clutter_power"] = float(np.mean(np.abs(batch.clutter_iq.astype(np.complex128)) ** 2))
        base_summary["mean_noise_power"] = float(np.mean(np.abs(batch.noise_iq.astype(np.complex128)) ** 2))
        base_summary["teacher_references"] = _teacher_diagnostic(
            batch,
            physics,
            t1_options,
            device=device,
            batch_size=int(r2_config["generation_chunk_size"]),
        )
        baseline_by_scenario[scenario] = base_summary
        numerical_floor = float(r1["r1"]["complex64_vs_complex128_reference"]["max_absolute_error"])
        g2_by_scenario[scenario] = {
            "median_feature_increment_rms": base_summary["paired_feature_increment_rms_median"],
            "measured_complex64_numerical_floor_abs": numerical_floor,
            "minimum_required_increment_abs": 10.0 * numerical_floor,
            "passed": base_summary["paired_feature_increment_rms_median"] > 10.0 * numerical_floor,
        }
        output_rows.append(_metric_row("base_observable_h1_vs_h0", scenario, "10dB_snr_10dB_cnr", base_summary))

        for multiplier in map(float, r2_config["target_amplitude_multipliers"]):
            scaled_h1 = batch.iq_h0 + batch.target_signal_iq * multiplier
            scaled_features = _feature_batch(frontend, scaled_h1.astype(np.complex64), device)
            scaled_summary = _feature_separation(f0, scaled_features)
            scaled_summary["target_amplitude_multiplier"] = multiplier
            scaled_summary["target_component_power_ratio"] = multiplier**2
            output_rows.append(_metric_row("target_scale", scenario, f"target_x{multiplier:g}", scaled_summary))

    for scenario_index, scenario in enumerate(scenario_ids):
        seed_base = family_seed_bases["clutter_scale"] + scenario_index * 100_000
        batch = _generate_in_chunks(
            int(r2_config["scenes_per_geometry"]), seed_base, r2_config["fixed_snr_db"],
            r2_config["fixed_cnr_db"], physics, scenario_id=scenario,
            chunk_size=int(r2_config["generation_chunk_size"]), seed_stride=int(config["data"]["scene_seed_stride"]),
        )
        scene_ids.setdefault("clutter_scale", {})[scenario] = batch.base_scene_id.astype(str).tolist()
        for multiplier in map(float, r2_config["clutter_amplitude_multipliers"]):
            h0 = batch.clutter_iq * multiplier + batch.noise_iq
            h1 = h0 + batch.target_signal_iq
            f0 = _feature_batch(frontend, h0.astype(np.complex64), device)
            f1 = _feature_batch(frontend, h1.astype(np.complex64), device)
            summary = _feature_separation(f0, f1)
            summary["clutter_amplitude_multiplier"] = multiplier
            summary["clutter_component_power_ratio"] = multiplier**2
            summary["mean_total_observation_power_h0"] = float(np.mean(np.abs(h0.astype(np.complex128)) ** 2))
            output_rows.append(_metric_row("clutter_scale", scenario, f"clutter_x{multiplier:g}", summary))

    pulse = np.arange(130, dtype=np.float64) / float(config["radar"]["prf_hz"])
    for scenario_index, scenario in enumerate(scenario_ids):
        seed_base = family_seed_bases["doppler_interventions"] + scenario_index * 100_000
        batch = _generate_in_chunks(
            int(r2_config["scenes_per_geometry"]), seed_base, r2_config["fixed_snr_db"],
            r2_config["fixed_cnr_db"], physics, scenario_id=scenario,
            chunk_size=int(r2_config["generation_chunk_size"]), seed_stride=int(config["data"]["scene_seed_stride"]),
        )
        scene_ids.setdefault("doppler_interventions", {})[scenario] = batch.base_scene_id.astype(str).tolist()
        for delta in map(float, r2_config["frequency_shifts_hz"]):
            modulation = np.exp(2j * np.pi * delta * pulse)[None, None, :, None]
            target_only = batch.iq_h0 + batch.target_signal_iq * modulation
            whole_observation = batch.iq_h1 * modulation
            for intervention_name, shifted in (("target_component_only", target_only), ("whole_observation", whole_observation)):
                shifted_features = _feature_batch(frontend, shifted.astype(np.complex64), device)
                base_features = _feature_batch(frontend, batch.iq_h1, device)
                feature_shift = np.sqrt(np.mean((shifted_features.astype(np.float64) - base_features.astype(np.float64)) ** 2, axis=(1, 2, 3)))
                output_rows.append(_metric_row(
                    intervention_name,
                    scenario,
                    f"delta_{delta:+g}Hz",
                    {
                        "n": int(len(feature_shift)),
                        "feature_change_rms_mean": float(np.mean(feature_shift)),
                        "feature_change_rms_p95": float(np.quantile(feature_shift, 0.95)),
                        "intervention_is_regenerated_geometry": False,
                        "truth_doppler_shift_wrapped_hz_mean": float(np.mean(wrap_periodic_difference(batch.truth_target_doppler_hz + delta - batch.truth_target_doppler_hz, period=1300.0))),
                    },
                ))

    for scenario_index, scenario in enumerate(scenario_ids):
        scene_seed_base = family_seed_bases["regenerated_geometry"] + scenario_index * 100_000
        baseline = _generate_in_chunks(
            int(r2_config["scenes_per_geometry"]), scene_seed_base, r2_config["fixed_snr_db"],
            r2_config["fixed_cnr_db"], physics, scenario_id=scenario,
            chunk_size=int(r2_config["generation_chunk_size"]), seed_stride=int(config["data"]["scene_seed_stride"]),
        )
        scene_ids.setdefault("regenerated_geometry", {})[scenario] = baseline.base_scene_id.astype(str).tolist()
        baseline_background = baseline.clutter_iq + baseline.noise_iq
        for velocity in map(float, r2_config["regenerated_target_radial_velocities_mps"]):
            regenerated = _generate_in_chunks(
                int(r2_config["scenes_per_geometry"]), scene_seed_base, r2_config["fixed_snr_db"],
                r2_config["fixed_cnr_db"], physics, scenario_id=scenario,
                chunk_size=int(r2_config["generation_chunk_size"]), seed_stride=int(config["data"]["scene_seed_stride"]),
                target_velocity_mps=velocity,
            )
            background_equal = bool(np.array_equal(baseline.clutter_iq, regenerated.clutter_iq) and np.array_equal(baseline.noise_iq, regenerated.noise_iq))
            if not background_equal:
                raise RuntimeError("regenerated-geometry R2 scenes did not keep paired background/noise")
            h0_features = _feature_batch(frontend, regenerated.iq_h0, device)
            h1_features = _feature_batch(frontend, regenerated.iq_h1, device)
            summary = _feature_separation(h0_features, h1_features)
            summary.update({
                "target_radial_velocity_mps": velocity,
                "truth_doppler_hz_mean": float(np.mean(regenerated.truth_target_doppler_hz)),
                "mean_target_component_power": float(np.mean(np.abs(regenerated.target_signal_iq.astype(np.complex128)) ** 2)),
                "background_and_noise_exactly_paired": background_equal,
                "same_base_scene_id_as_matched_velocity": bool(np.array_equal(baseline.base_scene_id, regenerated.base_scene_id)),
            })
            output_rows.append(_metric_row("regenerated_geometry", scenario, f"velocity_{velocity:+g}mps", summary))

    gate_passed = all(item["passed"] for item in g2_by_scenario.values())
    result = {
        "git_sha": preflight["git_sha"],
        "r2_conditions": baseline_by_scenario,
        "g2_non_degenerate_observable_increment": g2_by_scenario,
        "G2_controlled_information_passed": bool(gate_passed),
        "scene_ids": scene_ids,
        "summary": {
            "target_amplitude_changes_only_target_component": True,
            "clutter_changes_keep_target_and_noise_fixed": True,
            "frequency_interventions_separated": ["target_component_only", "whole_observation", "regenerated_geometry"],
            "teacher_references_are_descriptive_only": True,
            "oracle_components_are_not_student_inputs": True,
        },
        "data_generation_for_model_training_started": False,
        "final_blind_generated": False,
    }
    _write_json(OUT / "controlled_r2.json", result)
    with (OUT / "controlled_r2.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = sorted({key for row in output_rows for key in row})
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    return result


def _load_cached_split(cache_root: Path, split_name: str) -> dict[str, np.ndarray]:
    path = cache_root / split_name
    if not path.is_dir():
        raise FileNotFoundError(f"required E003 cached split is missing: {path}")
    return {item.stem: np.load(item, mmap_mode="r") for item in path.glob("*.npy")}


def _write_scene_manifest(splits: dict[str, dict[str, np.ndarray]]) -> int:
    path = OUT / "scene_manifest.csv"
    fields = (
        "split", "base_scene_id", "course_level", "scenario_id", "snr_db",
        "requested_cnr_db", "truth_range_cell", "truth_doppler_hz",
        "truth_radial_velocity_mps", "truth_azimuth_deg", "truth_amplitude",
    )
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for split_name, arrays in splits.items():
            count = len(arrays["base_scene_id"])
            for index in range(count):
                def value(name: str, fallback: Any = "") -> Any:
                    return arrays[name][index].item() if name in arrays and hasattr(arrays[name][index], "item") else (
                        arrays[name][index] if name in arrays else fallback
                    )
                writer.writerow({
                    "split": split_name,
                    "base_scene_id": str(value("base_scene_id")),
                    "course_level": str(value("course_level", "D" if split_name == "exact_overlap_boundary" else "")),
                    "scenario_id": str(value("scenario_id", "exact_overlap" if split_name == "exact_overlap_boundary" else "")),
                    "snr_db": value("snr_db"),
                    "requested_cnr_db": value("requested_cnr_db"),
                    "truth_range_cell": value("truth_range_cell"),
                    "truth_doppler_hz": value("truth_doppler_hz"),
                    "truth_radial_velocity_mps": value("truth_radial_velocity_mps"),
                    "truth_azimuth_deg": value("truth_azimuth_deg"),
                    "truth_amplitude": value("truth_amplitude", value("potential_target_amplitude")),
                })
                rows += 1
    return rows


def run_generate_data() -> dict[str, Any]:
    preflight = _require_preflight()
    r1 = json.loads((OUT / "r0_r1_metrics.json").read_text(encoding="utf-8"))
    r2 = json.loads((OUT / "controlled_r2.json").read_text(encoding="utf-8"))
    if not r1.get("G1_frontend_passed") or not r2.get("G2_controlled_information_passed"):
        raise RuntimeError("G1/G2 prerequisite gates did not pass; training data generation is stopped")
    config = _read_yaml(CONFIG_PATH)
    physics = _read_yaml(PHYSICS_PATH)
    cache_root = ROOT / config["data"]["cache_root"]
    if cache_root.exists() and any(cache_root.iterdir()):
        existing = OUT / "cache_manifest.json"
        if not existing.is_file():
            raise FileExistsError(f"E003 cache root already contains data without its manifest; preserving it: {cache_root}")
    started = time.perf_counter()
    splits: dict[str, dict[str, np.ndarray]] = {}
    print("[E003] generating paired train split (51,200 scenes)", flush=True)
    splits["train"] = materialize_e003_paired_split(cache_root, "train", config, physics)
    print("[E003] generating paired validation split (5,000 scenes)", flush=True)
    splits["validation"] = materialize_e003_paired_split(cache_root, "validation", config, physics)
    print("[E003] generating independent H0 calibration/verification splits", flush=True)
    levels = config["scenarios"]
    for name in ("h0_calibration", "h0_verification"):
        section = config["data"][name]
        splits[name] = materialize_e003_h0_split(
            cache_root,
            name,
            count=int(section["scenes"]),
            course_levels=levels,
            seed_base=int(section["seed_base"]),
            assignment_seed=int(section["mixture_assignment_seed"]),
            physics_config=physics,
            chunk_size=int(section["generation_chunk_size"]),
            seed_stride=int(config["data"]["scene_seed_stride"]),
        )
    print("[E003] generating dedicated C3 H0 verification (4,096 scenes)", flush=True)
    section = config["data"]["c3_h0_verification"]
    splits["c3_h0_verification"] = materialize_e003_h0_split(
        cache_root,
        "c3_h0_verification",
        count=int(section["scenes"]),
        course_levels=levels,
        seed_base=int(section["seed_base"]),
        assignment_seed=int(section["assignment_seed"]),
        physics_config=physics,
        chunk_size=int(section["generation_chunk_size"]),
        seed_stride=int(config["data"]["scene_seed_stride"]),
        dedicated_c3=True,
    )
    print("[E003] generating paired H1 development split (4,096 scenes)", flush=True)
    splits["development"] = materialize_e003_paired_split(cache_root, "development", config, physics)
    boundary = config["data"]["exact_overlap_boundary"]
    print("[E003] generating exact-overlap descriptive boundary split (512 scenes)", flush=True)
    splits["exact_overlap_boundary"] = materialize_identifiability_boundary(
        cache_root,
        "exact_overlap_boundary",
        int(boundary["paired_base_scenes"]),
        physics,
        snr_db=float(boundary["snr_db"]),
        cnr_db=float(boundary["cnr_db"]),
        seed_base=int(boundary["seed_base"]),
        chunk_size=int(boundary["generation_chunk_size"]),
        chunk_seed_stride=int(config["data"]["scene_seed_stride"]),
    )
    counts = validate_e003_split_disjointness(splits)
    scene_manifest_rows = _write_scene_manifest(splits)
    cache_files = [
        {"path": path.relative_to(cache_root).as_posix(), "size_bytes": int(path.stat().st_size)}
        for path in sorted(cache_root.rglob("*"))
        if path.is_file()
    ]
    _write_json(OUT / "cache_manifest.json", {
        "experiment": "D2-A-E003",
        "cache_root": config["data"]["cache_root"],
        "files": cache_files,
        "total_bytes": int(sum(item["size_bytes"] for item in cache_files)),
        "cleanup_eligible_files_only": True,
    })
    elapsed = time.perf_counter() - started
    result = {
        "git_sha": preflight["git_sha"],
        "split_counts": counts,
        "all_base_scene_ids_disjoint": True,
        "scene_manifest_rows": scene_manifest_rows,
        "cache_file_count": len(cache_files),
        "cache_bytes": int(sum(item["size_bytes"] for item in cache_files)),
        "generation_seconds": float(elapsed),
        "cache_path": str(cache_root.relative_to(ROOT)),
        "final_blind_generated": False,
    }
    _write_json(OUT / "data_generation.json", result)
    return result


def _task_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    return {**config, "training": {**config["training"], "loss": config["training"]["losses"]}}


def _build_model(
    arm: str,
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    *,
    projection_seed: int,
) -> torch.nn.Module:
    if arm == "S0":
        model: torch.nn.Module = TaskStudent(
            width=int(config["models"]["S0"].get("width", 32)),
            temporal_dilations=(1, 2, 4, 8, 16, 32, 64),
            include_spectrum_head=True,
        )
        model.load_state_dict(initial_state, strict=True)
        return model
    if arm != "S2":
        raise ValueError(f"unknown E003 arm {arm}")
    torch.manual_seed(projection_seed)
    model_s2 = PhysicsGuidedTaskStudent(include_spectrum_head=True)
    model_s2.student.load_state_dict(initial_state, strict=True)
    return model_s2


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    *,
    arm: str,
    update: int,
    microbatch: int,
    validation: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "experiment": "D2-A-E003",
        "protocol_version": 1.1,
        "arm": arm,
        "update": int(update),
        "microbatch_size": int(microbatch),
        "model_state_dict": _cpu_state_dict(model),
        "validation": dict(validation),
        "compute_spectrum_head_retained_for_matched_training_architecture": True,
    }, path)


def _train_arm_attempt(
    *,
    arm: str,
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    pair_plan: np.ndarray,
    microbatch: int,
    output_dir: Path,
) -> dict[str, Any]:
    training = config["training"]
    effective_batch = int(training["effective_batch_size"])
    effective_h1 = int(training["paired_base_scenes_per_update"])
    updates = int(training["optimizer_updates_per_model"])
    if effective_batch % microbatch or microbatch % 2:
        raise ValueError("microbatch must be an even divisor of the effective paired batch")
    if pair_plan.shape != (updates, effective_h1):
        raise ValueError("matched pair plan differs from frozen E003 update budget")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(training["training_seed"]))
    torch.cuda.manual_seed_all(int(training["training_seed"]))
    model = _build_model(arm, config, initial_state, projection_seed=int(training["training_seed"]) + 1).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate_initial"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_config = _task_loss_config(config)
    accumulations = effective_batch // microbatch
    best_validation_loss = float("inf")
    best_validation: dict[str, Any] = {}
    best_update = 0
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    initial_validation = _paired_validation(model, validation_arrays, loss_config, device, microbatch)
    _save_checkpoint(output_dir / "initial_checkpoint.pt", model, arm=arm, update=0, microbatch=microbatch, validation=initial_validation)
    best_validation_loss = float(initial_validation["task_loss"])
    best_validation = initial_validation
    best_update = 0
    _save_checkpoint(output_dir / "best_checkpoint.pt", model, arm=arm, update=0, microbatch=microbatch, validation=initial_validation)

    columns = (
        "update", "arm", "learning_rate", "train_total", "train_presence",
        "train_range_cell", "train_range_offset", "train_doppler", "validation_task_loss",
        "validation_auc", "validation_range_rmse_all_h1_cells", "validation_doppler_rmse_all_h1_hz",
    )
    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerow({
            "update": 0,
            "arm": arm,
            "validation_task_loss": initial_validation["task_loss"],
            "validation_auc": initial_validation["auc"],
            "validation_range_rmse_all_h1_cells": initial_validation["range_rmse_all_h1_cells"],
            "validation_doppler_rmse_all_h1_hz": initial_validation["doppler_rmse_all_h1_hz"],
        })
        stream.flush()
        model.train()
        for update_index, pair_ids in enumerate(pair_plan):
            fraction = update_index / max(updates - 1, 1)
            learning_rate = float(training["learning_rate_final"]) + 0.5 * (
                float(training["learning_rate_initial"]) - float(training["learning_rate_final"])
            ) * (1.0 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = make_paired_examples(train_arrays, pair_ids, include_teacher=False)
            optimizer.zero_grad(set_to_none=True)
            train_terms = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "task_total")}
            for micro_start in range(0, effective_batch, microbatch):
                micro_stop = micro_start + microbatch
                x = torch.from_numpy(batch["iq_ri"][micro_start:micro_stop]).to(device)
                truth = {
                    key: torch.from_numpy(batch[key][micro_start:micro_stop]).to(device)
                    for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
                }
                outputs = model(x, compute_spectrum=False)
                from radardistill.distillation.task_losses import task_loss
                losses = task_loss(outputs, truth, loss_config, teacher_weight=0.0)
                micro_h1 = int(truth["truth_presence"].sum().item())
                scaled = scale_d2a_e002_microbatch_loss(
                    losses,
                    loss_config,
                    microbatch_size=micro_stop - micro_start,
                    effective_batch_size=effective_batch,
                    microbatch_h1=micro_h1,
                    effective_h1=effective_h1,
                    teacher_weight=0.0,
                )
                if not torch.isfinite(scaled):
                    raise FloatingPointError(f"non-finite E003 {arm} loss at update {update_index + 1}")
                scaled.backward()
                sample_fraction = (micro_stop - micro_start) / effective_batch
                h1_fraction = micro_h1 / effective_h1
                train_terms["presence"] += float(losses["presence"].detach().item()) * sample_fraction
                for name in ("range_cell", "range_offset", "doppler"):
                    train_terms[name] += float(losses[name].detach().item()) * h1_fraction
                train_terms["task_total"] += float(scaled.detach().item())
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite E003 {arm} gradient at update {update_index + 1}")
            optimizer.step()
            update = update_index + 1
            row: dict[str, Any] = {
                "update": update,
                "arm": arm,
                "learning_rate": learning_rate,
                "train_total": train_terms["task_total"],
                "train_presence": train_terms["presence"],
                "train_range_cell": train_terms["range_cell"],
                "train_range_offset": train_terms["range_offset"],
                "train_doppler": train_terms["doppler"],
            }
            if update % int(training["validation_interval_updates"]) == 0 or update == updates:
                model.eval()
                validation = _paired_validation(model, validation_arrays, loss_config, device, microbatch)
                row.update({
                    "validation_task_loss": validation["task_loss"],
                    "validation_auc": validation["auc"],
                    "validation_range_rmse_all_h1_cells": validation["range_rmse_all_h1_cells"],
                    "validation_doppler_rmse_all_h1_hz": validation["doppler_rmse_all_h1_hz"],
                })
                _save_checkpoint(output_dir / "last_checkpoint.pt", model, arm=arm, update=update, microbatch=microbatch, validation=validation)
                if float(validation["task_loss"]) < best_validation_loss:
                    best_validation_loss = float(validation["task_loss"])
                    best_validation = validation
                    best_update = update
                    _save_checkpoint(output_dir / "best_checkpoint.pt", model, arm=arm, update=update, microbatch=microbatch, validation=validation)
                model.train()
            writer.writerow(row)
            if update % 20 == 0:
                stream.flush()
            if update % 100 == 0:
                print(f"[E003] {arm} update {update}/{updates}; validation={row.get('validation_task_loss', float('nan'))}", flush=True)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time
    torch.cuda.synchronize(device)
    result = {
        "arm": arm,
        "updates_completed": updates,
        "selected_update": best_update,
        "selected_validation": best_validation,
        "microbatch_size": microbatch,
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": accumulations,
        "training_seconds_including_validation": float(elapsed),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history_path": str(history_path.relative_to(ROOT)),
        "checkpoint_paths": {
            name: str((output_dir / name).relative_to(ROOT))
            for name in ("initial_checkpoint.pt", "best_checkpoint.pt", "last_checkpoint.pt")
        },
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def _train_arm(
    arm: str,
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    plan: np.ndarray,
) -> dict[str, Any]:
    training = config["training"]
    output_root = OUT / "runs" / "seed_101" / arm
    failures: list[dict[str, Any]] = []
    for microbatch in (int(training["microbatch_size_initial"]), *map(int, training["oom_fallbacks"])):
        output_dir = output_root / f"microbatch_{microbatch}"
        try:
            result = _train_arm_attempt(
                arm=arm,
                train_arrays=train_arrays,
                validation_arrays=validation_arrays,
                config=config,
                device=device,
                initial_state=initial_state,
                pair_plan=plan,
                microbatch=microbatch,
                output_dir=output_dir,
            )
            result["prior_oom_attempts"] = failures
            return result
        except torch.cuda.OutOfMemoryError as exc:
            failures.append({"microbatch_size": microbatch, "error": str(exc)[:1000]})
            torch.cuda.empty_cache()
            if microbatch == int(training["oom_fallbacks"][-1]):
                raise RuntimeError("E003 microbatch 4 OOMed; arm stopped without CPU/AMP/budget fallback") from exc
    raise RuntimeError(f"E003 {arm} training did not complete")


def run_training() -> dict[str, Any]:
    preflight = _require_preflight()
    r1 = json.loads((OUT / "r0_r1_metrics.json").read_text(encoding="utf-8"))
    r2 = json.loads((OUT / "controlled_r2.json").read_text(encoding="utf-8"))
    generation = json.loads((OUT / "data_generation.json").read_text(encoding="utf-8"))
    if not r1.get("G1_frontend_passed") or not r2.get("G2_controlled_information_passed"):
        raise RuntimeError("E003 G1/G2 prerequisite gate failed; student training not started")
    if not generation.get("all_base_scene_ids_disjoint"):
        raise RuntimeError("E003 data split audit did not pass")
    config = _read_yaml(CONFIG_PATH)
    device = _require_cuda()
    cache_root = ROOT / config["data"]["cache_root"]
    train_arrays = _load_cached_split(cache_root, "train")
    validation_arrays = _load_cached_split(cache_root, "validation")
    if len(train_arrays["base_scene_id"]) != 51_200 or len(validation_arrays["base_scene_id"]) != 5_000:
        raise RuntimeError("E003 train/validation scene counts differ from the frozen protocol")
    plan = build_e003_pair_plan(
        train_arrays["base_scene_id"],
        updates=int(config["training"]["optimizer_updates_per_model"]),
        pairs_per_update=int(config["training"]["paired_base_scenes_per_update"]),
        seed=int(config["training"]["pair_plan_seed"]),
    )
    with (OUT / "training_pair_plan.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("update", "slot", "base_scene_id", "course_level"))
        for update, indices in enumerate(plan, start=1):
            for slot, index in enumerate(indices, start=1):
                writer.writerow((update, slot, str(train_arrays["base_scene_id"][index]), str(train_arrays["course_level"][index])))

    seed = int(config["training"]["training_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial_s0 = TaskStudent(include_spectrum_head=True)
    initial_state = _cpu_state_dict(initial_s0)
    del initial_s0
    s0_initial_state = {key: value.clone() for key, value in initial_state.items()}
    s2_initial = _build_model("S2", config, initial_state, projection_seed=seed + 1)
    shared_initial_state_equal = all(
        torch.equal(s2_initial.student.state_dict()[key].cpu(), value)
        for key, value in initial_state.items()
    )
    s2_projection_parameters = s2_initial.projection_parameter_count
    s2_total_parameters = s2_initial.total_parameter_count
    s0_parameter_count = sum(value.numel() for value in initial_state.values())
    del s2_initial
    torch.cuda.empty_cache()
    if not shared_initial_state_equal:
        raise RuntimeError("S0 and S2 shared TaskStudent weights differ at initialization")
    if s2_projection_parameters / s0_parameter_count > 0.05:
        raise RuntimeError("S2 projection exceeds the frozen five-percent parameter-increase limit")

    print("[E003] starting sequential CUDA S0 training", flush=True)
    s0_result = _train_arm("S0", train_arrays, validation_arrays, config, device, s0_initial_state, plan)
    del s0_initial_state
    torch.cuda.empty_cache()
    print("[E003] starting sequential CUDA S2 training", flush=True)
    s2_result = _train_arm("S2", train_arrays, validation_arrays, config, device, initial_state, plan)
    torch.cuda.synchronize(device)
    result = {
        "git_sha": preflight["git_sha"],
        "training_seed": seed,
        "supervision": "physical_truth_only_E1",
        "teacher_losses_used": False,
        "final_blind_generated": False,
        "same_training_scene_multiset_and_order": True,
        "same_effective_batch_and_update_budget": True,
        "same_shared_taskstudent_initialization": shared_initial_state_equal,
        "parameter_counts": {
            "S0": int(s0_parameter_count),
            "S2_total": int(s2_total_parameters),
            "S2_projection": int(s2_projection_parameters),
            "S2_increase_fraction": float((s2_total_parameters - s0_parameter_count) / s0_parameter_count),
        },
        "S0": s0_result,
        "S2": s2_result,
    }
    _write_json(OUT / "training.json", result)
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple, np.ndarray)) else _json_safe(value)
                for key, value in row.items()
            })


def _predict_iq(
    model: torch.nn.Module,
    iq: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("E003 prediction input must be complex [B,3,130,32] IQ")
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("score", "range_cell", "doppler_hz")}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = np.asarray(values[start:start + batch_size], dtype=np.complex64)
            iq_ri = np.concatenate((batch.real, batch.imag), axis=1).astype(np.float32, copy=False)
            tensor = torch.from_numpy(np.ascontiguousarray(iq_ri)).to(device)
            outputs = model(tensor, compute_spectrum=False)
            decoded = decode_task_outputs(outputs, 1300.0)
            parts["score"].append(outputs["presence_logit"].float().cpu().numpy().reshape(-1))
            parts["range_cell"].append(decoded["range_cell"].float().cpu().numpy().reshape(-1))
            parts["doppler_hz"].append(decoded["doppler_hz"].float().cpu().numpy().reshape(-1))
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def _load_evaluation_model(arm: str, checkpoint_path: Path, config: dict[str, Any]) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("experiment") != "D2-A-E003" or payload.get("arm") != arm:
        raise ValueError(f"checkpoint identity does not match E003 {arm}: {checkpoint_path}")
    if arm == "S0":
        model: torch.nn.Module = TaskStudent(include_spectrum_head=True)
    elif arm == "S2":
        model = PhysicsGuidedTaskStudent(include_spectrum_head=True)
    else:
        raise ValueError(f"unknown E003 arm {arm}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _evaluate_paired(
    arrays: dict[str, np.ndarray],
    prediction: dict[str, dict[str, np.ndarray]],
    threshold: float,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return evaluate_paired_predictions(
        h0_scores=prediction["h0"]["score"],
        h1_scores=prediction["h1"]["score"],
        threshold=threshold,
        truth_range=np.asarray(arrays["truth_range_cell"], dtype=np.float64),
        predicted_range=prediction["h1"]["range_cell"],
        truth_doppler=np.asarray(arrays["truth_doppler_hz"], dtype=np.float64),
        predicted_doppler=prediction["h1"]["doppler_hz"],
        base_scene_ids=np.asarray(arrays["base_scene_id"]).astype(str),
        prf_hz=1300.0,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def _summarize_h0_groups(
    scores: np.ndarray,
    arrays: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    result = {"overall": summarize_h0_verification(scores, threshold)}
    for field in ("course_level", "scenario_id"):
        labels = np.asarray(arrays[field]).astype(str)
        result[f"by_{field}"] = {
            label: summarize_h0_verification(scores[labels == label], threshold)
            for label in sorted(np.unique(labels))
        }
    if "snr_db" in arrays:
        snr = np.asarray(arrays["snr_db"], dtype=np.float64)
        result["by_snr_db"] = {
            f"{value:g}": summarize_h0_verification(scores[np.isclose(snr, value)], threshold)
            for value in sorted(np.unique(snr))
        }
    if "requested_cnr_db" in arrays:
        cnr = np.asarray(arrays["requested_cnr_db"], dtype=np.float64)
        result["by_requested_cnr_db"] = {
            f"{value:g}": summarize_h0_verification(scores[np.isclose(cnr, value)], threshold)
            for value in sorted(np.unique(cnr))
        }
    return result


def _roc_rows(negative: np.ndarray, positive: np.ndarray, *, arm: str, split: str) -> list[dict[str, Any]]:
    scores = np.concatenate((np.asarray(negative, dtype=np.float64), np.asarray(positive, dtype=np.float64)))
    quantiles = np.quantile(scores, np.linspace(0.0, 1.0, 201))
    thresholds = np.unique(np.concatenate(([-np.inf], quantiles, [np.inf])))
    return [
        {
            "arm": arm,
            "split": split,
            "threshold": float(threshold),
            "false_positive_rate": float(np.mean(negative >= threshold)),
            "true_positive_rate": float(np.mean(positive >= threshold)),
        }
        for threshold in thresholds
    ]


def _save_prediction_archive(
    arm: str,
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    threshold: float,
) -> Path:
    path = OUT / f"predictions_{arm}.npz"
    flat: dict[str, np.ndarray] = {"calibrated_threshold": np.asarray(threshold, dtype=np.float64)}
    for split, values in predictions.items():
        for label, fields_value in values.items():
            for name, array in fields_value.items():
                flat[f"{split}_{label}_{name}"] = np.asarray(array)
    np.savez_compressed(path, **flat)
    return path


def _write_scene_prediction_rows(
    arm: str,
    threshold: float,
    split_data: dict[str, dict[str, Any]],
) -> Path:
    rows: list[dict[str, Any]] = []
    for split, item in split_data.items():
        arrays = item["arrays"]
        count = len(item["prediction"]["h0"]["score"])
        course = np.asarray(arrays["course_level"]).astype(str) if "course_level" in arrays else np.full(count, "")
        scenario = np.asarray(arrays["scenario_id"]).astype(str) if "scenario_id" in arrays else np.full(count, "")
        snr = np.asarray(arrays["snr_db"], dtype=np.float64) if "snr_db" in arrays else np.full(count, np.nan)
        if "h1" not in item["prediction"]:
            scores = item["prediction"]["h0"]["score"]
            for index, score in enumerate(scores):
                rows.append({
                    "arm": arm,
                    "split": split,
                    "base_scene_id": str(arrays["base_scene_id"][index]),
                    "condition": str(course[index]),
                    "scenario_id": str(scenario[index]),
                    "snr_db": float(snr[index]),
                    "label": 0,
                    "score_logit": float(score),
                    "threshold": float(threshold),
                    "detected_ge": bool(score >= threshold),
                    "truth_range_cell": None,
                    "estimated_range_cell": float(item["prediction"]["h0"]["range_cell"][index]),
                    "truth_doppler_hz": None,
                    "estimated_doppler_hz": float(item["prediction"]["h0"]["doppler_hz"][index]),
                })
            continue
        for label in ("h0", "h1"):
            values = item["prediction"][label]
            for index, score in enumerate(values["score"]):
                is_h1 = label == "h1"
                rows.append({
                    "arm": arm,
                    "split": split,
                    "base_scene_id": str(arrays["base_scene_id"][index]),
                    "condition": str(course[index]),
                    "scenario_id": str(scenario[index]),
                    "snr_db": float(snr[index]),
                    "label": int(is_h1),
                    "score_logit": float(score),
                    "threshold": float(threshold),
                    "detected_ge": bool(score >= threshold),
                    "truth_range_cell": float(arrays["truth_range_cell"][index]) if is_h1 else None,
                    "estimated_range_cell": float(values["range_cell"][index]),
                    "truth_doppler_hz": float(arrays["truth_doppler_hz"][index]) if is_h1 else None,
                    "estimated_doppler_hz": float(values["doppler_hz"][index]),
                })
    path = OUT / f"scene_predictions_{arm}.csv"
    _write_csv(path, rows)
    return path


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))


def _phase_diagnostics(
    model: torch.nn.Module,
    device: torch.device,
    arm: str,
    h0_arrays: dict[str, np.ndarray],
    h1_arrays: dict[str, np.ndarray],
    threshold: float,
    config: dict[str, Any],
    selected_h0_indices: np.ndarray,
    selected_h1_indices: np.ndarray,
) -> dict[str, Any]:
    h0_iq = np.asarray(h0_arrays["iq_h0"][selected_h0_indices])
    h1_iq = np.asarray(h1_arrays["iq_h1"][selected_h1_indices])
    iq = np.concatenate((h0_iq, h1_iq), axis=0)
    count_h0, count_h1 = len(h0_iq), len(h1_iq)
    truth_presence = np.concatenate((np.zeros(count_h0, dtype=bool), np.ones(count_h1, dtype=bool)))
    conditions = np.concatenate((
        np.asarray(h0_arrays["course_level"][selected_h0_indices]).astype(str),
        np.asarray(h1_arrays["course_level"][selected_h1_indices]).astype(str),
    ))
    truth_range = np.concatenate((np.zeros(count_h0), np.asarray(h1_arrays["truth_range_cell"][selected_h1_indices], dtype=np.float64)))
    truth_doppler = np.concatenate((np.zeros(count_h0), np.asarray(h1_arrays["truth_doppler_hz"][selected_h1_indices], dtype=np.float64)))
    baseline = _predict_iq(model, iq, device)
    rotation_results: list[dict[str, Any]] = []
    rotation_count = int(config["statistics"]["phase_rotations"])
    for rotation_index in range(rotation_count):
        angle = 2.0 * np.pi * rotation_index / rotation_count
        if rotation_index == 0:
            intervened = baseline
        else:
            rotated = iq * np.complex64(np.exp(1j * angle))
            intervened = _predict_iq(model, rotated, device)
        summary = summarize_prediction_intervention(
            baseline,
            intervened,
            truth_presence=truth_presence,
            condition=conditions,
            threshold=threshold,
            truth_range_cell=truth_range,
            truth_doppler_hz=truth_doppler,
            prf_hz=1300.0,
        )
        rotation_results.append({"rotation_index": rotation_index, "angle_rad": float(angle), **summary})
    return {
        "arm": arm,
        "sample_counts": {"H0": int(count_h0), "H1": int(count_h1), "per_course_each": int(config["statistics"]["physical_diagnostic_quota_per_course"])},
        "calibration_threshold_reused": float(threshold),
        "rotation_results": rotation_results,
    }


def _frequency_shift_summary(
    baseline: dict[str, np.ndarray],
    changed: dict[str, np.ndarray],
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    threshold: float,
    shift_hz: np.ndarray | float,
) -> dict[str, Any]:
    delta = np.broadcast_to(np.asarray(shift_hz, dtype=np.float64), baseline["score"].shape)
    moved_truth = wrap_periodic_difference(truth_doppler + delta, period=1300.0)
    estimated_delta = wrap_periodic_difference(changed["doppler_hz"] - baseline["doppler_hz"], period=1300.0)
    shift_error = wrap_periodic_difference(estimated_delta - delta, period=1300.0)
    base_detected = baseline["score"] >= threshold
    changed_detected = changed["score"] >= threshold
    base_range_error = baseline["range_cell"] - truth_range
    changed_range_error = changed["range_cell"] - truth_range
    base_doppler_error = wrap_periodic_difference(baseline["doppler_hz"] - truth_doppler, period=1300.0)
    changed_doppler_error = wrap_periodic_difference(changed["doppler_hz"] - moved_truth, period=1300.0)
    base_joint = base_detected & (np.abs(base_range_error) <= 1.0) & (np.abs(base_doppler_error) <= 20.0)
    changed_joint = changed_detected & (np.abs(changed_range_error) <= 1.0) & (np.abs(changed_doppler_error) <= 20.0)
    return {
        "n": int(len(delta)),
        "truth_doppler_shift_hz_mean": float(np.mean(delta)),
        "estimated_doppler_shift_hz_mean": float(np.mean(estimated_delta)),
        "estimated_doppler_shift_error_abs_p95_hz": float(np.quantile(np.abs(shift_error), 0.95)),
        "range_change_abs_p95_cells": float(np.quantile(np.abs(changed["range_cell"] - baseline["range_cell"]), 0.95)),
        "score_abs_change_p95": float(np.quantile(np.abs(changed["score"] - baseline["score"]), 0.95)),
        "detection_agreement": float(np.mean(changed_detected == base_detected)),
        "baseline_range_rmse_cells": _rmse(base_range_error),
        "intervention_range_rmse_cells": _rmse(changed_range_error),
        "baseline_doppler_rmse_hz": _rmse(base_doppler_error),
        "intervention_doppler_rmse_hz": _rmse(changed_doppler_error),
        "baseline_joint_success": float(np.mean(base_joint)),
        "intervention_joint_success": float(np.mean(changed_joint)),
        "wraps_truth_to_prf_interval": True,
    }


def _frequency_diagnostics(
    model: torch.nn.Module,
    device: torch.device,
    arm: str,
    config: dict[str, Any],
    physics: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    r2_config = config["data"]["controlled_r2"]
    seed_bases = list(map(int, r2_config["seed_bases_by_family"]))
    scenario_results: dict[str, Any] = {}
    pulse = np.arange(130, dtype=np.float64) / float(config["radar"]["prf_hz"])
    for scenario_index, scenario in enumerate(r2_config["scenario_ids"]):
        doppler_seed = seed_bases[2] + scenario_index * 100_000
        baseline_batch = _generate_in_chunks(
            int(r2_config["scenes_per_geometry"]), doppler_seed,
            r2_config["fixed_snr_db"], r2_config["fixed_cnr_db"], physics,
            scenario_id=scenario,
            chunk_size=int(r2_config["generation_chunk_size"]),
            seed_stride=int(config["data"]["scene_seed_stride"]),
        )
        baseline = _predict_iq(model, baseline_batch.iq_h1, device)
        scenario_result: dict[str, Any] = {
            "scene_ids": baseline_batch.base_scene_id.astype(str).tolist(),
            "matched_r2_intervention_seed_family": "doppler_interventions",
            "target_background_doppler_separation_hz": {
                "target_mean": float(np.mean(baseline_batch.truth_target_doppler_hz)),
                "target_p05": float(np.quantile(baseline_batch.truth_target_doppler_hz, 0.05)),
                "target_p95": float(np.quantile(baseline_batch.truth_target_doppler_hz, 0.95)),
                "stationary_clutter_model_hz": 0.0,
            },
            "shift_interventions": [],
            "regenerated_geometry": [],
        }
        for shift in map(float, r2_config["frequency_shifts_hz"]):
            modulation = np.exp(2j * np.pi * shift * pulse)[None, None, :, None].astype(np.complex64)
            alternatives = {
                "whole_observation": baseline_batch.iq_h1 * modulation,
                "target_component_only_counterfactual": baseline_batch.iq_h0 + baseline_batch.target_signal_iq * modulation,
            }
            for intervention_name, iq in alternatives.items():
                changed = _predict_iq(model, iq, device)
                scenario_result["shift_interventions"].append({
                    "intervention": intervention_name,
                    "requested_shift_hz": shift,
                    **_frequency_shift_summary(
                        baseline, changed,
                        np.asarray(baseline_batch.truth_target_range_cell, dtype=np.float64),
                        np.asarray(baseline_batch.truth_target_doppler_hz, dtype=np.float64),
                        threshold,
                        shift,
                    ),
                })
        baseline_background = baseline_batch.clutter_iq + baseline_batch.noise_iq
        for velocity in map(float, r2_config["regenerated_target_radial_velocities_mps"]):
            regenerated = _generate_in_chunks(
                int(r2_config["scenes_per_geometry"]), doppler_seed,
                r2_config["fixed_snr_db"], r2_config["fixed_cnr_db"], physics,
                scenario_id=scenario,
                chunk_size=int(r2_config["generation_chunk_size"]),
                seed_stride=int(config["data"]["scene_seed_stride"]),
                target_velocity_mps=velocity,
            )
            paired_background = bool(
                np.array_equal(baseline_background, regenerated.clutter_iq + regenerated.noise_iq)
                and np.array_equal(baseline_batch.base_scene_id, regenerated.base_scene_id)
            )
            if not paired_background:
                raise RuntimeError("E003 regenerated geometry lost the paired background/noise or scene IDs")
            changed = _predict_iq(model, regenerated.iq_h1, device)
            truth_delta = wrap_periodic_difference(
                regenerated.truth_target_doppler_hz - baseline_batch.truth_target_doppler_hz,
                period=1300.0,
            )
            estimated_delta = wrap_periodic_difference(
                changed["doppler_hz"] - baseline["doppler_hz"], period=1300.0
            )
            scenario_result["regenerated_geometry"].append({
                "target_radial_velocity_mps": velocity,
                "background_and_noise_exactly_paired": paired_background,
                "same_scene_ids": True,
                "truth_doppler_shift_hz_mean": float(np.mean(truth_delta)),
                "estimated_doppler_shift_hz_mean": float(np.mean(estimated_delta)),
                "estimated_shift_error_abs_p95_hz": float(np.quantile(np.abs(wrap_periodic_difference(estimated_delta - truth_delta, period=1300.0)), 0.95)),
                "range_change_abs_p95_cells": float(np.quantile(np.abs(changed["range_cell"] - baseline["range_cell"]), 0.95)),
                "detection_agreement": float(np.mean((changed["score"] >= threshold) == (baseline["score"] >= threshold))),
                "intervention_is_regenerated_target_geometry": True,
            })
        scenario_results[str(scenario)] = scenario_result
    return {"arm": arm, "threshold_reused": float(threshold), "scenarios": scenario_results}


def _cuda_timing(function, *, warmup: int = 5, repeats: int = 30) -> dict[str, float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(repeats):
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return {"mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)), "p95_ms": float(np.quantile(samples, 0.95)), "repeats": repeats}


def _benchmark_model(
    model: torch.nn.Module,
    arm: str,
    device: torch.device,
    sample_iq: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"arm": arm, "device": torch.cuda.get_device_name(device), "batch_sizes": {}}
    for batch_size in (1, 16):
        source = np.asarray(sample_iq[:batch_size], dtype=np.complex64)
        if len(source) < batch_size:
            repeats = int(np.ceil(batch_size / len(source)))
            source = np.tile(source, (repeats, 1, 1, 1))[:batch_size]
        cpu_ri = torch.from_numpy(np.ascontiguousarray(np.concatenate((source.real, source.imag), axis=1).astype(np.float32)))
        transfer_times: list[float] = []
        for _ in range(30):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            gpu_ri = cpu_ri.to(device)
            torch.cuda.synchronize(device)
            transfer_times.append((time.perf_counter() - start) * 1000.0)
        torch.cuda.reset_peak_memory_stats(device)
        full = _cuda_timing(lambda: model(gpu_ri, compute_spectrum=False))
        batch_result: dict[str, Any] = {
            "input_to_output": full,
            "cpu_to_cuda_transfer": {
                "mean_ms": float(np.mean(transfer_times)),
                "median_ms": float(np.median(transfer_times)),
                "p95_ms": float(np.quantile(transfer_times, 0.95)),
                "repeats": len(transfer_times),
            },
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
        if arm == "S2":
            iq_tensor = torch.complex(gpu_ri[:, :3], gpu_ri[:, 3:])
            batch_result["fixed_frontend"] = _cuda_timing(lambda: model.frontend(iq_tensor))
            with torch.inference_mode():
                projected_input = model.frontend(iq_tensor)
            batch_result["projection_and_task_network"] = _cuda_timing(
                lambda: model.student(model.projection(projected_input), compute_spectrum=False)
            )
            del projected_input, iq_tensor
        else:
            batch_result["task_network"] = full
        result["batch_sizes"][str(batch_size)] = batch_result
        del gpu_ri
        torch.cuda.empty_cache()
    return result


def _selected_course_indices(arrays: dict[str, np.ndarray], config: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    course = np.asarray(arrays["course_level"]).astype(str)
    ids = np.asarray(arrays["base_scene_id"]).astype(str)
    quota = int(config["statistics"]["physical_diagnostic_quota_per_course"])
    quotas = {name: quota for name in ("C0", "C1", "C2", "C3")}
    selected = select_scene_ids_by_stratum(
        ids, course, quotas, seed=int(config["statistics"]["physical_diagnostic_seed"])
    )
    lookup = {scene_id: index for index, scene_id in enumerate(ids)}
    indices: dict[str, np.ndarray] = {}
    rows: list[dict[str, str]] = []
    for level in ("C0", "C1", "C2", "C3"):
        chosen = selected[level]
        indices[level] = np.asarray([lookup[scene_id] for scene_id in chosen], dtype=np.int64)
        rows.extend({"course_level": level, "base_scene_id": str(scene_id)} for scene_id in chosen)
    return indices, rows


def _paired_differences(
    s2: np.ndarray,
    s0: np.ndarray,
    ids: np.ndarray,
    *,
    seed: int,
    statistic=None,
) -> dict[str, Any]:
    first = np.asarray(s2, dtype=np.float64).reshape(-1)
    second = np.asarray(s0, dtype=np.float64).reshape(-1)
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a - b))
    return {
        "point_difference_s2_minus_s0": float(statistic(first, second)),
        "paired_scene_bootstrap_ci95": list(paired_difference_interval(
            first, second, np.asarray(ids).astype(str),
            statistic=statistic,
            replicates=2000,
            seed=seed,
        )),
    }


def _run_arm_evaluation(
    arm: str,
    model: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    arrays: dict[str, dict[str, np.ndarray]],
    selected_h0_indices: np.ndarray,
    selected_h1_indices: np.ndarray,
    physics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prediction: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    calibration = arrays["h0_calibration"]
    calibration_scores = _predict_iq(model, calibration["iq_h0"], device)["score"]
    threshold_info = calibrate_student_threshold(
        calibration_scores,
        target_pfa=float(config["detection"]["target_pfa_cpi"]),
    )
    threshold = float(threshold_info["threshold"])
    prediction["h0_calibration"] = {"h0": {"score": calibration_scores}}
    prediction["h0_verification"] = {"h0": _predict_iq(model, arrays["h0_verification"]["iq_h0"], device)}
    prediction["c3_h0_verification"] = {"h0": _predict_iq(model, arrays["c3_h0_verification"]["iq_h0"], device)}
    prediction["development"] = {
        "h0": _predict_iq(model, arrays["development"]["iq_h0"], device),
        "h1": _predict_iq(model, arrays["development"]["iq_h1"], device),
    }
    prediction["exact_overlap_boundary"] = {
        "h0": _predict_iq(model, arrays["exact_overlap_boundary"]["iq_h0"], device),
        "h1": _predict_iq(model, arrays["exact_overlap_boundary"]["iq_h1"], device),
    }

    bootstrap_replicates = int(config["statistics"]["bootstrap_replicates"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    h0_verification = _summarize_h0_groups(
        prediction["h0_verification"]["h0"]["score"], arrays["h0_verification"], threshold
    )
    c3_verification = _summarize_h0_groups(
        prediction["c3_h0_verification"]["h0"]["score"], arrays["c3_h0_verification"], threshold
    )
    development = arrays["development"]
    development_prediction = prediction["development"]
    dev_metrics: dict[str, Any] = {
        "overall": _evaluate_paired(
            development, development_prediction, threshold,
            bootstrap_replicates=bootstrap_replicates, bootstrap_seed=bootstrap_seed,
        ),
        "by_course": {},
        "by_scenario": {},
        "by_scenario_snr": {},
    }
    for field, destination in (
        ("course_level", "by_course"),
        ("scenario_id", "by_scenario"),
    ):
        labels = np.asarray(development[field]).astype(str)
        for label in sorted(np.unique(labels)):
            mask = labels == label
            subset = {
                key: np.asarray(development[key])[mask]
                for key in ("base_scene_id", "truth_range_cell", "truth_doppler_hz")
            }
            subset_prediction = {
                name: {field_name: np.asarray(field_value)[mask] for field_name, field_value in values.items()}
                for name, values in development_prediction.items()
            }
            dev_metrics[destination][label] = _evaluate_paired(
                subset, subset_prediction, threshold,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + sum(label.encode("utf-8")),
            )
    courses = np.asarray(development["course_level"]).astype(str)
    scenarios = np.asarray(development["scenario_id"]).astype(str)
    snr_values = np.asarray(development["snr_db"], dtype=np.float64)
    cnr_values = np.asarray(development["requested_cnr_db"], dtype=np.float64)
    for course in sorted(np.unique(courses)):
        for scenario in sorted(np.unique(scenarios[courses == course])):
            course_scenario = (courses == course) & (scenarios == scenario)
            for snr in sorted(np.unique(snr_values[course_scenario])):
                for cnr in sorted(np.unique(cnr_values[course_scenario & np.isclose(snr_values, snr)])):
                    mask = course_scenario & np.isclose(snr_values, snr) & np.isclose(cnr_values, cnr)
                    subset = {
                        key: np.asarray(development[key])[mask]
                        for key in ("base_scene_id", "truth_range_cell", "truth_doppler_hz")
                    }
                    subset_prediction = {
                        name: {field_name: np.asarray(field_value)[mask] for field_name, field_value in values.items()}
                        for name, values in development_prediction.items()
                    }
                    label = f"{course}/{scenario}/SNR{snr:g}/CNR{cnr:g}"
                    dev_metrics["by_scenario_snr"][label] = _evaluate_paired(
                        subset, subset_prediction, threshold,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=bootstrap_seed + sum(label.encode("utf-8")),
                    )
    boundary_metrics = _evaluate_paired(
        arrays["exact_overlap_boundary"], prediction["exact_overlap_boundary"], threshold,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed + 900,
    )
    metrics = {
        "arm": arm,
        "threshold_calibration": threshold_info,
        "h0_verification": h0_verification,
        "c3_h0_verification": c3_verification,
        "development": dev_metrics,
        "exact_overlap_boundary_descriptive_only": boundary_metrics,
        "threshold_comparator": ">=",
        "teacher_oracle_inputs_used_for_prediction": False,
    }

    phase_metrics = _phase_diagnostics(
        model, device, arm,
        arrays["h0_verification"], arrays["development"], threshold, config,
        selected_h0_indices, selected_h1_indices,
    )
    frequency_metrics = _frequency_diagnostics(model, device, arm, config, physics, threshold)
    compute = _benchmark_model(model, arm, device, arrays["development"]["iq_h1"][:16])
    prediction["h0_calibration"] = {"h0": {"score": calibration_scores}}
    return (
        {"threshold": threshold, "predictions": prediction, "metrics": metrics},
        {"phase": phase_metrics, "frequency": frequency_metrics},
        _roc_rows(
            development_prediction["h0"]["score"], development_prediction["h1"]["score"],
            arm=arm, split="development",
        ),
        [
            {"arm": arm, "split": "h0_calibration", "condition": "overall", **threshold_info},
            {"arm": arm, "split": "h0_verification", "condition": "overall", **h0_verification["overall"]},
            *[
                {"arm": arm, "split": "h0_verification", "condition": f"course:{label}", **value}
                for label, value in h0_verification["by_course_level"].items()
            ],
            {"arm": arm, "split": "c3_h0_verification", "condition": "C3_overall", **c3_verification["overall"]},
            *[
                {"arm": arm, "split": "c3_h0_verification", "condition": f"{label}", **value}
                for label, value in c3_verification["by_scenario_id"].items()
            ],
            {"arm": arm, "split": "development", "condition": "overall", **dev_metrics["overall"]},
            *[
                {"arm": arm, "split": "development", "condition": f"course:{label}", **value}
                for label, value in dev_metrics["by_course"].items()
            ],
            *[
                {"arm": arm, "split": "development", "condition": f"scenario:{label}", **value}
                for label, value in dev_metrics["by_scenario"].items()
            ],
            *[
                {"arm": arm, "split": "development", "condition": f"scenario_snr:{label}", **value}
                for label, value in dev_metrics["by_scenario_snr"].items()
            ],
            {"arm": arm, "split": "exact_overlap_boundary", "condition": "D_descriptive_only", **boundary_metrics},
        ],
        [{"arm": arm, **compute}],
    )


def _compare_arms(
    arm_results: dict[str, dict[str, Any]],
    arrays: dict[str, dict[str, np.ndarray]],
    train_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    dev = arrays["development"]
    ids = np.asarray(dev["base_scene_id"]).astype(str)
    truth_range = np.asarray(dev["truth_range_cell"], dtype=np.float64)
    truth_doppler = np.asarray(dev["truth_doppler_hz"], dtype=np.float64)
    course = np.asarray(dev["course_level"]).astype(str)
    pred: dict[str, dict[str, Any]] = {}
    for arm in ("S0", "S2"):
        item = arm_results[arm]
        threshold = float(item["threshold"])
        h0 = item["predictions"]["development"]["h0"]
        h1 = item["predictions"]["development"]["h1"]
        detected = h1["score"] >= threshold
        range_error = h1["range_cell"] - truth_range
        doppler_error = wrap_periodic_difference(h1["doppler_hz"] - truth_doppler, period=1300.0)
        pred[arm] = {
            "pd": detected,
            "joint": detected & (np.abs(range_error) <= 1.0) & (np.abs(doppler_error) <= 20.0),
            "range_error": range_error,
            "doppler_error": doppler_error,
            "threshold": threshold,
            "h0_score": h0["score"],
        }
    c23 = np.isin(course, ("C2", "C3"))
    s0, s2 = pred["S0"], pred["S2"]
    seed = int(config["statistics"]["bootstrap_seed"])
    overall_pd = _paired_differences(s2["pd"], s0["pd"], ids, seed=seed + 11)
    overall_joint = _paired_differences(s2["joint"], s0["joint"], ids, seed=seed + 12)
    c23_joint = _paired_differences(
        s2["joint"][c23], s0["joint"][c23], ids[c23], seed=seed + 13
    )
    c23_range_ratio_stat = lambda a, b: _rmse(a) / max(_rmse(b), 1e-12)
    c23_doppler_ratio_stat = lambda a, b: _rmse(a) / max(_rmse(b), 1e-12)
    c23_range_ratio = _paired_differences(
        s2["range_error"][c23], s0["range_error"][c23], ids[c23],
        seed=seed + 14, statistic=c23_range_ratio_stat,
    )
    c23_doppler_ratio = _paired_differences(
        s2["doppler_error"][c23], s0["doppler_error"][c23], ids[c23],
        seed=seed + 15, statistic=c23_doppler_ratio_stat,
    )
    c3_h0 = arrays["c3_h0_verification"]
    c3_ids = np.asarray(c3_h0["base_scene_id"]).astype(str)
    c3_masks = {
        arm: arm_results[arm]["predictions"]["c3_h0_verification"]["h0"]["score"] >= pred[arm]["threshold"]
        for arm in ("S0", "S2")
    }
    c3_pfa_difference = _paired_differences(
        c3_masks["S2"], c3_masks["S0"], c3_ids, seed=seed + 16
    )
    train_range = np.asarray(train_arrays["truth_range_cell"], dtype=np.float64)
    train_doppler = np.asarray(train_arrays["truth_doppler_hz"], dtype=np.float64)
    constant_range = float(np.median(train_range))
    constant_doppler = float(wrap_periodic_difference(
        np.angle(np.mean(np.exp(2j * np.pi * train_doppler / 1300.0))) * 1300.0 / (2.0 * np.pi),
        period=1300.0,
    ))
    constant_range_error = np.full(len(truth_range), constant_range) - truth_range
    constant_doppler_error = wrap_periodic_difference(constant_doppler - truth_doppler, period=1300.0)
    g3: dict[str, Any] = {}
    for index, arm in enumerate(("S0", "S2")):
        dev_metrics = arm_results[arm]["metrics"]["development"]["overall"]
        range_diff = _paired_differences(
            pred[arm]["range_error"], constant_range_error, ids,
            seed=seed + 30 + index,
            statistic=lambda a, b: _rmse(a) - _rmse(b),
        )
        doppler_diff = _paired_differences(
            pred[arm]["doppler_error"], constant_doppler_error, ids,
            seed=seed + 32 + index,
            statistic=lambda a, b: _rmse(a) - _rmse(b),
        )
        g3[arm] = {
            "auc": dev_metrics["auc"],
            "auc_ci95": dev_metrics["auc_ci95_paired_bootstrap"],
            "range_rmse_difference_vs_truth_constant": range_diff,
            "doppler_rmse_difference_vs_truth_constant": doppler_diff,
            "passed": bool(
                dev_metrics["auc_ci95_paired_bootstrap"][0] > 0.5
                and range_diff["paired_scene_bootstrap_ci95"][1] < 0.0
                and doppler_diff["paired_scene_bootstrap_ci95"][1] < 0.0
            ),
        }
    c3_pfa_s0 = arm_results["S0"]["metrics"]["c3_h0_verification"]["overall"]["pfa_ge"]
    c3_pfa_s2 = arm_results["S2"]["metrics"]["c3_h0_verification"]["overall"]["pfa_ge"]
    gates = config["gates"]["G4_task_protection"]
    protections = {
        "overall_pd_point": overall_pd["point_difference_s2_minus_s0"] >= float(gates["overall_pd_difference_point_min"]),
        "overall_pd_ci_lower": overall_pd["paired_scene_bootstrap_ci95"][0] > float(gates["overall_pd_difference_ci_lower_min"]),
        "overall_joint_point": overall_joint["point_difference_s2_minus_s0"] >= float(gates["overall_joint_difference_point_min"]),
        "overall_joint_ci_lower": overall_joint["paired_scene_bootstrap_ci95"][0] > float(gates["overall_joint_difference_ci_lower_min"]),
        "c2_c3_joint_ci_lower": c23_joint["paired_scene_bootstrap_ci95"][0] > float(gates["C2_C3_joint_difference_ci_lower_min"]),
        "c2_c3_range_rmse_ratio_point": c23_range_ratio["point_difference_s2_minus_s0"] <= float(gates["C2_C3_rmse_ratio_point_max"]),
        "c2_c3_range_rmse_ratio_ci_upper": c23_range_ratio["paired_scene_bootstrap_ci95"][1] <= float(gates["C2_C3_rmse_ratio_ci_upper_max"]),
        "c2_c3_doppler_rmse_ratio_point": c23_doppler_ratio["point_difference_s2_minus_s0"] <= float(gates["C2_C3_rmse_ratio_point_max"]),
        "c2_c3_doppler_rmse_ratio_ci_upper": c23_doppler_ratio["paired_scene_bootstrap_ci95"][1] <= float(gates["C2_C3_rmse_ratio_ci_upper_max"]),
        "c3_h0_pfa_difference_ci_upper": c3_pfa_difference["paired_scene_bootstrap_ci95"][1] <= float(gates["C3_pfa_difference_ci_upper_max"]),
        "s0_c3_h0_pfa_point": c3_pfa_s0 <= float(gates["per_model_C3_pfa_point_max"]),
        "s2_c3_h0_pfa_point": c3_pfa_s2 <= float(gates["per_model_C3_pfa_point_max"]),
    }
    primary = {
        **c23_joint,
        "minimum_gain": float(gates["primary_C2_C3_joint_gain_point_min"]),
        "passed": bool(
            c23_joint["point_difference_s2_minus_s0"] >= float(gates["primary_C2_C3_joint_gain_point_min"])
            and c23_joint["paired_scene_bootstrap_ci95"][0] > float(gates["primary_C2_C3_joint_gain_ci_lower_min"])
        ),
    }
    return {
        "truth_constant_baseline": {
            "range_median_cell": constant_range,
            "doppler_circular_mean_hz": constant_doppler,
            "range_rmse_all_h1_cells": _rmse(constant_range_error),
            "doppler_rmse_all_h1_hz": _rmse(constant_doppler_error),
        },
        "G3_task_learning": g3,
        "G4_task_protection": {"components": protections, "passed": all(protections.values())},
        "G4_primary_C2_C3_joint_gain": primary,
        "paired_comparisons": {
            "overall_pd": overall_pd,
            "overall_joint_success": overall_joint,
            "C2_C3_joint_success": c23_joint,
            "C2_C3_range_rmse_ratio": c23_range_ratio,
            "C2_C3_doppler_rmse_ratio": c23_doppler_ratio,
            "C3_h0_pfa": c3_pfa_difference,
        },
        "C3_h0_pfa_by_arm": {"S0": c3_pfa_s0, "S2": c3_pfa_s2},
        "S2_route_candidate_passed": bool(all(protections.values()) and primary["passed"] and g3["S0"]["passed"] and g3["S2"]["passed"]),
        "bootstrap_unit": "paired base_scene_id; fixed-seed scene uncertainty, not training-seed uncertainty",
    }


def run_evaluate() -> dict[str, Any]:
    preflight = _require_preflight()
    config = _read_yaml(CONFIG_PATH)
    physics = _read_yaml(PHYSICS_PATH)
    training = json.loads((OUT / "training.json").read_text(encoding="utf-8"))
    if training.get("supervision") != "physical_truth_only_E1" or training.get("teacher_losses_used") is not False:
        raise RuntimeError("E003 evaluation requires the completed matched truth-only pilot")
    cache_root = ROOT / config["data"]["cache_root"]
    arrays = {
        name: _load_cached_split(cache_root, name)
        for name in ("h0_calibration", "h0_verification", "c3_h0_verification", "development", "exact_overlap_boundary")
    }
    train_arrays = _load_cached_split(cache_root, "train")
    selected_h0_indices_by_course, h0_rows = _selected_course_indices(
        arrays["h0_verification"], config
    )
    selected_h1_indices_by_course, h1_rows = _selected_course_indices(
        arrays["development"], config
    )
    selected_h0 = np.concatenate([selected_h0_indices_by_course[level] for level in ("C0", "C1", "C2", "C3")])
    selected_h1 = np.concatenate([selected_h1_indices_by_course[level] for level in ("C0", "C1", "C2", "C3")])
    selected_rows = [
        {"split": "h0_verification", **row} for row in h0_rows
    ] + [
        {"split": "development", **row} for row in h1_rows
    ]
    _write_csv(OUT / "selected_scene_ids.csv", selected_rows)

    device = _require_cuda()
    arm_results: dict[str, dict[str, Any]] = {}
    physical_results: dict[str, Any] = {}
    compute_rows: list[dict[str, Any]] = []
    roc_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    scene_prediction_paths: dict[str, str] = {}
    for arm in ("S0", "S2"):
        print(f"[E003] evaluating {arm} on independent H0/H1 splits", flush=True)
        microbatch = int(training[arm]["microbatch_size"])
        selected_checkpoint = OUT / "runs" / "seed_101" / arm / f"microbatch_{microbatch}" / "best_checkpoint.pt"
        if not selected_checkpoint.is_file():
            raise FileNotFoundError(f"selected {arm} checkpoint is missing: {selected_checkpoint}")
        model = _load_evaluation_model(arm, selected_checkpoint, config).to(device).eval()
        arm_data, physical, arm_roc, arm_rows, arm_compute = _run_arm_evaluation(
            arm, model, device, config, arrays, selected_h0, selected_h1, physics
        )
        arm_results[arm] = arm_data
        physical_results[arm] = physical
        compute_rows.extend(arm_compute)
        roc_rows.extend(arm_roc)
        result_rows.extend(arm_rows)
        scene_prediction_paths[arm] = str(_write_scene_prediction_rows(
            arm, arm_data["threshold"], {
                "h0_verification": {"arrays": arrays["h0_verification"], "prediction": arm_data["predictions"]["h0_verification"]},
                "c3_h0_verification": {"arrays": arrays["c3_h0_verification"], "prediction": arm_data["predictions"]["c3_h0_verification"]},
                "development": {"arrays": arrays["development"], "prediction": arm_data["predictions"]["development"]},
                "exact_overlap_boundary": {"arrays": arrays["exact_overlap_boundary"], "prediction": arm_data["predictions"]["exact_overlap_boundary"]},
            }
        ).relative_to(ROOT))
        archive_path = _save_prediction_archive(arm, arm_data["predictions"], arm_data["threshold"])
        arm_data["prediction_archive"] = str(archive_path.relative_to(ROOT))
        arm_data["best_checkpoint"] = str(selected_checkpoint.relative_to(ROOT))
        del model
        torch.cuda.empty_cache()

    comparison = _compare_arms(arm_results, arrays, train_arrays, config)
    metrics = {
        "experiment": "D2-A-E003",
        "git_sha": preflight["git_sha"],
        "supervision": "physical_truth_only_E1",
        "arms": {arm: arm_results[arm]["metrics"] for arm in ("S0", "S2")},
        "comparison_and_gates": comparison,
        "thresholds": {arm: arm_results[arm]["metrics"]["threshold_calibration"] for arm in ("S0", "S2")},
        "scene_prediction_tables": scene_prediction_paths,
        "final_blind_generated": False,
        "distillation_run": False,
    }
    _write_json(OUT / "metrics.json", metrics)
    _write_json(OUT / "physical_consistency.json", {
        "git_sha": preflight["git_sha"],
        "sampling": {
            "quota_per_course": int(config["statistics"]["physical_diagnostic_quota_per_course"]),
            "H0_source": "h0_verification",
            "H1_source": "development",
            "selected_scene_count": int(len(selected_rows)),
            "selected_scene_ids_file": "experiments/D2-A-E003/selected_scene_ids.csv",
        },
        "arms": physical_results,
        "interpretation": "physical diagnostics are descriptive and do not substitute for G3/G4 task gates",
    })
    _write_json(OUT / "compute.json", {
        "git_sha": preflight["git_sha"],
        "benchmark": compute_rows,
        "training": training,
        "r1_fixed_frontend_benchmark": json.loads((OUT / "r0_r1_metrics.json").read_text(encoding="utf-8"))["r1"]["cuda_feature_timing"],
        "realtime_claim": False,
    })
    _write_csv(OUT / "result.csv", result_rows)
    _write_csv(OUT / "roc.csv", roc_rows)
    return {
        "git_sha": preflight["git_sha"],
        "G3": comparison["G3_task_learning"],
        "G4_task_protection": comparison["G4_task_protection"],
        "G4_primary": comparison["G4_primary_C2_C3_joint_gain"],
        "S2_route_candidate_passed": comparison["S2_route_candidate_passed"],
        "final_blind_generated": False,
        "distillation_run": False,
    }


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "NA"
    return f"{numeric:.{digits}g}"


def _make_e003_figures(metrics: dict[str, Any], physical: dict[str, Any]) -> list[str]:
    import matplotlib.pyplot as plt

    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    arms = metrics["arms"]
    courses = ("C0", "C1", "C2", "C3")
    colors = {"S0": "#3569a8", "S2": "#d87928"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for arm in ("S0", "S2"):
        values = arms[arm]["development"]["by_course"]
        axes[0].plot(courses, [values[key]["range_rmse_all_h1_cells"] for key in courses], marker="o", label=arm, color=colors[arm])
        axes[1].plot(courses, [values[key]["doppler_rmse_all_h1_hz"] for key in courses], marker="o", label=arm, color=colors[arm])
    axes[0].set_ylabel("Range RMSE (cells)")
    axes[1].set_ylabel("Circular Doppler RMSE (Hz)")
    for axis in axes:
        axis.set_xlabel("Course condition")
        axis.grid(True, alpha=0.25)
        axis.legend()
    fig.suptitle("D2-A-E003 development error by preregistered course")
    fig.tight_layout()
    path = figure_dir / "rmse_by_course.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(ROOT)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for arm in ("S0", "S2"):
        groups = arms[arm]["development"]["by_scenario_snr"]
        for scenario in ("separated", "near_clutter_edge", "partial_overlap"):
            points = []
            for snr in (0.0, 10.0):
                key = f"C3/{scenario}/SNR{snr:g}/CNR20"
                if key in groups:
                    metric = groups[key]
                    points.append((snr, metric["range_rmse_all_h1_cells"], metric["doppler_rmse_all_h1_hz"]))
            if points:
                label = f"{arm} {scenario}"
                axes[0].plot([row[0] for row in points], [row[1] for row in points], marker="o", color=colors[arm], alpha=0.9)
                axes[1].plot([row[0] for row in points], [row[2] for row in points], marker="o", label=label, color=colors[arm], alpha=0.9)
    axes[0].set_ylabel("Range RMSE (cells)")
    axes[1].set_ylabel("Circular Doppler RMSE (Hz)")
    for axis in axes:
        axis.set_xlabel("C3 SNR (dB); CNR=20 dB")
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("C3 error by SNR and geometry (separate lines are scenario conditions)")
    fig.tight_layout()
    path = figure_dir / "c3_rmse_by_snr.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(ROOT)))

    fig, axis = plt.subplots(figsize=(5, 4.5))
    for arm in ("S0", "S2"):
        arm_rows = [row for row in physical["arms"][arm]["frequency"].get("roc_rows", [])]
        if arm_rows:
            axis.plot([row["false_positive_rate"] for row in arm_rows], [row["true_positive_rate"] for row in arm_rows], label=arm, color=colors[arm])
    # The exact ROC curve is stored in roc.csv; it is plotted below from the compact metric archive.
    if not axis.lines:
        roc_data = []
        with (OUT / "roc.csv").open(encoding="utf-8", newline="") as stream:
            roc_data = list(csv.DictReader(stream))
        for arm in ("S0", "S2"):
            rows = [row for row in roc_data if row["arm"] == arm]
            axis.plot([float(row["false_positive_rate"]) for row in rows], [float(row["true_positive_rate"]) for row in rows], label=arm, color=colors[arm])
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.4)
    axis.set(xlabel="False positive rate", ylabel="Detection probability", title="Development ROC")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "development_roc.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(ROOT)))

    fig, axis = plt.subplots(figsize=(6.5, 4.2))
    for arm in ("S0", "S2"):
        rotations = physical["arms"][arm]["phase"]["rotation_results"]
        axis.plot(
            [row["angle_rad"] for row in rotations],
            [row["H1"]["decision_agreement"] for row in rotations],
            label=arm,
            color=colors[arm],
        )
    axis.set(xlabel="Common phase rotation (rad)", ylabel="H1 decision agreement", ylim=(0, 1.02), title="Public phase consistency")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    path = figure_dir / "phase_decision_agreement.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    outputs.append(str(path.relative_to(ROOT)))
    return outputs


def run_analysis() -> dict[str, Any]:
    preflight_path = OUT / "preflight.json"
    r1_path = OUT / "r0_r1_metrics.json"
    r2_path = OUT / "controlled_r2.json"
    if not (preflight_path.is_file() and r1_path.is_file() and r2_path.is_file()):
        raise RuntimeError("E003 analysis requires completed preflight, R0/R1, and R2 artifacts")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    r1 = json.loads(r1_path.read_text(encoding="utf-8"))
    r2 = json.loads(r2_path.read_text(encoding="utf-8"))
    metrics_path = OUT / "metrics.json"
    training_path = OUT / "training.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else None
    training = json.loads(training_path.read_text(encoding="utf-8")) if training_path.is_file() else None
    physical_path = OUT / "physical_consistency.json"
    physical = json.loads(physical_path.read_text(encoding="utf-8")) if physical_path.is_file() else None
    if not r1.get("G1_frontend_passed") or not r2.get("G2_controlled_information_passed"):
        decision = "STOP"
        rationale = "G1/R1 correctness or G2/R2 observable-information prerequisite failed; no student training was authorized."
    elif metrics is None or training is None:
        raise RuntimeError("E003 prerequisites passed but the authorized matched student pilot or evaluation is missing")
    else:
        comparison = metrics["comparison_and_gates"]
        if comparison["S2_route_candidate_passed"]:
            decision = "CONDITIONAL GO"
            rationale = "The one-seed S2 pilot passed the frozen task-learning, protection and primary-gain screens; only a separately preregistered multi-seed confirmation may be proposed."
        elif not comparison["G3_task_learning"]["S0"]["passed"] or not comparison["G3_task_learning"]["S2"]["passed"]:
            decision = "STOP"
            rationale = "At least one arm failed the frozen task-learning screen against the truth-only constant baseline, so this comparison does not justify formal confirmation or distillation."
        else:
            decision = "STOP"
            rationale = "S2 did not pass the frozen protection and primary-gain conjunction; no formal multi-seed confirmation or distillation is authorized by this pilot."

    figure_paths: list[str] = []
    if metrics is not None and physical is not None:
        figure_paths = _make_e003_figures(metrics, physical)

    lines = [
        "# D2-A-E003 阶段分析",
        "",
        f"- 代码与协议来源提交：`{preflight['git_sha']}`",
        f"- 运行设备：{preflight['gpu']}；数据类型：synthetic；训练：E1 truth-only，未蒸馏。",
        f"- G1/R1 固定前端：**{'PASS' if r1.get('G1_frontend_passed') else 'FAIL'}**；G2/R2 可观测信息：**{'PASS' if r2.get('G2_controlled_information_passed') else 'FAIL'}**。",
        f"- E003 阶段建议：**{decision}**。{rationale}",
        "- 未生成 final blind；单个训练种子的场景 bootstrap 不是训练种子不确定性。",
        "",
        "## R1 数值与物理正确性",
        "",
        f"Complex64 相对最大前向误差为 `{_format_number(r1['r1']['complex64_vs_complex128_reference']['relative_max_error'], 6)}`，预注册限值 `1e-4`；absolute max error 为 `{_format_number(r1['r1']['complex64_vs_complex128_reference']['max_absolute_error'], 6)}`。该容差仅用于相对数值误差，物理映射和任务指标单独验收。",
        f"公共相位误差 `{_format_number(r1['r1']['common_phase_feature_error']['relative_max_error'], 6)}`；Lag-1 Doppler 圆周映射最大误差 `{_format_number(r1['r1']['lag1_frequency_map_max_circular_error_hz'], 6)} Hz`；范围泄漏计数 `{r1['r1']['range_locality_nonzero_leakage_count']}`。",
        "",
        "## R2 可观测信息与传统参考",
        "",
        "|几何|目标-背景特征增量中位数 RMS|10×数值地板|门|",
        "|---|---:|---:|---|",
    ]
    for scenario, item in r2["g2_non_degenerate_observable_increment"].items():
        lines.append(
            f"|{scenario}|{_format_number(item['median_feature_increment_rms'])}|{_format_number(item['minimum_required_increment_abs'])}|{'PASS' if item['passed'] else 'FAIL'}|"
        )
    if r2.get("r2_conditions"):
        lines.extend(["", "R2 中 B3/T1 教师分数与估计仅作描述性参考，并使用已审计的 CUDA 公式；不因此授予教师资格。", ""])
        for scenario, item in r2["r2_conditions"].items():
            lines.append(f"- {scenario}：B3 AUC `{_format_number(item['teacher_references']['B3']['auc'])}`，T1 AUC `{_format_number(item['teacher_references']['T1']['auc'])}`；B3/T1 参数估计与固定检测分数分开记录。")
    if metrics is not None:
        lines.extend([
            "",
            "## S0/S2 任务结果",
            "",
            "|臂|H0 verification Pfa|C3 H0 Pfa|开发 AUC (95% CI)|Pd|Joint success|Range RMSE (cell)|Doppler RMSE (Hz)|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for arm in ("S0", "S2"):
            item = metrics["arms"][arm]
            dev = item["development"]["overall"]
            lines.append(
                f"|{arm}|{_format_number(item['h0_verification']['overall']['pfa_ge'])}|{_format_number(item['c3_h0_verification']['overall']['pfa_ge'])}|"
                f"{_format_number(dev['auc'])} ({_format_number(dev['auc_ci95_paired_bootstrap'][0])}, {_format_number(dev['auc_ci95_paired_bootstrap'][1])})|"
                f"{_format_number(dev['pd'])}|{_format_number(dev['joint_success'])}|"
                f"{_format_number(dev['range_rmse_all_h1_cells'])}|{_format_number(dev['doppler_rmse_all_h1_hz'])}|"
            )
        comparison = metrics["comparison_and_gates"]
        lines.extend([
            "",
            f"主要 C2/C3 Joint success 差 S2−S0：`{_format_number(comparison['G4_primary_C2_C3_joint_gain']['point_difference_s2_minus_s0'])}`，配对 scene-bootstrap 95% CI `{comparison['G4_primary_C2_C3_joint_gain']['paired_scene_bootstrap_ci95']}`。",
            f"G3 S0/S2：`{comparison['G3_task_learning']['S0']['passed']}` / `{comparison['G3_task_learning']['S2']['passed']}`；G4 task protection：`{comparison['G4_task_protection']['passed']}`；主要增益门：`{comparison['G4_primary_C2_C3_joint_gain']['passed']}`。",
            "",
            "## 物理一致性与计算",
            "",
            f"公共相位 32 旋转、目标/整体频移与重新生成速度几何分别见 [`physical_consistency.json`](physical_consistency.json)；CUDA 端到端测量边界见 [`compute.json`](compute.json)。这些描述性诊断不替代 G3/G4。",
        ])
    else:
        lines.extend(["", "## 学生 pilot", "", "由于 G1 或 G2 前置门未通过，没有生成训练数据或运行 S0/S2。"])
    lines.extend([
        "",
        "## 图表",
        "",
        *[f"- [{Path(path).name}](figures/{Path(path).name})" for path in figure_paths],
        "",
        "## 研究解释",
        "",
        "E003 只测试一个固定相干输入表示在冻结合成静止杂波条件下对现有任务学生的作用。S2 的任何改善都可能来自显式前端与可学习投影的组合；本轮没有额外消融拆分两者的独立因果贡献。单 seed pilot 仅筛选路线，不证明跨训练种子泛化，也不支持蒸馏或盲测结论。",
    ])
    analysis_text = "\n".join(lines) + "\n"
    (OUT / "analysis.md").write_text(analysis_text, encoding="utf-8")

    target_sha = _git("rev-parse", "HEAD")
    stage_review = [
        "# D2-A-E003 阶段审查材料",
        "",
        f"- 当前审查目标提交：`{target_sha}`",
        "- 研究总纲：`docs/RESEARCH_CHARTER.md`，版本 v1.2；本阶段未修改。",
        "- 本阶段代码/协议来源提交：`" + str(preflight["git_sha"]) + "`。",
        "- 阶段决定：**" + decision + "**。",
        "- 证据：[`实验分析`](../../experiments/D2-A-E003/analysis.md)、[`指标`](../../experiments/D2-A-E003/metrics.json)、[`R0/R1`](../../experiments/D2-A-E003/r0_r1_metrics.json)、[`R2`](../../experiments/D2-A-E003/controlled_r2.json)、[`CUDA资源`](../../experiments/D2-A-E003/compute.json)。",
        "",
        "请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、总体目标、主要研究内容、核心科学问题和长期技术路线，然后结合这些内容审查本阶段的物理表征、学生结构、训练、代码、数据和研究结论。",
        "",
        "请独立判断：",
        "",
        "1. 本阶段是否推动总纲中的 GMTI 任务级学生目标，核心科学问题是否值得继续研究；",
        "2. 固定 28 通道相干前端、S0/S2 匹配设计、split/seed、训练和统计是否合理，有无泄漏或基线不公平；",
        "3. R1/R2、H0 阈值校准、H1 参数估计、物理一致性和 CUDA 成本结论是否由实际代码与结果支持；",
        "4. 1e-4 complex64 相对容差是否正确限定为数值精度判据，且没有掩盖物理映射或任务误差；",
        "5. 物理前端的贡献能否与可学习投影/TaskStudent贡献区分，本实验的创新边界是否适当；",
        "6. E003 是否有资格提出独立的三种子确认；是否应停止当前表示路线、调整长期技术路线或提出更有价值方向；",
        "7. 研究假设哪些得到支持、否定或仍未决；请给证据驱动的结论和下一步建议。",
        "",
        "本轮没有做知识蒸馏、正式三种子或 final blind。任何研究总纲正文修改都必须先由用户讨论并明确批准。",
        "",
        "审查文件：`docs/RESEARCH_CHARTER.md`、`docs/CURRENT_STATUS.md`、`docs/RESEARCH_ROADMAP.md`、`docs/research/ASSUMPTION_REGISTER.md`、`docs/task_alignments/D2-A-E003_alignment.md`、`experiments/D2-A-E003/protocol.md`、`experiments/D2-A-E003/feature_design.md`、`experiments/D2-A-E003/analysis.md`、`docs/stage_reviews/D2-A-E003_review.md`、相关源代码/测试及本阶段 JSON/CSV。",
    ]
    review_text = "\n".join(stage_review) + "\n"
    (OUT / "review_request.md").write_text(review_text, encoding="utf-8")
    review_path = ROOT / "docs/stage_reviews/D2-A-E003_review.md"
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(review_text, encoding="utf-8")
    return {
        "git_sha": target_sha,
        "decision": decision,
        "analysis_path": str((OUT / "analysis.md").relative_to(ROOT)),
        "stage_review_path": str(review_path.relative_to(ROOT)),
        "review_request_path": str((OUT / "review_request.md").relative_to(ROOT)),
        "figure_paths": figure_paths,
    }


def _project_size_bytes() -> int:
    output = subprocess.run(["du", "-sb", str(ROOT)], cwd=ROOT, check=True, text=True, capture_output=True).stdout
    return int(output.split()[0])


def run_cleanup() -> dict[str, Any]:
    manifest_path = OUT / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("E003-owned cache manifest is required before cache cleanup")
    for required in (OUT / "metrics.json", OUT / "analysis.md", OUT / "review_request.md", ROOT / "docs/stage_reviews/D2-A-E003_review.md"):
        if not required.is_file():
            raise FileNotFoundError(f"E003 review artifacts must be complete before cleanup: {required}")
    config = _read_yaml(CONFIG_PATH)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "D2-A-E003" or manifest.get("cleanup_eligible_files_only") is not True:
        raise ValueError("cache manifest does not authorize only E003-listed files")
    cache_root = (ROOT / config["data"]["cache_root"]).resolve()
    if (ROOT / manifest["cache_root"]).resolve() != cache_root:
        raise ValueError("cache manifest root differs from frozen E003 cache root")
    removed: list[dict[str, Any]] = []
    preserved_unlisted: list[str] = []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        candidate = cache_root / relative
        if candidate.resolve().parent != (cache_root / relative.parent).resolve() or not candidate.resolve().is_relative_to(cache_root):
            raise ValueError(f"cache manifest path escapes E003 cache root: {relative}")
        if not candidate.exists():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"refusing to remove non-regular E003 cache entry: {candidate}")
        actual_size = int(candidate.stat().st_size)
        if actual_size != int(entry["size_bytes"]):
            raise ValueError(f"E003 cache file size changed after manifest: {relative}")
        candidate.unlink()
        removed.append({"path": relative.as_posix(), "size_bytes": actual_size})
    if cache_root.exists():
        for directory in sorted((path for path in cache_root.rglob("*") if path.is_dir()), reverse=True):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            cache_root.rmdir()
        except OSError:
            preserved_unlisted = [str(path.relative_to(cache_root)) for path in cache_root.rglob("*") if path.is_file()]
    size = _project_size_bytes()
    limit = 20 * 1024**3
    result = {
        "experiment": "D2-A-E003",
        "cache_manifest": str(manifest_path.relative_to(ROOT)),
        "removed_file_count": len(removed),
        "removed_bytes": int(sum(item["size_bytes"] for item in removed)),
        "removed_files": removed,
        "unlisted_files_preserved": preserved_unlisted,
        "project_bytes_after_cleanup": size,
        "project_gib_after_cleanup": size / 1024**3,
        "project_limit_gib": 20,
        "under_limit": size < limit,
        "final_blind_generated": False,
        "user_untracked_files_touched": False,
    }
    _write_json(OUT / "cleanup.json", result)
    final_size = _project_size_bytes()
    if final_size >= limit:
        raise RuntimeError(f"project remains at {final_size / 1024**3:.2f} GiB after listed E003 cache removal")
    return {**result, "project_bytes_after_cleanup_record_written": final_size, "under_limit": True}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--stage",
        choices=("preflight", "r0-r1", "r2", "generate-data", "train", "evaluate", "analysis", "cleanup"),
        required=True,
    )
    args = parser.parse_args()
    if args.config.resolve() != CONFIG_PATH.resolve():
        raise ValueError("E003 config path must remain the committed experiments/D2-A-E003/config.yaml")
    if args.stage == "preflight":
        result = run_preflight()
    elif args.stage == "r0-r1":
        result = run_r0_r1()
    elif args.stage == "r2":
        result = run_r2()
    elif args.stage == "generate-data":
        result = run_generate_data()
    elif args.stage == "train":
        result = run_training()
    elif args.stage == "evaluate":
        result = run_evaluate()
    elif args.stage == "analysis":
        result = run_analysis()
    else:
        result = run_cleanup()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
