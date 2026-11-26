#!/usr/bin/env python3
"""Measure streaming D2-A scene generation and cell-scoring throughput.

This is a resource preflight only. Its convenience quantile is measured on the
same tiny batch and must never be used as a calibrated threshold or performance
result. No IQ or range-Doppler cache is written.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.d2a_baselines import run_b3
from radardistill.evaluation.d2a_r7_metrics import dense_h0_summary
from radardistill.simulation.d2a import generate_d2a_paired_batch


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


def run_preflight(
    *, scene_count: int, batch_size: int, seed: int, config_path: Path
) -> dict[str, Any]:
    if scene_count <= 0 or batch_size <= 0 or seed < 0:
        raise ValueError("scene_count and batch_size must be positive; seed nonnegative")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    radar = config["radar"]
    native_cells = int(radar["range_bin_count"]) * int(radar["pulse_count"])
    generation_seconds = 0.0
    scoring_seconds = 0.0
    summary_seconds = 0.0
    generated = 0
    scored_cells = 0
    max_batch_payload_bytes = 0
    observed_scores: list[np.ndarray] = []
    benchmark_maps: np.ndarray | None = None

    for offset in range(0, scene_count, batch_size):
        count = min(batch_size, scene_count - offset)
        started = time.perf_counter()
        batch = generate_d2a_paired_batch(
            count,
            seed + offset,
            10.0,
            10.0,
            config,
            scenario_id="separated",
        )
        generation_seconds += time.perf_counter() - started
        generated += count
        max_batch_payload_bytes = max(
            max_batch_payload_bytes,
            sum(
                int(value.nbytes)
                for value in vars(batch).values()
                if isinstance(value, np.ndarray)
            ),
        )

        started = time.perf_counter()
        prediction = run_b3(batch.iq_h0, float(radar["prf_hz"]), config)
        scoring_seconds += time.perf_counter() - started
        maps = np.asarray(prediction.rd_power, dtype=np.float64)
        if benchmark_maps is None:
            benchmark_maps = maps.copy()
        observed_scores.append(np.max(maps, axis=(1, 2)))
        scored_cells += int(maps.size)

    # The next operation only measures summary code cost. Its threshold is not
    # a held-out or calibrated operating point and has no scientific meaning.
    if benchmark_maps is None:
        raise RuntimeError("no scene batch was generated")
    scene_maxima = np.concatenate(observed_scores)
    benchmark_threshold = float(np.quantile(scene_maxima, 0.5))
    started = time.perf_counter()
    benchmark_summary = dense_h0_summary(
        benchmark_maps,
        benchmark_threshold,
        np.ones(benchmark_maps.shape[1:], dtype=bool),
        random_cut_seed=seed,
        bootstrap_replicates=200,
    )
    summary_seconds = time.perf_counter() - started

    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover - optional environment probe
        cuda = {"available": None, "probe_error": str(exc)}

    rss_max = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; the current project environment is Linux.
    peak_rss_bytes = rss_max * 1024 if platform.system() == "Linux" else rss_max
    return _json_safe(
        {
            "experiment": "D2-A-R7-A-resource-preflight",
            "evidence_class": "small_streaming_resource_preflight_only",
            "scientific_metrics": "NOT_EVALUATED",
            "scene_count": generated,
            "batch_size": batch_size,
            "seed_start": seed,
            "condition": {"scenario_id": "separated", "snr_db": 10.0, "cnr_db": 10.0},
            "grid": {
                "range_cells": int(radar["range_bin_count"]),
                "doppler_cells": int(radar["pulse_count"]),
                "physical_cells_per_cpi": native_cells,
                "scored_physical_cells": scored_cells,
            },
            "timing_seconds": {
                "generation_total": generation_seconds,
                "scoring_total": scoring_seconds,
                "summary_total": summary_seconds,
                "generation_scenes_per_second": generated / generation_seconds,
                "scoring_scenes_per_second": generated / scoring_seconds,
                "scoring_cells_per_second": scored_cells / scoring_seconds,
            },
            "memory_and_storage": {
                "process_peak_rss_bytes": peak_rss_bytes,
                "maximum_in_memory_batch_array_bytes": max_batch_payload_bytes,
                "full_dataset_cache_written": False,
                "estimated_hours_for_2995731_scenes_at_measured_generation_rate": (
                    2_995_731 / (generated / generation_seconds) / 3600.0
                ),
                "estimated_hours_for_2995731_scenes_at_measured_scoring_rate": (
                    2_995_731 / (generated / scoring_seconds) / 3600.0
                ),
            },
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
                "cpu_count": os.cpu_count(),
                "cpu_model": platform.processor() or None,
                "cuda": cuda,
                "scorer": "NumPy CPU B3",
            },
            "benchmark_only": {
                "threshold_source": "median of same preflight batch; invalid for inference",
                "pfa_cell": benchmark_summary["pfa_cell_pooled"],
                "summary_scenes": int(benchmark_maps.shape[0]),
                "summary_runtime_seconds": summary_seconds,
            },
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=872_611_01)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "experiments/D2-A-E001/config.yaml"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(
        scene_count=args.scenes,
        batch_size=args.batch_size,
        seed=args.seed,
        config_path=args.config,
    )
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
