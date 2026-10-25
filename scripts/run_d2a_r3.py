#!/usr/bin/env python3
"""Run the preregistered D2-A-R3 threshold and scale-nuisance diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.baselines import wrap_frequency
from radardistill.d2a_baselines import DetectionOutput, run_b0, run_b1, run_b2, run_b3
from radardistill.d2a_r3_baselines import R3DetectionOutput, run_t1_global_normalized, run_t2_local_cfar
from radardistill.evaluation.d2a_e001 import _paired_auc_interval, empirical_auc
from radardistill.evaluation.d2a_r3 import (
    binomial_upper_tail,
    calibrate_empirical_threshold,
    hierarchical_finite_sample_bootstrap,
    independent_mean_difference_interval,
    paired_mean_difference_interval,
    summarize_r3_scores,
    summarize_threshold,
    validate_disjoint_seed_groups,
    wilson_interval,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch

METHODS_T0 = ("B0", "B1", "B2", "B3")
METHODS_ALL = METHODS_T0 + ("T1", "T2")
SCALE_FACTORS = ((0.5 + 0j), (2.0 + 0j), (0.6 + 1.2j))


def build_scene_group_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand the frozen YAML group table into explicit, count-bearing groups."""
    sampling = config["sampling"]
    count = int(sampling["scenes_per_group"])
    primary = sampling["primary_condition"]
    groups: list[dict[str, Any]] = []

    def add(
        group_id: str,
        role: str,
        seed_start: int,
        snr_db: float,
        cnr_db: float,
        scenario_id: str,
        scope: str,
        scene_count: int = count,
    ) -> None:
        groups.append(
            {
                "group_id": group_id,
                "role": role,
                "seed_start": int(seed_start),
                "snr_db": float(snr_db),
                "cnr_db": float(cnr_db),
                "scenario_id": scenario_id,
                "scene_count": int(scene_count),
                "scope": scope,
            }
        )

    for index, seed in enumerate(primary["calibration_seed_starts"], start=1):
        add(
            f"primary_cal_{index}", "calibration", seed, primary["snr_db"],
            primary["cnr_db"], "separated", scope="primary",
        )
    for entry in primary["validation_groups"]:
        add(
            f"primary_{entry['name']}", "validation", entry["seed_start"],
            primary["snr_db"], primary["cnr_db"], entry["scenario_id"], scope="primary",
        )
    for condition in sampling["secondary_conditions"]:
        cnr_tag = f"cnr{int(condition['cnr_db'])}"
        add(
            f"{cnr_tag}_cal", "calibration", condition["calibration_seed_start"],
            condition["snr_db"], condition["cnr_db"], "separated", scope="secondary",
        )
        for entry in condition["validation_groups"]:
            add(
                f"{cnr_tag}_{entry['name']}", "validation", entry["seed_start"],
                condition["snr_db"], condition["cnr_db"], entry["scenario_id"],
                scope="secondary",
            )
    scale = sampling["scale_diagnostic"]
    add(
        "controlled_scale", "scale_diagnostic", scale["seed_start"],
        primary["snr_db"], primary["cnr_db"], "separated",
        scene_count=scale["scene_count"], scope="scale_diagnostic",
    )
    return groups


def assert_raw_phase_reviewed(output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    review_path = output_dir / "raw_phase_review.md"
    if not metrics_path.exists():
        raise ValueError("raw T0 metrics.json is missing")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not metrics.get("raw_phase", {}).get("completed"):
        raise ValueError("raw T0 phase is not marked completed")
    if not review_path.exists() or "T0 原始分数复核完成" not in review_path.read_text(encoding="utf-8"):
        raise ValueError("raw_phase_review.md is required before candidate phase")


def assert_replayed_scene_coverage(
    raw_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    replayed_ids_by_group: dict[str, list[str]],
) -> None:
    """Require candidate replay to cover every frozen T0 scene exactly once."""
    expected_groups = {
        str(group["group_id"]): int(group["scene_count"])
        for group in groups
        if group["role"] != "scale_diagnostic"
    }
    raw_ids_by_group: dict[str, list[str]] = {group_id: [] for group_id in expected_groups}
    for row in raw_rows:
        group_id = str(row["group_id"])
        if group_id in raw_ids_by_group:
            raw_ids_by_group[group_id].append(str(row["base_scene_id"]))

    if set(raw_ids_by_group) != set(replayed_ids_by_group):
        raise ValueError("raw T0 replay coverage group mismatch")
    seen_ids: set[str] = set()
    for group_id, expected_count in expected_groups.items():
        raw_ids = raw_ids_by_group[group_id]
        replayed_ids = [str(value) for value in replayed_ids_by_group[group_id]]
        if len(raw_ids) != expected_count or len(set(raw_ids)) != expected_count:
            raise ValueError(f"raw T0 replay coverage is incomplete or duplicated for {group_id}")
        if len(replayed_ids) != expected_count or len(set(replayed_ids)) != expected_count:
            raise ValueError(f"candidate T0 replay coverage is incomplete or duplicated for {group_id}")
        if set(raw_ids) != set(replayed_ids):
            raise ValueError(f"raw T0 replay coverage IDs differ for {group_id}")
        if seen_ids.intersection(replayed_ids):
            raise ValueError(f"candidate T0 replay scene IDs overlap across groups: {group_id}")
        seen_ids.update(replayed_ids)


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


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(_json_safe(data), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _assert_disjoint_group_ranges(groups: list[dict[str, Any]]) -> None:
    ranges = sorted(
        (
            int(group["seed_start"]),
            int(group["seed_start"]) + int(group["scene_count"]) - 1,
            str(group["group_id"]),
        )
        for group in groups
    )
    for left, right in zip(ranges, ranges[1:]):
        if right[0] <= left[1]:
            raise ValueError(f"scene seed ranges overlap: {left[2]} and {right[2]}")
    validate_disjoint_seed_groups(
        [
            {"name": group["group_id"], "seed_start": group["seed_start"]}
            for group in groups
            if group["scene_count"] == 4096
        ],
        scenes_per_group=4096,
    )


def _validate_legacy_thresholds(config: dict[str, Any]) -> None:
    source = ROOT / "experiments/D2-A-E001/metrics.json"
    baseline = json.loads(source.read_text(encoding="utf-8"))
    for cnr in (0, 10, 20):
        key = f"snr_10_cnr_{cnr}"
        baseline_key = f"snr=10,cnr={cnr}"
        for method in METHODS_T0:
            frozen = float(config["detection"]["frozen_legacy_thresholds"][key][method])
            recorded = float(baseline["threshold_calibration"][baseline_key][method]["threshold"])
            if frozen != recorded:
                raise ValueError(f"R3 frozen legacy threshold changed for {key}/{method}")


def _environment_record() -> dict[str, Any]:
    def git_value(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    cpu_name = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                cpu_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        cpu_name = platform.processor() or "unknown"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        gpu = f"unavailable: {exc}"
    try:
        import numpy
        numpy_version = numpy.__version__
    except Exception:
        numpy_version = "unavailable"
    return {
        "git_sha": git_value("rev-parse", "HEAD"),
        "git_status_porcelain": git_value("status", "--porcelain=v1"),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": cpu_name,
        "logical_cpu_count": os.cpu_count(),
        "numpy": numpy_version,
        "pyyaml": yaml.__version__,
        "gpu_visibility_recorded_but_not_used_for_r3": gpu,
        "r3_device": "CPU",
    }


def _iter_group_batches(
    group: dict[str, Any],
    simulation_config: dict[str, Any],
    batch_size: int,
):
    count = int(group["scene_count"])
    for start in range(0, count, batch_size):
        current = min(batch_size, count - start)
        batch = generate_d2a_paired_batch(
            current,
            int(group["seed_start"]) + start,
            float(group["snr_db"]),
            float(group["cnr_db"]),
            simulation_config,
            scenario_id=str(group["scenario_id"]),
        )
        yield start, batch


def _run_t0_methods(iq: np.ndarray, config: dict[str, Any]) -> tuple[dict[str, DetectionOutput], dict[str, float]]:
    prf = float(config["radar"]["prf_hz"])
    methods = {
        "B0": lambda: run_b0(iq, prf),
        "B1": lambda: run_b1(iq, prf),
        "B2": lambda: run_b2(iq, prf, config),
        "B3": lambda: run_b3(iq, prf, config),
    }
    outputs: dict[str, DetectionOutput] = {}
    seconds_per_scene: dict[str, float] = {}
    scene_count = int(iq.shape[0]) if iq.ndim == 4 else 1
    for method, run in methods.items():
        started = time.perf_counter()
        outputs[method] = run()
        seconds_per_scene[method] = (time.perf_counter() - started) / scene_count
    return outputs, seconds_per_scene


def _score_vector(output: DetectionOutput | R3DetectionOutput, attribute: str = "score") -> np.ndarray:
    return np.asarray(getattr(output, attribute), dtype=np.float64).reshape(-1)


def _legacy_threshold(config: dict[str, Any], cnr_db: float, method: str) -> float:
    key = f"snr_10_cnr_{int(cnr_db)}"
    return float(config["detection"]["frozen_legacy_thresholds"][key][method])


def _threshold_row(
    scores: np.ndarray,
    threshold: float,
    *,
    method: str,
    threshold_source: str,
    group: dict[str, Any],
    record_type: str = "validation_group",
    phase: str = "raw",
) -> dict[str, Any]:
    summary = summarize_threshold(scores, threshold)
    count = int(round(summary["pfa_ge"] * summary["n"]))
    return {
        "phase": phase,
        "record_type": record_type,
        "method": method,
        "threshold_source": threshold_source,
        "group_id": group["group_id"],
        "scope": group["scope"],
        "scenario_id": group["scenario_id"],
        "snr_db": group["snr_db"],
        "cnr_db": group["cnr_db"],
        "threshold": threshold,
        **summary,
        "pfa_wilson_ci95": list(wilson_interval(count, summary["n"])),
    }


def _collect_raw_group(
    group: dict[str, Any],
    simulation_config: dict[str, Any],
    r3_config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, float]]:
    scores: dict[str, list[np.ndarray]] = {method: [] for method in METHODS_T0}
    rows: list[dict[str, Any]] = []
    runtime: dict[str, list[float]] = {method: [] for method in METHODS_T0}
    for start, batch in _iter_group_batches(
        group, simulation_config, int(r3_config["sampling"]["batch_size"])
    ):
        outputs, seconds = _run_t0_methods(batch.iq_h0, simulation_config)
        for method in METHODS_T0:
            scores[method].append(_score_vector(outputs[method]))
            runtime[method].append(seconds[method])
        amp_db = 20.0 * np.log10(batch.truth_target_amplitude)
        measured_noise = np.mean(np.abs(batch.noise_iq.astype(np.complex128)) ** 2, axis=(1, 2, 3))
        measured_clutter = np.mean(np.abs(batch.clutter_iq.astype(np.complex128)) ** 2, axis=(1, 2, 3))
        h0_power = np.mean(np.abs(batch.iq_h0.astype(np.complex128)) ** 2, axis=(1, 2, 3))
        for local_index, scene_id in enumerate(batch.base_scene_id):
            row: dict[str, Any] = {
                "phase": "raw",
                "group_id": group["group_id"],
                "role": group["role"],
                "scope": group["scope"],
                "scenario_id": group["scenario_id"],
                "snr_db": group["snr_db"],
                "cnr_db": group["cnr_db"],
                "seed_start": group["seed_start"],
                "batch_seed": int(group["seed_start"]) + start,
                "batch_local_index": local_index,
                "base_scene_id": str(scene_id),
                "potential_target_amplitude": float(batch.truth_target_amplitude[local_index]),
                "potential_target_amplitude_db": float(amp_db[local_index]),
                "noise_power_parameter": float(batch.noise_power[local_index]),
                "measured_noise_power": float(measured_noise[local_index]),
                "measured_clutter_power": float(measured_clutter[local_index]),
                "measured_h0_total_power": float(h0_power[local_index]),
            }
            for method in METHODS_T0:
                row[f"score_{method}"] = float(_score_vector(outputs[method])[local_index])
            rows.append(row)
    merged = {method: np.concatenate(parts) for method, parts in scores.items()}
    timing = {method: float(np.mean(parts)) for method, parts in runtime.items()}
    return merged, rows, timing


def _collect_raw_scale_diagnostic(
    group: dict[str, Any],
    simulation_config: dict[str, Any],
    r3_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: dict[tuple[str, str], list[float]] = {
        (factor_name, method): []
        for factor_name in ("real_0p5", "real_2", "complex_0p6_1p2")
        for method in METHODS_T0
    }
    factor_names = ("real_0p5", "real_2", "complex_0p6_1p2")
    for _, batch in _iter_group_batches(
        group, simulation_config, int(r3_config["sampling"]["batch_size"])
    ):
        reference, _ = _run_t0_methods(batch.iq_h0, simulation_config)
        for factor_name, factor in zip(factor_names, SCALE_FACTORS, strict=True):
            scaled, _ = _run_t0_methods(factor * batch.iq_h0, simulation_config)
            for method in METHODS_T0:
                original_scores = _score_vector(reference[method])
                scaled_scores = _score_vector(scaled[method])
                expected = (abs(factor) ** 2) * original_scores
                relative = np.abs(scaled_scores - expected) / np.maximum(np.abs(expected), 1e-30)
                errors[(factor_name, method)].extend(relative.tolist())
        for index, scene_id in enumerate(batch.base_scene_id):
            row = {
                "phase": "raw",
                "group_id": group["group_id"],
                "role": group["role"],
                "scope": group["scope"],
                "scenario_id": group["scenario_id"],
                "snr_db": group["snr_db"],
                "cnr_db": group["cnr_db"],
                "seed_start": group["seed_start"],
                "base_scene_id": str(scene_id),
                "potential_target_amplitude": float(batch.truth_target_amplitude[index]),
                "potential_target_amplitude_db": float(20 * np.log10(batch.truth_target_amplitude[index])),
                "noise_power_parameter": float(batch.noise_power[index]),
                "measured_noise_power": float(np.mean(np.abs(batch.noise_iq[index].astype(np.complex128)) ** 2)),
                "measured_clutter_power": float(np.mean(np.abs(batch.clutter_iq[index].astype(np.complex128)) ** 2)),
                "measured_h0_total_power": float(np.mean(np.abs(batch.iq_h0[index].astype(np.complex128)) ** 2)),
            }
            for method in METHODS_T0:
                row[f"score_{method}"] = float(_score_vector(reference[method])[index])
            rows.append(row)
    summary = [
        {
            "factor": factor_name,
            "method": method,
            "max_relative_error_from_abs_a_squared": float(np.max(values)),
            "mean_relative_error_from_abs_a_squared": float(np.mean(values)),
            "n_scenes": len(values),
        }
        for (factor_name, method), values in errors.items()
    ]
    return summary, rows


def _amplitude_bin(amplitude_db: np.ndarray, edges: list[float]) -> np.ndarray:
    bins = np.searchsorted(np.asarray(edges), amplitude_db, side="right") - 1
    return np.clip(bins, 0, len(edges) - 2)


def _raw_amplitude_diagnostics(
    scene_rows: list[dict[str, Any]],
    thresholds_by_group: dict[str, dict[str, dict[str, Any]]],
    validation_groups: list[dict[str, Any]],
    r3_config: dict[str, Any],
) -> dict[str, Any]:
    primary_ids = {group["group_id"] for group in validation_groups if group["scope"] == "primary"}
    rows = [row for row in scene_rows if row["group_id"] in primary_ids]
    amp = np.asarray([row["potential_target_amplitude_db"] for row in rows], dtype=np.float64)
    edges = [float(value) for value in r3_config["detection"]["amplitude_bins_db"]]
    indices = _amplitude_bin(amp, edges)
    edges_summary: list[dict[str, Any]] = []
    by_bin: list[dict[str, Any]] = []
    for index in range(len(edges) - 1):
        mask = indices == index
        subset = [row for row, keep in zip(rows, mask, strict=True) if keep]
        by_bin.append(
            {
                "amplitude_bin_index": index,
                "amplitude_low_db": edges[index],
                "amplitude_high_db": edges[index + 1],
                "n_scenes": len(subset),
                "median_h0_total_power": float(np.median([row["measured_h0_total_power"] for row in subset])),
                "median_noise_power": float(np.median([row["measured_noise_power"] for row in subset])),
                "median_clutter_power": float(np.median([row["measured_clutter_power"] for row in subset])),
            }
        )
    for method in METHODS_T0:
        for cal_group in thresholds_by_group:
            if not cal_group.startswith("primary_cal_"):
                continue
            threshold = float(thresholds_by_group[cal_group][method]["threshold"])
            rates = []
            for index, row_bin in enumerate(by_bin):
                mask_rows = [row for row, bin_index in zip(rows, indices, strict=True) if bin_index == index]
                scores = np.asarray([row[f"score_{method}"] for row in mask_rows], dtype=np.float64)
                detections = int(np.sum(scores >= threshold))
                interval = wilson_interval(detections, scores.size)
                rate = float(detections / scores.size)
                rates.append(rate)
                edges_summary.append(
                    {
                        "method": method,
                        "threshold_source": cal_group,
                        "amplitude_bin_index": index,
                        "amplitude_low_db": edges[index],
                        "amplitude_high_db": edges[index + 1],
                        "n_scenes": int(scores.size),
                        "pfa_cpi": rate,
                        "pfa_wilson_ci95": list(interval),
                    }
                )
            edges_summary.append(
                {
                    "method": method,
                    "threshold_source": cal_group,
                    "amplitude_bin_index": "range",
                    "max_minus_min_pfa": float(max(rates) - min(rates)),
                }
            )
    x = amp
    total_db = 10.0 * np.log10(np.asarray([row["measured_h0_total_power"] for row in rows]))
    b3_db = 10.0 * np.log10(np.asarray([row["score_B3"] for row in rows]))
    return {
        "power_by_amplitude_bin": by_bin,
        "pfa_by_amplitude_bin": edges_summary,
        "potential_amplitude_db_vs_h0_total_power_db_pearson_r": float(np.corrcoef(x, total_db)[0, 1]),
        "potential_amplitude_db_vs_B3_score_db_pearson_r": float(np.corrcoef(x, b3_db)[0, 1]),
        "potential_amplitude_db_to_B3_score_db_linear_slope": float(np.polyfit(x, b3_db, 1)[0]),
        "interpretation": "Descriptive association only; latent amplitude and component powers are never detector inputs.",
    }


def _threshold_validation_rows(
    group_scores: dict[str, dict[str, np.ndarray]],
    group_specs: list[dict[str, Any]],
    thresholds_by_group: dict[str, dict[str, dict[str, Any]]],
    r3_config: dict[str, Any],
) -> list[dict[str, Any]]:
    validations = [group for group in group_specs if group["role"] == "validation"]
    calibrations = [group for group in group_specs if group["role"] == "calibration"]
    primary_calibrations = [group for group in calibrations if group["scope"] == "primary"]
    rows: list[dict[str, Any]] = []
    for group in validations:
        matched = [cal for cal in calibrations if cal["cnr_db"] == group["cnr_db"]]
        for method in METHODS_T0:
            for cal in matched:
                threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
                rows.append(
                    _threshold_row(
                        group_scores[group["group_id"]][method], threshold,
                        method=method, threshold_source=cal["group_id"], group=group,
                    )
                )
            legacy = _legacy_threshold(r3_config, group["cnr_db"], method)
            rows.append(
                _threshold_row(
                    group_scores[group["group_id"]][method], legacy,
                    method=method, threshold_source="E001_frozen_matched_CNR", group=group,
                )
            )
            if group["cnr_db"] != 10.0:
                for cal in primary_calibrations:
                    threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
                    rows.append(
                        _threshold_row(
                            group_scores[group["group_id"]][method], threshold,
                            method=method, threshold_source=f"primary_10dB_{cal['group_id']}", group=group,
                        )
                    )
                rows.append(
                    _threshold_row(
                        group_scores[group["group_id"]][method],
                        _legacy_threshold(r3_config, 10.0, method),
                        method=method, threshold_source="E001_frozen_primary_10dB", group=group,
                    )
                )
    for cnr in sorted({float(group["cnr_db"]) for group in validations}):
        matching_validation = [group for group in validations if group["cnr_db"] == cnr]
        matching_cal = [cal for cal in calibrations if cal["cnr_db"] == cnr]
        for method in METHODS_T0:
            for cal in matching_cal:
                scores = np.concatenate([group_scores[group["group_id"]][method] for group in matching_validation])
                threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
                combined_group = {
                    "group_id": f"cnr{int(cnr)}_all_validation",
                    "role": "validation_combined",
                    "scope": "combined_independent_groups",
                    "scenario_id": "A_B_C_D_or_A_C_combined",
                    "snr_db": 10.0,
                    "cnr_db": cnr,
                }
                rows.append(
                    _threshold_row(
                        scores, threshold, method=method,
                        threshold_source=cal["group_id"], group=combined_group,
                        record_type="combined_validation",
                    )
                )
    return rows


def _validate_r3_groups(groups: list[dict[str, Any]]) -> None:
    _assert_disjoint_group_ranges(groups)
    if len({group["group_id"] for group in groups}) != len(groups):
        raise ValueError("R3 group IDs must be unique")


def run_raw_phase(r3_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    group_specs = build_scene_group_specs(r3_config)
    _validate_r3_groups(group_specs)
    _validate_legacy_thresholds(r3_config)
    simulation_config = yaml.safe_load((ROOT / "experiments/D2-A-E001/config.yaml").read_text(encoding="utf-8"))
    if simulation_config["radar"] != r3_config["radar"]:
        raise ValueError("R3 radar parameters must exactly match the frozen E001 radar parameters")
    rows: list[dict[str, Any]] = []
    scores_by_group: dict[str, dict[str, np.ndarray]] = {}
    runtime_by_group: dict[str, dict[str, float]] = {}
    evaluation_groups = [group for group in group_specs if group["role"] != "scale_diagnostic"]
    for index, group in enumerate(evaluation_groups, start=1):
        scores, group_rows, timing = _collect_raw_group(group, simulation_config, r3_config)
        scores_by_group[group["group_id"]] = scores
        rows.extend(group_rows)
        runtime_by_group[group["group_id"]] = timing
        print(
            f"[raw {index}/{len(evaluation_groups)}] {group['group_id']} "
            f"n={group['scene_count']} SNR/CNR={group['snr_db']}/{group['cnr_db']} dB",
            flush=True,
        )

    thresholds_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for group in evaluation_groups:
        if group["role"] != "calibration":
            continue
        thresholds_by_group[group["group_id"]] = {
            method: calibrate_empirical_threshold(
                scores_by_group[group["group_id"]][method],
                float(r3_config["detection"]["target_pfa_cpi"]),
            )
            for method in METHODS_T0
        }
    validation_rows = _threshold_validation_rows(
        scores_by_group, evaluation_groups, thresholds_by_group, r3_config
    )
    primary_calibrators = [group for group in evaluation_groups if group["role"] == "calibration" and group["scope"] == "primary"]
    primary_validators = [group for group in evaluation_groups if group["role"] == "validation" and group["scope"] == "primary"]
    primary_b3_h0 = np.concatenate([scores_by_group[group["group_id"]]["B3"] for group in primary_validators])
    legacy_b3 = _legacy_threshold(r3_config, 10.0, "B3")
    legacy_summary = summarize_threshold(primary_b3_h0, legacy_b3)
    legacy_false_count = int(round(legacy_summary["pfa_ge"] * legacy_summary["n"]))
    estimate_pfa = float(legacy_summary["pfa_ge"])
    calibration_b3 = [scores_by_group[group["group_id"]]["B3"] for group in primary_calibrators]
    validation_b3 = [scores_by_group[group["group_id"]]["B3"] for group in primary_validators]
    finite_sample = hierarchical_finite_sample_bootstrap(
        calibration_b3,
        validation_b3,
        target_pfa=float(r3_config["detection"]["target_pfa_cpi"]),
        calibration_n=int(r3_config["detection"]["finite_sample_calibration_n"]),
        validation_n=int(r3_config["detection"]["finite_sample_validation_n"]),
        replicates=int(r3_config["detection"]["finite_sample_bootstrap_replicates"]),
        seed=2026102501,
    )
    finite_sample["legacy_threshold_validation_pfa"] = estimate_pfa
    finite_sample["binomial_tail_validation_only_pr_23_of_256"] = binomial_upper_tail(
        23, 256, estimate_pfa
    )
    finite_sample["legacy_threshold_pfa_wilson_ci95"] = list(
        wilson_interval(legacy_false_count, legacy_summary["n"])
    )
    scale_group = next(group for group in group_specs if group["role"] == "scale_diagnostic")
    scale_summary, scale_rows = _collect_raw_scale_diagnostic(scale_group, simulation_config, r3_config)
    rows.extend(scale_rows)
    amplitude_summary = _raw_amplitude_diagnostics(
        rows, thresholds_by_group, evaluation_groups, r3_config
    )
    metric = {
        "experiment_id": "D2-A-R3",
        "data_are_synthetic": True,
        "phase_order": ["raw T0 H0", "review raw T0", "T1/T2 candidates"],
        "raw_phase": {
            "completed": True,
            "methods": list(METHODS_T0),
            "scenes_per_primary_calibration_group": int(r3_config["sampling"]["scenes_per_group"]),
            "scenes_per_primary_validation_group": int(r3_config["sampling"]["scenes_per_group"]),
            "thresholds_by_group": thresholds_by_group,
            "validation_threshold_results": validation_rows,
            "legacy_b3_primary_validation": {
                **legacy_summary,
                "pfa_wilson_ci95": list(wilson_interval(legacy_false_count, legacy_summary["n"])),
                "e001_historical_point_pfa": 0.08984375,
                "e001_historical_validation_n": 256,
            },
            "finite_sample_diagnostic": finite_sample,
            "amplitude_and_component_power": amplitude_summary,
            "controlled_scale": scale_summary,
            "runtime_seconds": float(time.perf_counter() - started),
            "seconds_per_scene_by_group_and_method": runtime_by_group,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "h0_scene_diagnostics.csv", rows)
    _write_csv(output_dir / "result.csv", validation_rows)
    _write_json(output_dir / "metrics.json", metric)
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(r3_config, sort_keys=False), encoding="utf-8")
    _write_json(output_dir / "environment.json", _environment_record())
    _write_raw_phase_analysis(output_dir, metric)
    return metric


def _write_raw_phase_analysis(output_dir: Path, metrics: dict[str, Any]) -> None:
    raw = metrics["raw_phase"]
    legacy = raw["legacy_b3_primary_validation"]
    finite = raw["finite_sample_diagnostic"]
    lines = [
        "# D2-A-R3 第一阶段：原始 T0 H0 统计量",
        "",
        "本文件仅记录冻结 B0–B3 原始分数的 H0 诊断。T1/T2 尚未运行；所有阈值比较仍使用原实现。",
        "",
        f"- 新 primary 独立 H0 场景：{legacy['n']}。",
        f"- 将 E001 B3 10/10 阈值原样应用到新 H0 后，`Pfa_CPI`={legacy['pfa_ge']:.6f}，95% Wilson CI `{legacy['pfa_wilson_ci95']}`。",
        f"- E001 的 23/256 在固定该新估计率时的验证抽样尾概率：{finite['binomial_tail_validation_only_pr_23_of_256']:.6g}。",
        f"- 联合 n=512 calibration / n=256 validation 层级 bootstrap 中，Pfa≥23/256 的比例：{finite['probability_pfa_at_least_23_of_256']:.6g}。",
        "- 按固定潜在目标幅度箱的组件功率与原始分数关系见 `metrics.json` 和 `h0_scene_diagnostics.csv`。",
        "",
        "## T0 原始分数复核结果",
        "",
        "请在运行候选方法前审查每种基线的校准阈值变化、组间 Pfa、严格/非严格比较和平局数，以及有限样本 bootstrap。结论记录在 `raw_phase_review.md`；只有该文件写入“**T0 原始分数复核完成**”后，运行器才接受 `--phase candidates`。",
        "",
        "此处为阶段性诊断，不是最终阶段审查。",
    ]
    (output_dir / "raw_phase_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_all_methods(
    iq: np.ndarray,
    simulation_config: dict[str, Any],
    r3_config: dict[str, Any],
) -> dict[str, DetectionOutput | R3DetectionOutput]:
    raw, _ = _run_t0_methods(iq, simulation_config)
    prf = float(simulation_config["radar"]["prf_hz"])
    return {
        **raw,
        "T1": run_t1_global_normalized(
            iq, prf, simulation_config, r3_config["detection"]["t1_global_robust_background"]
        ),
        "T2": run_t2_local_cfar(
            iq, prf, simulation_config, r3_config["detection"]["t2_local_cfar_like"]
        ),
    }


def _candidate_threshold_validation_rows(
    group_data: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
    thresholds_by_group: dict[str, dict[str, dict[str, Any]]],
    r3_config: dict[str, Any],
) -> list[dict[str, Any]]:
    validations = [group for group in groups if group["role"] == "validation"]
    calibrations = [group for group in groups if group["role"] == "calibration"]
    primary_calibrations = [group for group in calibrations if group["scope"] == "primary"]
    rows: list[dict[str, Any]] = []
    for group in validations:
        matched = [cal for cal in calibrations if cal["cnr_db"] == group["cnr_db"]]
        for method in METHODS_ALL:
            scores = group_data[group["group_id"]]["h0_scores"][method]
            for cal in matched:
                threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
                rows.append(
                    _threshold_row(
                        scores, threshold, method=method,
                        threshold_source=cal["group_id"], group=group, phase="candidate",
                    )
                )
            if method in METHODS_T0:
                legacy = _legacy_threshold(r3_config, group["cnr_db"], method)
                rows.append(
                    _threshold_row(
                        scores, legacy, method=method,
                        threshold_source="E001_frozen_matched_CNR", group=group, phase="candidate",
                    )
                )
            if group["cnr_db"] != 10.0:
                for cal in primary_calibrations:
                    threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
                    rows.append(
                        _threshold_row(
                            scores, threshold, method=method,
                            threshold_source=f"primary_10dB_{cal['group_id']}", group=group,
                            phase="candidate",
                        )
                    )
                if method in METHODS_T0:
                    rows.append(
                        _threshold_row(
                            scores, _legacy_threshold(r3_config, 10.0, method),
                            method=method, threshold_source="E001_frozen_primary_10dB",
                            group=group, phase="candidate",
                        )
                    )
    for cnr in sorted({float(group["cnr_db"]) for group in validations}):
        matching_validation = [group for group in validations if group["cnr_db"] == cnr]
        matching_cal = [cal for cal in calibrations if cal["cnr_db"] == cnr]
        for method in METHODS_ALL:
            scores = np.concatenate(
                [group_data[group["group_id"]]["h0_scores"][method] for group in matching_validation]
            )
            for cal in matching_cal:
                rows.append(
                    _threshold_row(
                        scores,
                        float(thresholds_by_group[cal["group_id"]][method]["threshold"]),
                        method=method,
                        threshold_source=cal["group_id"],
                        group={
                            "group_id": f"cnr{int(cnr)}_all_validation",
                            "scope": "combined_independent_groups",
                            "scenario_id": "A_B_C_D_or_A_C_combined",
                            "snr_db": 10.0,
                            "cnr_db": cnr,
                        },
                        record_type="combined_validation",
                        phase="candidate",
                    )
                )
    return rows


def _candidate_amplitude_diagnostics(
    scene_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    thresholds_by_group: dict[str, dict[str, dict[str, Any]]],
    r3_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    primary_ids = {
        group["group_id"] for group in groups
        if group["role"] == "validation" and group["scope"] == "primary"
    }
    rows = [row for row in scene_rows if row["group_id"] in primary_ids]
    amplitude = np.asarray([float(row["potential_target_amplitude_db"]) for row in rows])
    edges = [float(value) for value in r3_config["detection"]["amplitude_bins_db"]]
    bin_index = _amplitude_bin(amplitude, edges)
    primary_calibrators = [
        group for group in groups if group["role"] == "calibration" and group["scope"] == "primary"
    ]
    amplitude_rows: list[dict[str, Any]] = []
    gates: dict[str, Any] = {}
    reps = int(r3_config["detection"]["bootstrap_replicates"])
    for method in METHODS_ALL:
        method_gates: list[dict[str, Any]] = []
        for cal in primary_calibrators:
            threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
            bin_detected: list[np.ndarray] = []
            rates: list[float] = []
            for index in range(len(edges) - 1):
                selected_rows = [row for row, value in zip(rows, bin_index, strict=True) if value == index]
                values = np.asarray([float(row[f"score_{method}"]) >= threshold for row in selected_rows])
                bin_detected.append(values)
                successes = int(np.sum(values))
                rate = successes / values.size
                rates.append(float(rate))
                amplitude_rows.append(
                    {
                        "method": method,
                        "threshold_source": cal["group_id"],
                        "amplitude_bin_index": index,
                        "amplitude_low_db": edges[index],
                        "amplitude_high_db": edges[index + 1],
                        "n_scenes": int(values.size),
                        "pfa_cpi": float(rate),
                        "pfa_wilson_ci95": list(wilson_interval(successes, values.size)),
                    }
                )
            difference_ci = independent_mean_difference_interval(
                bin_detected[-1],
                bin_detected[0],
                replicates=reps,
                seed=2026102601 + METHODS_ALL.index(method) * 10 + primary_calibrators.index(cal),
            )
            max_minus_min = float(max(rates) - min(rates))
            passed = (
                difference_ci[0] >= -0.02
                and difference_ci[1] <= 0.02
                and max_minus_min <= 0.03
            )
            method_gates.append(
                {
                    "threshold_source": cal["group_id"],
                    "max_minus_min_pfa": max_minus_min,
                    "high_minus_low_pfa_ci95": list(difference_ci),
                    "passed": passed,
                }
            )
        gates[method] = {
            "threshold_replicates_passing": sum(item["passed"] for item in method_gates),
            "required_threshold_replicates_passing": int(
                r3_config["gates"]["G3"]["min_calibration_replicates_passing"]
            ),
            "replicates": method_gates,
            "passed": sum(item["passed"] for item in method_gates)
            >= int(r3_config["gates"]["G3"]["min_calibration_replicates_passing"]),
        }
    return amplitude_rows, gates


def _candidate_g2(
    thresholds_by_group: dict[str, dict[str, dict[str, Any]]],
    h0_rows: list[dict[str, Any]],
    groups: list[dict[str, Any]],
    r3_config: dict[str, Any],
) -> dict[str, Any]:
    interval = [float(value) for value in r3_config["gates"]["G2"]["validation_pfa_interval"]]
    group_interval = [float(value) for value in r3_config["gates"]["G2"]["each_primary_validation_ci_must_intersect_interval"]]
    calibrators = [group for group in groups if group["role"] == "calibration"]
    output: dict[str, Any] = {}
    for method in METHODS_ALL:
        replicate_results: list[dict[str, Any]] = []
        for cal in calibrators:
            cal_summary = thresholds_by_group[cal["group_id"]][method]
            matched_validation = [
                group for group in groups
                if group["role"] == "validation" and group["cnr_db"] == cal["cnr_db"]
            ]
            combined = next(
                row for row in h0_rows
                if row["record_type"] == "combined_validation"
                and row["method"] == method
                and row["threshold_source"] == cal["group_id"]
                and float(row["cnr_db"]) == float(cal["cnr_db"])
            )
            group_results = []
            for group in matched_validation:
                row = next(
                    row for row in h0_rows
                    if row["record_type"] == "validation_group"
                    and row["method"] == method
                    and row["threshold_source"] == cal["group_id"]
                    and row["group_id"] == group["group_id"]
                )
                ci = [float(value) for value in row["pfa_wilson_ci95"]]
                intersects = ci[1] >= group_interval[0] and ci[0] <= group_interval[1]
                group_results.append(
                    {"group_id": group["group_id"], "pfa": float(row["pfa_ge"]), "ci95": ci, "ci_intersects_target_band": intersects}
                )
            combined_ci = [float(value) for value in combined["pfa_wilson_ci95"]]
            cal_pass = interval[0] <= float(cal_summary["pfa_ge"]) <= interval[1]
            val_pass = (
                interval[0] <= float(combined["pfa_ge"]) <= interval[1]
                and combined_ci[0] <= 0.05 <= combined_ci[1]
                and all(item["ci_intersects_target_band"] for item in group_results)
            )
            replicate_results.append(
                {
                    "calibration_group": cal["group_id"],
                    "calibration_pfa_ge": float(cal_summary["pfa_ge"]),
                    "calibration_pass": cal_pass,
                    "combined_validation_pfa": float(combined["pfa_ge"]),
                    "combined_validation_ci95": combined_ci,
                    "validation_groups": group_results,
                    "validation_pass": val_pass,
                    "passed": cal_pass and val_pass,
                }
            )
        output[method] = {"replicates": replicate_results, "passed": all(item["passed"] for item in replicate_results)}
    return output


def _candidate_g4(
    group_data: dict[str, dict[str, Any]],
    groups: list[dict[str, Any]],
    thresholds_by_group: dict[str, dict[str, dict[str, Any]]],
    r3_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reps = int(r3_config["detection"]["bootstrap_replicates"])
    limit = float(r3_config["gates"]["G4"]["max_allowed_pd_regression"])
    joint_limit = float(r3_config["gates"]["G4"]["max_allowed_joint_success_regression"])
    identifiable = set(r3_config["gates"]["G4"]["identifiable_scenarios"])
    validations = [group for group in groups if group["role"] == "validation" and group["scenario_id"] in identifiable]
    calibrations = [group for group in groups if group["role"] == "calibration"]
    rows: list[dict[str, Any]] = []
    for group in validations:
        matched_calibrators = [cal for cal in calibrations if cal["cnr_db"] == group["cnr_db"]]
        data = group_data[group["group_id"]]
        ids = data["base_scene_id"]
        truths_r = data["truth_range_cell"]
        truths_d = data["truth_doppler_hz"]
        for cal in matched_calibrators:
            b3_threshold = float(thresholds_by_group[cal["group_id"]]["B3"]["threshold"])
            b3_detect = data["h1_scores"]["B3"] >= b3_threshold
            b3_range_error = data["h1_range"]["B3"] - truths_r
            b3_doppler_error = np.asarray(
                wrap_frequency(data["h1_doppler"]["B3"] - truths_d, float(r3_config["radar"]["prf_hz"]))
            )
            b3_joint = b3_detect & (np.abs(b3_range_error) <= 1.0) & (np.abs(b3_doppler_error) <= 20.0)
            for method in METHODS_ALL:
                threshold = float(thresholds_by_group[cal["group_id"]][method]["threshold"])
                candidate_detect = data["h1_scores"][method] >= threshold
                range_error = data["h1_range"][method] - truths_r
                doppler_error = np.asarray(
                    wrap_frequency(data["h1_doppler"][method] - truths_d, float(r3_config["radar"]["prf_hz"]))
                )
                candidate_joint = candidate_detect & (np.abs(range_error) <= 1.0) & (np.abs(doppler_error) <= 20.0)
                pd_ci = paired_mean_difference_interval(
                    candidate_detect.astype(np.float64), b3_detect.astype(np.float64), ids,
                    replicates=reps, seed=2026102701 + METHODS_ALL.index(method),
                )
                joint_ci = paired_mean_difference_interval(
                    candidate_joint.astype(np.float64), b3_joint.astype(np.float64), ids,
                    replicates=reps, seed=2026102801 + METHODS_ALL.index(method),
                )
                rows.append(
                    {
                        "method": method,
                        "reference_method": "B3",
                        "threshold_source": cal["group_id"],
                        "group_id": group["group_id"],
                        "scenario_id": group["scenario_id"],
                        "cnr_db": group["cnr_db"],
                        "pd_difference": float(np.mean(candidate_detect) - np.mean(b3_detect)),
                        "pd_difference_ci95": list(pd_ci),
                        "joint_success_difference": float(np.mean(candidate_joint) - np.mean(b3_joint)),
                        "joint_success_difference_ci95": list(joint_ci),
                        "passed": pd_ci[0] > -limit and joint_ci[0] > -joint_limit,
                    }
                )
    result: dict[str, Any] = {}
    for method in METHODS_ALL:
        method_rows = [row for row in rows if row["method"] == method]
        result[method] = {
            "groups_evaluated": len(method_rows),
            "passed": all(row["passed"] for row in method_rows),
            "rows": method_rows,
        }
    return rows, result


def _candidate_gates(
    h0_rows: list[dict[str, Any]],
    g2: dict[str, Any],
    g3: dict[str, Any],
    g4: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for method in METHODS_ALL:
        output[method] = {
            "G2": bool(g2[method]["passed"]),
            "G3": bool(g3[method]["passed"]),
            "G4": bool(g4[method]["passed"]),
            "all_front_gates": bool(g2[method]["passed"] and g3[method]["passed"] and g4[method]["passed"]),
        }
    order = ("B3", "T1", "T2")
    selected = next((method for method in order if output[method]["all_front_gates"]), None)
    return {
        "methods": output,
        "selected_b3_based_statistic": selected,
        "any_b3_based_front_gate_passed": selected is not None,
    }


def _candidate_scale_diagnostic(
    group: dict[str, Any],
    simulation_config: dict[str, Any],
    r3_config: dict[str, Any],
) -> list[dict[str, Any]]:
    factor_names = ("real_0p5", "real_2", "complex_0p6_1p2")
    raw_errors: dict[tuple[str, str], list[float]] = {
        (factor_name, method): []
        for factor_name in factor_names
        for method in METHODS_T0
    }
    normalized_errors: dict[tuple[str, str], list[float]] = {
        (factor_name, method): []
        for factor_name in factor_names
        for method in ("T1", "T2")
    }
    for _, batch in _iter_group_batches(
        group, simulation_config, int(r3_config["sampling"]["batch_size"])
    ):
        original = _run_all_methods(batch.iq_h0, simulation_config, r3_config)
        for factor_name, factor in zip(factor_names, SCALE_FACTORS, strict=True):
            scaled = _run_all_methods(factor * batch.iq_h0, simulation_config, r3_config)
            for method in METHODS_T0:
                base_score = _score_vector(original[method])
                actual = _score_vector(scaled[method])
                expected = abs(factor) ** 2 * base_score
                raw_errors[(factor_name, method)].extend(
                    (np.abs(actual - expected) / np.maximum(np.abs(expected), 1e-30)).tolist()
                )
            for method in ("T1", "T2"):
                base_map = np.asarray(original[method].score_map)
                scaled_map = np.asarray(scaled[method].score_map)
                normalized_errors[(factor_name, method)].extend(
                    np.max(np.abs(scaled_map - base_map), axis=(1, 2)).tolist()
                )
    rows = []
    for (factor_name, method), values in raw_errors.items():
        rows.append(
            {
                "factor": factor_name,
                "method": method,
                "max_relative_error_from_abs_a_squared": float(np.max(values)),
                "mean_relative_error_from_abs_a_squared": float(np.mean(values)),
                "n_scenes": len(values),
            }
        )
    for (factor_name, method), values in normalized_errors.items():
        rows.append(
            {
                "factor": factor_name,
                "method": method,
                "max_abs_difference_normalized_map": float(np.max(values)),
                "mean_abs_difference_normalized_map": float(np.mean(values)),
                "n_scenes": len(values),
            }
        )
    return rows


def run_candidate_phase(r3_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    assert_raw_phase_reviewed(output_dir)
    started = time.perf_counter()
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    raw_metrics = metrics["raw_phase"]
    groups = build_scene_group_specs(r3_config)
    _validate_r3_groups(groups)
    simulation_config = yaml.safe_load((ROOT / "experiments/D2-A-E001/config.yaml").read_text(encoding="utf-8"))
    if simulation_config["radar"] != r3_config["radar"]:
        raise ValueError("R3 radar parameters must exactly match the frozen E001 radar parameters")

    h0_rows = _read_csv_rows(output_dir / "h0_scene_diagnostics.csv")
    row_by_id = {row["base_scene_id"]: row for row in h0_rows}
    if len(row_by_id) != len(h0_rows):
        raise ValueError("raw phase contains duplicate base_scene_id values")
    group_data: dict[str, dict[str, Any]] = {}
    h1_scene_rows: list[dict[str, Any]] = []
    reference_contamination: dict[str, dict[str, list[float]]] = {
        group["group_id"]: {"T1": [], "T2": []}
        for group in groups if group["role"] == "validation"
    }
    replayed_ids_by_group: dict[str, list[str]] = {}
    for index, group in enumerate([group for group in groups if group["role"] != "scale_diagnostic"], start=1):
        h0_parts = {method: [] for method in METHODS_ALL}
        h1_parts = {method: [] for method in METHODS_ALL}
        h1_range_parts = {method: [] for method in METHODS_ALL}
        h1_doppler_parts = {method: [] for method in METHODS_ALL}
        truth_range_parts: list[np.ndarray] = []
        truth_doppler_parts: list[np.ndarray] = []
        id_parts: list[np.ndarray] = []
        group_replayed_ids: list[str] = []
        for start, batch in _iter_group_batches(
            group, simulation_config, int(r3_config["sampling"]["batch_size"])
        ):
            group_replayed_ids.extend(str(value) for value in batch.base_scene_id)
            h0_outputs = _run_all_methods(batch.iq_h0, simulation_config, r3_config)
            for method in METHODS_T0:
                current = _score_vector(h0_outputs[method])
                for local_index, scene_id in enumerate(batch.base_scene_id):
                    stored = row_by_id.get(str(scene_id))
                    if stored is None:
                        raise ValueError(f"candidate phase generated unknown base_scene_id {scene_id}")
                    previous = float(stored[f"score_{method}"])
                    if current[local_index] != previous:
                        raise ValueError(f"frozen raw {method} score changed for scene {scene_id}")
            for method in METHODS_ALL:
                h0_parts[method].append(_score_vector(h0_outputs[method]))
            for local_index, scene_id in enumerate(batch.base_scene_id):
                stored = row_by_id[str(scene_id)]
                for method in ("T1", "T2"):
                    stored[f"score_{method}"] = float(_score_vector(h0_outputs[method])[local_index])

            if group["role"] == "validation":
                h1_outputs = _run_all_methods(batch.iq_h1, simulation_config, r3_config)
                for method in METHODS_ALL:
                    h1_parts[method].append(_score_vector(h1_outputs[method]))
                    h1_range_parts[method].append(_score_vector(h1_outputs[method], "range_cell"))
                    h1_doppler_parts[method].append(_score_vector(h1_outputs[method], "doppler_hz"))
                truth_range_parts.append(np.asarray(batch.truth_target_range_cell, dtype=np.float64))
                truth_doppler_parts.append(np.asarray(batch.truth_target_doppler_hz, dtype=np.float64))
                id_parts.append(np.asarray(batch.base_scene_id).astype(str))

                freq_axis = np.fft.fftshift(
                    np.fft.fftfreq(int(simulation_config["radar"]["pulse_count"]),
                                   d=1.0 / float(simulation_config["radar"]["prf_hz"]))
                )
                for method in ("T1", "T2"):
                    h0_bg = np.asarray(h0_outputs[method].background_estimate)
                    h1_bg = np.asarray(h1_outputs[method].background_estimate)
                    if method == "T1":
                        background0 = h0_bg[:, 0, 0]
                        background1 = h1_bg[:, 0, 0]
                    else:
                        background0 = np.empty(batch.iq_h0.shape[0], dtype=np.float64)
                        background1 = np.empty_like(background0)
                        for scene_index in range(batch.iq_h0.shape[0]):
                            doppler_index = int(np.argmin(np.abs(freq_axis - batch.truth_target_doppler_hz[scene_index])))
                            range_index = int(np.clip(round(float(batch.truth_target_range_cell[scene_index])), 0,
                                                      h0_bg.shape[2] - 1))
                            background0[scene_index] = h0_bg[scene_index, doppler_index, range_index]
                            background1[scene_index] = h1_bg[scene_index, doppler_index, range_index]
                    relative_change = np.abs(background1 - background0) / np.maximum(background0, 1e-30)
                    reference_contamination[group["group_id"]][method].extend(relative_change.tolist())

                for scene_index, scene_id in enumerate(batch.base_scene_id):
                    row = {
                        "group_id": group["group_id"],
                        "scenario_id": group["scenario_id"],
                        "snr_db": group["snr_db"],
                        "cnr_db": group["cnr_db"],
                        "base_scene_id": str(scene_id),
                        "truth_range_cell": float(batch.truth_target_range_cell[scene_index]),
                        "truth_doppler_hz": float(batch.truth_target_doppler_hz[scene_index]),
                    }
                    for method in METHODS_ALL:
                        row[f"h0_score_{method}"] = float(_score_vector(h0_outputs[method])[scene_index])
                        row[f"h1_score_{method}"] = float(_score_vector(h1_outputs[method])[scene_index])
                        row[f"h1_range_{method}"] = float(_score_vector(h1_outputs[method], "range_cell")[scene_index])
                        row[f"h1_doppler_{method}"] = float(_score_vector(h1_outputs[method], "doppler_hz")[scene_index])
                    h1_scene_rows.append(row)
        group_data[group["group_id"]] = {
            "h0_scores": {method: np.concatenate(parts) for method, parts in h0_parts.items()},
        }
        replayed_ids_by_group[group["group_id"]] = group_replayed_ids
        if group["role"] == "validation":
            group_data[group["group_id"]].update(
                {
                    "h1_scores": {method: np.concatenate(parts) for method, parts in h1_parts.items()},
                    "h1_range": {method: np.concatenate(parts) for method, parts in h1_range_parts.items()},
                    "h1_doppler": {method: np.concatenate(parts) for method, parts in h1_doppler_parts.items()},
                    "truth_range_cell": np.concatenate(truth_range_parts),
                    "truth_doppler_hz": np.concatenate(truth_doppler_parts),
                    "base_scene_id": np.concatenate(id_parts),
                }
            )
        print(
            f"[candidates {index}/{len([g for g in groups if g['role'] != 'scale_diagnostic'])}] "
            f"{group['group_id']} n={group['scene_count']}", flush=True,
        )

    assert_replayed_scene_coverage(h0_rows, groups, replayed_ids_by_group)

    thresholds_by_group: dict[str, dict[str, dict[str, Any]]] = {}
    for group in [group for group in groups if group["role"] == "calibration"]:
        thresholds_by_group[group["group_id"]] = {}
        for method in METHODS_ALL:
            result = calibrate_empirical_threshold(
                group_data[group["group_id"]]["h0_scores"][method],
                float(r3_config["detection"]["target_pfa_cpi"]),
            )
            if method in METHODS_T0:
                previous = float(raw_metrics["thresholds_by_group"][group["group_id"]][method]["threshold"])
                if result["threshold"] != previous:
                    raise ValueError(f"raw-phase calibration threshold changed for {group['group_id']}/{method}")
            thresholds_by_group[group["group_id"]][method] = result

    candidate_h0_rows = _candidate_threshold_validation_rows(
        group_data, groups, thresholds_by_group, r3_config
    )
    amplitude_rows, g3 = _candidate_amplitude_diagnostics(
        h0_rows, groups, thresholds_by_group, r3_config
    )
    g2 = _candidate_g2(thresholds_by_group, candidate_h0_rows, groups, r3_config)
    paired_effect_rows, g4 = _candidate_g4(group_data, groups, thresholds_by_group, r3_config)
    gates = _candidate_gates(candidate_h0_rows, g2, g3, g4)

    h1_metric_rows: list[dict[str, Any]] = []
    auc_cache: dict[tuple[str, str], tuple[float, tuple[float, float]]] = {}
    for group in [group for group in groups if group["role"] == "validation"]:
        data = group_data[group["group_id"]]
        matched_cal = [cal for cal in groups if cal["role"] == "calibration" and cal["cnr_db"] == group["cnr_db"]]
        for method in METHODS_ALL:
            h0_scores = data["h0_scores"][method]
            h1_scores = data["h1_scores"][method]
            auc_cache[(group["group_id"], method)] = (
                float(empirical_auc(h0_scores, h1_scores)),
                _paired_auc_interval(
                    h0_scores, h1_scores, data["base_scene_id"],
                    replicates=int(r3_config["detection"]["bootstrap_replicates"]),
                    seed=2026102901 + METHODS_ALL.index(method),
                ),
            )
            threshold_entries = [
                (cal["group_id"], float(thresholds_by_group[cal["group_id"]][method]["threshold"]))
                for cal in matched_cal
            ]
            if method in METHODS_T0:
                threshold_entries.append(
                    ("E001_frozen_matched_CNR", _legacy_threshold(r3_config, group["cnr_db"], method))
                )
            for threshold_source, threshold in threshold_entries:
                summary = summarize_r3_scores(
                    h0_scores,
                    h1_scores,
                    data["h1_range"][method],
                    data["h1_doppler"][method],
                    data["truth_range_cell"],
                    data["truth_doppler_hz"],
                    data["base_scene_id"],
                    threshold=threshold,
                    prf_hz=float(r3_config["radar"]["prf_hz"]),
                    bootstrap_replicates=int(r3_config["detection"]["bootstrap_replicates"]),
                    seed=2026103001 + METHODS_ALL.index(method),
                    auc_summary=auc_cache[(group["group_id"], method)],
                )
                h1_metric_rows.append(
                    {
                        "record_type": "h1_metrics",
                        "phase": "candidate",
                        "method": method,
                        "threshold_source": threshold_source,
                        "group_id": group["group_id"],
                        "scenario_id": group["scenario_id"],
                        "snr_db": group["snr_db"],
                        "cnr_db": group["cnr_db"],
                        **summary,
                    }
                )

    control_group = next(group for group in groups if group["role"] == "scale_diagnostic")
    controlled_scale = _candidate_scale_diagnostic(control_group, simulation_config, r3_config)
    contamination_summary = {
        group_id: {
            method: {
                "median_relative_background_change_h0_to_h1": float(np.median(values)),
                "p95_relative_background_change_h0_to_h1": float(np.quantile(values, 0.95)),
                "n_scenes": len(values),
            }
            for method, values in methods.items()
        }
        for group_id, methods in reference_contamination.items()
    }
    candidate_phase = {
        "completed": True,
        "methods": list(METHODS_ALL),
        "thresholds_by_group": thresholds_by_group,
        "validation_threshold_results": candidate_h0_rows,
        "amplitude_pfa_rows": amplitude_rows,
        "h1_results": h1_metric_rows,
        "paired_h1_effects_vs_b3": paired_effect_rows,
        "reference_background_contamination": contamination_summary,
        "controlled_scale": controlled_scale,
        "gates": {"G2": g2, "G3": g3, "G4": g4, "summary": gates},
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    metrics["candidate_phase"] = candidate_phase
    _write_csv(output_dir / "h0_scene_diagnostics.csv", h0_rows)
    _write_csv(output_dir / "h1_scene_results.csv", h1_scene_rows)
    all_result_rows = _read_csv_rows(output_dir / "result.csv") + candidate_h0_rows + h1_metric_rows + paired_effect_rows
    _write_csv(output_dir / "result.csv", all_result_rows)
    _write_json(output_dir / "metrics.json", metrics)
    _write_candidate_analysis(output_dir, metrics)
    _plot_candidate_results(output_dir, candidate_h0_rows, h1_metric_rows, amplitude_rows)
    return candidate_phase


def _write_candidate_analysis(output_dir: Path, metrics: dict[str, Any]) -> None:
    candidate = metrics["candidate_phase"]
    gate_summary = candidate["gates"]["summary"]
    lines = [
        "# D2-A-R3 阶段结果分析",
        "",
        "本文件由预注册代码生成初稿；阶段结论须逐项核对 `metrics.json`、CSV、代码和统计区间后完善。",
        "",
        f"- T0 原始 H0 分数场景记录：{sum(int(row.get('n', 0)) for row in candidate['validation_threshold_results'] if row.get('record_type') == 'combined_validation')} 个 score-threshold 评估。",
        f"- R3 候选统计量：T1/T2；总体前置门选择：`{gate_summary['selected_b3_based_statistic']}`。",
        "- T1/T2 的物理前端都是 B3 DPCA，不能解释成纯学习结果。",
        "- D 场景是同流形来源不可辨识参考，不是唯一源归因门槛。",
        "",
        "## 前置门汇总",
        "",
        "| 方法 | G2 虚警 | G3 幅度稳定性 | G4 Pd/联合性能保护 | 总门 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, gates in gate_summary["methods"].items():
        lines.append(
            f"| {method} | {gates['G2']} | {gates['G3']} | {gates['G4']} | {gates['all_front_gates']} |"
        )
    lines.extend(
        [
            "",
            "请补充十个研究问题的证据化回答、对有限样本和尺度 nuisance 的归因、教师可用性以及唯一下一步。",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_candidate_results(
    output_dir: Path,
    h0_rows: list[dict[str, Any]],
    h1_rows: list[dict[str, Any]],
    amplitude_rows: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    combined = [row for row in h0_rows if row.get("record_type") == "combined_validation"]
    methods = METHODS_ALL
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for method in methods:
        selected = [row for row in combined if row["method"] == method and row["threshold_source"] == row["group_id"].replace("_all_validation", "_cal")]
        if not selected:
            selected = [row for row in combined if row["method"] == method]
        by_cnr: dict[float, list[float]] = {}
        for row in selected:
            by_cnr.setdefault(float(row["cnr_db"]), []).append(float(row["pfa_ge"]))
        if by_cnr:
            x = sorted(by_cnr)
            y = [float(np.mean(by_cnr[key])) for key in x]
            axes[0].plot(x, y, marker="o", label=method)
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1)
    axes[0].set(title="Matched-calibration H0 Pfa_CPI", xlabel="CNR (dB)", ylabel="Pfa_CPI")
    axes[0].legend()
    for method in methods:
        selected = [row for row in h1_rows if row["method"] == method and row["threshold_source"] != "E001_frozen_matched_CNR"]
        grouped: dict[str, list[float]] = {}
        for row in selected:
            grouped.setdefault(str(row["scenario_id"]), []).append(float(row["pd"]))
        if grouped:
            x = list(grouped)
            y = [float(np.mean(grouped[key])) for key in x]
            axes[1].plot(x, y, marker="o", label=method)
    axes[1].set(title="Pd at matched R3 thresholds", xlabel="Identifiability scenario", ylabel="Pd", ylim=(0, 1.02))
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "pfa_pd_by_condition.png", dpi=150)
    plt.close(fig)

    bins = [row for row in amplitude_rows if isinstance(row.get("amplitude_bin_index"), int)]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        selected = [row for row in bins if row["method"] == method]
        if not selected:
            continue
        by_bin: dict[int, list[float]] = {}
        for row in selected:
            by_bin.setdefault(int(row["amplitude_bin_index"]), []).append(float(row["pfa_cpi"]))
        x = sorted(by_bin)
        ax.plot(x, [float(np.mean(by_bin[key])) for key in x], marker="o", label=method)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
    ax.set(title="H0 Pfa by latent reference-amplitude bin", xlabel="Amplitude bin index", ylabel="Pfa_CPI")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "pfa_by_amplitude.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/D2-A-R3/config.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/D2-A-R3")
    parser.add_argument("--phase", choices=("raw", "candidates"), required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.phase == "raw":
        run_raw_phase(config, args.output_dir)
    else:
        run_candidate_phase(config, args.output_dir)


if __name__ == "__main__":
    main()
