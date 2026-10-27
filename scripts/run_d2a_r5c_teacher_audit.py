#!/usr/bin/env python3
"""Run the preregistered R5-C teacher-response audit on new paired scenes."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d2a_e001 import cluster_bootstrap_interval, empirical_auc
from radardistill.simulation.d2a import D2APairedBatch, generate_d2a_paired_batch
from radardistill.teachers.d2a_k2_cuda import k2_response_cuda


CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-C/config.yaml"
PROTOCOL_PATH = ROOT / "experiments/D2-A-R5/R5-C/protocol.md"
OUTPUT_ROOT = ROOT / "experiments/D2-A-R5/R5-C"
PHYSICS_PATH = ROOT / "experiments/D2-A-E001/config.yaml"
FROZEN_FILES = (
    "docs/RESEARCH_CHARTER.md",
    "docs/task_alignments/D2-A-R5C_alignment.md",
    "experiments/D2-A-E001/config.yaml",
    "experiments/D2-A-R5/R5-C/protocol.md",
    "experiments/D2-A-R5/R5-C/config.yaml",
    "src/radardistill/d2a_baselines.py",
    "src/radardistill/evaluation/d2a_e001.py",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/teachers/d2a_k2.py",
    "src/radardistill/teachers/d2a_k2_cuda.py",
    "scripts/run_d2a_r5c_teacher_audit.py",
)
_ID_SEED = re.compile(r"^d2a-scene-(\d+)-")
_EXPECTED_CONDITIONS = (
    ("C0", "separated", 20.0, -5.0, 256),
    ("C1", "separated", 15.0, 0.0, 128),
    ("C1", "near_clutter_edge", 15.0, 0.0, 128),
    ("C2", "separated", 10.0, 10.0, 256),
    ("C2", "near_clutter_edge", 10.0, 10.0, 256),
    ("C2", "partial_overlap", 10.0, 10.0, 256),
    ("C3", "separated", 0.0, 20.0, 512),
    ("C3", "separated", 10.0, 20.0, 512),
    ("C3", "near_clutter_edge", 0.0, 20.0, 512),
    ("C3", "near_clutter_edge", 10.0, 20.0, 512),
    ("C3", "partial_overlap", 0.0, 20.0, 512),
    ("C3", "partial_overlap", 10.0, 20.0, 512),
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


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
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(_json_safe(row) for row in rows)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _repo_size_bytes() -> int:
    result = subprocess.run(
        ["du", "-sb", str(ROOT)], check=True, text=True, capture_output=True
    )
    return int(result.stdout.split()[0])


def _validate_config(config: dict[str, Any]) -> None:
    experiment = config.get("experiment", {})
    data = config.get("data", {})
    teacher = config.get("teacher", {})
    evaluation = config.get("evaluation", {})
    resources = config.get("resources", {})
    if experiment.get("id") != "D2-A-R5-C" or float(experiment.get("protocol_version", -1)) != 1.1:
        raise ValueError("R5-C config must match protocol v1.1")
    if any(
        experiment.get(key) is not False
        for key in ("student_training_authorized", "distillation_authorized", "final_blind_generation_authorized")
    ):
        raise ValueError("R5-C teacher audit does not authorize training, KD, or final blind")
    actual_conditions = tuple(
        (
            str(row.get("course")),
            str(row.get("scenario_id")),
            float(row.get("snr_db")),
            float(row.get("cnr_db")),
            int(row.get("scenes")),
        )
        for row in data.get("conditions", ())
    )
    if actual_conditions != _EXPECTED_CONDITIONS:
        actual_total = sum(row[-1] for row in actual_conditions)
        raise ValueError(
            "R5-C conditions differ from the frozen 12-stratum allocation "
            f"(registered total must be 4352; found {actual_total})"
        )
    if sum(row[-1] for row in actual_conditions) != 4352 or int(data.get("total_paired_base_scenes", -1)) != 4352:
        raise ValueError("R5-C registered sample total must remain 4352 paired base scenes")
    if int(data.get("batch_size", -1)) != 16:
        raise ValueError("R5-C CUDA teacher batch size is frozen at 16")
    if (
        int(data.get("seed_block_start", -1)) != 726000000
        or int(data.get("seed_block_stride", -1)) != 100000
        or int(data.get("chunk_seed_stride", -1)) != 1009
    ):
        raise ValueError("R5-C scene seed allocation differs from the frozen protocol")
    if teacher.get("device") != "cuda:0" or teacher.get("cpu_fallback") != "forbidden":
        raise ValueError("R5-C teacher execution requires cuda:0 with CPU fallback forbidden")
    if tuple(teacher.get("input_fields", ())) != ("observable_iq_only", "frozen_physics_config"):
        raise ValueError("R5-C teacher input must contain observable IQ and frozen physics config only")
    if bool(evaluation.get("reliability_threshold_tuning", True)):
        raise ValueError("R5-C cannot tune a reliability threshold")
    if int(resources.get("max_batch_size", -1)) != 16 or resources.get("preserve_untracked_user_files") is not True:
        raise ValueError("R5-C resource and user-data protections differ from the protocol")


def _build_batch_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    _validate_config(config)
    batch_size = int(config["data"]["batch_size"])
    block_start = int(config["data"]["seed_block_start"])
    block_stride = int(config["data"]["seed_block_stride"])
    chunk_seed_stride = int(config["data"]["chunk_seed_stride"])
    plan: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(config["data"]["conditions"]):
        count = int(condition["scenes"])
        if count % batch_size:
            raise ValueError("every R5-C condition count must be divisible by the fixed batch size")
        seed_block = block_start + condition_index * block_stride
        if ((count // batch_size) - 1) * chunk_seed_stride >= block_stride:
            raise ValueError("R5-C chunk seeds would overlap the next condition seed block")
        for chunk_index in range(count // batch_size):
            plan.append(
                {
                    **condition,
                    "condition_index": condition_index,
                    "chunk_index": chunk_index,
                    "scene_count": batch_size,
                    "seed_block_start": seed_block,
                    "seed": seed_block + chunk_index * chunk_seed_stride,
                }
            )
    return plan


def _validate_pair_batch(
    batch: D2APairedBatch,
    *,
    expected_count: int,
    expected_seed: int | None = None,
) -> list[str]:
    ids = [str(value) for value in np.asarray(batch.base_scene_id).reshape(-1)]
    if len(ids) != expected_count or len(set(ids)) != expected_count:
        raise ValueError("generated paired scene IDs must be unique and match the registered batch size")
    if expected_seed is not None:
        expected_ids = {f"d2a-scene-{expected_seed}-{index}" for index in range(expected_count)}
        if set(ids) != expected_ids:
            raise ValueError("generated paired scene IDs do not match the preregistered batch seed")
    expected_shape = (expected_count, 3, 130, 32)
    for name in ("iq_h0", "iq_h1", "target_signal_iq"):
        value = np.asarray(getattr(batch, name))
        if value.shape != expected_shape or not np.iscomplexobj(value) or np.any(~np.isfinite(value)):
            raise ValueError(f"generated {name} is not finite complex IQ with shape {expected_shape}")
    if not np.allclose(
        np.asarray(batch.iq_h1),
        np.asarray(batch.iq_h0) + np.asarray(batch.target_signal_iq),
        rtol=0.0,
        atol=1e-6,
    ):
        raise ValueError("paired H1 must equal H0 plus the simulated target IQ")
    return ids


def _wrap_hz(values: np.ndarray | float, prf_hz: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.remainder(array + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def _window_mask(
    *,
    truth_range_cell: float,
    truth_doppler_hz: float,
    frequency_axis: np.ndarray,
    range_count: int,
    prf_hz: float,
) -> np.ndarray:
    doppler_error = _wrap_hz(frequency_axis - truth_doppler_hz, prf_hz)
    range_axis = np.arange(range_count, dtype=np.float64)
    return (
        (np.abs(doppler_error[:, None]) <= 20.0)
        & (np.abs(range_axis[None, :] - truth_range_cell) <= 1.0)
    )


def _map_response_metrics(
    *,
    candidate: str,
    course: str,
    scenario_id: str,
    snr_db: float,
    cnr_db: float,
    base_scene_ids: np.ndarray,
    h0_response: np.ndarray,
    h1_response: np.ndarray,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    prf_hz: float,
    fft_factor: int,
) -> list[dict[str, Any]]:
    ids = np.asarray(base_scene_ids).astype(str).reshape(-1)
    h0 = np.asarray(h0_response, dtype=np.float64)
    h1 = np.asarray(h1_response, dtype=np.float64)
    ranges = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    frequencies = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    count = ids.size
    frequency_count = 130 * fft_factor
    if (
        count == 0
        or h0.shape != (count, frequency_count, 32)
        or h1.shape != h0.shape
        or ranges.shape != (count,)
        or frequencies.shape != (count,)
    ):
        raise ValueError("R5-C response maps and truth arrays must share the registered [B,F,32] shape")
    if any(np.any(~np.isfinite(value)) for value in (h0, h1, ranges, frequencies)):
        raise ValueError("R5-C response maps and truth arrays must be finite")
    for response in (h0, h1):
        totals = response.sum(axis=(1, 2))
        if np.any(response < 0.0) or np.any(np.abs(totals - 1.0) > 1e-6):
            raise ValueError("R5-C support maps must be nonnegative and normalized per CPI")
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(frequency_count, d=1.0 / prf_hz))
    rows: list[dict[str, Any]] = []
    for index, scene_id in enumerate(ids):
        truth_mask = _window_mask(
            truth_range_cell=float(ranges[index]),
            truth_doppler_hz=float(frequencies[index]),
            frequency_axis=frequency_axis,
            range_count=32,
            prf_hz=prf_hz,
        )
        h0_mass = float(h0[index][truth_mask].sum())
        h1_mass = float(h1[index][truth_mask].sum())
        flat_peak = int(np.argmax(h1[index]))
        peak_frequency_index, peak_range = divmod(flat_peak, 32)
        peak_frequency = float(frequency_axis[peak_frequency_index])
        range_error = float(peak_range - ranges[index])
        doppler_error = float(_wrap_hz(peak_frequency - frequencies[index], prf_hz))
        confidence_mask = _window_mask(
            truth_range_cell=float(peak_range),
            truth_doppler_hz=peak_frequency,
            frequency_axis=frequency_axis,
            range_count=32,
            prf_hz=prf_hz,
        )
        confidence = float(h1[index][confidence_mask].sum())
        h0_peak = int(np.argmax(h0[index]))
        h0_f, h0_r = divmod(h0_peak, 32)
        h0_doppler_error = float(_wrap_hz(frequency_axis[h0_f] - frequencies[index], prf_hz))
        h0_hit = abs(h0_r - ranges[index]) <= 1.0 and abs(h0_doppler_error) <= 20.0
        rows.append(
            {
                "base_scene_id": scene_id,
                "candidate": candidate,
                "course": course,
                "scenario_id": scenario_id,
                "condition_id": f"{course}_{scenario_id}_snr_{snr_db:g}_cnr_{cnr_db:g}",
                "snr_db": float(snr_db),
                "cnr_db": float(cnr_db),
                "truth_range_cell": float(ranges[index]),
                "truth_doppler_hz": float(frequencies[index]),
                "h0_window_mass": h0_mass,
                "h1_window_mass": h1_mass,
                "target_window_mass_lift": h1_mass - h0_mass,
                "predicted_range_cell": int(peak_range),
                "predicted_doppler_hz": peak_frequency,
                "range_abs_error_cells": abs(range_error),
                "doppler_wrapped_error_hz": doppler_error,
                "top_cell_hit": bool(abs(range_error) <= 1.0 and abs(doppler_error) <= 20.0),
                "h0_pseudo_target_hit": bool(h0_hit),
                "response_confidence": confidence,
            }
        )
    return rows


def _local_response_metrics(
    *,
    course: str,
    scenario_id: str,
    snr_db: float,
    cnr_db: float,
    base_scene_ids: np.ndarray,
    h0_response: np.ndarray,
    h1_response: np.ndarray,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    prf_hz: float,
) -> list[dict[str, Any]]:
    ids = np.asarray(base_scene_ids).astype(str).reshape(-1)
    h0 = np.asarray(h0_response, dtype=np.float64)
    h1 = np.asarray(h1_response, dtype=np.float64)
    ranges = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    frequencies = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    if h0.shape != (ids.size, 520, 32) or h1.shape != h0.shape:
        raise ValueError("K2-local requires paired 4x B3 response maps with shape [B,520,32]")
    if ranges.shape != ids.shape or frequencies.shape != ids.shape:
        raise ValueError("K2-local truth arrays must match the paired scene IDs")
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(520, d=1.0 / prf_hz))
    range_axis = np.arange(32)
    rows: list[dict[str, Any]] = []
    for index, scene_id in enumerate(ids):
        h0_peak = int(np.argmax(h0[index]))
        h1_peak = int(np.argmax(h1[index]))
        h0_f, h0_r = divmod(h0_peak, 32)
        h1_f, h1_r = divmod(h1_peak, 32)
        h0_local = h0[index, :, h0_r]
        h1_local = h1[index, :, h1_r]
        h0_total = float(h0_local.sum())
        h1_total = float(h1_local.sum())
        if h0_total <= 0.0 or h1_total <= 0.0:
            raise ValueError("K2-local cannot normalize a zero predicted-range response")
        h0_local = h0_local / h0_total
        h1_local = h1_local / h1_total
        h1_frequency = float(frequency_axis[h1_f])
        range_error = float(h1_r - ranges[index])
        doppler_error = float(_wrap_hz(h1_frequency - frequencies[index], prf_hz))
        truth_frequency_window = np.abs(_wrap_hz(frequency_axis - frequencies[index], prf_hz)) <= 20.0
        h0_range_ok = abs(h0_r - ranges[index]) <= 1.0
        h1_range_ok = abs(range_error) <= 1.0
        h0_mass = float(h0_local[truth_frequency_window].sum()) if h0_range_ok else 0.0
        h1_mass = float(h1_local[truth_frequency_window].sum()) if h1_range_ok else 0.0
        confidence_window = np.abs(_wrap_hz(frequency_axis - h1_frequency, prf_hz)) <= 20.0
        confidence = float(h1_local[confidence_window].sum())
        h0_frequency = float(frequency_axis[h0_f])
        h0_doppler_error = float(_wrap_hz(h0_frequency - frequencies[index], prf_hz))
        rows.append(
            {
                "base_scene_id": scene_id,
                "candidate": "K2_local",
                "course": course,
                "scenario_id": scenario_id,
                "condition_id": f"{course}_{scenario_id}_snr_{snr_db:g}_cnr_{cnr_db:g}",
                "snr_db": float(snr_db),
                "cnr_db": float(cnr_db),
                "truth_range_cell": float(ranges[index]),
                "truth_doppler_hz": float(frequencies[index]),
                "h0_window_mass": h0_mass,
                "h1_window_mass": h1_mass,
                "target_window_mass_lift": h1_mass - h0_mass,
                "predicted_range_cell": int(h1_r),
                "predicted_doppler_hz": h1_frequency,
                "range_abs_error_cells": abs(range_error),
                "doppler_wrapped_error_hz": doppler_error,
                "top_cell_hit": bool(h1_range_ok and abs(doppler_error) <= 20.0),
                "h0_pseudo_target_hit": bool(h0_range_ok and abs(h0_doppler_error) <= 20.0),
                "response_confidence": confidence,
            }
        )
    return rows


def _rmse(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2))) if array.size else float("nan")


def _confidence_reliability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confidence = np.asarray([row["response_confidence"] for row in rows], dtype=np.float64)
    correct = np.asarray([row["top_cell_hit"] for row in rows], dtype=np.bool_)
    auc = None
    if np.any(correct) and np.any(~correct):
        auc = float(empirical_auc(confidence[~correct], confidence[correct]))
    edges = np.quantile(confidence, [0.25, 0.5, 0.75])
    bins = np.digitize(confidence, edges, right=True)
    reliability_rows = []
    for index in range(4):
        selected = bins == index
        reliability_rows.append(
            {
                "quartile": index + 1,
                "n": int(np.count_nonzero(selected)),
                "mean_confidence": float(np.mean(confidence[selected])) if np.any(selected) else None,
                "localization_rate": float(np.mean(correct[selected])) if np.any(selected) else None,
            }
        )
    rank_corr = None
    if np.std(confidence) > 0.0 and np.std(correct.astype(np.float64)) > 0.0:
        order = np.argsort(confidence, kind="mergesort")
        ranks = np.empty(confidence.size, dtype=np.float64)
        sorted_confidence = confidence[order]
        start = 0
        while start < confidence.size:
            stop = start + 1
            while stop < confidence.size and sorted_confidence[stop] == sorted_confidence[start]:
                stop += 1
            ranks[order[start:stop]] = (start + stop - 1) / 2.0
            start = stop
        rank_corr = float(np.corrcoef(ranks, correct.astype(np.float64))[0, 1])
    return {
        "confidence_auc": auc,
        "confidence_rank_correlation": rank_corr,
        "quartiles": reliability_rows,
    }


def _summarize_group(
    rows: list[dict[str, Any]], *, bootstrap_replicates: int, bootstrap_seed: int
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty R5-C scene group")
    ids = np.asarray([row["base_scene_id"] for row in rows])
    lift = np.asarray([row["target_window_mass_lift"] for row in rows], dtype=np.float64)
    hit = np.asarray([row["top_cell_hit"] for row in rows], dtype=np.float64)
    range_error = np.asarray([row["range_abs_error_cells"] for row in rows], dtype=np.float64)
    doppler_error = np.asarray([row["doppler_wrapped_error_hz"] for row in rows], dtype=np.float64)
    correct = hit.astype(np.bool_)
    if not np.any(correct):
        range_hit_rmse = None
        doppler_hit_rmse = None
    else:
        range_hit_rmse = _rmse(range_error[correct])
        doppler_hit_rmse = _rmse(doppler_error[correct])
    lift_ci = cluster_bootstrap_interval(
        lift, ids, statistic=np.mean, replicates=bootstrap_replicates, seed=bootstrap_seed
    )
    hit_ci = cluster_bootstrap_interval(
        hit, ids, statistic=np.mean, replicates=bootstrap_replicates, seed=bootstrap_seed + 1
    )
    range_ci = cluster_bootstrap_interval(
        range_error,
        ids,
        statistic=_rmse,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 2,
    )
    doppler_ci = cluster_bootstrap_interval(
        doppler_error,
        ids,
        statistic=_rmse,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 3,
    )
    confidence = _confidence_reliability(rows)
    return {
        "n_paired_scenes": len(rows),
        "target_window_mass_h0_mean": float(np.mean([row["h0_window_mass"] for row in rows])),
        "target_window_mass_h1_mean": float(np.mean([row["h1_window_mass"] for row in rows])),
        "target_window_mass_lift_mean": float(np.mean(lift)),
        "target_window_mass_lift_ci95": [float(value) for value in lift_ci],
        "teacher_top_cell_localization_rate": float(np.mean(hit)),
        "teacher_top_cell_localization_ci95": [float(value) for value in hit_ci],
        "range_rmse_all_h1_cells": _rmse(range_error),
        "range_rmse_ci95": [float(value) for value in range_ci],
        "doppler_rmse_all_h1_hz": _rmse(doppler_error),
        "doppler_rmse_ci95": [float(value) for value in doppler_ci],
        "range_rmse_when_jointly_localized": range_hit_rmse,
        "doppler_rmse_when_jointly_localized_hz": doppler_hit_rmse,
        "h0_pseudo_target_top_hit_rate": float(np.mean([row["h0_pseudo_target_hit"] for row in rows])),
        **confidence,
    }


def _paired_rmse_difference_ci(
    error_a: np.ndarray,
    error_b: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    ids = np.asarray(cluster_ids).reshape(-1)
    unique_ids = np.unique(ids)
    clusters = {scene_id: np.flatnonzero(ids == scene_id) for scene_id in unique_ids}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    a = np.asarray(error_a, dtype=np.float64)
    b = np.asarray(error_b, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.choice(unique_ids, size=unique_ids.size, replace=True)
        indices = np.concatenate([clusters[scene_id] for scene_id in selected])
        estimates[replicate] = _rmse(a[indices]) - _rmse(b[indices])
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _primary_contrasts(
    rows: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    contrasts: dict[str, Any] = {}
    for scope, predicate in (
        ("identifiable_C2_C3", lambda row: row["course"] in {"C2", "C3"}),
        ("C3", lambda row: row["course"] == "C3"),
    ):
        selected = [row for row in rows if predicate(row)]
        by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
        for row in selected:
            by_candidate.setdefault(row["candidate"], {}).setdefault(row["base_scene_id"], {})["row"] = row
        ids_k1 = set(by_candidate.get("K1", {}))
        ids_k3 = set(by_candidate.get("K3", {}))
        if ids_k1 != ids_k3 or not ids_k1:
            raise ValueError("K1/K3 paired-scene IDs must match exactly for the primary contrast")
        ordered_ids = np.asarray(sorted(ids_k1))
        k1 = [by_candidate["K1"][scene_id]["row"] for scene_id in ordered_ids]
        k3 = [by_candidate["K3"][scene_id]["row"] for scene_id in ordered_ids]
        localization_delta = np.asarray(
            [float(b["top_cell_hit"]) - float(a["top_cell_hit"]) for a, b in zip(k1, k3)],
            dtype=np.float64,
        )
        lift_delta = np.asarray(
            [b["target_window_mass_lift"] - a["target_window_mass_lift"] for a, b in zip(k1, k3)],
            dtype=np.float64,
        )
        loc_ci = cluster_bootstrap_interval(
            localization_delta,
            ordered_ids,
            statistic=np.mean,
            replicates=replicates,
            seed=seed,
        )
        lift_ci = cluster_bootstrap_interval(
            lift_delta,
            ordered_ids,
            statistic=np.mean,
            replicates=replicates,
            seed=seed + 1,
        )
        doppler_delta = _rmse(
            np.asarray([row["doppler_wrapped_error_hz"] for row in k3])
        ) - _rmse(np.asarray([row["doppler_wrapped_error_hz"] for row in k1]))
        doppler_ci = _paired_rmse_difference_ci(
            np.asarray([row["doppler_wrapped_error_hz"] for row in k3]),
            np.asarray([row["doppler_wrapped_error_hz"] for row in k1]),
            ordered_ids,
            replicates=replicates,
            seed=seed + 2,
        )
        contrasts[scope] = {
            "n_paired_scenes": int(ordered_ids.size),
            "K3_minus_K1_localization_rate": float(np.mean(localization_delta)),
            "K3_minus_K1_localization_rate_ci95": [float(value) for value in loc_ci],
            "K3_minus_K1_target_window_lift": float(np.mean(lift_delta)),
            "K3_minus_K1_target_window_lift_ci95": [float(value) for value in lift_ci],
            "K3_minus_K1_doppler_rmse_hz": float(doppler_delta),
            "K3_minus_K1_doppler_rmse_ci95_hz": [float(value) for value in doppler_ci],
        }
    return contrasts


def _registered_summaries(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    conditions = config["data"]["conditions"]
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    scopes: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("identifiable_C2_C3", lambda row: row["course"] in {"C2", "C3"}),
        ("C3", lambda row: row["course"] == "C3"),
    ]
    for condition in conditions:
        condition_id = (
            f"{condition['course']}_{condition['scenario_id']}_"
            f"snr_{float(condition['snr_db']):g}_cnr_{float(condition['cnr_db']):g}"
        )
        scopes.append((condition_id, lambda row, key=condition_id: row["condition_id"] == key))
    result: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(("K1", "K3", "K2_local")):
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        if not candidate_rows:
            raise ValueError(f"R5-C output has no rows for {candidate}")
        candidate_result = {}
        for scope_index, (scope_name, predicate) in enumerate(scopes):
            selected = [row for row in candidate_rows if predicate(row)]
            candidate_result[scope_name] = _summarize_group(
                selected,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + candidate_index * 100 + scope_index * 10,
            )
        result[candidate] = candidate_result
    contrasts = _primary_contrasts(
        rows,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 1000,
    )
    gates: dict[str, Any] = {}
    for candidate in ("K1", "K3", "K2_local"):
        summary = result[candidate]["identifiable_C2_C3"]
        c3_groups = [
            result[candidate][
                f"C3_{condition['scenario_id']}_snr_{float(condition['snr_db']):g}_cnr_{float(condition['cnr_db']):g}"
            ]
            for condition in conditions
            if condition["course"] == "C3"
        ]
        c3_scope_names = [
            f"C3_{condition['scenario_id']}_snr_{float(condition['snr_db']):g}_cnr_{float(condition['cnr_db']):g}"
            for condition in conditions
            if condition["course"] == "C3"
        ]
        severe = []
        for scope_name, group in zip(c3_scope_names, c3_groups):
            if (
                group["target_window_mass_lift_ci95"][1] <= 0.0
                and group["teacher_top_cell_localization_rate"] <= 0.05
            ):
                severe.append(
                    {
                        "condition": scope_name,
                        "confidence_auc": group["confidence_auc"],
                        "reliability_salvage": bool(
                            group["confidence_auc"] is not None
                            and group["confidence_auc"] >= float(config["evaluation"]["reliability_salvage_minimum_auc"])
                        ),
                    }
                )
        info_pass = summary["target_window_mass_lift_ci95"][0] > 0.0
        reliability_pass = all(item["reliability_salvage"] for item in severe)
        gates[candidate] = {
            "identifiable_pooled_information_pass": bool(info_pass),
            "severe_C3_subgroups": severe,
            "unconditional_all_C3_quality_pass": bool(info_pass and not severe),
            "reliability_gated_candidate_pass": bool(info_pass and severe and reliability_pass),
            "G4_knowledge_quality_pass": bool(info_pass and (not severe or reliability_pass)),
            "fixed_threshold_detector_qualification": False,
            "student_KD_authorized": False,
        }
    contrasts["K3_gain_gate"] = {}
    thresholds = config["evaluation"]["minimum_meaningful_gain"]
    for scope_name in ("identifiable_C2_C3", "C3"):
        contrast = contrasts[scope_name]
        localization_pass = (
            contrast["K3_minus_K1_localization_rate"] >= float(thresholds["localization_rate"])
            and contrast["K3_minus_K1_localization_rate_ci95"][0] > 0.0
        )
        doppler_pass = (
            contrast["K3_minus_K1_doppler_rmse_hz"] <= -float(thresholds["circular_doppler_rmse_reduction_hz"])
            and contrast["K3_minus_K1_doppler_rmse_ci95_hz"][1]
            <= -float(thresholds["circular_doppler_rmse_reduction_hz"])
        )
        contrasts["K3_gain_gate"][scope_name] = bool(localization_pass or doppler_pass)
    return {"candidate_summaries": result, "primary_contrasts": contrasts, "g4": gates}


def _manifest_seed_conflicts(
    manifest: Any,
    *,
    reserved_low: int,
    reserved_high: int,
    source: str = "<manifest>",
) -> list[tuple[str, int, str]]:
    conflicts: list[tuple[str, int, str]] = []

    def visit(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, f"{location}.{key}" if location else str(key))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{location}[{index}]")
        elif isinstance(value, int) and not isinstance(value, bool):
            if reserved_low <= value < reserved_high:
                conflicts.append((source, value, location))
        elif isinstance(value, str):
            match = _ID_SEED.match(value)
            if match:
                seed = int(match.group(1))
                if reserved_low <= seed < reserved_high:
                    conflicts.append((source, seed, value))

    visit(manifest, "")
    return conflicts


def _existing_scene_seeds(reserved_low: int, reserved_high: int) -> list[tuple[str, int, str]]:
    conflicts: list[tuple[str, int, str]] = []
    for path in ROOT.glob("experiments/**/base_scene_id.npy"):
        try:
            ids = np.load(path, mmap_mode="r", allow_pickle=False)
        except Exception as error:
            raise RuntimeError(f"cannot inspect existing scene manifest {path}") from error
        for value in ids:
            match = _ID_SEED.match(str(value))
            if match:
                seed = int(match.group(1))
                if reserved_low <= seed < reserved_high:
                    conflicts.append((str(path.relative_to(ROOT)), seed, str(value)))
    for path in ROOT.glob("experiments/**/*.csv"):
        try:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not ({"base_scene_id", "scene_id"} & set(reader.fieldnames)):
                    continue
                field = "base_scene_id" if "base_scene_id" in reader.fieldnames else "scene_id"
                for row in reader:
                    value = str(row.get(field, ""))
                    match = _ID_SEED.match(value)
                    if match:
                        seed = int(match.group(1))
                        if reserved_low <= seed < reserved_high:
                            conflicts.append((str(path.relative_to(ROOT)), seed, value))
        except (OSError, UnicodeDecodeError, csv.Error) as error:
            raise RuntimeError(f"cannot inspect existing scene IDs in {path}") from error
    manifest_paths = sorted(
        set(ROOT.glob("experiments/**/*manifest*.json"))
        | set(ROOT.glob("experiments/**/*seed*.json"))
    )
    for path in manifest_paths:
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"cannot inspect seed metadata in {path}") from error
        conflicts.extend(
            _manifest_seed_conflicts(
                manifest,
                reserved_low=reserved_low,
                reserved_high=reserved_high,
                source=str(path.relative_to(ROOT)),
            )
        )
    return conflicts


def _require_committed_main() -> str:
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R5-C preflight requires the authorized main branch")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError("R5-C requires the frozen protocol and code pushed to origin/main")
    for relative in FROZEN_FILES:
        _git("ls-files", "--error-unmatch", relative)
        status = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT
        ).returncode
        if status != 0:
            raise RuntimeError(f"R5-C frozen input has tracked modifications: {relative}")
    return head


def _cuda_environment(device_name: str, minimum_free: int) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R5-C requires visible CUDA; CPU fallback is forbidden")
    if device_name != "cuda:0":
        raise RuntimeError("R5-C is frozen to cuda:0")
    torch.cuda.set_device(torch.device(device_name))
    device_name_actual = torch.cuda.get_device_name(0)
    if "RTX 3050 Laptop" not in device_name_actual:
        raise RuntimeError(f"R5-C requires RTX 3050 Laptop GPU, found {device_name_actual}")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < minimum_free:
        raise RuntimeError(
            f"R5-C requires {minimum_free} free CUDA bytes, found {free_bytes}"
        )
    return {
        "device": device_name,
        "gpu_name": device_name_actual,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count(),
        "free_bytes_before": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _reset_cuda_peak_memory_stats(device_name: str) -> None:
    import torch

    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)


def run_preflight() -> dict[str, Any]:
    config = _read_yaml(CONFIG_PATH)
    _validate_config(config)
    physics = _read_yaml(PHYSICS_PATH)
    radar = config["radar"]
    if (
        int(physics["radar"]["channel_count"]) != int(radar["channel_count"])
        or int(physics["radar"]["pulse_count"]) != int(radar["pulse_count"])
        or int(physics["radar"]["range_bin_count"]) != int(radar["range_bin_count"])
        or float(physics["radar"]["prf_hz"]) != float(radar["prf_hz"])
        or (int(radar["channel_count"]), int(radar["pulse_count"]), int(radar["range_bin_count"])) != (3, 130, 32)
        or float(radar["prf_hz"]) != 1300.0
    ):
        raise RuntimeError("R5-C physical model differs from the frozen 3×130×32, PRF=1300 signal model")
    git_sha = _require_committed_main()
    seed_low = int(config["data"]["seed_block_start"])
    seed_high = seed_low + len(config["data"]["conditions"]) * int(config["data"]["seed_block_stride"])
    conflicts = _existing_scene_seeds(seed_low, seed_high)
    if conflicts:
        raise RuntimeError(f"R5-C seed blocks overlap existing scene IDs: {conflicts[:5]}")
    cuda = _cuda_environment(
        str(config["teacher"]["device"]), int(config["resources"]["minimum_free_vram_bytes"])
    )
    stat = os.statvfs(ROOT)
    free_disk = int(stat.f_bavail * stat.f_frsize)
    minimum_free_disk = int(config["resources"]["minimum_free_disk_bytes"])
    if free_disk < minimum_free_disk:
        raise RuntimeError(f"R5-C needs {minimum_free_disk} free bytes; only {free_disk} remain")
    project_size = _repo_size_bytes()
    project_limit = int(config["resources"]["project_size_final_limit_gib"]) * 1024**3
    if project_size >= project_limit:
        raise RuntimeError(f"R5-C preflight sees repository size above the final limit: {project_size}")
    plan = _build_batch_plan(config)
    environment = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_sha": git_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cuda": cuda,
        "disk_free_bytes": free_disk,
        "project_size_bytes_before": project_size,
        "project_size_gib_before": project_size / 1024**3,
        "scene_seed_block": [seed_low, seed_high - 1],
        "existing_scene_seed_collisions": len(conflicts),
    }
    result = {
        "protocol_version": 1.1,
        "git_sha": git_sha,
        "branch": "main",
        "pushed_to_origin_main": True,
        "cuda_device": cuda,
        "condition_count": len(config["data"]["conditions"]),
        "batch_count": len(plan),
        "paired_base_scenes": sum(batch["scene_count"] for batch in plan),
        "scene_seed_block": environment["scene_seed_block"],
        "existing_scene_seed_collisions": 0,
        "final_blind": False,
        "student_training": False,
        "distillation": False,
        "passed": True,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_ROOT / "environment.json", environment)
    _write_json(OUTPUT_ROOT / "preflight.json", result)
    return result


def _require_preflight(config: dict[str, Any]) -> dict[str, Any]:
    path = OUTPUT_ROOT / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run R5-C preflight before teacher scene generation")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("passed") or record.get("git_sha") != _require_committed_main():
        raise RuntimeError("R5-C preflight does not match the pushed protocol/code commit")
    if record.get("protocol_version") != float(config["experiment"]["protocol_version"]):
        raise RuntimeError("R5-C preflight protocol version mismatch")
    return record


def run_audit() -> dict[str, Any]:
    config = _read_yaml(CONFIG_PATH)
    _validate_config(config)
    preflight = _require_preflight(config)
    output_csv = OUTPUT_ROOT / "teacher_scene_metrics.csv"
    selected_ids_path = OUTPUT_ROOT / "selected_scene_ids.csv"
    if output_csv.exists() or selected_ids_path.exists():
        raise RuntimeError("R5-C scene results already exist; refusing to overwrite archived evidence")
    physics = _read_yaml(PHYSICS_PATH)
    plan = _build_batch_plan(config)
    seen_ids: set[str] = set()
    metric_rows: list[dict[str, Any]] = []
    selected_id_rows: list[dict[str, Any]] = []
    generation_seconds = 0.0
    cuda_k1_seconds = 0.0
    cuda_k3_seconds = 0.0
    import torch

    _reset_cuda_peak_memory_stats("cuda:0")
    for batch_index, descriptor in enumerate(plan):
        started = time.perf_counter()
        batch = generate_d2a_paired_batch(
            int(descriptor["scene_count"]),
            int(descriptor["seed"]),
            float(descriptor["snr_db"]),
            float(descriptor["cnr_db"]),
            physics,
            scenario_id=str(descriptor["scenario_id"]),
        )
        generation_seconds += time.perf_counter() - started
        ids = _validate_pair_batch(
            batch,
            expected_count=int(descriptor["scene_count"]),
            expected_seed=int(descriptor["seed"]),
        )
        if seen_ids.intersection(ids):
            raise RuntimeError("R5-C generated duplicate base_scene_id values")
        seen_ids.update(ids)
        started = time.perf_counter()
        k1_h0 = k2_response_cuda(
            batch.iq_h0, physics, device="cuda:0", oversampling_factor=1
        )
        k1_h1 = k2_response_cuda(
            batch.iq_h1, physics, device="cuda:0", oversampling_factor=1
        )
        cuda_k1_seconds += time.perf_counter() - started
        started = time.perf_counter()
        k3_h0 = k2_response_cuda(
            batch.iq_h0, physics, device="cuda:0", oversampling_factor=4
        )
        k3_h1 = k2_response_cuda(
            batch.iq_h1, physics, device="cuda:0", oversampling_factor=4
        )
        cuda_k3_seconds += time.perf_counter() - started
        row_args = {
            "course": str(descriptor["course"]),
            "scenario_id": str(descriptor["scenario_id"]),
            "snr_db": float(descriptor["snr_db"]),
            "cnr_db": float(descriptor["cnr_db"]),
            "base_scene_ids": np.asarray(ids),
            "truth_range_cell": batch.truth_target_range_cell,
            "truth_doppler_hz": batch.truth_target_doppler_hz,
            "prf_hz": float(config["radar"]["prf_hz"]),
        }
        metric_rows.extend(
            _map_response_metrics(
                candidate="K1", h0_response=k1_h0, h1_response=k1_h1, fft_factor=1, **row_args
            )
        )
        metric_rows.extend(
            _map_response_metrics(
                candidate="K3", h0_response=k3_h0, h1_response=k3_h1, fft_factor=4, **row_args
            )
        )
        metric_rows.extend(
            _local_response_metrics(
                h0_response=k3_h0, h1_response=k3_h1, **row_args
            )
        )
        for index, scene_id in enumerate(ids):
            selected_id_rows.append(
                {
                    "base_scene_id": scene_id,
                    "course": descriptor["course"],
                    "scenario_id": descriptor["scenario_id"],
                    "snr_db": descriptor["snr_db"],
                    "cnr_db": descriptor["cnr_db"],
                    "seed": descriptor["seed"],
                    "scene_index_in_batch": index,
                }
            )
        del batch, k1_h0, k1_h1, k3_h0, k3_h1
        if batch_index % 16 == 15 or batch_index == len(plan) - 1:
            print(
                f"R5-C batch {batch_index + 1}/{len(plan)}; "
                f"scenes={len(seen_ids)}/{config['data']['total_paired_base_scenes']}; "
                f"CUDA allocated={torch.cuda.memory_allocated(0) / 1024**2:.1f} MiB",
                flush=True,
            )
    if len(seen_ids) != int(config["data"]["total_paired_base_scenes"]):
        raise RuntimeError("R5-C completed scene count differs from the frozen allocation")
    _write_csv(output_csv, metric_rows)
    _write_csv(selected_ids_path, selected_id_rows)
    project_size_after = _repo_size_bytes()
    runtime = {
        "git_sha": preflight["git_sha"],
        "batch_count": len(plan),
        "paired_base_scenes": len(seen_ids),
        "candidate_row_count": len(metric_rows),
        "scene_generation_cpu_seconds": generation_seconds,
        "K1_cuda_seconds": cuda_k1_seconds,
        "K3_cuda_seconds": cuda_k3_seconds,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "project_size_bytes_after": project_size_after,
        "project_size_gib_after": project_size_after / 1024**3,
        "iq_or_teacher_map_cache_written": False,
        "student_training": False,
        "distillation": False,
        "final_blind": False,
    }
    _write_json(OUTPUT_ROOT / "run_resources.json", runtime)
    return runtime


def _plot_metrics(summary: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = ("K1", "K3", "K2_local")
    labels = ("K1 full RD", "K3 4× RD", "K2 local FD")
    c3 = summary["candidate_summaries"]
    lift_values = [c3[name]["C3"]["target_window_mass_lift_mean"] for name in candidates]
    lift_intervals = [c3[name]["C3"]["target_window_mass_lift_ci95"] for name in candidates]
    localization = [c3[name]["C3"]["teacher_top_cell_localization_rate"] for name in candidates]
    localization_intervals = [c3[name]["C3"]["teacher_top_cell_localization_ci95"] for name in candidates]
    positions = np.arange(len(candidates))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    for axis, values, intervals, title, ylabel in (
        (axes[0], lift_values, lift_intervals, "C3 paired target-window lift", "H1−H0 support mass"),
        (axes[1], localization, localization_intervals, "C3 teacher top-cell localization", "Rate"),
    ):
        lower = np.asarray([value - ci[0] for value, ci in zip(values, intervals)])
        upper = np.asarray([ci[1] - value for value, ci in zip(values, intervals)])
        axis.bar(positions, values, color=("#4776a8", "#cf6d3b", "#6a9c70"))
        axis.errorbar(positions, values, yerr=np.vstack([lower, upper]), fmt="none", color="black", capsize=3)
        axis.set_xticks(positions, labels, rotation=15, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    path = OUTPUT_ROOT / "figures/teacher_quality_c3.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def analyze_results() -> dict[str, Any]:
    config = _read_yaml(CONFIG_PATH)
    _validate_config(config)
    rows = list(csv.DictReader((OUTPUT_ROOT / "teacher_scene_metrics.csv").open(encoding="utf-8")))
    for row in rows:
        for key in (
            "snr_db", "cnr_db", "truth_range_cell", "truth_doppler_hz", "h0_window_mass",
            "h1_window_mass", "target_window_mass_lift", "predicted_range_cell", "predicted_doppler_hz",
            "range_abs_error_cells", "doppler_wrapped_error_hz", "response_confidence",
        ):
            row[key] = float(row[key])
        for key in ("top_cell_hit", "h0_pseudo_target_hit"):
            row[key] = row[key].lower() == "true"
    expected_rows = int(config["data"]["total_paired_base_scenes"]) * 3
    if len(rows) != expected_rows:
        raise RuntimeError(f"R5-C metrics CSV has {len(rows)} rows, expected {expected_rows}")
    expected_scene_count = int(config["data"]["total_paired_base_scenes"])
    selected_ids = [
        row["base_scene_id"]
        for row in csv.DictReader(
            (OUTPUT_ROOT / "selected_scene_ids.csv").open(encoding="utf-8")
        )
    ]
    if len(selected_ids) != expected_scene_count or len(set(selected_ids)) != expected_scene_count:
        raise RuntimeError("R5-C selected scene manifest has missing or duplicate base_scene_id values")
    selected_id_set = set(selected_ids)
    for candidate in ("K1", "K3", "K2_local"):
        candidate_rows = [row for row in rows if row["candidate"] == candidate]
        candidate_ids = [row["base_scene_id"] for row in candidate_rows]
        if len(candidate_ids) != expected_scene_count or set(candidate_ids) != selected_id_set:
            raise RuntimeError(
                f"R5-C {candidate} rows must contain every selected base_scene_id exactly once"
            )
    summary = _registered_summaries(rows, config)
    run_resources = json.loads((OUTPUT_ROOT / "run_resources.json").read_text(encoding="utf-8"))
    preflight = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    summary.update(
        {
            "experiment": "D2-A-R5-C",
            "protocol_version": 1.1,
            "git_sha": preflight["git_sha"],
            "research_charter": "docs/RESEARCH_CHARTER.md v1.3",
            "paired_base_scenes": int(config["data"]["total_paired_base_scenes"]),
            "candidate_rows": len(rows),
            "environment": json.loads((OUTPUT_ROOT / "environment.json").read_text(encoding="utf-8")),
            "runtime_resources": run_resources,
            "blind": False,
            "student_training": False,
            "distillation": False,
            "fixed_threshold_teacher_qualification": False,
        }
    )
    figure_path = _plot_metrics(summary)
    summary["figure"] = str(figure_path.relative_to(ROOT))
    _write_json(OUTPUT_ROOT / "metrics.json", summary)
    _write_analysis(summary)
    return summary


def _write_analysis(summary: dict[str, Any]) -> None:
    candidates = summary["candidate_summaries"]
    g4 = summary["g4"]
    lines = [
        "# D2-A-R5-C 教师知识质量审计分析",
        "",
        f"**状态：** 真实新场景教师审计完成；协议 v1.1；Git SHA `{summary['git_sha']}`。",
        "**数据性质：** 合成开发审计，不是 blind；没有训练学生、KD 或固定阈值 detector 认证。",
        "**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3，正文未修改。",
        "",
        "## 研究问题与执行范围",
        "",
        "固定 B3 DPCA 前端下，比较原生全 RD 支持图 K1、4× 零填充全图 K3，以及从 K3 自身最大响应距离门抽取的 K2-local Doppler 支持。教师仅由可观测 IQ 与冻结几何计算，离线真值只用于质量评价。归一化功率支持不解释为后验概率。",
        "",
        f"共完成 {summary['paired_base_scenes']} 个新配对 base scenes、{summary['candidate_rows']} 条候选记录。实际 CUDA 与仿真资源见 `run_resources.json`；运行时间不是实时性结论。",
        "",
        "## 预注册结果摘要",
        "",
        "| 候选 | scope | H1−H0 target support lift (95% CI) | top-cell localization (95% CI) | Doppler RMSE Hz (95% CI) | confidence AUC | G4 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for candidate in ("K1", "K3", "K2_local"):
        for scope in ("identifiable_C2_C3", "C3"):
            metric = candidates[candidate][scope]
            lift_ci = metric["target_window_mass_lift_ci95"]
            loc_ci = metric["teacher_top_cell_localization_ci95"]
            doppler_ci = metric["doppler_rmse_ci95"]
            lines.append(
                f"| {candidate} | {scope} | {metric['target_window_mass_lift_mean']:.6g} "
                f"[{lift_ci[0]:.6g}, {lift_ci[1]:.6g}] | "
                f"{metric['teacher_top_cell_localization_rate']:.4f} [{loc_ci[0]:.4f}, {loc_ci[1]:.4f}] | "
                f"{metric['doppler_rmse_all_h1_hz']:.3f} [{doppler_ci[0]:.3f}, {doppler_ci[1]:.3f}] | "
                f"{metric['confidence_auc']} | {g4[candidate]['G4_knowledge_quality_pass']} |"
            )
    lines.extend(
        [
            "",
            "## K3 相对 K1 的预注册配对对比",
            "",
            "| scope | localization difference, K3−K1 (95% CI) | Doppler RMSE difference Hz, K3−K1 (95% CI) | meaningful gain gate |",
            "|---|---:|---:|---|",
        ]
    )
    contrast = summary["primary_contrasts"]
    for scope in ("identifiable_C2_C3", "C3"):
        value = contrast[scope]
        lines.append(
            f"| {scope} | {value['K3_minus_K1_localization_rate']:.4f} "
            f"[{value['K3_minus_K1_localization_rate_ci95'][0]:.4f}, {value['K3_minus_K1_localization_rate_ci95'][1]:.4f}] | "
            f"{value['K3_minus_K1_doppler_rmse_hz']:.3f} "
            f"[{value['K3_minus_K1_doppler_rmse_ci95_hz'][0]:.3f}, {value['K3_minus_K1_doppler_rmse_ci95_hz'][1]:.3f}] | "
            f"{contrast['K3_gain_gate'][scope]} |"
        )
    lines.extend(
        [
            "",
            "## 教师资格与结论边界",
            "",
            "- `G4_knowledge_quality_pass` 只代表满足协议中 IQ 输入隔离、数值正确和可辨识 target-window 信息门；出现的 C3 严重误导子组及可观测置信信号逐一见 `metrics.json`。",
            "- 任何 G4 结果都不授予固定阈值 detector 资格，不代表学生能吸收该响应，也不构成蒸馏收益。教师固定阈值历史结论保持不变。",
            "- 对 K3 的复杂性保留只按冻结最小有意义增益门判断；没有通过时不继续搜索 FFT 倍率、温度或窗口。",
            "- R5-A Medium 的 C3 虚警保护失败、R5-B TFAS G3 失败及 E002 K2 未显示 student-side 增益均保持历史结果。",
            "",
            "## 复现材料",
            "",
            "- 协议/config：`protocol.md`、`config.yaml`。",
            "- CUDA 教师：`src/radardistill/teachers/d2a_k2_cuda.py`。",
            "- runner：`scripts/run_d2a_r5c_teacher_audit.py`。",
            "- 输出：`preflight.json`、`environment.json`、`selected_scene_ids.csv`、`teacher_scene_metrics.csv`、`run_resources.json`、`metrics.json`、`figures/teacher_quality_c3.png`。",
            "- 本报告基于新 scene seeds 的实际运行；不含完整 IQ 或 teacher map 缓存。",
        ]
    )
    (OUTPUT_ROOT / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "run", "analyze"))
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        result = run_preflight()
    elif args.stage == "run":
        result = run_audit()
    else:
        result = analyze_results()
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
