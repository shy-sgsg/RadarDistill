"""CUDA-only, matched-budget trainer for D1-E002."""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..distillation.task_losses import decode_task_outputs, task_loss
from ..simulation.d0 import D0Batch, generate_d0_batch
from ..students.d1_e002 import D1E002Student
from ..students.task_student import TaskStudent


_ARRAY_NAMES = (
    "iq",
    "truth_presence",
    "truth_range_cell",
    "truth_doppler_hz",
    "truth_initial_phase_rad",
    "truth_target_amplitude",
    "snr_db",
)


def require_cuda(
    device: str = "cuda:0",
    *,
    cuda_available: bool | None = None,
    device_count: int | None = None,
) -> torch.device:
    """Reject CPU fallback before creating data or a model."""
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("D1-E002 requires CUDA cuda:0; CPU fallback is forbidden")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    count = torch.cuda.device_count() if device_count is None else device_count
    if not available or count < 1:
        raise RuntimeError("CUDA is required for D1-E002; no training was started")
    selected = torch.device("cuda:0")
    torch.cuda.set_device(selected)
    return selected


def balanced_batch_orders(
    examples_per_class: int,
    effective_batch_size: int,
    updates: int,
    seed: int,
) -> np.ndarray:
    """Create a deterministic H0/H1 balanced order with paired sample IDs."""
    if examples_per_class <= 0 or effective_batch_size <= 0 or updates <= 0:
        raise ValueError("sample count, batch size, and updates must be positive")
    if effective_batch_size % 2:
        raise ValueError("effective batch size must be even for balanced H0/H1 batches")
    per_class_batch = effective_batch_size // 2
    if examples_per_class % per_class_batch:
        raise ValueError("examples_per_class must divide into balanced class batches")
    rng = np.random.default_rng(int(seed))
    orders = np.empty((updates, effective_batch_size), dtype=np.int64)
    permutation = rng.permutation(examples_per_class)
    cursor = 0
    for update in range(updates):
        if cursor + per_class_batch > examples_per_class:
            permutation = rng.permutation(examples_per_class)
            cursor = 0
        sample_ids = permutation[cursor : cursor + per_class_batch]
        cursor += per_class_batch
        paired = np.concatenate((sample_ids, examples_per_class + sample_ids))
        orders[update] = paired[rng.permutation(effective_batch_size)]
    return orders


def _expand(base: int, count: int, stride: int) -> list[int]:
    return [int(base + stride * index) for index in range(count)]


def _evaluation_stream_seeds(
    seed_base: int,
    total_per_snr: int,
    chunk_size: int,
    snr_count: int,
    stride: int,
) -> list[int]:
    chunks = math.ceil(total_per_snr / chunk_size)
    return [
        int(seed_base + snr_index * stride + chunk_index * stride * snr_count)
        for snr_index in range(snr_count)
        for chunk_index in range(chunks)
    ]


def e002_seed_values(config: dict[str, Any]) -> list[int]:
    """Expand every E002 data, initialization, order, and bootstrap seed."""
    stride = int(config["seeds"]["stream_stride"])
    snr_count = len(config["data"]["snr_db"])
    splits = config["splits"]
    values: list[int] = []
    for split in [splits["pilot"], *splits["formal"]]:
        chunks = int(split["train_examples_per_class"]) // int(
            split["train_examples_per_chunk"]
        )
        values.extend(_expand(int(split["train_seed"]), chunks, stride))
        values.extend(_expand(int(split["validation_seed"]), snr_count, stride))
        values.extend(
            (int(split["initialization_seed"]), int(split["batch_order_seed"]))
        )
    calibration = splits["calibration"]
    chunk_size = int(config["data"]["evaluation_generation_chunk_size"])
    values.extend(
        _evaluation_stream_seeds(
            int(calibration["h0_seed_base"]),
            int(calibration["h0_examples_per_snr"]),
            chunk_size,
            snr_count,
            stride,
        )
    )
    blind = splits["blind"]
    values.extend(
        _evaluation_stream_seeds(
            int(blind["h0_seed_base"]),
            int(blind["h0_examples_per_snr"]),
            chunk_size,
            snr_count,
            stride,
        )
    )
    values.extend(
        _evaluation_stream_seeds(
            int(blind["h1_seed_base"]),
            int(blind["h1_examples_per_snr"]),
            chunk_size,
            snr_count,
            stride,
        )
    )
    values.append(int(splits["physical_consistency"]["seed"]))
    values.append(int(config["seeds"]["bootstrap"]))
    return values


def validate_e002_seed_disjointness(
    config: dict[str, Any], *, existing_seed_values: set[int] | None = None
) -> set[int]:
    """Check all expanded streams are unique and disjoint from prior seeds."""
    values = e002_seed_values(config)
    if len(values) != len(set(values)):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        raise ValueError(f"D1-E002 seed streams overlap internally: {duplicates[:8]}")
    existing = set(existing_seed_values or ())
    overlap = sorted(set(values) & existing)
    if overlap:
        raise ValueError(f"D1-E002 seed streams overlap prior experiments: {overlap[:8]}")
    return set(values)


def _collect_seed_values(value: Any, key: str = "") -> set[int]:
    if isinstance(value, dict):
        result: set[int] = set()
        for child_key, child_value in value.items():
            result.update(_collect_seed_values(child_value, str(child_key)))
        return result
    if isinstance(value, list):
        result: set[int] = set()
        for child_value in value:
            result.update(_collect_seed_values(child_value, key))
        return result
    if "seed" in key.lower() and isinstance(value, int) and not isinstance(value, bool):
        return {value}
    return set()


def prior_registered_seed_values(root: Path) -> set[int]:
    """Read registered seeds and expand D1-E001 block/SNR seed streams."""
    values: set[int] = set()
    for path in (root / "experiments").rglob("config.yaml"):
        if path.parent.name == "D1-E002":
            continue
        try:
            values.update(_collect_seed_values(yaml.safe_load(path.read_text(encoding="utf-8"))))
        except (OSError, yaml.YAMLError):
            continue

    d1_path = root / "experiments" / "D1-E001" / "config.yaml"
    d1 = yaml.safe_load(d1_path.read_text(encoding="utf-8"))
    stride = int(d1["data"]["training_chunk_seed_stride_by_index"])
    snr_count = len(d1["data"]["evaluation_snr_db"])
    splits = d1["splits"]
    formal = splits["formal"]
    for split in [splits["pilot"], *formal["training_seeds"]]:
        train_base = int(split.get("train_seed", split.get("train")))
        validation_base = int(split.get("validation_seed", split.get("validation")))
        examples = int(split.get("train_examples_per_class", formal["train_examples_per_class"]))
        values.update(_expand(train_base, examples // 64, stride))
        values.update(_expand(validation_base, snr_count, stride))
    calibration = splits["detection_calibration"]
    blind = splits["final_blind"]
    values.update(_expand(int(calibration["seed_base"]), snr_count, stride))
    values.update(_expand(int(blind["seed_base_h0"]), snr_count, stride))
    values.update(_expand(int(blind["seed_base_h1"]), snr_count, stride))
    return values


def select_formal_candidates(
    pilot_metrics: dict[str, dict[str, Any]],
    s0_metrics: dict[str, Any],
    matched_fft_metrics: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen validation-only screen; never reads blind data."""
    screen = config["acceptance"]["pilot_screen"]
    candidates: dict[str, Any] = {}
    formal: list[str] = []
    snrs = [str(value) for value in config["acceptance"]["primary_endpoint"]["snr_db"]]
    low_auc_margin = float(screen["minimum_low_snr_auc_delta_vs_s0"])
    max_regression = float(screen["maximum_single_primary_snr_regression_fraction"])
    range_margin = float(screen["maximum_high_snr_range_rmse_increase_cells_vs_s0"])
    s0_auc = float(s0_metrics["low_snr_mean_auc"])
    s0_range = float(s0_metrics["high_snr_mean_range_rmse_cells"])
    for name in config["training"]["candidate_order"]:
        current = pilot_metrics.get(name)
        reasons: list[str] = []
        if not isinstance(current, dict):
            candidates[name] = {"eligible": False, "reasons": ["missing_pilot_metrics"]}
            continue
        expected_updates = int(config["models"][name]["optimizer_updates"])
        training_summary = current.get("training", {})
        completed_updates = training_summary.get(
            "completed_updates", current.get("completed_updates", -1)
        )
        if int(completed_updates) != expected_updates:
            reasons.append("pilot_update_budget_incomplete")
        if not current.get("finite_outputs", False):
            reasons.append("non_finite_outputs")
        if not current.get("wrapped_frequency_decode_valid", False):
            reasons.append("periodic_frequency_decode_invalid")
        candidate_auc = float(current.get("low_snr_mean_auc", math.nan))
        if not math.isfinite(candidate_auc) or candidate_auc - s0_auc < low_auc_margin:
            reasons.append("low_snr_auc_guard_failed")
        candidate_range = float(current.get("high_snr_mean_range_rmse_cells", math.nan))
        if not math.isfinite(candidate_range) or candidate_range - s0_range > range_margin:
            reasons.append("high_snr_range_guard_failed")

        if name == "S3":
            comparator = matched_fft_metrics
            required_gain = float(screen["s3_minimum_gain_vs_matched_nonlearning_fft"])
        else:
            comparator = s0_metrics
            required_gain = float(screen["s1_s2_minimum_gain_vs_s0"])
        current_rmse = float(current.get("high_snr_mean_doppler_rmse_hz", math.nan))
        comparator_rmse = float(comparator.get("high_snr_mean_doppler_rmse_hz", math.nan))
        if not math.isfinite(current_rmse) or not math.isfinite(comparator_rmse) or comparator_rmse <= 0.0:
            reasons.append("primary_validation_metric_unavailable")
        else:
            relative_gain = (comparator_rmse - current_rmse) / comparator_rmse
            if relative_gain < required_gain:
                reasons.append("primary_validation_gain_below_screen")
        per_snr = current.get("doppler_rmse_by_snr_db", {})
        comparator_per_snr = comparator.get("doppler_rmse_by_snr_db", {})
        for snr in snrs:
            candidate_value = float(per_snr.get(snr, math.nan))
            comparator_value = float(comparator_per_snr.get(snr, math.nan))
            if (
                not math.isfinite(candidate_value)
                or not math.isfinite(comparator_value)
                or comparator_value <= 0.0
                or candidate_value > comparator_value * (1.0 + max_regression)
            ):
                reasons.append(f"primary_snr_{snr}_regression")
        eligible = not reasons
        candidates[name] = {
            "eligible": eligible,
            "comparator": "matched_nonlearning_three_point_fft" if name == "S3" else "S0",
            "relative_gain": (
                (comparator_rmse - current_rmse) / comparator_rmse
                if math.isfinite(current_rmse) and math.isfinite(comparator_rmse) and comparator_rmse > 0
                else None
            ),
            "reasons": reasons,
        }
        if eligible:
            formal.append(name)
    return {
        "selection_source": "pilot_validation_only",
        "blind_data_used": False,
        "formal_candidates": formal,
        "candidates": candidates,
        "no_candidate_action": "stop_without_formal_training_or_blind_generation" if not formal else None,
    }


def assert_preregistered_files(root: Path, config_path: Path) -> str:
    """Require the live config and protocol bytes to match the current commit."""
    root = root.resolve()
    config_relative = config_path.resolve().relative_to(root).as_posix()
    protocol_relative = "experiments/D1-E002/protocol.md"
    for relative in (config_relative, protocol_relative):
        try:
            committed = subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"],
                check=True,
                capture_output=True,
            ).stdout
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"preregistered file is not in HEAD: {relative}") from exc
        if (root / relative).read_bytes() != committed:
            raise RuntimeError(f"preregistered file differs from HEAD; freeze it before running: {relative}")
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def assert_committed_implementation(root: Path) -> str:
    """Require all executable E002 code paths to be tracked and clean before a run."""
    paths = (
        "experiments/D1-E002/config.yaml",
        "experiments/D1-E002/protocol.md",
        "src/radardistill/training/d1_e002.py",
        "src/radardistill/students/d1_e002.py",
        "src/radardistill/evaluation/d1_e002.py",
        "scripts/run_d1_e002.py",
        "scripts/evaluate_d1_e002.py",
        "scripts/validate_d1_e002_protocol.py",
    )
    for relative in paths:
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
        )
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", *paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError(f"commit D1-E002 implementation before starting an experiment: {dirty}")
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_d0_config(root: Path) -> dict[str, Any]:
    return yaml.safe_load(
        (root / "experiments" / "D0-E001" / "config.yaml").read_text(encoding="utf-8")
    )


def _array_specs(total: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    return {
        "iq": ((total, 3, 130, 32), np.complex64),
        "truth_presence": ((total,), np.bool_),
        "truth_range_cell": ((total,), np.float32),
        "truth_doppler_hz": ((total,), np.float32),
        "truth_initial_phase_rad": ((total,), np.float32),
        "truth_target_amplitude": ((total,), np.float32),
        "snr_db": ((total,), np.float32),
    }


def _write_examples(
    maps: dict[str, np.ndarray],
    start: int,
    batch: D0Batch,
) -> None:
    fields = {
        "iq": batch.iq,
        "truth_presence": batch.truth_presence,
        "truth_range_cell": batch.truth_range_cell,
        "truth_doppler_hz": batch.truth_doppler_hz,
        "truth_initial_phase_rad": batch.truth_initial_phase_rad,
        "truth_target_amplitude": batch.truth_target_amplitude,
        "snr_db": batch.snr_db,
    }
    end = start + len(batch.iq)
    for name, values in fields.items():
        maps[name][start:end] = values


def _open_or_generate_split(
    *,
    root: Path,
    config: dict[str, Any],
    split: dict[str, Any],
    split_kind: str,
    cache_path: Path,
) -> dict[str, np.ndarray]:
    stride = int(config["seeds"]["stream_stride"])
    if split_kind == "train":
        examples_per_class = int(split["train_examples_per_class"])
        chunk_size = int(split["train_examples_per_chunk"])
        if examples_per_class % chunk_size:
            raise ValueError("training examples per class must divide into fixed cache chunks")
        expected = {
            "experiment_id": "D1-E002",
            "split_id": split["id"],
            "kind": "train",
            "train_seed": int(split["train_seed"]),
            "stream_stride": stride,
            "examples_per_class": examples_per_class,
            "examples_per_chunk": chunk_size,
            "snr_interval_db": [float(item) for item in config["data"]["training_snr_interval_db"]],
            "class_layout": "all_H0_then_all_H1_with_paired_chunk_sample_ids",
        }
        total_per_class = examples_per_class
    elif split_kind == "validation":
        examples_per_snr = int(split["validation_examples_per_class_per_snr"])
        snr_values = [float(value) for value in config["data"]["snr_db"]]
        total_per_class = examples_per_snr * len(snr_values)
        expected = {
            "experiment_id": "D1-E002",
            "split_id": split["id"],
            "kind": "validation",
            "validation_seed": int(split["validation_seed"]),
            "stream_stride": stride,
            "examples_per_class_per_snr": examples_per_snr,
            "snr_db": snr_values,
            "class_layout": "all_H0_then_all_H1_stratified_by_snr",
        }
    else:
        raise ValueError("split_kind must be train or validation")

    manifest_path = cache_path / "manifest.json"
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"cached E002 split differs from frozen config: {cache_path}")
        missing = [name for name in _ARRAY_NAMES if not (cache_path / f"{name}.npy").is_file()]
        if missing:
            raise FileNotFoundError(f"cached E002 split is incomplete: {missing}")
        return {name: np.load(cache_path / f"{name}.npy", mmap_mode="r") for name in _ARRAY_NAMES}
    if cache_path.exists():
        raise FileExistsError(
            f"unmanifested E002 cache exists at {cache_path}; preserving it and requiring a new path"
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    building = cache_path.with_name(cache_path.name + ".building")
    if building.exists():
        raise FileExistsError(f"partial cache build exists at {building}; preserving it")
    building.mkdir(parents=True, exist_ok=False)
    total = 2 * total_per_class
    maps = {
        name: np.lib.format.open_memmap(
            building / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in _array_specs(total).items()
    }
    d0_config = _load_d0_config(root)
    if split_kind == "train":
        snr_range = tuple(float(item) for item in config["data"]["training_snr_interval_db"])
        for chunk_index, start in enumerate(range(0, total_per_class, chunk_size)):
            count = min(chunk_size, total_per_class - start)
            seed = int(split["train_seed"]) + stride * chunk_index
            h0 = generate_d0_batch(count, seed, snr_range, 0.0, d0_config)
            h1 = generate_d0_batch(count, seed, snr_range, 1.0, d0_config)
            _write_examples(maps, start, h0)
            _write_examples(maps, total_per_class + start, h1)
    else:
        for snr_index, snr in enumerate(snr_values):
            seed = int(split["validation_seed"]) + stride * snr_index
            h0 = generate_d0_batch(examples_per_snr, seed, snr, 0.0, d0_config)
            h1 = generate_d0_batch(examples_per_snr, seed, snr, 1.0, d0_config)
            offset = snr_index * examples_per_snr
            _write_examples(maps, offset, h0)
            _write_examples(maps, total_per_class + offset, h1)
    for array in maps.values():
        array.flush()
    (building / "manifest.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(building, cache_path)
    return {name: np.load(cache_path / f"{name}.npy", mmap_mode="r") for name in _ARRAY_NAMES}


def _as_iq_planes(iq: np.ndarray) -> torch.Tensor:
    values = np.asarray(iq, dtype=np.complex64)
    planes = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(planes))


def _truth_batch(arrays: dict[str, np.ndarray], indices: np.ndarray, device) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.asarray(arrays[name][indices]).copy()).to(device)
        for name in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[sorted_labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def _wrap_numpy(error_hz: np.ndarray, prf_hz: float) -> np.ndarray:
    return (np.asarray(error_hz, dtype=np.float64) + prf_hz / 2.0) % prf_hz - prf_hz / 2.0


def _summarize_predictions(
    *,
    score: np.ndarray,
    predicted_range: np.ndarray,
    predicted_doppler: np.ndarray,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, Any]:
    truth_presence = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    truth_range = np.asarray(arrays["truth_range_cell"], dtype=np.float64)
    truth_doppler = np.asarray(arrays["truth_doppler_hz"], dtype=np.float64)
    prf_hz = float(config["radar"]["prf_hz"])
    snr_values = [float(value) for value in config["data"]["snr_db"]]
    auc_by_snr: dict[str, float] = {}
    range_by_snr: dict[str, float] = {}
    doppler_by_snr: dict[str, float] = {}
    range_mae_by_snr: dict[str, float] = {}
    doppler_mae_by_snr: dict[str, float] = {}
    for value in snr_values:
        key = str(int(value))
        group = np.isclose(snr, value)
        auc_by_snr[key] = _auc(score[group], truth_presence[group])
        h1 = group & truth_presence
        if not np.any(h1):
            continue
        range_error = predicted_range[h1] - truth_range[h1]
        doppler_error = _wrap_numpy(predicted_doppler[h1] - truth_doppler[h1], prf_hz)
        range_by_snr[key] = float(np.sqrt(np.mean(range_error**2)))
        doppler_by_snr[key] = float(np.sqrt(np.mean(doppler_error**2)))
        range_mae_by_snr[key] = float(np.mean(np.abs(range_error)))
        doppler_mae_by_snr[key] = float(np.mean(np.abs(doppler_error)))
    low_keys = [str(value) for value in (-20, -15, -10, -5)]
    high_keys = [str(value) for value in (0, 5, 10, 20)]
    high_doppler = [doppler_by_snr[key] for key in high_keys]
    high_range = [range_by_snr[key] for key in high_keys]
    low_auc = [auc_by_snr[key] for key in low_keys]
    return {
        "auc_by_snr_db": auc_by_snr,
        "range_rmse_by_snr_db": range_by_snr,
        "doppler_rmse_by_snr_db": doppler_by_snr,
        "range_mae_by_snr_db": range_mae_by_snr,
        "doppler_mae_by_snr_db": doppler_mae_by_snr,
        "high_snr_mean_doppler_rmse_hz": float(np.mean(high_doppler)),
        "high_snr_mean_range_rmse_cells": float(np.mean(high_range)),
        "low_snr_mean_auc": float(np.mean(low_auc)),
        "finite_outputs": bool(
            np.isfinite(score).all()
            and np.isfinite(predicted_range).all()
            and np.isfinite(predicted_doppler).all()
        ),
        "wrapped_frequency_decode_valid": bool(
            np.all(predicted_doppler >= -prf_hz / 2.0)
            and np.all(predicted_doppler < prf_hz / 2.0)
        ),
        "examples": int(len(score)),
        "h1_examples": int(truth_presence.sum()),
    }


def _evaluate_model(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    *,
    variant: str | None,
    batch_size: int = 64,
) -> dict[str, Any]:
    from ..evaluation.d1_e002 import estimate_fft_at_range, estimate_lag_one_at_range

    score_parts: list[np.ndarray] = []
    range_parts: list[np.ndarray] = []
    doppler_parts: list[np.ndarray] = []
    lag_parts: list[np.ndarray] = []
    fft_three_parts: list[np.ndarray] = []
    fft_centroid_parts: list[np.ndarray] = []
    prf_hz = float(config["radar"]["prf_hz"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(arrays["iq"]), batch_size):
            indices = np.arange(start, min(start + batch_size, len(arrays["iq"])), dtype=np.int64)
            iq = np.asarray(arrays["iq"][indices]).copy()
            inputs = _as_iq_planes(iq).to(device)
            if isinstance(model, TaskStudent):
                outputs = model(inputs, compute_spectrum=False)
            else:
                outputs = model(inputs)
            decoded = decode_task_outputs(outputs, prf_hz)
            pred_bin = outputs["range_logits"].argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
            score_parts.append(outputs["presence_logit"].detach().cpu().numpy().astype(np.float64))
            range_parts.append(decoded["range_cell"].detach().cpu().numpy().astype(np.float64))
            doppler_parts.append(decoded["doppler_hz"].detach().cpu().numpy().astype(np.float64))
            if variant == "S2":
                lag_parts.append(estimate_lag_one_at_range(iq, pred_bin, prf_hz))
            elif variant == "S3":
                fft_three_parts.append(
                    estimate_fft_at_range(iq, pred_bin, prf_hz, nfft=520)
                )
                fft_centroid_parts.append(
                    estimate_fft_at_range(
                        iq,
                        pred_bin,
                        prf_hz,
                        nfft=520,
                        estimator="five_bin_excess_centroid",
                    )
                )
    score = np.concatenate(score_parts)
    predicted_range = np.concatenate(range_parts)
    predicted_doppler = np.concatenate(doppler_parts)
    result = _summarize_predictions(
        score=score,
        predicted_range=predicted_range,
        predicted_doppler=predicted_doppler,
        arrays=arrays,
        config=config,
    )
    if variant == "S2":
        lag = np.concatenate(lag_parts)
        result["matched_lag_one_equal_channel"] = _summarize_predictions(
            score=score,
            predicted_range=predicted_range,
            predicted_doppler=lag,
            arrays=arrays,
            config=config,
        )
    elif variant == "S3":
        fft_three = np.concatenate(fft_three_parts)
        fft_centroid = np.concatenate(fft_centroid_parts)
        result["matched_nonlearning_fft"] = {
            "three_point_log_parabolic": _summarize_predictions(
                score=score,
                predicted_range=predicted_range,
                predicted_doppler=fft_three,
                arrays=arrays,
                config=config,
            ),
            "five_bin_excess_centroid": _summarize_predictions(
                score=score,
                predicted_range=predicted_range,
                predicted_doppler=fft_centroid,
                arrays=arrays,
                config=config,
            ),
        }
    return result


def _e002_loss_config(config: dict[str, Any]) -> dict[str, Any]:
    weights = config["training"]["common_task_loss_weights"]
    return {
        "radar": config["radar"],
        "training": {
            "loss": {
                "presence_bce_weight": float(weights["presence_bce"]),
                "range_cell_cross_entropy_weight": float(weights["range_cell_cross_entropy"]),
                "range_offset_smooth_l1_weight": float(weights["range_offset_smooth_l1"]),
                "range_offset_smooth_l1_beta": 0.1,
                "doppler_circular_cosine_weight": float(weights["doppler_circular_cosine"]),
                "teacher_response_kl_weight": 0.0,
            }
        },
    }


def _learning_rate(update: int, total_updates: int, initial: float, final: float) -> float:
    fraction = min(max(update / max(total_updates - 1, 1), 0.0), 1.0)
    return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * fraction))


def _initial_candidate(variant: str, split: dict[str, Any], config: dict[str, Any]) -> D1E002Student:
    seed = int(split["initialization_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return D1E002Student(
        variant=variant,
        width=32,
        temporal_dilations=(1, 2, 4, 8, 16, 32, 64),
        prf_hz=float(config["radar"]["prf_hz"]),
        fft_nfft=int(config["models"]["S3"]["fft"]["nfft"]),
        coherence_epsilon=float(config["models"]["S2"]["statistic_epsilon"]),
        fft_epsilon=float(config["models"]["S3"]["fft"]["epsilon"]),
    )


def _clone_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_candidate_checkpoint(
    path: Path,
    model: D1E002Student,
    *,
    variant: str,
    split: dict[str, Any],
    config: dict[str, Any],
    update: int,
    validation: dict[str, Any],
    code_commit: str,
    microbatch: int,
) -> None:
    torch.save(
        {
            "state_dict": _clone_state_to_cpu(model),
            "student": {
                "variant": variant,
                "width": model.width,
                "temporal_dilations": [1, 2, 4, 8, 16, 32, 64],
                "prf_hz": model.prf_hz,
                "fft_nfft": model.fft_nfft,
                "method_kind": model.method_kind,
                "is_hybrid": model.is_hybrid,
            },
            "experiment": {
                "id": "D1-E002",
                "code_commit": code_commit,
                "seed_id": split["id"],
                "train_seed": int(split["train_seed"]),
                "validation_seed": int(split["validation_seed"]),
                "initialization_seed": int(split["initialization_seed"]),
                "batch_order_seed": int(split["batch_order_seed"]),
                "update": int(update),
                "actual_microbatch": int(microbatch),
                "teacher_labels_used": False,
                "teacher_soft_responses_used": False,
            },
            "validation": validation,
        },
        path,
    )


def _run_train_attempt(
    *,
    variant: str,
    split: dict[str, Any],
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    code_commit: str,
    microbatch: int,
) -> dict[str, Any]:
    training = config["training"]
    effective_batch = int(training["effective_batch_size"])
    updates = int(config["models"][variant]["optimizer_updates"])
    if effective_batch % microbatch:
        raise ValueError("microbatch must divide the frozen effective batch")
    model = _initial_candidate(variant, split, config).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["optimizer"]["learning_rate"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    orders = balanced_batch_orders(
        int(split["train_examples_per_class"]),
        effective_batch,
        updates,
        int(split["batch_order_seed"]),
    )
    loss_config = _e002_loss_config(config)
    validation_interval = int(training["validation_interval_updates"])
    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_validation = _evaluate_model(
        model, validation_arrays, config, device, variant=variant
    )
    best_validation["validation_primary_rmse_hz"] = best_validation[
        "high_snr_mean_doppler_rmse_hz"
    ]
    best_update = 0
    best_state = _clone_state_to_cpu(model)
    out_dir.mkdir(parents=True, exist_ok=False)
    _save_candidate_checkpoint(
        out_dir / "initial_inference.pt",
        model,
        variant=variant,
        split=split,
        config=config,
        update=0,
        validation=best_validation,
        code_commit=code_commit,
        microbatch=microbatch,
    )
    history.append(
        {
            "update": 0,
            "stage": "initial",
            "train_total": None,
            "validation_primary_rmse_hz": best_validation["validation_primary_rmse_hz"],
        }
    )
    _write_json(out_dir / "latest_progress.json", {"update": 0, "validation": best_validation})

    model.train()
    for update in range(updates):
        lr = _learning_rate(
            update,
            updates,
            float(training["optimizer"]["learning_rate"]),
            float(training["optimizer"]["final_learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch_indices = orders[update]
        effective_h1 = int(np.count_nonzero(train_arrays["truth_presence"][batch_indices]))
        if effective_h1 != effective_batch // 2:
            raise RuntimeError("balanced batch order lost its H0/H1 contract")
        optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "total")}
        for offset in range(0, effective_batch, microbatch):
            indices = batch_indices[offset : offset + microbatch]
            inputs = _as_iq_planes(train_arrays["iq"][indices]).to(device)
            truth = _truth_batch(train_arrays, indices, device)
            outputs = model(inputs)
            losses = task_loss(outputs, truth, loss_config)
            if any(not torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f"non-finite D1-E002 loss: {variant} update {update + 1}")
            micro_h1 = int(np.count_nonzero(train_arrays["truth_presence"][indices]))
            loss_weights = training["common_task_loss_weights"]
            scaled = (
                float(loss_weights["presence_bce"])
                * losses["presence"]
                * (len(indices) / effective_batch)
                + (
                    float(loss_weights["range_cell_cross_entropy"])
                    * losses["range_cell"]
                    + float(loss_weights["range_offset_smooth_l1"])
                    * losses["range_offset"]
                    + float(loss_weights["doppler_circular_cosine"])
                    * losses["doppler"]
                )
                * (micro_h1 / effective_h1)
            )
            scaled.backward()
            totals["presence"] += float(losses["presence"].detach().item()) * len(indices) / effective_batch
            if micro_h1:
                fraction = micro_h1 / effective_h1
                totals["range_cell"] += float(losses["range_cell"].detach().item()) * fraction
                totals["range_offset"] += float(losses["range_offset"].detach().item()) * fraction
                totals["doppler"] += float(losses["doppler"].detach().item()) * fraction
            totals["total"] += float(scaled.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["optimizer"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite D1-E002 gradient: {variant} update {update + 1}")
        optimizer.step()
        row: dict[str, Any] = {
            "update": update + 1,
            "stage": "truth_supervised",
            "learning_rate": lr,
            "train_total": totals["total"],
            "train_presence": totals["presence"],
            "train_range_cell": totals["range_cell"],
            "train_range_offset": totals["range_offset"],
            "train_doppler": totals["doppler"],
        }
        if (update + 1) % validation_interval == 0 or update + 1 == updates:
            validation = _evaluate_model(
                model, validation_arrays, config, device, variant=variant
            )
            primary = float(validation["high_snr_mean_doppler_rmse_hz"])
            row["validation_primary_rmse_hz"] = primary
            row["validation_low_snr_mean_auc"] = validation["low_snr_mean_auc"]
            row["validation_high_snr_mean_range_rmse_cells"] = validation[
                "high_snr_mean_range_rmse_cells"
            ]
            if math.isfinite(primary) and primary < float(best_validation["validation_primary_rmse_hz"]):
                best_validation = validation
                best_validation["validation_primary_rmse_hz"] = primary
                best_update = update + 1
                best_state = _clone_state_to_cpu(model)
                _save_candidate_checkpoint(
                    out_dir / "best_inference.pt",
                    model,
                    variant=variant,
                    split=split,
                    config=config,
                    update=best_update,
                    validation=best_validation,
                    code_commit=code_commit,
                    microbatch=microbatch,
                )
            _save_candidate_checkpoint(
                out_dir / "latest_inference.pt",
                model,
                variant=variant,
                split=split,
                config=config,
                update=update + 1,
                validation=validation,
                code_commit=code_commit,
                microbatch=microbatch,
            )
            _write_json(
                out_dir / "latest_progress.json",
                {"update": update + 1, "validation": validation, "best_update": best_update},
            )
            print(
                f"[{variant}/{split['id']}] update {update + 1}/{updates} "
                f"val_doppler={primary:.4f} Hz; best={best_validation['validation_primary_rmse_hz']:.4f} Hz",
                flush=True,
            )
        history.append(row)

    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start_time
    _save_candidate_checkpoint(
        out_dir / "last_inference.pt",
        model,
        variant=variant,
        split=split,
        config=config,
        update=updates,
        validation=best_validation,
        code_commit=code_commit,
        microbatch=microbatch,
    )
    model.load_state_dict(best_state, strict=True)
    _save_candidate_checkpoint(
        out_dir / "best_inference.pt",
        model,
        variant=variant,
        split=split,
        config=config,
        update=best_update,
        validation=best_validation,
        code_commit=code_commit,
        microbatch=microbatch,
    )
    fieldnames = sorted({key for row in history for key in row})
    with (out_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "variant": variant,
        "seed_id": split["id"],
        "optimizer_updates": updates,
        "completed_updates": updates,
        "best_update": best_update,
        "best_validation": best_validation,
        "initial_validation": history[0],
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "train_seconds": train_seconds,
        "samples_per_second": effective_batch * updates / max(train_seconds, 1e-12),
        "actual_microbatch": microbatch,
        "effective_batch_size": effective_batch,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "best_checkpoint": str(out_dir / "best_inference.pt"),
    }
    model.to("cpu")
    del model, optimizer
    return summary


def _is_cuda_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or (
        isinstance(error, RuntimeError) and "out of memory" in str(error).lower()
    )


def _train_candidate_with_fallback(
    *,
    variant: str,
    split: dict[str, Any],
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    code_commit: str,
) -> dict[str, Any]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite prior E002 output: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    sizes = [int(config["training"]["initial_microbatch_size"]), *map(int, config["training"]["microbatch_fallbacks"])]
    for microbatch in sizes:
        attempt = out_dir.with_name(out_dir.name + f".microbatch{microbatch}.building")
        if attempt.exists():
            raise FileExistsError(f"partial training attempt exists; preserving it: {attempt}")
        try:
            summary = _run_train_attempt(
                variant=variant,
                split=split,
                train_arrays=train_arrays,
                validation_arrays=validation_arrays,
                config=config,
                device=device,
                out_dir=attempt,
                code_commit=code_commit,
                microbatch=microbatch,
            )
            os.replace(attempt, out_dir)
            summary["best_checkpoint"] = str(out_dir / "best_inference.pt")
            summary["actual_microbatch"] = microbatch
            return summary
        except BaseException as error:
            if not _is_cuda_oom(error):
                raise
            if attempt.exists():
                shutil.rmtree(attempt)
            gc.collect()
            torch.cuda.empty_cache()
    out_dir.mkdir(parents=True, exist_ok=False)
    failure = {
        "variant": variant,
        "seed_id": split["id"],
        "status": "cuda_oom_after_microbatch_fallbacks",
        "attempted_microbatches": sizes,
        "completed_updates": 0,
    }
    (out_dir / "failure.json").write_text(
        json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return failure


def _load_d1_anchor(root: Path, checkpoint_relative: str, device: torch.device) -> TaskStudent:
    d1_config = yaml.safe_load(
        (root / "experiments" / "D1-E001" / "config.yaml").read_text(encoding="utf-8")
    )
    student_config = d1_config["student"]
    model = TaskStudent(
        width=int(student_config["width"]),
        temporal_dilations=student_config["temporal_dilations"],
        include_spectrum_head=False,
    )
    artifact = torch.load(root / checkpoint_relative, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("state_dict"), dict):
        raise ValueError(f"D1 anchor checkpoint has an invalid envelope: {checkpoint_relative}")
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _environment(root: Path, device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        driver = None
    code_paths = (
        "experiments/D1-E002/config.yaml",
        "experiments/D1-E002/protocol.md",
        "src/radardistill/training/d1_e002.py",
        "src/radardistill/students/d1_e002.py",
        "src/radardistill/evaluation/d1_e002.py",
        "scripts/run_d1_e002.py",
        "scripts/evaluate_d1_e002.py",
        "scripts/validate_d1_e002_protocol.py",
    )
    return {
        "git_sha": subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dirty_state": {
            "e002_implementation_paths_clean": not bool(
                subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain", "--", *code_paths],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
            "workspace_has_uncommitted_changes": bool(
                subprocess.run(
                    ["git", "-C", str(root), "status", "--porcelain"],
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            ),
        },
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "gpu": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "cuda_device": str(device),
    }


def validate_e002_config(config: dict[str, Any]) -> None:
    """Reject changes to the E1 CUDA-only budget and matched candidate contract."""
    if config.get("experiment_id") != "D1-E002":
        raise ValueError("expected experiment_id D1-E002")
    radar = config["radar"]
    if (
        float(radar["prf_hz"]) != 1300.0
        or int(radar["pulse_count"]) != 130
        or int(radar["channel_count"]) != 3
        or int(radar["range_bin_count"]) != 32
    ):
        raise ValueError("D1-E002 radar dimensions differ from the preregistered D1 task")
    data = config["data"]
    if data.get("normalization_uses_truth_or_snr") is not False:
        raise ValueError("truth/SNR-dependent input normalization is forbidden")
    if data.get("training_snr_distribution") != "uniform_continuous":
        raise ValueError("D1-E002 training SNR must remain continuous-uniform")
    if [float(value) for value in data["training_snr_interval_db"]] != [-20.0, 20.0]:
        raise ValueError("D1-E002 training SNR interval is frozen at [-20,20] dB")
    if int(data["evaluation_generation_chunk_size"]) != 64:
        raise ValueError("D1-E002 evaluation stream chunks are frozen at 64 examples")
    training = config["training"]
    if (
        training.get("device") != "cuda:0"
        or training.get("cpu_fallback") is not False
        or training.get("precision") != "float32"
        or training.get("objective") != "E1_physics_simulation_truth_supervision_only"
        or training.get("teacher_labels_used") is not False
        or training.get("teacher_soft_responses_used") is not False
    ):
        raise ValueError("D1-E002 training must remain CUDA FP32 truth-only E1")
    if training.get("candidate_order") != ["S1", "S2", "S3"]:
        raise ValueError("candidate order is frozen as S1, S2, S3")
    if int(training["effective_batch_size"]) % 2:
        raise ValueError("balanced H0/H1 training requires an even effective batch")
    budgets = {
        (
            int(config["models"][name]["optimizer_updates"]),
            int(config["models"][name]["effective_batch_size"]),
        )
        for name in ("S1", "S2", "S3")
    }
    if len(budgets) != 1:
        raise ValueError("candidate optimizer budgets must match")
    if budgets != {(1600, 64)} or int(training["effective_batch_size"]) != 64:
        raise ValueError("D1-E002 freezes 1600 updates and effective batch size 64 per candidate")


def d1_anchor_checkpoint(root: Path, split_id: str, config: dict[str, Any]) -> str:
    """Resolve the paired frozen D1-A checkpoint for pilot or formal evaluation."""
    if split_id == config["splits"]["pilot"]["id"]:
        return "experiments/D1-E001/pilot/A_truth_only/best_inference.pt"
    formal = config["splits"]["formal"]
    for index, split in enumerate(formal):
        if split_id == split["id"]:
            return str(config["models"]["S0"]["checkpoints"][index])
    raise KeyError(f"no frozen D1-A anchor is registered for split {split_id!r}")


def _load_committed_config(config_path: Path) -> dict[str, Any]:
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"invalid D1-E002 config: {config_path}")
    validate_e002_config(value)
    return value


def _run_pilot(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
    code_commit: str,
) -> Path:
    output_dir = root / "experiments" / "D1-E002" / "pilot"
    if output_dir.exists():
        raise FileExistsError(f"pilot output already exists; preserving it: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / "config_used.yaml")
    split = config["splits"]["pilot"]
    manifest = {
        "experiment_id": "D1-E002",
        "phase": "pilot",
        "status": "RUNNING",
        "code_commit": code_commit,
        "split_id": split["id"],
        "seed_values": sorted(e002_seed_values(config)),
        "environment": _environment(root, device),
        "blind_data_generated": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "environment.json", manifest["environment"])
    results: dict[str, Any] = {"status": "RUNNING", "baseline": {}, "candidates": {}}
    _write_json(output_dir / "metrics.json", results)
    try:
        cache = root / "experiments" / "D1-E002" / "data" / "cache" / split["id"]
        train_arrays = _open_or_generate_split(
            root=root,
            config=config,
            split=split,
            split_kind="train",
            cache_path=cache / "train",
        )
        validation_arrays = _open_or_generate_split(
            root=root,
            config=config,
            split=split,
            split_kind="validation",
            cache_path=cache / "validation",
        )
        anchor_relative = d1_anchor_checkpoint(root, split["id"], config)
        anchor = _load_d1_anchor(root, anchor_relative, device)
        s0_validation = _evaluate_model(anchor, validation_arrays, config, device, variant=None)
        anchor.to("cpu")
        del anchor
        gc.collect()
        torch.cuda.empty_cache()
        results["baseline"] = {"S0": {"checkpoint": anchor_relative, "validation": s0_validation}}
        _write_json(output_dir / "metrics.json", results)

        for variant in config["training"]["candidate_order"]:
            candidate_dir = output_dir / variant
            summary = _train_candidate_with_fallback(
                variant=variant,
                split=split,
                train_arrays=train_arrays,
                validation_arrays=validation_arrays,
                config=config,
                device=device,
                out_dir=candidate_dir,
                code_commit=code_commit,
            )
            if summary.get("completed_updates") == int(config["models"][variant]["optimizer_updates"]):
                results["candidates"][variant] = {
                    **summary["best_validation"],
                    "training": summary,
                }
            else:
                results["candidates"][variant] = {"training": summary}
            _write_json(candidate_dir / "summary.json", summary)
            _write_json(output_dir / "metrics.json", results)

        pilot_metrics = {
            name: value
            for name, value in results["candidates"].items()
            if value.get("training", {}).get("completed_updates")
            == int(config["models"][name]["optimizer_updates"])
        }
        s3_validation = pilot_metrics.get("S3", {}).get("matched_nonlearning_fft", {})
        three_point = s3_validation.get("three_point_log_parabolic", {})
        selection = select_formal_candidates(
            pilot_metrics,
            s0_validation,
            three_point,
            config,
        )
        selection["pilot_screen"] = config["acceptance"]["pilot_screen"]
        selection["code_commit"] = code_commit
        _write_json(output_dir / "selection.json", selection)
        results["selection"] = selection
        results["status"] = "COMPLETED"
        _write_json(output_dir / "metrics.json", results)
        manifest["status"] = "COMPLETED"
        manifest["formal_candidates"] = selection["formal_candidates"]
        _write_json(output_dir / "run_manifest.json", manifest)
        return output_dir
    except BaseException as error:
        results["status"] = "FAILED"
        results["failure"] = f"{type(error).__name__}: {error}"
        _write_json(output_dir / "metrics.json", results)
        manifest["status"] = "FAILED"
        manifest["failure"] = results["failure"]
        _write_json(output_dir / "run_manifest.json", manifest)
        raise


def reselect_d1_e002_pilot(root: Path, config_path: Path) -> Path:
    """Recompute the pilot screen from committed validation metrics after a selector fix."""
    root = root.resolve()
    config_path = config_path.resolve()
    registered_config = (root / "experiments" / "D1-E002" / "config.yaml").resolve()
    if config_path != registered_config:
        raise ValueError("D1-E002 reselection only accepts its registered config.yaml")
    config = _load_committed_config(config_path)
    assert_preregistered_files(root, config_path)
    selector_code_commit = assert_committed_implementation(root)
    selector_script = root / "scripts" / "reselect_d1_e002_pilot.py"
    subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", "scripts/reselect_d1_e002_pilot.py"],
        check=True,
        capture_output=True,
    )
    script_status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--", "scripts/reselect_d1_e002_pilot.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if script_status or not selector_script.is_file():
        raise RuntimeError("commit the clean D1-E002 pilot reselection utility before use")

    pilot_dir = root / "experiments" / "D1-E002" / "pilot"
    metrics_path = pilot_dir / "metrics.json"
    selection_path = pilot_dir / "selection.json"
    manifest_path = pilot_dir / "run_manifest.json"
    corrected_path = pilot_dir / "selection_corrected.json"
    correction_path = pilot_dir / "selection_correction.json"
    if corrected_path.exists() or correction_path.exists():
        raise FileExistsError("a D1-E002 pilot reselection already exists; preserving existing artifacts")
    for path in (metrics_path, selection_path, manifest_path):
        relative = path.resolve().relative_to(root).as_posix()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", relative],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError(f"commit the original pilot artifact before reselection: {relative}")
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
        )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    original_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    pilot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pilot_code_commit = str(pilot_manifest.get("code_commit", ""))
    if (
        metrics.get("status") != "COMPLETED"
        or pilot_manifest.get("status") != "COMPLETED"
        or metrics.get("selection") != original_selection
        or original_selection.get("code_commit") != pilot_code_commit
        or metrics.get("blind_data_generated", False)
        or pilot_manifest.get("blind_data_generated", False)
    ):
        raise RuntimeError("reselection requires the original completed pilot with no blind data")

    pilot_metrics = {
        name: value
        for name, value in metrics.get("candidates", {}).items()
        if int(value.get("training", {}).get("completed_updates", -1))
        == int(config["models"][name]["optimizer_updates"])
    }
    matched_fft = (
        pilot_metrics.get("S3", {})
        .get("matched_nonlearning_fft", {})
        .get("three_point_log_parabolic", {})
    )
    selection = select_formal_candidates(
        pilot_metrics,
        metrics["baseline"]["S0"]["validation"],
        matched_fft,
        config,
    )
    correction = {
        "status": "CORRECTED_FROM_PILOT_VALIDATION",
        "reason": "pilot candidate summaries store completed_updates under training, while the original selector read the top-level field",
        "pilot_code_commit": pilot_code_commit,
        "selector_code_commit": selector_code_commit,
        "source_metrics": "experiments/D1-E002/pilot/metrics.json",
        "original_selection": "experiments/D1-E002/pilot/selection.json",
        "corrected_selection": "experiments/D1-E002/pilot/selection_corrected.json",
        "formal_candidates": selection["formal_candidates"],
        "blind_data_generated": False,
        "thresholds_changed": False,
        "pilot_data_regenerated": False,
    }
    selection["pilot_screen"] = config["acceptance"]["pilot_screen"]
    selection["code_commit"] = pilot_code_commit
    selection["selection_correction"] = {
        "pilot_code_commit": pilot_code_commit,
        "selector_code_commit": selector_code_commit,
        "reason": correction["reason"],
    }
    _write_json(corrected_path, selection)
    _write_json(correction_path, correction)
    return corrected_path


def _require_committed_pilot(root: Path, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    pilot_dir = root / "experiments" / "D1-E002" / "pilot"
    original_selection_path = pilot_dir / "selection.json"
    corrected_selection_path = pilot_dir / "selection_corrected.json"
    correction_path = pilot_dir / "selection_correction.json"
    manifest_path = pilot_dir / "run_manifest.json"
    if not original_selection_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("formal training requires a completed, committed E002 pilot")
    original_selection = json.loads(original_selection_path.read_text(encoding="utf-8"))
    pilot_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frozen_code = str(pilot_manifest.get("code_commit", ""))
    if pilot_manifest.get("status") != "COMPLETED" or original_selection.get("code_commit") != frozen_code:
        raise RuntimeError("formal training requires a completed pilot with a matching frozen code commit")
    corrected = corrected_selection_path.is_file() or correction_path.is_file()
    if corrected_selection_path.is_file() != correction_path.is_file():
        raise RuntimeError("pilot selection correction artifacts are incomplete")
    selection = (
        json.loads(corrected_selection_path.read_text(encoding="utf-8"))
        if corrected
        else original_selection
    )
    required_paths = [
        Path("experiments/D1-E002/pilot/config_used.yaml"),
        Path("experiments/D1-E002/pilot/metrics.json"),
        Path("experiments/D1-E002/pilot/selection.json"),
        Path("experiments/D1-E002/pilot/run_manifest.json"),
    ]
    if corrected:
        required_paths.extend(
            (
                Path("experiments/D1-E002/pilot/selection_corrected.json"),
                Path("experiments/D1-E002/pilot/selection_correction.json"),
            )
        )
    pilot_metrics = json.loads((pilot_dir / "metrics.json").read_text(encoding="utf-8"))
    if pilot_metrics.get("status") != "COMPLETED":
        raise RuntimeError("formal training requires completed pilot metrics")
    if pilot_metrics.get("blind_data_generated", False) or pilot_manifest.get("blind_data_generated", False):
        raise RuntimeError("pilot data must not include any blind evaluation")
    for candidate in ("S1", "S2", "S3"):
        candidate_result = pilot_metrics.get("candidates", {}).get(candidate, {})
        training_result = candidate_result.get("training", {})
        required_paths.append(Path(f"experiments/D1-E002/pilot/{candidate}/summary.json"))
        if int(training_result.get("completed_updates", -1)) == int(
            config["models"][candidate]["optimizer_updates"]
        ):
            required_paths.extend(
                Path(f"experiments/D1-E002/pilot/{candidate}/{name}")
                for name in ("initial_inference.pt", "best_inference.pt", "training_history.csv")
            )
        else:
            required_paths.append(Path(f"experiments/D1-E002/pilot/{candidate}/failure.json"))
    for relative in required_paths:
        absolute = root / relative
        if not absolute.is_file():
            raise FileNotFoundError(f"pilot artifact required before formal training: {relative}")
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", relative.as_posix()],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if status:
            raise RuntimeError(f"commit pilot artifact before formal training: {relative}")
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative.as_posix()],
            check=True,
            capture_output=True,
        )
    if corrected:
        correction = json.loads(correction_path.read_text(encoding="utf-8"))
        metadata = selection.get("selection_correction", {})
        if (
            correction.get("status") != "CORRECTED_FROM_PILOT_VALIDATION"
            or correction.get("pilot_code_commit") != frozen_code
            or selection.get("code_commit") != frozen_code
            or metadata.get("pilot_code_commit") != frozen_code
            or metadata.get("selector_code_commit") != correction.get("selector_code_commit")
            or correction.get("blind_data_generated", True)
            or correction.get("thresholds_changed", True)
            or correction.get("pilot_data_regenerated", True)
            or correction.get("formal_candidates") != selection.get("formal_candidates")
            or pilot_metrics.get("selection") != original_selection
        ):
            raise RuntimeError("pilot selection correction provenance is inconsistent")
        pilot_candidates = {
            name: value
            for name, value in pilot_metrics.get("candidates", {}).items()
            if int(value.get("training", {}).get("completed_updates", -1))
            == int(config["models"][name]["optimizer_updates"])
        }
        matched_fft = (
            pilot_candidates.get("S3", {})
            .get("matched_nonlearning_fft", {})
            .get("three_point_log_parabolic", {})
        )
        expected = select_formal_candidates(
            pilot_candidates,
            pilot_metrics["baseline"]["S0"]["validation"],
            matched_fft,
            config,
        )
        for key in ("selection_source", "blind_data_used", "formal_candidates", "candidates", "no_candidate_action"):
            if selection.get(key) != expected.get(key):
                raise RuntimeError(f"corrected pilot selection does not match saved metrics: {key}")
        correction_code = str(correction.get("selector_code_commit", ""))
        if not correction_code:
            raise RuntimeError("pilot selection correction lacks its code commit")
    else:
        correction_code = frozen_code
    frozen_paths = [
        "experiments/D1-E002/config.yaml",
        "experiments/D1-E002/protocol.md",
        "src/radardistill/training/d1_e002.py",
        "src/radardistill/students/d1_e002.py",
        "src/radardistill/evaluation/d1_e002.py",
        "scripts/run_d1_e002.py",
        "scripts/evaluate_d1_e002.py",
        "scripts/validate_d1_e002_protocol.py",
    ]
    changed = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{frozen_code}..HEAD", "--", *frozen_paths],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if corrected:
        allowed = ["src/radardistill/training/d1_e002.py"]
        if changed != allowed:
            raise RuntimeError(f"pilot correction changed files outside the selector implementation: {changed}")
        changed_after_correction = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", f"{correction_code}..HEAD", "--", *frozen_paths],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        if changed_after_correction:
            raise RuntimeError(f"D1-E002 implementation changed after the recorded selector correction: {changed_after_correction}")
        subprocess.run(
            ["git", "-C", str(root), "merge-base", "--is-ancestor", correction_code, "HEAD"],
            check=True,
            capture_output=True,
        )
    elif changed:
        raise RuntimeError(f"D1-E002 frozen implementation changed after pilot: {changed}")
    return selection, correction_code


def _run_formal(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> Path:
    selection, code_commit = _require_committed_pilot(root, config)
    candidates = list(selection.get("formal_candidates", []))
    if any(name not in {"S1", "S2", "S3"} for name in candidates):
        raise ValueError(f"pilot selection contains an unknown candidate: {candidates}")
    output_dir = root / "experiments" / "D1-E002" / "formal"
    if output_dir.exists():
        raise FileExistsError(f"formal output already exists; preserving it: {output_dir}")
    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / "config_used.yaml")
    if not candidates:
        _write_json(
            output_dir / "skipped.json",
            {
                "status": "STOP_NO_PILOT_CANDIDATE_PASSED",
                "formal_candidates": [],
                "blind_data_generated": False,
                "pilot_selection": (
                    "experiments/D1-E002/pilot/selection_corrected.json"
                    if selection.get("selection_correction")
                    else "experiments/D1-E002/pilot/selection.json"
                ),
            },
        )
        return output_dir
    manifest = {
        "experiment_id": "D1-E002",
        "phase": "formal",
        "status": "RUNNING",
        "code_commit": code_commit,
        "pilot_code_commit": selection.get("code_commit"),
        "pilot_selection_file": (
            "experiments/D1-E002/pilot/selection_corrected.json"
            if selection.get("selection_correction")
            else "experiments/D1-E002/pilot/selection.json"
        ),
        "formal_candidates": candidates,
        "environment": _environment(root, device),
        "blind_data_generated": False,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    _write_json(output_dir / "environment.json", manifest["environment"])
    results: dict[str, Any] = {"status": "RUNNING", "pilot_selection": selection, "seeds": {}}
    _write_json(output_dir / "metrics.json", results)
    try:
        for index, split in enumerate(config["splits"]["formal"]):
            seed_dir = output_dir / split["id"]
            seed_dir.mkdir()
            cache = root / "experiments" / "D1-E002" / "data" / "cache" / split["id"]
            train_arrays = _open_or_generate_split(
                root=root,
                config=config,
                split=split,
                split_kind="train",
                cache_path=cache / "train",
            )
            validation_arrays = _open_or_generate_split(
                root=root,
                config=config,
                split=split,
                split_kind="validation",
                cache_path=cache / "validation",
            )
            anchor_relative = d1_anchor_checkpoint(root, split["id"], config)
            anchor = _load_d1_anchor(root, anchor_relative, device)
            s0_validation = _evaluate_model(anchor, validation_arrays, config, device, variant=None)
            anchor.to("cpu")
            del anchor
            gc.collect()
            torch.cuda.empty_cache()
            seed_result: dict[str, Any] = {
                "S0": {"checkpoint": anchor_relative, "validation": s0_validation},
                "candidates": {},
            }
            results["seeds"][split["id"]] = seed_result
            for variant in candidates:
                candidate_dir = seed_dir / variant
                summary = _train_candidate_with_fallback(
                    variant=variant,
                    split=split,
                    train_arrays=train_arrays,
                    validation_arrays=validation_arrays,
                    config=config,
                    device=device,
                    out_dir=candidate_dir,
                    code_commit=code_commit,
                )
                candidate_result: dict[str, Any] = {"training": summary}
                if summary.get("completed_updates") == int(config["models"][variant]["optimizer_updates"]):
                    candidate_result["validation"] = summary["best_validation"]
                seed_result["candidates"][variant] = candidate_result
                _write_json(candidate_dir / "summary.json", summary)
                _write_json(output_dir / "metrics.json", results)
        incomplete = [
            f"{seed_id}/{variant}"
            for seed_id, seed_result in results["seeds"].items()
            for variant in candidates
            if seed_result["candidates"].get(variant, {}).get("training", {}).get("completed_updates")
            != int(config["models"][variant]["optimizer_updates"])
        ]
        if incomplete:
            results["status"] = "FAILED_INCOMPLETE_BUDGET"
            results["incomplete_runs"] = incomplete
            _write_json(output_dir / "metrics.json", results)
            manifest["status"] = "FAILED_INCOMPLETE_BUDGET"
            manifest["incomplete_runs"] = incomplete
            _write_json(output_dir / "run_manifest.json", manifest)
            raise RuntimeError(f"formal training did not complete the frozen budget: {incomplete}")
        combined_rows: list[dict[str, Any]] = []
        fieldnames: set[str] = {"seed_id", "variant"}
        for split in config["splits"]["formal"]:
            for variant in candidates:
                history_path = output_dir / split["id"] / variant / "training_history.csv"
                with history_path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                    fieldnames.update(reader.fieldnames or ())
                combined_rows.extend(
                    {"seed_id": split["id"], "variant": variant, **row} for row in rows
                )
        with (output_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as handle:
            ordered_fields = ["seed_id", "variant", *sorted(fieldnames - {"seed_id", "variant"})]
            writer = csv.DictWriter(handle, fieldnames=ordered_fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(combined_rows)
        results["status"] = "COMPLETED"
        _write_json(output_dir / "metrics.json", results)
        manifest["status"] = "COMPLETED"
        _write_json(output_dir / "run_manifest.json", manifest)
        return output_dir
    except BaseException as error:
        results["status"] = "FAILED"
        results["failure"] = f"{type(error).__name__}: {error}"
        _write_json(output_dir / "metrics.json", results)
        manifest["status"] = "FAILED"
        manifest["failure"] = results["failure"]
        _write_json(output_dir / "run_manifest.json", manifest)
        raise


def run_d1_e002(root: Path, config_path: Path, phase: str) -> Path:
    """Execute one frozen pilot/formal phase, sequentially on CUDA:0."""
    root = root.resolve()
    config_path = config_path.resolve()
    if phase not in {"pilot", "formal"}:
        raise ValueError("phase must be pilot or formal")
    if config_path != (root / "experiments" / "D1-E002" / "config.yaml").resolve():
        raise ValueError("D1-E002 runner only accepts its registered config.yaml")
    config = _load_committed_config(config_path)
    assert_preregistered_files(root, config_path)
    code_commit = assert_committed_implementation(root)
    validate_e002_seed_disjointness(config, existing_seed_values=prior_registered_seed_values(root))
    device = require_cuda(str(config["training"]["device"]))
    if phase == "formal":
        return _run_formal(root, config_path, config, device)
    return _run_pilot(root, config_path, config, device, code_commit)
