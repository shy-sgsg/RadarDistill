"""CUDA-only, matched-budget trainer for D1-E001."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..distillation.task_losses import decode_task_outputs, task_loss
from ..simulation.d0 import D0Batch, generate_d0_batch
from ..students.task_student import TaskStudent, export_inference_model
from ..teachers.apx15_doppler import APX15DopplerTeacher, TeacherBatch


_CACHE_ARRAYS = (
    "iq",
    "truth_presence",
    "truth_range_cell",
    "truth_doppler_hz",
    "truth_initial_phase_rad",
    "truth_azimuth_deg",
    "truth_elevation_deg",
    "truth_target_amplitude",
    "nuisance_initial_phase_rad",
    "nuisance_azimuth_deg",
    "nuisance_elevation_deg",
    "reference_amplitude",
    "noise_power",
    "snr_db",
    "teacher_response",
    "teacher_detection_score",
    "teacher_range_cell",
    "teacher_doppler_hz",
    "teacher_confidence",
)


def require_cuda(
    device: str = "cuda",
    *,
    cuda_available: bool | None = None,
    device_count: int | None = None,
) -> torch.device:
    """Fail before data work if D1 training cannot use CUDA; never fall back."""
    if device not in {"cuda", "cuda:0"}:
        raise ValueError("D1 training.device must be cuda:0; CPU fallback is forbidden")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    count = torch.cuda.device_count() if device_count is None else device_count
    if not available or count < 1:
        raise RuntimeError("CUDA is required for D1; no model training was started")
    selected = torch.device("cuda:0")
    torch.cuda.set_device(selected)
    return selected


def strategy_update_counts(methods: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for name, method in methods.items():
        stages = method.get("stages", ())
        if not stages or any(int(stage.get("updates", 0)) <= 0 for stage in stages):
            raise ValueError(f"method {name} must have positive updates in every stage")
        counts[name] = sum(int(stage["updates"]) for stage in stages)
    return counts


def validation_seeds_by_snr(base_seed: int, stride: int, snr_count: int) -> list[int]:
    if base_seed < 0 or stride <= 0 or snr_count <= 0:
        raise ValueError("validation seed base, stride, and SNR count must be positive")
    return [int(base_seed + stride * index) for index in range(snr_count)]


def calibrate_score_threshold(scores: Any, pfa: float) -> float:
    """Finite-sample order-statistic threshold with strict score > threshold."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("calibration scores must be nonempty and finite")
    if not math.isfinite(pfa) or not 0.0 < pfa < 1.0:
        raise ValueError("pfa must lie strictly between zero and one")
    index = int(math.ceil((values.size + 1) * (1.0 - pfa))) - 1
    index = min(max(index, 0), values.size - 1)
    return float(np.partition(values, index)[index])


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.bool_).reshape(-1)
    positives = int(labels.sum())
    negatives = labels.size - positives
    if positives == 0 or negatives == 0:
        return float("nan")
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
    positive_rank_sum = ranks[sorted_labels].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _write_split_cache(
    *,
    cache_root: Path,
    tag: str,
    base_seed: int,
    examples_per_class: int,
    snr_spec: tuple[float, float] | list[float],
    validation_snr: list[float] | None,
    validation_seed_stride: int,
    d0_config: dict[str, Any],
    teacher: APX15DopplerTeacher,
) -> dict[str, np.ndarray]:
    cache_dir = cache_root / tag
    manifest_path = cache_dir / "manifest.json"
    snr_count = len(validation_snr) if validation_snr is not None else 1
    total = 2 * examples_per_class * snr_count
    expected = {
        "tag": tag,
        "base_seed": int(base_seed),
        "examples_per_class": int(examples_per_class),
        "snr_spec": list(snr_spec) if isinstance(snr_spec, tuple) else snr_spec,
        "validation_snr": validation_snr,
        "validation_seed_stride": int(validation_seed_stride),
        "snr_count": snr_count,
        "total_examples": total,
        "shape": [total, 3, 130, 32],
        "teacher_response_shape": [total, 130, 32],
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"cached D1 split {tag} does not match the frozen config")
        missing = [name for name in _CACHE_ARRAYS if not (cache_dir / f"{name}.npy").is_file()]
        if missing:
            raise FileNotFoundError(f"cached D1 split {tag} is incomplete: {missing}")
        return {
            name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in _CACHE_ARRAYS
        }
    if cache_dir.exists() and any(cache_dir.iterdir()):
        raise FileExistsError(
            f"partial D1 cache exists at {cache_dir}; preserving it, choose a new cache root"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    shapes = {
        "iq": (total, 3, 130, 32),
        "truth_presence": (total,),
        "truth_range_cell": (total,),
        "truth_doppler_hz": (total,),
        "truth_initial_phase_rad": (total,),
        "truth_azimuth_deg": (total,),
        "truth_elevation_deg": (total,),
        "truth_target_amplitude": (total,),
        "nuisance_initial_phase_rad": (total,),
        "nuisance_azimuth_deg": (total,),
        "nuisance_elevation_deg": (total,),
        "reference_amplitude": (total,),
        "noise_power": (total,),
        "snr_db": (total,),
        "teacher_response": (total, 130, 32),
        "teacher_detection_score": (total,),
        "teacher_range_cell": (total,),
        "teacher_doppler_hz": (total,),
        "teacher_confidence": (total,),
    }
    dtypes = {
        "iq": np.complex64,
        "truth_presence": np.bool_,
        "truth_range_cell": np.float32,
        "truth_doppler_hz": np.float32,
        "truth_initial_phase_rad": np.float32,
        "truth_azimuth_deg": np.float32,
        "truth_elevation_deg": np.float32,
        "truth_target_amplitude": np.float32,
        "nuisance_initial_phase_rad": np.float32,
        "nuisance_azimuth_deg": np.float32,
        "nuisance_elevation_deg": np.float32,
        "reference_amplitude": np.float32,
        "noise_power": np.float32,
        "snr_db": np.float32,
        "teacher_response": np.float32,
        "teacher_detection_score": np.float32,
        "teacher_range_cell": np.float32,
        "teacher_doppler_hz": np.float32,
        "teacher_confidence": np.float32,
    }
    maps = {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy", mode="w+", dtype=dtypes[name], shape=shape
        )
        for name, shape in shapes.items()
    }
    if validation_snr is None:
        chunk = int(d0_config["execution"]["batch_size_cpu_teacher"])
        if examples_per_class % chunk:
            raise ValueError("training examples per class must divide the D0 teacher batch size")
        for chunk_index, start in enumerate(range(0, examples_per_class, chunk)):
            count = min(chunk, examples_per_class - start)
            sample_seed = int(base_seed + validation_seed_stride * chunk_index)
            h0 = generate_d0_batch(count, sample_seed, tuple(snr_spec), 0.0, d0_config)
            h1 = generate_d0_batch(count, sample_seed, tuple(snr_spec), 1.0, d0_config)
            batch = _join_classes(h0, h1)
            prediction = teacher.predict(batch.iq)
            _write_batch_to_maps(maps, start, examples_per_class, batch, prediction)
    else:
        snr_values = list(validation_snr)
        stride = int(validation_seed_stride)
        samples_per_snr = examples_per_class
        for snr_index, snr in enumerate(snr_values):
            seed = int(base_seed + stride * snr_index)
            h0 = generate_d0_batch(samples_per_snr, seed, float(snr), 0.0, d0_config)
            h1 = generate_d0_batch(samples_per_snr, seed, float(snr), 1.0, d0_config)
            batch = _join_classes(h0, h1)
            prediction = teacher.predict(batch.iq)
            start = snr_index * 2 * samples_per_snr
            _write_batch_to_maps(maps, start, samples_per_snr, batch, prediction, paired_snr=True)

    for array in maps.values():
        array.flush()
    manifest_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in _CACHE_ARRAYS}


def _join_classes(h0: D0Batch, h1: D0Batch) -> D0Batch:
    values: dict[str, Any] = {}
    for field in D0Batch.__dataclass_fields__:
        left = getattr(h0, field)
        right = getattr(h1, field)
        values[field] = np.concatenate((left, right), axis=0)
    return D0Batch(**values)


def _write_batch_to_maps(
    maps: dict[str, np.ndarray],
    start: int,
    examples_per_class: int,
    batch: D0Batch,
    teacher: TeacherBatch,
    *,
    paired_snr: bool = False,
) -> None:
    if paired_snr:
        per_class = batch.iq.shape[0] // 2
        h0_slice = slice(start, start + per_class)
        h1_slice = slice(start + per_class, start + 2 * per_class)
    else:
        count = batch.iq.shape[0] // 2
        h0_start = start
        h1_start = examples_per_class + start
        h0_slice = slice(h0_start, h0_start + count)
        h1_slice = slice(h1_start, h1_start + count)
    for class_slice, indices in ((h0_slice, np.arange(batch.iq.shape[0] // 2)), (h1_slice, np.arange(batch.iq.shape[0] // 2, batch.iq.shape[0]))):
        maps["iq"][class_slice] = batch.iq[indices]
        maps["truth_presence"][class_slice] = batch.truth_presence[indices]
        maps["truth_range_cell"][class_slice] = batch.truth_range_cell[indices]
        maps["truth_doppler_hz"][class_slice] = batch.truth_doppler_hz[indices]
        maps["truth_initial_phase_rad"][class_slice] = batch.truth_initial_phase_rad[indices]
        maps["truth_azimuth_deg"][class_slice] = batch.truth_azimuth_deg[indices]
        maps["truth_elevation_deg"][class_slice] = batch.truth_elevation_deg[indices]
        maps["truth_target_amplitude"][class_slice] = batch.truth_target_amplitude[indices]
        maps["nuisance_initial_phase_rad"][class_slice] = batch.nuisance_initial_phase_rad[indices]
        maps["nuisance_azimuth_deg"][class_slice] = batch.nuisance_azimuth_deg[indices]
        maps["nuisance_elevation_deg"][class_slice] = batch.nuisance_elevation_deg[indices]
        maps["reference_amplitude"][class_slice] = batch.reference_amplitude[indices]
        maps["noise_power"][class_slice] = batch.noise_power[indices]
        maps["snr_db"][class_slice] = batch.snr_db[indices]
        maps["teacher_response"][class_slice] = teacher.teacher_response[indices]
        maps["teacher_detection_score"][class_slice] = teacher.teacher_detection_score[indices]
        maps["teacher_range_cell"][class_slice] = teacher.teacher_range_cell[indices]
        maps["teacher_doppler_hz"][class_slice] = teacher.teacher_doppler_hz[indices]
        maps["teacher_confidence"][class_slice] = teacher.teacher_confidence[indices]


def _load_d0_config(root: Path) -> dict[str, Any]:
    path = root / "experiments" / "D0-E001" / "config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_memmap_truth(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "truth_presence": torch.from_numpy(np.asarray(arrays["truth_presence"][indices]).copy()),
        "truth_range_cell": torch.from_numpy(np.asarray(arrays["truth_range_cell"][indices]).copy()),
        "truth_doppler_hz": torch.from_numpy(np.asarray(arrays["truth_doppler_hz"][indices]).copy()),
        "teacher_response": torch.from_numpy(np.asarray(arrays["teacher_response"][indices]).copy()),
    }


def _iq_to_ri(iq: np.ndarray) -> torch.Tensor:
    values = np.asarray(iq)
    if values.dtype != np.complex64:
        values = values.astype(np.complex64, copy=False)
    planes = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(planes))


def _batch_orders(count: int, batch_size: int, updates: int, seed: int) -> np.ndarray:
    if count % batch_size:
        raise ValueError("training set size must divide the effective batch size")
    rng = np.random.default_rng(seed)
    result = np.empty((updates, batch_size), dtype=np.int64)
    permutation = rng.permutation(count)
    cursor = 0
    for update in range(updates):
        if cursor + batch_size > count:
            permutation = rng.permutation(count)
            cursor = 0
        result[update] = permutation[cursor : cursor + batch_size]
        cursor += batch_size
    return result


def _learning_rate(step: int, total_steps: int, initial: float, final: float) -> float:
    fraction = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * fraction))


def _stage_for_update(method: dict[str, Any], update: int) -> tuple[dict[str, Any], int, int]:
    offset = 0
    for stage in method["stages"]:
        end = offset + int(stage["updates"])
        if update < end:
            return stage, update - offset, int(stage["updates"])
        offset = end
    raise IndexError(f"update {update} exceeds strategy schedule")


def _teacher_weight(stage: dict[str, Any], stage_update: int, stage_updates: int) -> float:
    if "teacher_weight_start" in stage:
        start = float(stage["teacher_weight_start"])
        end = float(stage["teacher_weight_end"])
        fraction = stage_update / max(stage_updates - 1, 1)
        return start + (end - start) * fraction
    return float(stage.get("teacher_weight", 0.0))


def _accumulation_scaled_loss(
    losses: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    microbatch_size: int,
    effective_batch_size: int,
    microbatch_h1: int,
    effective_h1: int,
    task_weight: float,
    teacher_weight: float,
) -> torch.Tensor:
    """Scale microbatch means to reproduce one loss over the effective batch."""
    if effective_batch_size <= 0 or not 0 < microbatch_size <= effective_batch_size:
        raise ValueError("microbatch size must lie within a positive effective batch")
    if effective_h1 < 0 or not 0 <= microbatch_h1 <= effective_h1:
        raise ValueError("microbatch H1 count must lie within the effective H1 count")
    sample_fraction = microbatch_size / effective_batch_size
    h1_fraction = microbatch_h1 / effective_h1 if effective_h1 else 0.0
    weights = config["training"]["loss"]
    task = (
        float(weights["presence_bce_weight"]) * losses["presence"] * sample_fraction
        + float(weights["range_cell_cross_entropy_weight"]) * losses["range_cell"] * h1_fraction
        + float(weights["range_offset_smooth_l1_weight"]) * losses["range_offset"] * h1_fraction
        + float(weights["doppler_circular_cosine_weight"]) * losses["doppler"] * h1_fraction
    )
    return float(task_weight) * task + float(teacher_weight) * losses["teacher_kl"] * sample_fraction


def _task_metrics(
    model: TaskStudent,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    batch_size = int(config["training"]["effective_batch_size"])
    count = len(arrays["truth_presence"])
    logits: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    range_errors: list[np.ndarray] = []
    doppler_errors: list[np.ndarray] = []
    parameter_snr: list[np.ndarray] = []
    all_snr: list[np.ndarray] = []
    presence_loss_sum = 0.0
    range_loss_sum = 0.0
    offset_loss_sum = 0.0
    doppler_loss_sum = 0.0
    h1_count_total = 0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            indices = np.arange(start, min(start + batch_size, count), dtype=np.int64)
            inputs = _iq_to_ri(arrays["iq"][indices]).to(device, non_blocking=False)
            truth = _load_memmap_truth(arrays, indices)
            truth = {key: value.to(device) for key, value in truth.items() if key != "teacher_response"}
            outputs = model(inputs, compute_spectrum=False)
            losses = task_loss(outputs, truth, config, teacher_weight=0.0)
            batch_h1 = int(np.count_nonzero(arrays["truth_presence"][indices]))
            presence_loss_sum += float(losses["presence"].item()) * len(indices)
            if batch_h1:
                range_loss_sum += float(losses["range_cell"].item()) * batch_h1
                offset_loss_sum += float(losses["range_offset"].item()) * batch_h1
                doppler_loss_sum += float(losses["doppler"].item()) * batch_h1
                h1_count_total += batch_h1
            decoded = decode_task_outputs(outputs, float(config["radar"]["prf_hz"]))
            logits.append(outputs["presence_logit"].detach().cpu().numpy())
            truths.append(np.asarray(arrays["truth_presence"][indices], dtype=np.bool_))
            all_snr.append(np.asarray(arrays["snr_db"][indices], dtype=np.float64))
            h1 = np.asarray(arrays["truth_presence"][indices], dtype=np.bool_)
            if np.any(h1):
                true_range = np.asarray(arrays["truth_range_cell"][indices][h1], dtype=np.float64)
                true_doppler = np.asarray(arrays["truth_doppler_hz"][indices][h1], dtype=np.float64)
                pred_range = decoded["range_cell"].detach().cpu().numpy()[h1]
                pred_doppler = decoded["doppler_hz"].detach().cpu().numpy()[h1]
                range_errors.append(pred_range - true_range)
                doppler_errors.append(
                    (pred_doppler - true_doppler + 650.0) % 1300.0 - 650.0
                )
                parameter_snr.append(np.asarray(arrays["snr_db"][indices][h1], dtype=np.float64))
    score = np.concatenate(logits)
    label = np.concatenate(truths)
    range_error = np.concatenate(range_errors)
    doppler_error = np.concatenate(doppler_errors)
    snr_all = np.concatenate(all_snr)
    snr_parameters = np.concatenate(parameter_snr)
    loss_weights = config["training"]["loss"]
    validation_task_loss = (
        float(loss_weights["presence_bce_weight"]) * presence_loss_sum / count
        + float(loss_weights["range_cell_cross_entropy_weight"]) * range_loss_sum / h1_count_total
        + float(loss_weights["range_offset_smooth_l1_weight"]) * offset_loss_sum / h1_count_total
        + float(loss_weights["doppler_circular_cosine_weight"]) * doppler_loss_sum / h1_count_total
    )
    by_snr: dict[str, dict[str, float]] = {}
    for snr in sorted(np.unique(snr_all)):
        snr_key = str(int(snr)) if float(snr).is_integer() else str(float(snr))
        sample_mask = snr_all == snr
        h1_mask = snr_parameters == snr
        by_snr[snr_key] = {
            "presence_auc": _auc(score[sample_mask], label[sample_mask]),
            "range_rmse_cells": float(np.sqrt(np.mean(range_error[h1_mask] ** 2))),
            "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error[h1_mask] ** 2))),
            "n_h0": int(np.count_nonzero(~label[sample_mask])),
            "n_h1": int(np.count_nonzero(label[sample_mask])),
        }
    return {
        "task_loss": float(validation_task_loss),
        "presence_auc": _auc(score, label),
        "range_rmse_cells": float(np.sqrt(np.mean(range_error**2))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error**2))),
        "doppler_bias_hz": float(np.mean(doppler_error)),
        "by_snr": by_snr,
    }


def _teacher_metrics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    labels = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    scores = np.asarray(arrays["teacher_detection_score"], dtype=np.float64)
    h1 = labels
    range_error = np.asarray(arrays["teacher_range_cell"][h1], dtype=np.float64) - np.asarray(
        arrays["truth_range_cell"][h1], dtype=np.float64
    )
    doppler_error = (
        np.asarray(arrays["teacher_doppler_hz"][h1], dtype=np.float64)
        - np.asarray(arrays["truth_doppler_hz"][h1], dtype=np.float64)
        + 650.0
    ) % 1300.0 - 650.0
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    by_snr: dict[str, Any] = {}
    for value in sorted(np.unique(snr)):
        key = str(int(value)) if float(value).is_integer() else str(float(value))
        sample_mask = snr == value
        h1_mask = sample_mask & labels
        range_part = np.asarray(arrays["teacher_range_cell"][h1_mask], dtype=np.float64) - np.asarray(
            arrays["truth_range_cell"][h1_mask], dtype=np.float64
        )
        doppler_part = (
            np.asarray(arrays["teacher_doppler_hz"][h1_mask], dtype=np.float64)
            - np.asarray(arrays["truth_doppler_hz"][h1_mask], dtype=np.float64)
            + 650.0
        ) % 1300.0 - 650.0
        by_snr[key] = {
            "presence_auc": _auc(scores[sample_mask], labels[sample_mask]),
            "range_rmse_cells": float(np.sqrt(np.mean(range_part**2))),
            "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_part**2))),
            "n_h0": int(np.count_nonzero(~labels[sample_mask])),
            "n_h1": int(np.count_nonzero(labels[sample_mask])),
        }
    return {
        "presence_auc": _auc(scores, labels),
        "range_rmse_cells_all_h1": float(np.sqrt(np.mean(range_error**2))),
        "doppler_rmse_hz_all_h1": float(np.sqrt(np.mean(doppler_error**2))),
        "by_snr": by_snr,
    }


def _quantile_summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    absolute = np.abs(data)
    return {
        "mean_abs": float(np.mean(absolute)),
        "p95_abs": float(np.quantile(absolute, 0.95)),
        "max_abs": float(np.max(absolute)),
    }


def _grouped_quantile_summary(
    values: np.ndarray,
    eligible: np.ndarray,
    snr_db: np.ndarray,
) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64).reshape(-1)
    mask = np.asarray(eligible, dtype=np.bool_).reshape(-1)
    snr = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if data.shape != mask.shape or data.shape != snr.shape:
        raise ValueError("diagnostic values, eligibility mask, and SNR must have equal length")
    if not np.any(mask):
        raise ValueError("physical-consistency diagnostic group must not be empty")
    result: dict[str, Any] = {"n": int(np.count_nonzero(mask)), **_quantile_summary(data[mask])}
    by_snr: dict[str, Any] = {}
    for value in sorted(np.unique(snr[mask])):
        group = mask & (snr == value)
        key = str(int(value)) if float(value).is_integer() else str(float(value))
        by_snr[key] = {"n": int(np.count_nonzero(group)), **_quantile_summary(data[group])}
    result["by_snr_db"] = by_snr
    return result


def _physical_consistency_validation(
    model: TaskStudent,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    selection = config["splits"]["physical_consistency_validation"]
    count = min(int(selection["examples"]), len(arrays["iq"]))
    rng = np.random.default_rng(int(selection["seed"]))
    indices = np.sort(rng.choice(len(arrays["iq"]), size=count, replace=False))
    iq = np.asarray(arrays["iq"][indices]).copy()
    truth_presence = np.asarray(arrays["truth_presence"][indices], dtype=np.bool_)
    snr_db = np.asarray(arrays["snr_db"][indices], dtype=np.float64)
    prf_hz = float(config["radar"]["prf_hz"])
    model.eval()
    inference_batch = min(
        int(config["training"]["initial_microbatch_size"]),
        int(config["training"]["effective_batch_size"]),
    )

    def predict(values: np.ndarray) -> dict[str, np.ndarray]:
        predictions: dict[str, list[np.ndarray]] = {
            "presence": [],
            "range": [],
            "doppler": [],
        }
        with torch.no_grad():
            for start in range(0, len(values), inference_batch):
                inputs = _iq_to_ri(values[start : start + inference_batch]).to(device)
                outputs = model(inputs, compute_spectrum=False)
                decoded = decode_task_outputs(outputs, prf_hz)
                predictions["presence"].append(decoded["presence_probability"].cpu().numpy())
                predictions["range"].append(decoded["range_cell"].cpu().numpy())
                predictions["doppler"].append(decoded["doppler_hz"].cpu().numpy())
        return {name: np.concatenate(parts) for name, parts in predictions.items()}

    baseline = predict(iq)
    common_phase: dict[str, Any] = {}
    for angle in config["evaluation"]["diagnostics"]["common_phase_rotation_radians"]:
        rotated = iq * np.complex64(np.exp(1j * float(angle)))
        shifted = predict(rotated)
        delta_doppler = (shifted["doppler"] - baseline["doppler"] + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
        common_phase[str(angle)] = {
            "presence_probability_delta": _grouped_quantile_summary(
                shifted["presence"] - baseline["presence"], np.ones(count, dtype=np.bool_), snr_db
            ),
            "range_cell_delta": _grouped_quantile_summary(
                shifted["range"] - baseline["range"], truth_presence, snr_db
            ),
            "doppler_hz_delta": _grouped_quantile_summary(
                delta_doppler, truth_presence, snr_db
            ),
        }

    pulse = np.arange(iq.shape[2], dtype=np.float64) / prf_hz
    frequency_shift: dict[str, Any] = {}
    for amount in config["evaluation"]["diagnostics"]["report_frequency_shift_equivariance_hz"]:
        phasor = np.exp(2j * np.pi * float(amount) * pulse).astype(np.complex64)
        shifted = predict(iq * phasor[None, None, :, None])
        residual = (
            shifted["doppler"] - baseline["doppler"] - float(amount) + prf_hz / 2.0
        ) % prf_hz - prf_hz / 2.0
        frequency_shift[str(amount)] = {
            "presence_probability_delta": _grouped_quantile_summary(
                shifted["presence"] - baseline["presence"], np.ones(count, dtype=np.bool_), snr_db
            ),
            "range_cell_delta": _grouped_quantile_summary(
                shifted["range"] - baseline["range"], truth_presence, snr_db
            ),
            "doppler_equivariance_residual_hz": _grouped_quantile_summary(
                residual, truth_presence, snr_db
            ),
        }
    sample_counts = {
        str(int(value)) if float(value).is_integer() else str(float(value)): {
            "h0": int(np.count_nonzero((snr_db == value) & ~truth_presence)),
            "h1": int(np.count_nonzero((snr_db == value) & truth_presence)),
        }
        for value in sorted(np.unique(snr_db))
    }
    return {
        "n_validation_samples": count,
        "sample_counts_by_snr_db": sample_counts,
        "common_phase_rotation": common_phase,
        "frequency_shift_equivariance": frequency_shift,
    }


def _clone_state_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _save_checkpoint(
    path: Path,
    model: TaskStudent,
    *,
    config: dict[str, Any],
    method_name: str,
    seed_id: str,
    update: int,
    validation: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inference_model = export_inference_model(model).cpu().eval()
    torch.save(
        {
            "state_dict": inference_model.state_dict(),
            "student": config["student"],
            "method": method_name,
            "seed_id": seed_id,
            "update": int(update),
            "validation": validation,
        },
        path,
    )


def _train_strategy(
    *,
    model_name: str,
    method: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    out_dir: Path,
    seed_id: str,
    order_seed: int,
    requested_microbatch: int,
) -> tuple[dict[str, Any], TaskStudent]:
    training = config["training"]
    effective_batch = int(training["effective_batch_size"])
    updates = strategy_update_counts({model_name: method})[model_name]
    if effective_batch % requested_microbatch:
        raise ValueError("microbatch must divide the frozen effective batch size")
    model = TaskStudent(width=int(config["student"]["width"]), temporal_dilations=config["student"]["temporal_dilations"])
    model.load_state_dict(initial_state)
    model.to(device)
    torch.cuda.reset_peak_memory_stats(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["optimizer"]["learning_rate"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    orders = _batch_orders(
        len(train_arrays["truth_presence"]), effective_batch, updates, int(order_seed)
    )
    validation_interval = int(training["validation_interval_updates"])
    start_time = time.perf_counter()
    best_validation = math.inf
    best_update = 0
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    initial_validation = _task_metrics(model, validation_arrays, config, device)
    history.append({"update": 0, "stage": "initial", "train_total": float("nan"), **{f"validation_{k}": v for k, v in initial_validation.items()}})
    best_validation = initial_validation["task_loss"]
    best_metrics = initial_validation
    best_state = _clone_state_to_cpu(model)
    _save_checkpoint(
        out_dir / "initial_inference.pt",
        model,
        config=config,
        method_name=model_name,
        seed_id=seed_id,
        update=0,
        validation=initial_validation,
    )

    model.train()
    for update in range(updates):
        stage, stage_update, stage_updates = _stage_for_update(method, update)
        if model_name == "B_teacher_pretrain_then_truth" and stage["name"] == "truth_finetune":
            if model.spectrum_head is not None:
                for parameter in model.spectrum_head.parameters():
                    parameter.requires_grad_(False)
        teacher_weight = _teacher_weight(stage, stage_update, stage_updates)
        compute_spectrum = teacher_weight > 0.0
        lr = _learning_rate(
            update,
            updates,
            float(training["optimizer"]["learning_rate"]),
            float(training["optimizer"]["final_learning_rate"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        batch_indices = orders[update]
        effective_h1 = int(np.count_nonzero(train_arrays["truth_presence"][batch_indices]))
        loss_totals = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "teacher_kl", "total")}
        for offset in range(0, effective_batch, requested_microbatch):
            indices = batch_indices[offset : offset + requested_microbatch]
            inputs = _iq_to_ri(train_arrays["iq"][indices]).to(device)
            truth = _load_memmap_truth(train_arrays, indices)
            truth = {key: value.to(device) for key, value in truth.items()}
            if not compute_spectrum:
                truth.pop("teacher_response", None)
            outputs = model(inputs, compute_spectrum=compute_spectrum)
            losses = task_loss(
                outputs,
                truth,
                config,
                task_weight=float(stage.get("task_weight", 1.0)),
                teacher_weight=teacher_weight,
            )
            if any(not torch.isfinite(value).all() for value in losses.values()):
                raise FloatingPointError(f"non-finite D1 loss in {model_name} update {update + 1}")
            microbatch_h1 = int(np.count_nonzero(train_arrays["truth_presence"][indices]))
            sample_fraction = len(indices) / effective_batch
            h1_fraction = microbatch_h1 / effective_h1 if effective_h1 else 0.0
            scaled = _accumulation_scaled_loss(
                losses,
                config,
                microbatch_size=len(indices),
                effective_batch_size=effective_batch,
                microbatch_h1=microbatch_h1,
                effective_h1=effective_h1,
                task_weight=float(stage.get("task_weight", 1.0)),
                teacher_weight=teacher_weight,
            )
            scaled.backward()
            loss_totals["presence"] += float(losses["presence"].detach().item()) * sample_fraction
            loss_totals["range_cell"] += float(losses["range_cell"].detach().item()) * h1_fraction
            loss_totals["range_offset"] += float(losses["range_offset"].detach().item()) * h1_fraction
            loss_totals["doppler"] += float(losses["doppler"].detach().item()) * h1_fraction
            loss_totals["teacher_kl"] += float(losses["teacher_kl"].detach().item()) * sample_fraction
            loss_totals["total"] += float(scaled.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["optimizer"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite D1 gradient in {model_name} update {update + 1}")
        optimizer.step()
        row: dict[str, Any] = {
            "update": update + 1,
            "stage": str(stage["name"]),
            "teacher_weight": teacher_weight,
            "learning_rate": lr,
            "train_total": loss_totals["total"],
            "train_presence": loss_totals["presence"],
            "train_range_cell": loss_totals["range_cell"],
            "train_range_offset": loss_totals["range_offset"],
            "train_doppler": loss_totals["doppler"],
            "train_teacher_kl": loss_totals["teacher_kl"],
        }
        if (update + 1) % validation_interval == 0 or update + 1 == updates:
            validation = _task_metrics(model, validation_arrays, config, device)
            row.update({f"validation_{key}": value for key, value in validation.items()})
            if validation["task_loss"] < best_validation:
                best_validation = validation["task_loss"]
                best_update = update + 1
                best_metrics = validation
                best_state = _clone_state_to_cpu(model)
        history.append(row)

    if best_state is None:
        raise RuntimeError("D1 trainer failed to produce a selected checkpoint")
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start_time
    last_model = model
    _save_checkpoint(
        out_dir / "last_inference.pt",
        last_model,
        config=config,
        method_name=model_name,
        seed_id=seed_id,
        update=updates,
        validation=best_metrics,
    )
    model.load_state_dict(best_state)
    _save_checkpoint(
        out_dir / "best_inference.pt",
        model,
        config=config,
        method_name=model_name,
        seed_id=seed_id,
        update=best_update,
        validation=best_metrics,
    )
    history_path = out_dir / "training_history.csv"
    fieldnames = sorted({key for row in history for key in row})
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "method": model_name,
        "seed_id": seed_id,
        "parameter_count_training_model": int(sum(parameter.numel() for parameter in model.parameters())),
        "optimizer_updates": updates,
        "best_update": best_update,
        "best_validation": best_metrics,
        "initial_validation": initial_validation,
        "train_seconds": train_seconds,
        "samples_per_second": effective_batch * updates / train_seconds,
        "actual_microbatch": requested_microbatch,
        "effective_batch_size": effective_batch,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "history_path": str(history_path),
        "best_checkpoint": str(out_dir / "best_inference.pt"),
    }
    return summary, model


def _environment(root: Path, device: torch.device) -> dict[str, Any]:
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        tracked_changes = bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        untracked_present = bool(
            subprocess.run(
                ["git", "-C", str(root), "ls-files", "--others", "--exclude-standard"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_sha, tracked_changes, untracked_present = "unknown", True, True
    props = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        driver = "unavailable"
    return {
        "git_sha": git_sha,
        "tracked_worktree_changes": tracked_changes,
        "untracked_worktree_files_present": untracked_present,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_driver": driver,
        "cuda_total_memory_bytes": int(props.total_memory),
        "cpu_count": __import__("os").cpu_count(),
        "training_device": str(device),
        "precision": "float32",
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _write_analysis(
    path: Path,
    summaries: list[dict[str, Any]],
    phase: str,
    teacher_validation: dict[str, Any],
    physical_consistency: dict[str, Any],
) -> None:
    lines = [
        f"# D1-E001 {phase} 训练记录",
        "",
        "本文件只报告运行记录，不代替独立盲测阶段审查。参数估计与检测指标来自独立验证数据。",
        "",
        "| 方法 | seed | 更新数 | 最佳验证更新 | 验证 AUC | range RMSE (cell) | Doppler RMSE (Hz) | 训练秒 | 峰值显存 (MiB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        metrics = item["best_validation"]
        lines.append(
            f"| {item['method']} | {item['seed_id']} | {item['optimizer_updates']} | {item['best_update']} | "
            f"{metrics['presence_auc']:.4f} | {metrics['range_rmse_cells']:.4f} | "
            f"{metrics['doppler_rmse_hz']:.3f} | {item['train_seconds']:.2f} | "
            f"{item['cuda_peak_allocated_bytes'] / (1024**2):.1f} |"
        )
    lines.extend(
        [
            "",
            "## 学生逐 SNR 验证指标",
            "",
            "| 方法 | SNR (dB) | AUC | range RMSE (cell) | Doppler RMSE (Hz) | H0 | H1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in summaries:
        for snr, metrics in sorted(
            item["best_validation"]["by_snr"].items(), key=lambda pair: float(pair[0])
        ):
            lines.append(
                f"| {item['method']} | {snr} | {metrics['presence_auc']:.4f} | "
                f"{metrics['range_rmse_cells']:.4f} | {metrics['doppler_rmse_hz']:.3f} | "
                f"{metrics['n_h0']} | {metrics['n_h1']} |"
            )
    lines.extend(
        [
            "",
            "## APX15 教师验证集基线",
            "",
            "| SNR (dB) | AUC | range RMSE (cell) | Doppler RMSE (Hz) | H0 | H1 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for snr, metrics in teacher_validation["by_snr"].items():
        lines.append(
            f"| {snr} | {metrics['presence_auc']:.4f} | {metrics['range_rmse_cells']:.4f} | "
            f"{metrics['doppler_rmse_hz']:.3f} | {metrics['n_h0']} | {metrics['n_h1']} |"
        )
    lines.extend(["", "## 物理一致性验证集诊断", "", f"样本数：{physical_consistency['n_validation_samples']}", ""])
    for angle, metrics in physical_consistency["common_phase_rotation"].items():
        lines.append(
            f"- 公共相位旋转 {angle} rad：Doppler delta mean-abs/p95/max = "
            f"{metrics['doppler_hz_delta']['mean_abs']:.4f}/"
            f"{metrics['doppler_hz_delta']['p95_abs']:.4f}/"
            f"{metrics['doppler_hz_delta']['max_abs']:.4f} Hz。"
        )
    for amount, metrics in physical_consistency["frequency_shift_equivariance"].items():
        lines.append(
            f"- 输入频移 {amount} Hz：等变残差 mean-abs/p95/max = "
            f"{metrics['doppler_equivariance_residual_hz']['mean_abs']:.4f}/"
            f"{metrics['doppler_equivariance_residual_hz']['p95_abs']:.4f}/"
            f"{metrics['doppler_equivariance_residual_hz']['max_abs']:.4f} Hz。"
        )
    lines.extend(
        [
            "",
            "### H1 参数输出分 SNR 变化",
            "",
            "| 变换 | SNR (dB) | H1 样本 | range delta p95 (cell) | Doppler delta/residual p95 (Hz) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for angle, metrics in physical_consistency["common_phase_rotation"].items():
        for snr, doppler in metrics["doppler_hz_delta"]["by_snr_db"].items():
            range_summary = metrics["range_cell_delta"]["by_snr_db"][snr]
            lines.append(
                f"| 公共相位 {angle} rad | {snr} | {doppler['n']} | "
                f"{range_summary['p95_abs']:.4f} | {doppler['p95_abs']:.4f} |"
            )
    for amount, metrics in physical_consistency["frequency_shift_equivariance"].items():
        for snr, doppler in metrics["doppler_equivariance_residual_hz"]["by_snr_db"].items():
            range_summary = metrics["range_cell_delta"]["by_snr_db"][snr]
            lines.append(
                f"| 输入频移 {amount} Hz | {snr} | {doppler['n']} | "
                f"{range_summary['p95_abs']:.4f} | {doppler['p95_abs']:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_phase(
    *,
    phase: str,
    config: dict[str, Any],
    root: Path,
    experiment_output: Path,
    device: torch.device,
    seed_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if phase == "pilot":
        split = config["splits"]["pilot"]
        train_tag, validation_tag = "pilot_train", "pilot_validation"
        seed_id = "pilot"
        train_seed = split["train_seed"]
        validation_seed = split["validation_seed"]
        init_seed = split["initialization_seed"]
        phase_output = experiment_output / "pilot"
    elif phase == "formal":
        if seed_record is None:
            raise ValueError("formal D1 training requires an explicit training seed record")
        split = config["splits"]["formal"]
        seed_id = str(seed_record["id"])
        train_seed = int(seed_record["train"])
        validation_seed = int(seed_record["validation"])
        init_seed = int(seed_record["initialization"])
        train_tag = f"{seed_id}_train"
        validation_tag = f"{seed_id}_validation"
        phase_output = experiment_output / "formal" / seed_id
    else:
        raise ValueError(f"unknown D1 phase: {phase}")

    if phase_output.exists() and any(phase_output.iterdir()):
        raise FileExistsError(f"{phase_output} already has artifacts; use a new --output directory")
    phase_output.mkdir(parents=True, exist_ok=True)
    d0_config = _load_d0_config(root)
    teacher = APX15DopplerTeacher(d0_config)
    cache_root = root / config["data"]["cache"]["directory"]
    train_arrays = _write_split_cache(
        cache_root=cache_root,
        tag=train_tag,
        base_seed=int(train_seed),
        examples_per_class=int(split["train_examples_per_class"]),
        snr_spec=tuple(config["data"]["training_snr_db_uniform"]),
        validation_snr=None,
        validation_seed_stride=int(config["data"]["training_chunk_seed_stride_by_index"]),
        d0_config=d0_config,
        teacher=teacher,
    )
    validation_arrays = _write_split_cache(
        cache_root=cache_root,
        tag=validation_tag,
        base_seed=int(validation_seed),
        examples_per_class=int(split["validation_examples_per_class_per_snr"]),
        snr_spec=config["data"]["evaluation_snr_db"],
        validation_snr=config["data"]["evaluation_snr_db"],
        validation_seed_stride=int(split["validation_seed_stride_by_snr_index"]),
        d0_config=d0_config,
        teacher=teacher,
    )

    seed_model = int(init_seed)
    torch.manual_seed(seed_model)
    torch.cuda.manual_seed_all(seed_model)
    initial_model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
    )
    initial_state = _clone_state_to_cpu(initial_model)
    del initial_model
    methods = config["training"]["methods"]
    counts = strategy_update_counts(methods)
    if set(counts.values()) != {int(config["training"]["total_optimizer_updates_per_strategy"])}:
        raise ValueError(f"A/B/C/D schedules do not share the frozen update budget: {counts}")
    order_seed = seed_model + 77
    summaries: list[dict[str, Any]] = []
    physical_consistency: dict[str, Any] | None = None
    torch.cuda.reset_peak_memory_stats(device)
    for name, method in methods.items():
        torch.cuda.empty_cache()
        strategy_dir = phase_output / name
        if strategy_dir.exists() and any(strategy_dir.iterdir()):
            raise FileExistsError(f"strategy output already exists: {strategy_dir}")
        allowed_microbatches = [
            int(config["training"]["initial_microbatch_size"]),
            *[int(value) for value in config["training"]["oom_microbatch_fallbacks"]],
        ]
        attempts: list[dict[str, Any]] = []
        for microbatch in allowed_microbatches:
            torch.cuda.empty_cache()
            try:
                summary, trained_model = _train_strategy(
                    model_name=name,
                    method=method,
                    initial_state=initial_state,
                    train_arrays=train_arrays,
                    validation_arrays=validation_arrays,
                    config=config,
                    device=device,
                    out_dir=strategy_dir,
                    seed_id=seed_id,
                    order_seed=order_seed,
                    requested_microbatch=microbatch,
                )
                summary["oom_fallback_attempts"] = attempts
                break
            except torch.cuda.OutOfMemoryError as exc:
                attempts.append({"microbatch": microbatch, "error": str(exc).splitlines()[0]})
                for generated in ("best_inference.pt", "last_inference.pt", "training_history.csv"):
                    candidate = strategy_dir / generated
                    if candidate.exists():
                        candidate.unlink()
                if microbatch == allowed_microbatches[-1]:
                    _write_json(strategy_dir / "oom_attempts.json", attempts)
                    raise
                torch.cuda.empty_cache()
        else:  # pragma: no cover - loop exits through success or final OOM
            raise RuntimeError("no D1 microbatch size could be run")
        summaries.append(summary)
        if name == "A_truth_only":
            physical_consistency = _physical_consistency_validation(
                trained_model, validation_arrays, config, device
            )
        del trained_model
        torch.cuda.empty_cache()
        _write_json(phase_output / "metrics_partial.json", summaries)
    if physical_consistency is None:
        raise RuntimeError("D1 pilot did not produce the required physical-consistency diagnostics")
    teacher_validation = _teacher_metrics(validation_arrays)
    metrics = {
        "experiment_id": "D1-E001",
        "phase": phase,
        "status": "COMPLETED",
        "seed_id": seed_id,
        "pilot_or_formal_seed": {
            "train": int(train_seed),
            "validation": int(validation_seed),
            "initialization": int(init_seed),
        },
        "training_environment": _environment(root, device),
        "strategy_summaries": summaries,
        "teacher_validation_baseline": teacher_validation,
        "physical_consistency_validation": physical_consistency,
    }
    if phase == "pilot":
        metrics["pilot_gate"] = _pilot_gate(summaries, config)
    _write_json(phase_output / "metrics.json", metrics)
    _write_analysis(
        phase_output / "analysis.md",
        summaries,
        phase,
        teacher_validation,
        physical_consistency,
    )
    _write_json(phase_output / "environment.json", metrics["training_environment"])
    (phase_output / "metrics_partial.json").unlink(missing_ok=True)
    teacher.close()
    return metrics


def _pilot_gate(summaries: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_name = {item["method"]: item for item in summaries}
    direct = by_name.get("A_truth_only")
    if direct is None:
        return {
            "passed": False,
            "technical_feasibility_passed": False,
            "learning_screen_passed": False,
            "reason": "A strategy missing",
        }
    initial = direct["initial_validation"]
    final = direct["best_validation"]
    finite = all(
        math.isfinite(float(value))
        for item in summaries
        for metrics in (item["initial_validation"], item["best_validation"])
        for key, value in metrics.items()
        if key != "by_snr"
    )
    expected_updates = int(config["training"]["total_optimizer_updates_per_strategy"])
    expected_methods = set(config["training"]["methods"])
    complete = set(by_name) == expected_methods and all(
        item["optimizer_updates"] == expected_updates for item in summaries
    )
    technical_passed = bool(finite and complete)
    screen = config["gates"]["G1_truth_supervised_task_learning"]
    screen_snr = float(screen["snr_db"])
    snr_key = str(int(screen_snr)) if screen_snr.is_integer() else str(screen_snr)
    high_snr = final.get("by_snr", {}).get(snr_key, {})
    task_loss_decreased = final["task_loss"] < initial["task_loss"]
    screen_metrics_finite = all(
        key in high_snr and math.isfinite(float(high_snr[key]))
        for key in ("presence_auc", "range_rmse_cells", "doppler_rmse_hz")
    )
    learning_passed = bool(
        task_loss_decreased
        and screen_metrics_finite
        and high_snr.get("presence_auc", -math.inf) >= float(screen["minimum_detection_auc"])
        and high_snr.get("range_rmse_cells", math.inf)
        <= float(screen["maximum_range_rmse_cells_all_h1"])
        and high_snr.get("doppler_rmse_hz", math.inf)
        <= float(screen["maximum_doppler_rmse_hz_all_h1"])
    )
    return {
        "passed": bool(technical_passed and learning_passed),
        "technical_feasibility_passed": technical_passed,
        "learning_screen_passed": learning_passed,
        "criteria": (
            "pilot screen only: all four strategies complete the frozen update budget with finite "
            "validation metrics, and A improves validation task loss while meeting the frozen "
            "20 dB AUC/range/Doppler numeric references; this is not the formal G1 decision"
        ),
        "learning_diagnostic_not_a_gate": {
            "A_initial_validation_task_loss": initial["task_loss"],
            "A_best_validation_task_loss": final["task_loss"],
            "A_validation_at_screen_snr": high_snr,
            "screen_snr_db": screen_snr,
            "task_loss_decreased": task_loss_decreased,
        },
        "direct_validation_initial": initial,
        "direct_validation_best": final,
    }


def run_d1_e001(
    config: dict[str, Any],
    output: str | Path,
    *,
    root: str | Path,
    phase: str = "pilot",
    device: str = "cuda",
) -> dict[str, Any]:
    """Run the pilot or all three formal training seeds, sequentially on CUDA."""
    cuda_device = require_cuda(device)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if phase not in {"pilot", "formal"}:
        raise ValueError("phase must be 'pilot' or 'formal'")
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    if phase == "formal":
        pilot_metrics = output_path / "pilot" / "metrics.json"
        if not pilot_metrics.is_file():
            raise FileNotFoundError("D1 formal training requires a completed pilot at output/pilot")
        pilot = json.loads(pilot_metrics.read_text(encoding="utf-8"))
        pilot_summaries = pilot.get("strategy_summaries", [])
        recomputed_gate = _pilot_gate(pilot_summaries, config)
        if not recomputed_gate["passed"]:
            raise RuntimeError("D1 recomputed pilot gate failed; formal training is not started")
        seed_records = _formal_seed_records(config)
        formal_output = output_path / "formal"
        if (formal_output / "metrics.json").exists():
            raise FileExistsError(f"{formal_output} already contains completed formal results")
        result: dict[str, Any] = {
            "experiment_id": "D1-E001",
            "phase": "formal",
            "status": "RUNNING",
            "completed_seed_ids": [],
            "seed_summaries": [],
        }
        formal_output.mkdir(parents=True, exist_ok=True)
        for seed_record in seed_records:
            seed_metrics = _run_phase(
                phase="formal",
                config=config,
                root=root_path,
                experiment_output=output_path,
                device=cuda_device,
                seed_record=seed_record,
            )
            result["completed_seed_ids"].append(str(seed_record["id"]))
            result["seed_summaries"].append(seed_metrics)
            _write_json(formal_output / "metrics_partial.json", result)
            torch.cuda.empty_cache()
        result["status"] = "COMPLETED"
        _write_json(formal_output / "metrics.json", result)
        (formal_output / "metrics_partial.json").unlink(missing_ok=True)
        return result
    return _run_phase(
        phase="pilot",
        config=config,
        root=root_path,
        experiment_output=output_path,
        device=cuda_device,
    )


def _formal_seed_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    records = config["splits"]["formal"].get("training_seeds", [])
    if len(records) != 3:
        raise ValueError("D1 formal protocol requires exactly three training seed records")
    identifiers = [str(record["id"]) for record in records]
    seed_values = [
        int(record[key])
        for record in records
        for key in ("train", "validation", "initialization")
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("formal seed identifiers must be unique")
    if any(value < 0 for value in seed_values) or len(set(seed_values)) != len(seed_values):
        raise ValueError("formal training, validation, and initialization seeds must be unique")
    return [dict(record) for record in records]
