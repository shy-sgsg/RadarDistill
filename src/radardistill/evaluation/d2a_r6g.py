"""Independent evaluation for the matched D2-A-R6-G teaching pilot."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from radardistill.distillation.task_losses import decode_task_outputs
from radardistill.evaluation.d2a_r5c_teaching_pilot import predict_k3_teacher_localization
from radardistill.evaluation.d2a_r6a import (
    calibrate_threshold,
    summarize_parameter_estimator,
    wrapped_error,
)
from radardistill.evaluation.d2a_r6e import exact_two_sided, h0_metrics, h1_metrics
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_architectures import R6ECEStudent
from radardistill.students.r6g_student import R6GCEStudent, export_r6g_inference_model
from radardistill.training import d2a_r6g as training


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-G"
RUNS = training.RUNS
ARMS = ("T0", "T1", "T3")
VALIDATION_POINTS = (0, 12_800, 25_600, 51_200)


def teacher_correctness_mask(
    *,
    predicted_range_bin: Any,
    predicted_doppler_hz: Any,
    truth_range_cell: Any,
    truth_doppler_hz: Any,
    prf_hz: float,
    range_gate_cells: float = 1.0,
    doppler_gate_hz: float = 20.0,
) -> np.ndarray:
    """Return the offline K3 peak correctness label used by the frozen protocol."""
    arrays = [
        np.asarray(predicted_range_bin, dtype=np.float64),
        np.asarray(predicted_doppler_hz, dtype=np.float64),
        np.asarray(truth_range_cell, dtype=np.float64),
        np.asarray(truth_doppler_hz, dtype=np.float64),
    ]
    if any(value.ndim != 1 or not np.isfinite(value).all() for value in arrays):
        raise ValueError("R6-G teacher and truth values must be finite vectors")
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("R6-G teacher and truth vectors must align")
    if not np.isfinite(range_gate_cells) or range_gate_cells < 0.0:
        raise ValueError("range_gate_cells must be finite and nonnegative")
    if not np.isfinite(doppler_gate_hz) or doppler_gate_hz < 0.0:
        raise ValueError("doppler_gate_hz must be finite and nonnegative")
    predicted_range, predicted_doppler, truth_range, truth_doppler = arrays
    doppler_error = np.asarray(wrapped_error(predicted_doppler, truth_doppler, prf_hz))
    return (np.abs(predicted_range - truth_range) <= range_gate_cells) & (
        np.abs(doppler_error) <= doppler_gate_hz
    )


def _finite_vector(
    predictions: Mapping[str, np.ndarray], key: str, *, dtype: Any = np.float64
) -> np.ndarray:
    if key not in predictions:
        raise ValueError(f"R6-G prediction is missing {key}")
    values = np.asarray(predictions[key], dtype=dtype)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"R6-G prediction {key} must be a finite vector")
    return values


def _stratified_sample(
    rng: np.random.Generator,
    condition_index: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    sampled: list[np.ndarray] = []
    for condition in np.unique(condition_index[selected]):
        members = np.flatnonzero(selected & (condition_index == condition))
        sampled.append(rng.choice(members, size=len(members), replace=True))
    if not sampled:
        raise ValueError("R6-G bootstrap selection is empty")
    return np.concatenate(sampled)


def _quantile_interval(samples: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [float(value) for value in np.quantile(samples, [tail, 1.0 - tail])]


def paired_r6g_contrast(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    threshold_baseline: float,
    threshold_candidate: float,
    conditions: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
    prf_hz: float,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Compute the paired, condition-stratified R6-G task contrast."""
    baseline_ids = np.asarray(baseline.get("scene_id", [])).astype(str)
    candidate_ids = np.asarray(candidate.get("scene_id", [])).astype(str)
    if baseline_ids.ndim != 1 or not len(baseline_ids) or not np.array_equal(baseline_ids, candidate_ids):
        raise ValueError("R6-G contrasts require paired scene IDs in identical order")
    first_index = _finite_vector(baseline, "condition_index", dtype=np.int64)
    second_index = _finite_vector(candidate, "condition_index", dtype=np.int64)
    if first_index.shape != baseline_ids.shape or not np.array_equal(first_index, second_index):
        raise ValueError("R6-G contrasts require paired condition assignments")
    if np.any(first_index < 0) or np.any(first_index >= len(conditions)):
        raise ValueError("R6-G condition index is outside the frozen condition table")
    if isinstance(replicates, bool) or replicates <= 0 or seed < 0:
        raise ValueError("invalid R6-G bootstrap configuration")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if not np.isfinite(threshold_baseline) or not np.isfinite(threshold_candidate):
        raise ValueError("R6-G thresholds must be finite")

    first_score = _finite_vector(baseline, "score")
    second_score = _finite_vector(candidate, "score")
    first_range = _finite_vector(baseline, "predicted_range_cell")
    second_range = _finite_vector(candidate, "predicted_range_cell")
    truth_range = _finite_vector(baseline, "truth_range_cell")
    first_doppler = _finite_vector(baseline, "predicted_doppler_hz")
    second_doppler = _finite_vector(candidate, "predicted_doppler_hz")
    truth_doppler = _finite_vector(baseline, "truth_doppler_hz")
    if any(
        value.shape != baseline_ids.shape
        for value in (
            first_score, second_score, first_range, second_range, truth_range,
            first_doppler, second_doppler, truth_doppler,
        )
    ):
        raise ValueError("R6-G paired predictions and truth must align with scene IDs")
    if not np.array_equal(truth_range, _finite_vector(candidate, "truth_range_cell")) or not np.array_equal(
        truth_doppler, _finite_vector(candidate, "truth_doppler_hz")
    ):
        raise ValueError("R6-G paired arms must use identical physical truth")

    courses = np.asarray([str(item["course"]) for item in conditions])
    primary = np.isin(courses[first_index], ("C2", "C3"))
    if not primary.any():
        raise ValueError("R6-G primary C2+C3 contrast is empty")
    first_detected = first_score >= float(threshold_baseline)
    second_detected = second_score >= float(threshold_candidate)
    first_range_error = first_range - truth_range
    second_range_error = second_range - truth_range
    first_doppler_error = np.asarray(wrapped_error(first_doppler, truth_doppler, prf_hz), dtype=np.float64)
    second_doppler_error = np.asarray(wrapped_error(second_doppler, truth_doppler, prf_hz), dtype=np.float64)
    first_joint = first_detected & (np.abs(first_range_error) <= 1.0) & (np.abs(first_doppler_error) <= 20.0)
    second_joint = second_detected & (np.abs(second_range_error) <= 1.0) & (np.abs(second_doppler_error) <= 20.0)
    joint_delta = second_joint.astype(np.float64) - first_joint.astype(np.float64)
    pd_delta = second_detected.astype(np.float64) - first_detected.astype(np.float64)

    rng = np.random.default_rng(seed)
    joint_samples = np.empty(replicates, dtype=np.float64)
    pd_samples = np.empty(replicates, dtype=np.float64)
    range_ratio_samples = np.empty(replicates, dtype=np.float64)
    doppler_ratio_samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        primary_indices = _stratified_sample(rng, first_index, primary)
        all_indices = _stratified_sample(rng, first_index, np.ones(len(first_index), dtype=bool))
        joint_samples[replicate] = float(np.mean(joint_delta[primary_indices]))
        pd_samples[replicate] = float(np.mean(pd_delta[all_indices]))
        range_ratio_samples[replicate] = math.sqrt(
            float(np.mean(second_range_error[all_indices] ** 2))
            / max(float(np.mean(first_range_error[all_indices] ** 2)), 1e-24)
        )
        doppler_ratio_samples[replicate] = math.sqrt(
            float(np.mean(second_doppler_error[all_indices] ** 2))
            / max(float(np.mean(first_doppler_error[all_indices] ** 2)), 1e-24)
        )

    joint_difference = float(np.mean(joint_delta[primary]))
    pd_difference = float(np.mean(pd_delta))
    range_ratio = math.sqrt(float(np.mean(second_range_error**2)) / max(float(np.mean(first_range_error**2)), 1e-24))
    doppler_ratio = math.sqrt(float(np.mean(second_doppler_error**2)) / max(float(np.mean(first_doppler_error**2)), 1e-24))
    joint_ci = _quantile_interval(joint_samples, confidence)
    pd_ci = _quantile_interval(pd_samples, confidence)
    range_ci = _quantile_interval(range_ratio_samples, confidence)
    doppler_ci = _quantile_interval(doppler_ratio_samples, confidence)
    return {
        "n_paired_scenes": int(len(baseline_ids)),
        "primary_scope": "identifiable_C2_C3",
        "n_primary_scenes": int(primary.sum()),
        "joint_success_difference": joint_difference,
        "joint_success_ci95": joint_ci,
        "confidence_level": confidence,
        "bootstrap_method": "paired_scene_condition_stratified_percentile",
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "minimum_primary_gain": 0.02,
        "primary_gain_pass": bool(joint_difference >= 0.02 and joint_ci[0] > 0.0),
        "task_protection": {
            "pd_difference": pd_difference,
            "pd_difference_ci95": pd_ci,
            "pd_lower_minimum": -0.02,
            "pd_pass": bool(pd_ci[0] >= -0.02),
            "range_rmse_ratio": range_ratio,
            "range_rmse_ratio_ci95": range_ci,
            "range_rmse_ratio_upper_maximum": 1.05,
            "range_pass": bool(range_ci[1] <= 1.05),
            "doppler_rmse_ratio": doppler_ratio,
            "doppler_rmse_ratio_ci95": doppler_ci,
            "doppler_rmse_ratio_upper_maximum": 1.05,
            "doppler_pass": bool(doppler_ci[1] <= 1.05),
        },
    }


def teacher_correctness_strata(
    predictions: Mapping[str, np.ndarray],
    *,
    threshold: float,
    conditions: Sequence[Mapping[str, Any]],
    prf_hz: float,
) -> dict[str, Any]:
    """Summarize each student on the same H1 scenes where K3 is right or wrong."""
    count = len(np.asarray(predictions.get("scene_id", [])))
    index = _finite_vector(predictions, "condition_index", dtype=np.int64)
    teacher_correct = np.asarray(predictions.get("teacher_correct"), dtype=bool)
    if index.shape != (count,) or teacher_correct.shape != (count,):
        raise ValueError("R6-G teacher strata require aligned condition and correctness vectors")
    if np.any(index < 0) or np.any(index >= len(conditions)):
        raise ValueError("R6-G teacher-strata condition index is outside the frozen table")
    course = np.asarray([str(item["course"]) for item in conditions])
    scenario = np.asarray([str(item["scenario_id"]) for item in conditions])
    snr = np.asarray([float(item["snr_db"]) for item in conditions])
    cnr = np.asarray([float(item["cnr_db"]) for item in conditions])
    groups = {
        "all_H1": np.ones(count, dtype=bool),
        "C3": course[index] == "C3",
        "C3_partial_overlap_snr0_cnr20": (
            (course[index] == "C3") & (scenario[index] == "partial_overlap")
            & (snr[index] == 0.0) & (cnr[index] == 20.0)
        ),
    }
    correct_rates: dict[str, float | None] = {}
    correct_rates_by_condition: dict[str, float | None] = {}
    strata: dict[str, dict[str, Any]] = {"teacher_correct": {}, "teacher_incorrect": {}}
    score = _finite_vector(predictions, "score")
    predicted_range = _finite_vector(predictions, "predicted_range_cell")
    truth_range = _finite_vector(predictions, "truth_range_cell")
    predicted_doppler = _finite_vector(predictions, "predicted_doppler_hz")
    truth_doppler = _finite_vector(predictions, "truth_doppler_hz")
    if any(value.shape != (count,) for value in (score, predicted_range, truth_range, predicted_doppler, truth_doppler)):
        raise ValueError("R6-G teacher-strata task vectors must align with scene IDs")
    detected = score >= float(threshold)
    for condition_number in range(len(conditions)):
        mask = index == condition_number
        correct_rates_by_condition[str(condition_number)] = (
            float(np.mean(teacher_correct[mask])) if mask.any() else None
        )
    for group_name, group_mask in groups.items():
        correct_rates[group_name] = (
            float(np.mean(teacher_correct[group_mask])) if group_mask.any() else None
        )
        for correct, label in ((True, "teacher_correct"), (False, "teacher_incorrect")):
            mask = group_mask & (teacher_correct == correct)
            n = int(mask.sum())
            if not n:
                strata[label][group_name] = {
                    "n": 0,
                    "pd": None,
                    "joint_success": None,
                    "all_h1": None,
                    "detection_hit": None,
                }
                continue
            summary = summarize_parameter_estimator(
                predicted_range[mask], truth_range[mask],
                predicted_doppler[mask], truth_doppler[mask], detected[mask],
                prf_hz=prf_hz, range_gate_cells=1.0, doppler_gate_hz=20.0,
            )
            summary["n"] = n
            summary["pd"] = float(np.mean(detected[mask]))
            summary["pd_ci95_exact"] = exact_two_sided(int(detected[mask].sum()), n)
            summary["joint_success_ci95_exact"] = exact_two_sided(
                int(summary["joint_success_count"]), n
            )
            strata[label][group_name] = summary
    return {
        "k3_correct_rate": correct_rates,
        "k3_correct_rate_by_condition": correct_rates_by_condition,
        "student_by_teacher_correctness": strata,
    }


def predict_r6g_eval_split(
    model: torch.nn.Module,
    split: str,
    plan: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    *,
    device: torch.device,
    microbatch: int,
    prf_hz: float,
    compute_teacher_correctness: bool = False,
) -> dict[str, np.ndarray]:
    """Stream one independent R6-G split; K3 correctness is an offline H1 diagnostic."""
    if split not in {"h0_calibration", "h0_verification", "h1_development"}:
        raise ValueError("unknown R6-G evaluation split")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R6-G formal evaluation requires CUDA; CPU fallback is forbidden")
    if device.index not in (None, 0) or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3050 Laptop GPU":
        raise RuntimeError("R6-G formal evaluation requires the registered RTX 3050 cuda:0")
    if microbatch <= 0 or not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("R6-G evaluation microbatch and PRF must be positive")
    is_h1 = split == "h1_development"
    if compute_teacher_correctness and not is_h1:
        raise ValueError("R6-G K3 correctness is defined only for offline H1 evaluation")
    keys = ["scene_id", "condition_index", "score"]
    if is_h1:
        keys += [
            "predicted_range_cell", "predicted_doppler_hz",
            "truth_range_cell", "truth_doppler_hz",
        ]
        if compute_teacher_correctness:
            keys += [
                "teacher_predicted_range_bin", "teacher_predicted_doppler_hz",
                "teacher_range_abs_error_cells", "teacher_doppler_wrapped_error_hz",
                "teacher_correct",
            ]
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    model.eval()
    for spec in plan:
        condition = spec["condition"]
        generated = generate_d2a_paired_batch(
            int(spec["count"]), int(spec["seed"]),
            float(condition["snr_db"]), float(condition["cnr_db"]),
            dict(physics), scenario_id=str(condition["scenario_id"]),
        )
        iq = generated.iq_h1 if is_h1 else generated.iq_h0
        iq_ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
        split_values: dict[str, list[np.ndarray]] = {
            "score": [], "predicted_range_cell": [], "predicted_doppler_hz": [],
        }
        if compute_teacher_correctness:
            split_values.update({
                "teacher_predicted_range_bin": [], "teacher_predicted_doppler_hz": [],
                "teacher_range_abs_error_cells": [], "teacher_doppler_wrapped_error_hz": [],
                "teacher_correct": [],
            })
        with torch.inference_mode():
            for start in range(0, len(iq_ri), microbatch):
                stop = min(start + microbatch, len(iq_ri))
                output = model(torch.from_numpy(np.ascontiguousarray(iq_ri[start:stop])).to(device))
                decoded = decode_task_outputs(output, prf_hz)
                split_values["score"].append(decoded["presence_probability"].cpu().numpy())
                if is_h1:
                    split_values["predicted_range_cell"].append(decoded["range_cell"].cpu().numpy())
                    split_values["predicted_doppler_hz"].append(decoded["doppler_hz"].cpu().numpy())
                del output, decoded
                if compute_teacher_correctness:
                    teacher = predict_k3_teacher_localization(
                        iq[start:stop],
                        np.asarray(generated.truth_target_range_cell[start:stop]),
                        np.asarray(generated.truth_target_doppler_hz[start:stop]),
                        physics,
                        device,
                        prf_hz=prf_hz,
                    )
                    for key in split_values:
                        if key.startswith("teacher_"):
                            split_values[key].append(np.asarray(teacher[key]))
        count = len(generated.base_scene_id)
        values["scene_id"].append(np.asarray(generated.base_scene_id).astype("<U48"))
        values["condition_index"].append(np.full(count, int(spec["condition_index"]), dtype=np.uint8))
        values["score"].append(np.concatenate(split_values["score"]).astype(np.float32, copy=False))
        if is_h1:
            values["predicted_range_cell"].append(
                np.concatenate(split_values["predicted_range_cell"]).astype(np.float32, copy=False)
            )
            values["predicted_doppler_hz"].append(
                np.concatenate(split_values["predicted_doppler_hz"]).astype(np.float32, copy=False)
            )
            values["truth_range_cell"].append(np.asarray(generated.truth_target_range_cell, dtype=np.float32))
            values["truth_doppler_hz"].append(np.asarray(generated.truth_target_doppler_hz, dtype=np.float32))
            if compute_teacher_correctness:
                for key in split_values:
                    if key.startswith("teacher_"):
                        values[key].append(np.concatenate(split_values[key]))
    result = {key: np.concatenate(parts) for key, parts in values.items()}
    if len(set(result["scene_id"].astype(str))) != len(result["scene_id"]):
        raise ValueError(f"R6-G {split} generated duplicate scene IDs")
    for key, array in result.items():
        if key not in {"scene_id", "condition_index", "teacher_correct"} and not np.isfinite(array).all():
            raise FloatingPointError(f"R6-G {split} produced nonfinite {key}")
    return result


def predict_r6g_validation_cache(
    model: torch.nn.Module,
    validation: Mapping[str, np.ndarray],
    *,
    device: torch.device,
    forward_batch_cpi: int,
    prf_hz: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Predict paired cached validation IQ for the preregistered learning curve."""
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R6-G validation-curve evaluation requires CUDA")
    if forward_batch_cpi <= 0 or forward_batch_cpi % 2:
        raise ValueError("validation forward batch must be a positive even CPI count")
    count = len(validation["scene_id"])
    paired_per_batch = forward_batch_cpi // 2
    h0_parts: dict[str, list[np.ndarray]] = {key: [] for key in ("scene_id", "condition_index", "score")}
    h1_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "scene_id", "condition_index", "score", "predicted_range_cell",
            "predicted_doppler_hz", "truth_range_cell", "truth_doppler_hz",
        )
    }
    model.eval()
    with torch.inference_mode():
        for base_start in range(0, count, paired_per_batch):
            indices = np.arange(base_start, min(base_start + paired_per_batch, count), dtype=np.int64)
            batch = training._r6g_training_batch(validation, indices, indices)
            output = model(torch.from_numpy(batch["iq_ri"]).to(device))
            decoded = decode_task_outputs(output, prf_hz)
            score = decoded["presence_probability"].cpu().numpy()
            predicted_range = decoded["range_cell"].cpu().numpy()
            predicted_doppler = decoded["doppler_hz"].cpu().numpy()
            scene_ids = np.repeat(np.asarray(validation["scene_id"])[indices].astype("<U48"), 2)
            condition_index = np.repeat(
                np.asarray(validation["condition_index"], dtype=np.uint8)[indices], 2
            )
            h0_parts["scene_id"].append(scene_ids[0::2])
            h0_parts["condition_index"].append(condition_index[0::2])
            h0_parts["score"].append(score[0::2].astype(np.float32, copy=False))
            h1_parts["scene_id"].append(scene_ids[1::2])
            h1_parts["condition_index"].append(condition_index[1::2])
            h1_parts["score"].append(score[1::2].astype(np.float32, copy=False))
            h1_parts["predicted_range_cell"].append(predicted_range[1::2].astype(np.float32, copy=False))
            h1_parts["predicted_doppler_hz"].append(predicted_doppler[1::2].astype(np.float32, copy=False))
            h1_parts["truth_range_cell"].append(
                np.asarray(validation["truth_range_cell"], dtype=np.float32)[indices]
            )
            h1_parts["truth_doppler_hz"].append(
                np.asarray(validation["truth_doppler_hz"], dtype=np.float32)[indices]
            )
            del output, decoded
    return (
        {key: np.concatenate(parts) for key, parts in h0_parts.items()},
        {key: np.concatenate(parts) for key, parts in h1_parts.items()},
    )


def summarize_validation_history(path: Path) -> dict[str, Any]:
    """Summarize the recorded truth-task validation loss curve and fixed points."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("R6-G training history is empty")
    updates = np.asarray([int(row["update"]) for row in rows], dtype=np.int64)
    losses = np.asarray([
        float(row.get("validation_total") or row.get("validation_truth_total") or "nan")
        for row in rows
    ], dtype=np.float64)
    if np.any(np.diff(updates) <= 0) or updates[0] != 0 or not np.isfinite(losses).all():
        raise ValueError("R6-G validation history must contain finite, strictly increasing losses from update 0")
    if updates[-1] != 51_200:
        raise ValueError("R6-G validation history does not reach the frozen 51,200-update endpoint")
    by_update = {int(row["update"]): row for row in rows}
    points: dict[str, Any] = {}
    for update in VALIDATION_POINTS:
        if update not in by_update:
            raise ValueError(f"R6-G validation history is missing update {update}")
        row = by_update[update]
        points[str(update)] = {
            key: float(value)
            for key, value in row.items()
            if key.startswith("validation_") and value not in (None, "")
        }
    area = float(np.trapezoid(losses, updates) / (updates[-1] - updates[0]))
    return {
        "updates": updates.tolist(),
        "validation_truth_loss_points": points,
        "normalized_validation_truth_loss_area": area,
        "area_definition": "trapezoidal integral over updates divided by total update span",
    }


def capture_validation_checkpoint(
    source: Path,
    *,
    arm: str,
    target_updates: Sequence[int],
) -> int | None:
    """Copy a training checkpoint only when it matches a preregistered curve update."""
    if arm not in ARMS:
        raise ValueError(f"unknown R6-G arm: {arm}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    update = int(checkpoint.get("update", -1))
    if update not in set(map(int, target_updates)):
        return None
    owner = checkpoint.get("owner", {})
    if owner.get("owner") != "D2-A-R6-G-training-run" or owner.get("arm") != arm:
        raise RuntimeError(f"R6-G {arm} curve checkpoint owner mismatch")
    run_dir = Path(source).parent
    destination_dir = run_dir / "validation_checkpoints"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"update_{update:05d}.pt"
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if int(existing.get("update", -1)) != update or existing.get("owner") != owner:
            raise RuntimeError(f"R6-G {arm} curve checkpoint already exists with a different owner")
        return update
    temporary = destination.with_suffix(".pt.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)
    return update


def ensure_initial_validation_checkpoint(arm: str) -> bool:
    """Save the shared, pre-update initialization as the registered update-0 snapshot."""
    if arm not in ARMS:
        raise ValueError(f"unknown R6-G arm: {arm}")
    run_dir = RUNS / arm
    destination = run_dir / "validation_checkpoints" / "update_00000.pt"
    if destination.is_file():
        return True
    owner_path = run_dir / "owner.json"
    initial_path = RUNS / "shared_initial_state.pt"
    if not owner_path.is_file() or not initial_path.is_file():
        return False
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    shared = torch.load(initial_path, map_location="cpu", weights_only=False)
    init_owner = shared.get("owner", {})
    if (
        owner.get("owner") != "D2-A-R6-G-training-run"
        or owner.get("arm") != arm
        or owner.get("git_sha") != init_owner.get("git_sha")
        or owner.get("config_sha256") != init_owner.get("config_sha256")
        or owner.get("initialization_seed") != init_owner.get("seed")
    ):
        raise RuntimeError(f"R6-G {arm} shared update-0 initialization owner mismatch")
    state = shared.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("R6-G shared initialization is missing its state_dict")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save({"owner": owner, "update": 0, "model_state_dict": state}, temporary)
    temporary.replace(destination)
    return True


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_predictions(path: Path, values: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def _load_predictions(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _summarize_validation_checkpoint(
    checkpoint_path: Path,
    arm: str,
    update: int,
    validation: Mapping[str, np.ndarray],
    conditions: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    owner = checkpoint.get("owner", {})
    if owner.get("arm") != arm or int(checkpoint.get("update", -1)) != update:
        raise RuntimeError(f"R6-G {arm} validation checkpoint owner/update mismatch at {update}")
    training_model = R6GCEStudent(width=int(config["student"]["width"])).to(device)
    training_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = export_r6g_inference_model(training_model).to(device).eval()
    h0, h1 = predict_r6g_validation_cache(
        model,
        validation,
        device=device,
        forward_batch_cpi=int(config["evaluation"]["validation_cpi_forward_batch_size"]),
        prf_hz=float(config["radar"]["prf_hz"]),
    )
    threshold_info = calibrate_threshold(
        h0["score"], float(config["evaluation"]["target_pfa_cpi"])
    )
    summary = h1_metrics(
        h1,
        threshold=float(threshold_info["threshold"]),
        conditions=conditions,
        h0_verification=h0,
        prf_hz=float(config["radar"]["prf_hz"]),
    )
    result = {
        "checkpoint_update": update,
        "threshold_source": "paired_validation_H0_only_for_curve_diagnostic",
        "threshold_calibration": threshold_info,
        "pd": summary["pd"],
        "joint_success_all_H1": summary["by_group"]["all"]["joint_success"],
        "range_rmse_all_H1_cells": summary["by_group"]["all"]["all_h1"]["range_rmse_cells"],
        "doppler_rmse_all_H1_hz": summary["by_group"]["all"]["all_h1"]["doppler_rmse_hz"],
        "C2_C3": summary["by_group"]["C2_C3"],
        "C3": summary["by_group"]["C3"],
        "validation_h0_pfa_at_reference_threshold": float(np.mean(h0["score"] >= float(threshold_info["threshold"]))),
    }
    del checkpoint, training_model, model, h0, h1
    torch.cuda.empty_cache()
    return result


def evaluate_r6g() -> dict[str, Any]:
    """Run the frozen independent H0 calibration/verification and H1 development evaluation."""
    g0, _ = training._require_g0()
    training._require_matching_g1(g0)
    config, r6a, physics = training._configs()
    conditions = r6a["radar"]["conditions"]
    plans = training.build_r6g_plans(config, r6a)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != config["resources"]["required_gpu_name"]:
        raise RuntimeError("R6-G evaluation requires the registered CUDA device")
    microbatch = int(training._require_matching_g1(g0)["formal_microbatch"])
    started = time.perf_counter()
    arms: dict[str, Any] = {}
    thresholds: dict[str, float] = {}

    for arm in ARMS:
        run_dir = RUNS / arm
        record_path = run_dir / "training_record.json"
        if not record_path.is_file():
            raise RuntimeError(f"R6-G {arm} lacks a completed training record")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or int(record.get("updates", -1)) != int(config["training"]["fixed_updates_per_arm"]):
            raise RuntimeError(f"R6-G {arm} did not complete the frozen optimizer budget")
        checkpoint_path = ROOT / record["inference_checkpoint"]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("owner") != record.get("owner") or int(checkpoint.get("update", -1)) != int(record["best_update"]):
            raise RuntimeError(f"R6-G {arm} inference checkpoint does not match its training record")
        model = R6ECEStudent(width=int(config["student"]["width"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        prediction_paths: dict[str, str] = {}
        predictions: dict[str, dict[str, np.ndarray]] = {}
        threshold_info: dict[str, Any] | None = None
        for split in ("h0_calibration", "h0_verification", "h1_development"):
            predictions[split] = predict_r6g_eval_split(
                model,
                split,
                plans[split],
                physics,
                device=device,
                microbatch=microbatch,
                prf_hz=float(config["radar"]["prf_hz"]),
                compute_teacher_correctness=(arm == "T0" and split == "h1_development"),
            )
            path = STAGE / "predictions" / arm / f"{split}.npz"
            _save_predictions(path, predictions[split])
            prediction_paths[split] = str(path.relative_to(ROOT))
            if split == "h0_calibration":
                threshold_info = calibrate_threshold(
                    predictions[split]["score"], float(config["evaluation"]["target_pfa_cpi"])
                )
        assert threshold_info is not None
        threshold = float(threshold_info["threshold"])
        thresholds[arm] = threshold
        h0_summary = h0_metrics(
            predictions["h0_calibration"],
            predictions["h0_verification"],
            threshold=threshold,
            conditions=conditions,
        )
        h1_summary = h1_metrics(
            predictions["h1_development"],
            threshold=threshold,
            conditions=conditions,
            h0_verification=predictions["h0_verification"],
            prf_hz=float(config["radar"]["prf_hz"]),
        )
        arms[arm] = {
            "training": record,
            "inference_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_checkpoint_update": int(record["best_update"]),
            "threshold_calibration": threshold_info,
            "h0": h0_summary,
            "h1_development": h1_summary,
            "prediction_files": prediction_paths,
        }
        del model, checkpoint, predictions
        torch.cuda.empty_cache()

    teacher_predictions = _load_predictions(STAGE / "predictions/T0/h1_development.npz")
    for arm in ARMS:
        predictions = _load_predictions(STAGE / "predictions" / arm / "h1_development.npz")
        if not np.array_equal(predictions["scene_id"], teacher_predictions["scene_id"]):
            raise RuntimeError(f"R6-G {arm} H1 predictions are not paired with the K3 correctness audit")
        for key, value in teacher_predictions.items():
            if key.startswith("teacher_") and key not in predictions:
                predictions[key] = value
        strata = teacher_correctness_strata(
            predictions,
            threshold=thresholds[arm],
            conditions=conditions,
            prf_hz=float(config["radar"]["prf_hz"]),
        )
        arms[arm]["teacher_k3_correctness_strata"] = strata
        if arm != "T0":
            _save_predictions(STAGE / "predictions" / arm / "h1_development.npz", predictions)

    contrasts: dict[str, Any] = {}
    for baseline_name, candidate_name, key in (
        ("T0", "T1", "T1_minus_T0"),
        ("T3", "T1", "T1_minus_T3"),
        ("T0", "T3", "T3_minus_T0"),
    ):
        baseline = _load_predictions(STAGE / "predictions" / baseline_name / "h1_development.npz")
        candidate = _load_predictions(STAGE / "predictions" / candidate_name / "h1_development.npz")
        contrasts[key] = paired_r6g_contrast(
            baseline,
            candidate,
            threshold_baseline=thresholds[baseline_name],
            threshold_candidate=thresholds[candidate_name],
            conditions=conditions,
            replicates=int(config["evaluation"]["bootstrap_replicates"]),
            seed=int(config["evaluation"]["bootstrap_seed"]),
            prf_hz=float(config["radar"]["prf_hz"]),
            confidence=0.95,
        )

    trajectories: dict[str, Any] = {}
    for arm in ARMS:
        run_dir = RUNS / arm
        history = summarize_validation_history(run_dir / "training_history.csv")
        checkpoints = run_dir / "validation_checkpoints"
        validation = training._load_pair_cache("validation")
        curve_points: dict[str, Any] = {}
        for update in VALIDATION_POINTS:
            path = checkpoints / f"update_{update:05d}.pt"
            if not path.is_file():
                raise RuntimeError(f"R6-G {arm} lacks the registered validation checkpoint at update {update}")
            curve_points[str(update)] = _summarize_validation_checkpoint(
                path, arm, update, validation, conditions,
                device=device, config=config,
            )
        trajectories[arm] = {**history, "task_metrics_at_registered_updates": curve_points}

    result = {
        "status": "development_evaluation_complete",
        "experiment": "D2-A-R6-G",
        "protocol_version": str(config["experiment"]["protocol_version"]),
        "git_sha": training._git("rev-parse", "HEAD"),
        "preflight_git_sha": g0["git_sha"],
        "protocol_sha256": g0["protocol_sha256"],
        "config_sha256": g0["config_sha256"],
        "development_only": True,
        "final_blind_generated": False,
        "independent_scene_counts": training._expected_scene_counts(plans),
        "primary_contrasts": contrasts,
        "arms": arms,
        "validation_trajectories": trajectories,
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "precision": "FP32",
        "microbatch": microbatch,
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "evaluation_seconds": time.perf_counter() - started,
    }
    _write_json(STAGE / "metrics.json", result)
    return result
