"""Development evaluation for the matched D2-A-R6-E architecture screen."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from scipy.stats import beta

from radardistill.distillation.task_losses import decode_task_outputs
from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_r6a import summarize_parameter_estimator, wrapped_error
from radardistill.simulation.d2a import generate_d2a_paired_batch


def predict_eval_split(
    model: torch.nn.Module,
    split: str,
    plan: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    *,
    device: torch.device,
    microbatch: int,
    prf_hz: float,
) -> dict[str, np.ndarray]:
    """Stream an independent evaluation split without persisting IQ."""
    if split not in {"h0_calibration", "h0_verification", "h1_development"}:
        raise ValueError("unknown R6-E evaluation split")
    if microbatch <= 0 or prf_hz <= 0:
        raise ValueError("microbatch and PRF must be positive")
    is_h1 = split == "h1_development"
    names = ["scene_id", "condition_index", "score"]
    if is_h1:
        names += ["predicted_range_cell", "predicted_doppler_hz", "truth_range_cell", "truth_doppler_hz"]
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    model.eval()
    with torch.inference_mode():
        for chunk_index, spec in enumerate(plan, start=1):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]),
                int(spec["seed"]),
                float(condition["snr_db"]),
                float(condition["cnr_db"]),
                dict(physics),
                scenario_id=str(condition["scenario_id"]),
            )
            iq = generated.iq_h1 if is_h1 else generated.iq_h0
            iq_ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
            scores: list[np.ndarray] = []
            ranges: list[np.ndarray] = []
            dopplers: list[np.ndarray] = []
            for start in range(0, len(iq_ri), microbatch):
                array = np.ascontiguousarray(iq_ri[start : start + microbatch])
                output = model(torch.from_numpy(array).to(device))
                decoded = decode_task_outputs(output, prf_hz)
                scores.append(decoded["presence_probability"].cpu().numpy())
                if is_h1:
                    ranges.append(decoded["range_cell"].cpu().numpy())
                    dopplers.append(decoded["doppler_hz"].cpu().numpy())
            count = len(generated.base_scene_id)
            values["scene_id"].append(np.asarray(generated.base_scene_id).astype("<U48"))
            values["condition_index"].append(np.full(count, int(spec["condition_index"]), dtype=np.uint8))
            values["score"].append(np.concatenate(scores).astype(np.float32, copy=False))
            if is_h1:
                values["predicted_range_cell"].append(np.concatenate(ranges).astype(np.float32, copy=False))
                values["predicted_doppler_hz"].append(np.concatenate(dopplers).astype(np.float32, copy=False))
                values["truth_range_cell"].append(np.asarray(generated.truth_target_range_cell, dtype=np.float32))
                values["truth_doppler_hz"].append(np.asarray(generated.truth_target_doppler_hz, dtype=np.float32))
    result = {key: np.concatenate(parts) for key, parts in values.items()}
    if any(not np.isfinite(value).all() for key, value in result.items() if key not in {"scene_id", "condition_index"}):
        raise FloatingPointError(f"R6-E {split} produced a nonfinite prediction")
    return result


def exact_upper(events: int, total: int, alpha: float = 0.05) -> float:
    if total <= 0 or not 0 <= events <= total or not 0 < alpha < 1:
        raise ValueError("invalid exact-binomial interval inputs")
    return 1.0 if events == total else float(beta.ppf(1.0 - alpha, events + 1, total - events))


def exact_two_sided(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    if total <= 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("invalid exact-binomial interval inputs")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return [lower, upper]


def h0_metrics(
    calibration: Mapping[str, np.ndarray],
    verification: Mapping[str, np.ndarray],
    *,
    threshold: float,
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cal_score = np.asarray(calibration["score"], dtype=np.float64)
    ver_score = np.asarray(verification["score"], dtype=np.float64)
    cal_index = np.asarray(calibration["condition_index"], dtype=np.int64)
    ver_index = np.asarray(verification["condition_index"], dtype=np.int64)
    if cal_score.ndim != 1 or ver_score.ndim != 1 or not np.isfinite(cal_score).all() or not np.isfinite(ver_score).all():
        raise ValueError("H0 scores must be finite vectors")
    if cal_score.shape != cal_index.shape or ver_score.shape != ver_index.shape:
        raise ValueError("H0 scores and condition indices must align")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    alarms = ver_score >= threshold
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "calibration": {
            "n": int(len(cal_score)),
            "pfa_gt": float(np.mean(cal_score > threshold)),
            "pfa_ge": float(np.mean(cal_score >= threshold)),
            "tie_count": int(np.count_nonzero(cal_score == threshold)),
        },
        "verification": {
            "n": int(len(ver_score)),
            "false_alarms": int(alarms.sum()),
            "pfa": float(alarms.mean()),
            "pfa_ci95_exact_upper": exact_upper(int(alarms.sum()), len(alarms)),
            "tie_count": int(np.count_nonzero(ver_score == threshold)),
        },
        "verification_by_condition": {},
        "c3_pooled": {},
        "c3_cell_familywise_alpha": 0.05,
    }
    course_by_condition = np.asarray([str(item["course"]) for item in conditions])
    if np.any(cal_index < 0) or np.any(ver_index < 0) or np.any(cal_index >= len(conditions)) or np.any(ver_index >= len(conditions)):
        raise ValueError("condition index outside registered condition table")
    c3_mask = course_by_condition[ver_index] == "C3"
    c3_alarms = alarms[c3_mask]
    result["c3_pooled"] = {
        "n": int(c3_mask.sum()),
        "false_alarms": int(c3_alarms.sum()),
        "pfa": None if not len(c3_alarms) else float(c3_alarms.mean()),
        "pfa_ci95_exact_upper": None if not len(c3_alarms) else exact_upper(int(c3_alarms.sum()), len(c3_alarms)),
    }
    for index, condition in enumerate(conditions):
        selected_cal = cal_index == index
        selected_ver = ver_index == index
        selected_alarms = alarms[selected_ver]
        entry = {
            "condition": dict(condition),
            "calibration_n": int(selected_cal.sum()),
            "calibration_pfa_ge": float(np.mean(cal_score[selected_cal] >= threshold)) if selected_cal.any() else None,
            "verification_n": int(selected_ver.sum()),
            "false_alarms": int(selected_alarms.sum()),
            "pfa": float(selected_alarms.mean()) if selected_ver.any() else None,
            "pfa_ci95_exact_upper": exact_upper(int(selected_alarms.sum()), int(selected_ver.sum())) if selected_ver.any() else None,
        }
        if str(condition["course"]) == "C3":
            entry["pfa_familywise_exact_upper"] = (
                exact_upper(int(selected_alarms.sum()), int(selected_ver.sum()), alpha=0.05 / 6.0)
                if selected_ver.any() else None
            )
        result["verification_by_condition"][str(index)] = entry
    c3_entries = [entry for entry in result["verification_by_condition"].values()
                  if entry["condition"]["course"] == "C3"]
    result["c3_six_cells_familywise_pass"] = bool(
        len(c3_entries) == 6 and all(entry.get("pfa_familywise_exact_upper") is not None
                                     and entry["pfa_familywise_exact_upper"] <= 0.10
                                     for entry in c3_entries)
    )
    return result


def h1_metrics(
    predictions: Mapping[str, np.ndarray],
    *,
    threshold: float,
    conditions: Sequence[Mapping[str, Any]],
    h0_verification: Mapping[str, np.ndarray],
    prf_hz: float,
) -> dict[str, Any]:
    condition_index = np.asarray(predictions["condition_index"], dtype=np.int64)
    if condition_index.ndim != 1 or not len(condition_index) or np.any(condition_index >= len(conditions)):
        raise ValueError("H1 condition indices do not match the registered condition table")
    detected = np.asarray(predictions["score"], dtype=np.float64) >= threshold
    course = np.asarray([str(item["course"]) for item in conditions])
    groups = {
        "all": np.ones(len(condition_index), dtype=bool),
        "C2_C3": np.isin(course[condition_index], ("C2", "C3")),
        "C3": course[condition_index] == "C3",
    }
    result: dict[str, Any] = {
        "n": int(len(condition_index)),
        "pd": float(detected.mean()),
        "pd_ci95_exact": exact_two_sided(int(detected.sum()), len(detected)),
        "by_group": {},
        "by_condition": {},
    }
    pred_range = np.asarray(predictions["predicted_range_cell"], dtype=np.float64)
    truth_range = np.asarray(predictions["truth_range_cell"], dtype=np.float64)
    pred_doppler = np.asarray(predictions["predicted_doppler_hz"], dtype=np.float64)
    truth_doppler = np.asarray(predictions["truth_doppler_hz"], dtype=np.float64)
    for name, mask in groups.items():
        summary = summarize_parameter_estimator(
            pred_range[mask], truth_range[mask], pred_doppler[mask], truth_doppler[mask],
            detected[mask], prf_hz=prf_hz, range_gate_cells=1.0, doppler_gate_hz=20.0,
        )
        summary["n"] = int(mask.sum())
        summary["pd"] = float(detected[mask].mean())
        summary["pd_ci95_exact"] = exact_two_sided(int(detected[mask].sum()), int(mask.sum()))
        summary["joint_success_ci95_exact"] = exact_two_sided(
            int(summary["joint_success_count"]), int(mask.sum())
        )
        result["by_group"][name] = summary
    for index, condition in enumerate(conditions):
        mask = condition_index == index
        if not mask.any():
            result["by_condition"][str(index)] = {"condition": dict(condition), "n": 0}
            continue
        summary = summarize_parameter_estimator(
            pred_range[mask], truth_range[mask], pred_doppler[mask], truth_doppler[mask],
            detected[mask], prf_hz=prf_hz, range_gate_cells=1.0, doppler_gate_hz=20.0,
        )
        summary["n"] = int(mask.sum())
        summary["pd"] = float(detected[mask].mean())
        summary["pd_ci95_exact"] = exact_two_sided(int(detected[mask].sum()), int(mask.sum()))
        summary["joint_success_ci95_exact"] = exact_two_sided(
            int(summary["joint_success_count"]), int(mask.sum())
        )
        result["by_condition"][str(index)] = {"condition": dict(condition), **summary}
    h0_index = np.asarray(h0_verification["condition_index"], dtype=np.int64)
    h0_score = np.asarray(h0_verification["score"], dtype=np.float64)
    result["roc_auc_vs_h0_verification"] = float(empirical_auc(h0_score, predictions["score"]))
    result["roc_auc_by_condition"] = {
        str(index): float(empirical_auc(
            h0_score[h0_index == index], np.asarray(predictions["score"])[condition_index == index]
        ))
        for index in range(len(conditions))
    }
    return result


def paired_quality_bootstrap(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    threshold_baseline: float,
    threshold_candidate: float,
    conditions: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
    confidence: float = 0.975,
) -> dict[str, Any]:
    ids = np.asarray(baseline["scene_id"]).astype(str)
    if not np.array_equal(ids, np.asarray(candidate["scene_id"]).astype(str)):
        raise ValueError("R6-E candidate comparison requires paired scene IDs in the same order")
    first_index = np.asarray(baseline["condition_index"], dtype=np.int64)
    second_index = np.asarray(candidate["condition_index"], dtype=np.int64)
    if not np.array_equal(first_index, second_index) or len(ids) != len(first_index):
        raise ValueError("R6-E candidate comparison requires matched condition assignments")
    if replicates <= 0 or seed < 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid paired bootstrap configuration")
    courses = np.asarray([str(item["course"]) for item in conditions])
    primary = np.isin(courses[first_index], ("C2", "C3"))
    if not primary.any():
        raise ValueError("R6-E primary C2+C3 comparison is empty")

    first_detected = np.asarray(baseline["score"], dtype=np.float64) >= threshold_baseline
    second_detected = np.asarray(candidate["score"], dtype=np.float64) >= threshold_candidate
    first_range_error = np.asarray(baseline["predicted_range_cell"], dtype=np.float64) - np.asarray(baseline["truth_range_cell"], dtype=np.float64)
    second_range_error = np.asarray(candidate["predicted_range_cell"], dtype=np.float64) - np.asarray(candidate["truth_range_cell"], dtype=np.float64)
    first_doppler_error = wrapped_error(baseline["predicted_doppler_hz"], baseline["truth_doppler_hz"], 1300.0)
    second_doppler_error = wrapped_error(candidate["predicted_doppler_hz"], candidate["truth_doppler_hz"], 1300.0)
    first_joint = first_detected & (np.abs(first_range_error) <= 1.0) & (np.abs(first_doppler_error) <= 20.0)
    second_joint = second_detected & (np.abs(second_range_error) <= 1.0) & (np.abs(second_doppler_error) <= 20.0)
    joint_delta = second_joint[primary].astype(np.float64) - first_joint[primary].astype(np.float64)
    pd_delta = second_detected.astype(np.float64) - first_detected.astype(np.float64)
    rng = np.random.default_rng(seed)
    joint_samples = np.empty(replicates, dtype=np.float64)
    pd_samples = np.empty(replicates, dtype=np.float64)
    range_ratio_samples = np.empty(replicates, dtype=np.float64)
    doppler_ratio_samples = np.empty(replicates, dtype=np.float64)
    primary_indices = np.flatnonzero(primary)
    for replicate in range(replicates):
        joint_indices = rng.choice(primary_indices, size=len(primary_indices), replace=True)
        all_indices = rng.integers(0, len(ids), size=len(ids))
        joint_samples[replicate] = np.mean(
            second_joint[joint_indices].astype(np.float64)
            - first_joint[joint_indices].astype(np.float64)
        )
        pd_samples[replicate] = np.mean(pd_delta[all_indices])
        range_ratio_samples[replicate] = math.sqrt(
            np.mean(second_range_error[all_indices] ** 2) /
            max(np.mean(first_range_error[all_indices] ** 2), 1e-24)
        )
        doppler_ratio_samples[replicate] = math.sqrt(
            np.mean(second_doppler_error[all_indices] ** 2) /
            max(np.mean(first_doppler_error[all_indices] ** 2), 1e-24)
        )
    tail = (1.0 - confidence) / 2.0
    quantiles = (tail, 1.0 - tail)
    joint_ci = [float(item) for item in np.quantile(joint_samples, quantiles)]
    pd_ci = [float(item) for item in np.quantile(pd_samples, quantiles)]
    range_ci = [float(item) for item in np.quantile(range_ratio_samples, quantiles)]
    doppler_ci = [float(item) for item in np.quantile(doppler_ratio_samples, quantiles)]
    joint_difference = float(np.mean(joint_delta))
    range_ratio = float(math.sqrt(np.mean(second_range_error ** 2) / max(np.mean(first_range_error ** 2), 1e-24)))
    doppler_ratio = float(math.sqrt(np.mean(second_doppler_error ** 2) / max(np.mean(first_doppler_error ** 2), 1e-24)))
    return {
        "group": "identifiable_C2_C3",
        "n_paired_scenes": int(primary.sum()),
        "joint_success_difference": joint_difference,
        "joint_success_ci97_5": joint_ci,
        "confidence_level": confidence,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "meaningful_gain_threshold": 0.02,
        "candidate_screening_pass": bool(joint_difference >= 0.02 and joint_ci[0] > 0.0),
        "protection": {
            "pd_difference": float(np.mean(pd_delta)),
            "pd_difference_ci97_5": pd_ci,
            "pd_lower_minimum": -0.02,
            "pd_pass": bool(pd_ci[0] >= -0.02),
            "range_rmse_ratio": range_ratio,
            "range_rmse_ratio_ci97_5": range_ci,
            "range_rmse_ratio_upper_maximum": 1.05,
            "range_pass": bool(range_ci[1] <= 1.05),
            "doppler_rmse_ratio": doppler_ratio,
            "doppler_rmse_ratio_ci97_5": doppler_ci,
            "doppler_rmse_ratio_upper_maximum": 1.05,
            "doppler_pass": bool(doppler_ci[1] <= 1.05),
        },
    }


def intervention_summary(
    baseline: Mapping[str, np.ndarray],
    intervention: Mapping[str, np.ndarray],
    *,
    threshold: float,
    prf_hz: float,
    expected_doppler_shift_hz: float | None = None,
) -> dict[str, Any]:
    """Summarize paired H1 prediction sensitivity to a controlled IQ transform."""
    fields = ("score", "predicted_range_cell", "predicted_doppler_hz")
    if any(np.asarray(baseline[key]).shape != np.asarray(intervention[key]).shape for key in fields):
        raise ValueError("intervention outputs must remain paired")
    score_before = np.asarray(baseline["score"], dtype=np.float64)
    score_after = np.asarray(intervention["score"], dtype=np.float64)
    range_before = np.asarray(baseline["predicted_range_cell"], dtype=np.float64)
    range_after = np.asarray(intervention["predicted_range_cell"], dtype=np.float64)
    doppler_before = np.asarray(baseline["predicted_doppler_hz"], dtype=np.float64)
    doppler_after = np.asarray(intervention["predicted_doppler_hz"], dtype=np.float64)
    expected = 0.0 if expected_doppler_shift_hz is None else float(expected_doppler_shift_hz)
    response_residual = wrapped_error(doppler_after - doppler_before, np.full(len(doppler_before), expected), prf_hz)
    return {
        "n": int(len(score_before)),
        "decision_agreement": float(np.mean((score_before >= threshold) == (score_after >= threshold))),
        "score_absolute_change_mean": float(np.mean(np.abs(score_after - score_before))),
        "score_absolute_change_p95": float(np.quantile(np.abs(score_after - score_before), 0.95)),
        "range_absolute_change_p95_cells": float(np.quantile(np.abs(range_after - range_before), 0.95)),
        "doppler_wrapped_change_p95_hz": float(np.quantile(np.abs(wrapped_error(doppler_after, doppler_before, prf_hz)), 0.95)),
        "expected_doppler_shift_hz": expected_doppler_shift_hz,
        "doppler_shift_response_residual_mean_absolute_hz": float(np.mean(np.abs(response_residual))),
        "doppler_shift_response_residual_p95_absolute_hz": float(np.quantile(np.abs(response_residual), 0.95)),
    }


def _predict_iq(
    model: torch.nn.Module,
    iq: np.ndarray,
    *,
    device: torch.device,
    microbatch: int,
    prf_hz: float,
) -> dict[str, np.ndarray]:
    iq_ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
    score: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    dopplers: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(iq_ri), microbatch):
            x = torch.from_numpy(np.ascontiguousarray(iq_ri[start : start + microbatch])).to(device)
            decoded = decode_task_outputs(model(x), prf_hz)
            score.append(decoded["presence_probability"].cpu().numpy())
            ranges.append(decoded["range_cell"].cpu().numpy())
            dopplers.append(decoded["doppler_hz"].cpu().numpy())
    return {
        "score": np.concatenate(score),
        "predicted_range_cell": np.concatenate(ranges),
        "predicted_doppler_hz": np.concatenate(dopplers),
    }


def physical_diagnostics(
    model: torch.nn.Module,
    plan: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    base_predictions: Mapping[str, np.ndarray],
    *,
    threshold: float,
    device: torch.device,
    microbatch: int,
    prf_hz: float,
    phase_count: int = 16,
    frequency_shifts_hz: Sequence[float] = (-130.0, 130.0),
) -> dict[str, Any]:
    """Apply common phase rotations and coherent Doppler shifts to the same H1 scenes."""
    if phase_count < 2 or microbatch <= 0:
        raise ValueError("phase_count must be at least two and microbatch positive")
    count = len(base_predictions["scene_id"])
    phase_parts: list[dict[str, list[np.ndarray]]] = [
        {key: [] for key in ("score", "predicted_range_cell", "predicted_doppler_hz")}
        for _ in range(phase_count)
    ]
    shift_parts: dict[float, dict[str, list[np.ndarray]]] = {
        float(shift): {key: [] for key in ("score", "predicted_range_cell", "predicted_doppler_hz")}
        for shift in frequency_shifts_hz
    }
    offset = 0
    model.eval()
    for spec in plan:
        condition = spec["condition"]
        generated = generate_d2a_paired_batch(
            int(spec["count"]),
            int(spec["seed"]),
            float(condition["snr_db"]),
            float(condition["cnr_db"]),
            dict(physics),
            scenario_id=str(condition["scenario_id"]),
        )
        n = len(generated.base_scene_id)
        stop = offset + n
        if not np.array_equal(
            np.asarray(generated.base_scene_id).astype(str),
            np.asarray(base_predictions["scene_id"])[offset:stop].astype(str),
        ):
            raise RuntimeError("physical diagnostic scene IDs differ from the frozen H1 prediction order")
        iq = np.asarray(generated.iq_h1, dtype=np.complex64)
        base_slice = {
            "score": np.asarray(base_predictions["score"])[offset:stop],
            "predicted_range_cell": np.asarray(base_predictions["predicted_range_cell"])[offset:stop],
            "predicted_doppler_hz": np.asarray(base_predictions["predicted_doppler_hz"])[offset:stop],
        }
        for phase_index, phase in enumerate(np.arange(phase_count, dtype=np.float64) * (2.0 * math.pi / phase_count)):
            if phase_index == 0:
                changed = base_slice
            else:
                rotated = iq * np.complex64(np.exp(1j * phase))
                changed = _predict_iq(model, rotated, device=device, microbatch=microbatch, prf_hz=prf_hz)
            for key in phase_parts[phase_index]:
                phase_parts[phase_index][key].append(np.asarray(changed[key]))
        time_index = np.arange(iq.shape[2], dtype=np.float64).reshape(1, 1, -1, 1)
        for shift, parts in shift_parts.items():
            factor = np.exp(1j * 2.0 * math.pi * shift * time_index / prf_hz).astype(np.complex64)
            shifted = iq * factor
            changed = _predict_iq(model, shifted, device=device, microbatch=microbatch, prf_hz=prf_hz)
            for key in parts:
                parts[key].append(np.asarray(changed[key]))
        offset = stop
    if offset != count:
        raise RuntimeError("physical diagnostic prediction count differs from H1 development")

    baseline = {key: np.asarray(base_predictions[key]) for key in phase_parts[0]}
    phase_rows = []
    for index, parts in enumerate(phase_parts):
        changed = {key: np.concatenate(values) for key, values in parts.items()}
        phase_rows.append({
            "phase_index": index,
            "phase_rad": float(index * 2.0 * math.pi / phase_count),
            "summary": intervention_summary(
                baseline, changed, threshold=threshold, prf_hz=prf_hz,
            ),
        })
    shift_rows: dict[str, Any] = {}
    for shift, parts in shift_parts.items():
        changed = {key: np.concatenate(values) for key, values in parts.items()}
        shift_rows[f"{shift:g}"] = intervention_summary(
            baseline, changed, threshold=threshold, prf_hz=prf_hz,
            expected_doppler_shift_hz=shift,
        )
    return {
        "common_phase_rotations": phase_rows,
        "frequency_shift_interventions": shift_rows,
        "n_scenes": int(count),
        "scene_unit": "base_scene_id",
    }
