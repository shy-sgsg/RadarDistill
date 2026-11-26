"""Unified no-clutter historical checkpoint evaluation for D2-A-R7-D."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def make_no_clutter_pair(batch: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return paired H0/H1 inputs after exactly discarding simulated clutter."""
    noise = np.asarray(batch.noise_iq)
    target = np.asarray(batch.target_signal_iq)
    if noise.shape != target.shape or noise.ndim != 4:
        raise ValueError("noise_iq and target_signal_iq must share [B,C,N,R] shape")
    if not np.iscomplexobj(noise) or not np.iscomplexobj(target):
        raise ValueError("IQ components must be complex arrays")
    h0 = np.asarray(noise, dtype=np.complex64)
    h1 = np.asarray(noise + target, dtype=np.complex64)
    return h0, h1


def _finite_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return array


def calibrate_score_threshold(calibration_scores: Any, target_pfa: float) -> float:
    """Calibrate a right-tail score threshold with the registered ``higher`` rule."""
    scores = _finite_vector(calibration_scores, "calibration_scores")
    if not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must be strictly between zero and one")
    return float(np.quantile(scores, 1.0 - target_pfa, method="higher"))


def _clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> list[float | None]:
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if trials == 0:
        return [None, None]
    from scipy.stats import beta

    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [lower, upper]


def summarize_detection_scores(
    verification_scores: Any,
    h1_scores: Any,
    threshold: float,
) -> dict[str, Any]:
    """Summarize strict ``score > threshold`` H0/H1 decisions and exact intervals."""
    h0 = _finite_vector(verification_scores, "verification_scores")
    h1 = _finite_vector(h1_scores, "h1_scores")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    false_alarms = int(np.count_nonzero(h0 > threshold))
    detections = int(np.count_nonzero(h1 > threshold))
    h0_fraction = false_alarms / int(h0.size)
    pd = detections / int(h1.size)
    return {
        "verification_scenes": int(h0.size),
        "verification_false_alarms": false_alarms,
        "verification_pfa": h0_fraction,
        "verification_pfa_ci95": _clopper_pearson(false_alarms, int(h0.size)),
        "h1_scenes": int(h1.size),
        "h1_detected": detections,
        "h1_pd": pd,
        "h1_pd_ci95": _clopper_pearson(detections, int(h1.size)),
        "threshold": float(threshold),
    }


def _error_summary(error: np.ndarray) -> dict[str, Any]:
    if error.size == 0:
        return {
            "n": 0,
            "bias": None,
            "mae": None,
            "rmse": None,
            "p95_abs_error": None,
        }
    absolute = np.abs(error)
    return {
        "n": int(error.size),
        "bias": float(np.mean(error)),
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_abs_error": float(np.quantile(absolute, 0.95)),
    }


def summarize_parameter_errors(
    truth_range: Any,
    predicted_range: Any,
    truth_doppler: Any,
    predicted_doppler: Any,
    prf_hz: float,
    detection_mask: Any,
    range_gate_cells: float = 1.0,
    doppler_gate_hz: float = 20.0,
) -> dict[str, Any]:
    """Summarize all-H1 and detected-H1 errors with PRF-periodic Doppler error."""
    range_true = _finite_vector(truth_range, "truth_range")
    range_pred = _finite_vector(predicted_range, "predicted_range")
    doppler_true = _finite_vector(truth_doppler, "truth_doppler")
    doppler_pred = _finite_vector(predicted_doppler, "predicted_doppler")
    detected = np.asarray(detection_mask, dtype=np.bool_)
    if not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    if not np.isfinite(range_gate_cells) or range_gate_cells < 0.0:
        raise ValueError("range_gate_cells must be finite and non-negative")
    if not np.isfinite(doppler_gate_hz) or doppler_gate_hz < 0.0:
        raise ValueError("doppler_gate_hz must be finite and non-negative")
    if not (range_true.size == range_pred.size == doppler_true.size == doppler_pred.size == detected.size):
        raise ValueError("truth, predictions, and detection mask must have equal lengths")

    range_error = range_pred - range_true
    doppler_error = np.remainder(doppler_pred - doppler_true + prf_hz / 2.0, prf_hz) - prf_hz / 2.0
    success = (
        detected
        & (np.abs(range_error) <= range_gate_cells)
        & (np.abs(doppler_error) <= doppler_gate_hz)
    )

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        range_summary = _error_summary(range_error[mask])
        doppler_summary = _error_summary(doppler_error[mask])
        return {
            "n": int(np.count_nonzero(mask)),
            "range_error_cells": range_summary,
            "doppler_error_hz": doppler_summary,
            "range_rmse_cells": range_summary["rmse"],
            "doppler_rmse_hz": doppler_summary["rmse"],
            "range_mae_cells": range_summary["mae"],
            "doppler_mae_hz": doppler_summary["mae"],
        }

    return {
        "all_h1": summarize(np.ones(detected.shape, dtype=np.bool_)),
        "detected_h1": summarize(detected),
        "joint_success_count": int(np.count_nonzero(success)),
        "joint_success": float(np.mean(success)) if success.size else None,
        "joint_success_ci95": _clopper_pearson(int(np.count_nonzero(success)), int(success.size)),
    }


def decode_task_outputs(outputs: Mapping[str, Any], prf_hz: float) -> dict[str, np.ndarray]:
    """Convert common student output dictionaries into one CPI score/range/Doppler."""
    import torch
    from radardistill.distillation.task_losses import decode_task_outputs as decode

    decoded = decode(dict(outputs), float(prf_hz))
    return {
        "score": decoded["presence_probability"].detach().cpu().numpy().astype(np.float64),
        "range_cell": decoded["range_cell"].detach().cpu().numpy().astype(np.float64),
        "doppler_hz": decoded["doppler_hz"].detach().cpu().numpy().astype(np.float64),
    }


def extract_checkpoint_state(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the state dict from the frozen checkpoint payload formats in R7-D."""
    for name in ("state_dict", "model_state_dict", "student_state_dict"):
        state = payload.get(name)
        if isinstance(state, Mapping) and state:
            return state
    raise ValueError("checkpoint payload has no recognized non-empty state dictionary")


def _infer_width(state: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        tensor = state.get(key)
        if tensor is not None:
            shape = tuple(tensor.shape)
            if shape:
                return int(shape[0])
    raise ValueError(f"cannot infer model width from checkpoint keys: {keys}")


def build_historical_student(
    checkpoint_path: str | Path,
    payload: Mapping[str, Any],
    physics_config: Mapping[str, Any],
) -> tuple[Any, str]:
    """Construct the exact historical model family named by a frozen path."""
    from radardistill.d2a_baselines import dpca_time_shift_s
    from radardistill.students.d1_e002 import D1E002Student
    from radardistill.students.physics_guided_student import PhysicsGuidedTaskStudent
    from radardistill.students.r6_architectures import build_r6e_student
    from radardistill.students.r6_j4 import R6J4Student
    from radardistill.students.r6_joint import R6J1Student, build_r6d_student
    from radardistill.students.r6_scaled_cnn import build_r6c_student
    from radardistill.students.r6_ssm import build_r6h_student
    from radardistill.students.r6g_student import R6GCEStudent
    from radardistill.students.task_student import TaskStudent
    from radardistill.students.temporal_frequency_attention import TemporalFrequencyAttentionStudent

    path = Path(checkpoint_path)
    parts = path.parts
    state = extract_checkpoint_state(payload)
    path_text = path.as_posix()

    def task_model(prefix: str = "") -> TaskStudent:
        width = _infer_width(state, prefix + "stem.0.weight")
        spectrum_key = prefix + "spectrum_head.weight"
        include_spectrum = spectrum_key in state
        bins = int(state[spectrum_key].shape[0]) if include_spectrum else 130
        return TaskStudent(width=width, include_spectrum_head=include_spectrum,
                           spectrum_frequency_bins=bins)

    if "D1-E001" in parts:
        model = task_model()
        label = "D1-E001 lightweight depthwise-separable TCN"
    elif "D1-E002" in parts:
        variant = path.parent.name.upper()
        width = _infer_width(state, "backbone.stem.0.weight")
        model = D1E002Student(variant=variant, width=width)
        label = f"D1-E002 {variant}"
    elif "D2-A-R5" in parts and "R5-B" in parts:
        model = TemporalFrequencyAttentionStudent()
        label = "R5-B TemporalFrequencyAttentionStudent"
    elif "D2-A-R5" in parts and "R5-A" in parts and "projection.weight" in state:
        width = _infer_width(state, "student.stem.0.weight")
        spectrum_key = "student.spectrum_head.weight"
        model = PhysicsGuidedTaskStudent(width=width, include_spectrum_head=spectrum_key in state)
        if spectrum_key in state:
            import torch.nn as nn

            model.student.spectrum_head = nn.Conv1d(width, int(state[spectrum_key].shape[0]), 1)
        label = "R5-A S2 fixed-coherence physics-guided student"
    elif "D2-A-R5" in parts or ("D2-A-R6" in parts and "R6-B" in parts):
        model = task_model()
        label = f"{parts[parts.index('D2-A-R5') + 1] if 'D2-A-R5' in parts else 'R6-B'} TaskStudent"
    elif "R6-C" in parts:
        arm = path.parent.name
        model = build_r6c_student(arm)
        label = f"R6-C {arm} TaskStudent"
    elif "R6-D" in parts:
        arm = path.parent.name
        width = _infer_width(state, "task_student.stem.0.weight", "trunk.stem.0.weight")
        model = build_r6d_student(arm, width=width)
        label = f"R6-D {arm}"
    elif "R6-E" in parts:
        arm = path.parent.name
        model = build_r6e_student(arm)
        label = f"R6-E {arm}"
    elif "R6-G" in parts:
        width = _infer_width(state, "stem.weight_real")
        response_key = "training_response_head.weight"
        bins = int(state[response_key].shape[0]) if response_key in state else 520
        model = R6GCEStudent(width=width, spectrum_frequency_bins=bins)
        label = f"R6-G {path.parent.name} CE-TCN"
    elif "R6-H" in parts:
        arm = path.parent.name
        model = build_r6h_student(arm)
        label = f"R6-H {arm}"
    elif "R6-J4" in parts and path.parent.name == "j4_joint":
        width = _infer_width(state, "stem.weight")
        model = R6J4Student(
            dpca_delay_s=dpca_time_shift_s(dict(physics_config)),
            width=width,
            prf_hz=float(physics_config["radar"]["prf_hz"]),
        )
        label = "R6-J4 fixed DPCA/FFT candidate student"
    elif "R6-J4" in parts:
        width = _infer_width(state, "task_student.stem.0.weight")
        model = R6J1Student(width=width)
        label = "R6-J4 J1-J raw-IQ student"
    else:
        raise ValueError(f"no historical model adapter registered for {path_text}")

    return model, label


def parameter_counts(model: Any) -> dict[str, int]:
    """Count learned parameters (excluding buffers such as fixed FFT grids)."""
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)
    return {"parameter_count": total, "trainable_parameter_count": trainable}


def binary_auc(h0_scores: Any, h1_scores: Any) -> float | None:
    """Compute tie-aware ROC AUC from scores, returning None for a missing class."""
    from scipy.stats import rankdata

    h0 = _finite_vector(h0_scores, "h0_scores")
    h1 = _finite_vector(h1_scores, "h1_scores")
    n0, n1 = int(h0.size), int(h1.size)
    ranks = rankdata(np.concatenate((h0, h1)), method="average")
    return float((np.sum(ranks[n0:]) - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def bootstrap_error_intervals(
    error: Any,
    *,
    replicates: int = 1000,
    seed: int = 0,
) -> dict[str, list[float] | None]:
    """Scene-bootstrap RMSE and MAE intervals with a fixed reproducible seed."""
    values = _finite_vector(error, "error")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(int(seed))
    result: dict[str, list[float] | None] = {}
    for metric in ("rmse", "mae"):
        draws = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 64):
            count = min(64, replicates - start)
            indices = rng.integers(0, values.size, size=(count, values.size))
            sampled = values[indices]
            if metric == "rmse":
                draws[start : start + count] = np.sqrt(np.mean(sampled**2, axis=1))
            else:
                draws[start : start + count] = np.mean(np.abs(sampled), axis=1)
        result[f"{metric}_ci95"] = [float(value) for value in np.quantile(draws, [0.025, 0.975])]
    return result
