"""Detection and task metrics for the D2-A-E002 pilot."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution

from radardistill.baselines import wrap_frequency
from radardistill.evaluation.d2a_e001 import _paired_auc_interval, cluster_bootstrap_interval, empirical_auc


def calibrate_student_threshold(scores: Any, target_pfa: float = 0.05) -> dict[str, Any]:
    """Use the preregistered higher empirical quantile and inclusive comparator."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("calibration scores must be nonempty and finite")
    if not np.isfinite(target_pfa) or not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must lie strictly between zero and one")
    threshold = float(np.quantile(values, 1.0 - target_pfa, method="higher"))
    greater = values > threshold
    equal = values == threshold
    inclusive = greater | equal
    return {
        "threshold": threshold,
        "target_pfa": float(target_pfa),
        "quantile_method": "higher",
        "decision_comparator": ">=",
        "n": int(values.size),
        "pfa_gt": float(np.mean(greater)),
        "pfa_ge": float(np.mean(inclusive)),
        "tie_count": int(np.sum(equal)),
        "tie_fraction": float(np.mean(equal)),
    }


def exact_binomial_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    """One-sided Clopper-Pearson upper confidence limit."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must define a valid binomial count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if successes == total:
        return 1.0
    return float(beta_distribution.ppf(confidence, successes + 1, total - successes))


def e0_energy_score(iq: np.ndarray) -> np.ndarray:
    """Frozen E0 noncoherent energy detector: max range of CPI IQ power."""
    values = np.asarray(iq)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32):
        raise ValueError("iq must have shape [B,3,130,32] or [3,130,32]")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    range_energy = np.sum(np.abs(values.astype(np.complex128)) ** 2, axis=(1, 2))
    return np.max(range_energy, axis=-1)


def _rmse(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    return float(np.sqrt(np.mean(values[finite]))) if np.any(finite) else float("nan")


def evaluate_paired_predictions(
    *,
    h0_scores: np.ndarray,
    h1_scores: np.ndarray,
    threshold: float,
    truth_range: np.ndarray,
    predicted_range: np.ndarray,
    truth_doppler: np.ndarray,
    predicted_doppler: np.ndarray,
    base_scene_ids: np.ndarray,
    prf_hz: float,
    range_gate_bins: float = 1.0,
    doppler_gate_hz: float = 20.0,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Summarize paired H0/H1 detection and task estimates with scene bootstrap."""
    negative = np.asarray(h0_scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(h1_scores, dtype=np.float64).reshape(-1)
    truth_r = np.asarray(truth_range, dtype=np.float64).reshape(-1)
    pred_r = np.asarray(predicted_range, dtype=np.float64).reshape(-1)
    truth_f = np.asarray(truth_doppler, dtype=np.float64).reshape(-1)
    pred_f = np.asarray(predicted_doppler, dtype=np.float64).reshape(-1)
    clusters = np.asarray(base_scene_ids).reshape(-1)
    n = positive.size
    if n == 0 or negative.size != n or any(
        values.size != n for values in (truth_r, pred_r, truth_f, pred_f, clusters)
    ):
        raise ValueError("paired predictions, truths, and base_scene_ids must have equal nonzero size")
    if any(np.any(~np.isfinite(values)) for values in (negative, positive, truth_r, pred_r, truth_f, pred_f)):
        raise ValueError("paired scores, truth, and prediction arrays must be finite")
    if not np.isfinite(threshold) or not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("threshold must be finite and PRF positive")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

    false_alarm = negative >= threshold
    detected = positive >= threshold
    range_error = pred_r - truth_r
    doppler_error = np.asarray(wrap_frequency(pred_f - truth_f, prf_hz), dtype=np.float64)
    joint = detected & (np.abs(range_error) <= range_gate_bins) & (np.abs(doppler_error) <= doppler_gate_hz)
    auc = empirical_auc(negative, positive)
    auc_ci = _paired_auc_interval(
        negative,
        positive,
        clusters,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 701,
    )

    def mean_ci(values: np.ndarray, label: str) -> list[float]:
        return list(
            cluster_bootstrap_interval(
                values.astype(np.float64),
                clusters,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + sum(label.encode("utf-8")),
            )
        )

    def rmse_ci(values: np.ndarray, label: str) -> list[float]:
        return list(
            cluster_bootstrap_interval(
                values.astype(np.float64),
                clusters,
                statistic=_rmse,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + sum(label.encode("utf-8")),
            )
        )

    range_squared = range_error**2
    doppler_squared = doppler_error**2
    conditional_range_squared = np.where(detected, range_squared, np.nan)
    conditional_doppler_squared = np.where(detected, doppler_squared, np.nan)
    pfa_count = int(false_alarm.sum())
    return {
        "n_paired_scenes": int(n),
        "threshold": float(threshold),
        "pfa_gt": float(np.mean(negative > threshold)),
        "pfa_ge": float(np.mean(false_alarm)),
        "pfa_ci95_exact_upper": exact_binomial_upper(pfa_count, n),
        "tie_count_h0": int(np.sum(negative == threshold)),
        "pd": float(np.mean(detected)),
        "pd_ci95_bootstrap": mean_ci(detected, "pd"),
        "auc": auc,
        "auc_ci95_paired_bootstrap": [float(value) for value in auc_ci],
        "range_mae_all_h1_cells": float(np.mean(np.abs(range_error))),
        "range_rmse_all_h1_cells": _rmse(range_squared),
        "range_rmse_all_h1_ci95": rmse_ci(range_squared, "range_rmse_all"),
        "range_rmse_detected_h1_cells": _rmse(conditional_range_squared),
        "range_rmse_detected_h1_ci95": rmse_ci(conditional_range_squared, "range_rmse_detected"),
        "doppler_mae_all_h1_hz": float(np.mean(np.abs(doppler_error))),
        "doppler_rmse_all_h1_hz": _rmse(doppler_squared),
        "doppler_rmse_all_h1_ci95": rmse_ci(doppler_squared, "doppler_rmse_all"),
        "doppler_bias_all_h1_hz": float(np.mean(doppler_error)),
        "doppler_rmse_detected_h1_hz": _rmse(conditional_doppler_squared),
        "doppler_rmse_detected_h1_ci95": rmse_ci(conditional_doppler_squared, "doppler_rmse_detected"),
        "joint_success": float(np.mean(joint)),
        "joint_success_ci95_bootstrap": mean_ci(joint, "joint_success"),
        "misses_count_as_joint_failure": True,
        "detected_count": int(detected.sum()),
        "missed_count": int(n - detected.sum()),
        "range_gate_bins": float(range_gate_bins),
        "doppler_gate_hz": float(doppler_gate_hz),
    }


def summarize_h0_verification(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Return empirical false alarms and the preregistered exact upper bound."""
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)) or not np.isfinite(threshold):
        raise ValueError("verification scores and threshold must be finite and nonempty")
    greater = values > threshold
    equal = values == threshold
    count = int(np.sum(greater | equal))
    return {
        "n": int(values.size),
        "threshold": float(threshold),
        "pfa_gt": float(np.mean(greater)),
        "pfa_ge": float(count / values.size),
        "false_alarm_count_ge": count,
        "pfa_ci95_exact_upper": exact_binomial_upper(count, values.size),
        "tie_count": int(equal.sum()),
    }


def paired_difference_interval(
    first: np.ndarray,
    second: np.ndarray,
    base_scene_ids: np.ndarray,
    *,
    statistic: Callable[[np.ndarray, np.ndarray], float] | None = None,
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a paired statistic difference by resampling shared scenes."""
    values_a = np.asarray(first, dtype=np.float64).reshape(-1)
    values_b = np.asarray(second, dtype=np.float64).reshape(-1)
    clusters = np.asarray(base_scene_ids).reshape(-1)
    if values_a.size == 0 or values_a.size != values_b.size or values_a.size != clusters.size:
        raise ValueError("paired values and base_scene_ids must have matching nonzero size")
    if np.any(~np.isfinite(values_a)) or np.any(~np.isfinite(values_b)):
        raise ValueError("paired values must be finite")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("replicates and confidence must be positive and valid")
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a - b))
    unique = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([members[cluster] for cluster in sampled])
        estimates[index] = float(statistic(values_a[rows], values_b[rows]))
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)


def evaluate_detection_scores(
    *,
    h0_scores: np.ndarray,
    h1_scores: np.ndarray,
    threshold: float,
    base_scene_ids: np.ndarray,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Evaluate a CPI detector when it has no range/Doppler parameter output."""
    negative = np.asarray(h0_scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(h1_scores, dtype=np.float64).reshape(-1)
    clusters = np.asarray(base_scene_ids).reshape(-1)
    if negative.size == 0 or positive.size != negative.size or clusters.size != negative.size:
        raise ValueError("paired detection scores and base_scene_ids must have matching nonzero size")
    if np.any(~np.isfinite(negative)) or np.any(~np.isfinite(positive)) or not np.isfinite(threshold):
        raise ValueError("detection scores and threshold must be finite")
    false_alarm = negative >= threshold
    detected = positive >= threshold
    auc = empirical_auc(negative, positive)
    auc_ci = _paired_auc_interval(
        negative,
        positive,
        clusters,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 701,
    )
    pd_ci = cluster_bootstrap_interval(
        detected.astype(np.float64),
        clusters,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 809,
    )
    return {
        "n_paired_scenes": int(negative.size),
        "threshold": float(threshold),
        "pfa_gt": float(np.mean(negative > threshold)),
        "pfa_ge": float(np.mean(false_alarm)),
        "pfa_ci95_exact_upper": exact_binomial_upper(int(false_alarm.sum()), negative.size),
        "tie_count_h0": int(np.sum(negative == threshold)),
        "pd": float(np.mean(detected)),
        "pd_ci95_bootstrap": [float(value) for value in pd_ci],
        "auc": float(auc),
        "auc_ci95_paired_bootstrap": [float(value) for value in auc_ci],
    }
