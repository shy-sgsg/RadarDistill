"""Paired scene-level metrics and cluster bootstrap for D2-A-E001."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import warnings

import numpy as np

from radardistill.baselines import wrap_frequency
from radardistill.d2a_baselines import DetectionOutput


def empirical_auc(negative_scores: np.ndarray, positive_scores: np.ndarray) -> float:
    """Compute empirical ROC AUC by the Mann-Whitney rank identity."""
    negative = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    positive = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    if negative.size == 0 or positive.size == 0:
        raise ValueError("AUC requires at least one negative and one positive score")
    if np.any(~np.isfinite(negative)) or np.any(~np.isfinite(positive)):
        raise ValueError("AUC scores must be finite")
    combined = np.concatenate((negative, positive))
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty(combined.size, dtype=np.float64)
    sorted_values = combined[order]
    start = 0
    while start < sorted_values.size:
        stop = start + 1
        while stop < sorted_values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        average_rank = (start + 1 + stop) / 2.0
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = float(np.sum(ranks[negative.size :]))
    return (positive_rank_sum - positive.size * (positive.size + 1) / 2.0) / (
        negative.size * positive.size
    )


def cluster_bootstrap_interval(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    statistic: Callable[[np.ndarray], float] = np.nanmean,
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI that resamples whole base-scene clusters with replacement."""
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if samples.size == 0 or clusters.shape != samples.shape:
        raise ValueError("values and cluster_ids must be nonempty one-dimensional arrays of equal size")
    if replicates <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("replicates must be positive and confidence must lie in (0,1)")
    unique_clusters = np.unique(clusters)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in selected])
        selected_values = samples[indices]
        if not np.any(np.isfinite(selected_values)):
            estimates[replicate] = np.nan
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            estimates[replicate] = float(statistic(selected_values))
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high)


def _paired_auc_interval(
    negative: np.ndarray,
    positive: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    clusters = np.asarray(cluster_ids).reshape(-1)
    if negative.shape != positive.shape or negative.shape != clusters.shape:
        raise ValueError("paired AUC scores and cluster IDs must have equal shapes")
    unique_clusters = np.unique(clusters)
    cluster_indices = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique_clusters}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.choice(unique_clusters, size=unique_clusters.size, replace=True)
        indices = np.concatenate([cluster_indices[cluster] for cluster in selected])
        estimates[replicate] = empirical_auc(negative[indices], positive[indices])
    tail = 0.025
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def _score_array(output: DetectionOutput) -> np.ndarray:
    return np.atleast_1d(np.asarray(output.score, dtype=np.float64))


def summarize_detector(
    h0_output: DetectionOutput,
    h1_output: DetectionOutput,
    threshold: float,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    prf_hz: float,
    doppler_gate_hz: float = 20.0,
    range_gate_bins: float = 1.0,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Summarize one detector at an independently calibrated CPI threshold."""
    h0_score = _score_array(h0_output)
    h1_score = _score_array(h1_output)
    cluster_ids = np.asarray(cluster_ids)
    truth_range = np.asarray(truth_range_cell, dtype=np.float64)
    truth_doppler = np.asarray(truth_doppler_hz, dtype=np.float64)
    if not (h0_score.shape == h1_score.shape == truth_range.shape == truth_doppler.shape == cluster_ids.shape):
        raise ValueError("H0/H1 scores, truth, and scene IDs must have matching sample counts")

    h0_detected = h0_score >= threshold
    h1_detected = h1_score >= threshold
    h0_rd = np.asarray(h0_output.rd_power, dtype=np.float64)
    h1_rd = np.asarray(h1_output.rd_power, dtype=np.float64)
    if h0_rd.ndim == 2:
        h0_rd = h0_rd[None, ...]
    if h1_rd.ndim == 2:
        h1_rd = h1_rd[None, ...]
    if h0_rd.shape[0] != h0_score.size or h1_rd.shape[0] != h1_score.size:
        raise ValueError("range-Doppler maps must have one map per scene")

    false_alarm_counts = np.sum(h0_rd >= threshold, axis=(1, 2)).astype(np.float64)
    pfa_cut_values = np.mean(h0_rd >= threshold, axis=(1, 2)).astype(np.float64)
    predicted_range = np.atleast_1d(np.asarray(h1_output.range_cell, dtype=np.float64))
    predicted_doppler = np.atleast_1d(np.asarray(h1_output.doppler_hz, dtype=np.float64))
    range_error = predicted_range - truth_range
    doppler_error = np.asarray(wrap_frequency(predicted_doppler - truth_doppler, prf_hz))
    joint = (
        h1_detected
        & (np.abs(range_error) <= range_gate_bins)
        & (np.abs(doppler_error) <= doppler_gate_hz)
    )

    def mean_ci(values: np.ndarray, label: str) -> list[float]:
        return list(
            cluster_bootstrap_interval(
                values,
                cluster_ids,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + sum(label.encode("utf-8")),
            )
        )

    def rmse_stat(values: np.ndarray) -> float:
        valid = np.isfinite(values)
        return float(np.sqrt(np.mean(values[valid]))) if np.any(valid) else float("nan")

    def rmse_ci(values: np.ndarray, label: str) -> list[float]:
        return list(
            cluster_bootstrap_interval(
                values,
                cluster_ids,
                statistic=rmse_stat,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + sum(label.encode("utf-8")),
            )
        )

    conditional_range_sq = np.where(h1_detected, range_error**2, np.nan)
    conditional_doppler_sq = np.where(h1_detected, doppler_error**2, np.nan)
    conditional_range_rmse = rmse_stat(conditional_range_sq)
    conditional_doppler_rmse = rmse_stat(conditional_doppler_sq)

    auc = empirical_auc(h0_score, h1_score)
    auc_ci = _paired_auc_interval(
        h0_score,
        h1_score,
        cluster_ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 701,
    )

    return {
        "threshold": float(threshold),
        "n_scenes": int(h0_score.size),
        "pfa_cpi": float(np.mean(h0_detected)),
        "pfa_cpi_ci95": mean_ci(h0_detected.astype(np.float64), "pfa_cpi"),
        "pfa_cut": float(np.mean(pfa_cut_values)),
        "false_alarms_per_cpi": float(np.mean(h0_detected)),
        "suprathreshold_rd_cells_per_cpi": float(np.mean(false_alarm_counts)),
        "pd": float(np.mean(h1_detected)),
        "pd_ci95": mean_ci(h1_detected.astype(np.float64), "pd"),
        "roc_auc": auc,
        "roc_auc_ci95": [float(auc_ci[0]), float(auc_ci[1])],
        "range_rmse_cells_detected": conditional_range_rmse,
        "range_rmse_cells_detected_ci95": rmse_ci(conditional_range_sq, "range_rmse"),
        "doppler_rmse_hz_detected": conditional_doppler_rmse,
        "doppler_rmse_hz_detected_ci95": rmse_ci(conditional_doppler_sq, "doppler_rmse"),
        "joint_success": float(np.mean(joint)),
        "joint_success_ci95": mean_ci(joint.astype(np.float64), "joint_success"),
        "range_gate_bins": float(range_gate_bins),
        "doppler_gate_hz": float(doppler_gate_hz),
    }
