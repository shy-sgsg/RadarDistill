"""Statistics and grouping checks for the independent D2-A-R3 experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import exp, lgamma, log, log1p
from statistics import NormalDist
from typing import Any

import numpy as np

from radardistill.baselines import wrap_frequency
from radardistill.d2a_baselines import DetectionOutput
from radardistill.d2a_r3_baselines import R3DetectionOutput
from radardistill.evaluation.d2a_e001 import (
    _paired_auc_interval,
    cluster_bootstrap_interval,
    empirical_auc,
    summarize_detector,
)


def summarize_threshold(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("scores must be nonempty and finite")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    greater = values > threshold
    equal = values == threshold
    inclusive = greater | equal
    return {
        "n": int(values.size),
        "pfa_gt": float(np.mean(greater)),
        "pfa_ge": float(np.mean(inclusive)),
        "tie_count": int(np.sum(equal)),
        "tie_fraction": float(np.mean(equal)),
    }


def calibrate_empirical_threshold(
    scores: np.ndarray,
    target_pfa: float = 0.05,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("scores must be nonempty and finite")
    if not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must be between zero and one")
    threshold = float(np.quantile(values, 1.0 - target_pfa, method="higher"))
    result: dict[str, Any] = {"threshold": threshold, "quantile_method": "higher"}
    result.update(summarize_threshold(values, threshold))
    return result


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must define a valid binomial count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    p = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denominator
    half_width = z * np.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total))) / denominator
    return float(center - half_width), float(center + half_width)


def binomial_upper_tail(observed_count: int, total: int, probability: float) -> float:
    """Return P(X >= observed_count) for a Binomial(total, probability)."""
    if total < 0 or observed_count < 0 or observed_count > total:
        raise ValueError("observed_count and total must define a valid binomial event")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    if observed_count == 0:
        return 1.0
    if probability == 0.0:
        return 0.0
    if probability == 1.0:
        return 1.0
    log_p = log(probability)
    log_q = log1p(-probability)
    terms = []
    for count in range(observed_count, total + 1):
        log_mass = (
            lgamma(total + 1)
            - lgamma(count + 1)
            - lgamma(total - count + 1)
            + count * log_p
            + (total - count) * log_q
        )
        terms.append(exp(log_mass))
    return float(min(1.0, sum(terms)))


def validate_disjoint_seed_groups(
    groups: Sequence[Mapping[str, Any]],
    scenes_per_group: int,
) -> list[tuple[int, int, str]]:
    if scenes_per_group <= 0:
        raise ValueError("scenes_per_group must be positive")
    ranges: list[tuple[int, int, str]] = []
    for group in groups:
        name = str(group["name"])
        start = int(group["seed_start"])
        if start < 0:
            raise ValueError("seed_start must be nonnegative")
        ranges.append((start, start + scenes_per_group - 1, name))
    ordered = sorted(ranges)
    for left, right in zip(ordered, ordered[1:]):
        if right[0] <= left[1]:
            raise ValueError(f"seed group ranges overlap: {left[2]} and {right[2]}")
    return ordered


def hierarchical_finite_sample_bootstrap(
    calibration_groups: Sequence[np.ndarray],
    validation_groups: Sequence[np.ndarray],
    *,
    target_pfa: float = 0.05,
    calibration_n: int = 512,
    validation_n: int = 256,
    replicates: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Estimate finite calibration/validation variability without pooling groups.

    Each replicate first selects a calibration and validation group, then draws
    the frozen sample sizes with replacement within those groups.
    """
    calibration = [np.asarray(group, dtype=np.float64).reshape(-1) for group in calibration_groups]
    validation = [np.asarray(group, dtype=np.float64).reshape(-1) for group in validation_groups]
    if not calibration or not validation:
        raise ValueError("at least one calibration and validation group are required")
    if any(group.size < calibration_n for group in calibration):
        raise ValueError("each calibration group must contain calibration_n scores")
    if any(group.size < validation_n for group in validation):
        raise ValueError("each validation group must contain validation_n scores")
    if any(np.any(~np.isfinite(group)) for group in calibration + validation):
        raise ValueError("all scores must be finite")
    if not 0.0 < target_pfa < 1.0 or replicates <= 0:
        raise ValueError("target_pfa and replicates must be positive and valid")

    rng = np.random.default_rng(seed)
    thresholds = np.empty(replicates, dtype=np.float64)
    pfa = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        cal_group = calibration[int(rng.integers(len(calibration)))]
        val_group = validation[int(rng.integers(len(validation)))]
        cal_sample = cal_group[rng.integers(cal_group.size, size=calibration_n)]
        val_sample = val_group[rng.integers(val_group.size, size=validation_n)]
        threshold = float(np.quantile(cal_sample, 1.0 - target_pfa, method="higher"))
        thresholds[index] = threshold
        pfa[index] = np.mean(val_sample >= threshold)
    return {
        "threshold_q025_q50_q975": [float(value) for value in np.quantile(thresholds, [0.025, 0.5, 0.975])],
        "pfa_q025_q50_q975": [float(value) for value in np.quantile(pfa, [0.025, 0.5, 0.975])],
        "probability_pfa_at_least_23_of_256": float(np.mean(pfa >= 23 / 256)),
        "replicates": int(replicates),
        "calibration_n": int(calibration_n),
        "validation_n": int(validation_n),
        "threshold_method": "higher",
        "decision_comparator": ">=",
    }


def _as_e001_output(output: R3DetectionOutput) -> DetectionOutput:
    """Adapt the normalized detector map without mixing its units with raw power."""
    return DetectionOutput(
        score=output.score,
        doppler_hz=output.doppler_hz,
        range_cell=output.range_cell,
        angle_deg=None,
        rd_power=output.score_map,
    )


def summarize_r3_detector(
    h0_output: R3DetectionOutput,
    h1_output: R3DetectionOutput,
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
    """Summarize a raw or normalized R3 score at an independent threshold."""
    result = summarize_detector(
        _as_e001_output(h0_output),
        _as_e001_output(h1_output),
        threshold,
        truth_range_cell,
        truth_doppler_hz,
        cluster_ids,
        prf_hz=prf_hz,
        doppler_gate_hz=doppler_gate_hz,
        range_gate_bins=range_gate_bins,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    n = int(result["n_scenes"])
    pfa_count = int(round(float(result["pfa_cpi"]) * n))
    pd_count = int(round(float(result["pd"]) * n))
    result["pfa_cpi_cluster_bootstrap_ci95"] = result["pfa_cpi_ci95"]
    result["pd_cluster_bootstrap_ci95"] = result["pd_ci95"]
    result["pfa_cpi_ci95"] = list(wilson_interval(pfa_count, n))
    result["pd_ci95"] = list(wilson_interval(pd_count, n))
    result["pfa_ci_method"] = "wilson_score"
    result["pd_ci_method"] = "wilson_score"
    return result


def paired_mean_difference_interval(
    candidate_values: np.ndarray,
    reference_values: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    candidate = np.asarray(candidate_values, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if candidate.shape != reference.shape or candidate.shape != clusters.shape or candidate.size == 0:
        raise ValueError("paired values and scene IDs must be equal nonempty vectors")
    difference = candidate - reference
    if np.unique(clusters).size == clusters.size:
        return _bootstrap_mean_interval(difference, replicates, seed, confidence)
    return cluster_bootstrap_interval(
        difference,
        clusters,
        statistic=np.mean,
        replicates=replicates,
        seed=seed,
        confidence=confidence,
    )


def _bootstrap_mean_interval(
    values: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    samples = np.asarray(values, dtype=np.float64).reshape(-1)
    if samples.size == 0 or replicates <= 0:
        raise ValueError("bootstrap requires nonempty values and positive replicates")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        indices = rng.integers(0, samples.size, size=(count, samples.size))
        estimates[start : start + count] = np.mean(samples[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [tail, 1.0 - tail])
    return float(low), float(high)


def _bootstrap_rmse_interval(
    squared_errors: np.ndarray,
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    values = np.asarray(squared_errors, dtype=np.float64).reshape(-1)
    if values.size == 0 or replicates <= 0:
        raise ValueError("bootstrap requires nonempty values and positive replicates")
    rng = np.random.default_rng(seed)
    estimates = np.full(replicates, np.nan, dtype=np.float64)
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        selected = values[indices]
        valid = np.isfinite(selected)
        denominators = np.sum(valid, axis=1)
        means = np.divide(
            np.nansum(selected, axis=1),
            denominators,
            out=np.full(count, np.nan, dtype=np.float64),
            where=denominators > 0,
        )
        estimates[start : start + count] = np.sqrt(means)
    finite = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(finite, [tail, 1.0 - tail])
    return float(low), float(high)


def independent_mean_difference_interval(
    first_values: np.ndarray,
    second_values: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap a difference of means from two disjoint independent groups."""
    first = np.asarray(first_values, dtype=np.float64).reshape(-1)
    second = np.asarray(second_values, dtype=np.float64).reshape(-1)
    if first.size == 0 or second.size == 0 or replicates <= 0:
        raise ValueError("both independent groups must be nonempty and replicates positive")
    if np.any(~np.isfinite(first)) or np.any(~np.isfinite(second)):
        raise ValueError("independent group values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        count = min(64, replicates - start)
        first_indices = rng.integers(0, first.size, size=(count, first.size))
        second_indices = rng.integers(0, second.size, size=(count, second.size))
        differences[start : start + count] = (
            np.mean(first[first_indices], axis=1) - np.mean(second[second_indices], axis=1)
        )
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(differences, [tail, 1.0 - tail])
    return float(low), float(high)


def summarize_r3_scores(
    h0_scores: np.ndarray,
    h1_scores: np.ndarray,
    predicted_range_cell: np.ndarray,
    predicted_doppler_hz: np.ndarray,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    threshold: float,
    prf_hz: float,
    doppler_gate_hz: float = 20.0,
    range_gate_bins: float = 1.0,
    bootstrap_replicates: int = 2000,
    seed: int = 0,
    auc_summary: tuple[float, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compute R3 CPI-level detection and task metrics from streamed scores."""
    h0 = np.asarray(h0_scores, dtype=np.float64).reshape(-1)
    h1 = np.asarray(h1_scores, dtype=np.float64).reshape(-1)
    predicted_range = np.asarray(predicted_range_cell, dtype=np.float64).reshape(-1)
    predicted_doppler = np.asarray(predicted_doppler_hz, dtype=np.float64).reshape(-1)
    truth_range = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    truth_doppler = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    clusters = np.asarray(cluster_ids).reshape(-1)
    if not (h0.shape == h1.shape == predicted_range.shape == predicted_doppler.shape
            == truth_range.shape == truth_doppler.shape == clusters.shape):
        raise ValueError("R3 H0/H1 scores, estimates, truth, and scene IDs must match")
    if h0.size == 0 or any(np.any(~np.isfinite(values)) for values in
                           (h0, h1, predicted_range, predicted_doppler, truth_range, truth_doppler)):
        raise ValueError("R3 scores, estimates, and truth must be nonempty and finite")
    if not np.isfinite(threshold) or not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("threshold and positive PRF must be finite")

    h0_detected = h0 >= threshold
    h1_detected = h1 >= threshold
    range_error = predicted_range - truth_range
    doppler_error = np.asarray(wrap_frequency(predicted_doppler - truth_doppler, prf_hz))
    joint = (
        h1_detected
        & (np.abs(range_error) <= range_gate_bins)
        & (np.abs(doppler_error) <= doppler_gate_hz)
    )

    def mean_ci(values: np.ndarray, name: str) -> list[float]:
        values = np.asarray(values, dtype=np.float64)
        if np.unique(clusters).size == clusters.size:
            return list(_bootstrap_mean_interval(
                values, bootstrap_replicates, seed + sum(name.encode("utf-8"))
            ))
        return list(cluster_bootstrap_interval(
            values, clusters, statistic=np.mean, replicates=bootstrap_replicates,
            seed=seed + sum(name.encode("utf-8")),
        ))

    def rmse_stat(values: np.ndarray) -> float:
        finite = np.isfinite(values)
        return float(np.sqrt(np.mean(values[finite]))) if np.any(finite) else float("nan")

    def rmse_ci(values: np.ndarray, name: str) -> list[float]:
        if np.unique(clusters).size == clusters.size:
            return list(_bootstrap_rmse_interval(
                values, bootstrap_replicates, seed + sum(name.encode("utf-8"))
            ))
        return list(cluster_bootstrap_interval(
            values, clusters, statistic=rmse_stat, replicates=bootstrap_replicates,
            seed=seed + sum(name.encode("utf-8")),
        ))

    range_sq = np.where(h1_detected, range_error**2, np.nan)
    doppler_sq = np.where(h1_detected, doppler_error**2, np.nan)
    pfa_count = int(np.sum(h0_detected))
    pd_count = int(np.sum(h1_detected))
    if auc_summary is None:
        auc_point = float(empirical_auc(h0, h1))
        auc_interval = _paired_auc_interval(
            h0, h1, clusters, replicates=bootstrap_replicates, seed=seed + 701
        )
    else:
        auc_point, auc_interval = auc_summary
    return {
        "threshold": float(threshold),
        "n_scenes": int(h0.size),
        "pfa_cpi": float(np.mean(h0_detected)),
        "pfa_cpi_ci95": list(wilson_interval(pfa_count, h0.size)),
        "pfa_cpi_cluster_bootstrap_ci95": mean_ci(h0_detected.astype(np.float64), "pfa"),
        "pfa_gt": float(np.mean(h0 > threshold)),
        "pfa_tie_count": int(np.sum(h0 == threshold)),
        "pd": float(np.mean(h1_detected)),
        "pd_ci95": list(wilson_interval(pd_count, h1.size)),
        "pd_cluster_bootstrap_ci95": mean_ci(h1_detected.astype(np.float64), "pd"),
        "roc_auc": float(auc_point),
        "roc_auc_ci95": [float(auc_interval[0]), float(auc_interval[1])],
        "range_rmse_cells_detected": rmse_stat(range_sq),
        "range_rmse_cells_detected_ci95": rmse_ci(range_sq, "range_rmse"),
        "doppler_rmse_hz_detected": rmse_stat(doppler_sq),
        "doppler_rmse_hz_detected_ci95": rmse_ci(doppler_sq, "doppler_rmse"),
        "joint_success": float(np.mean(joint)),
        "joint_success_ci95": mean_ci(joint.astype(np.float64), "joint"),
        "range_gate_bins": float(range_gate_bins),
        "doppler_gate_hz": float(doppler_gate_hz),
    }
