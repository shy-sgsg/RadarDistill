"""Stable scene sampling and periodic-error helpers for physical diagnostics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def select_scene_ids_by_stratum(
    scene_ids: np.ndarray,
    strata: np.ndarray,
    quotas: Mapping[str, int],
    *,
    seed: int,
) -> dict[str, np.ndarray]:
    """Select a deterministic quota of stable scene IDs within every stratum.

    Candidate IDs are sorted before the seeded draw. Therefore a permutation of
    input rows does not change the selected ID set. Returned IDs are sorted as
    well, so the manifest itself is stable across row orders.
    """
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    labels = np.asarray(strata).reshape(-1).astype(str)
    if ids.size == 0 or ids.shape != labels.shape:
        raise ValueError("scene_ids and strata must be nonempty arrays with equal length")
    if np.unique(ids).size != ids.size:
        raise ValueError("scene_ids must be globally unique")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not quotas:
        raise ValueError("at least one stratum quota is required")

    normalized_quotas = {str(name): int(count) for name, count in quotas.items()}
    if any(count < 0 for count in normalized_quotas.values()):
        raise ValueError("stratum quotas must be nonnegative")
    actual_strata = set(labels.tolist())
    if actual_strata != set(normalized_quotas):
        raise ValueError(
            "registered strata must exactly match input strata; "
            f"found={sorted(actual_strata)}, quotas={sorted(normalized_quotas)}"
        )

    selected: dict[str, np.ndarray] = {}
    for ordinal, name in enumerate(sorted(normalized_quotas)):
        candidates = np.sort(ids[labels == name])
        quota = normalized_quotas[name]
        if len(candidates) < quota:
            raise ValueError(
                f"stratum {name!r} has {len(candidates)} scenes, below quota {quota}"
            )
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(ordinal,)))
        chosen = candidates if quota == len(candidates) else rng.choice(
            candidates, size=quota, replace=False
        )
        selected[name] = np.sort(chosen.astype(str, copy=False))
    return selected


def wrap_periodic_difference(difference: np.ndarray | float, *, period: float) -> np.ndarray:
    """Wrap a scalar/vector difference to the half-open interval [-period/2, period/2)."""
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("period must be positive and finite")
    values = np.asarray(difference, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("difference must contain only finite values")
    return np.remainder(values + period / 2.0, period) - period / 2.0


def summarize_prediction_intervention(
    baseline: Mapping[str, np.ndarray],
    intervention: Mapping[str, np.ndarray],
    *,
    truth_presence: np.ndarray,
    condition: np.ndarray,
    threshold: float,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    prf_hz: float,
    range_gate_cells: float = 1.0,
    doppler_gate_hz: float = 20.0,
) -> dict[str, object]:
    """Summarize paired prediction changes overall and by H0/H1/condition.

    Besides decision agreement this reports logit changes, score threshold
    crossings, H1 localization/joint success, and wrapped parameter changes.
    The condition labels are diagnostic strata only and are never model inputs.
    """
    presence = np.asarray(truth_presence, dtype=bool).reshape(-1)
    labels = np.asarray(condition).reshape(-1).astype(str)
    range_truth = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    doppler_truth = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    count = presence.size
    if count == 0 or labels.size != count or range_truth.size != count or doppler_truth.size != count:
        raise ValueError("truth and condition arrays must have equal nonzero length")
    if not np.isfinite(threshold) or range_gate_cells < 0.0 or doppler_gate_hz < 0.0:
        raise ValueError("threshold and localization gates must be finite and nonnegative")
    if not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")

    def vector(source: Mapping[str, np.ndarray], name: str) -> np.ndarray:
        result = np.asarray(source[name], dtype=np.float64).reshape(-1)
        if result.size != count or not np.all(np.isfinite(result)):
            raise ValueError(f"prediction field {name!r} must be finite and have length {count}")
        return result

    base_score = vector(baseline, "score")
    changed_score = vector(intervention, "score")
    base_range = vector(baseline, "range_cell")
    changed_range = vector(intervention, "range_cell")
    base_doppler = vector(baseline, "doppler_hz")
    changed_doppler = vector(intervention, "doppler_hz")
    base_detected = base_score >= threshold
    changed_detected = changed_score >= threshold
    base_range_error = base_range - range_truth
    changed_range_error = changed_range - range_truth
    base_doppler_error = wrap_periodic_difference(base_doppler - doppler_truth, period=prf_hz)
    changed_doppler_error = wrap_periodic_difference(changed_doppler - doppler_truth, period=prf_hz)
    base_localized = (np.abs(base_range_error) <= range_gate_cells) & (
        np.abs(base_doppler_error) <= doppler_gate_hz
    )
    changed_localized = (np.abs(changed_range_error) <= range_gate_cells) & (
        np.abs(changed_doppler_error) <= doppler_gate_hz
    )
    base_joint = base_detected & base_localized & presence
    changed_joint = changed_detected & changed_localized & presence

    masks: dict[str, np.ndarray] = {"all": np.ones(count, dtype=bool), "H0": ~presence, "H1": presence}
    for name in np.unique(labels):
        condition_mask = labels == name
        masks[f"condition:{name}"] = condition_mask
        masks[f"condition:{name}/H0"] = condition_mask & ~presence
        masks[f"condition:{name}/H1"] = condition_mask & presence
    masks["H1/baseline_detected"] = presence & base_detected
    masks["H1/baseline_missed"] = presence & ~base_detected
    masks["H1/baseline_parameter_localized"] = presence & base_localized
    masks["H1/baseline_parameter_mislocalized"] = presence & ~base_localized

    result: dict[str, object] = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if not n:
            result[name] = {"n": 0}
            continue
        row: dict[str, object] = {
            "n": n,
            "score_abs_change_mean": float(np.mean(np.abs(changed_score[mask] - base_score[mask]))),
            "score_abs_change_p95": float(np.quantile(np.abs(changed_score[mask] - base_score[mask]), 0.95)),
            "decision_agreement": float(np.mean(changed_detected[mask] == base_detected[mask])),
            "baseline_detected_count": int(base_detected[mask].sum()),
            "intervention_detected_count": int(changed_detected[mask].sum()),
            "baseline_to_intervention_false_to_true": int(np.sum(mask & ~base_detected & changed_detected)),
            "baseline_to_intervention_true_to_false": int(np.sum(mask & base_detected & ~changed_detected)),
        }
        h1_mask = mask & presence
        if h1_mask.any():
            row.update(
                {
                    "range_abs_change_p95_cells": float(np.quantile(np.abs(changed_range[h1_mask] - base_range[h1_mask]), 0.95)),
                    "doppler_wrapped_abs_change_p95_hz": float(np.quantile(np.abs(wrap_periodic_difference(changed_doppler[h1_mask] - base_doppler[h1_mask], period=prf_hz)), 0.95)),
                    "baseline_parameter_localization_rate": float(np.mean(base_localized[h1_mask])),
                    "intervention_parameter_localization_rate": float(np.mean(changed_localized[h1_mask])),
                    "baseline_joint_success_rate": float(np.mean(base_joint[h1_mask])),
                    "intervention_joint_success_rate": float(np.mean(changed_joint[h1_mask])),
                }
            )
        if np.all(~presence[mask]):
            row["baseline_h0_false_alarm_rate"] = float(np.mean(base_detected[mask]))
            row["intervention_h0_false_alarm_rate"] = float(np.mean(changed_detected[mask]))
        result[name] = row

    result["threshold"] = float(threshold)
    result["comparator"] = ">="
    result["doppler_difference_interval_hz"] = [-float(prf_hz) / 2.0, float(prf_hz) / 2.0]
    return result

