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


def _cluster_bootstrap_mean_ci(
    values: np.ndarray,
    scene_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    observations = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    if observations.size == 0 or ids.size != observations.size:
        raise ValueError("bootstrap observations and scene IDs must have equal nonzero length")
    if not np.all(np.isfinite(observations)):
        raise ValueError("bootstrap observations must be finite")
    if replicates <= 0 or seed < 0:
        raise ValueError("bootstrap replicates must be positive and seed nonnegative")

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    clusters = [np.flatnonzero(inverse == cluster) for cluster in range(unique_ids.size)]
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, unique_ids.size, size=unique_ids.size)
        rows = np.concatenate([clusters[index] for index in selected])
        bootstrap_means[replicate] = np.mean(observations[rows])
    return [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])]


def _summarize_clustered_delta(
    values: np.ndarray,
    scene_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    delta = np.asarray(values, dtype=np.float64).reshape(-1)
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    return {
        "mean_change": float(np.mean(delta)),
        "ci95": _cluster_bootstrap_mean_ci(
            delta,
            ids,
            replicates=replicates,
            seed=seed,
        ),
    }


def _summarize_paired_correctness(
    teacher_correct: np.ndarray,
    student_correct: np.ndarray,
    scene_ids: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    teacher = np.asarray(teacher_correct, dtype=bool).reshape(-1)
    student = np.asarray(student_correct, dtype=bool).reshape(-1)
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    if teacher.shape != student.shape or ids.size != teacher.size:
        raise ValueError("correctness arrays and scene IDs must have equal length")

    n = teacher.size
    counts = {
        "both_correct": int(np.sum(teacher & student)),
        "teacher_only_correct": int(np.sum(teacher & ~student)),
        "student_only_correct": int(np.sum(~teacher & student)),
        "both_wrong": int(np.sum(~teacher & ~student)),
    }
    if n == 0:
        return {
            "n": 0,
            "n_base_scenes": 0,
            "counts": counts,
            "proportions": {name: None for name in counts},
            "teacher_accuracy": None,
            "student_accuracy": None,
            "student_minus_teacher_accuracy": None,
            "scene_bootstrap_95_ci": None,
            "student_rescue_given_teacher_wrong": None,
            "student_regression_given_teacher_correct": None,
        }

    unique_ids, inverse = np.unique(ids, return_inverse=True)
    cluster_rows = [np.flatnonzero(inverse == cluster) for cluster in range(unique_ids.size)]
    delta = student.astype(np.float64) - teacher.astype(np.float64)
    rng = np.random.default_rng(bootstrap_seed)
    bootstrap_delta = np.empty(bootstrap_replicates, dtype=np.float64)
    for replicate in range(bootstrap_replicates):
        selected_clusters = rng.integers(0, unique_ids.size, size=unique_ids.size)
        sampled_rows = np.concatenate([cluster_rows[index] for index in selected_clusters])
        bootstrap_delta[replicate] = np.mean(delta[sampled_rows])

    teacher_wrong = ~teacher
    teacher_correct_mask = teacher
    return {
        "n": int(n),
        "n_base_scenes": int(unique_ids.size),
        "counts": counts,
        "proportions": {name: count / n for name, count in counts.items()},
        "teacher_accuracy": float(np.mean(teacher)),
        "student_accuracy": float(np.mean(student)),
        "student_minus_teacher_accuracy": float(np.mean(delta)),
        "scene_bootstrap_95_ci": [
            float(value) for value in np.quantile(bootstrap_delta, [0.025, 0.975])
        ],
        "student_rescue_given_teacher_wrong": (
            float(np.mean(student[teacher_wrong])) if np.any(teacher_wrong) else None
        ),
        "student_regression_given_teacher_correct": (
            float(np.mean(~student[teacher_correct_mask]))
            if np.any(teacher_correct_mask)
            else None
        ),
    }


def summarize_teacher_student_error_matrix(
    teacher_predictions: Mapping[str, np.ndarray],
    student_predictions: Mapping[str, np.ndarray],
    *,
    scene_ids: np.ndarray,
    truth_presence: np.ndarray,
    condition: np.ndarray,
    teacher_threshold: float,
    student_threshold: float,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    prf_hz: float,
    range_gate_cells: float = 1.0,
    doppler_gate_hz: float = 20.0,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, object]:
    """Compare teacher and student correctness by task and scene stratum.

    Detection is evaluated on H0 and H1 separately; range and circular-Doppler
    parameter correctness are evaluated on H1 independently of detection;
    joint correctness requires detection and both parameter gates. Bootstrap
    intervals resample unique ``base_scene_id`` rows and do not estimate
    training-seed variation. Teacher and student prediction rows must already
    be aligned to the shared ``scene_ids`` array.
    """
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    presence_values = np.asarray(truth_presence).reshape(-1)
    labels = np.asarray(condition).reshape(-1).astype(str)
    count = ids.size
    if count == 0 or presence_values.size != count or labels.size != count:
        raise ValueError("scene IDs, presence labels, and conditions must have equal nonzero length")
    if np.any(ids == ""):
        raise ValueError("scene_ids must be nonempty")
    if np.any(labels == ""):
        raise ValueError("condition labels must be nonempty")
    if not np.all(np.isin(presence_values, (False, True, 0, 1))):
        raise ValueError("truth_presence must contain only boolean values")
    presence = presence_values.astype(bool)

    settings = (teacher_threshold, student_threshold, prf_hz, range_gate_cells, doppler_gate_hz)
    if not np.all(np.isfinite(settings)):
        raise ValueError("thresholds, PRF, and error gates must be finite")
    if prf_hz <= 0.0 or range_gate_cells < 0.0 or doppler_gate_hz < 0.0:
        raise ValueError("PRF must be positive and error gates must be nonnegative")
    if bootstrap_replicates <= 0 or bootstrap_seed < 0:
        raise ValueError("bootstrap_replicates must be positive and bootstrap_seed nonnegative")

    def prediction_vector(
        source: Mapping[str, np.ndarray], name: str, *, finite_mask: np.ndarray | None = None
    ) -> np.ndarray:
        if name not in source:
            raise ValueError(f"prediction field {name!r} is missing")
        values = np.asarray(source[name], dtype=np.float64).reshape(-1)
        mask = np.ones(count, dtype=bool) if finite_mask is None else finite_mask
        if values.size != count or not np.all(np.isfinite(values[mask])):
            raise ValueError(
                f"prediction field {name!r} must be finite on required rows and have length {count}"
            )
        return values

    def truth_vector(source: np.ndarray, name: str) -> np.ndarray:
        values = np.asarray(source, dtype=np.float64).reshape(-1)
        if values.size != count or not np.all(np.isfinite(values[presence])):
            raise ValueError(f"{name} must be finite for every H1 scene and have length {count}")
        return values

    teacher_score = prediction_vector(teacher_predictions, "score")
    student_score = prediction_vector(student_predictions, "score")
    teacher_range = prediction_vector(teacher_predictions, "range_cell", finite_mask=presence)
    student_range = prediction_vector(student_predictions, "range_cell", finite_mask=presence)
    teacher_doppler = prediction_vector(teacher_predictions, "doppler_hz", finite_mask=presence)
    student_doppler = prediction_vector(student_predictions, "doppler_hz", finite_mask=presence)
    range_truth = truth_vector(truth_range_cell, "truth_range_cell")
    doppler_truth = truth_vector(truth_doppler_hz, "truth_doppler_hz")

    teacher_detected = teacher_score >= teacher_threshold
    student_detected = student_score >= student_threshold
    teacher_range_correct = np.zeros(count, dtype=bool)
    student_range_correct = np.zeros(count, dtype=bool)
    teacher_doppler_correct = np.zeros(count, dtype=bool)
    student_doppler_correct = np.zeros(count, dtype=bool)
    teacher_range_correct[presence] = (
        np.abs(teacher_range[presence] - range_truth[presence]) <= range_gate_cells
    )
    student_range_correct[presence] = (
        np.abs(student_range[presence] - range_truth[presence]) <= range_gate_cells
    )
    teacher_doppler_correct[presence] = np.abs(
        wrap_periodic_difference(
            teacher_doppler[presence] - doppler_truth[presence], period=prf_hz
        )
    ) <= doppler_gate_hz
    student_doppler_correct[presence] = np.abs(
        wrap_periodic_difference(
            student_doppler[presence] - doppler_truth[presence], period=prf_hz
        )
    ) <= doppler_gate_hz
    teacher_presence_correct = teacher_detected == presence
    student_presence_correct = student_detected == presence
    teacher_joint_correct = (
        teacher_detected & teacher_range_correct & teacher_doppler_correct
    )
    student_joint_correct = (
        student_detected & student_range_correct & student_doppler_correct
    )

    groups: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(count, dtype=bool)),
        ("H0", ~presence),
        ("H1", presence),
    ]
    for label in sorted(np.unique(labels).tolist()):
        condition_mask = labels == label
        groups.extend(
            (
                (f"condition:{label}", condition_mask),
                (f"condition:{label}/H0", condition_mask & ~presence),
                (f"condition:{label}/H1", condition_mask & presence),
            )
        )

    grouped: dict[str, object] = {}
    for group_index, (name, mask) in enumerate(groups):
        h1_mask = mask & presence
        task_scene_ids = ids[mask]
        h1_scene_ids = ids[h1_mask]
        task_inputs = (
            (
                "presence",
                teacher_presence_correct[mask],
                student_presence_correct[mask],
                task_scene_ids,
            ),
            ("range", teacher_range_correct[h1_mask], student_range_correct[h1_mask], h1_scene_ids),
            (
                "doppler",
                teacher_doppler_correct[h1_mask],
                student_doppler_correct[h1_mask],
                h1_scene_ids,
            ),
            ("joint", teacher_joint_correct[h1_mask], student_joint_correct[h1_mask], h1_scene_ids),
        )
        tasks: dict[str, object] = {}
        for task_index, (task, teacher_ok, student_ok, task_ids) in enumerate(task_inputs):
            tasks[task] = _summarize_paired_correctness(
                teacher_ok,
                student_ok,
                task_ids,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + group_index * 10 + task_index,
            )
        grouped[name] = {
            "n": int(np.sum(mask)),
            "n_base_scenes": int(np.unique(task_scene_ids).size),
            "tasks": tasks,
        }

    return {
        "bootstrap": {
            "unit": "base_scene_id",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "interval": "two_sided_95_percentile",
        },
        "thresholds": {
            "teacher": float(teacher_threshold),
            "student": float(student_threshold),
            "comparator": ">=",
        },
        "gates": {
            "range_error_cells": float(range_gate_cells),
            "doppler_error_hz": float(doppler_gate_hz),
            "doppler_period_hz": float(prf_hz),
        },
        "uncertainty_is_training_seed_variance": False,
        "groups": grouped,
    }


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
    expected_doppler_shift_hz: float | None = None,
    scene_ids: np.ndarray | None = None,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
    range_gate_cells: float = 1.0,
    doppler_gate_hz: float = 20.0,
) -> dict[str, object]:
    """Summarize paired prediction changes overall and by H0/H1/condition.

    Besides decision agreement this reports score changes, threshold crossings,
    H1 localization/joint success, and wrapped parameter changes. When
    ``scene_ids`` are provided, paired deltas receive scene-cluster bootstrap
    intervals. ``expected_doppler_shift_hz`` maps H1 truth by the specified
    periodic shift and adds prediction-response residuals. Condition labels
    are diagnostic strata only and are never model inputs.
    """
    presence = np.asarray(truth_presence, dtype=bool).reshape(-1)
    labels = np.asarray(condition).reshape(-1).astype(str)
    range_truth = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    doppler_truth = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    count = presence.size
    if count == 0 or labels.size != count or range_truth.size != count or doppler_truth.size != count:
        raise ValueError("truth and condition arrays must have equal nonzero length")
    if (
        not np.isfinite(threshold)
        or not np.isfinite(range_gate_cells)
        or not np.isfinite(doppler_gate_hz)
        or range_gate_cells < 0.0
        or doppler_gate_hz < 0.0
    ):
        raise ValueError("threshold and localization gates must be finite and nonnegative")
    if not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")

    intervention_scene_ids: np.ndarray | None = None
    if scene_ids is not None:
        intervention_scene_ids = np.asarray(scene_ids).reshape(-1).astype(str)
        if intervention_scene_ids.size != count or np.any(intervention_scene_ids == ""):
            raise ValueError("scene_ids must be nonempty and match the prediction rows")
        if bootstrap_replicates <= 0 or bootstrap_seed < 0:
            raise ValueError("bootstrap_replicates must be positive and bootstrap_seed nonnegative")
    if expected_doppler_shift_hz is not None:
        if not np.isfinite(expected_doppler_shift_hz):
            raise ValueError("expected_doppler_shift_hz must be finite")
        if intervention_scene_ids is None:
            raise ValueError("scene_ids are required for a Doppler-shift intervention")

    def vector(
        source: Mapping[str, np.ndarray], name: str, *, finite_mask: np.ndarray | None = None
    ) -> np.ndarray:
        result = np.asarray(source[name], dtype=np.float64).reshape(-1)
        mask = np.ones(count, dtype=bool) if finite_mask is None else finite_mask
        if result.size != count or not np.all(np.isfinite(result[mask])):
            raise ValueError(
                f"prediction field {name!r} must be finite on required rows and have length {count}"
            )
        return result

    base_score = vector(baseline, "score")
    changed_score = vector(intervention, "score")
    base_range = vector(baseline, "range_cell", finite_mask=presence)
    changed_range = vector(intervention, "range_cell", finite_mask=presence)
    base_doppler = vector(baseline, "doppler_hz", finite_mask=presence)
    changed_doppler = vector(intervention, "doppler_hz", finite_mask=presence)
    if not np.all(np.isfinite(range_truth[presence])) or not np.all(
        np.isfinite(doppler_truth[presence])
    ):
        raise ValueError("range and Doppler truth must be finite for every H1 row")
    base_detected = base_score >= threshold
    changed_detected = changed_score >= threshold
    base_range_error = np.full(count, np.nan, dtype=np.float64)
    base_range_error[presence] = base_range[presence] - range_truth[presence]
    intervention_doppler_truth = doppler_truth.copy()
    if expected_doppler_shift_hz is not None:
        intervention_doppler_truth[presence] = wrap_periodic_difference(
            doppler_truth[presence] + expected_doppler_shift_hz,
            period=prf_hz,
        )
    changed_range_error = np.full(count, np.nan, dtype=np.float64)
    changed_range_error[presence] = changed_range[presence] - range_truth[presence]
    base_doppler_error = np.full(count, np.nan, dtype=np.float64)
    changed_doppler_error = np.full(count, np.nan, dtype=np.float64)
    base_doppler_error[presence] = wrap_periodic_difference(
        base_doppler[presence] - doppler_truth[presence], period=prf_hz
    )
    changed_doppler_error[presence] = wrap_periodic_difference(
        changed_doppler[presence] - intervention_doppler_truth[presence], period=prf_hz
    )
    doppler_response_residual: np.ndarray | None = None
    wrapped_expected_shift: float | None = None
    if expected_doppler_shift_hz is not None:
        doppler_response_residual = np.full(count, np.nan, dtype=np.float64)
        wrapped_expected_shift = float(
            wrap_periodic_difference(expected_doppler_shift_hz, period=prf_hz)
        )
        observed_shift = wrap_periodic_difference(
            changed_doppler[presence] - base_doppler[presence], period=prf_hz
        )
        doppler_response_residual[presence] = wrap_periodic_difference(
            observed_shift - wrapped_expected_shift, period=prf_hz
        )
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
    for group_index, (name, mask) in enumerate(masks.items()):
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
            if doppler_response_residual is not None and intervention_scene_ids is not None:
                residuals = doppler_response_residual[h1_mask]
                h1_scene_ids = intervention_scene_ids[h1_mask]
                absolute_residuals = np.abs(residuals)
                row["doppler_response"] = {
                    "n": int(residuals.size),
                    "n_base_scenes": int(np.unique(h1_scene_ids).size),
                    "expected_shift_hz": float(expected_doppler_shift_hz),
                    "wrapped_expected_shift_hz": wrapped_expected_shift,
                    "mean_signed_residual_hz": float(np.mean(residuals)),
                    "mean_absolute_residual_hz": float(np.mean(absolute_residuals)),
                    "p95_absolute_residual_hz": float(np.quantile(absolute_residuals, 0.95)),
                    "mean_signed_residual_ci95_hz": _cluster_bootstrap_mean_ci(
                        residuals,
                        h1_scene_ids,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed,
                    ),
                    "mean_absolute_residual_ci95_hz": _cluster_bootstrap_mean_ci(
                        absolute_residuals,
                        h1_scene_ids,
                        replicates=bootstrap_replicates,
                        seed=bootstrap_seed,
                    ),
                    "bootstrap_unit": "base_scene_id",
                }
        if intervention_scene_ids is not None:
            group_scene_ids = intervention_scene_ids[mask]
            paired_bootstrap: dict[str, object] = {
                "unit": "base_scene_id",
                "replicates": int(bootstrap_replicates),
                "n_base_scenes": int(np.unique(group_scene_ids).size),
                "seed_base": int(bootstrap_seed + group_index * 100),
                "interval": "two_sided_95_percentile",
                "delta_convention": "intervention_minus_baseline",
            }
            paired_deltas: dict[str, object] = {
                "detection_rate": _summarize_clustered_delta(
                    changed_detected[mask].astype(np.float64)
                    - base_detected[mask].astype(np.float64),
                    group_scene_ids,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + group_index * 100,
                )
            }
            h0_mask = mask & ~presence
            if h0_mask.any():
                h0_ids = intervention_scene_ids[h0_mask]
                paired_deltas["false_alarm_rate"] = _summarize_clustered_delta(
                    changed_detected[h0_mask].astype(np.float64)
                    - base_detected[h0_mask].astype(np.float64),
                    h0_ids,
                    replicates=bootstrap_replicates,
                    seed=bootstrap_seed + group_index * 100 + 1,
                )
            if h1_mask.any():
                h1_ids = intervention_scene_ids[h1_mask]
                paired_deltas.update(
                    {
                        "pd": _summarize_clustered_delta(
                            changed_detected[h1_mask].astype(np.float64)
                            - base_detected[h1_mask].astype(np.float64),
                            h1_ids,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + group_index * 100 + 2,
                        ),
                        "parameter_localization_rate": _summarize_clustered_delta(
                            changed_localized[h1_mask].astype(np.float64)
                            - base_localized[h1_mask].astype(np.float64),
                            h1_ids,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + group_index * 100 + 3,
                        ),
                        "joint_success_rate": _summarize_clustered_delta(
                            changed_joint[h1_mask].astype(np.float64)
                            - base_joint[h1_mask].astype(np.float64),
                            h1_ids,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + group_index * 100 + 4,
                        ),
                        "mean_absolute_range_error_cells": _summarize_clustered_delta(
                            np.abs(changed_range_error[h1_mask])
                            - np.abs(base_range_error[h1_mask]),
                            h1_ids,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + group_index * 100 + 5,
                        ),
                        "mean_absolute_doppler_error_hz": _summarize_clustered_delta(
                            np.abs(changed_doppler_error[h1_mask])
                            - np.abs(base_doppler_error[h1_mask]),
                            h1_ids,
                            replicates=bootstrap_replicates,
                            seed=bootstrap_seed + group_index * 100 + 6,
                        ),
                    }
                )
            paired_bootstrap["deltas"] = paired_deltas
            row["paired_scene_bootstrap"] = paired_bootstrap
        if np.all(~presence[mask]):
            row["baseline_h0_false_alarm_rate"] = float(np.mean(base_detected[mask]))
            row["intervention_h0_false_alarm_rate"] = float(np.mean(changed_detected[mask]))
        result[name] = row

    result["threshold"] = float(threshold)
    result["comparator"] = ">="
    result["doppler_difference_interval_hz"] = [-float(prf_hz) / 2.0, float(prf_hz) / 2.0]
    if expected_doppler_shift_hz is not None:
        result["doppler_intervention"] = {
            "expected_shift_hz": float(expected_doppler_shift_hz),
            "wrapped_expected_shift_hz": wrapped_expected_shift,
            "period_hz": float(prf_hz),
            "truth_mapping": "wrap_periodic_difference(truth_doppler_hz + expected_shift_hz, period=prf_hz)",
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
            "bootstrap_unit": "base_scene_id",
        }
    return result

