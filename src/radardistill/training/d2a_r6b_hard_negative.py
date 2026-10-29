"""Frozen helpers for the R6-B matched H0 hard-negative experiment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def required_zero_event_sample_size(target_pfa: float, alpha: float = 0.05) -> int:
    """Smallest n whose one-sided zero-event exact upper bound is <= target."""
    if not np.isfinite(target_pfa) or not 0 < target_pfa < 1:
        raise ValueError("target_pfa must lie strictly between zero and one")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    return int(math.ceil(math.log(alpha) / math.log1p(-float(target_pfa))))


def clopper_pearson_upper(events: int, trials: int, alpha: float = 0.05) -> float:
    """One-sided exact Clopper-Pearson upper confidence limit."""
    if isinstance(events, bool) or isinstance(trials, bool):
        raise ValueError("events and trials must be integers")
    if not isinstance(events, (int, np.integer)) or not isinstance(trials, (int, np.integer)):
        raise ValueError("events and trials must be integers")
    if trials <= 0 or not 0 <= events <= trials:
        raise ValueError("require 0 <= events <= positive trials")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if events == trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - alpha, events + 1, trials - events))


def build_r6b_paired_split_plan(
    data_config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    """Build course-balanced paired-scene chunks with unique deterministic seeds."""
    if split_name not in {"train", "validation"}:
        raise ValueError("paired split must be train or validation")
    splits = data_config.get("splits")
    if not isinstance(splits, Mapping) or split_name not in splits:
        raise ValueError("R6-B split config is missing")
    split = splits[split_name]
    courses = ("C0", "C1", "C2", "C3")
    total = int(split.get("paired_scenes", 0))
    bases = tuple(int(value) for value in split.get("seed_bases_by_course", ()))
    chunk_size = int(data_config.get("chunk_size", 0))
    chunk_stride = int(data_config.get("chunk_seed_stride", 0))
    condition_stride = int(data_config.get("condition_seed_stride", 0))
    assignment_seed = int(split.get("assignment_seed", -1))
    if total <= 0 or total % len(courses) or len(bases) != len(courses):
        raise ValueError("paired split size and seed bases must divide evenly across four courses")
    if min(chunk_size, chunk_stride, condition_stride) <= 0 or assignment_seed < 0:
        raise ValueError("invalid paired split chunk or assignment configuration")
    if len(conditions) != int(data_config.get("condition_count", -1)):
        raise ValueError("condition table size differs from the frozen R6-B config")
    by_course: dict[str, list[tuple[int, Mapping[str, Any]]]] = {course: [] for course in courses}
    for index, condition in enumerate(conditions):
        course = str(condition.get("course"))
        if course not in by_course:
            raise ValueError(f"unknown registered course: {course}")
        by_course[course].append((index, condition))
    if any(not rows for rows in by_course.values()):
        raise ValueError("every registered course must have at least one condition")

    per_course = total // len(courses)
    rng = np.random.default_rng(assignment_seed)
    plan: list[dict[str, Any]] = []
    for course_index, course in enumerate(courses):
        rows = by_course[course]
        counts = [per_course // len(rows)] * len(rows)
        remainder_targets = rng.permutation(len(rows))[: per_course % len(rows)]
        for index in remainder_targets:
            counts[int(index)] += 1
        chunk_index = 0
        for (condition_index, condition), condition_count in zip(rows, counts, strict=True):
            remaining = condition_count
            while remaining:
                count = min(chunk_size, remaining)
                seed = bases[course_index] + chunk_index * chunk_stride
                if chunk_index * chunk_stride >= condition_stride:
                    raise ValueError("paired split chunk seeds exceed their course seed block")
                plan.append({
                    "split": split_name,
                    "presence": "paired",
                    "course": course,
                    "condition_index": condition_index,
                    "condition": dict(condition),
                    "count": count,
                    "seed": seed,
                })
                remaining -= count
                chunk_index += 1
    seeds = [row["seed"] for row in plan]
    if len(seeds) != len(set(seeds)):
        raise ValueError("paired split generated duplicate chunk seeds")
    return plan


def build_hard_negative_pools(
    scores: np.ndarray,
    condition_indices: np.ndarray,
    *,
    top_fraction: float = 0.25,
) -> dict[int, np.ndarray]:
    """Select deterministic within-condition top-score H0 indices."""
    values = np.asarray(scores, dtype=np.float64)
    conditions = np.asarray(condition_indices)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("H0 hardness scores must be a finite nonempty vector")
    if conditions.shape != values.shape or not np.issubdtype(conditions.dtype, np.integer):
        raise ValueError("H0 condition indices must be aligned integers")
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1]")
    pools: dict[int, np.ndarray] = {}
    for condition in np.unique(conditions):
        indices = np.flatnonzero(conditions == condition)
        if not indices.size:
            continue
        count = max(1, int(math.ceil(indices.size * float(top_fraction))))
        order = np.lexsort((indices, -values[indices]))
        pools[int(condition)] = indices[order[:count]].astype(np.int64, copy=False)
    return pools


def build_repeated_h1_plan(
    condition_indices: np.ndarray,
    *,
    updates: int,
    h1_per_update: int,
    seed: int,
) -> np.ndarray:
    """Shuffle a full training cohort each epoch into the fixed H1 update plan."""
    conditions = np.asarray(condition_indices)
    if conditions.ndim != 1 or not len(conditions) or not np.issubdtype(conditions.dtype, np.integer):
        raise ValueError("training conditions must be a nonempty integer vector")
    if updates <= 0 or h1_per_update <= 0 or seed < 0:
        raise ValueError("updates, H1 batch size, and seed must be valid")
    exposures = int(updates) * int(h1_per_update)
    if exposures % len(conditions):
        raise ValueError("fixed updates must traverse the full cohort an integer number of times")
    updates_per_epoch, remainder = divmod(len(conditions), int(h1_per_update))
    if remainder or updates_per_epoch * (exposures // len(conditions)) != updates:
        raise ValueError("training cohort must divide into complete fixed H1 update epochs")
    epochs = exposures // len(conditions)
    rng = np.random.default_rng(seed)
    return np.stack([
        rng.permutation(len(conditions)).reshape(updates_per_epoch, h1_per_update)
        for _ in range(epochs)
    ]).reshape(updates, h1_per_update).astype(np.int64, copy=False)


def build_h0_sample_plan(
    h1_plan: np.ndarray,
    h1_condition_indices: np.ndarray,
    h0_condition_indices: np.ndarray,
    hard_pools: Mapping[int, np.ndarray],
    *,
    mode: str,
    seed: int,
    hard_fraction: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample H0 slots only within each paired H1 condition, with replacement."""
    plan = np.asarray(h1_plan, dtype=np.int64)
    h1_conditions = np.asarray(h1_condition_indices)
    h0_conditions = np.asarray(h0_condition_indices)
    if plan.ndim != 2 or not plan.size or h1_conditions.ndim != 1 or h0_conditions.ndim != 1:
        raise ValueError("H1 plan and train condition vectors must be nonempty")
    if np.any(plan < 0) or np.any(plan >= len(h1_conditions)):
        raise ValueError("H1 plan index outside the registered training split")
    if not np.issubdtype(h1_conditions.dtype, np.integer) or not np.issubdtype(h0_conditions.dtype, np.integer):
        raise ValueError("condition indices must be integers")
    if mode not in {"uniform", "hard_negative"} or seed < 0:
        raise ValueError("unknown H0 sampler mode or invalid seed")
    if not np.isfinite(hard_fraction) or not 0 <= hard_fraction <= 1:
        raise ValueError("hard_fraction must lie in [0, 1]")
    if mode == "hard_negative" and hard_fraction > 0 and not hard_pools:
        raise ValueError("hard pool mapping is required for hard-negative sampling")

    slot_conditions = h1_conditions[plan]
    flat_conditions = slot_conditions.reshape(-1)
    result = np.empty(flat_conditions.shape, dtype=np.int64)
    rng = np.random.default_rng(seed)
    audit_counts: dict[str, Any] = {}
    for condition in np.unique(flat_conditions):
        condition_id = int(condition)
        slots = np.flatnonzero(flat_conditions == condition)
        uniform_pool = np.flatnonzero(h0_conditions == condition)
        if not uniform_pool.size:
            raise ValueError(f"no H0 training pool for registered condition {condition_id}")
        if mode == "uniform":
            result[slots] = rng.choice(uniform_pool, size=len(slots), replace=True)
            audit_counts[str(condition_id)] = {
                "slots": int(len(slots)), "hard": 0, "uniform": int(len(slots)),
                "hard_fraction": 0.0,
            }
            continue
        raw_hard_pool = np.asarray(hard_pools.get(condition_id, ()), dtype=np.int64)
        if not raw_hard_pool.size or np.any(raw_hard_pool < 0) or np.any(raw_hard_pool >= len(h0_conditions)):
            raise ValueError(f"missing or invalid hard pool for condition {condition_id}")
        if np.any(h0_conditions[raw_hard_pool] != condition):
            raise ValueError(f"hard pool contains H0 data from another condition: {condition_id}")
        hard_count = int(math.floor(len(slots) * float(hard_fraction) + 0.5))
        uniform_count = len(slots) - hard_count
        sampled = np.concatenate((
            rng.choice(raw_hard_pool, size=hard_count, replace=True) if hard_count else np.empty(0, dtype=np.int64),
            rng.choice(uniform_pool, size=uniform_count, replace=True) if uniform_count else np.empty(0, dtype=np.int64),
        ))
        rng.shuffle(sampled)
        result[slots] = sampled
        audit_counts[str(condition_id)] = {
            "slots": int(len(slots)), "hard": int(hard_count), "uniform": int(uniform_count),
            "hard_fraction": float(hard_count / len(slots)),
        }
    sampled = result.reshape(plan.shape)
    if not np.array_equal(h0_conditions[sampled], slot_conditions):
        raise RuntimeError("internal error: H0 sample condition differs from matched H1 condition")
    audit = {
        "mode": mode,
        "seed": int(seed),
        "hard_fraction_requested": float(hard_fraction if mode == "hard_negative" else 0.0),
        "hard_fraction_by_condition": {key: value["hard_fraction"] for key, value in audit_counts.items()},
        "h0_slots_by_condition": {key: value["slots"] for key, value in audit_counts.items()},
        "sampling_counts_by_condition": audit_counts,
    }
    return sampled, audit


def assemble_independent_h0_h1_batch(
    arrays: Mapping[str, np.ndarray], h0_indices: np.ndarray, h1_indices: np.ndarray
) -> dict[str, np.ndarray]:
    """Interleave independently sampled H0 and H1 IQ while keeping H1 truth isolated."""
    h0 = np.asarray(h0_indices, dtype=np.int64).reshape(-1)
    h1 = np.asarray(h1_indices, dtype=np.int64).reshape(-1)
    if h0.shape != h1.shape or not h0.size:
        raise ValueError("H0 and H1 index vectors must be aligned and nonempty")
    try:
        iq_h0 = arrays["iq_h0"]
        iq_h1 = arrays["iq_h1"]
        range_truth = arrays["truth_range_cell"]
        doppler_truth = arrays["truth_doppler_hz"]
    except KeyError as exc:
        raise ValueError("paired cache lacks IQ or physical truth") from exc
    count = len(iq_h0)
    if len(iq_h1) != count or len(range_truth) != count or len(doppler_truth) != count:
        raise ValueError("paired cache arrays have inconsistent scene counts")
    if np.any(h0 < 0) or np.any(h1 < 0) or np.any(h0 >= count) or np.any(h1 >= count):
        raise ValueError("sampled train index outside paired cache")
    selected_h0 = np.asarray(iq_h0[h0])
    selected_h1 = np.asarray(iq_h1[h1])
    expected = (len(h0), 3, 130, 32)
    if selected_h0.shape != expected or selected_h1.shape != expected:
        raise ValueError("paired IQ must have frozen [N,3,130,32] shape")
    if not np.iscomplexobj(selected_h0) or not np.iscomplexobj(selected_h1):
        raise ValueError("paired IQ cache must remain complex")
    interleaved = np.empty((len(h0), 2, 3, 130, 32), dtype=np.complex64)
    interleaved[:, 0] = selected_h0
    interleaved[:, 1] = selected_h1
    flat = interleaved.reshape(2 * len(h0), 3, 130, 32)
    iq_ri = np.concatenate((flat.real, flat.imag), axis=1).astype(np.float32, copy=False)
    presence = np.tile(np.asarray([False, True]), len(h0))
    range_values = np.zeros(2 * len(h0), dtype=np.float32)
    doppler_values = np.zeros(2 * len(h0), dtype=np.float32)
    range_values[1::2] = np.asarray(range_truth[h1], dtype=np.float32)
    doppler_values[1::2] = np.asarray(doppler_truth[h1], dtype=np.float32)
    return {
        "iq_ri": np.ascontiguousarray(iq_ri),
        "truth_presence": presence,
        "truth_range_cell": range_values,
        "truth_doppler_hz": doppler_values,
    }
