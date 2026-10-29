"""Preregistered data, schedule, and continuation helpers for D2-A-R6-C."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from radardistill.training.d2a_r6b_hard_negative import (
    build_h0_sample_plan,
    build_repeated_h1_plan,
)


R6C_COURSES = ("C0", "C1", "C2", "C3")
R6C_TREND_OFFSETS = (1600, 1200, 800, 400, 0)
R6C_REQUIRED_TREND_FIELDS = (
    "validation_task_loss",
    "interval_train_task_loss",
    "validation_presence_loss",
    "validation_range_cell_loss",
    "validation_range_offset_loss",
    "validation_doppler_loss",
    "validation_range_rmse_cells",
    "validation_doppler_rmse_hz",
    "validation_auc",
    "validation_range_cell_mismatch_rate",
)


def _allocate(total: int, group_count: int, rng: np.random.Generator) -> list[int]:
    if total < 0 or group_count <= 0:
        raise ValueError("allocation requires a nonnegative total and positive group count")
    counts = [total // group_count] * group_count
    remainder = total % group_count
    for index in rng.permutation(group_count)[:remainder]:
        counts[int(index)] += 1
    return counts


def build_r6c_paired_split_plan(
    data_config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    split_name: str,
) -> list[dict[str, Any]]:
    """Return deterministic, course/condition-balanced paired-scene chunks.

    Train-core and train-extended use the same condition-local seed prefixes;
    training chunks are exactly ``chunk_size`` scenes so their generated scenes
    are identical across the nested split. Validation permits a shorter tail.
    """
    if split_name not in {"train_core", "train_extended", "validation"}:
        raise ValueError("R6-C split must be train_core, train_extended, or validation")
    chunk_size = int(data_config.get("chunk_size", 0))
    chunk_stride = int(data_config.get("chunk_seed_stride", 0))
    condition_stride = int(data_config.get("condition_seed_stride", 0))
    condition_count = int(data_config.get("condition_count", len(conditions)))
    if chunk_size <= 0 or chunk_stride <= 0 or condition_stride <= 0:
        raise ValueError("R6-C chunk and condition seed strides must be positive")
    if len(conditions) != condition_count:
        raise ValueError("R6-C condition table size differs from config")

    split_config_name = "train_extended" if split_name == "train_core" else split_name
    splits = data_config.get("splits")
    if not isinstance(splits, Mapping) or split_config_name not in splits:
        raise ValueError(f"R6-C config is missing {split_config_name} split metadata")
    split = splits[split_config_name]
    seed_bases = tuple(int(value) for value in split.get("seed_bases_by_course", ()))
    assignment_seed = int(split.get("assignment_seed", -1))
    if len(seed_bases) != len(R6C_COURSES) or assignment_seed < 0:
        raise ValueError("R6-C split needs four course seed bases and a nonnegative assignment seed")

    if split_name == "train_core":
        total = int(data_config.get("core_train_pairs", 0))
    elif split_name == "train_extended":
        total = int(data_config.get("extended_train_pairs", 0))
    else:
        total = int(data_config.get("validation_pairs", 0))
    if total <= 0 or total % len(R6C_COURSES):
        raise ValueError("R6-C paired split size must divide evenly across four courses")

    by_course: dict[str, list[tuple[int, Mapping[str, Any]]]] = {course: [] for course in R6C_COURSES}
    for index, condition in enumerate(conditions):
        course = str(condition.get("course"))
        if course not in by_course:
            raise ValueError(f"unknown R6-C course: {course}")
        by_course[course].append((index, condition))
    if any(not rows for rows in by_course.values()):
        raise ValueError("every R6-C course must have at least one condition")

    rng = np.random.default_rng(assignment_seed)
    per_course = total // len(R6C_COURSES)
    if split_name != "validation" and per_course % chunk_size:
        raise ValueError("R6-C training course quotas must be exact multiples of chunk_size")

    plan: list[dict[str, Any]] = []
    course_offsets = {course: index for index, course in enumerate(R6C_COURSES)}
    for course in R6C_COURSES:
        rows = by_course[course]
        course_index = course_offsets[course]
        if split_name == "validation":
            condition_counts = _allocate(per_course, len(rows), rng)
        else:
            chunks_per_course = per_course // chunk_size
            condition_chunk_counts = _allocate(chunks_per_course, len(rows), rng)
            condition_counts = [count * chunk_size for count in condition_chunk_counts]

        for condition_ordinal, ((condition_index, condition), count) in enumerate(
            zip(rows, condition_counts, strict=True)
        ):
            remaining = count
            chunk_index = 0
            while remaining:
                current_count = min(chunk_size, remaining)
                seed = (
                    seed_bases[course_index]
                    + condition_ordinal * condition_stride
                    + chunk_index * chunk_stride
                )
                plan.append({
                    "split": split_name,
                    "presence": "paired",
                    "course": course,
                    "condition_index": int(condition_index),
                    "condition_ordinal": int(condition_ordinal),
                    "condition": dict(condition),
                    "count": int(current_count),
                    "seed": int(seed),
                })
                remaining -= current_count
                chunk_index += 1

    seeds = [row["seed"] for row in plan]
    if len(seeds) != len(set(seeds)):
        raise ValueError(f"R6-C {split_name} produced duplicate scene chunk seeds")
    return plan


def build_r6c_h1_plan(
    condition_indices: np.ndarray,
    *,
    updates: int,
    h1_per_update: int,
    seed: int,
) -> np.ndarray:
    """Build a deterministic, complete-epoch H1 scene traversal plan."""
    return build_repeated_h1_plan(
        condition_indices,
        updates=updates,
        h1_per_update=h1_per_update,
        seed=seed,
    )


def build_r6c_h0_plan(
    h1_plan: np.ndarray,
    h1_condition_indices: np.ndarray,
    h0_condition_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample uniform H0s within each H1 condition while preserving 1:1 balance."""
    return build_h0_sample_plan(
        h1_plan,
        h1_condition_indices,
        h0_condition_indices,
        {},
        mode="uniform",
        seed=seed,
        hard_fraction=0.0,
    )


def r6c_learning_rate(update: int) -> float:
    """R6-B cosine schedule followed by the two registered continuation stages."""
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= 51_200:
        raise ValueError("R6-C update must be an integer in [1, 51200]")
    segments = (
        (1, 6_400, 1e-3, 1e-4),
        (6_401, 12_800, 1e-4, 1e-5),
        (12_801, 25_600, 1e-5, 1e-6),
        (25_601, 51_200, 1e-6, 1e-7),
    )
    for start, stop, lr_start, lr_stop in segments:
        if start <= update <= stop:
            fraction = (update - start) / max(stop - start, 1)
            return float(lr_stop + 0.5 * (lr_start - lr_stop) * (1.0 + math.cos(math.pi * fraction)))
    raise AssertionError("unreachable update schedule state")


def r6c_select_extension_candidate(validation_by_model: Mapping[str, Mapping[str, float]]) -> str:
    """Choose Large/XLarge solely by common-validation task loss; tie favors Large."""
    if set(validation_by_model) != {"Large", "XLarge"}:
        raise ValueError("extension selection requires exactly Large and XLarge validation records")
    losses: dict[str, float] = {}
    for name in ("Large", "XLarge"):
        value = float(validation_by_model[name]["task_loss"])
        if not math.isfinite(value):
            raise ValueError("validation task losses must be finite")
        losses[name] = value
    return min(("Large", "XLarge"), key=lambda name: (losses[name], name != "Large"))


def r6c_extension_gate(
    history: Sequence[Mapping[str, Any]],
    *,
    start_update: int,
    end_update: int,
    relative_task_loss_drop_minimum: float = 0.01,
    required_nonworsening_intervals: int = 3,
    allowed_relative_interval_increase: float = 0.005,
    max_subtask_or_rmse_degradation: float = 0.05,
    max_auc_drop: float = 0.01,
    max_range_mismatch_increase: float = 0.01,
    max_overfit_gap_increase: float = 0.01,
) -> dict[str, Any]:
    """Evaluate the frozen five-point continuation gate for one model arm."""
    if start_update >= end_update:
        raise ValueError("continuation trend start must precede end")
    expected_updates = [end_update - offset for offset in R6C_TREND_OFFSETS]
    if expected_updates[0] != start_update:
        raise ValueError("continuation window must span the registered 1,600-update trend interval")
    by_update: dict[int, Mapping[str, Any]] = {}
    for row in history:
        try:
            update = int(row["update"])
        except (KeyError, TypeError, ValueError):
            continue
        if update in expected_updates:
            by_update[update] = row
    if any(update not in by_update for update in expected_updates):
        missing = [update for update in expected_updates if update not in by_update]
        return {"eligible": False, "reason": "missing_trend_points", "missing_updates": missing}
    rows = [by_update[update] for update in expected_updates]
    values: list[dict[str, float]] = []
    for row in rows:
        try:
            current = {field: float(row[field]) for field in R6C_REQUIRED_TREND_FIELDS}
        except (KeyError, TypeError, ValueError):
            return {"eligible": False, "reason": "missing_required_metric"}
        if not all(math.isfinite(value) for value in current.values()):
            return {"eligible": False, "reason": "nonfinite_required_metric"}
        values.append(current)

    first, last = values[0], values[-1]
    if first["validation_task_loss"] <= 0 or first["interval_train_task_loss"] < 0:
        return {"eligible": False, "reason": "invalid_loss_value"}
    drop = 1.0 - last["validation_task_loss"] / first["validation_task_loss"]
    interval_nonworsening = sum(
        later["validation_task_loss"]
        <= earlier["validation_task_loss"] * (1.0 + allowed_relative_interval_increase)
        for earlier, later in zip(values, values[1:])
    )
    first_gap = max(0.0, first["validation_task_loss"] - first["interval_train_task_loss"])
    last_gap = max(0.0, last["validation_task_loss"] - last["interval_train_task_loss"])
    gap_increase = last_gap - first_gap

    protections: dict[str, bool] = {}
    for field in (
        "validation_presence_loss",
        "validation_range_cell_loss",
        "validation_range_offset_loss",
        "validation_doppler_loss",
        "validation_range_rmse_cells",
        "validation_doppler_rmse_hz",
    ):
        baseline = first[field]
        protections[field] = baseline > 0 and last[field] <= baseline * (1.0 + max_subtask_or_rmse_degradation)
    protections["validation_auc"] = last["validation_auc"] >= first["validation_auc"] - max_auc_drop
    protections["validation_range_cell_mismatch_rate"] = (
        last["validation_range_cell_mismatch_rate"]
        <= first["validation_range_cell_mismatch_rate"] + max_range_mismatch_increase
    )
    protections["overfit_gap"] = gap_increase <= max_overfit_gap_increase

    checks = {
        "task_loss_drop": drop >= relative_task_loss_drop_minimum,
        "nonworsening_intervals": interval_nonworsening >= required_nonworsening_intervals,
        **protections,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "relative_task_loss_drop": float(drop),
        "nonworsening_intervals": int(interval_nonworsening),
        "overfit_gap_start": float(first_gap),
        "overfit_gap_end": float(last_gap),
        "overfit_gap_increase": float(gap_increase),
        "required_updates": expected_updates,
    }


def r6c_extension_pair_decision(
    medium_gate: Mapping[str, Any], large_gate: Mapping[str, Any]
) -> dict[str, bool]:
    """Apply the preregistered all-or-neither continuation rule."""
    medium_eligible = bool(medium_gate.get("eligible", False))
    large_eligible = bool(large_gate.get("eligible", False))
    return {
        "continue_both": medium_eligible and large_eligible,
        "medium_eligible": medium_eligible,
        "large_eligible": large_eligible,
    }
