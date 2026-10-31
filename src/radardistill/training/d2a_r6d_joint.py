"""Truth-only losses and fixed-budget utilities for D2-A-R6-D."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from radardistill.distillation.task_losses import (
    circular_doppler_error,
    doppler_circular_loss,
    encode_doppler_hz,
)
from radardistill.students.r6_joint import R6D_DOPPLER_BINS, R6D_RANGE_BINS


R6D_COURSES = ("C0", "C1", "C2", "C3")


def r6d_learning_rate(
    update: int, *, total_updates: int = 25_600, initial: float = 1e-3, final: float = 1e-5
) -> float:
    """One cosine decay spanning the whole matched update budget."""
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= total_updates:
        raise ValueError("update must be an integer within the frozen training budget")
    if total_updates < 2 or not math.isfinite(initial) or not math.isfinite(final) or initial <= final or final <= 0:
        raise ValueError("invalid cosine learning-rate schedule")
    fraction = (update - 1) / (total_updates - 1)
    return float(final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * fraction)))


def _allocate_r6d(total: int, groups: int, rng: np.random.Generator) -> list[int]:
    if total < 0 or groups <= 0:
        raise ValueError("invalid R6-D allocation")
    counts = [total // groups] * groups
    for index in rng.permutation(groups)[: total % groups]:
        counts[int(index)] += 1
    return counts


def build_r6d_paired_plan(
    data_config: Mapping[str, Any], conditions: list[Mapping[str, Any]], split_name: str
) -> list[dict[str, Any]]:
    """Build course-balanced, deterministic H0/H1 paired train or validation chunks."""
    if split_name not in {"train", "validation"}:
        raise ValueError("R6-D paired split must be train or validation")
    chunk_size = int(data_config["chunk_size"])
    chunk_stride = int(data_config["chunk_seed_stride"])
    condition_stride = int(data_config["condition_offset_stride"])
    if min(chunk_size, chunk_stride, condition_stride) <= 0:
        raise ValueError("R6-D seed strides and chunk size must be positive")
    if len(conditions) != int(data_config.get("condition_count", len(conditions))):
        raise ValueError("R6-D condition count mismatch")
    split = data_config["splits"][split_name]
    total = int(split["paired_scenes"])
    if total <= 0 or total % len(R6D_COURSES):
        raise ValueError("R6-D paired scene count must be positive and divide across courses")
    bases = tuple(int(value) for value in split["seed_bases_by_course"])
    assignment_seed = int(split["assignment_seed"])
    if len(bases) != len(R6D_COURSES) or assignment_seed < 0:
        raise ValueError("R6-D split requires four seed bases and a nonnegative assignment seed")
    by_course = {course: [] for course in R6D_COURSES}
    for condition_index, condition in enumerate(conditions):
        course = str(condition.get("course"))
        if course not in by_course:
            raise ValueError(f"unknown R6-D course: {course}")
        by_course[course].append((condition_index, condition))
    if any(not values for values in by_course.values()):
        raise ValueError("every R6-D course must contain registered conditions")

    rng = np.random.default_rng(assignment_seed)
    per_course = total // len(R6D_COURSES)
    if split_name == "train" and per_course % chunk_size:
        raise ValueError("R6-D training course allocation must use complete chunks")
    plans: list[dict[str, Any]] = []
    for course_index, course in enumerate(R6D_COURSES):
        rows = by_course[course]
        if split_name == "train":
            chunk_counts = _allocate_r6d(per_course // chunk_size, len(rows), rng)
            counts = [value * chunk_size for value in chunk_counts]
        else:
            counts = _allocate_r6d(per_course, len(rows), rng)
        for condition_ordinal, ((condition_index, condition), count) in enumerate(zip(rows, counts, strict=True)):
            remaining = count
            chunk_index = 0
            while remaining:
                current = min(chunk_size, remaining)
                seed = bases[course_index] + condition_ordinal * condition_stride + chunk_index * chunk_stride
                plans.append({
                    "split": split_name,
                    "course": course,
                    "condition_index": int(condition_index),
                    "condition_ordinal": int(condition_ordinal),
                    "condition": dict(condition),
                    "count": int(current),
                    "seed": int(seed),
                })
                remaining -= current
                chunk_index += 1
    seeds = [int(row["seed"]) for row in plans]
    if len(seeds) != len(set(seeds)):
        raise RuntimeError(f"R6-D {split_name} plan has duplicate chunk seeds")
    if sum(int(row["count"]) for row in plans) != total:
        raise RuntimeError(f"R6-D {split_name} plan has an incorrect scene total")
    return plans


def r6d_scene_ids(plan: list[Mapping[str, Any]]) -> set[str]:
    return {
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in plan
        for index in range(int(row["count"]))
    }


def audit_r6d_scene_plans(
    plans: Mapping[str, list[Mapping[str, Any]]],
    *,
    expected_counts: Mapping[str, int],
    prior_scene_ids: set[str],
) -> dict[str, Any]:
    if set(plans) != set(expected_counts):
        raise ValueError("R6-D plan names and expected scene counts differ")
    ids_by_split: dict[str, set[str]] = {}
    seen: set[str] = set()
    for split, plan in plans.items():
        ids = r6d_scene_ids(list(plan))
        expected = int(expected_counts[split])
        if len(ids) != expected:
            raise RuntimeError(f"R6-D {split} plan contains {len(ids)} unique IDs, expected {expected}")
        if len(ids) != sum(int(row["count"]) for row in plan):
            raise RuntimeError(f"R6-D {split} repeats a scene ID within the split")
        if ids & seen:
            raise RuntimeError(f"R6-D split scene IDs overlap at {split}")
        overlap = ids & prior_scene_ids
        if overlap:
            raise RuntimeError(f"R6-D {split} reuses {len(overlap)} prior scene IDs")
        ids_by_split[split] = ids
        seen.update(ids)
    return {"scene_ids": ids_by_split, "scene_counts": {name: len(ids) for name, ids in ids_by_split.items()},
            "overlap_count": 0, "prior_overlap_count": 0}


def _masked_zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def r6d_loss(
    outputs: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    *,
    arm: str,
    prf_hz: float = 1300.0,
    weights: Mapping[str, float] | None = None,
    range_beta: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute E1 task losses while masking H0 parameter supervision.

    ``j1_detection`` qualifies detection only; ``j1_parameter`` is conditional
    H1 parameter regression and has no detector claim; ``j1_joint`` and
    ``j2_joint`` optimize all registered task terms.
    """
    if arm not in {"j1_detection", "j1_parameter", "j1_joint", "j2_joint"}:
        raise ValueError(f"unknown R6-D arm {arm!r}")
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if not math.isfinite(range_beta) or range_beta <= 0:
        raise ValueError("range_beta must be positive and finite")
    default_weights = {
        "presence": 1.0,
        "range_cell": 1.0,
        "range_offset": 1.0,
        "doppler": 1.0,
        "candidate": 1.0,
    }
    if weights is not None:
        default_weights.update({key: float(value) for key, value in weights.items()})
    if any(not math.isfinite(value) or value < 0 for value in default_weights.values()):
        raise ValueError("loss weights must be finite and non-negative")

    presence_logit = outputs["presence_logit"]
    presence_truth = truth["truth_presence"].to(device=presence_logit.device, dtype=presence_logit.dtype)
    if presence_truth.shape != presence_logit.shape:
        raise ValueError("truth_presence must have one label per sample")
    h1 = presence_truth > 0.5
    zero_reference = outputs["range_logits"] if arm == "j1_parameter" else presence_logit
    zero = _masked_zero(zero_reference)
    presence = F.binary_cross_entropy_with_logits(presence_logit, presence_truth)
    range_cell = zero
    range_offset = zero
    doppler = zero
    candidate = zero

    if arm in {"j1_parameter", "j1_joint"} and torch.any(h1):
        range_logits = outputs["range_logits"]
        range_truth = truth["truth_range_cell"].to(device=range_logits.device, dtype=range_logits.dtype)
        doppler_truth = truth["truth_doppler_hz"].to(device=range_logits.device, dtype=range_logits.dtype)
        if range_truth.shape != presence_truth.shape or doppler_truth.shape != presence_truth.shape:
            raise ValueError("range and Doppler truths must have one value per sample")
        range_index = torch.floor(range_truth[h1]).long().clamp(0, range_logits.shape[-1] - 1)
        range_cell = F.cross_entropy(range_logits[h1], range_index)
        rows = torch.nonzero(h1, as_tuple=False).squeeze(-1)
        offset_target = range_truth[h1] - range_index.to(range_truth.dtype)
        selected_offset = outputs["range_offset"][rows, range_index]
        range_offset = F.smooth_l1_loss(selected_offset, offset_target, beta=range_beta)
        target_vector = encode_doppler_hz(doppler_truth[h1], prf_hz)
        selected_vector = outputs["doppler_vector"][rows, range_index]
        doppler = doppler_circular_loss(selected_vector, target_vector).mean()

    if arm == "j2_joint":
        if torch.any(h1):
            range_truth = truth["truth_range_cell"].to(device=presence_logit.device, dtype=presence_logit.dtype)
            doppler_truth = truth["truth_doppler_hz"].to(device=presence_logit.device, dtype=presence_logit.dtype)
            range_index = torch.floor(range_truth[h1]).long().clamp(0, R6D_RANGE_BINS - 1)
            wrapped = circular_doppler_error(doppler_truth[h1], torch.zeros_like(doppler_truth[h1]), prf_hz)
            bin_width = prf_hz / R6D_DOPPLER_BINS
            doppler_index = torch.floor((wrapped + prf_hz / 2.0) / bin_width).long().clamp(0, R6D_DOPPLER_BINS - 1)
            candidate_index = doppler_index * R6D_RANGE_BINS + range_index
            conditional_logits = outputs["candidate_logits"].flatten(start_dim=1)
            candidate = F.cross_entropy(conditional_logits[h1], candidate_index)
            range_offset = F.smooth_l1_loss(
                outputs["range_cell_estimate"][h1], range_truth[h1], beta=range_beta
            )
            target_vector = encode_doppler_hz(doppler_truth[h1], prf_hz)
            doppler = doppler_circular_loss(outputs["doppler_vector_pooled"][h1], target_vector).mean()

    if arm == "j1_detection" or arm == "j1_parameter":
        # The specialized arms intentionally do not optimize the other task.
        if arm == "j1_detection":
            presence_term = presence
        else:
            presence_term = zero
    else:
        presence_term = presence

    total = (
        default_weights["presence"] * presence_term
        + default_weights["range_cell"] * range_cell
        + default_weights["range_offset"] * range_offset
        + default_weights["doppler"] * doppler
        + default_weights["candidate"] * candidate
    )
    return {
        "presence": presence_term,
        "range_cell": range_cell,
        "range_offset": range_offset,
        "doppler": doppler,
        "candidate": candidate,
        "total": total,
    }
