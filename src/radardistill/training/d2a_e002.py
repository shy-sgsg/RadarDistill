"""Matched-course data planning and training utilities for D2-A-E002."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


COURSE_LEVELS = ("C0", "C1", "C2", "C3")
PILOT_ARMS = ("A", "B", "C")


def validate_e002_config(config: dict[str, Any]) -> None:
    """Reject a config that diverges from the frozen E002 v2 pilot budget."""
    try:
        experiment = config["experiment"]
        data = config["data"]
        training = config["training"]
        student = config["student"]
        teacher = config["teacher"]
    except (KeyError, TypeError) as exc:
        raise ValueError("E002 config is missing a required top-level section") from exc
    if experiment.get("id") != "D2-A-E002" or float(experiment.get("protocol_version", -1)) != 2.0:
        raise ValueError("config must identify preregistered D2-A-E002 protocol v2.0")
    if experiment.get("final_blind_generation_authorized") is not False:
        raise ValueError("final blind generation is not authorized by E002 v2")
    if training.get("cuda_required") is not True or training.get("cpu_fallback") != "forbidden":
        raise ValueError("E002 training requires CUDA and forbids CPU fallback")
    if training.get("device") != "cuda:0":
        raise ValueError("E002 training device must remain cuda:0")
    if training.get("optimizer_updates_per_arm") != 1600:
        raise ValueError("E002 pilot update budget is frozen at 1600 updates per arm")
    if tuple(training.get("arms", ())) != (
        "A_direct_truth",
        "B_curriculum_truth",
        "C_curriculum_K2",
    ):
        raise ValueError("E002 arm order must remain A, B, C")
    if teacher.get("official_teacher_approved") is not False:
        raise ValueError("K2 remains a research soft-response candidate, not an approved teacher")
    if tuple(teacher.get("input_fields", ())) != ("observable_iq_only",):
        raise ValueError("K2 generation must depend only on observable IQ")
    if training.get("paired_base_scenes_per_update") != 32:
        raise ValueError("E002 requires 32 paired base scenes per optimizer update")
    if training.get("effective_batch_size") != 64:
        raise ValueError("E002 effective batch is frozen at 64 IQ examples")
    if student.get("same_initial_state_for_all_arms") is not True:
        raise ValueError("A/B/C must start from the same initialization")

    levels = data.get("course_levels", {})
    if tuple(levels) != COURSE_LEVELS:
        raise ValueError("course levels must be ordered C0, C1, C2, C3")
    expected_pairs = int(data["train"]["pairs_per_course_level"])
    for level, specification in levels.items():
        counts = specification.get("scenario_counts", specification.get("scenario_snr_counts"))
        if not isinstance(counts, Mapping) or sum(int(value) for value in counts.values()) != expected_pairs:
            raise ValueError(f"{level} training cell counts must sum to {expected_pairs}")
    updates = training.get("curriculum_updates", {})
    if sum(int(value) for value in updates.values()) != training["optimizer_updates_per_arm"]:
        raise ValueError("curriculum update allocation must sum to the frozen 1600 updates")
    if tuple(updates) != ("C0", "C1", "C2", "C3", "C4"):
        raise ValueError("curriculum stages must be ordered C0 through C4")
    if int(data["train"]["paired_base_scenes"]) != (
        int(training["optimizer_updates_per_arm"])
        * int(training["paired_base_scenes_per_update"])
    ):
        raise ValueError("train pair count must equal updates times paired scenes per update")
    early_per_course = int(updates["C0"]) * int(training["paired_base_scenes_per_update"])
    mixed_per_course = int(updates["C4"]) * int(training["paired_base_scenes_per_update"]) // 4
    if early_per_course + mixed_per_course != expected_pairs:
        raise ValueError("C0-C3 plus balanced C4 must consume each course pair exactly once")
    for section_name in ("validation", "teacher_audit"):
        section = data[section_name]
        expected = int(section["pairs_per_course_level"])
        for level, counts in section["cell_counts"].items():
            if sum(int(value) for value in counts.values()) != expected:
                raise ValueError(f"{section_name} {level} cells must sum to {expected}")
    if data["development"].get("cell_counts_match_validation") is not True:
        raise ValueError("development cell allocation must match validation")


def build_matched_pair_plans(
    course_pair_ids: Mapping[str, np.ndarray],
    *,
    updates_per_course: int,
    mixed_updates: int,
    batch_pairs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Create A full-mix and identical B/C course schedules over one cohort.

    Every pair ID appears exactly once. B/C first use equal-sized pure course
    stages, then each mixed-stage batch draws the same number from every course.
    """
    if tuple(course_pair_ids) != COURSE_LEVELS:
        raise ValueError("course_pair_ids must be ordered C0, C1, C2, C3")
    if updates_per_course <= 0 or mixed_updates <= 0 or batch_pairs <= 0 or seed < 0:
        raise ValueError("course updates, mixed updates, batch pairs, and seed must be valid")
    if batch_pairs % 4:
        raise ValueError("balanced mixed batches require batch_pairs divisible by four")
    values = {name: np.asarray(ids).reshape(-1) for name, ids in course_pair_ids.items()}
    if any(ids.size == 0 for ids in values.values()):
        raise ValueError("every course level must have at least one pair ID")
    all_ids = np.concatenate(tuple(values.values()))
    if np.unique(all_ids).size != all_ids.size:
        raise ValueError("course pair IDs must be globally unique")
    early_count = updates_per_course * batch_pairs
    mixed_per_course = mixed_updates * (batch_pairs // 4)
    required = early_count + mixed_per_course
    if any(ids.size != required for ids in values.values()):
        raise ValueError(f"each course must have exactly {required} unique pair IDs")

    rng = np.random.default_rng(seed)
    shuffled = {name: rng.permutation(ids) for name, ids in values.items()}
    b_rows: list[np.ndarray] = []
    for name in COURSE_LEVELS:
        course = shuffled[name]
        for update in range(updates_per_course):
            start = update * batch_pairs
            b_rows.append(course[start : start + batch_pairs])
    for update in range(mixed_updates):
        row_parts = []
        per_course = batch_pairs // 4
        for name in COURSE_LEVELS:
            start = early_count + update * per_course
            row_parts.append(shuffled[name][start : start + per_course])
        mixed_row = np.concatenate(row_parts)
        b_rows.append(mixed_row[rng.permutation(batch_pairs)])

    b_plan = np.stack(b_rows)
    a_plan = rng.permutation(all_ids).reshape(b_plan.shape)
    c_plan = b_plan.copy()
    for name, plan in (("A", a_plan), ("B", b_plan), ("C", c_plan)):
        if not np.array_equal(np.sort(plan.reshape(-1)), np.sort(all_ids)):
            raise RuntimeError(f"internal error: {name} schedule changed the training cohort")
    return {"A": a_plan, "B": b_plan, "C": c_plan}


def require_cuda_device(device: str = "cuda:0"):
    """Select cuda:0 only when the configured RTX 3050 Laptop is visible."""
    import torch

    if device != "cuda:0":
        raise ValueError("D2-A-E002 requires cuda:0; CPU fallback is forbidden")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required for E002; no data generation or training was started")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    expected = "NVIDIA GeForce RTX 3050 Laptop GPU"
    if device_name != expected:
        raise RuntimeError(f"E002 requires {expected}, found {device_name}")
    return torch.device("cuda:0")
