"""Matched-course data planning and training utilities for D2-A-E002."""

from __future__ import annotations

from collections.abc import Mapping
import csv
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radardistill.distillation.d2a_e002_losses import d2a_e002_loss
from radardistill.distillation.task_losses import decode_task_outputs, task_loss
from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.students.task_student import TaskStudent, export_inference_model


COURSE_LEVELS = ("C0", "C1", "C2", "C3")
PILOT_ARMS = ("A", "B", "C")
PROTOCOL_VERSION = 2.1


def e002_scene_seed_streams(config: dict[str, Any]) -> dict[str, list[int]]:
    """Expand every fixed scene-generation seed/chunk into its used stream."""
    data = config["data"]
    stride = int(data["chunk_seed_stride"])
    streams: dict[str, list[int]] = {}

    def add_bases(name: str, bases: list[int], counts: list[int], chunk: int) -> None:
        for level, (base, count) in enumerate(zip(bases, counts, strict=True)):
            stream_name = f"{name}_C{level}" if len(bases) == 4 else name
            chunks = math.ceil(count / chunk)
            streams[stream_name] = [int(base) + index * stride for index in range(chunks)]

    train = data["train"]
    add_bases(
        "train",
        train["seed_bases_by_course"],
        [int(train["pairs_per_course_level"])] * 4,
        int(train["generation_chunk_size"]),
    )
    for name in ("validation", "teacher_audit"):
        section = data[name]
        add_bases(
            name,
            section["seed_bases_by_course"],
            [int(section["pairs_per_course_level"])] * 4,
            int(section["generation_chunk_size"]),
        )
    for name in ("h0_calibration", "h0_verification"):
        section = data[name]
        count = int(section["scenes"])
        chunk = int(section["generation_chunk_size"])
        streams[name] = [
            int(section["seed_base"]) + index * int(section["chunk_seed_stride"])
            for index in range(math.ceil(count / chunk))
        ]
    development = data["development"]
    dev_per_course = int(development["paired_base_scenes"]) // 4
    add_bases(
        "development",
        development["seed_bases_by_course"],
        [dev_per_course] * 4,
        int(development["generation_chunk_size"]),
    )
    boundary = data["identifiability_boundary"]
    streams["identifiability_boundary"] = [
        int(boundary["seed_base"]) + index * int(boundary["chunk_seed_stride"])
        for index in range(
            math.ceil(int(boundary["paired_base_scenes"]) / int(boundary["generation_chunk_size"]))
        )
    ]
    return streams


def validate_e002_scene_seed_streams(config: dict[str, Any]) -> None:
    streams = e002_scene_seed_streams(config)
    seen: dict[int, str] = {}
    for stream_name, seeds in streams.items():
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"E002 scene seed stream repeats within {stream_name}")
        for seed in seeds:
            previous = seen.get(seed)
            if previous is not None:
                raise ValueError(f"E002 scene seed streams overlap: {previous} and {stream_name}")
            seen[seed] = stream_name


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
    if experiment.get("id") != "D2-A-E002" or float(experiment.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("config must identify preregistered D2-A-E002 protocol v2.1")
    if experiment.get("final_blind_generation_authorized") is not False:
        raise ValueError("final blind generation is not authorized by E002 v2")
    if training.get("cuda_required") is not True or training.get("cpu_fallback") != "forbidden":
        raise ValueError("E002 training requires CUDA and forbids CPU fallback")
    if training.get("device") != "cuda:0":
        raise ValueError("E002 training device must remain cuda:0")
    if training.get("optimizer_updates_per_arm") != 1600:
        raise ValueError("E002 pilot update budget is frozen at 1600 updates per arm")
    if training.get("training_seed") != 101:
        raise ValueError("E002 pilot training seed is frozen at 101")
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
    if int(config.get("statistics", {}).get("physical_diagnostic_pairs", -1)) != 256:
        raise ValueError("E002 physical diagnostic subset is frozen at 256 paired scenes")
    sanity = training.get("micro_overfit_sanity", {})
    if (
        sanity.get("c0_pairs") != 16
        or sanity.get("optimizer_updates") != 100
        or sanity.get("effective_batch_size") != 32
        or float(sanity.get("learning_rate", -1.0)) != 1e-3
        or float(sanity.get("weight_decay", -1.0)) != 1e-4
        or float(sanity.get("gradient_clip_norm", -1.0)) != 1.0
        or float(sanity.get("final_truth_task_loss_ratio_max", -1.0)) != 0.75
        or sanity.get("use_first_rows_after_fixed_c0_assignment") is not True
        or sanity.get("supervision") != "physical_truth_only"
        or tuple(sanity.get("required_truth_gradient_groups", ()))
        != ("stem", "presence_head", "range_head", "range_offset_head", "doppler_head")
        or sanity.get("require_nonzero_k2_head_gradient_when_cache_present") is not True
    ):
        raise ValueError("E002 G1 micro-overfit and gradient sanity settings are frozen")

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
    if any(int(updates[level]) != 256 for level in COURSE_LEVELS) or int(updates["C4"]) != 576:
        raise ValueError("E002 curriculum is frozen at 256 updates for C0-C3 and 576 for C4")
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
    validate_e002_scene_seed_streams(config)


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


def run_micro_overfit_sanity(
    train_arrays: Mapping[str, np.ndarray],
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Run the frozen C0 truth-task engineering sanity check on a small cohort."""
    sanity = config["training"]["micro_overfit_sanity"]
    course = np.asarray(train_arrays["course_level"]).astype(str)
    c0_ids = np.flatnonzero(course == "C0")
    pair_count = int(sanity["c0_pairs"])
    if pair_count <= 0 or c0_ids.size < pair_count:
        raise ValueError("micro-overfit sanity requires its preregistered number of C0 pairs")
    batch = make_paired_examples(
        train_arrays,
        c0_ids[:pair_count],
        include_teacher=("teacher_response" in train_arrays),
    )
    initialization_seed = int(config["data"]["training_seeds"]["initialization"])
    torch.manual_seed(initialization_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(initialization_seed)
    model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=True,
    ).to(device)
    x = torch.from_numpy(batch["iq_ri"]).to(device)
    truth = {
        key: torch.from_numpy(batch[key]).to(device)
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    initial_teacher = (
        torch.from_numpy(batch["teacher_response"]).to(device)
        if "teacher_response" in batch
        else None
    )
    model.eval()
    outputs = model(x, compute_spectrum=("teacher_response" in batch))
    initial_terms = d2a_e002_loss(
        outputs,
        truth,
        config,
        teacher_response=initial_teacher,
        teacher_weight=0.0,
    )
    groups = {
        "stem": model.stem,
        "presence_head": model.presence_per_range,
        "range_head": model.range_head,
        "range_offset_head": model.range_offset_head,
        "doppler_head": model.doppler_head,
    }
    gradient_norms: dict[str, float] = {}
    for name, module in groups.items():
        gradients = torch.autograd.grad(
            initial_terms["task_total"],
            tuple(module.parameters()),
            retain_graph=True,
            allow_unused=True,
        )
        squares = [gradient.detach().double().square().sum() for gradient in gradients if gradient is not None]
        gradient_norms[name] = float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0
    gradient_checks_passed = all(
        np.isfinite(value) and value > 0.0 for value in gradient_norms.values()
    )
    task_head_modules = {
        "presence": model.presence_per_range,
        "range_cell": model.range_head,
        "range_offset": model.range_offset_head,
        "doppler": model.doppler_head,
    }
    single_task_gradient_norms: dict[str, dict[str, float]] = {}
    for task_name, head in task_head_modules.items():
        stem_parameters = tuple(model.stem.parameters())
        head_parameters = tuple(head.parameters())
        gradients = torch.autograd.grad(
            initial_terms[task_name],
            (*stem_parameters, *head_parameters),
            retain_graph=True,
            allow_unused=True,
        )
        gradient_parts = (gradients[: len(stem_parameters)], gradients[len(stem_parameters) :])
        norms: dict[str, float] = {}
        for group_name, group_gradients in zip(("stem", "head"), gradient_parts, strict=True):
            squares = [
                gradient.detach().double().square().sum()
                for gradient in group_gradients
                if gradient is not None
            ]
            norms[group_name] = float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0
        single_task_gradient_norms[task_name] = norms
    single_task_gradients_passed = all(
        np.isfinite(value) and value > 0.0
        for groups_by_task in single_task_gradient_norms.values()
        for value in groups_by_task.values()
    )

    k2_gradient_norm: float | None = None
    k2_gradient_passed = True
    if initial_teacher is not None:
        gradients = torch.autograd.grad(
            initial_terms["teacher_kl"],
            tuple(model.spectrum_head.parameters()),
            allow_unused=True,
        )
        squares = [gradient.detach().double().square().sum() for gradient in gradients if gradient is not None]
        k2_gradient_norm = float(torch.sqrt(torch.stack(squares).sum()).item()) if squares else 0.0
        k2_gradient_passed = np.isfinite(k2_gradient_norm) and k2_gradient_norm > 0.0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(sanity["learning_rate"]),
        weight_decay=float(sanity["weight_decay"]),
    )
    initial_loss = float(initial_terms["task_total"].detach().item())
    model.train()
    last_grad_norm = float("nan")
    for _ in range(int(sanity["optimizer_updates"])):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(x, compute_spectrum=False)
        loss = d2a_e002_loss(
            outputs,
            truth,
            config,
            teacher_response=None,
            teacher_weight=0.0,
        )["task_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite micro-overfit truth-task loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(sanity["gradient_clip_norm"])
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite micro-overfit gradient")
        last_grad_norm = float(grad_norm.detach().item())
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final_outputs = model(x, compute_spectrum=False)
        final_loss = float(
            d2a_e002_loss(
                final_outputs,
                truth,
                config,
                teacher_response=None,
                teacher_weight=0.0,
            )["task_total"].item()
        )
    ratio = final_loss / max(initial_loss, np.finfo(np.float64).tiny)
    loss_ratio_max = float(sanity["final_truth_task_loss_ratio_max"])
    loss_reduction_passed = np.isfinite(ratio) and ratio <= loss_ratio_max
    return {
        "n_paired_scenes": int(pair_count),
        "n_truth_task_examples": int(2 * pair_count),
        "optimizer_updates": int(sanity["optimizer_updates"]),
        "initial_truth_loss_components": {
            name: float(initial_terms[name].detach().item())
            for name in ("presence", "range_cell", "range_offset", "doppler", "teacher_kl")
        },
        "initial_truth_task_loss": initial_loss,
        "final_truth_task_loss": final_loss,
        "truth_task_final_to_initial_loss_ratio": float(ratio),
        "truth_task_loss_ratio_max": loss_ratio_max,
        "truth_loss_reduction_passed": bool(loss_reduction_passed),
        "truth_gradient_norms": gradient_norms,
        "single_task_gradient_norms": single_task_gradient_norms,
        "truth_gradient_groups_passed": bool(gradient_checks_passed and single_task_gradients_passed),
        "k2_gradient_norm": k2_gradient_norm,
        "k2_gradient_passed": bool(k2_gradient_passed),
        "last_training_gradient_norm": last_grad_norm,
        "passed": bool(
            loss_reduction_passed
            and gradient_checks_passed
            and single_task_gradients_passed
            and k2_gradient_passed
        ),
        "not_part_of_formal_arm_updates": True,
    }


def make_paired_examples(
    arrays: Mapping[str, np.ndarray],
    pair_ids: np.ndarray,
    *,
    include_teacher: bool,
) -> dict[str, np.ndarray]:
    """Interleave each scene's H0/H1 IQ and labels without exposing metadata."""
    ids = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    if ids.size == 0 or np.any(ids < 0) or np.any(ids >= len(arrays["iq_h0"])):
        raise ValueError("pair_ids must be nonempty indices into the paired cache")
    iq = np.empty((ids.size, 2, 3, 130, 32), dtype=np.complex64)
    iq[:, 0] = arrays["iq_h0"][ids]
    iq[:, 1] = arrays["iq_h1"][ids]
    flat_iq = iq.reshape(2 * ids.size, 3, 130, 32)
    iq_ri = np.concatenate((flat_iq.real, flat_iq.imag), axis=1).astype(np.float32, copy=False)
    presence = np.tile(np.asarray([False, True]), ids.size)
    truth_range = np.zeros(2 * ids.size, dtype=np.float32)
    truth_doppler = np.zeros(2 * ids.size, dtype=np.float32)
    truth_range[1::2] = np.asarray(arrays["truth_range_cell"][ids], dtype=np.float32)
    truth_doppler[1::2] = np.asarray(arrays["truth_doppler_hz"][ids], dtype=np.float32)
    result = {
        "iq_ri": np.ascontiguousarray(iq_ri),
        "truth_presence": presence,
        "truth_range_cell": truth_range,
        "truth_doppler_hz": truth_doppler,
    }
    if include_teacher:
        if "teacher_response" not in arrays:
            raise ValueError("C arm requires the preregistered H1 K2 response cache")
        response = np.zeros((2 * ids.size, 130, 32), dtype=np.float32)
        response[1::2] = np.asarray(arrays["teacher_response"][ids], dtype=np.float32)
        result["teacher_response"] = response
    return result


def scale_d2a_e002_microbatch_loss(
    losses: Mapping[str, torch.Tensor],
    config: dict[str, Any],
    *,
    microbatch_size: int,
    effective_batch_size: int,
    microbatch_h1: int,
    effective_h1: int,
    teacher_weight: float,
) -> torch.Tensor:
    """Weight microbatch means so accumulation equals one effective-batch loss."""
    if effective_batch_size <= 0 or not 0 < microbatch_size <= effective_batch_size:
        raise ValueError("microbatch size must lie within a positive effective batch")
    if effective_h1 <= 0 or not 0 <= microbatch_h1 <= effective_h1:
        raise ValueError("microbatch H1 count must lie within a positive effective H1 count")
    if not np.isfinite(teacher_weight) or teacher_weight < 0.0:
        raise ValueError("teacher_weight must be finite and nonnegative")
    sample_fraction = microbatch_size / effective_batch_size
    h1_fraction = microbatch_h1 / effective_h1
    weights = config["training"]["losses"]
    scaled = float(weights["presence_bce_weight"]) * losses["presence"] * sample_fraction
    scaled = scaled + h1_fraction * (
        float(weights["range_cell_cross_entropy_weight"]) * losses["range_cell"]
        + float(weights["range_offset_smooth_l1_weight"]) * losses["range_offset"]
        + float(weights["doppler_circular_cosine_weight"]) * losses["doppler"]
        + float(teacher_weight) * losses["teacher_kl"]
    )
    return scaled


def _task_config(config: dict[str, Any]) -> dict[str, Any]:
    training = dict(config["training"])
    training["loss"] = training["losses"]
    return {**config, "training": training}


def _paired_validation(
    model: TaskStudent,
    arrays: Mapping[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    microbatch_size: int,
) -> dict[str, float]:
    model.eval()
    count = len(arrays["base_scene_id"])
    if microbatch_size % 2 or microbatch_size > int(config["training"]["effective_batch_size"]):
        raise ValueError("validation microbatch must be even and no larger than effective batch")
    loss_sums = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler")}
    h1_total = 0
    example_total = 0
    h0_scores: list[np.ndarray] = []
    h1_scores: list[np.ndarray] = []
    range_estimates: list[np.ndarray] = []
    doppler_estimates: list[np.ndarray] = []
    true_cell_doppler_estimates: list[np.ndarray] = []
    range_truths: list[np.ndarray] = []
    doppler_truths: list[np.ndarray] = []
    mismatch_count = 0
    h1_count = 0
    prf_hz = float(config["radar"]["prf_hz"])
    loss_config = _task_config(config)

    with torch.no_grad():
        for pair_start in range(0, count, int(config["training"]["paired_base_scenes_per_update"])):
            pair_ids = np.arange(
                pair_start,
                min(pair_start + int(config["training"]["paired_base_scenes_per_update"]), count),
                dtype=np.int64,
            )
            batch = make_paired_examples(arrays, pair_ids, include_teacher=False)
            size = len(batch["truth_presence"])
            for start in range(0, size, microbatch_size):
                stop = min(start + microbatch_size, size)
                if (stop - start) % 2:
                    raise RuntimeError("paired validation microbatch split an H0/H1 pair")
                x = torch.from_numpy(batch["iq_ri"][start:stop]).to(device)
                truth = {
                    key: torch.from_numpy(batch[key][start:stop]).to(device)
                    for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
                }
                outputs = model(x, compute_spectrum=False)
                terms = task_loss(outputs, truth, loss_config, teacher_weight=0.0)
                micro_h1 = int(truth["truth_presence"].sum().item())
                for name in loss_sums:
                    loss_sums[name] += float(terms[name].item()) * (micro_h1 if name != "presence" else stop - start)
                h1_total += micro_h1
                example_total += stop - start
                decoded = decode_task_outputs(outputs, prf_hz)
                presence_scores = outputs["presence_logit"].detach().cpu().numpy()
                h0_scores.append(presence_scores[0::2])
                h1_scores.append(presence_scores[1::2])
                range_estimates.append(decoded["range_cell"].detach().cpu().numpy()[1::2])
                doppler_estimates.append(decoded["doppler_hz"].detach().cpu().numpy()[1::2])
                true_range = truth["truth_range_cell"].detach().cpu().numpy()[1::2]
                true_doppler = truth["truth_doppler_hz"].detach().cpu().numpy()[1::2]
                true_cells = np.floor(true_range).astype(np.int64).clip(0, 31)
                rows = torch.arange(x.shape[0], device=device)[1::2]
                selected_cells = torch.as_tensor(true_cells, device=device)
                vectors = outputs["doppler_vector"][rows, selected_cells]
                angles = torch.atan2(vectors[:, 1], vectors[:, 0])
                true_cell_frequency = torch.remainder(
                    angles * (prf_hz / (2.0 * math.pi)) + prf_hz / 2.0, prf_hz
                ) - prf_hz / 2.0
                true_cell_doppler_estimates.append(true_cell_frequency.cpu().numpy())
                range_truths.append(true_range)
                doppler_truths.append(true_doppler)
                predicted_cells = outputs["range_logits"].argmax(dim=-1)[1::2].cpu().numpy()
                mismatch_count += int(np.count_nonzero(predicted_cells != true_cells))
                h1_count += true_cells.size

    if example_total != 2 * count or h1_total != count or h1_count != count:
        raise RuntimeError("validation cache did not provide exactly one paired H0/H1 per scene")
    presence_loss = loss_sums["presence"] / example_total
    parameter_losses = {name: loss_sums[name] / h1_total for name in ("range_cell", "range_offset", "doppler")}
    weights = config["training"]["losses"]
    task_total = (
        float(weights["presence_bce_weight"]) * presence_loss
        + float(weights["range_cell_cross_entropy_weight"]) * parameter_losses["range_cell"]
        + float(weights["range_offset_smooth_l1_weight"]) * parameter_losses["range_offset"]
        + float(weights["doppler_circular_cosine_weight"]) * parameter_losses["doppler"]
    )
    h0_score = np.concatenate(h0_scores)
    h1_score = np.concatenate(h1_scores)
    range_truth = np.concatenate(range_truths)
    doppler_truth = np.concatenate(doppler_truths)
    range_error = np.concatenate(range_estimates) - range_truth
    doppler_error = (
        np.concatenate(doppler_estimates) - doppler_truth + prf_hz / 2.0
    ) % prf_hz - prf_hz / 2.0
    true_cell_doppler_error = (
        np.concatenate(true_cell_doppler_estimates) - doppler_truth + prf_hz / 2.0
    ) % prf_hz - prf_hz / 2.0
    return {
        "task_loss": float(task_total),
        "presence_loss": float(presence_loss),
        "range_cell_loss": float(parameter_losses["range_cell"]),
        "range_offset_loss": float(parameter_losses["range_offset"]),
        "doppler_loss": float(parameter_losses["doppler"]),
        "auc": float(empirical_auc(h0_score, h1_score)),
        "range_rmse_all_h1_cells": float(np.sqrt(np.mean(range_error**2))),
        "doppler_rmse_all_h1_hz": float(np.sqrt(np.mean(doppler_error**2))),
        "true_cell_doppler_rmse_hz": float(np.sqrt(np.mean(true_cell_doppler_error**2))),
        "predicted_range_cell_mismatch_rate": float(mismatch_count / h1_count),
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _save_training_checkpoint(
    path: Path,
    model: TaskStudent,
    *,
    arm: str,
    update: int,
    microbatch_size: int,
    validation: dict[str, float],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload: dict[str, Any] = {
        "experiment": "D2-A-E002",
        "protocol_version": PROTOCOL_VERSION,
        "arm": arm,
        "update": int(update),
        "microbatch_size": int(microbatch_size),
        "model_state_dict": _cpu_state_dict(model),
        "validation": dict(validation),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _train_arm_attempt(
    *,
    arm: str,
    pair_plan: np.ndarray,
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    config: dict[str, Any],
    device: torch.device,
    initial_state: dict[str, torch.Tensor],
    out_dir: Path,
    microbatch_size: int,
    a_best_validation_loss: float | None,
) -> dict[str, Any]:
    training = config["training"]
    effective_batch = int(training["effective_batch_size"])
    if microbatch_size <= 0 or effective_batch % microbatch_size or microbatch_size % 2:
        raise ValueError("microbatch must be an even divisor of the effective paired batch")
    accumulation = effective_batch // microbatch_size
    start_time = time.perf_counter()
    model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=True,
    ).to(device)
    model.load_state_dict(initial_state)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["optimizer"]["learning_rate_initial"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    updates = int(training["optimizer_updates_per_arm"])
    learning_rate_initial = float(training["optimizer"]["learning_rate_initial"])
    learning_rate_final = float(training["optimizer"]["learning_rate_final"])
    validation_interval = int(training["validation_interval_updates"])
    best_validation = float("inf")
    best_update = 0
    best_validation_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    updates_to_a_best: int | None = None
    history_path = out_dir / "training_history.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    initial_validation = _paired_validation(
        model, validation_arrays, config, device, microbatch_size
    )
    best_validation = float(initial_validation["task_loss"])
    best_validation_metrics = initial_validation
    best_state = _cpu_state_dict(model)
    last_validation_metrics = initial_validation
    _save_training_checkpoint(
        out_dir / "initial_checkpoint.pt",
        model,
        arm=arm,
        update=0,
        microbatch_size=microbatch_size,
        validation=initial_validation,
    )

    history_fields = [
        "update", "arm", "teacher_weight", "learning_rate", "train_total",
        "train_presence", "train_range_cell", "train_range_offset", "train_doppler",
        "train_teacher_kl", "validation_task_loss", "validation_auc",
        "validation_presence_loss", "validation_range_cell_loss",
        "validation_range_offset_loss", "validation_doppler_loss",
        "validation_range_rmse_all_h1_cells", "validation_doppler_rmse_all_h1_hz",
        "validation_true_cell_doppler_rmse_hz", "validation_predicted_range_cell_mismatch_rate",
    ]
    model.train()
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=history_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({"update": 0, "arm": arm, **{f"validation_{key}": value for key, value in initial_validation.items()}})
        history_file.flush()
        for update_index, pair_ids in enumerate(pair_plan):
            model.train()
            fraction = update_index / max(updates - 1, 1)
            learning_rate = learning_rate_final + 0.5 * (learning_rate_initial - learning_rate_final) * (
                1.0 + math.cos(math.pi * fraction)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            teacher_weight = (
                float(training["k2_loss"]["lambda_initial"]) * (1.0 - fraction)
                if arm == "C"
                else 0.0
            )
            batch = make_paired_examples(
                train_arrays,
                pair_ids,
                include_teacher=(arm == "C"),
            )
            optimizer.zero_grad(set_to_none=True)
            terms_sum = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "teacher_kl")}
            for micro_start in range(0, effective_batch, microbatch_size):
                micro_stop = micro_start + microbatch_size
                x = torch.from_numpy(batch["iq_ri"][micro_start:micro_stop]).to(device)
                truth = {
                    key: torch.from_numpy(batch[key][micro_start:micro_stop]).to(device)
                    for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
                }
                teacher = None
                if arm == "C":
                    teacher = torch.from_numpy(
                        batch["teacher_response"][micro_start:micro_stop]
                    ).to(device)
                outputs = model(x, compute_spectrum=(arm == "C"))
                losses = d2a_e002_loss(
                    outputs,
                    truth,
                    config,
                    teacher_response=teacher,
                    teacher_weight=teacher_weight,
                )
                if any(not torch.isfinite(value).all() for value in losses.values()):
                    raise FloatingPointError(f"non-finite E002 loss in arm {arm}, update {update_index + 1}")
                micro_h1 = int(truth["truth_presence"].sum().item())
                scaled = scale_d2a_e002_microbatch_loss(
                    losses,
                    config,
                    microbatch_size=microbatch_size,
                    effective_batch_size=effective_batch,
                    microbatch_h1=micro_h1,
                    effective_h1=effective_batch // 2,
                    teacher_weight=teacher_weight if arm == "C" else 0.0,
                )
                scaled.backward()
                terms_sum["presence"] += float(losses["presence"].detach().item()) / accumulation
                h1_fraction = micro_h1 / (effective_batch // 2)
                for name in ("range_cell", "range_offset", "doppler", "teacher_kl"):
                    terms_sum[name] += float(losses[name].detach().item()) * h1_fraction

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["optimizer"]["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite E002 gradient in arm {arm}, update {update_index + 1}")
            optimizer.step()
            row: dict[str, Any] = {
                "update": update_index + 1,
                "arm": arm,
                "teacher_weight": teacher_weight,
                "learning_rate": learning_rate,
                "train_presence": terms_sum["presence"],
                "train_range_cell": terms_sum["range_cell"],
                "train_range_offset": terms_sum["range_offset"],
                "train_doppler": terms_sum["doppler"],
                "train_teacher_kl": terms_sum["teacher_kl"],
                "train_total": (
                    float(training["losses"]["presence_bce_weight"]) * terms_sum["presence"]
                    + float(training["losses"]["range_cell_cross_entropy_weight"]) * terms_sum["range_cell"]
                    + float(training["losses"]["range_offset_smooth_l1_weight"]) * terms_sum["range_offset"]
                    + float(training["losses"]["doppler_circular_cosine_weight"]) * terms_sum["doppler"]
                    + teacher_weight * terms_sum["teacher_kl"]
                ),
            }
            completed = update_index + 1
            should_validate = completed % validation_interval == 0 or completed == updates
            if should_validate:
                validation = _paired_validation(
                    model, validation_arrays, config, device, microbatch_size
                )
                last_validation_metrics = validation
                row.update({f"validation_{key}": value for key, value in validation.items()})
                current_validation = float(validation["task_loss"])
                if current_validation < best_validation:
                    best_validation = current_validation
                    best_update = completed
                    best_validation_metrics = validation
                    best_state = _cpu_state_dict(model)
                    if arm in {"B", "C"} and a_best_validation_loss is not None and updates_to_a_best is None:
                        if current_validation <= a_best_validation_loss:
                            updates_to_a_best = completed
                    _save_training_checkpoint(
                        out_dir / "best_checkpoint.pt",
                        model,
                        arm=arm,
                        update=completed,
                        microbatch_size=microbatch_size,
                        validation=validation,
                    )
                if completed % 100 == 0 or completed == updates:
                    _save_training_checkpoint(
                        out_dir / "progress_checkpoint.pt",
                        model,
                        arm=arm,
                        update=completed,
                        microbatch_size=microbatch_size,
                        validation=validation,
                        optimizer=optimizer,
                    )
            writer.writerow(row)
            history_file.flush()

    if best_state is None:
        raise RuntimeError("E002 training did not produce a best checkpoint")
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start_time
    _save_training_checkpoint(
        out_dir / "last_checkpoint.pt",
        model,
        arm=arm,
        update=updates,
        microbatch_size=microbatch_size,
        validation=last_validation_metrics,
    )
    model.load_state_dict(best_state)
    _save_training_checkpoint(
        out_dir / "best_checkpoint.pt",
        model,
        arm=arm,
        update=best_update,
        microbatch_size=microbatch_size,
        validation=best_validation_metrics,
    )
    inference_model = export_inference_model(model).cpu()
    torch.save(
        {
            "experiment": "D2-A-E002",
            "protocol_version": PROTOCOL_VERSION,
            "arm": arm,
            "update": best_update,
            "model_state_dict": _cpu_state_dict(inference_model),
            "has_k2_head": False,
        },
        out_dir / "best_inference.pt",
    )
    return {
        "arm": arm,
        "optimizer_updates": updates,
        "best_update": best_update,
        "updates_to_A_best_validation_loss": updates_to_a_best,
        "initial_validation": initial_validation,
        "best_validation": best_validation_metrics,
        "best_validation_task_loss": best_validation,
        "train_seconds": train_seconds,
        "effective_examples_per_second": float(effective_batch * updates / train_seconds),
        "microbatch_size": microbatch_size,
        "effective_batch_size": effective_batch,
        "parameter_count_training_model": int(sum(parameter.numel() for parameter in model.parameters())),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "training_history": str(history_path),
        "best_checkpoint": str(out_dir / "best_checkpoint.pt"),
        "best_inference_checkpoint": str(out_dir / "best_inference.pt"),
    }


def train_d2a_e002_pilot(
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    config: dict[str, Any],
    output_root: str | Path,
    *,
    device_name: str = "cuda:0",
    arms: tuple[str, ...] = ("A", "B", "C"),
) -> dict[str, Any]:
    """Train selected A/B/C pilot arms serially on the required RTX 3050 CUDA."""
    validate_e002_config(config)
    device = require_cuda_device(device_name)
    if not arms or any(arm not in PILOT_ARMS for arm in arms) or tuple(arm for arm in PILOT_ARMS if arm in arms) != arms:
        raise ValueError("pilot arms must be a nonempty ordered subset of A, B, C")
    if "C" in arms and "teacher_response" not in train_arrays:
        raise ValueError("C arm requires the preregistered train K2 cache")
    train_count = len(train_arrays["base_scene_id"])
    course_labels = np.asarray(train_arrays["course_level"]).astype(str)
    course_ids = {
        level: np.flatnonzero(course_labels == level).astype(np.int64)
        for level in COURSE_LEVELS
    }
    if any(ids.size != int(config["data"]["train"]["pairs_per_course_level"]) for ids in course_ids.values()):
        raise ValueError("cached train data do not match the frozen per-course pair counts")
    plans = build_matched_pair_plans(
        course_ids,
        updates_per_course=int(config["training"]["curriculum_updates"]["C0"]),
        mixed_updates=int(config["training"]["curriculum_updates"]["C4"]),
        batch_pairs=int(config["training"]["paired_base_scenes_per_update"]),
        seed=int(config["data"]["training_seeds"]["batch_plan"]),
    )
    expected_train_count = int(config["data"]["train"]["paired_base_scenes"])
    if train_count != expected_train_count:
        raise ValueError("train cache size differs from the preregistered 51,200 pairs")

    torch.manual_seed(int(config["data"]["training_seeds"]["initialization"]))
    torch.cuda.manual_seed_all(int(config["data"]["training_seeds"]["initialization"]))
    initial_model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=True,
    )
    initial_state = _cpu_state_dict(initial_model)
    output_root = Path(output_root)
    result: dict[str, Any] = {
        "experiment": "D2-A-E002",
        "protocol_version": PROTOCOL_VERSION,
        "training_seed": int(config["training"]["training_seed"]),
        "plan_seed": int(config["data"]["training_seeds"]["batch_plan"]),
        "initialization_seed": int(config["data"]["training_seeds"]["initialization"]),
        "arms": {},
    }
    a_best: float | None = None
    base_microbatch = int(config["training"]["initial_microbatch_size"])
    fallbacks = [int(value) for value in config["training"]["oom_microbatch_fallbacks"]]
    for arm in arms:
        success: dict[str, Any] | None = None
        for microbatch in (base_microbatch, *fallbacks):
            arm_dir = output_root / f"seed_101" / arm / f"microbatch_{microbatch}"
            try:
                torch.manual_seed(int(config["training"]["training_seed"]))
                torch.cuda.manual_seed_all(int(config["training"]["training_seed"]))
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                success = _train_arm_attempt(
                    arm=arm,
                    pair_plan=plans[arm],
                    train_arrays=train_arrays,
                    validation_arrays=validation_arrays,
                    config=config,
                    device=device,
                    initial_state=initial_state,
                    out_dir=arm_dir,
                    microbatch_size=microbatch,
                    a_best_validation_loss=a_best,
                )
                break
            except torch.cuda.OutOfMemoryError:
                if microbatch == fallbacks[-1]:
                    raise
                torch.cuda.empty_cache()
                import gc

                gc.collect()
                continue
        if success is None:
            raise RuntimeError(f"E002 arm {arm} did not complete")
        result["arms"][arm] = success
        if arm == "A":
            a_best = float(success["best_validation_task_loss"])
        torch.cuda.empty_cache()
    return result
