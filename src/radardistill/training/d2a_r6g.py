"""Matched teacher-content pilot for D2-A-R6-G."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

from radardistill.distillation.r5c_teaching import truth_dirichlet_targets
from radardistill.evaluation.d2a_r6a import build_split_plan
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.training.d2a_r5c_teaching_pilot import compute_h1_teacher_maps
from radardistill.training.d2a_r6d_joint import (
    audit_r6d_scene_plans,
    build_r6d_paired_plan,
    r6d_learning_rate,
    r6d_loss,
)
from radardistill.training.d2a_r6b_hard_negative import (
    assemble_independent_h0_h1_batch,
    build_h0_sample_plan,
    build_repeated_h1_plan,
)
from radardistill.students.r6g_student import R6GCEStudent
from radardistill.students.r6g_student import export_r6g_inference_model


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-G"
CACHE = STAGE / "cache/training"
RUNS = STAGE / "runs/seed_707"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"


def _expect(section: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    if section.get(key) != expected:
        raise ValueError(f"{label}.{key} must equal {expected!r}")


def validate_r6g_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the frozen R6-G pilot's scientific design."""
    if not isinstance(config, Mapping):
        raise ValueError("R6-G config must be a mapping")
    sections = ("experiment", "radar", "student", "teacher", "data", "training", "evaluation", "resources")
    for name in sections:
        if not isinstance(config.get(name), Mapping):
            raise ValueError(f"R6-G config section {name} must be a mapping")

    experiment = config["experiment"]
    for key, expected in {
        "id": "D2-A-R6",
        "subphase": "R6-G",
        "protocol_version": "1.0.0",
        "execution_authorized": True,
        "execution_gate": "implement_G0_G1_then_train",
        "final_blind_authorized": False,
        "status": "preregistered",
    }.items():
        _expect(experiment, key, expected, "experiment")

    radar = config["radar"]
    for key, expected in {
        "channel_count": 3,
        "prf_hz": 1300.0,
        "pulse_count": 130,
        "range_bin_count": 32,
        "clutter": "stationary",
        "condition_count": 12,
    }.items():
        _expect(radar, key, expected, "radar")

    _expect(config, "arms", ["T0", "T1", "T3"], "config")
    student = config["student"]
    for key, expected in {
        "architecture": "complex_equivariant_residual_tcn",
        "source_architecture": "R6ECEStudent",
        "width": 64,
        "inference_parameters_expected": 63429,
    }.items():
        _expect(student, key, expected, "student")
    initialization = student.get("initialization")
    if not isinstance(initialization, Mapping):
        raise ValueError("student.initialization must be a mapping")
    for key, expected in {
        "seed": 707,
        "mode": "from_scratch",
        "clone_bit_identical_state_across_arms": True,
        "warm_start_checkpoint": None,
    }.items():
        _expect(initialization, key, expected, "student.initialization")

    teacher = config["teacher"]
    for key, expected in {
        "arm": "T1_only",
        "method": "K3_B3_DPCA_complex128_cuda_4x",
        "device": "cuda:0",
        "teacher_inputs": ["observable_H1_IQ", "frozen_physics_config"],
        "response_shape": [520, 32],
        "storage_dtype": "float16",
        "compute_only_for": "H1",
        "hard_peak_used_for_training": False,
        "truth_or_snr_cnr_input": False,
        "reliability_gating": False,
    }.items():
        _expect(teacher, key, expected, "teacher")

    data = config["data"]
    for key, expected in {
        "training_seed": 707,
        "sampler_seed": 708,
        "reuse_any_existing_scene_ids": False,
        "same_scene_order_across_arms": True,
        "persist_evaluation_iq": False,
    }.items():
        _expect(data, key, expected, "data")
    splits = data.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("data.splits must be a mapping")
    expected_split_sizes = {
        "train": (51200, None),
        "validation": (5000, None),
        "h0_calibration": (120000, 10000),
        "h0_verification": (120000, 10000),
        "h1_development": (6144, 512),
    }
    for split_name, (count, per_condition) in expected_split_sizes.items():
        spec = splits.get(split_name)
        if not isinstance(spec, Mapping):
            raise ValueError(f"data.splits.{split_name} must be a mapping")
        count_key = "paired_scenes" if split_name in {"train", "validation"} else "scene_count"
        _expect(spec, count_key, count, f"data.splits.{split_name}")
        if per_condition is not None:
            _expect(spec, "count_per_condition" if split_name.startswith("h0_") else "count_per_condition", per_condition, f"data.splits.{split_name}")

    training = config["training"]
    for key, expected in {
        "optimizer": "AdamW",
        "learning_rate_initial": 0.001,
        "learning_rate_final": 0.00001,
        "learning_rate_schedule": "single_cosine_over_all_updates",
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "effective_batch_size_cpi": 64,
        "per_update_examples": {"H0": 32, "H1": 32},
        "fixed_updates_per_arm": 51200,
        "validation_interval_updates": 400,
        "resume_checkpoint_interval_updates": 800,
        "history_interval_updates": 400,
        "history_written_every_update": False,
        "checkpoint_selection": "minimum_common_validation_truth_task_loss",
        "microbatch_initial": 4,
        "microbatch_oom_fallbacks": [2, 1],
        "preserve_effective_batch_on_oom": True,
        "precision": "FP32",
        "amp": False,
        "arm_order": ["T0", "T1", "T3"],
        "teacher_loss_weight": 0.05,
        "teacher_loss_rows": "H1_only",
        "t0_teacher_map_computed": False,
        "t3_teacher_map_computed": False,
        "t3_truth_response_used": True,
    }.items():
        _expect(training, key, expected, "training")

    evaluation = config["evaluation"]
    for key, expected in {
        "target_pfa_cpi": 0.05,
        "range_gate_cells": 1.0,
        "doppler_gate_hz": 20.0,
        "decision_comparator": ">=",
        "c3_familywise_alpha": 0.05,
        "c3_per_cell_upper_maximum": 0.10,
        "bootstrap_replicates": 5000,
        "bootstrap_seed": 2026110401,
        "primary_arm_contrast": "T1_minus_T0",
        "primary_scope": "identifiable_C2_C3_joint_success",
        "minimum_primary_gain": 0.02,
        "primary_ci": "two_sided_95_percent",
        "final_blind_authorized": False,
    }.items():
        _expect(evaluation, key, expected, "evaluation")

    resources = config["resources"]
    for key, expected in {
        "required_gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "required_device": "cuda:0",
        "simultaneous_models_resident": 1,
        "teacher_targets_computed_on_cuda": True,
        "target_generation_batch_size": 16,
    }.items():
        _expect(resources, key, expected, "resources")


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read_yaml(STAGE / "config.yaml")
    r6a = _read_yaml(ROOT / config["radar"]["condition_source"])
    physics = _read_yaml(ROOT / config["radar"]["physics_source"])
    return config, r6a, physics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def build_r6g_plans(
    config: Mapping[str, Any], r6a: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Build the exact R6-G train, validation, evaluation, and G1 scene plans."""
    conditions = list(r6a["radar"]["conditions"])
    data = config["data"]
    plans: dict[str, list[dict[str, Any]]] = {
        "train": build_r6d_paired_plan(data, conditions, "train"),
        "validation": build_r6d_paired_plan(data, conditions, "validation"),
    }
    evaluation_data = {
        key: data[key]
        for key in ("condition_seed_stride", "chunk_size", "chunk_seed_stride")
    }
    evaluation_data["condition_count"] = int(config["radar"]["condition_count"])
    evaluation_names = ("h0_calibration", "h0_verification", "h1_development")
    evaluation_data["splits"] = {name: data["splits"][name] for name in evaluation_names}
    evaluation_config = {
        "data": evaluation_data,
        "radar": {"conditions": conditions},
    }
    plans.update({name: build_split_plan(evaluation_config, name) for name in evaluation_names})

    g1_conditions = [
        (index, condition)
        for index, condition in enumerate(conditions)
        if condition.get("course") == "C3"
        and condition.get("scenario_id") == "partial_overlap"
        and float(condition.get("snr_db")) == 0.0
        and float(condition.get("cnr_db")) == 20.0
    ]
    if len(g1_conditions) != 1:
        raise ValueError("R6-G requires exactly one registered C3 partial-overlap 0/20 dB G1 condition")
    condition_index, condition = g1_conditions[0]
    plans["g1_test"] = [{
        "split": "g1_test",
        "presence": "H1",
        "condition_index": int(condition_index),
        "condition": dict(condition),
        "count": 16,
        "seed": int(data["g1_test_seed"]),
    }]
    return plans


def build_r6g_sampler_plans(
    condition_indices: np.ndarray,
    *,
    updates: int,
    effective_batch: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build a fixed H1 epoch plan and within-condition uniform H0 plan."""
    conditions = np.asarray(condition_indices)
    if conditions.ndim != 1 or not len(conditions) or not np.issubdtype(conditions.dtype, np.integer):
        raise ValueError("R6-G train conditions must be a nonempty integer vector")
    if isinstance(effective_batch, bool) or not isinstance(effective_batch, int) or effective_batch <= 0 or effective_batch % 2:
        raise ValueError("R6-G effective_batch must be a positive even integer")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates <= 0 or seed < 0:
        raise ValueError("R6-G updates and sampler seed must be valid")
    h1_per_update = effective_batch // 2
    h1_plan = build_repeated_h1_plan(
        conditions, updates=updates, h1_per_update=h1_per_update, seed=seed
    )
    h0_plan, h0_audit = build_h0_sample_plan(
        h1_plan,
        conditions,
        conditions,
        {},
        mode="uniform",
        seed=seed + 1,
        hard_fraction=0.0,
    )
    if not np.array_equal(conditions[h0_plan], conditions[h1_plan]):
        raise RuntimeError("R6-G H0 sampler crossed registered condition groups")
    exposure = updates * h1_per_update
    if exposure % len(conditions):
        raise RuntimeError("R6-G H1 plan does not provide equal whole-cohort exposure")
    audit = {
        "updates": updates,
        "effective_batch": effective_batch,
        "h1_per_update": h1_per_update,
        "h1_exposures": exposure,
        "h1_scene_visits_per_presence": exposure // len(conditions),
        "h0_condition_matched": True,
        "h0_sampling": h0_audit,
        "h1_plan_sha256": hashlib.sha256(h1_plan.tobytes()).hexdigest(),
        "h0_plan_sha256": hashlib.sha256(h0_plan.tobytes()).hexdigest(),
    }
    return h1_plan, h0_plan, audit


def audit_r6g_plans(
    plans: Mapping[str, list[Mapping[str, Any]]],
    *,
    expected_counts: Mapping[str, int],
    prior_scene_ids: set[str],
) -> dict[str, Any]:
    """Reject duplicates, cross-split reuse, and any historical scene reuse."""
    audit = audit_r6d_scene_plans(
        plans,
        expected_counts=expected_counts,
        prior_scene_ids=prior_scene_ids,
    )
    return {
        "scene_counts": audit["scene_counts"],
        "cross_split_overlap_count": audit["overlap_count"],
        "prior_scene_overlap_count": audit["prior_overlap_count"],
        "scene_ids": audit["scene_ids"],
    }


def validate_r6g_target_maps(
    maps: np.ndarray | torch.Tensor,
    *,
    expected_count: int | None = None,
    sum_tolerance: float = 3e-3,
) -> None:
    """Validate cached normalized nonnegative [B,520,32] response maps."""
    values = maps.detach().cpu().numpy() if torch.is_tensor(maps) else np.asarray(maps)
    if values.ndim != 3 or values.shape[1:] != (520, 32):
        raise ValueError("R6-G response maps must have shape [B,520,32]")
    if values.shape[0] <= 0 or (expected_count is not None and values.shape[0] != expected_count):
        raise ValueError("R6-G response map count differs from its registered batch")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("R6-G response maps must be finite floating-point values")
    if np.any(values < 0):
        raise ValueError("R6-G response maps must be nonnegative")
    totals = values.astype(np.float64, copy=False).sum(axis=(1, 2))
    if not np.all(np.abs(totals - 1.0) <= sum_tolerance):
        raise ValueError("R6-G response maps must be normalized to unit mass")


def build_r6g_t3_maps(
    truth_range_cell: torch.Tensor,
    truth_doppler_hz: torch.Tensor,
    *,
    device: torch.device,
    prf_hz: float = 1300.0,
    pulse_count: int = 130,
) -> torch.Tensor:
    """Generate the T3 finite-CPI response on the truth range gate."""
    ranges = torch.as_tensor(truth_range_cell, dtype=torch.float32, device=device)
    dopplers = torch.as_tensor(truth_doppler_hz, dtype=torch.float32, device=device)
    range_indices, frequency_response = truth_dirichlet_targets(
        ranges, dopplers, prf_hz=prf_hz, pulse_count=pulse_count
    )
    result = torch.zeros(
        (len(ranges), 520, 32), dtype=frequency_response.dtype, device=device
    )
    rows = torch.arange(len(ranges), device=device)
    result[rows, :, range_indices] = frequency_response
    if not bool(torch.isfinite(result).all()) or bool(torch.any(result < 0)):
        raise FloatingPointError("R6-G T3 target contains invalid values")
    return result


def r6g_response_map_kl(student_logits: torch.Tensor, target_map: torch.Tensor) -> torch.Tensor:
    """KL divergence from a cached normalized [B,520,32] response map."""
    if not isinstance(student_logits, torch.Tensor) or not isinstance(target_map, torch.Tensor):
        raise ValueError("R6-G logits and target maps must be torch tensors")
    expected_shape = (student_logits.shape[0], 520, 32) if student_logits.ndim == 3 else None
    if expected_shape is None or student_logits.shape != expected_shape or target_map.shape != expected_shape:
        raise ValueError("R6-G logits and target maps must share shape [B,520,32]")
    if not torch.is_floating_point(student_logits) or not torch.is_floating_point(target_map):
        raise ValueError("R6-G logits and target maps must be floating point")
    if not bool(torch.isfinite(student_logits).all()) or not bool(torch.isfinite(target_map).all()):
        raise ValueError("R6-G logits and target maps must be finite")
    if bool(torch.any(target_map < 0.0)):
        raise ValueError("R6-G target maps must be nonnegative")
    target = target_map.to(device=student_logits.device, dtype=student_logits.dtype)
    totals = target.sum(dim=(1, 2), keepdim=True)
    if bool(torch.any(totals <= 0.0)):
        raise ValueError("each R6-G target map must have positive total mass")
    target = target / totals
    target = target.clamp_min(1e-12)
    target = target / target.sum(dim=(1, 2), keepdim=True)
    target_flat = target.reshape(target.shape[0], -1)
    target_log = torch.log(target_flat)
    student_log = F.log_softmax(student_logits.reshape(student_logits.shape[0], -1), dim=-1)
    return torch.sum(target_flat * (target_log - student_log), dim=-1).mean()


def _paired_iq_and_truth(generated: Any) -> dict[str, np.ndarray]:
    paired = np.stack((generated.iq_h0, generated.iq_h1), axis=1)
    flat = paired.reshape(-1, 3, 130, 32)
    iq_ri = np.concatenate((flat.real, flat.imag), axis=1).astype(np.float32, copy=False)
    presence = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), len(generated.base_scene_id))
    range_cell = np.zeros(len(presence), dtype=np.float32)
    doppler_hz = np.zeros(len(presence), dtype=np.float32)
    range_cell[1::2] = np.asarray(generated.truth_target_range_cell, dtype=np.float32)
    doppler_hz[1::2] = np.asarray(generated.truth_target_doppler_hz, dtype=np.float32)
    return {
        "iq_ri": np.ascontiguousarray(iq_ri),
        "truth_presence": presence,
        "truth_range_cell": range_cell,
        "truth_doppler_hz": doppler_hz,
    }


def _r6g_g1_teacher_maps(
    generated: Any,
    physics: Mapping[str, Any],
    device: torch.device,
    *,
    batch_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    k3_chunks: list[torch.Tensor] = []
    t3_chunks: list[torch.Tensor] = []
    for start in range(0, len(generated.base_scene_id), batch_size):
        stop = min(start + batch_size, len(generated.base_scene_id))
        # The K3 call intentionally receives only observable H1 IQ and frozen physics.
        k3 = compute_h1_teacher_maps(generated.iq_h1[start:stop], physics, device)
        validate_r6g_target_maps(k3, expected_count=stop - start)
        ranges = torch.as_tensor(generated.truth_target_range_cell[start:stop], device=device)
        dopplers = torch.as_tensor(generated.truth_target_doppler_hz[start:stop], device=device)
        t3 = build_r6g_t3_maps(ranges, dopplers, device=device)
        validate_r6g_target_maps(t3, expected_count=stop - start)
        k3_chunks.append(k3.detach())
        t3_chunks.append(t3.detach())
    return torch.cat(k3_chunks, dim=0), torch.cat(t3_chunks, dim=0)


def _r6g_g1_arm_probe(
    arm: str,
    state: Mapping[str, torch.Tensor],
    batch: Mapping[str, np.ndarray],
    k3_maps: torch.Tensor,
    t3_maps: torch.Tensor,
    *,
    microbatch: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = R6GCEStudent(width=int(config["student"]["width"])).to(device)
    model.load_state_dict(state, strict=True)
    model.train()
    inputs = torch.from_numpy(batch["iq_ri"])
    labels = {
        key: torch.from_numpy(batch[key])
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    model.zero_grad(set_to_none=True)
    effective_count = len(inputs)
    if effective_count % microbatch:
        raise ValueError("R6-G G1 microbatch must divide its physical probe batch")
    loss_total = 0.0
    for start in range(0, effective_count, microbatch):
        stop = start + microbatch
        x = inputs[start:stop].to(device)
        truth = {key: value[start:stop].to(device) for key, value in labels.items()}
        outputs = model(x)
        truth_loss = r6d_loss(outputs, truth, arm="j1_joint", prf_hz=1300.0)["total"]
        h1_mask = truth["truth_presence"].bool()
        auxiliary = truth_loss * 0.0
        h1_offset = int(np.count_nonzero(batch["truth_presence"][:start]))
        h1_count = int(h1_mask.sum().item())
        if arm != "T0" and h1_count:
            if arm == "T1":
                target = k3_maps[h1_offset:h1_offset + h1_count]
            else:
                target = t3_maps[h1_offset:h1_offset + h1_count]
            auxiliary = r6g_response_map_kl(outputs["spectrum_logits"][h1_mask], target)
        loss = truth_loss + float(config["training"]["teacher_loss_weight"]) * auxiliary
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"R6-G {arm} G1 loss is nonfinite")
        (loss * (microbatch / effective_count)).backward()
        loss_total += float(loss.detach().item()) * (microbatch / effective_count)
    task_gradients = [
        parameter.grad for name, parameter in model.named_parameters()
        if not name.startswith("training_response_head.") and parameter.grad is not None
    ]
    if not task_gradients or any(not bool(torch.isfinite(grad).all()) for grad in task_gradients):
        raise FloatingPointError(f"R6-G {arm} task gradients are missing or nonfinite")
    task_gradient_norm = float(torch.sqrt(sum(grad.detach().square().sum() for grad in task_gradients)).item())
    if task_gradient_norm <= 0.0:
        raise FloatingPointError(f"R6-G {arm} task gradients are zero")
    response_gradients = [parameter.grad for parameter in model.training_response_head.parameters()]
    response_finite = all(grad is None or bool(torch.isfinite(grad).all()) for grad in response_gradients)
    response_norm = float(torch.sqrt(sum(
        grad.detach().square().sum() for grad in response_gradients if grad is not None
    )).item()) if any(grad is not None for grad in response_gradients) else 0.0
    expected_response_gradient = arm in {"T1", "T3"}
    if not response_finite or (expected_response_gradient and response_norm <= 0.0):
        raise FloatingPointError(f"R6-G {arm} response head gradient failed its G1 check")
    if arm == "T0" and response_norm != 0.0:
        raise RuntimeError("R6-G T0 unexpectedly used an auxiliary response target")
    record = {
        "loss": loss_total,
        "task_gradient_finite": True,
        "task_gradient_norm": task_gradient_norm,
        "response_gradient_norm": response_norm,
        "response_gradient_expected": expected_response_gradient,
        "inference_parameters": sum(
            parameter.numel() for name, parameter in model.named_parameters()
            if not name.startswith("training_response_head.")
        ),
        "training_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model
    torch.cuda.empty_cache()
    return record


def _r6g_g1_effective_batch_smoke(
    generated: Any,
    initial_state: Mapping[str, torch.Tensor],
    k3_maps: torch.Tensor,
    t3_maps: torch.Tensor,
    *,
    microbatch: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    data = {
        "iq_h0": generated.iq_h0,
        "iq_h1": generated.iq_h1,
        "truth_range_cell": np.asarray(generated.truth_target_range_cell, dtype=np.float32),
        "truth_doppler_hz": np.asarray(generated.truth_target_doppler_hz, dtype=np.float32),
    }
    rng = np.random.default_rng(int(config["data"]["sampler_seed"]))
    h0_indices = rng.integers(0, len(generated.base_scene_id), size=32, dtype=np.int64)
    h1_indices = rng.integers(0, len(generated.base_scene_id), size=32, dtype=np.int64)
    batch = assemble_independent_h0_h1_batch(data, h0_indices, h1_indices)
    targets = {
        "k3": k3_maps.detach().cpu().numpy().astype(np.float16),
        "t3": t3_maps.detach().cpu().numpy().astype(np.float16),
    }
    results: dict[str, Any] = {}
    for arm in ("T0", "T1", "T3"):
        model = R6GCEStudent(width=int(config["student"]["width"])).to(device)
        model.load_state_dict(initial_state, strict=True)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate_initial"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        torch.cuda.reset_peak_memory_stats(device)
        try:
            terms = _r6g_training_update(
                model, optimizer, batch, h1_indices, arm=arm,
                target_maps=None if arm == "T0" else targets,
                microbatch=microbatch,
                effective_batch=int(config["training"]["effective_batch_size_cpi"]),
                device=device, config=config,
            )
            results[arm] = {
                "finite_losses": all(np.isfinite(value) for value in terms.values()),
                "truth_loss": terms["truth"],
                "auxiliary_loss": terms["auxiliary"],
                "total_loss": terms["total"],
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        finally:
            del model, optimizer
            torch.cuda.empty_cache()
    return results


def r6g_run_g1() -> dict[str, Any]:
    """G1: run the physical-to-CUDA target and student-gradient smoke test."""
    g0, _ = _require_g0()
    start_time = time.perf_counter()
    config, r6a, physics = _configs()
    device, gpu, _, _ = _require_cuda(config)
    plan = build_r6g_plans(config, r6a)["g1_test"][0]
    condition = plan["condition"]
    generated = generate_d2a_paired_batch(
        int(plan["count"]), int(plan["seed"]), float(condition["snr_db"]),
        float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
    )
    if len(generated.base_scene_id) != 16:
        raise RuntimeError("R6-G G1 simulator returned an unexpected scene count")
    batch = _paired_iq_and_truth(generated)
    k3_maps, t3_maps = _r6g_g1_teacher_maps(generated, physics, device)
    if k3_maps.device != device or t3_maps.device != device:
        raise RuntimeError("R6-G G1 teacher targets did not remain on CUDA")

    seed = int(config["student"]["initialization"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    template = R6GCEStudent(width=int(config["student"]["width"]))
    initial_state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
    del template
    parameter_count = sum(
        value.numel() for key, value in initial_state.items()
        if not key.startswith("training_response_head.")
    )
    expected_parameters = int(config["student"]["inference_parameters_expected"])
    if parameter_count != expected_parameters:
        raise RuntimeError(f"R6-G inference parameter count {parameter_count} != {expected_parameters}")

    reports: dict[str, Any] = {}
    formal_microbatch = None
    for candidate in [int(config["training"]["microbatch_initial"]), *map(int, config["training"]["microbatch_oom_fallbacks"])]:
        if len(batch["iq_ri"]) % candidate:
            continue
        reports[str(candidate)] = {}
        try:
            for arm in config["training"]["arm_order"]:
                torch.cuda.reset_peak_memory_stats(device)
                arm_report = _r6g_g1_arm_probe(
                    arm, initial_state, batch, k3_maps, t3_maps,
                    microbatch=candidate, device=device, config=config,
                )
                arm_report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                arm_report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
                reports[str(candidate)][arm] = arm_report
            reports[str(candidate)]["effective_batch_64_smoke"] = _r6g_g1_effective_batch_smoke(
                generated, initial_state, k3_maps, t3_maps,
                microbatch=candidate, device=device, config=config,
            )
            formal_microbatch = candidate
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            reports[str(candidate)]["status"] = "oom"
    if formal_microbatch is None:
        raise RuntimeError("R6-G cannot complete its G1 physical batch even at microbatch 1")
    record = {
        "status": "G1_PASS",
        "git_sha": g0["git_sha"],
        "protocol_sha256": g0["protocol_sha256"],
        "config_sha256": g0["config_sha256"],
        "gpu": gpu,
        "device": str(device),
        "precision": "FP32",
        "physical_test_seed": int(plan["seed"]),
        "physical_test_scene_ids": [str(value) for value in generated.base_scene_id],
        "condition": dict(condition),
        "teacher_map_shape": list(k3_maps.shape),
        "teacher_map_normalized": True,
        "t3_map_shape": list(t3_maps.shape),
        "t3_map_normalized": True,
        "inference_parameters": parameter_count,
        "formal_microbatch": formal_microbatch,
        "arm_probes": reports,
        "paired_scene_order_shared": True,
        "teacher_k3_truth_inputs": False,
        "h0_auxiliary_targets_computed": False,
        "seconds": time.perf_counter() - start_time,
    }
    _write_json(STAGE / "g1.json", record)
    return record


def _prior_scene_ids() -> set[str]:
    """Read historical scene IDs from experiment manifests, excluding this stage."""
    result: set[str] = set()
    for path in sorted((ROOT / "experiments").rglob("*.csv")):
        if "manifest" not in path.name and path.name != "selected_scene_ids.csv":
            continue
        if path == STAGE or STAGE in path.parents:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            key = next(
                (name for name in ("scene_id", "base_scene_id", "base_scene_ids") if name in fields),
                None,
            )
            if key is None:
                continue
            for row in reader:
                value = row.get(key, "")
                if value:
                    result.update(item for item in str(value).split(";") if item)
    return result


def _expected_scene_counts(plans: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, int]:
    return {name: sum(int(row["count"]) for row in rows) for name, rows in plans.items()}


def _plan_summaries(plans: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for name, rows in plans.items():
        seeds = [int(row["seed"]) for row in rows]
        serialized = json.dumps(list(rows), sort_keys=True, separators=(",", ":"), allow_nan=False)
        summaries[name] = {
            "scene_count": sum(int(row["count"]) for row in rows),
            "chunk_count": len(rows),
            "seed_min": min(seeds),
            "seed_max": max(seeds),
            "plan_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        }
    return summaries


def _require_cuda(config: Mapping[str, Any]) -> tuple[torch.device, str, int, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-G requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    required = str(config["resources"]["required_gpu_name"])
    if device_name != required:
        raise RuntimeError(f"R6-G requires {required}, found {device_name}")
    free, total = torch.cuda.mem_get_info(0)
    return torch.device("cuda:0"), device_name, int(free), int(total)


def _source_hashes() -> dict[str, str]:
    config = _read_yaml(STAGE / "config.yaml")
    condition_path = ROOT / config["radar"]["condition_source"]
    physics_path = ROOT / config["radar"]["physics_source"]
    paths = {
        "trainer": Path(__file__).resolve(),
        "runner": ROOT / "scripts/run_d2a_r6g.py",
        "student": ROOT / "src/radardistill/students/r6g_student.py",
        "simulation": ROOT / "src/radardistill/simulation/d2a.py",
        "teacher": ROOT / "src/radardistill/teachers/d2a_k2_cuda.py",
        "teacher_adapter": ROOT / "src/radardistill/training/d2a_r5c_teaching_pilot.py",
        "task_loss": ROOT / "src/radardistill/training/d2a_r6d_joint.py",
        "sampler": ROOT / "src/radardistill/training/d2a_r6b_hard_negative.py",
        "response_loss": ROOT / "src/radardistill/distillation/r5c_teaching.py",
        "condition_config": condition_path,
        "physics_config": physics_path,
        "protocol": STAGE / "protocol.md",
        "config": STAGE / "config.yaml",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def r6g_preflight() -> dict[str, Any]:
    """G0: validate frozen code, fresh scene plans, disk headroom, and actual CUDA."""
    config, r6a, _ = _configs()
    validate_r6g_config(config)
    head = _git("rev-parse", "HEAD")
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R6-G G0 requires the authorized main branch")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R6-G G0 requires all implementation changes committed")
    plans = build_r6g_plans(config, r6a)
    expected = _expected_scene_counts(plans)
    historical = _prior_scene_ids()
    scene_audit = audit_r6g_plans(
        plans, expected_counts=expected, prior_scene_ids=historical
    )
    disk = shutil.disk_usage(STAGE)
    remaining_cache = _remaining_owned_cache_bytes(config, r6a, head)
    cache_exists = any((CACHE / name).exists() for name in ("train", "validation", "targets"))
    minimum_free = remaining_cache + 4 * 1024**3
    if not cache_exists:
        minimum_free = max(
            minimum_free,
            int(config["data"]["minimum_free_disk_bytes_before_generation"]),
        )
    if disk.free < minimum_free:
        raise RuntimeError(
            f"R6-G needs {minimum_free} free disk bytes before generation; found {disk.free}"
        )
    device, gpu, free_vram, total_vram = _require_cuda(config)
    record = {
        "status": "G0_PASS",
        "git_sha": head,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_hashes(),
        "split_scene_counts": scene_audit["scene_counts"],
        "historical_scene_id_count": len(historical),
        "historical_scene_id_overlap": scene_audit["prior_scene_overlap_count"],
        "cross_split_scene_id_overlap": scene_audit["cross_split_overlap_count"],
        "disk_free_before_generation_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "minimum_free_bytes": minimum_free,
        "device": str(device),
        "gpu": gpu,
        "free_vram_before_g1_bytes": free_vram,
        "total_vram_bytes": total_vram,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "scene_plan_summaries": _plan_summaries(plans),
    }
    _write_json(STAGE / "g0.json", record)
    return record


def _require_g0() -> tuple[dict[str, Any], str]:
    path = STAGE / "g0.json"
    if not path.is_file():
        raise RuntimeError("R6-G requires a successful G0 preflight")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "G0_PASS" or not isinstance(record.get("git_sha"), str):
        raise RuntimeError("R6-G G0 record is missing its committed source revision")
    if _git("branch", "--show-current") != "main" or _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R6-G requires a clean tracked worktree on main")
    if record.get("protocol_sha256") != _sha256(STAGE / "protocol.md"):
        raise RuntimeError("R6-G protocol changed after G0")
    if record.get("config_sha256") != _sha256(STAGE / "config.yaml"):
        raise RuntimeError("R6-G config changed after G0")
    if record.get("source_sha256") != _source_hashes():
        raise RuntimeError("R6-G implementation sources changed after G0")
    return record, str(record["git_sha"])


def _paired_bytes_per_scene() -> int:
    return int(
        2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
        + 2 * np.dtype(np.float32).itemsize
        + np.dtype(np.uint8).itemsize
        + 48 * np.dtype("<U1").itemsize
    )


def _paired_cache_paths(split: str) -> dict[str, Path]:
    directory = CACHE / split
    names = ("iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id")
    return {name: directory / f"{name}.npy" for name in names}


def _remaining_owned_cache_bytes(
    config: Mapping[str, Any], r6a: Mapping[str, Any], git_sha: str
) -> int:
    plans = build_r6g_plans(config, r6a)
    remaining = 0
    for split in ("train", "validation"):
        plan = plans[split]
        count = sum(int(row["count"]) for row in plan)
        directory = CACHE / split
        written = 0
        if directory.exists():
            owner_path, progress_path = directory / "owner.json", directory / "progress.json"
            if not owner_path.is_file() or not progress_path.is_file():
                raise RuntimeError(f"refusing to reuse unowned R6-G cache: {directory}")
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            _assert_cache_owner(_cache_owner(split, plan, git_sha), owner, f"R6-G {split}")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("owner") != owner:
                raise RuntimeError(f"R6-G cache progress owner mismatch: {directory}")
            written = int(progress.get("rows_written", -1))
            if not 0 <= written <= count:
                raise RuntimeError(f"R6-G cache progress is invalid: {directory}")
        remaining += (count - written) * _paired_bytes_per_scene()

    train_dir = CACHE / "train"
    target_dir = CACHE / "targets"
    if target_dir.exists():
        train_owner_path = train_dir / "owner.json"
        if not train_owner_path.is_file():
            raise RuntimeError("R6-G target cache exists without its train cache owner")
        train_owner = json.loads(train_owner_path.read_text(encoding="utf-8"))
        target_owner = _target_owner(git_sha, {**train_owner, "rows": int(config["data"]["splits"]["train"]["paired_scenes"])})
        owner_path, progress_path = target_dir / "owner.json", target_dir / "progress.json"
        if not owner_path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"refusing to reuse unowned R6-G target cache: {target_dir}")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _assert_cache_owner(target_owner, owner, "R6-G target")
        if progress.get("owner") != owner:
            raise RuntimeError("R6-G target cache owner differs from its registered plan")
        target_rows_written = int(progress.get("completed_rows", -1))
        target_rows = int(config["data"]["splits"]["train"]["paired_scenes"])
        if not 0 <= target_rows_written <= target_rows:
            raise RuntimeError("R6-G target cache progress is invalid")
    else:
        target_rows = int(config["data"]["splits"]["train"]["paired_scenes"])
        target_rows_written = 0
    map_bytes_per_row = 2 * 520 * 32 * np.dtype(np.float16).itemsize
    remaining += (target_rows - target_rows_written) * map_bytes_per_row
    return int(remaining + 2 * 1024 * 1024)


def _cache_owner(split: str, plan: list[Mapping[str, Any]], git_sha: str) -> dict[str, Any]:
    plan_hash = hashlib.sha256(
        json.dumps(list(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "owner": "D2-A-R6-G-paired-cache",
        "split": split,
        "git_sha": git_sha,
        "plan_sha256": plan_hash,
        "config_sha256": _sha256(STAGE / "config.yaml"),
    }


def _assert_cache_owner(expected: Mapping[str, Any], actual: Mapping[str, Any], label: str) -> None:
    expected_identity = dict(expected)
    actual_identity = dict(actual)
    expected_identity.pop("git_sha", None)
    actual_identity.pop("git_sha", None)
    if not isinstance(actual.get("git_sha"), str) or actual_identity != expected_identity:
        raise RuntimeError(f"{label} cache owner differs from the registered plan")


def _load_pair_cache(split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("R6-G persists only train and validation paired scenes")
    g0, git_sha = _require_g0()
    config, r6a, _ = _configs()
    plan = build_r6g_plans(config, r6a)[split]
    count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    owner_path, progress_path = directory / "owner.json", directory / "progress.json"
    expected_owner = _cache_owner(split, plan, git_sha)
    if not owner_path.is_file() or not progress_path.is_file():
        raise RuntimeError(f"R6-G {split} cache owner/progress is missing")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    _assert_cache_owner(expected_owner, owner, f"R6-G {split}")
    if progress.get("owner") != owner or progress.get("rows_written") != count:
        raise RuntimeError(f"R6-G {split} cache is incomplete or has a different owner")
    paths = _paired_cache_paths(split)
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    shapes = {
        "iq_h0": (count, 3, 130, 32),
        "iq_h1": (count, 3, 130, 32),
        "truth_range_cell": (count,),
        "truth_doppler_hz": (count,),
        "condition_index": (count,),
        "scene_id": (count,),
    }
    dtypes = {
        "iq_h0": np.dtype(np.complex64),
        "iq_h1": np.dtype(np.complex64),
        "truth_range_cell": np.dtype(np.float32),
        "truth_doppler_hz": np.dtype(np.float32),
        "condition_index": np.dtype(np.uint8),
        "scene_id": np.dtype("<U48"),
    }
    if any(arrays[name].shape != shapes[name] or arrays[name].dtype != dtypes[name] for name in arrays):
        raise RuntimeError(f"R6-G {split} cache shape/dtype mismatch")
    manifest = STAGE / f"scene_manifest_{split}.csv"
    if not manifest.is_file():
        raise RuntimeError(f"R6-G {split} scene manifest is missing")
    with manifest.open(newline="", encoding="utf-8") as handle:
        manifest_ids = np.asarray([row["scene_id"] for row in csv.DictReader(handle)], dtype=str)
    cached_ids = np.asarray(arrays["scene_id"]).astype(str)
    planned_ids = np.asarray([
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in plan for index in range(int(row["count"]))
    ], dtype=str)
    if not (np.array_equal(manifest_ids, cached_ids) and np.array_equal(cached_ids, planned_ids)):
        raise RuntimeError(f"R6-G {split} cache scene IDs differ from G0 plan/manifest")
    if g0.get("split_scene_counts", {}).get(split) != count:
        raise RuntimeError(f"R6-G {split} cache count differs from G0")
    return arrays


def _require_matching_g1(g0: Mapping[str, Any]) -> dict[str, Any]:
    path = STAGE / "g1.json"
    if not path.is_file():
        raise RuntimeError("R6-G requires a successful G1 physical/CUDA preflight")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("status") != "G1_PASS"
        or record.get("git_sha") != g0.get("git_sha")
        or record.get("protocol_sha256") != g0.get("protocol_sha256")
        or record.get("config_sha256") != g0.get("config_sha256")
    ):
        raise RuntimeError("R6-G G1 record does not match the current G0 source/protocol/config")
    return record


def generate_r6g_pair_cache(split: str) -> dict[str, Any]:
    """Generate/resume an owner-scoped paired IQ memmap without GPU residency."""
    g0, git_sha = _require_g0()
    _require_matching_g1(g0)
    if split not in {"train", "validation"}:
        raise ValueError("R6-G paired cache split must be train or validation")
    config, r6a, physics = _configs()
    plan = build_r6g_plans(config, r6a)[split]
    count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    expected_owner = _cache_owner(split, plan, git_sha)
    disk_free = int(shutil.disk_usage(STAGE).free)
    reserve = 4 * 1024**3
    remaining_cache = _remaining_owned_cache_bytes(config, r6a, git_sha)
    if disk_free < remaining_cache + reserve:
        raise RuntimeError(
            f"R6-G cache generation needs {remaining_cache + reserve} free bytes; found {disk_free}"
        )
    if directory.exists():
        owner_path = directory / "owner.json"
        if not owner_path.is_file():
            raise RuntimeError(f"refusing to reuse an unowned R6-G cache: {directory}")
        _assert_cache_owner(expected_owner, json.loads(owner_path.read_text(encoding="utf-8")), f"R6-G {split}")
        owner = expected_owner
    else:
        directory.mkdir(parents=True)
        owner = expected_owner
        _write_json(directory / "owner.json", owner)

    paths = _paired_cache_paths(split)
    manifest = STAGE / f"scene_manifest_{split}.csv"
    progress_path = directory / "progress.json"
    shapes = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "iq_h1": ((count, 3, 130, 32), np.complex64),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
        "condition_index": ((count,), np.uint8),
        "scene_id": ((count,), "<U48"),
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("R6-G paired cache progress owner mismatch")
        completed = int(progress["completed_chunks"])
        offset = int(progress["rows_written"])
        if not 0 <= completed <= len(plan) or offset != sum(int(row["count"]) for row in plan[:completed]):
            raise RuntimeError("R6-G paired cache cursor differs from the frozen plan")
        if manifest.is_file():
            with manifest.open("r+b") as handle:
                handle.truncate(int(progress["manifest_offset"]))
        elif offset:
            raise RuntimeError("R6-G scene manifest missing for a partial cache")
    else:
        if any(path.name != "owner.json" for path in directory.iterdir()):
            raise RuntimeError("R6-G partial cache has no valid progress record")
        if manifest.exists():
            raise FileExistsError(f"refusing to overwrite an unowned R6-G manifest: {manifest}")
        completed = offset = 0
        progress = {"owner": owner, "completed_chunks": 0, "rows_written": 0, "manifest_offset": 0}
        _write_json(progress_path, progress)
    for name, (shape, dtype) in shapes.items():
        if not paths[name].is_file():
            np.lib.format.open_memmap(paths[name], mode="w+", dtype=dtype, shape=shape).flush()
        else:
            array = np.load(paths[name], mmap_mode="r")
            if array.shape != shape or array.dtype != np.dtype(dtype):
                raise RuntimeError(f"R6-G cache shape/dtype mismatch for {name}")
    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if handle.tell() == 0:
            writer.writerow(("scene_id", "split", "seed", "condition_index", "course", "scenario_id", "snr_db", "cnr_db", "truth_range_cell", "truth_doppler_hz"))
            handle.flush()
            progress["manifest_offset"] = handle.tell()
            _write_json(progress_path, progress)
        for chunk_index, spec in enumerate(plan[completed:], start=completed):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            stop = offset + int(spec["count"])
            arrays["iq_h0"][offset:stop] = generated.iq_h0
            arrays["iq_h1"][offset:stop] = generated.iq_h1
            arrays["truth_range_cell"][offset:stop] = np.asarray(generated.truth_target_range_cell, dtype=np.float32)
            arrays["truth_doppler_hz"][offset:stop] = np.asarray(generated.truth_target_doppler_hz, dtype=np.float32)
            arrays["condition_index"][offset:stop] = int(spec["condition_index"])
            arrays["scene_id"][offset:stop] = np.asarray(generated.base_scene_id).astype("<U48")
            for index, scene_id in enumerate(generated.base_scene_id):
                writer.writerow((
                    str(scene_id), split, int(spec["seed"]), int(spec["condition_index"]),
                    condition["course"], condition["scenario_id"],
                    repr(float(generated.snr_db[index])), repr(float(generated.requested_cnr_db[index])),
                    repr(float(generated.truth_target_range_cell[index])),
                    repr(float(generated.truth_target_doppler_hz[index])),
                ))
            offset = stop
            if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
                for array in arrays.values():
                    array.flush()
                handle.flush()
                progress = {
                    "owner": owner,
                    "completed_chunks": chunk_index + 1,
                    "rows_written": offset,
                    "manifest_offset": handle.tell(),
                }
                _write_json(progress_path, progress)
                print(json.dumps({
                    "stage": "R6-G paired-cache generation", "split": split,
                    "chunks_done": chunk_index + 1, "chunks_total": len(plan), "scenes_done": offset,
                }), flush=True)
    del arrays
    _load_pair_cache(split)
    return {
        "split": split,
        "complete": True,
        "scenes": count,
        "cache_bytes": sum(path.stat().st_size for path in paths.values()),
        "manifest": str(manifest.relative_to(ROOT)),
        "g0_git_sha": git_sha,
        "cache_owner": expected_owner,
    }


def _target_owner(git_sha: str, train_owner: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "owner": "D2-A-R6-G-H1-target-cache",
        "git_sha": git_sha,
        "train_cache_plan_sha256": train_owner["plan_sha256"],
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "rows": int(train_owner.get("rows", 51_200)),
        "targets": ["K3_IQ_only", "T3_truth_Dirichlet"],
        "shape_per_target": [520, 32],
        "dtype": "float16",
    }


def generate_r6g_target_cache(*, batch_size: int | None = None) -> dict[str, Any]:
    """Build CUDA K3 and truth-response target maps for training H1 rows only."""
    g0, git_sha = _require_g0()
    config, r6a, physics = _configs()
    _require_matching_g1(g0)
    if batch_size is None:
        batch_size = int(config["resources"]["target_generation_batch_size"])
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("R6-G target batch_size must be a positive integer")
    device, gpu, _, _ = _require_cuda(config)
    train = _load_pair_cache("train")
    count = len(train["iq_h1"])
    train_owner = json.loads((CACHE / "train" / "owner.json").read_text(encoding="utf-8"))
    owner = _target_owner(git_sha, {**train_owner, "rows": count})
    directory = CACHE / "targets"
    free = int(shutil.disk_usage(STAGE).free)
    required = _remaining_owned_cache_bytes(config, r6a, git_sha) + 4 * 1024**3
    if free < required:
        raise RuntimeError(f"R6-G target generation needs {required} free bytes; found {free}")
    if directory.exists():
        owner_path = directory / "owner.json"
        if not owner_path.is_file():
            raise RuntimeError(f"refusing to reuse an unowned R6-G target cache: {directory}")
        _assert_cache_owner(owner, json.loads(owner_path.read_text(encoding="utf-8")), "R6-G target")
    else:
        directory.mkdir(parents=True)
        _write_json(directory / "owner.json", owner)
    paths = {name: directory / f"{name}.npy" for name in ("k3", "t3")}
    shapes = {name: (count, 520, 32) for name in paths}
    progress_path = directory / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("R6-G target cache progress owner mismatch")
        start = int(progress["completed_rows"])
        if not 0 <= start <= count:
            raise RuntimeError("R6-G target cache cursor is invalid")
        if start and any(not path.is_file() for path in paths.values()):
            raise RuntimeError("R6-G target cache file missing for a nonzero progress cursor")
    else:
        if any(path.name not in {"owner.json"} for path in directory.iterdir()):
            raise RuntimeError("R6-G partial target cache has no valid progress record")
        start = 0
        progress = {"owner": owner, "completed_rows": 0}
        _write_json(progress_path, progress)
    for name, path in paths.items():
        if not path.is_file():
            np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shapes[name]).flush()
        else:
            mapped = np.load(path, mmap_mode="r")
            if mapped.shape != shapes[name] or mapped.dtype != np.dtype(np.float16):
                raise RuntimeError(f"R6-G {name} target cache shape/dtype mismatch")
    targets = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    start_time = time.perf_counter()
    for begin in range(start, count, batch_size):
        end = min(begin + batch_size, count)
        # The source contains H1 IQ only. Truth labels are passed solely to the
        # T3 analytic target builder, never to the IQ-only K3 teacher.
        k3 = compute_h1_teacher_maps(train["iq_h1"][begin:end], physics, device)
        validate_r6g_target_maps(k3, expected_count=end - begin)
        ranges = torch.as_tensor(train["truth_range_cell"][begin:end], device=device)
        dopplers = torch.as_tensor(train["truth_doppler_hz"][begin:end], device=device)
        t3 = build_r6g_t3_maps(
            ranges, dopplers, device=device,
            prf_hz=float(config["radar"]["prf_hz"]),
            pulse_count=int(config["radar"]["pulse_count"]),
        )
        validate_r6g_target_maps(t3, expected_count=end - begin)
        targets["k3"][begin:end] = k3.detach().cpu().numpy().astype(np.float16)
        targets["t3"][begin:end] = t3.detach().cpu().numpy().astype(np.float16)
        if end % (batch_size * 128) == 0 or end == count:
            for value in targets.values():
                value.flush()
            progress = {"owner": owner, "completed_rows": end}
            _write_json(progress_path, progress)
            print(json.dumps({
                "stage": "R6-G teacher-target generation", "rows_done": end,
                "rows_total": count, "gpu": gpu,
            }), flush=True)
    del targets
    for path in paths.values():
        validate_r6g_target_maps(np.load(path, mmap_mode="r"), expected_count=count)
    return {
        "complete": True,
        "rows": count,
        "target_cache_bytes": sum(path.stat().st_size for path in paths.values()),
        "gpu": gpu,
        "device": str(device),
        "batch_size": int(batch_size),
        "seconds": time.perf_counter() - start_time,
        "targets": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "owner": owner,
    }


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_history(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def _load_target_cache(config: Mapping[str, Any], git_sha: str) -> dict[str, np.ndarray]:
    count = int(config["data"]["splits"]["train"]["paired_scenes"])
    train_owner_path = CACHE / "train" / "owner.json"
    if not train_owner_path.is_file():
        raise RuntimeError("R6-G target cache requires the registered training IQ cache")
    train_owner = json.loads(train_owner_path.read_text(encoding="utf-8"))
    expected_owner = _target_owner(git_sha, {**train_owner, "rows": count})
    directory = CACHE / "targets"
    owner_path, progress_path = directory / "owner.json", directory / "progress.json"
    if not owner_path.is_file() or not progress_path.is_file():
        raise RuntimeError("R6-G target cache owner/progress is missing")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    _assert_cache_owner(expected_owner, owner, "R6-G target")
    if progress.get("owner") != owner or int(progress.get("completed_rows", -1)) != count:
        raise RuntimeError("R6-G target cache is incomplete or has a different progress owner")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in ("k3", "t3")}
    for name, array in arrays.items():
        if array.shape != (count, 520, 32) or array.dtype != np.dtype(np.float16):
            raise RuntimeError(f"R6-G {name} response cache shape/dtype mismatch")
    return arrays


def _r6g_training_batch(
    train: Mapping[str, np.ndarray],
    h0_indices: np.ndarray,
    h1_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return assemble_independent_h0_h1_batch(train, h0_indices, h1_indices)


def _r6g_training_update(
    model: R6GCEStudent,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    h1_indices: np.ndarray,
    *,
    arm: str,
    target_maps: Mapping[str, np.ndarray] | None,
    microbatch: int,
    effective_batch: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, float]:
    if arm not in {"T0", "T1", "T3"}:
        raise ValueError(f"unknown R6-G arm {arm!r}")
    if len(batch["iq_ri"]) != effective_batch or effective_batch % microbatch:
        raise ValueError("R6-G microbatch must divide the frozen effective batch")
    if len(h1_indices) * 2 != effective_batch:
        raise ValueError("R6-G H1 target indices do not match the 50/50 task batch")
    if arm != "T0" and target_maps is None:
        raise RuntimeError(f"R6-G {arm} requires cached H1 response targets")

    inputs = torch.from_numpy(batch["iq_ri"])
    labels = {
        key: torch.from_numpy(np.asarray(batch[key], dtype=np.float32))
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("total", "truth", "auxiliary", "presence", "range_cell", "range_offset", "doppler", "candidate")}
    presence_np = np.asarray(batch["truth_presence"], dtype=bool)
    for start in range(0, effective_batch, microbatch):
        stop = start + microbatch
        x = inputs[start:stop].to(device)
        truth = {key: value[start:stop].to(device) for key, value in labels.items()}
        outputs = model(x, compute_spectrum=arm != "T0")
        task_terms = r6d_loss(
            outputs, truth, arm="j1_joint", prf_hz=float(config["radar"]["prf_hz"])
        )
        task_loss = task_terms["total"]
        auxiliary = task_loss.detach() * 0.0
        h1_mask = truth["truth_presence"].bool()
        h1_count = int(h1_mask.sum().item())
        h1_offset = int(np.count_nonzero(presence_np[:start]))
        if arm != "T0" and h1_count:
            target_name = "k3" if arm == "T1" else "t3"
            rows = np.asarray(h1_indices[h1_offset:h1_offset + h1_count], dtype=np.int64)
            target_np = np.asarray(target_maps[target_name][rows], dtype=np.float32)
            target = torch.from_numpy(target_np).to(device)
            auxiliary = r6g_response_map_kl(outputs["spectrum_logits"][h1_mask], target)
        loss = task_loss + float(config["training"]["teacher_loss_weight"]) * auxiliary
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"R6-G {arm} update loss is nonfinite")
        scale = microbatch / effective_batch
        (loss * scale).backward()
        totals["total"] += float(loss.detach().item()) * scale
        totals["truth"] += float(task_loss.detach().item()) * scale
        totals["auxiliary"] += float(auxiliary.detach().item()) * scale
        for key in ("presence", "range_cell", "range_offset", "doppler", "candidate"):
            totals[key] += float(task_terms[key].detach().item()) * scale
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), float(config["training"]["gradient_clip_norm"])
    )
    if not bool(torch.isfinite(gradient_norm)):
        raise FloatingPointError(f"R6-G {arm} gradient norm is nonfinite")
    optimizer.step()
    return totals


def _r6g_validation_summary(
    model: R6GCEStudent,
    validation: Mapping[str, np.ndarray],
    *,
    microbatch: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, float]:
    count = len(validation["scene_id"])
    keys = ("total", "presence", "range_cell", "range_offset", "doppler", "candidate")
    totals = {key: 0.0 for key in keys}
    model.eval()
    with torch.inference_mode():
        for base_start in range(0, count, 32):
            base_indices = np.arange(base_start, min(base_start + 32, count), dtype=np.int64)
            batch = _r6g_training_batch(validation, base_indices, base_indices)
            if len(batch["iq_ri"]) % microbatch:
                raise ValueError("R6-G validation cohort does not divide by the selected microbatch")
            inputs = torch.from_numpy(batch["iq_ri"])
            labels = {
                key: torch.from_numpy(np.asarray(batch[key], dtype=np.float32))
                for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
            }
            for start in range(0, len(inputs), microbatch):
                stop = start + microbatch
                x = inputs[start:stop].to(device)
                truth = {key: value[start:stop].to(device) for key, value in labels.items()}
                outputs = model(x, compute_spectrum=False)
                terms = r6d_loss(
                    outputs, truth, arm="j1_joint", prf_hz=float(config["radar"]["prf_hz"])
                )
                weight = len(inputs[start:stop]) / (2 * count)
                for key in keys:
                    totals[key] += float(terms[key].item()) * weight
    if not all(np.isfinite(value) for value in totals.values()):
        raise FloatingPointError("R6-G validation truth-task loss is nonfinite")
    return totals


def _r6g_shared_initial_state(
    config: Mapping[str, Any], git_sha: str
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = RUNS / "shared_initial_state.pt"
    owner = {
        "owner": "D2-A-R6-G-shared-initialization",
        "git_sha": git_sha,
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "seed": int(config["student"]["initialization"]["seed"]),
        "width": int(config["student"]["width"]),
    }
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        _assert_cache_owner(owner, saved.get("owner", {}), "R6-G shared initialization")
        state = saved.get("state_dict")
        if not isinstance(state, dict):
            raise RuntimeError("R6-G shared initial state is missing")
        return state, owner
    RUNS.mkdir(parents=True, exist_ok=True)
    if any(item.name != path.name for item in RUNS.iterdir()):
        raise RuntimeError("R6-G run directory has files without an initialization owner")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(owner["seed"]))
        template = R6GCEStudent(width=int(owner["width"]))
        state = {key: value.detach().cpu().clone() for key, value in template.state_dict().items()}
        del template
    _atomic_torch_save({"owner": owner, "state_dict": state}, path)
    return state, owner


def _r6g_training_owner(
    arm: str,
    config: Mapping[str, Any],
    git_sha: str,
    sampler_audit: Mapping[str, Any],
) -> dict[str, Any]:
    target_owner = None
    if arm in {"T1", "T3"}:
        target_owner = json.loads((CACHE / "targets" / "owner.json").read_text(encoding="utf-8"))
    return {
        "owner": "D2-A-R6-G-training-run",
        "git_sha": git_sha,
        "arm": arm,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "initialization_seed": int(config["student"]["initialization"]["seed"]),
        "sampler_seed": int(config["data"]["sampler_seed"]),
        "updates": int(config["training"]["fixed_updates_per_arm"]),
        "effective_batch": int(config["training"]["effective_batch_size_cpi"]),
        "h1_plan_sha256": sampler_audit["h1_plan_sha256"],
        "h0_plan_sha256": sampler_audit["h0_plan_sha256"],
        "target_owner": target_owner,
    }


def _r6g_load_resume(
    path: Path,
    owner: Mapping[str, Any],
    model: R6GCEStudent,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _assert_cache_owner(owner, saved.get("owner", {}), "R6-G resume")
    model.load_state_dict(saved["model_state_dict"], strict=True)
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    if "torch_rng_state" in saved:
        torch.set_rng_state(saved["torch_rng_state"])
    if "cuda_rng_state" in saved:
        torch.cuda.set_rng_state(saved["cuda_rng_state"], device=device)
    return saved


def r6g_train_arm(
    arm: str,
    *,
    microbatch: int | None = None,
    oom_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Train/resume one R6-G arm with fixed effective batch and common validation selection."""
    if arm not in {"T0", "T1", "T3"}:
        raise ValueError(f"unknown R6-G arm: {arm}")
    g0, git_sha = _require_g0()
    config, _, _ = _configs()
    g1 = _require_matching_g1(g0)
    device, gpu, _, _ = _require_cuda(config)
    if g1.get("device") != str(device):
        raise RuntimeError("R6-G G1 device does not match the active training device")
    if microbatch is None:
        microbatch = int(g1["formal_microbatch"])
    effective_batch = int(config["training"]["effective_batch_size_cpi"])
    if microbatch not in (int(config["training"]["microbatch_initial"]), *map(int, config["training"]["microbatch_oom_fallbacks"])):
        raise ValueError("R6-G microbatch is outside the frozen fallback sequence")
    if effective_batch % microbatch:
        raise ValueError("R6-G microbatch must divide the fixed effective batch")
    if oom_events is None:
        oom_events = []

    train = _load_pair_cache("train")
    validation = _load_pair_cache("validation")
    target_maps = _load_target_cache(config, git_sha) if arm in {"T1", "T3"} else None
    updates = int(config["training"]["fixed_updates_per_arm"])
    sampler_seed = int(config["data"]["sampler_seed"])
    h1_plan, h0_plan, sampler_audit = build_r6g_sampler_plans(
        train["condition_index"], updates=updates, effective_batch=effective_batch, seed=sampler_seed
    )
    if sampler_audit["h1_scene_visits_per_presence"] != int(config["training"]["expected_scene_visits_per_presence"]):
        raise RuntimeError("R6-G H1 exposure differs from the frozen scene-visit target")
    owner = _r6g_training_owner(arm, config, git_sha, sampler_audit)
    initial_state, initialization_owner = _r6g_shared_initial_state(config, git_sha)
    run_dir = RUNS / arm
    run_dir.mkdir(parents=True, exist_ok=True)
    owner_path = run_dir / "owner.json"
    if owner_path.is_file():
        _assert_cache_owner(owner, json.loads(owner_path.read_text(encoding="utf-8")), f"R6-G {arm} run")
    else:
        if any(path.name != "owner.json" for path in run_dir.iterdir()):
            raise RuntimeError(f"R6-G {arm} run has files without an owner record")
        _write_json(owner_path, owner)

    record_path = run_dir / "training_record.json"
    if record_path.is_file():
        record = json.loads(record_path.read_text(encoding="utf-8"))
        _assert_cache_owner(owner, record.get("owner", {}), f"R6-G {arm} record")
        if record.get("status") == "complete" and int(record.get("updates", -1)) == updates:
            return record

    torch.manual_seed(int(config["student"]["initialization"]["seed"]))
    torch.cuda.manual_seed_all(int(config["student"]["initialization"]["seed"]))
    model = R6GCEStudent(width=int(config["student"]["width"])).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate_initial"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    history_path = run_dir / "training_history.csv"
    last_path = run_dir / "last_checkpoint.pt"
    best_path = run_dir / "best_checkpoint.pt"
    validation_interval = int(config["training"]["validation_interval_updates"])
    checkpoint_interval = int(config["training"]["resume_checkpoint_interval_updates"])
    history: list[dict[str, Any]] = []
    best_objective = float("inf")
    best_update = -1
    start_update = 0
    elapsed_before = 0.0
    if last_path.is_file():
        saved = _r6g_load_resume(last_path, owner, model, optimizer, device)
        start_update = int(saved["update"])
        history = list(saved.get("history", []))
        best_objective = float(saved.get("best_objective", float("inf")))
        best_update = int(saved.get("best_update", -1))
        elapsed_before = float(saved.get("training_seconds_completed", 0.0))
        _write_history(history_path, history)
    elif any(path.name not in {"owner.json", "training_record.json"} for path in run_dir.iterdir()):
        raise RuntimeError(f"R6-G {arm} artifacts exist without a resumable checkpoint")
    else:
        initial_validation = _r6g_validation_summary(
            model, validation, microbatch=microbatch, device=device, config=config
        )
        history.append({"update": 0, "learning_rate": float(config["training"]["learning_rate_initial"]),
                        "validation_truth_total": initial_validation["total"],
                        **{f"validation_{key}": value for key, value in initial_validation.items()}})
        _write_history(history_path, history)
        _atomic_torch_save({
            "owner": owner,
            "update": 0,
            "model_state_dict": _cpu_tree(model.state_dict()),
            "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            "history": history,
            "best_objective": best_objective,
            "best_update": best_update,
            "training_seconds_completed": 0.0,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
            "microbatch": microbatch,
            "oom_events": list(oom_events),
        }, last_path)

    if best_path.is_file():
        saved_best = torch.load(best_path, map_location="cpu", weights_only=False)
        _assert_cache_owner(owner, saved_best.get("owner", {}), f"R6-G {arm} best checkpoint")
        saved_objective = float(saved_best["validation_truth_total"])
        if saved_objective < best_objective:
            best_objective = saved_objective
            best_update = int(saved_best["update"])

    target_updates = updates
    history_every = int(config["training"]["history_interval_updates"])
    interval_sums = {key: 0.0 for key in ("total", "truth", "auxiliary", "presence", "range_cell", "range_offset", "doppler", "candidate")}
    interval_count = 0
    run_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    update = start_update + 1
    while update <= target_updates:
        try:
            for group in optimizer.param_groups:
                group["lr"] = r6d_learning_rate(
                    update,
                    total_updates=target_updates,
                    initial=float(config["training"]["learning_rate_initial"]),
                    final=float(config["training"]["learning_rate_final"]),
                )
            batch = _r6g_training_batch(train, h0_plan[update - 1], h1_plan[update - 1])
            terms = _r6g_training_update(
                model, optimizer, batch, h1_plan[update - 1], arm=arm,
                target_maps=target_maps, microbatch=microbatch,
                effective_batch=effective_batch, device=device, config=config,
            )
            for key in interval_sums:
                interval_sums[key] += terms[key]
            interval_count += 1

            if update % validation_interval == 0:
                validation_metrics = _r6g_validation_summary(
                    model, validation, microbatch=microbatch, device=device, config=config
                )
                objective = float(validation_metrics["total"])
                if objective < best_objective:
                    best_objective, best_update = objective, update
                    _atomic_torch_save({
                        "owner": owner,
                        "update": update,
                        "validation_truth_total": objective,
                        "validation_metrics": validation_metrics,
                        "model_state_dict": _cpu_tree(model.state_dict()),
                    }, best_path)
                row = {
                    "update": update,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **{f"interval_train_{key}": interval_sums[key] / max(interval_count, 1) for key in interval_sums},
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "microbatch": microbatch,
                }
                history.append(row)
                if update % history_every == 0:
                    _write_history(history_path, history)
                interval_sums = {key: 0.0 for key in interval_sums}
                interval_count = 0
                model.train()

            if update % checkpoint_interval == 0 or update == target_updates:
                elapsed = elapsed_before + time.perf_counter() - run_start
                _atomic_torch_save({
                    "owner": owner,
                    "update": update,
                    "model_state_dict": _cpu_tree(model.state_dict()),
                    "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                    "history": history,
                    "best_objective": best_objective,
                    "best_update": best_update,
                    "training_seconds_completed": elapsed,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
                    "microbatch": microbatch,
                    "oom_events": list(oom_events),
                }, last_path)
                print(json.dumps({
                    "stage": "R6-G train", "arm": arm, "update": update,
                    "train_truth": history[-1].get("interval_train_truth"),
                    "validation_truth": history[-1].get("validation_truth_total"),
                    "best_update": best_update, "microbatch": microbatch,
                    "gpu": gpu,
                }), flush=True)
            update += 1
        except torch.cuda.OutOfMemoryError:
            fallbacks = [int(value) for value in config["training"]["microbatch_oom_fallbacks"]]
            next_microbatch = next((value for value in fallbacks if value < microbatch), None)
            if next_microbatch is None:
                raise RuntimeError(
                    f"R6-G {arm} OOM at microbatch 1; the fixed effective batch cannot fit"
                ) from None
            oom_events.append({"update": update, "from_microbatch": microbatch, "to_microbatch": next_microbatch})
            _write_json(run_dir / "oom_events.json", {"owner": owner, "events": oom_events})
            del model, optimizer
            torch.cuda.empty_cache()
            return r6g_train_arm(arm, microbatch=next_microbatch, oom_events=oom_events)

    total_training_seconds = elapsed_before + time.perf_counter() - run_start
    if not best_path.is_file():
        raise RuntimeError(f"R6-G {arm} completed without a validation-selected checkpoint")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    _assert_cache_owner(owner, best.get("owner", {}), f"R6-G {arm} final best checkpoint")
    model.load_state_dict(best["model_state_dict"], strict=True)
    inference_model = export_r6g_inference_model(model)
    inference_path = run_dir / "inference_checkpoint.pt"
    _atomic_torch_save({
        "owner": owner,
        "update": int(best["update"]),
        "model_state_dict": _cpu_tree(inference_model.state_dict()),
    }, inference_path)
    _write_history(history_path, history)
    record = {
        "status": "complete",
        "arm": arm,
        "owner": owner,
        "updates": target_updates,
        "best_update": int(best["update"]),
        "best_validation_truth_total": float(best["validation_truth_total"]),
        "training_seconds": total_training_seconds,
        "effective_batch_size": effective_batch,
        "microbatch": microbatch,
        "microbatch_oom_events": list(oom_events),
        "unique_train_paired_scenes": len(train["scene_id"]),
        "scene_visits_per_presence": sampler_audit["h1_scene_visits_per_presence"],
        "total_cpi_exposures_per_presence": sampler_audit["h1_exposures"],
        "sampler_audit": sampler_audit,
        "inference_parameters": sum(parameter.numel() for parameter in inference_model.parameters()),
        "training_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu": gpu,
        "device": str(device),
        "precision": "FP32",
        "history": str(history_path.relative_to(ROOT)),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
        "inference_checkpoint": str(inference_path.relative_to(ROOT)),
        "preflight_git_sha": g0["git_sha"],
        "initialization_owner": initialization_owner,
    }
    _write_json(record_path, record)
    del model, optimizer, inference_model, train, validation, target_maps
    torch.cuda.empty_cache()
    return record


def r6g_train_all() -> dict[str, Any]:
    """Run or resume T0, T1, T3 sequentially using the shared fixed scene plan."""
    config, _, _ = _configs()
    g0, _ = _require_g0()
    _require_matching_g1(g0)
    results: dict[str, Any] = {}
    for arm in config["training"]["arm_order"]:
        results[arm] = r6g_train_arm(str(arm))
    expected_updates = int(config["training"]["fixed_updates_per_arm"])
    if any(int(row["updates"]) != expected_updates or row["status"] != "complete" for row in results.values()):
        raise RuntimeError("R6-G arms did not complete identical frozen update budgets")
    return {
        "status": "all_arms_complete",
        "git_sha": g0["git_sha"],
        "arm_order": list(config["training"]["arm_order"]),
        "updates_per_arm": expected_updates,
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "arms": results,
    }
