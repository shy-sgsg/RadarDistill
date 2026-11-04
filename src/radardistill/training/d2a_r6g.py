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
    r6d_loss,
)
from radardistill.students.r6g_student import R6GCEStudent


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-G"
CACHE = STAGE / "cache/training"
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
    paths = {
        "trainer": Path(__file__).resolve(),
        "student": ROOT / "src/radardistill/students/r6g_student.py",
        "simulation": ROOT / "src/radardistill/simulation/d2a.py",
        "teacher": ROOT / "src/radardistill/teachers/d2a_k2_cuda.py",
        "teacher_adapter": ROOT / "src/radardistill/training/d2a_r5c_teaching_pilot.py",
        "task_loss": ROOT / "src/radardistill/training/d2a_r6d_joint.py",
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
    minimum_free = int(config["data"]["minimum_free_disk_bytes_before_generation"])
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
        "scene_plan_seed_bases": {
            split: [int(row["seed"]) for row in rows]
            for split, rows in plans.items()
        },
    }
    _write_json(STAGE / "g0.json", record)
    return record


def _require_g0() -> tuple[dict[str, Any], str]:
    path = STAGE / "g0.json"
    if not path.is_file():
        raise RuntimeError("R6-G requires a successful G0 preflight")
    record = json.loads(path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    if record.get("status") != "G0_PASS" or record.get("git_sha") != head:
        raise RuntimeError("R6-G G0 record does not match the current committed source")
    if record.get("protocol_sha256") != _sha256(STAGE / "protocol.md"):
        raise RuntimeError("R6-G protocol changed after G0")
    if record.get("config_sha256") != _sha256(STAGE / "config.yaml"):
        raise RuntimeError("R6-G config changed after G0")
    return record, head


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
            if owner != _cache_owner(split, plan, git_sha):
                raise RuntimeError(f"R6-G cache owner differs from its frozen plan: {directory}")
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
        if owner != target_owner or progress.get("owner") != owner:
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
    if owner != expected_owner or progress.get("owner") != owner or progress.get("rows_written") != count:
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
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != expected_owner:
            raise RuntimeError(f"refusing to reuse an unowned R6-G cache: {directory}")
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
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
            raise RuntimeError(f"refusing to reuse an unowned R6-G target cache: {directory}")
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
