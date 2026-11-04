"""Matched teacher-content pilot for D2-A-R6-G."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


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
    }.items():
        _expect(resources, key, expected, "resources")
