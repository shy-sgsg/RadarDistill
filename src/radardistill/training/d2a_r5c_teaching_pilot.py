"""Frozen protocol and data-plan helpers for the R5-C teaching pilot."""

from __future__ import annotations

import math
import csv
import json
import os
import random
import subprocess
import time
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radardistill.distillation.r5c_teaching import response_kl
from radardistill.distillation.task_losses import task_loss
from radardistill.students.task_student import TaskStudent
from radardistill.teachers.d2a_k2_cuda import k2_response_cuda
from radardistill.training.d2a_e002 import (
    _paired_validation,
    make_paired_examples,
    scale_d2a_e002_microbatch_loss,
)
from radardistill.training.d2a_r5 import R5CosineSchedule

_CONDITIONS = (
    ("C0", "separated", 20.0, -5.0),
    ("C1", "separated", 15.0, 0.0),
    ("C1", "near_clutter_edge", 15.0, 0.0),
    ("C2", "separated", 10.0, 10.0),
    ("C2", "near_clutter_edge", 10.0, 10.0),
    ("C2", "partial_overlap", 10.0, 10.0),
    ("C3", "separated", 0.0, 20.0),
    ("C3", "separated", 10.0, 20.0),
    ("C3", "near_clutter_edge", 0.0, 20.0),
    ("C3", "near_clutter_edge", 10.0, 20.0),
    ("C3", "partial_overlap", 0.0, 20.0),
    ("C3", "partial_overlap", 10.0, 20.0),
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STAGE_ROOT = _PROJECT_ROOT / "experiments/D2-A-R5/R5-C-teaching"
_REQUIRED_R5C_REVIEW_TARGET_COMMIT = "7563c822e70afe761897ddd5e91bb64207720f89"
FROZEN_INPUTS = (
    "AGENTS.md",
    "docs/RESEARCH_CHARTER.md",
    "docs/CURRENT_STATUS.md",
    "docs/RESEARCH_ROADMAP.md",
    "docs/research/ASSUMPTION_REGISTER.md",
    "experiments/D2-A-R5/R5-C-teaching/protocol.md",
    "experiments/D2-A-R5/R5-C-teaching/config.yaml",
    "experiments/D2-A-R5/R5-C/protocol.md",
    "experiments/D2-A-R5/R5-C/config.yaml",
    "experiments/D2-A-R5/R5-C/review_request.md",
    "experiments/D2-A-R5/R5-A/config.yaml",
    "docs/stage_reviews/D2-A-R5C_review.md",
    "experiments/D2-A-E003/config.yaml",
    "experiments/D2-A-E001/config.yaml",
    "src/radardistill/students/task_student.py",
    "src/radardistill/distillation/r5c_teaching.py",
    "src/radardistill/distillation/task_losses.py",
    "src/radardistill/training/d2a_r5c_teaching_pilot.py",
    "src/radardistill/d2a_baselines.py",
    "src/radardistill/baselines.py",
    "src/radardistill/teachers/d2a_k2_cuda.py",
    "src/radardistill/training/d2a_e002.py",
    "src/radardistill/training/d2a_e002_data.py",
    "src/radardistill/training/d2a_r5.py",
    "src/radardistill/evaluation/d2a_e001.py",
    "src/radardistill/evaluation/d2a_e002.py",
    "src/radardistill/evaluation/d2a_r5c_teaching_pilot.py",
    "src/radardistill/simulation/d0.py",
    "src/radardistill/simulation/d2a.py",
    "scripts/run_d2a_r5.py",
    "scripts/run_d2a_e003.py",
    "scripts/run_d2a_r5c_teacher_audit.py",
    "scripts/run_d2a_r5c_teaching_pilot.py",
    "tests/test_d2a_r5c_teaching.py",
    "tests/test_d2a_r5c_teaching_runner.py",
    "tests/test_d2a_r5c_teaching_evaluation.py",
)

_SPLITS = {
    "h0_calibration": {
        "presence": "H0",
        "scene_count": 8192,
        "condition_indices": list(range(12)),
        "condition_counts": [2048, 1024, 1024, 683, 683, 682, 342, 342, 341, 341, 341, 341],
        "seed_base": 1_048_000_000,
    },
    "h0_verification": {
        "presence": "H0",
        "scene_count": 8192,
        "condition_indices": list(range(12)),
        "condition_counts": [2048, 1024, 1024, 683, 683, 682, 342, 342, 341, 341, 341, 341],
        "seed_base": 1_068_000_000,
    },
    "c3_h0_verification": {
        "presence": "H0",
        "scene_count": 4096,
        "condition_indices": list(range(6, 12)),
        "condition_counts": [683, 683, 683, 683, 682, 682],
        "seed_base": 1_088_000_000,
    },
    "h1_development": {
        "presence": "H1",
        "scene_count": 4352,
        "condition_indices": list(range(12)),
        "condition_counts": [256, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        "seed_base": 1_108_000_000,
    },
}


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"R5-C teaching config requires a {name} mapping")
    return section


def _expect_keys(section: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(section) != expected:
        raise ValueError(f"R5-C teaching config fields differ from the draft: {label}")


def _expect(section: Mapping[str, Any], key: str, expected: Any, label: str) -> None:
    actual = section.get(key)
    if not _matches(actual, expected):
        raise ValueError(f"R5-C teaching frozen setting changed: {label}.{key}")


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(actual, bool) and actual is expected
    if isinstance(expected, int):
        return isinstance(actual, Integral) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return isinstance(actual, Real) and not isinstance(actual, bool) and float(actual) == expected
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            _matches(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and set(actual) == set(expected) and all(
            _matches(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def validate_r5c_teaching_config(config: Mapping[str, Any]) -> None:
    """Validate the frozen pilot design and the exact returned review record."""
    if not isinstance(config, Mapping):
        raise ValueError("R5-C teaching config must be a mapping")
    _expect_keys(
        config,
        {"experiment", "radar", "model", "data", "training", "evaluation", "gates", "resources"},
        "root",
    )
    experiment = _section(config, "experiment")
    radar = _section(config, "radar")
    model = _section(config, "model")
    data = _section(config, "data")
    training = _section(config, "training")
    evaluation = _section(config, "evaluation")
    resources = _section(config, "resources")
    expected_section_keys = {
        "experiment": {
            "id", "subphase", "protocol_version", "status", "charter_version",
            "independent_chatgpt_review", "run_authorized", "final_blind_generation_authorized",
        },
        "radar": {"channel_count", "prf_hz", "pulse_count", "range_bin_count", "clutter", "source_config"},
        "model": {
            "architecture", "width", "spectrum_frequency_bins", "training_parameter_count_expected",
            "inference_parameter_count_expected", "spectrum_head_training_only",
            "remove_spectrum_head_at_inference",
        },
        "data": {
            "training_source", "training_pairs", "validation_source", "validation_pairs", "pair_plan_source",
            "training_seed_bases_by_course", "training_assignment_seed", "training_cell_counts",
            "validation_seed_bases_by_course", "validation_assignment_seed", "validation_cell_counts",
            "pair_plan_seed_base", "fixed_pair_dataset_passes", "reuse_same_pair_order_for_all_arms",
            "chunk_size", "condition_seed_stride", "chunk_seed_stride", "split_seed_bases",
            "conditions", "splits", "persist_full_iq", "persist_teacher_maps",
            "final_blind_generation_authorized",
        },
        "training": {
            "arms", "initialization_seed", "matched_initial_state_bit_identical", "supervision",
            "task_losses", "teacher_loss_weight", "teacher_loss_weight_schedule", "teacher_loss_rows", "teacher_input",
            "teacher_backend", "teacher_map_storage", "updates", "paired_scenes_per_update",
            "effective_batch_size_cpi", "initial_microbatch_size", "oom_microbatch_fallbacks",
            "preserve_effective_batch_on_oom", "optimizer", "weight_decay", "gradient_clip_norm",
            "learning_rate_initial", "learning_rate_at_update_6400", "learning_rate_final", "schedule",
            "validation_interval_updates", "selection", "precision", "amp",
            "auxiliary_response_support", "auxiliary_target_embedding", "execution_authorized",
        },
        "evaluation": {
            "threshold_target_pfa_cpi", "calibration_quantile", "calibration_quantile_method",
            "score_type", "auc_method", "range_gate_bins", "doppler_gate_hz",
            "decision_comparator", "exact_pfa_interval", "bootstrap_unit", "bootstrap_replicates",
            "bootstrap_seed", "primary_scope", "primary_contrasts", "minimum_primary_gain",
            "simultaneous_one_sided_confidence", "primary_bootstrap_lower_quantile",
            "protection_intervals", "minimum_pd_difference_point",
            "minimum_joint_difference_point", "minimum_protection_ci_lower",
            "minimum_c2_c3_joint_ci_lower", "maximum_c2_c3_rmse_ratio_point",
            "maximum_c2_c3_rmse_ratio_nominal_95_upper",
            "maximum_c3_h0_pfa_difference_nominal_95_upper", "c3_h0_pfa_point_limit",
            "final_blind_generation_authorized",
        },
        "gates": {"g1_overfit"},
        "resources": {
            "gpu", "device", "cuda_required", "cpu_fallback", "simultaneous_models_on_gpu",
            "minimum_free_vram_bytes", "minimum_free_disk_bytes", "persist_full_iq",
            "persist_teacher_maps", "project_size_final_limit_gib",
            "preserve_untracked_user_files",
        },
    }
    for name, keys in expected_section_keys.items():
        _expect_keys(_section(config, name), keys, name)

    for key, value in {
        "id": "D2-A-R5",
        "subphase": "R5-C-teaching",
        "charter_version": 1.3,
        "final_blind_generation_authorized": False,
    }.items():
        _expect(experiment, key, value, "experiment")
    review = experiment.get("independent_chatgpt_review")
    if not isinstance(review, Mapping):
        raise ValueError("R5-C teaching config requires an explicit independent review record")
    _expect_keys(review, {"status", "target_commit", "decision", "supports_pilot"}, "independent_chatgpt_review")
    if review.get("target_commit") != _REQUIRED_R5C_REVIEW_TARGET_COMMIT:
        raise ValueError(
            "R5-C teaching review target commit must equal the frozen teacher-audit commit "
            f"{_REQUIRED_R5C_REVIEW_TARGET_COMMIT}"
        )
    if (
        experiment.get("status") != "preregistered"
        or experiment.get("protocol_version") != 1.0
        or review.get("status") != "returned"
        or review.get("decision") != "REFINE"
        or review.get("supports_pilot") is not True
        or experiment.get("run_authorized") is not True
        or config["training"].get("execution_authorized") is not True
    ):
        raise ValueError("R5-C teaching run requires the returned REFINE review to authorize this pilot")

    for key, value in {
        "channel_count": 3,
        "prf_hz": 1300.0,
        "pulse_count": 130,
        "range_bin_count": 32,
        "clutter": "stationary",
        "source_config": "experiments/D2-A-E001/config.yaml",
    }.items():
        _expect(radar, key, value, "radar")
    for key, value in {
        "architecture": "TaskStudent",
        "width": 96,
        "spectrum_frequency_bins": 520,
        "training_parameter_count_expected": 120717,
        "inference_parameter_count_expected": 70277,
        "spectrum_head_training_only": True,
        "remove_spectrum_head_at_inference": True,
    }.items():
        _expect(model, key, value, "model")

    for key, value in {
        "training_source": "experiments/D2-A-R5/R5-C-teaching/data/cache/train",
        "training_pairs": 51200,
        "validation_source": "experiments/D2-A-R5/R5-C-teaching/data/cache/validation",
        "validation_pairs": 5000,
        "pair_plan_source": "experiments/D2-A-R5/R5-C-teaching/training_pair_plan.npz",
        "training_seed_bases_by_course": [1_008_000_000, 1_013_000_000, 1_018_000_000, 1_023_000_000],
        "training_assignment_seed": 2_026_102_831,
        "training_cell_counts": {
            "C0": {"separated": 12800},
            "C1": {"separated": 6400, "near_clutter_edge": 6400},
            "C2": {"separated": 4288, "near_clutter_edge": 4256, "partial_overlap": 4256},
            "C3": {
                "separated_0db": 2144, "separated_10db": 2144,
                "near_clutter_edge_0db": 2128, "near_clutter_edge_10db": 2128,
                "partial_overlap_0db": 2128, "partial_overlap_10db": 2128,
            },
        },
        "validation_seed_bases_by_course": [1_028_000_000, 1_033_000_000, 1_038_000_000, 1_043_000_000],
        "validation_assignment_seed": 2_026_102_832,
        "validation_cell_counts": {
            "C0": {"separated": 1250},
            "C1": {"separated": 625, "near_clutter_edge": 625},
            "C2": {"separated": 417, "near_clutter_edge": 417, "partial_overlap": 416},
            "C3": {
                "separated_0db": 209, "separated_10db": 208,
                "near_clutter_edge_0db": 208, "near_clutter_edge_10db": 209,
                "partial_overlap_0db": 208, "partial_overlap_10db": 208,
            },
        },
        "pair_plan_seed_base": 2026102821,
        "fixed_pair_dataset_passes": 8,
        "reuse_same_pair_order_for_all_arms": True,
        "chunk_size": 16,
        "condition_seed_stride": 1_000_000,
        "chunk_seed_stride": 1009,
        "persist_full_iq": False,
        "persist_teacher_maps": False,
        "final_blind_generation_authorized": False,
    }.items():
        _expect(data, key, value, "data")
    actual_conditions = tuple(
        (
            str(row.get("course")),
            str(row.get("scenario_id")),
            float(row.get("snr_db")),
            float(row.get("cnr_db")),
        )
        for row in data.get("conditions", ())
        if isinstance(row, Mapping)
    )
    if actual_conditions != _CONDITIONS or len(actual_conditions) != len(data.get("conditions", ())):
        raise ValueError("R5-C teaching condition order/physics differs from the draft design")
    for row in data["conditions"]:
        _expect_keys(row, {"course", "scenario_id", "snr_db", "cnr_db"}, "condition")
    split_bases = data.get("split_seed_bases")
    if not isinstance(split_bases, Mapping):
        raise ValueError("R5-C teaching data requires split_seed_bases")
    _expect_keys(split_bases, set(_SPLITS), "split_seed_bases")
    for name, split in _SPLITS.items():
        _expect(split_bases, name, split["seed_base"], "split_seed_bases")
    actual_splits = data.get("splits")
    if not isinstance(actual_splits, Mapping) or set(actual_splits) != set(_SPLITS):
        raise ValueError("R5-C teaching split names differ from the draft design")
    for name, expected in _SPLITS.items():
        actual = actual_splits.get(name)
        if not isinstance(actual, Mapping):
            raise ValueError(f"R5-C teaching split {name} must be a mapping")
        _expect_keys(actual, {"presence", "scene_count", "condition_indices", "condition_counts"}, name)
        for key in ("presence", "scene_count", "condition_indices", "condition_counts"):
            _expect(actual, key, expected[key], name)
        if sum(actual["condition_counts"]) != actual["scene_count"]:
            raise ValueError(f"R5-C teaching split {name} counts do not sum to its scene total")

    for key, value in {
        "arms": ["T0", "T1", "T2", "T3"],
        "initialization_seed": 303,
        "matched_initial_state_bit_identical": True,
        "supervision": "physical_truth_task_loss_all_arms",
        "task_losses": {
            "presence_bce_weight": 1.0,
            "range_cell_cross_entropy_weight": 1.0,
            "range_offset_smooth_l1_weight": 1.0,
            "range_offset_smooth_l1_beta": 0.1,
            "doppler_circular_cosine_weight": 1.0,
        },
        "teacher_loss_weight": 0.05,
        "teacher_loss_weight_schedule": "constant",
        "teacher_loss_rows": "H1_only",
        "teacher_input": "observable_H1_IQ_and_frozen_physics_config_only",
        "teacher_backend": "K3_B3_DPCA_complex128_cuda_4x",
        "teacher_map_storage": "online_update_temporary_only",
        "updates": 12800,
        "paired_scenes_per_update": 32,
        "effective_batch_size_cpi": 64,
        "initial_microbatch_size": 16,
        "oom_microbatch_fallbacks": [8, 4],
        "preserve_effective_batch_on_oom": True,
        "optimizer": "AdamW",
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "learning_rate_initial": 0.001,
        "learning_rate_at_update_6400": 0.0001,
        "learning_rate_final": 0.00001,
        "schedule": "two_segment_R5_cosine",
        "validation_interval_updates": 400,
        "selection": "minimum_common_validation_truth_task_loss",
        "precision": "FP32",
        "amp": False,
        "auxiliary_response_support": "shared_normalized_520x32_joint_map",
        "auxiliary_target_embedding": "local_frequency_target_at_teacher_or_truth_range_gate",
    }.items():
        _expect(training, key, value, "training")

    for key, value in {
        "threshold_target_pfa_cpi": 0.05,
        "score_type": "presence_logit",
        "auc_method": "independent_h0_condition_stratified_h1_count_weighted",
        "range_gate_bins": 1.0,
        "doppler_gate_hz": 20.0,
        "calibration_quantile": 0.95,
        "calibration_quantile_method": "higher",
        "decision_comparator": ">=",
        "exact_pfa_interval": "one_sided_clopper_pearson_95",
        "bootstrap_unit": "paired_base_scene_id",
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 2026102828,
        "primary_scope": "identifiable_C2_C3_joint_success",
        "primary_contrasts": ["T1_minus_T0", "T2_minus_T0"],
        "minimum_primary_gain": 0.02,
        "simultaneous_one_sided_confidence": 0.975,
        "primary_bootstrap_lower_quantile": 0.025,
        "protection_intervals": "nominal_95_percent_pilot_screening",
        "minimum_pd_difference_point": -0.01,
        "minimum_joint_difference_point": -0.01,
        "minimum_protection_ci_lower": -0.05,
        "minimum_c2_c3_joint_ci_lower": -0.05,
        "maximum_c2_c3_rmse_ratio_point": 1.05,
        "maximum_c2_c3_rmse_ratio_nominal_95_upper": 1.10,
        "maximum_c3_h0_pfa_difference_nominal_95_upper": 0.02,
        "c3_h0_pfa_point_limit": 0.10,
        "final_blind_generation_authorized": False,
    }.items():
        _expect(evaluation, key, value, "evaluation")

    for key, value in {
        "gpu": "NVIDIA_GeForce_RTX_3050_Laptop_GPU",
        "device": "cuda:0",
        "cuda_required": True,
        "cpu_fallback": "forbidden",
        "simultaneous_models_on_gpu": 1,
        "minimum_free_vram_bytes": 536870912,
        "minimum_free_disk_bytes": 16106127360,
        "persist_full_iq": False,
        "persist_teacher_maps": False,
        "project_size_final_limit_gib": 20,
        "preserve_untracked_user_files": True,
    }.items():
        _expect(resources, key, value, "resources")
    gates = _section(config, "gates")
    overfit = gates.get("g1_overfit")
    if not isinstance(overfit, Mapping):
        raise ValueError("R5-C teaching config requires the frozen G1 overfit screen")
    for key, value in {
        "paired_scenes": 16,
        "updates": 200,
        "minimum_truth_task_loss_relative_drop": 0.50,
        "minimum_presence_accuracy": 1.0,
        "minimum_h1_range_cell_accuracy": 0.95,
        "task_head_gradients_finite_and_nonzero": True,
        "auxiliary_response_head_gradients_finite_and_nonzero": True,
        "excluded_from_scientific_metrics": True,
    }.items():
        _expect(overfit, key, value, "gates.g1_overfit")


def build_split_seed_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand the fixed split counts into condition/chunk seed specifications."""
    validate_r5c_teaching_config(config)
    data = config["data"]
    chunk_size = int(data["chunk_size"])
    condition_stride = int(data["condition_seed_stride"])
    chunk_stride = int(data["chunk_seed_stride"])
    specs: list[dict[str, Any]] = []
    for split_name in _SPLITS:
        split = data["splits"][split_name]
        split_base = int(data["split_seed_bases"][split_name])
        for local_condition_index, (condition_index, count) in enumerate(
            zip(split["condition_indices"], split["condition_counts"], strict=True)
        ):
            if count <= 0:
                raise ValueError(f"R5-C teaching split {split_name} has an empty condition")
            condition = data["conditions"][condition_index]
            block_start = split_base + local_condition_index * condition_stride
            for chunk_index in range(math.ceil(count / chunk_size)):
                seed = block_start + chunk_index * chunk_stride
                if seed >= block_start + condition_stride:
                    raise ValueError(
                        f"R5-C teaching chunks overflow the seed block for {split_name}"
                    )
                specs.append(
                    {
                        "split": split_name,
                        "presence": split["presence"],
                        "condition_index": local_condition_index,
                        "source_condition_index": condition_index,
                        "condition": dict(condition),
                        "chunk_index": chunk_index,
                        "scene_count": min(chunk_size, count - chunk_index * chunk_size),
                        "seed_block_start": block_start,
                        "seed_block_end_exclusive": block_start + condition_stride,
                        "seed": seed,
                    }
                )
    seeds = [item["seed"] for item in specs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("R5-C teaching generated duplicate chunk seeds")
    return specs


def build_paired_seed_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand fresh paired train/validation generation into reserved seed blocks."""
    validate_r5c_teaching_config(config)
    data = config["data"]
    chunk_size = int(data["chunk_size"])
    chunk_stride = int(data["chunk_seed_stride"])
    specs: list[dict[str, Any]] = []
    for split_name, bases_key, counts_key in (
        ("training", "training_seed_bases_by_course", "training_cell_counts"),
        ("validation", "validation_seed_bases_by_course", "validation_cell_counts"),
    ):
        bases = data[bases_key]
        counts_by_course = data[counts_key]
        for course_index, course in enumerate(("C0", "C1", "C2", "C3")):
            count = sum(int(value) for value in counts_by_course[course].values())
            block_start = int(bases[course_index])
            block_end = block_start + int(data["condition_seed_stride"])
            for chunk_index, start in enumerate(range(0, count, chunk_size)):
                seed = block_start + chunk_index * chunk_stride
                if seed >= block_end:
                    raise ValueError(f"R5-C teaching {split_name} chunks overflow {course} seed block")
                specs.append(
                    {
                        "split": split_name,
                        "presence": "paired_H0_H1",
                        "condition_index": course_index,
                        "course": course,
                        "chunk_index": chunk_index,
                        "scene_count": min(chunk_size, count - start),
                        "seed_block_start": block_start,
                        "seed_block_end_exclusive": block_end,
                        "seed": seed,
                    }
                )
    seeds = [spec["seed"] for spec in specs]
    if len(seeds) != len(set(seeds)):
        raise ValueError("R5-C teaching paired splits contain duplicate chunk seeds")
    return specs


def build_all_seed_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return train, validation, calibration, verification, and development seeds."""
    return build_paired_seed_specs(config) + build_split_seed_specs(config)


def assert_split_seed_disjoint(
    specs: Iterable[Mapping[str, Any]], existing_manifest_seeds: Iterable[int]
) -> None:
    """Stop on duplicated seeds or any existing seed inside a reserved block."""
    spec_list = list(specs)
    if not spec_list:
        raise ValueError("R5-C teaching seed specs must be nonempty")
    try:
        raw_seeds = [item["seed"] for item in spec_list]
        raw_blocks = [
            (item["seed_block_start"], item["seed_block_end_exclusive"])
            for item in spec_list
        ]
        existing = list(existing_manifest_seeds)
    except (KeyError, TypeError) as error:
        raise ValueError("R5-C teaching seed specs/manifests must contain integer seeds") from error
    integer_values = raw_seeds + [value for block in raw_blocks for value in block] + existing
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in integer_values):
        raise ValueError("R5-C teaching seed specs/manifests must contain integer seeds")
    seeds = [int(value) for value in raw_seeds]
    blocks = sorted((int(start), int(end)) for start, end in raw_blocks)
    existing = [int(seed) for seed in existing]
    if any(seed < 0 for seed in seeds + existing):
        raise ValueError("R5-C teaching and existing manifest seeds must be nonnegative")
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed in R5-C teaching split specs")
    if len(existing) != len(set(existing)):
        raise ValueError("duplicate seed in existing manifests")
    prior_start = -1
    prior_end = -1
    for start, end in blocks:
        if start < 0 or end <= start:
            raise ValueError("R5-C teaching seed block bounds are invalid")
        if start < prior_end:
            # Repeated chunks in one condition share the same reserved block.
            # That is valid; a distinct overlapping block start is not.
            if start != prior_start or end != prior_end:
                raise ValueError("R5-C teaching seed blocks overlap")
        prior_start, prior_end = start, end
    for spec, seed in zip(spec_list, seeds, strict=True):
        if not int(spec["seed_block_start"]) <= seed < int(spec["seed_block_end_exclusive"]):
            raise ValueError("R5-C teaching chunk seed falls outside its reserved block")
    for seed in existing:
        if any(start <= seed < end for start, end in blocks):
            raise ValueError(f"R5-C teaching seed collision with existing manifest seed {seed}")


def _require_training_authorization(config: Mapping[str, Any]) -> None:
    validate_r5c_teaching_config(config)
    experiment = config["experiment"]
    review = experiment["independent_chatgpt_review"]
    if (
        experiment["status"] != "preregistered"
        or review["status"] != "returned"
        or review["supports_pilot"] is not True
        or experiment["run_authorized"] is not True
        or config["training"]["execution_authorized"] is not True
    ):
        raise RuntimeError(
            "independent R5-C review must support the pilot and protocol/config must be frozen before training"
        )


def _require_training_cuda(device: str | torch.device) -> torch.device:
    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid R5-C teaching device: {device!r}") from error
    if selected.type != "cuda" or selected.index != 0:
        raise ValueError("R5-C teaching training requires explicit cuda:0; CPU fallback is forbidden")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R5-C teaching requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    expected = "NVIDIA GeForce RTX 3050 Laptop GPU"
    actual = torch.cuda.get_device_name(0)
    if actual != expected:
        raise RuntimeError(f"R5-C teaching requires {expected}, found {actual}")
    return torch.device("cuda:0")


def build_initial_state(seed: int) -> dict[str, torch.Tensor]:
    """Create the shared Medium/520-bin CPU initialization deterministically."""
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("initialization seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = TaskStudent(width=96, spectrum_frequency_bins=520)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    del model
    return state


def build_arm_model(arm: str, initial_state: Mapping[str, torch.Tensor]) -> TaskStudent:
    """Build a matched arm from the same frozen initialization state."""
    if arm not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("arm must be one of T0, T1, T2, or T3")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(303)
        model = TaskStudent(width=96, spectrum_frequency_bins=520)
    model.load_state_dict(initial_state, strict=True)
    return model


def compute_h1_teacher_maps(
    iq_h1: np.ndarray,
    physics_config: Mapping[str, Any],
    device: str | torch.device,
) -> torch.Tensor:
    """Compute temporary K3 maps for only the current H1 IQ microbatch."""
    values = np.asarray(iq_h1)
    if values.ndim != 4 or values.shape[0] <= 0 or values.shape[1:] != (3, 130, 32):
        raise ValueError("iq_h1 must have shape [B,3,130,32] with a nonempty batch")
    if not np.iscomplexobj(values) or not np.isfinite(values).all():
        raise ValueError("iq_h1 must contain finite complex samples")
    selected = _require_training_cuda(device)
    maps = k2_response_cuda(
        values,
        dict(physics_config),
        device="cuda:0",
        oversampling_factor=4,
    )
    result = torch.as_tensor(maps, dtype=torch.float32, device=selected)
    if result.shape != (values.shape[0], 520, 32) or not bool(torch.isfinite(result).all()):
        raise ValueError("K3 CUDA teacher must return finite maps with shape [B,520,32]")
    return result


def h1_response_loss(
    outputs: Mapping[str, torch.Tensor],
    truth: Mapping[str, torch.Tensor],
    arm: str,
    *,
    teacher_map: torch.Tensor | None,
) -> torch.Tensor:
    """Compute an auxiliary response loss on H1 rows only."""
    if arm not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("arm must be one of T0, T1, T2, or T3")
    if "truth_presence" not in truth:
        raise ValueError("truth must contain truth_presence")
    reference = outputs["spectrum_logits"] if "spectrum_logits" in outputs else outputs["presence_logit"]
    presence = torch.as_tensor(truth["truth_presence"], device=reference.device).reshape(-1).bool()
    if presence.shape != (reference.shape[0],):
        raise ValueError("truth_presence must have one value per output row")
    if arm == "T0":
        if teacher_map is not None:
            raise ValueError("T0 cannot receive teacher maps")
        return reference.sum() * 0.0
    if not bool(torch.any(presence)):
        if teacher_map is not None:
            raise ValueError("empty H1 batch cannot receive teacher maps")
        return reference.sum() * 0.0
    logits = outputs.get("spectrum_logits")
    if logits is None:
        raise ValueError(f"{arm} requires the training-only spectrum head")
    if arm in {"T1", "T2"}:
        if teacher_map is None:
            raise ValueError(f"{arm} requires K3 maps for H1 rows")
        if teacher_map.shape[0] != int(presence.sum().item()):
            raise ValueError("teacher_map must contain one map per H1 row and no H0 maps")
        return response_kl(logits[presence], arm, teacher_map=teacher_map)
    if teacher_map is not None:
        raise ValueError("T3 uses truth-assisted analytic targets, not teacher maps")
    return response_kl(
        logits[presence],
        "T3",
        truth_range_cell=truth["truth_range_cell"][presence],
        truth_doppler_hz=truth["truth_doppler_hz"][presence],
    )


def _pilot_task_config(config: Mapping[str, Any]) -> dict[str, Any]:
    training = config["training"]
    task_losses = dict(training["task_losses"])
    return {
        "radar": {"prf_hz": float(config["radar"]["prf_hz"])},
        "training": {
            "loss": task_losses,
            "losses": task_losses,
            "effective_batch_size": int(training["effective_batch_size_cpi"]),
            "paired_base_scenes_per_update": int(training["paired_scenes_per_update"]),
        },
    }


def build_pilot_schedule(config: Mapping[str, Any]) -> R5CosineSchedule:
    """Create the R5 two-segment schedule with its frozen update boundaries."""
    training = config["training"]
    return R5CosineSchedule(
        initial_updates=6400,
        final_updates=12800,
        initial_lr=float(training["learning_rate_initial"]),
        initial_final_lr=float(training["learning_rate_at_update_6400"]),
        extension_final_lr=float(training["learning_rate_final"]),
    )


def scale_pilot_microbatch_loss(
    losses: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    arm: str,
    microbatch_size: int,
    effective_batch_size: int,
    microbatch_h1: int,
    effective_h1: int,
) -> torch.Tensor:
    """Scale truth and response terms to the matched effective batch."""
    if arm not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("arm must be one of T0, T1, T2, or T3")
    task_config = _pilot_task_config(config)
    teacher_weight = 0.0 if arm == "T0" else float(config["training"]["teacher_loss_weight"])
    return scale_d2a_e002_microbatch_loss(
        losses,
        task_config,
        microbatch_size=microbatch_size,
        effective_batch_size=effective_batch_size,
        microbatch_h1=microbatch_h1,
        effective_h1=effective_h1,
        teacher_weight=teacher_weight,
    )


def compute_training_microbatch_loss(
    model: TaskStudent,
    batch: Mapping[str, np.ndarray],
    arm: str,
    config: Mapping[str, Any],
    device: str | torch.device,
    *,
    physics_config: Mapping[str, Any],
    h1_iq: np.ndarray | None,
    effective_batch_size: int | None = None,
    effective_h1: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute one fixed-size training microbatch without retaining teacher maps."""
    selected = torch.device(device)
    x = torch.as_tensor(batch["iq_ri"], dtype=torch.float32, device=selected)
    truth = {
        key: torch.as_tensor(batch[key], device=selected)
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    if x.shape[0] <= 0 or x.shape[0] % 2:
        raise ValueError("pilot microbatches must contain complete H0/H1 pairs")
    outputs = model(x, compute_spectrum=(arm != "T0"))
    h1_count = int(truth["truth_presence"].sum().item())
    if arm in {"T1", "T2"}:
        if h1_iq is None or len(h1_iq) != h1_count:
            raise ValueError("T1/T2 require only the paired H1 IQ rows for teacher computation")
        teacher_map = compute_h1_teacher_maps(h1_iq, physics_config, selected)
    else:
        if h1_iq is not None:
            raise ValueError("T0/T3 must not compute or receive teacher IQ/maps")
        teacher_map = None
    task_terms = task_loss(outputs, truth, _pilot_task_config(config), teacher_weight=0.0)
    auxiliary = h1_response_loss(outputs, truth, arm, teacher_map=teacher_map)
    losses = dict(task_terms)
    losses["teacher_kl"] = auxiliary
    scaled = scale_pilot_microbatch_loss(
        losses,
        config,
        arm=arm,
        microbatch_size=int(x.shape[0]),
        effective_batch_size=int(
            effective_batch_size or config["training"]["effective_batch_size_cpi"]
        ),
        microbatch_h1=h1_count,
        effective_h1=int(effective_h1 or config["training"]["paired_scenes_per_update"]),
    )
    return scaled, {**task_terms, "teacher_kl": auxiliary, "scaled_total": scaled}


def _h1_iq_for_microbatch(
    train_arrays: Mapping[str, np.ndarray], pair_ids: np.ndarray, micro_start: int, micro_stop: int
) -> np.ndarray:
    if micro_start < 0 or micro_stop <= micro_start or micro_start % 2 or micro_stop % 2:
        raise ValueError("microbatch boundaries must preserve the interleaved H0/H1 pair order")
    first_pair, stop_pair = micro_start // 2, micro_stop // 2
    ids = np.asarray(pair_ids, dtype=np.int64).reshape(-1)
    if stop_pair > len(ids):
        raise ValueError("microbatch extends beyond this update's paired scene IDs")
    return np.asarray(train_arrays["iq_h1"][ids[first_pair:stop_pair]])


def _validate_training_arrays(arrays: Mapping[str, np.ndarray], expected_count: int) -> None:
    required = ("base_scene_id", "iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz")
    if any(key not in arrays for key in required):
        raise ValueError("R5-C training arrays are missing paired IQ, IDs, or physical truth")
    if len(arrays["base_scene_id"]) != expected_count:
        raise ValueError(f"R5-C training split must contain {expected_count} paired scenes")
    ids = np.asarray(arrays["base_scene_id"]).reshape(-1).astype(str)
    if np.unique(ids).size != expected_count:
        raise ValueError("R5-C paired training base_scene_id values must be unique")
    for key in ("iq_h0", "iq_h1"):
        value = arrays[key]
        if value.shape != (expected_count, 3, 130, 32) or not np.iscomplexobj(value):
            raise ValueError(f"{key} must have shape [N,3,130,32] and complex dtype")
    for key in ("truth_range_cell", "truth_doppler_hz"):
        value = np.asarray(arrays[key])
        if value.shape != (expected_count,) or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain one finite value per paired scene")


def _validate_pair_plan(pair_plan: np.ndarray, scene_count: int) -> np.ndarray:
    plan = np.asarray(pair_plan)
    expected_shape = (12800, 32)
    if plan.shape != expected_shape or not np.issubdtype(plan.dtype, np.integer):
        raise ValueError(f"R5-C teaching pair plan must have shape {expected_shape} and integer indices")
    if np.any(plan < 0) or np.any(plan >= scene_count):
        raise ValueError("R5-C teaching pair plan contains out-of-range training scene indices")
    for epoch in range(8):
        values = plan[epoch * 1600 : (epoch + 1) * 1600].reshape(-1)
        if not np.array_equal(np.sort(values), np.arange(scene_count)):
            raise ValueError("each R5-C teaching epoch must traverse every fixed training scene once")
    return plan.astype(np.int64, copy=False)


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _append_history(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "update", "epoch", "arm", "microbatch_size", "learning_rate",
        "train_presence", "train_range_cell", "train_range_offset", "train_doppler",
        "train_teacher_kl", "train_scaled_total", "validation_task_loss",
        "validation_presence_loss", "validation_range_cell_loss", "validation_range_offset_loss",
        "validation_doppler_loss",
    )
    new_file = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fields})
        stream.flush()
        os.fsync(stream.fileno())


def _truncate_history_to_update(path: Path, cursor: int) -> None:
    """Discard update rows beyond the latest resumable checkpoint cursor."""
    if isinstance(cursor, bool) or not isinstance(cursor, Integral) or cursor < 0:
        raise ValueError("training history cursor must be a nonnegative integer")
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or "update" not in fields:
            raise ValueError("R5-C teaching training history must contain an update column")
        rows = [row for row in reader if int(row["update"]) <= int(cursor)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())


def _task_head_gradient_norms(
    model: TaskStudent, losses: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    """Measure per-task gradients at all four direct output heads."""
    heads = (
        ("presence", "presence_per_range", "presence"),
        ("range", "range_head", "range_cell"),
        ("range_offset", "range_offset_head", "range_offset"),
        ("doppler", "doppler_head", "doppler"),
    )
    norms: dict[str, float] = {}
    for output, attribute, loss_name in heads:
        module = getattr(model, attribute)
        gradients = torch.autograd.grad(
            losses[loss_name], tuple(module.parameters()), retain_graph=True, allow_unused=True
        )
        squared_norms = [
            gradient.detach().double().square().sum()
            for gradient in gradients
            if gradient is not None
        ]
        norms[output] = (
            float(torch.sqrt(torch.stack(squared_norms).sum()).item())
            if squared_norms
            else 0.0
        )
    return norms


def _auxiliary_head_gradient_norm(model: TaskStudent, loss: torch.Tensor) -> float:
    """Measure the spectrum head's response to its auxiliary objective."""
    head = getattr(model, "spectrum_head", None)
    if head is None:
        return 0.0
    gradients = torch.autograd.grad(
        loss, tuple(head.parameters()), retain_graph=True, allow_unused=True
    )
    squared_norms = [
        gradient.detach().double().square().sum()
        for gradient in gradients
        if gradient is not None
    ]
    return (
        float(torch.sqrt(torch.stack(squared_norms).sum()).item())
        if squared_norms
        else 0.0
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _require_training_main_head() -> str:
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=_PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    head = _git_head()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=_PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if branch != "main" or head != origin:
        raise RuntimeError("R5-C teaching training requires the frozen code pushed to main")
    for relative in FROZEN_INPUTS:
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=_PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        if tracked.returncode != 0:
            raise RuntimeError(f"R5-C teaching frozen input is not tracked: {relative}")
        unchanged = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative], cwd=_PROJECT_ROOT
        )
        if unchanged.returncode != 0:
            raise RuntimeError(f"R5-C teaching frozen input has local changes: {relative}")
    return head


def _require_training_preflight(
    config: Mapping[str, Any], *, require_g1: bool = False
) -> None:
    head = _require_training_main_head()
    preflight_path = _STAGE_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("R5-C teaching requires a passed preflight before execution")
    try:
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("R5-C teaching preflight record is unreadable") from error
    if (
        preflight.get("git_sha") != head
        or preflight.get("passed") is not True
        or preflight.get("config") != dict(config)
    ):
        raise RuntimeError("R5-C teaching preflight does not match the pushed code/config")
    if not require_g1:
        return
    g1_path = _STAGE_ROOT / "g1_overfit_sanity.json"
    if not g1_path.is_file():
        raise RuntimeError("R5-C teaching training requires the G1 overfit gate")
    try:
        g1 = json.loads(g1_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("R5-C teaching G1 record is unreadable") from error
    arms = g1.get("arms")
    if (
        g1.get("git_sha") != head
        or g1.get("config") != dict(config)
        or g1.get("passed") is not True
        or not isinstance(arms, Mapping)
        or set(arms) != set(config["training"]["arms"])
        or any(arms[arm].get("passed") is not True for arm in config["training"]["arms"])
    ):
        raise RuntimeError("R5-C teaching training requires a passing G1 record for all arms")


def train_arm(
    arm: str,
    initial_state: Mapping[str, torch.Tensor],
    train_arrays: Mapping[str, np.ndarray],
    pair_plan: np.ndarray,
    config: Mapping[str, Any],
    device: str | torch.device,
    microbatch: int,
    *,
    validation_arrays: Mapping[str, np.ndarray],
    arm_root: Path,
    physics_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Train one authorized arm, checkpointing state and pair-plan cursor."""
    _require_training_authorization(config)
    _require_training_preflight(config, require_g1=True)
    if arm not in config["training"]["arms"]:
        raise ValueError(f"unregistered R5-C teaching arm: {arm}")
    selected = _require_training_cuda(device)
    if isinstance(microbatch, bool) or microbatch not in (16, 8, 4):
        raise ValueError("R5-C teaching microbatch must be 16, 8, or 4")
    if 64 % microbatch or microbatch % 2:
        raise ValueError("R5-C teaching microbatch must evenly preserve the effective batch and pairs")
    _validate_training_arrays(train_arrays, 51200)
    _validate_training_arrays(validation_arrays, 5000)
    plan = _validate_pair_plan(pair_plan, len(train_arrays["base_scene_id"]))
    output = Path(arm_root)
    output.mkdir(parents=True, exist_ok=True)
    loss_config = _pilot_task_config(config)
    torch.manual_seed(int(config["training"]["initialization_seed"]))
    torch.cuda.manual_seed_all(int(config["training"]["initialization_seed"]))
    model = build_arm_model(arm, initial_state).to(selected)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate_initial"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    schedule = build_pilot_schedule(config)
    history_path = output / "training_history.csv"
    best_path = output / "best_checkpoint.pt"
    state_path = output / "latest_training_state.pt"
    source_sha = _git_head()
    start_update = 0
    best_loss = float("inf")
    best_update = 0

    def save_resume(global_update: int, validation: Mapping[str, Any]) -> None:
        _atomic_torch_save(
            {
                "experiment": "D2-A-R5",
                "subphase": "R5-C-teaching",
                "protocol_version": float(config["experiment"]["protocol_version"]),
                "arm": arm,
                "source_git_sha": source_sha,
                "config": dict(config),
                "global_update": int(global_update),
                "data_cursor": int(global_update),
                "microbatch_size": int(microbatch),
                "model_state_dict": _cpu_state_dict(model),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": schedule.state_dict(),
                "rng_state": _capture_rng_state(),
                "best_validation_loss": float(best_loss),
                "best_update": int(best_update),
                "latest_validation": dict(validation),
                "pair_plan_seed_base": int(config["data"]["pair_plan_seed_base"]),
            },
            state_path,
        )

    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        identity = (state.get("experiment"), state.get("subphase"), state.get("arm"))
        if identity != ("D2-A-R5", "R5-C-teaching", arm):
            raise ValueError("R5-C teaching resume checkpoint identity mismatch")
        if state.get("source_git_sha") != source_sha or state.get("config") != dict(config):
            raise ValueError("R5-C teaching resume source/config differs from this committed run")
        if int(state.get("pair_plan_seed_base", -1)) != int(config["data"]["pair_plan_seed_base"]):
            raise ValueError("R5-C teaching resume pair-plan seed differs from config")
        start_update = int(state["global_update"])
        if not 0 <= start_update <= 12800:
            raise ValueError("R5-C teaching resume update is outside the training schedule")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        schedule_state = state["scheduler_state_dict"]
        schedule = R5CosineSchedule.from_state_dict(schedule_state)
        best_loss = float(state["best_validation_loss"])
        best_update = int(state["best_update"])
        _restore_rng_state(state["rng_state"])
    else:
        validation = _paired_validation(model, validation_arrays, loss_config, selected, microbatch)
        if not np.isfinite(float(validation["task_loss"])):
            raise FloatingPointError("R5-C teaching initial validation loss is non-finite")
        best_loss = float(validation["task_loss"])
        best_update = 0
        _atomic_torch_save(
            {
                "experiment": "D2-A-R5",
                "subphase": "R5-C-teaching",
                "arm": arm,
                "source_git_sha": source_sha,
                "config": dict(config),
                "global_update": 0,
                "model_state_dict": _cpu_state_dict(model),
                "validation": dict(validation),
            },
            best_path,
        )
        _append_history(
            history_path,
            {
                "update": 0,
                "epoch": 0,
                "arm": arm,
                "microbatch_size": microbatch,
                "learning_rate": config["training"]["learning_rate_initial"],
                "validation_task_loss": validation["task_loss"],
                "validation_presence_loss": validation["presence_loss"],
                "validation_range_cell_loss": validation["range_cell_loss"],
                "validation_range_offset_loss": validation["range_offset_loss"],
                "validation_doppler_loss": validation["doppler_loss"],
            },
        )
        save_resume(0, validation)

    # The cursor may still be zero if interruption occurred before the first
    # validation checkpoint; in that case remove all uncheckpointed updates.
    _truncate_history_to_update(history_path, start_update)

    torch.cuda.reset_peak_memory_stats(selected)
    started = time.perf_counter()
    latest_validation: dict[str, Any] = {}
    model.train()
    for update_index in range(start_update, 12800):
        update = update_index + 1
        learning_rate = schedule.lr_for_update(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        pair_ids = plan[update_index]
        paired = make_paired_examples(train_arrays, pair_ids, include_teacher=False)
        optimizer.zero_grad(set_to_none=True)
        accum = {key: 0.0 for key in ("presence", "range_cell", "range_offset", "doppler", "teacher_kl", "scaled_total")}
        for micro_start in range(0, 64, microbatch):
            micro_stop = micro_start + microbatch
            batch = {key: value[micro_start:micro_stop] for key, value in paired.items()}
            h1_iq = (
                _h1_iq_for_microbatch(train_arrays, pair_ids, micro_start, micro_stop)
                if arm in {"T1", "T2"}
                else None
            )
            scaled, terms = compute_training_microbatch_loss(
                model,
                batch,
                arm,
                config,
                selected,
                physics_config=physics_config,
                h1_iq=h1_iq,
            )
            if not bool(torch.isfinite(scaled)):
                raise FloatingPointError(f"non-finite R5-C teaching {arm} loss at update {update}")
            scaled.backward()
            sample_fraction = (micro_stop - micro_start) / 64.0
            h1_fraction = int(np.count_nonzero(batch["truth_presence"])) / 32.0
            accum["presence"] += float(terms["presence"].detach().item()) * sample_fraction
            for key in ("range_cell", "range_offset", "doppler", "teacher_kl"):
                accum[key] += float(terms[key].detach().item()) * h1_fraction
            accum["scaled_total"] += float(scaled.detach().item())
            del batch, h1_iq, scaled, terms
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["training"]["gradient_clip_norm"])
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError(f"non-finite R5-C teaching {arm} gradient at update {update}")
        optimizer.step()
        row: dict[str, Any] = {
            "update": update,
            "epoch": int(math.ceil(update / 1600)),
            "arm": arm,
            "microbatch_size": microbatch,
            "learning_rate": learning_rate,
            **{f"train_{key}": value for key, value in accum.items()},
        }
        if update % int(config["training"]["validation_interval_updates"]) == 0:
            model.eval()
            latest_validation = _paired_validation(
                model, validation_arrays, loss_config, selected, microbatch
            )
            if not all(np.isfinite(float(value)) for value in latest_validation.values()):
                raise FloatingPointError(f"non-finite R5-C teaching {arm} validation at update {update}")
            row.update(
                {
                    "validation_task_loss": latest_validation["task_loss"],
                    "validation_presence_loss": latest_validation["presence_loss"],
                    "validation_range_cell_loss": latest_validation["range_cell_loss"],
                    "validation_range_offset_loss": latest_validation["range_offset_loss"],
                    "validation_doppler_loss": latest_validation["doppler_loss"],
                }
            )
            if float(latest_validation["task_loss"]) < best_loss:
                best_loss = float(latest_validation["task_loss"])
                best_update = update
                _atomic_torch_save(
                    {
                        "experiment": "D2-A-R5",
                        "subphase": "R5-C-teaching",
                        "arm": arm,
                        "source_git_sha": source_sha,
                        "config": dict(config),
                        "global_update": update,
                        "model_state_dict": _cpu_state_dict(model),
                        "validation": dict(latest_validation),
                    },
                    best_path,
                )
            model.train()
        _append_history(history_path, row)
        if update % int(config["training"]["validation_interval_updates"]) == 0:
            save_resume(update, latest_validation)
        del paired

    torch.cuda.synchronize(selected)
    training_seconds = time.perf_counter() - started
    result = {
        "arm": arm,
        "updates_completed": 12800,
        "unique_training_pairs": 51200,
        "training_pair_scene_exposures": 12800 * 32,
        "cpi_example_exposures": 12800 * 64,
        "fixed_pair_dataset_passes": 8,
        "effective_cpi_per_update": 64,
        "best_update": best_update,
        "best_validation_task_loss": best_loss,
        "microbatch_size": microbatch,
        "effective_batch_size_cpi": 64,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "training_seconds": training_seconds,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(selected)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(selected)),
        "history_path": str(history_path),
        "best_checkpoint_path": str(best_path),
        "resume_checkpoint_path": str(state_path),
    }
    del optimizer, model
    return result


def run_overfit_sanity(
    *,
    initial_state: Mapping[str, torch.Tensor],
    train_arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    device: str | torch.device,
    physics_config: Mapping[str, Any],
    microbatch: int = 16,
) -> dict[str, Any]:
    """Run the frozen 16-pair, 200-update engineering sanity check per arm."""
    _require_training_authorization(config)
    _require_training_preflight(config)
    selected = _require_training_cuda(device)
    pair_count = int(config["gates"]["g1_overfit"]["paired_scenes"])
    updates = int(config["gates"]["g1_overfit"]["updates"])
    if pair_count != 16 or updates != 200:
        raise ValueError("R5-C teaching G1 overfit budget differs from its frozen configuration")
    if isinstance(microbatch, bool) or microbatch not in (16, 8, 4):
        raise ValueError("R5-C teaching G1 microbatch must be 16, 8, or 4")
    effective_batch_size = 2 * pair_count
    if effective_batch_size % microbatch or microbatch % 2:
        raise ValueError("R5-C teaching G1 microbatch must preserve complete H0/H1 pairs")
    _validate_training_arrays(train_arrays, 51200)
    pair_ids = np.arange(pair_count, dtype=np.int64)
    paired = make_paired_examples(train_arrays, pair_ids, include_teacher=False)
    initial_truth = {
        key: torch.as_tensor(paired[key], device=selected)
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    results: dict[str, Any] = {"pair_count": pair_count, "updates": updates, "arms": {}}
    for arm in config["training"]["arms"]:
        torch.manual_seed(int(config["training"]["initialization_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["initialization_seed"]))
        model = build_arm_model(arm, initial_state).to(selected)
        model.eval()
        x = torch.as_tensor(paired["iq_ri"], dtype=torch.float32, device=selected)
        initial_h1_iq = (
            _h1_iq_for_microbatch(train_arrays, pair_ids, 0, effective_batch_size)
            if arm in {"T1", "T2"}
            else None
        )
        initial_scaled_loss, initial_terms = compute_training_microbatch_loss(
            model,
            paired,
            arm,
            config,
            selected,
            physics_config=physics_config,
            h1_iq=initial_h1_iq,
            effective_batch_size=effective_batch_size,
            effective_h1=pair_count,
        )
        initial_task_loss = float(initial_terms["task_total"].detach().item())
        gradient_norms = _task_head_gradient_norms(model, initial_terms)
        auxiliary_gradient_norm = _auxiliary_head_gradient_norm(
            model, initial_terms["teacher_kl"]
        )
        auxiliary_gradient_passed = (
            auxiliary_gradient_norm == 0.0
            if arm == "T0"
            else np.isfinite(auxiliary_gradient_norm) and auxiliary_gradient_norm > 0.0
        )
        gradients_passed = (
            all(np.isfinite(value) and value > 0.0 for value in gradient_norms.values())
            and auxiliary_gradient_passed
        )
        del initial_scaled_loss, initial_terms

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(config["training"]["learning_rate_initial"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        last_gradient_norm = float("nan")
        model.train()
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, effective_batch_size, microbatch):
                micro_stop = micro_start + microbatch
                batch = {key: value[micro_start:micro_stop] for key, value in paired.items()}
                h1_iq = _h1_iq_for_microbatch(
                    train_arrays, pair_ids, micro_start, micro_stop
                ) if arm in {"T1", "T2"} else None
                scaled, _ = compute_training_microbatch_loss(
                    model,
                    batch,
                    arm,
                    config,
                    selected,
                    physics_config=physics_config,
                    h1_iq=h1_iq,
                    effective_batch_size=effective_batch_size,
                    effective_h1=pair_count,
                )
                if not bool(torch.isfinite(scaled)):
                    raise FloatingPointError(f"non-finite R5-C teaching overfit loss for {arm}")
                scaled.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise FloatingPointError(f"non-finite R5-C teaching overfit gradient for {arm}")
            last_gradient_norm = float(gradient_norm.detach().item())
            optimizer.step()

        model.eval()
        with torch.no_grad():
            final_outputs = model(x, compute_spectrum=(arm != "T0"))
            final_terms = task_loss(
                final_outputs, initial_truth, _pilot_task_config(config), teacher_weight=0.0
            )
            h1_mask = initial_truth["truth_presence"].bool()
            presence_correct = (
                (final_outputs["presence_logit"] >= 0.0) == initial_truth["truth_presence"].bool()
            )
            range_accuracy = (
                final_outputs["range_logits"][h1_mask].argmax(dim=-1)
                == torch.floor(initial_truth["truth_range_cell"][h1_mask]).long()
            ).float().mean()
        final_task_loss = float(final_terms["task_total"].item())
        loss_ratio = final_task_loss / max(initial_task_loss, np.finfo(np.float64).tiny)
        presence_accuracy = float(presence_correct.float().mean().item())
        h1_range_accuracy = float(range_accuracy.item())
        passed = bool(
            gradients_passed
            and np.isfinite(last_gradient_norm)
            and loss_ratio <= 1.0 - float(
                config["gates"]["g1_overfit"]["minimum_truth_task_loss_relative_drop"]
            )
            and presence_accuracy >= float(
                config["gates"]["g1_overfit"]["minimum_presence_accuracy"]
            )
            and h1_range_accuracy >= float(
                config["gates"]["g1_overfit"]["minimum_h1_range_cell_accuracy"]
            )
        )
        results["arms"][arm] = {
            "updates": updates,
            "paired_scenes": pair_count,
            "microbatch_size": microbatch,
            "initial_truth_task_loss": initial_task_loss,
            "final_truth_task_loss": final_task_loss,
            "loss_ratio_final_to_initial": loss_ratio,
            "presence_accuracy": presence_accuracy,
            "h1_range_cell_accuracy": h1_range_accuracy,
            "initial_task_head_gradient_norms": gradient_norms,
            "initial_auxiliary_response_head_gradient_norm": auxiliary_gradient_norm,
            "auxiliary_response_gradient_passed": bool(auxiliary_gradient_passed),
            "last_gradient_norm": last_gradient_norm,
            "passed": passed,
        }
        results["passed"] = results.get("passed", True) and passed
        del optimizer, model, x
        torch.cuda.empty_cache()
    return results


def train_arms_sequentially(
    *,
    initial_state: Mapping[str, torch.Tensor],
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    pair_plan: np.ndarray,
    config: Mapping[str, Any],
    device: str | torch.device,
    microbatch: int,
    arm_root: Path,
    physics_config: Mapping[str, Any],
    trainer: Any | None = None,
) -> list[dict[str, Any]]:
    """Run T0–T3 in fixed order, retrying OOM with smaller microbatches."""
    train_fn = train_arm if trainer is None else trainer
    if list(config["training"]["arms"]) != ["T0", "T1", "T2", "T3"]:
        raise ValueError("R5-C teaching arms must run sequentially in T0, T1, T2, T3 order")
    results: list[dict[str, Any]] = []
    root = Path(arm_root)
    for arm in config["training"]["arms"]:
        attempts: list[int] = []
        candidates = [microbatch, *[value for value in (8, 4) if value < microbatch]]
        for candidate in candidates:
            attempts.append(candidate)
            try:
                result = train_fn(
                    arm,
                    initial_state=initial_state,
                    train_arrays=train_arrays,
                    validation_arrays=validation_arrays,
                    pair_plan=pair_plan,
                    config=config,
                    device=device,
                    microbatch=candidate,
                    arm_root=root / arm,
                    physics_config=physics_config,
                )
                result["oom_microbatch_attempts"] = attempts
                results.append(result)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if candidate == candidates[-1]:
                    raise
        torch.cuda.empty_cache()
    return results
