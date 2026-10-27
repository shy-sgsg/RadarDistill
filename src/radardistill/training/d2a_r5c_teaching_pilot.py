"""Draft protocol and disjoint seed allocation for the R5-C teaching pilot."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Integral, Real
from typing import Any


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

_SPLITS = {
    "h0_calibration": {
        "presence": "H0",
        "scene_count": 8192,
        "condition_indices": list(range(12)),
        "condition_counts": [2048, 1024, 1024, 683, 683, 682, 342, 342, 341, 341, 341, 341],
        "seed_base": 728_000_000,
    },
    "h0_verification": {
        "presence": "H0",
        "scene_count": 8192,
        "condition_indices": list(range(12)),
        "condition_counts": [2048, 1024, 1024, 683, 683, 682, 342, 342, 341, 341, 341, 341],
        "seed_base": 748_000_000,
    },
    "c3_h0_verification": {
        "presence": "H0",
        "scene_count": 4096,
        "condition_indices": list(range(6, 12)),
        "condition_counts": [683, 683, 683, 683, 682, 682],
        "seed_base": 768_000_000,
    },
    "h1_development": {
        "presence": "H1",
        "scene_count": 4352,
        "condition_indices": list(range(12)),
        "condition_counts": [256, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
        "seed_base": 788_000_000,
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
    """Validate the exact draft pilot design and keep execution unauthorized."""
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
            "training_source", "training_pairs", "validation_source", "validation_pairs",
            "pair_plan_seed_base", "fixed_pair_dataset_passes", "reuse_same_pair_order_for_all_arms",
            "chunk_size", "condition_seed_stride", "chunk_seed_stride", "split_seed_bases",
            "conditions", "splits", "persist_full_iq", "persist_teacher_maps",
            "final_blind_generation_authorized",
        },
        "training": {
            "arms", "initialization_seed", "matched_initial_state_bit_identical", "supervision",
            "teacher_loss_weight", "teacher_loss_weight_schedule", "teacher_loss_rows", "teacher_input",
            "teacher_backend", "teacher_map_storage", "updates", "paired_scenes_per_update",
            "effective_batch_size_cpi", "initial_microbatch_size", "oom_microbatch_fallbacks",
            "preserve_effective_batch_on_oom", "optimizer", "weight_decay", "gradient_clip_norm",
            "learning_rate_initial", "learning_rate_at_update_6400", "learning_rate_final", "schedule",
            "validation_interval_updates", "selection", "precision", "amp", "execution_authorized",
        },
        "evaluation": {
            "threshold_target_pfa_cpi", "calibration_quantile", "calibration_quantile_method",
            "decision_comparator", "exact_pfa_interval", "bootstrap_unit", "bootstrap_replicates",
            "bootstrap_seed", "primary_scope", "primary_contrasts", "minimum_primary_gain",
            "simultaneous_one_sided_confidence", "primary_bootstrap_lower_quantile",
            "protection_intervals", "c3_h0_pfa_point_limit", "final_blind_generation_authorized",
        },
        "gates": {"g1_overfit"},
        "resources": {
            "gpu", "device", "cuda_required", "cpu_fallback", "simultaneous_models_on_gpu",
            "persist_full_iq", "persist_teacher_maps", "project_size_final_limit_gib",
            "preserve_untracked_user_files",
        },
    }
    for name, keys in expected_section_keys.items():
        _expect_keys(_section(config, name), keys, name)

    for key, value in {
        "id": "D2-A-R5",
        "subphase": "R5-C-teaching",
        "protocol_version": 0.1,
        "status": "draft",
        "charter_version": 1.3,
        "independent_chatgpt_review": "pending",
        "run_authorized": False,
        "final_blind_generation_authorized": False,
    }.items():
        _expect(experiment, key, value, "experiment")

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
        "training_source": "experiments/D2-A-R5/R5-A/training_pair_plan.npz",
        "training_pairs": 51200,
        "validation_source": "frozen_R5-A_common_validation_ids",
        "validation_pairs": 5000,
        "pair_plan_seed_base": 2026102711,
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
        "execution_authorized": False,
    }.items():
        _expect(training, key, value, "training")

    for key, value in {
        "threshold_target_pfa_cpi": 0.05,
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
