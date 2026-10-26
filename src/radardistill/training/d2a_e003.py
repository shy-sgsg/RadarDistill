"""Frozen data-plan validation helpers for D2-A-E003."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np


COURSE_LEVELS = ("C0", "C1", "C2", "C3")
PROTOCOL_VERSION = 1.1


def _chunk_seeds(seed_base: int, count: int, chunk_size: int, stride: int) -> list[int]:
    if seed_base < 0 or count <= 0 or chunk_size <= 0 or stride <= 0:
        raise ValueError("seed base, count, chunk size and stride must be valid")
    return [int(seed_base) + index * int(stride) for index in range(math.ceil(count / chunk_size))]


def e003_seed_manifest(config: Mapping[str, Any]) -> dict[str, list[int]]:
    """Expand all non-repeated E003 generation streams before data creation."""
    data = config["data"]
    chunk = int(data["chunk_size"])
    stride = int(data["scene_seed_stride"])
    streams: dict[str, list[int]] = {}

    for split in ("train", "validation"):
        section = data[split]
        count = int(section["pairs_per_course_level"])
        for level, base in zip(COURSE_LEVELS, section["seed_bases_by_course"], strict=True):
            streams[f"{split}_{level}"] = _chunk_seeds(int(base), count, chunk, stride)
    for split in ("h0_calibration", "h0_verification"):
        section = data[split]
        streams[split] = _chunk_seeds(
            int(section["seed_base"]), int(section["scenes"]),
            int(section["generation_chunk_size"]), stride,
        )
    c3 = data["c3_h0_verification"]
    streams["c3_h0_verification"] = _chunk_seeds(
        int(c3["seed_base"]), int(c3["scenes"]), int(c3["generation_chunk_size"]), stride
    )
    development = data["development"]
    per_course = int(development["paired_base_scenes"]) // 4
    for level, base in zip(COURSE_LEVELS, development["seed_bases_by_course"], strict=True):
        streams[f"development_{level}"] = _chunk_seeds(
            int(base), per_course, int(development["generation_chunk_size"]), stride
        )
    boundary = data["exact_overlap_boundary"]
    streams["exact_overlap_boundary"] = _chunk_seeds(
        int(boundary["seed_base"]), int(boundary["paired_base_scenes"]),
        int(boundary["generation_chunk_size"]), stride,
    )
    r2 = data["controlled_r2"]
    for name, base in zip(
        ("target_scale", "clutter_scale", "doppler_interventions", "regenerated_geometry"),
        r2["seed_bases_by_family"],
        strict=True,
    ):
        family_seeds: list[int] = []
        scenario_block = 100_000
        for scenario_index in range(len(r2["scenario_ids"])):
            family_seeds.extend(
                _chunk_seeds(
                    int(base) + scenario_index * scenario_block,
                    int(r2["scenes_per_geometry"]),
                    int(r2["generation_chunk_size"]),
                    stride,
                )
            )
        streams[f"r2_{name}"] = family_seeds

    used: dict[int, str] = {}
    for name, seeds in streams.items():
        for seed in seeds:
            previous = used.get(seed)
            if previous is not None:
                raise ValueError(f"E003 generation seed collision: {previous} and {name} at {seed}")
            used[seed] = name
    return streams


def validate_e003_config(config: Mapping[str, Any]) -> None:
    """Reject changes to the preregistered E003 scientific comparison."""
    try:
        experiment = config["experiment"]
        radar = config["radar"]
        data = config["data"]
        representation = config["representation"]
        training = config["training"]
        models = config["models"]
    except (KeyError, TypeError) as exc:
        raise ValueError("E003 config is missing a required section") from exc
    if experiment.get("id") != "D2-A-E003" or float(experiment.get("protocol_version", -1)) != PROTOCOL_VERSION:
        raise ValueError("config must identify D2-A-E003 protocol v1.1")
    if experiment.get("final_blind_generation_authorized") is not False:
        raise ValueError("E003 final blind generation is not authorized")
    if experiment.get("distillation_authorized") is not False:
        raise ValueError("E003 authorizes E1 truth-only training only; distillation is forbidden")
    if (
        float(radar.get("prf_hz", -1.0)) != 1300.0
        or int(radar.get("pulse_count", -1)) != 130
        or int(radar.get("channel_count", -1)) != 3
        or int(radar.get("range_bin_count", -1)) != 32
    ):
        raise ValueError("E003 must preserve the frozen E001 3x130x32 PRF=1300 radar model")
    if (
        representation.get("same_channel_lags") != [1, 4, 16]
        or representation.get("cross_channel_pairs") != [[0, 1], [0, 2], [1, 2]]
        or int(representation.get("feature_channels", -1)) != 28
        or representation.get("raw_iq_bypass") is not False
    ):
        raise ValueError("E003 physical feature set or no-bypass rule changed")
    if tuple(models.get("model_order", ())) != ("S0", "S2"):
        raise ValueError("E003 model order must remain S0 then S2")
    if training.get("supervision") != "physical_truth_only_E1":
        raise ValueError("E003 training must remain physical-truth-only E1")
    if (
        training.get("optimizer_updates_per_model") != 1600
        or training.get("effective_batch_size") != 64
        or training.get("paired_base_scenes_per_update") != 32
        or training.get("training_seed") != 101
        or training.get("device") != "cuda:0"
        or training.get("cuda_required") is not True
        or training.get("cpu_fallback") != "forbidden"
    ):
        raise ValueError("E003 requires 1600 updates, seed 101, effective batch 64, and CUDA-only training")
    if tuple(training.get("oom_fallbacks", ())) != (8, 4):
        raise ValueError("E003 OOM fallback order must remain microbatch 8 then 4")
    if len(data.get("train", {}).get("seed_bases_by_course", ())) != 4:
        raise ValueError("E003 train requires one seed base for each course level")
    if len(data.get("validation", {}).get("seed_bases_by_course", ())) != 4:
        raise ValueError("E003 validation requires one seed base for each course level")
    if len(data.get("development", {}).get("seed_bases_by_course", ())) != 4:
        raise ValueError("E003 development requires one seed base for each course level")

    scenarios = config.get("scenarios", {})
    if tuple(scenarios) != COURSE_LEVELS:
        raise ValueError("E003 scenarios must be ordered C0 through C3")
    for split in ("train", "validation", "development"):
        section = data[split]
        cells = section["cell_counts"]
        expected = int(section["pairs_per_course_level"] if split != "development" else section["paired_base_scenes"] // 4)
        if tuple(cells) != COURSE_LEVELS:
            raise ValueError(f"E003 {split} cell counts must be ordered C0-C3")
        for level, level_counts in cells.items():
            if sum(int(value) for value in level_counts.values()) != expected:
                raise ValueError(f"E003 {split}/{level} cell counts must sum to {expected}")
    if int(data["train"]["paired_base_scenes"]) != (
        int(training["optimizer_updates_per_model"]) * int(training["paired_base_scenes_per_update"])
    ):
        raise ValueError("E003 training cohort must be consumed once in 1600 matched updates")
    if int(data["development"]["paired_base_scenes"]) != sum(data["development"]["course_level_counts"]):
        raise ValueError("E003 development course counts must sum to the paired cohort size")
    if not math.isclose(float(config["statistics"]["primary_ci_lower_bound"]), 0.0):
        raise ValueError("E003 primary gain CI lower bound must remain above zero")
    if float(config["gates"]["G2_controlled_information"]["increment_over_measured_numerical_floor_min"]) != 10.0:
        raise ValueError("E003 R2 observable increments must exceed the measured numerical floor by 10x")
    if int(data["c3_h0_verification"].get("assignment_seed", -1)) < 0:
        raise ValueError("E003 dedicated C3 H0 composition requires a fixed assignment seed")
    if (
        int(config["statistics"].get("physical_diagnostic_quota_per_course", 0)) != 64
        or config["statistics"].get("physical_diagnostic_h0_source") != "h0_verification"
        or config["statistics"].get("physical_diagnostic_h1_source") != "development"
    ):
        raise ValueError("E003 physical diagnostic selection must remain 64 scenes per course from fixed H0/H1 splits")
    e003_seed_manifest(config)


def build_e003_pair_plan(
    scene_ids: np.ndarray,
    *,
    updates: int,
    pairs_per_update: int,
    seed: int,
) -> np.ndarray:
    """Create one fixed scene-index permutation consumed identically by S0/S2."""
    ids = np.asarray(scene_ids).reshape(-1).astype(str)
    if updates <= 0 or pairs_per_update <= 0 or seed < 0:
        raise ValueError("budget and seed must be valid positive values")
    expected = int(updates) * int(pairs_per_update)
    if ids.size != expected:
        raise ValueError(f"pair plan budget requires exactly {expected} unique scenes")
    if np.unique(ids).size != ids.size:
        raise ValueError("scene IDs must be unique")
    return np.random.default_rng(seed).permutation(ids.size).reshape(updates, pairs_per_update)

