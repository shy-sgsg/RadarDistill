from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from radardistill.training.d2a_r5c_teaching_pilot import (
    assert_split_seed_disjoint,
    build_split_seed_specs,
    validate_r5c_teaching_config,
)


CONFIG_PATH = Path("experiments/D2-A-R5/R5-C-teaching/config.yaml")


@pytest.fixture
def config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_draft_config_freezes_four_matched_arms_and_resource_budget(config: dict) -> None:
    validate_r5c_teaching_config(config)

    assert config["experiment"]["status"] == "draft"
    assert config["experiment"]["run_authorized"] is False
    assert config["training"]["arms"] == ["T0", "T1", "T2", "T3"]
    assert config["training"]["updates"] == 12800
    assert config["training"]["effective_batch_size_cpi"] == 64
    assert config["resources"]["device"] == "cuda:0"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "arms", ["T0", "T1"]),
        ("training", "updates", 6400),
        ("training", "teacher_loss_weight", 0.1),
        ("resources", "cpu_fallback", "allowed"),
        ("data", "condition_seed_stride", 100),
    ],
)
def test_draft_validator_rejects_changes_to_frozen_design(
    config: dict, section: str, key: str, value
) -> None:
    invalid = deepcopy(config)
    invalid[section][key] = value

    with pytest.raises(ValueError):
        validate_r5c_teaching_config(invalid)


def test_draft_validator_rejects_changed_false_alarm_and_overfit_gates(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["evaluation"]["c3_h0_pfa_point_limit"] = 0.20
    with pytest.raises(ValueError):
        validate_r5c_teaching_config(invalid)

    invalid = deepcopy(config)
    invalid["gates"]["g1_overfit"]["updates"] = 100
    with pytest.raises(ValueError):
        validate_r5c_teaching_config(invalid)


def test_draft_validator_rejects_unregistered_extra_fields(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["training"]["unregistered_loss_search"] = True

    with pytest.raises(ValueError, match="fields differ"):
        validate_r5c_teaching_config(invalid)


def test_seed_specs_have_exact_scene_counts_and_blocked_chunks(config: dict) -> None:
    validate_r5c_teaching_config(config)
    specs = build_split_seed_specs(config)

    assert len(specs) == 1564
    assert sum(spec["scene_count"] for spec in specs) == 24832
    assert {spec["scene_count"] for spec in specs} <= set(range(1, 17))
    assert len({spec["seed"] for spec in specs}) == len(specs)
    split_scenes = {}
    for spec in specs:
        split_scenes[spec["split"]] = split_scenes.get(spec["split"], 0) + spec["scene_count"]
    assert split_scenes == {
        "h0_calibration": 8192,
        "h0_verification": 8192,
        "c3_h0_verification": 4096,
        "h1_development": 4352,
    }


def test_seed_formula_uses_local_condition_indices_and_fixed_strides(config: dict) -> None:
    specs = build_split_seed_specs(config)
    calibration = [spec for spec in specs if spec["split"] == "h0_calibration"]
    c3_verification = [spec for spec in specs if spec["split"] == "c3_h0_verification"]

    assert calibration[0]["seed"] == 728_000_000
    assert calibration[1]["seed"] == 728_001_009
    assert calibration[128]["condition_index"] == 1
    assert calibration[128]["seed"] == 729_000_000
    assert c3_verification[0]["condition_index"] == 0
    assert c3_verification[0]["seed"] == 768_000_000
    assert c3_verification[43]["condition_index"] == 1
    assert c3_verification[43]["seed"] == 769_000_000


def test_split_seed_disjointness_rejects_any_reserved_block_collision(config: dict) -> None:
    specs = build_split_seed_specs(config)

    assert_split_seed_disjoint(specs, [727_999_999, 729_000_000_000])
    with pytest.raises(ValueError, match="collision"):
        assert_split_seed_disjoint(specs, [728_000_123])
    with pytest.raises(ValueError, match="duplicate"):
        assert_split_seed_disjoint(specs, [727_999_999, 727_999_999])


def test_split_seed_disjointness_does_not_auto_renumber_collisions(config: dict) -> None:
    specs = build_split_seed_specs(config)
    collision = specs[0]["seed"]

    with pytest.raises(ValueError, match="collision"):
        assert_split_seed_disjoint(specs, [collision])
