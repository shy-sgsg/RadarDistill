from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from radardistill.training.d2a_e003 import (
    build_e003_pair_plan,
    e003_seed_manifest,
    validate_e003_config,
)
from radardistill.training import d2a_e003_data
from radardistill.training.d2a_e003_data import h0_mixture_assignments


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / "experiments/D2-A-E003/config.yaml").read_text())


def test_frozen_e003_config_and_seed_streams_are_valid_and_disjoint() -> None:
    config = _config()
    validate_e003_config(config)
    streams = e003_seed_manifest(config)
    flattened = [seed for seeds in streams.values() for seed in seeds]
    assert len(flattened) == len(set(flattened))
    assert all(seed > 10_000_000 for seed in flattened)
    assert all(not 9_000_000 <= seed < 10_000_000 for seed in flattened)


def test_config_validator_rejects_training_budget_or_final_blind_drift() -> None:
    config = _config()
    config["training"]["optimizer_updates_per_model"] = 1599
    with pytest.raises(ValueError, match="1600"):
        validate_e003_config(config)

    config = _config()
    config["experiment"]["final_blind_generation_authorized"] = True
    with pytest.raises(ValueError, match="blind"):
        validate_e003_config(config)


def test_matched_pair_plan_uses_every_scene_once_and_is_reproducible() -> None:
    scene_ids = np.asarray([f"e003-scene-{index:05d}" for index in range(512)])
    first = build_e003_pair_plan(scene_ids, updates=16, pairs_per_update=32, seed=101)
    second = build_e003_pair_plan(scene_ids, updates=16, pairs_per_update=32, seed=101)
    assert first.shape == (16, 32)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first.reshape(-1)), np.arange(512))


def test_matched_pair_plan_rejects_nonmatching_budget_and_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="budget"):
        build_e003_pair_plan(np.arange(31), updates=1, pairs_per_update=32, seed=1)
    with pytest.raises(ValueError, match="unique"):
        build_e003_pair_plan(np.asarray(["a"] * 32), updates=1, pairs_per_update=32, seed=1)


def test_h0_mixture_has_exact_course_and_c3_snr_composition() -> None:
    config = _config()
    levels = config["scenarios"]
    mixed = h0_mixture_assignments(count=40, course_levels=levels, assignment_seed=51)
    counts = {name: int(np.sum(mixed["course_level"] == name)) for name in ("C0", "C1", "C2", "C3")}
    assert counts == {"C0": 10, "C1": 10, "C2": 10, "C3": 10}
    c3 = mixed["course_level"] == "C3"
    assert int(np.sum(c3 & (mixed["snr_db"] == 0.0))) == 5
    assert int(np.sum(c3 & (mixed["snr_db"] == 10.0))) == 5
    repeated = h0_mixture_assignments(count=40, course_levels=levels, assignment_seed=51)
    np.testing.assert_array_equal(mixed["scenario_id"], repeated["scenario_id"])
    np.testing.assert_array_equal(mixed["snr_db"], repeated["snr_db"])

    c3_only = h0_mixture_assignments(
        count=12, course_levels=levels, assignment_seed=52, dedicated_c3=True
    )
    assert set(c3_only["course_level"]) == {"C3"}
    assert len(c3_only["snr_db"]) == 12
    assert {float(value) for value in c3_only["snr_db"]} == {0.0, 10.0}


def test_h0_memmap_materializer_records_and_reuses_only_matching_manifest(tmp_path, monkeypatch) -> None:
    calls: list[tuple[int, int]] = []

    def fake_generate(count, seed, snr, cnr, physics, *, scenario_id):
        calls.append((count, seed))
        return SimpleNamespace(
            iq_h0=np.full((count, 3, 130, 32), seed, dtype=np.complex64),
            base_scene_id=np.asarray([f"fake-{seed}-{index}" for index in range(count)]),
            snr_db=np.asarray(snr, dtype=np.float64),
            requested_cnr_db=np.asarray(cnr, dtype=np.float64),
            truth_target_amplitude=np.ones(count, dtype=np.float64),
            noise_power=np.ones(count, dtype=np.float64),
        )

    monkeypatch.setattr(d2a_e003_data, "generate_d2a_paired_batch", fake_generate)
    config = _config()
    split = d2a_e003_data.materialize_e003_h0_split(
        tmp_path,
        "h0_calibration",
        count=40,
        course_levels=config["scenarios"],
        seed_base=991000,
        assignment_seed=51,
        physics_config={},
        chunk_size=16,
        seed_stride=1009,
    )
    assert len(split["base_scene_id"]) == 40
    assert len(np.unique(split["base_scene_id"])) == 40
    assert sum(count for count, _ in calls) == 40
    assert [seed for _, seed in calls] == [991000, 992009, 993018]
    again = d2a_e003_data.materialize_e003_h0_split(
        tmp_path,
        "h0_calibration",
        count=40,
        course_levels=config["scenarios"],
        seed_base=991000,
        assignment_seed=51,
        physics_config={},
        chunk_size=16,
        seed_stride=1009,
    )
    assert len(calls) == 3
    np.testing.assert_array_equal(split["base_scene_id"], again["base_scene_id"])
    with pytest.raises(ValueError, match="differs"):
        d2a_e003_data.materialize_e003_h0_split(
            tmp_path,
            "h0_calibration",
            count=40,
            course_levels=config["scenarios"],
            seed_base=991000,
            assignment_seed=52,
            physics_config={},
            chunk_size=16,
            seed_stride=1009,
        )

