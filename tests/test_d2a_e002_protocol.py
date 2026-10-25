from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.training.d2a_e002 import (
    build_matched_pair_plans,
    e002_scene_seed_streams,
    validate_e002_config,
)
from radardistill.training.d2a_e002_data import (
    H0_ARRAYS,
    _paired_shapes,
    check_cache_disk_space,
    estimate_cache_bytes,
)


@pytest.fixture
def e002_config():
    return yaml.safe_load(Path("experiments/D2-A-E002/config.yaml").read_text())


def test_frozen_e002_config_has_exact_counts_and_cuda_only(e002_config):
    validate_e002_config(e002_config)
    assert e002_config["experiment"]["protocol_version"] == 2.1
    assert e002_config["training"]["cuda_required"] is True
    assert e002_config["training"]["cpu_fallback"] == "forbidden"
    assert e002_config["experiment"]["final_blind_generation_authorized"] is False
    assert e002_config["teacher"]["official_teacher_approved"] is False
    assert e002_config["statistics"]["physical_diagnostic_pairs"] == 256


def test_config_rejects_changed_update_or_course_budget(e002_config):
    altered = yaml.safe_load(Path("experiments/D2-A-E002/config.yaml").read_text())
    altered["training"]["optimizer_updates_per_arm"] = 1599
    with pytest.raises(ValueError, match="1600|update"):
        validate_e002_config(altered)


def test_config_rejects_changed_physical_diagnostic_sample(e002_config):
    altered = yaml.safe_load(Path("experiments/D2-A-E002/config.yaml").read_text())
    altered["statistics"]["physical_diagnostic_pairs"] = 255
    with pytest.raises(ValueError, match="physical diagnostic"):
        validate_e002_config(altered)


def test_config_rejects_unmatched_curriculum_level_update_counts(e002_config):
    altered = yaml.safe_load(Path("experiments/D2-A-E002/config.yaml").read_text())
    altered["training"]["curriculum_updates"]["C1"] = 257
    altered["training"]["curriculum_updates"]["C2"] = 255
    with pytest.raises(ValueError, match="C0-C3|curriculum update"):
        validate_e002_config(altered)


def test_scene_generation_seed_streams_are_disjoint_and_fully_expanded(e002_config):
    streams = e002_scene_seed_streams(e002_config)
    assert len(streams["train_C0"]) == 800
    assert len(streams["validation_C0"]) == 79
    assert len(streams["h0_calibration"]) == 938
    all_seeds = [seed for values in streams.values() for seed in values]
    assert len(all_seeds) == len(set(all_seeds))


def test_pre_registered_cache_estimate_is_below_local_working_budget(e002_config):
    estimate = estimate_cache_bytes(e002_config)
    assert estimate["total"] > 12 * 1024**3
    assert estimate["total"] < 20 * 1024**3
    assert estimate["h0_calibration"] > 1 * 1024**3
    assert estimate["h0_verification"] > 1 * 1024**3


def test_disk_check_accepts_complete_cache_when_only_rerun_space_is_needed(
    e002_config, tmp_path, monkeypatch
):
    cache_root = tmp_path / "cache"
    data = e002_config["data"]
    paired = (
        ("train", data["train"]["paired_base_scenes"], True, False),
        ("validation", data["validation"]["paired_base_scenes"], False, False),
        ("teacher_audit", data["teacher_audit"]["paired_base_scenes"], True, True),
        ("development", data["development"]["paired_base_scenes"], False, False),
    )
    for name, count, include_k2, include_k2_h0 in paired:
        split = cache_root / name
        split.mkdir(parents=True)
        (split / "manifest.json").write_text("{}", encoding="utf-8")
        for array_name in _paired_shapes(count, include_k2, include_k2_h0):
            (split / f"{array_name}.npy").touch()
    for name in ("h0_calibration", "h0_verification"):
        split = cache_root / name
        split.mkdir(parents=True)
        (split / "manifest.json").write_text("{}", encoding="utf-8")
        for array_name in H0_ARRAYS:
            (split / f"{array_name}.npy").touch()
    boundary = cache_root / "identifiability_boundary"
    boundary.mkdir(parents=True)
    (boundary / "manifest.json").write_text("{}", encoding="utf-8")
    for array_name in ("iq_h0", "iq_h1", "base_scene_id", "truth_range_cell", "truth_doppler_hz"):
        (boundary / f"{array_name}.npy").touch()

    monkeypatch.setattr(
        "radardistill.training.d2a_e002_data.shutil.disk_usage",
        lambda _: type("Usage", (), {"free": 1024**3})(),
    )
    result = check_cache_disk_space(cache_root, e002_config, reserve_gib=3.0)
    assert result["cache_is_complete"] is True
    assert result["cache_bytes_remaining"] == 0


def test_matched_plans_use_each_pair_once_and_keep_b_c_order_identical():
    cohorts = {
        name: np.arange(index * 9, (index + 1) * 9)
        for index, name in enumerate(("C0", "C1", "C2", "C3"))
    }
    plans = build_matched_pair_plans(
        cohorts,
        updates_per_course=2,
        mixed_updates=1,
        batch_pairs=4,
        seed=917,
    )

    assert plans["A"].shape == plans["B"].shape == plans["C"].shape == (9, 4)
    np.testing.assert_array_equal(plans["B"], plans["C"])
    all_pair_ids = np.concatenate(tuple(cohorts.values()))
    for name in ("A", "B", "C"):
        np.testing.assert_array_equal(np.sort(plans[name].ravel()), np.sort(all_pair_ids))
    # The first two batches per course are homogeneous; the last batch uses
    # one not-yet-seen pair from each course.
    for course_index in range(4):
        for update in range(2):
            row = plans["B"][course_index * 2 + update]
            assert np.all((row // 9) == course_index)
    assert set(plans["B"][-1] // 9) == {0, 1, 2, 3}


def test_matched_plan_rejects_duplicate_or_insufficient_pair_ids():
    duplicated = {"C0": np.array([0, 1, 2]), "C1": np.array([2, 4, 5]), "C2": np.arange(6, 9), "C3": np.arange(9, 12)}
    with pytest.raises(ValueError, match="unique|duplicate"):
        build_matched_pair_plans(
            duplicated,
            updates_per_course=1,
            mixed_updates=1,
            batch_pairs=4,
            seed=1,
        )
