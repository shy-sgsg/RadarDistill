from __future__ import annotations

from pathlib import Path
import csv
import json
import sys

import numpy as np
import pytest
import yaml

from radardistill.training.d2a_r5 import (
    R5CosineSchedule,
    build_r5_repeated_pair_plan,
    should_extend_r5_training,
    validate_r5a_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-A/config.yaml"
sys.path.insert(0, str(ROOT / "scripts"))
from run_d2a_r5 import _load_split, _write_scene_predictions


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_r5a_config_freezes_cuda_budget_and_no_blind():
    config = _config()
    validate_r5a_config(config)
    assert config["training"]["device"] == "cuda:0"
    assert config["training"]["max_updates_initial"] == 6400
    assert config["training"]["extension_learning_rate_final"] == pytest.approx(1e-5)
    assert config["experiment"]["final_blind_generation_authorized"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["radar"].update(prf_hz=1200.0), "radar"),
        (lambda c: c["training"].update(device="cpu"), "CUDA"),
        (lambda c: c["training"].update(amp=True), "FP32"),
        (lambda c: c["experiment"].update(final_blind_generation_authorized=True), "blind"),
    ],
)
def test_r5a_config_rejects_scientific_or_resource_drift(mutate, message):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        validate_r5a_config(config)


def test_repeated_pair_plan_consumes_each_scene_once_per_epoch_and_is_matched():
    scene_ids = np.asarray([f"scene-{index}" for index in range(64)])
    plan = build_r5_repeated_pair_plan(
        scene_ids,
        updates=64,
        pairs_per_update=4,
        seed_base=2026102711,
    )
    other = build_r5_repeated_pair_plan(
        scene_ids,
        updates=64,
        pairs_per_update=4,
        seed_base=2026102711,
    )
    assert plan.shape == (64, 4)
    np.testing.assert_array_equal(plan, other)
    for epoch in range(4):
        epoch_indices = plan[epoch * 16 : (epoch + 1) * 16].reshape(-1)
        np.testing.assert_array_equal(np.sort(epoch_indices), np.arange(64))
    assert not np.array_equal(plan[:16], plan[16:32])


def test_repeated_pair_plan_rejects_partial_or_duplicate_epoch_ids():
    with pytest.raises(ValueError, match="complete dataset passes"):
        build_r5_repeated_pair_plan(
            np.asarray([f"scene-{i}" for i in range(64)]),
            updates=20,
            pairs_per_update=4,
            seed_base=1,
        )
    with pytest.raises(ValueError, match="unique"):
        build_r5_repeated_pair_plan(
            np.asarray(["a", "a", "c", "d"]), updates=2, pairs_per_update=2, seed_base=1
        )


def test_r5_extension_schedule_is_continuous_and_reproducible():
    schedule = R5CosineSchedule(
        initial_updates=6400,
        final_updates=12800,
        initial_lr=1e-3,
        initial_final_lr=1e-4,
        extension_final_lr=1e-5,
    )
    assert schedule.lr_for_update(1) == pytest.approx(1e-3)
    assert schedule.lr_for_update(6400) == pytest.approx(1e-4)
    assert schedule.lr_for_update(6401) == pytest.approx(1e-4)
    assert schedule.lr_for_update(12800) == pytest.approx(1e-5)
    state = schedule.state_dict()
    restored = R5CosineSchedule.from_state_dict(state)
    assert restored.state_dict() == state


def test_r5_extension_requires_frozen_late_validation_trend():
    good = [
        {
            "update": update,
            "validation_task_loss": loss,
            "validation_presence_loss": 0.7,
            "validation_range_cell_loss": 0.5,
            "validation_range_offset_loss": 0.4,
            "validation_doppler_loss": 0.3,
            "validation_auc": 0.9,
            "validation_range_rmse_cells": 0.5,
            "validation_doppler_rmse_hz": 20.0,
            "validation_true_cell_doppler_rmse_hz": 19.0,
            "validation_range_cell_mismatch_rate": 0.1,
        }
        for update, loss in zip(
            (4800, 5200, 5600, 6000, 6400), (1.00, 0.997, 0.994, 0.991, 0.985), strict=True
        )
    ]
    assert should_extend_r5_training(good) is True
    bad_drop = [dict(row) for row in good]
    bad_drop[-1]["validation_task_loss"] = 0.995
    assert should_extend_r5_training(bad_drop) is False
    missing = [row for row in good if row["update"] != 5600]
    assert should_extend_r5_training(missing) is False
    protected_metric_regression = [dict(row) for row in good]
    protected_metric_regression[-1]["validation_doppler_rmse_hz"] = 21.1
    assert should_extend_r5_training(protected_metric_regression) is False


def test_scene_prediction_csv_keeps_both_hypotheses_for_paired_splits(tmp_path):
    arrays = {
        "development": {
            "base_scene_id": np.asarray(["dev-1"]),
            "iq_h1": np.zeros((1, 1)),
            "truth_range_cell": np.asarray([4.25]),
            "truth_doppler_hz": np.asarray([12.0]),
            "course_level": np.asarray(["C2"]),
            "scenario_id": np.asarray(["A"]),
            "snr_db": np.asarray([0.0]),
        },
        "h0_verification": {
            "base_scene_id": np.asarray(["h0-1"]),
            "iq_h0": np.zeros((1, 1)),
            "course_level": np.asarray(["C2"]),
            "scenario_id": np.asarray(["A"]),
            "snr_db": np.asarray([0.0]),
        },
    }
    predictions = {
        "S0": {
            "development": {
                "h0": {"score": np.asarray([0.1]), "range_cell": np.asarray([0.0]), "doppler_hz": np.asarray([0.0])},
                "h1": {"score": np.asarray([0.9]), "range_cell": np.asarray([4.0]), "doppler_hz": np.asarray([11.0])},
            },
            "h0_verification": {
                "h0": {"score": np.asarray([0.2]), "range_cell": np.asarray([0.0]), "doppler_hz": np.asarray([0.0])},
            },
        }
    }
    output = tmp_path / "scene_predictions.csv"

    _write_scene_predictions(output, arrays, predictions)

    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    paired = [row for row in rows if row["split"] == "development"]
    assert [row["truth_presence"] for row in paired] == ["0", "1"]
    assert [float(row["score"]) for row in paired] == [0.1, 0.9]
    assert paired[0]["predicted_range_cell"] == ""
    assert paired[1]["predicted_range_cell"] == "4.0"
    assert len([row for row in rows if row["split"] == "h0_verification"]) == 1


def test_exact_overlap_split_loads_e002_manifest_format(tmp_path):
    split_dir = tmp_path / "exact_overlap_boundary"
    split_dir.mkdir()
    (split_dir / "manifest.json").write_text(
        json.dumps({"split": "exact_overlap_boundary", "scenes": 2, "scenario": "exact_overlap"}),
        encoding="utf-8",
    )
    expected = {
        "iq_h0": np.zeros((2, 3), dtype=np.complex64),
        "iq_h1": np.ones((2, 3), dtype=np.complex64),
        "base_scene_id": np.asarray(["boundary-1", "boundary-2"]),
        "truth_range_cell": np.asarray([2.0, 3.0], dtype=np.float32),
        "truth_doppler_hz": np.asarray([-1.0, 1.0], dtype=np.float32),
    }
    for name, values in expected.items():
        np.save(split_dir / f"{name}.npy", values)

    loaded = _load_split(tmp_path, "exact_overlap_boundary")

    assert set(loaded) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(loaded[name], expected[name])
