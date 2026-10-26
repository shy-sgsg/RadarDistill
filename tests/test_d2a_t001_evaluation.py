from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from radardistill.evaluation.d2a_t001 import (
    build_development_assignments,
    build_h0_mixture_assignments,
    summarize_soft_response,
    summarize_soft_response_per_scene,
)


def load_config(path: str):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_h0_mixture_assignment_is_reproducible_and_balanced_by_course():
    config = load_config("experiments/D2-A-T001/config.yaml")
    spec = config["data"]["course_levels"]
    first = build_h0_mixture_assignments(12000, 8101, spec)
    second = build_h0_mixture_assignments(12000, 8101, spec)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    course_counts = {level: int(np.sum(first["course_level"] == level)) for level in ("C0", "C1", "C2", "C3")}
    assert all(2700 <= count <= 3300 for count in course_counts.values())
    assert set(np.unique(first["snr_db"])) == {0.0, 10.0, 15.0, 20.0}
    assert np.all(first["cnr_db"][first["course_level"] == "C3"] == 20.0)


def test_development_assignment_matches_all_frozen_condition_counts():
    config = load_config("experiments/D2-A-T001/config.yaml")
    e002 = load_config("experiments/D2-A-E002/config.yaml")
    first = build_development_assignments(config, e002["data"]["course_levels"])
    second = build_development_assignments(config, e002["data"]["course_levels"])
    assert first["condition_id"].size == 12000
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert {level: int(np.sum(first["course_level"] == level)) for level in ("C0", "C1", "C2", "C3")} == {
        "C0": 3000,
        "C1": 3000,
        "C2": 3000,
        "C3": 3000,
    }
    assert int(np.sum(first["condition_id"] == "C3_partial_overlap_snr_0_cnr_20")) == 500
    assert set(np.unique(first["scenario_id"])) == {"separated", "near_clutter_edge", "partial_overlap"}


def test_soft_response_metrics_capture_target_mass_localization_and_entropy():
    frequency = np.fft.fftshift(np.fft.fftfreq(130, d=1.0 / 1300.0))
    h0 = np.full((2, 130, 32), 1.0 / (130 * 32), dtype=np.float64)
    h1 = np.zeros_like(h0)
    h1[:, 65, 7] = 1.0
    per_scene = summarize_soft_response_per_scene(
        h0,
        h1,
        np.asarray([7.0, 7.0]),
        np.asarray([frequency[65], frequency[65]]),
        prf_hz=1300.0,
    )
    assert np.allclose(per_scene["h0_target_mass"], 15.0 / (130 * 32))
    assert np.all(per_scene["h1_target_mass"] == 1.0)
    assert np.all(per_scene["h1_top_cell_localized"] == 1.0)
    summary = summarize_soft_response(
        per_scene,
        np.asarray(["a", "b"]),
        bootstrap_replicates=20,
        bootstrap_seed=3,
    )
    assert summary["target_window_mass_lift_mean"] > 0.99
    assert summary["teacher_top_cell_localization_rate"] == 1.0
    assert summary["h0_normalized_entropy_mean"] > 0.99
    assert summary["h1_normalized_entropy_mean"] == 0.0
