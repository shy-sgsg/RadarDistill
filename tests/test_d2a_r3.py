from __future__ import annotations

import numpy as np
import pytest
import yaml

from scripts.run_d2a_r3 import (
    assert_raw_phase_reviewed,
    assert_replayed_scene_coverage,
    build_scene_group_specs,
    _raw_score_quantile_rows,
    _validate_r3_groups,
)
from radardistill.d2a_baselines import run_b3
from radardistill.d2a_r3_baselines import (
    R3DetectionOutput,
    run_t1_global_normalized,
    run_t2_local_cfar,
)
from radardistill.evaluation.d2a_r3 import (
    hierarchical_finite_sample_bootstrap,
    paired_mean_difference_interval,
    calibrate_empirical_threshold,
    summarize_r3_scores,
    summarize_score_distribution,
    binomial_upper_tail,
    independent_mean_difference_interval,
    summarize_r3_detector,
    summarize_threshold,
    validate_disjoint_seed_groups,
    wilson_interval,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch


@pytest.fixture(scope="module")
def d2a_config():
    with open("experiments/D2-A-E001/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture(scope="module")
def r3_config():
    with open("experiments/D2-A-R3/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _noise_scene(seed: int = 91001) -> np.ndarray:
    rng = np.random.default_rng(seed)
    iq = rng.standard_normal((3, 130, 32)) + 1j * rng.standard_normal((3, 130, 32))
    return iq.astype(np.complex64)


def test_t0_raw_b3_power_obeys_quadratic_global_scale(d2a_config):
    iq = _noise_scene()
    factor = 0.6 + 1.2j
    raw = run_b3(iq, 1300.0, d2a_config)
    scaled = run_b3(factor * iq, 1300.0, d2a_config)
    assert scaled.score == pytest.approx(abs(factor) ** 2 * raw.score, rel=1e-6)


@pytest.mark.parametrize(
    ("detector", "option_key"),
    [
        (run_t1_global_normalized, "t1_global_robust_background"),
        (run_t2_local_cfar, "t2_local_cfar_like"),
    ],
)
def test_normalized_b3_statistic_is_invariant_to_global_complex_scale(
    d2a_config, r3_config, detector, option_key
):
    iq = _noise_scene(91002)
    factor = 0.6 + 1.2j
    options = r3_config["detection"][option_key]
    original = detector(iq, 1300.0, d2a_config, options)
    scaled = detector(factor * iq, 1300.0, d2a_config, options)
    np.testing.assert_allclose(scaled.score_map, original.score_map, rtol=1e-6, atol=1e-8)
    assert scaled.score == pytest.approx(original.score, rel=1e-6)


def test_t1_excludes_a_fixed_guard_around_the_global_peak(d2a_config, r3_config):
    iq = _noise_scene(91003)
    result = run_t1_global_normalized(
        iq, 1300.0, d2a_config, r3_config["detection"]["t1_global_robust_background"]
    )
    peak_doppler, peak_range = np.unravel_index(np.argmax(result.raw_rd_power), result.raw_rd_power.shape)
    mask = np.ones(result.raw_rd_power.shape, dtype=bool)
    doppler_indices = (peak_doppler + np.arange(-2, 3)) % result.raw_rd_power.shape[0]
    range_indices = np.arange(max(0, peak_range - 2), min(result.raw_rd_power.shape[1], peak_range + 3))
    mask[np.ix_(doppler_indices, range_indices)] = False
    expected = np.median(result.raw_rd_power[mask])
    assert result.background_estimate[peak_doppler, peak_range] == pytest.approx(expected, rel=1e-12)
    assert result.score == pytest.approx(np.max(result.score_map), rel=1e-12)


def test_t2_returns_raw_power_separately_from_dimensionless_detection_map(d2a_config, r3_config):
    iq = _noise_scene(91004)
    result = run_t2_local_cfar(
        iq, 1300.0, d2a_config, r3_config["detection"]["t2_local_cfar_like"]
    )
    raw = run_b3(iq, 1300.0, d2a_config)
    np.testing.assert_array_equal(result.raw_rd_power, raw.rd_power)
    assert result.score_map.shape == (130, 32)
    assert np.all(np.isfinite(result.score_map))
    assert result.score == pytest.approx(np.max(result.score_map), rel=1e-12)


def test_threshold_summary_reports_inclusive_strict_and_ties():
    result = summarize_threshold(np.asarray([1.0, 2.0, 2.0, 3.0]), threshold=2.0)
    assert result["pfa_ge"] == pytest.approx(0.75)
    assert result["pfa_gt"] == pytest.approx(0.25)
    assert result["tie_count"] == 2
    assert result["tie_fraction"] == pytest.approx(0.5)


def test_calibration_uses_higher_empirical_quantile_and_inclusive_rate():
    result = calibrate_empirical_threshold(np.asarray([1.0, 2.0, 3.0, 4.0]), target_pfa=0.25)
    assert result["threshold"] == 4.0
    assert result["quantile_method"] == "higher"
    assert result["pfa_ge"] == pytest.approx(0.25)
    assert result["pfa_gt"] == 0.0


def test_wilson_interval_is_reproducible_and_bounded():
    low, high = wilson_interval(23, 256)
    assert 0.0 < low < 23 / 256 < high < 1.0


def test_seed_group_validation_rejects_overlapping_scene_ranges():
    groups = [
        {"name": "cal", "seed_start": 1000},
        {"name": "validation", "seed_start": 1200},
    ]
    with pytest.raises(ValueError, match="overlap"):
        validate_disjoint_seed_groups(groups, scenes_per_group=256)


def test_seed_group_validation_accepts_disjoint_scene_ranges():
    groups = [
        {"name": "cal", "seed_start": 1000},
        {"name": "validation", "seed_start": 1300},
    ]
    validate_disjoint_seed_groups(groups, scenes_per_group=256)


def test_hierarchical_finite_sample_bootstrap_keeps_replicates_reproducible():
    calibration = [np.asarray([0.0, 1.0, 2.0, 3.0]), np.asarray([0.1, 1.1, 2.1, 3.1])]
    validation = [np.asarray([0.2, 1.2, 2.2, 3.2]), np.asarray([0.3, 1.3, 2.3, 3.3])]
    kwargs = dict(
        target_pfa=0.5,
        calibration_n=2,
        validation_n=2,
        replicates=40,
        seed=91005,
    )
    first = hierarchical_finite_sample_bootstrap(calibration, validation, **kwargs)
    second = hierarchical_finite_sample_bootstrap(calibration, validation, **kwargs)
    assert first == second
    assert first["replicates"] == 40
    assert first["calibration_n"] == 2
    assert first["validation_n"] == 2


def test_summarize_r3_uses_score_map_units_and_wilson_detection_intervals():
    h0_scores = np.asarray([0.0, 2.0, 0.0, 2.0])
    h1_scores = np.asarray([2.0, 2.0, 0.0, 2.0])
    h0_map = np.zeros((4, 4, 1), dtype=np.float64)
    h1_map = np.zeros((4, 4, 1), dtype=np.float64)
    h0_map[:, 0, 0] = h0_scores
    h1_map[:, 0, 0] = h1_scores
    zeros = np.zeros(4, dtype=np.float64)
    h0 = R3DetectionOutput(h0_scores, h0_map, h0_map, h0_map, zeros, zeros)
    h1 = R3DetectionOutput(h1_scores, h1_map, h1_map, h1_map, zeros, zeros)
    result = summarize_r3_detector(
        h0,
        h1,
        threshold=1.0,
        truth_range_cell=zeros,
        truth_doppler_hz=zeros,
        cluster_ids=np.asarray(["a", "b", "c", "d"]),
        prf_hz=1300.0,
        bootstrap_replicates=100,
        bootstrap_seed=91006,
    )
    assert result["pfa_cpi"] == pytest.approx(0.5)
    assert result["pd"] == pytest.approx(0.75)
    assert result["pfa_ci_method"] == "wilson_score"
    assert result["pd_ci_method"] == "wilson_score"
    assert result["pfa_cpi_cluster_bootstrap_ci95"] != result["pfa_cpi_ci95"]


def test_paired_mean_difference_interval_is_reproducible():
    candidate = np.asarray([1.0, 1.0, 0.0, 1.0])
    reference = np.asarray([0.0, 1.0, 0.0, 0.0])
    scenes = np.asarray(["a", "b", "c", "d"])
    first = paired_mean_difference_interval(candidate, reference, scenes, replicates=200, seed=91007)
    second = paired_mean_difference_interval(candidate, reference, scenes, replicates=200, seed=91007)
    assert first == second
    assert first[0] <= 0.5 <= first[1]


def test_score_summary_reports_pd_pfa_auc_and_joint_success():
    h0_scores = np.asarray([0.0, 2.0, 0.0, 2.0])
    h1_scores = np.asarray([2.0, 2.0, 0.0, 2.0])
    zeros = np.zeros(4)
    result = summarize_r3_scores(
        h0_scores,
        h1_scores,
        zeros,
        zeros,
        zeros,
        zeros,
        np.asarray(["a", "b", "c", "d"]),
        threshold=1.0,
        prf_hz=1300.0,
        bootstrap_replicates=100,
        seed=91008,
    )
    assert result["pfa_cpi"] == pytest.approx(0.5)
    assert result["pd"] == pytest.approx(0.75)
    assert result["roc_auc"] == pytest.approx(0.625)
    assert result["joint_success"] == pytest.approx(0.75)
    assert result["range_rmse_cells_detected"] == 0.0
    assert result["doppler_rmse_hz_detected"] == 0.0


def test_score_distribution_summary_reports_preregistered_quantiles():
    result = summarize_score_distribution(np.asarray([1.0, 2.0, 3.0, 4.0]))
    assert result["n"] == 4
    assert result["quantile_method"] == "linear"
    assert result["q50"] == pytest.approx(2.5)
    assert result["q90"] == pytest.approx(3.7)
    assert result["q95"] == pytest.approx(3.85)
    assert result["q99"] == pytest.approx(3.97)
    assert result["max"] == 4.0


def test_raw_score_quantiles_are_kept_separate_by_scene_group():
    groups = [
        {
            "group_id": "cal",
            "role": "calibration",
            "scope": "primary",
            "scenario_id": "separated",
            "snr_db": 10.0,
            "cnr_db": 10.0,
            "scene_count": 2,
        }
    ]
    rows = [
        {"group_id": "cal", **{f"score_{method}": float(value) for method in ("B0", "B1", "B2", "B3")}}
        for value in (1.0, 3.0)
    ]
    summaries = _raw_score_quantile_rows(rows, groups)
    assert len(summaries) == 4
    assert {row["group_id"] for row in summaries} == {"cal"}
    assert {row["q50"] for row in summaries} == {2.0}


def test_binomial_tail_quantifies_observed_e001_false_alarm_count():
    assert binomial_upper_tail(23, 256, 0.05) == pytest.approx(0.005181457236626827, rel=1e-12)


def test_independent_mean_difference_bootstrap_handles_separate_groups():
    first = independent_mean_difference_interval(
        np.ones(20), np.zeros(30), replicates=100, seed=91009
    )
    assert first == (1.0, 1.0)


def test_r3_scene_groups_match_preregistered_budget_and_disjoint_seeds(r3_config):
    groups = build_scene_group_specs(r3_config)
    assert len(groups) == 14
    assert sum(group["role"] == "calibration" for group in groups) == 5
    assert sum(group["role"] == "validation" for group in groups) == 8
    assert sum(group["scene_count"] == 4096 for group in groups) == 13
    ordered = sorted(groups, key=lambda group: group["seed_start"])
    for left, right in zip(ordered, ordered[1:]):
        assert left["seed_start"] + left["scene_count"] <= right["seed_start"]
    _validate_r3_groups(groups)


def test_candidate_phase_requires_completed_and_reviewed_raw_phase(tmp_path):
    metrics = {"raw_phase": {"completed": True}}
    (tmp_path / "metrics.json").write_text(__import__("json").dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_phase_review"):
        assert_raw_phase_reviewed(tmp_path)
    (tmp_path / "raw_phase_review.md").write_text(
        "T0 原始分数复核完成", encoding="utf-8"
    )
    assert_raw_phase_reviewed(tmp_path)


def test_g1_h0_is_identical_across_target_identifiability_scenarios(d2a_config):
    scenarios = ("separated", "near_clutter_edge", "partial_overlap", "exact_overlap")
    batches = [
        generate_d2a_paired_batch(
            2, 91010, 10.0, 10.0, d2a_config, scenario_id=scenario
        )
        for scenario in scenarios
    ]
    reference = batches[0]
    for batch in batches[1:]:
        np.testing.assert_array_equal(batch.iq_h0, reference.iq_h0)
        np.testing.assert_array_equal(batch.clutter_iq, reference.clutter_iq)
        np.testing.assert_array_equal(batch.noise_iq, reference.noise_iq)
    for batch in batches:
        np.testing.assert_allclose(
            batch.iq_h1 - batch.iq_h0,
            batch.target_signal_iq,
            rtol=1e-6,
            atol=1e-6,
        )


def test_candidate_phase_requires_complete_per_group_t0_replay():
    groups = [
        {"group_id": "cal", "role": "calibration", "scene_count": 2},
        {"group_id": "validation", "role": "validation", "scene_count": 1},
        {"group_id": "scale", "role": "scale_diagnostic", "scene_count": 1},
    ]
    raw_rows = [
        {"group_id": "cal", "base_scene_id": "a"},
        {"group_id": "cal", "base_scene_id": "b"},
        {"group_id": "validation", "base_scene_id": "c"},
        {"group_id": "scale", "base_scene_id": "d"},
    ]
    assert_replayed_scene_coverage(raw_rows, groups, {"cal": ["a", "b"], "validation": ["c"]})
    with pytest.raises(ValueError, match="coverage"):
        assert_replayed_scene_coverage(raw_rows, groups, {"cal": ["a"], "validation": ["c"]})
