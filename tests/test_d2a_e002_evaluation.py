from __future__ import annotations

import numpy as np
import pytest

from radardistill.evaluation.d2a_e002 import (
    calibrate_student_threshold,
    e0_energy_score,
    evaluate_detection_scores,
    evaluate_paired_predictions,
    exact_binomial_upper,
    paired_difference_interval,
)


def test_calibration_uses_higher_quantile_and_inclusive_threshold():
    scores = np.arange(20, dtype=np.float64)
    result = calibrate_student_threshold(scores, target_pfa=0.1)
    assert result["threshold"] == 18.0
    assert result["quantile_method"] == "higher"
    assert result["decision_comparator"] == ">="
    assert result["pfa_ge"] == pytest.approx(0.1)


def test_clopper_pearson_upper_is_exact_one_sided_bound():
    upper = exact_binomial_upper(0, 100)
    assert upper == pytest.approx(1.0 - 0.05 ** (1.0 / 100.0), rel=1e-8)
    assert upper < 0.03


def test_e0_energy_score_has_expected_quadratic_scale_behavior():
    rng = np.random.default_rng(7)
    iq = rng.standard_normal((2, 3, 130, 32)) + 1j * rng.standard_normal((2, 3, 130, 32))
    score = e0_energy_score(iq)
    scaled = e0_energy_score(iq * (0.5 + 0.0j))
    np.testing.assert_allclose(scaled, score * 0.25, rtol=1e-12)


def test_paired_prediction_metrics_wrap_doppler_and_count_misses_as_joint_failures():
    result = evaluate_paired_predictions(
        h0_scores=np.asarray([0.0, 1.0, 1.0]),
        h1_scores=np.asarray([2.0, 1.0, 0.0]),
        threshold=1.0,
        truth_range=np.asarray([4.0, 5.0, 6.0]),
        predicted_range=np.asarray([4.5, 5.5, 6.5]),
        truth_doppler=np.asarray([649.0, 0.0, -649.0]),
        predicted_doppler=np.asarray([-649.0, 0.0, 649.0]),
        base_scene_ids=np.asarray(["s0", "s1", "s2"]),
        prf_hz=1300.0,
        bootstrap_replicates=50,
        bootstrap_seed=5,
    )
    assert result["pfa_ge"] == pytest.approx(2.0 / 3.0)
    assert result["pd"] == pytest.approx(2.0 / 3.0)
    assert result["joint_success"] == pytest.approx(2.0 / 3.0)
    assert result["doppler_rmse_all_h1_hz"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert result["tie_count_h0"] == 2


def test_detection_only_summary_supports_energy_baseline_without_parameter_outputs():
    result = evaluate_detection_scores(
        h0_scores=np.asarray([0.0, 1.0, 2.0, 1.0]),
        h1_scores=np.asarray([3.0, 2.0, 1.0, 0.0]),
        threshold=1.0,
        base_scene_ids=np.asarray(["a", "b", "c", "d"]),
        bootstrap_replicates=40,
        bootstrap_seed=12,
    )
    assert result["pfa_ge"] == pytest.approx(0.75)
    assert result["pd"] == pytest.approx(0.75)
    assert 0.0 <= result["auc"] <= 1.0
    assert result["pfa_ci95_exact_upper"] > result["pfa_ge"]


def test_paired_difference_interval_resamples_shared_scene_ids():
    low, high = paired_difference_interval(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray(["a", "b", "c"]),
        replicates=50,
        seed=4,
    )
    assert low == pytest.approx(1.0)
    assert high == pytest.approx(1.0)
