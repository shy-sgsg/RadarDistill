from __future__ import annotations

import numpy as np

from radardistill.evaluation.d2a_r6e import (
    h0_metrics,
    h1_metrics,
    intervention_summary,
    paired_quality_bootstrap,
)


def test_h0_metrics_reports_independent_exact_bounds_and_c3_familywise_gate() -> None:
    conditions = [
        {"course": "C2", "snr_db": 0.0, "cnr_db": 20.0},
        {"course": "C3", "snr_db": 0.0, "cnr_db": 20.0},
    ]
    calibration = {"score": np.linspace(0.0, 1.0, 100), "condition_index": np.repeat([0, 1], 50)}
    verification = {
        "score": np.asarray([0.1, 0.2, 0.9, 0.8]),
        "condition_index": np.asarray([0, 0, 1, 1]),
    }

    result = h0_metrics(calibration, verification, threshold=0.5, conditions=conditions)

    assert result["verification"]["false_alarms"] == 2
    assert 0.0 < result["verification"]["pfa_ci95_exact_upper"] < 1.0
    assert result["verification_by_condition"]["1"]["pfa_familywise_exact_upper"] > 0.5
    assert result["c3_six_cells_familywise_pass"] is False


def test_h1_metrics_reports_all_c2_c3_joint_and_condition_results() -> None:
    conditions = [
        {"course": "C2", "snr_db": 0.0, "cnr_db": 20.0},
        {"course": "C3", "snr_db": 0.0, "cnr_db": 20.0},
    ]
    predictions = {
        "scene_id": np.asarray(["a", "b", "c", "d"]),
        "condition_index": np.asarray([0, 0, 1, 1]),
        "score": np.asarray([0.9, 0.1, 0.9, 0.9]),
        "predicted_range_cell": np.asarray([10.0, 10.0, 10.0, 13.0]),
        "truth_range_cell": np.asarray([10.0, 10.0, 10.0, 10.0]),
        "predicted_doppler_hz": np.asarray([100.0, 100.0, 100.0, 100.0]),
        "truth_doppler_hz": np.asarray([100.0, 100.0, 100.0, 100.0]),
    }
    h0_verification = {
        "condition_index": np.asarray([0, 1]),
        "score": np.asarray([0.2, 0.2]),
    }

    result = h1_metrics(predictions, threshold=0.5, conditions=conditions,
                        h0_verification=h0_verification, prf_hz=1300.0)

    assert result["by_group"]["C2_C3"]["n"] == 4
    assert result["by_group"]["C2_C3"]["joint_success"] == 0.5
    assert set(result["by_condition"]) == {"0", "1"}
    assert result["roc_auc_by_condition"] == {"0": 0.5, "1": 1.0}


def test_candidate_bootstrap_uses_preregistered_familywise_interval() -> None:
    baseline = {
        "scene_id": np.asarray(["a", "b", "c", "d"]),
        "condition_index": np.asarray([0, 1, 2, 3]),
        "score": np.zeros(4),
        "predicted_range_cell": np.asarray([10.0, 11.0, 12.0, 13.0]),
        "truth_range_cell": np.asarray([10.0, 11.0, 12.0, 13.0]),
        "predicted_doppler_hz": np.asarray([100.0, 200.0, 300.0, 400.0]),
        "truth_doppler_hz": np.asarray([100.0, 200.0, 300.0, 400.0]),
    }
    candidate = {key: np.array(value, copy=True) for key, value in baseline.items()}
    candidate["score"][:] = 1.0
    conditions = [{"course": "C2"}, {"course": "C2"}, {"course": "C3"}, {"course": "C3"}]

    result = paired_quality_bootstrap(
        baseline, candidate, threshold_baseline=0.5, threshold_candidate=0.5,
        conditions=conditions, replicates=100, seed=4,
    )

    assert result["joint_success_difference"] == 1.0
    assert result["joint_success_ci97_5"] == [1.0, 1.0]
    assert result["candidate_screening_pass"] is True
    assert result["protection"]["pd_difference_ci97_5"][0] >= -0.02
    assert result["protection"]["range_rmse_ratio_ci97_5"][1] <= 1.05
    assert result["protection"]["doppler_rmse_ratio_ci97_5"][1] <= 1.05


def test_frequency_intervention_summary_checks_expected_wrapped_shift() -> None:
    baseline = {
        "score": np.asarray([0.8, 0.4]),
        "predicted_range_cell": np.asarray([10.0, 11.0]),
        "predicted_doppler_hz": np.asarray([100.0, -200.0]),
    }
    intervention = {
        "score": np.asarray([0.8, 0.4]),
        "predicted_range_cell": np.asarray([10.0, 11.0]),
        "predicted_doppler_hz": np.asarray([230.0, -70.0]),
    }

    result = intervention_summary(
        baseline, intervention, threshold=0.5, prf_hz=1300.0,
        expected_doppler_shift_hz=130.0,
    )

    assert result["decision_agreement"] == 1.0
    assert result["doppler_shift_response_residual_p95_absolute_hz"] == 0.0
