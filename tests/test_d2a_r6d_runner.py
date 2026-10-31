from __future__ import annotations

import numpy as np

from scripts.run_d2a_r6d_joint import _exact_two_sided, _h1_metrics, _paired_quality_bootstrap


def _conditions() -> list[dict[str, str]]:
    return [
        {"course": "C0"},
        {"course": "C1"}, {"course": "C1"},
        {"course": "C2"}, {"course": "C2"}, {"course": "C2"},
        {"course": "C3"}, {"course": "C3"}, {"course": "C3"},
        {"course": "C3"}, {"course": "C3"}, {"course": "C3"},
    ]


def _predictions(*, detected: bool, range_error: float, doppler_error: float) -> dict[str, np.ndarray]:
    count = 120
    return {
        "scene_id": np.asarray([f"scene-{index}" for index in range(count)]),
        "condition_index": np.tile(np.arange(12, dtype=np.uint8), 10),
        "score": np.full(count, 0.9 if detected else 0.1, dtype=np.float32),
        "truth_range_cell": np.full(count, 12.25, dtype=np.float32),
        "predicted_range_cell": np.full(count, 12.25 + range_error, dtype=np.float32),
        "truth_doppler_hz": np.full(count, 100.0, dtype=np.float32),
        "predicted_doppler_hz": np.full(count, 100.0 + doppler_error, dtype=np.float32),
    }


def test_exact_two_sided_clopper_pearson_edges_and_center() -> None:
    assert _exact_two_sided(0, 20)[0] == 0.0
    assert _exact_two_sided(20, 20)[1] == 1.0
    low, high = _exact_two_sided(10, 20)
    assert low < 0.5 < high


def test_specialized_h1_metrics_do_not_claim_untrained_task_outputs() -> None:
    conditions = _conditions()
    ids = np.arange(120, dtype=np.uint8) % 12
    parameter_predictions = {
        "condition_index": ids,
        "predicted_range_cell": np.full(120, 12.5),
        "truth_range_cell": np.full(120, 12.25),
        "predicted_doppler_hz": np.full(120, 102.0),
        "truth_doppler_hz": np.full(120, 100.0),
    }
    parameter_result = _h1_metrics(
        parameter_predictions, threshold=None, conditions=conditions, estimate_parameters=True
    )
    assert "pd" not in parameter_result
    assert "all_h1" in parameter_result["by_group"]["all"]
    assert "joint_success" not in parameter_result["by_group"]["all"]

    detection_predictions = {"condition_index": ids, "score": np.full(120, 0.9)}
    detection_result = _h1_metrics(
        detection_predictions, threshold=0.5, conditions=conditions, estimate_parameters=False
    )
    assert detection_result["pd"] == 1.0
    assert detection_result["by_group"] == {}
    assert detection_result["parameter_metrics"] == "undefined_for_detection_only_arm"


def test_paired_quality_bootstrap_reports_registered_guards() -> None:
    first = _predictions(detected=True, range_error=0.4, doppler_error=4.0)
    second = _predictions(detected=True, range_error=0.3, doppler_error=3.0)
    result = _paired_quality_bootstrap(
        first,
        second,
        threshold_first=0.5,
        threshold_second=0.5,
        conditions=_conditions(),
        replicates=100,
        seed=10,
    )
    assert result["screening_pass"] is False
    assert result["overall_pd_difference_j2_minus_j1"]["protection_pass"] is True
    assert result["all_h1_range_rmse_ratio_j2_over_j1"]["ratio"] < 1.0
    assert result["all_h1_range_rmse_ratio_j2_over_j1"]["protection_pass"] is True
    assert result["all_h1_doppler_rmse_ratio_j2_over_j1"]["ratio"] < 1.0
    assert result["all_h1_doppler_rmse_ratio_j2_over_j1"]["protection_pass"] is True
