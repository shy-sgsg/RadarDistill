from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from radardistill.evaluation.d2a_r5c_teaching_pilot import (
    iter_generated_split_chunks,
    paired_teaching_contrasts,
    predict_task_student_chunk,
    summarize_pilot_arm,
)
from radardistill.evaluation.d2a_e002 import exact_binomial_upper
from radardistill.training.d2a_r5c_teaching_pilot import validate_r5c_teaching_config


CONFIG_PATH = Path("experiments/D2-A-R5/R5-C-teaching/config.yaml")


@pytest.fixture
def config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _metadata(ids: np.ndarray, courses: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "base_scene_ids": ids,
        "course_level": courses,
        "scenario_id": np.asarray(["separated"] * len(ids)),
        "snr_db": np.zeros(len(ids), dtype=np.float64),
        "requested_cnr_db": np.full(len(ids), 20.0, dtype=np.float64),
    }


def test_summary_uses_higher_quantile_inclusive_threshold_and_exact_pfa(config: dict):
    ids = np.asarray([f"scene-{index}" for index in range(4)])
    courses = np.asarray(["C2", "C2", "C3", "C3"])
    development = {
        **_metadata(ids, courses),
        "h1_score": np.asarray([5.0, 3.0, 4.0, 6.0]),
        "truth_range_cell": np.asarray([1.0, 2.0, 3.0, 4.0]),
        "predicted_range_cell": np.asarray([1.5, 2.5, 4.2, 5.2]),
        "predicted_range_bin": np.asarray([1, 2, 4, 5]),
        "truth_doppler_hz": np.asarray([649.0, -649.0, 0.0, 600.0]),
        "predicted_doppler_hz": np.asarray([-649.0, 649.0, 0.0, -600.0]),
        "true_range_gated_doppler_hz": np.asarray([649.5, -649.5, 1.0, 601.0]),
    }
    calibration = {
        **_metadata(np.asarray([f"cal-{i}" for i in range(6)]), np.asarray(["C0"] * 6)),
        "h0_score": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 4.0]),
    }
    verification = {
        "overall": {
            **_metadata(ids, courses),
            "h0_score": np.asarray([3.0, 4.0, 4.0, 5.0]),
        },
        "c3": {
            **_metadata(ids[2:], courses[2:]),
            "h0_score": np.asarray([4.0, 5.0]),
        },
    }

    result = summarize_pilot_arm(
        {"development": development}, calibration, verification, config
    )

    assert result["threshold_calibration"]["quantile_method"] == "higher"
    assert result["threshold_calibration"]["threshold"] == 4.0
    assert result["threshold_calibration"]["tie_count"] == 2
    assert result["threshold_calibration"]["pfa_ge"] == pytest.approx(2 / 6)
    overall_h0 = result["h0_verification"]["overall"]["overall"]
    assert overall_h0["pfa_ge"] == pytest.approx(3 / 4)
    assert overall_h0["pfa_gt"] == pytest.approx(1 / 4)
    assert overall_h0["tie_count"] == 2
    assert overall_h0["pfa_ci95_exact_upper"] == pytest.approx(exact_binomial_upper(3, 4))

    development_metrics = result["development"]["overall"]
    assert development_metrics["pd"] == pytest.approx(3 / 4)
    assert development_metrics["auc"] == pytest.approx(0.625)
    assert "pfa_ge" not in development_metrics
    assert development_metrics["joint_success"] == pytest.approx(1 / 4)
    assert development_metrics["missed_count"] == 1
    assert development_metrics["range_bias_all_h1_cells"] == pytest.approx(0.85)
    assert development_metrics["range_cell_mismatch_rate"] == pytest.approx(0.5)
    assert development_metrics["doppler_mae_all_h1_hz"] == pytest.approx(26.0)
    assert development_metrics["true_range_gated_doppler_mae_hz"] == pytest.approx(0.75)


def test_auc_is_h1_count_weighted_across_independent_condition_matches(config: dict):
    development = {
        "base_scene_ids": np.asarray(["a0", "a1", "b0"]),
        "course_level": np.asarray(["C2", "C2", "C3"]),
        "scenario_id": np.asarray(["separated"] * 3),
        "snr_db": np.asarray([0.0, 0.0, 10.0]),
        "requested_cnr_db": np.asarray([20.0, 20.0, 20.0]),
        "h1_score": np.asarray([1.0, 2.0, 3.0]),
        "truth_range_cell": np.asarray([1.0, 2.0, 3.0]),
        "predicted_range_cell": np.asarray([1.0, 2.0, 3.0]),
        "predicted_range_bin": np.asarray([1, 2, 3]),
        "truth_doppler_hz": np.asarray([10.0, 20.0, 30.0]),
        "predicted_doppler_hz": np.asarray([10.0, 20.0, 30.0]),
        "true_range_gated_doppler_hz": np.asarray([10.0, 20.0, 30.0]),
    }
    calibration = {
        "h0_score": np.asarray([0.0, 1.0, 2.0, 3.0, 4.0]),
    }
    verification = {
        "overall": {
            "course_level": np.asarray(["C2", "C2", "C3", "C3"]),
            "scenario_id": np.asarray(["separated"] * 4),
            "snr_db": np.asarray([0.0, 0.0, 10.0, 10.0]),
            "requested_cnr_db": np.asarray([20.0] * 4),
            "h0_score": np.asarray([0.0, 3.0, 0.0, 1.0]),
        },
        "c3": {
            "course_level": np.asarray(["C3", "C3"]),
            "scenario_id": np.asarray(["separated"] * 2),
            "snr_db": np.asarray([10.0, 10.0]),
            "requested_cnr_db": np.asarray([20.0, 20.0]),
            "h0_score": np.asarray([0.0, 1.0]),
        },
    }

    result = summarize_pilot_arm({"development": development}, calibration, verification, config)

    # Condition A AUC=0.5 (two H1 scenes); condition B AUC=1.0 (one H1 scene).
    assert result["development"]["overall"]["auc"] == pytest.approx(2 / 3)


def _arm_prediction(
    ids: np.ndarray,
    courses: np.ndarray,
    h1_scores: np.ndarray,
    *,
    reverse: bool = False,
) -> dict:
    order = np.arange(len(ids))[::-1] if reverse else np.arange(len(ids))
    development = {
        **_metadata(ids[order], courses[order]),
        "h0_score": np.zeros(len(ids), dtype=np.float64)[order],
        "h1_score": h1_scores[order],
        "truth_range_cell": np.full(len(ids), 5.0)[order],
        "predicted_range_cell": np.full(len(ids), 5.2)[order],
        "predicted_range_bin": np.full(len(ids), 5, dtype=np.int64)[order],
        "truth_doppler_hz": np.full(len(ids), 100.0)[order],
        "predicted_doppler_hz": np.full(len(ids), 100.5)[order],
        "true_range_gated_doppler_hz": np.full(len(ids), 100.25)[order],
    }
    c3_ids = np.asarray([f"c3-{i}" for i in range(4)])
    c3_courses = np.asarray(["C3"] * 4)
    c3_order = np.arange(4)[::-1] if reverse else np.arange(4)
    return {
        "calibration": {"h0_score": np.asarray([0.0, 0.25, 0.5, 1.0])},
        "development": development,
        "c3_h0_verification": {
            **_metadata(c3_ids[c3_order], c3_courses[c3_order]),
            "h0_score": np.zeros(4)[c3_order],
        },
    }


def test_contrasts_pair_by_scene_id_and_report_bonferroni_lower_bound(config: dict):
    ids = np.asarray([f"scene-{i}" for i in range(8)])
    courses = np.asarray(["C2"] * 4 + ["C3"] * 4)
    arms = {
        "T0": _arm_prediction(ids, courses, np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)),
        "T1": _arm_prediction(ids, courses, np.ones(8), reverse=True),
        "T2": _arm_prediction(ids, courses, np.ones(8)),
        "T3": _arm_prediction(ids, courses, np.ones(8)),
    }

    result = paired_teaching_contrasts(arms, config)

    primary = result["primary"]["T1_minus_T0"]
    assert primary["point_difference"] == pytest.approx(0.5)
    assert primary["nominal_95_ci"] == pytest.approx([0.125, 0.875])
    assert primary["simultaneous_one_sided_97_5_lower_bound"] == pytest.approx(0.125)
    assert primary["passed"] is True
    assert result["task_protection"]["T1"]["passed"] is True
    assert result["task_protection"]["T1"]["C2_C3_range_rmse_ratio"]["point_ratio"] == pytest.approx(1.0)
    assert "T3_minus_T0" not in result["primary"]
    assert result["secondary"]["T3_minus_T0"]["role"] == "secondary_descriptive"


def test_primary_bootstrap_resamples_only_identifiable_scene_ids(config: dict):
    ids = np.asarray([f"scene-{i}" for i in range(8)])
    courses = np.asarray(["C0", "C1", "C2", "C2", "C3", "C3", "C3", "C3"])
    arms = {
        "T0": _arm_prediction(ids, courses, np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)),
        "T1": _arm_prediction(ids, courses, np.ones(8)),
        "T2": _arm_prediction(ids, courses, np.ones(8)),
        "T3": _arm_prediction(ids, courses, np.ones(8)),
    }

    result = paired_teaching_contrasts(arms, config)

    assert result["primary"]["T1_minus_T0"]["n_paired_scenes"] == 6
    assert result["primary"]["T1_minus_T0"]["point_difference"] == pytest.approx(0.5)


def test_secondary_contrast_compares_local_teacher_with_truth_assisted_control(config: dict):
    ids = np.asarray([f"scene-{i}" for i in range(10)])
    courses = np.asarray(["C2"] * 4 + ["C3"] * 4 + ["C0", "C1"])
    arms = {
        "T0": _arm_prediction(ids, courses, np.zeros(10)),
        "T1": _arm_prediction(ids, courses, np.ones(10)),
        "T2": _arm_prediction(ids, courses, np.asarray([1] * 8 + [0, 0], dtype=float)),
        "T3": _arm_prediction(
            ids, courses, np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 1, 1], dtype=float)
        ),
    }

    result = paired_teaching_contrasts(arms, config)

    contrast = result["secondary"]["T2_minus_T3"]
    assert contrast["point_difference"] == pytest.approx(0.5)
    assert contrast["n_paired_scenes"] == 8
    assert contrast["role"] == "teacher_vs_truth_assisted_descriptive"


def test_contrasts_reject_different_or_duplicate_scene_ids(config: dict):
    ids = np.asarray([f"scene-{i}" for i in range(8)])
    courses = np.asarray(["C2"] * 4 + ["C3"] * 4)
    arms = {
        name: _arm_prediction(ids, courses, np.ones(8))
        for name in ("T0", "T1", "T2", "T3")
    }
    arms["T1"]["development"]["base_scene_ids"][0] = "unknown-scene"
    with pytest.raises(ValueError, match="same base_scene_id set"):
        paired_teaching_contrasts(arms, config)
    arms["T1"] = _arm_prediction(ids, courses, np.ones(8))
    arms["T1"]["development"]["base_scene_ids"][0] = ids[1]
    with pytest.raises(ValueError, match="unique base_scene_id"):
        paired_teaching_contrasts(arms, config)


def test_contrasts_reorder_each_arm_by_base_scene_id(config: dict):
    ids = np.asarray([f"scene-{i}" for i in range(8)])
    courses = np.asarray(["C2"] * 4 + ["C3"] * 4)
    baseline_scores = np.asarray([1, 1, 0, 0, 1, 1, 0, 0], dtype=float)
    teacher_scores = np.asarray([1, 0, 0, 0, 1, 0, 0, 0], dtype=float)
    arms = {
        "T0": _arm_prediction(ids, courses, baseline_scores),
        "T1": _arm_prediction(ids, courses, teacher_scores, reverse=True),
        "T2": _arm_prediction(ids, courses, baseline_scores),
        "T3": _arm_prediction(ids, courses, baseline_scores),
    }

    result = paired_teaching_contrasts(arms, config)

    assert result["primary"]["T1_minus_T0"]["point_difference"] == pytest.approx(-0.25)


def test_split_generation_yields_fixed_size_scene_chunks(monkeypatch, config: dict):
    generated = []
    authorized = deepcopy(config)
    authorized["experiment"]["status"] = "preregistered"
    authorized["experiment"]["protocol_version"] = 1.0
    authorized["experiment"]["independent_chatgpt_review"] = {
        "status": "returned",
        "target_commit": "a" * 40,
        "decision": "CONTINUE",
        "supports_pilot": True,
    }
    authorized["experiment"]["run_authorized"] = True
    authorized["training"]["execution_authorized"] = True

    def fake_generate(count, seed, snr_db, cnr_db, physics_config, *, scenario_id):
        generated.append((count, seed, float(snr_db), float(cnr_db), scenario_id))
        return SimpleNamespace(
            base_scene_id=np.asarray([f"seed-{seed}-{index}" for index in range(count)]),
            iq_h0=np.zeros((count, 3, 130, 32), dtype=np.complex64),
            iq_h1=np.ones((count, 3, 130, 32), dtype=np.complex64),
            truth_target_range_cell=np.ones(count),
            truth_target_doppler_hz=np.zeros(count),
            snr_db=np.full(count, float(snr_db)),
            requested_cnr_db=np.full(count, float(cnr_db)),
        )

    monkeypatch.setattr(
        "radardistill.evaluation.d2a_r5c_teaching_pilot.generate_d2a_paired_batch",
        fake_generate,
    )
    chunks = iter_generated_split_chunks(authorized, {}, "c3_h0_verification")
    first = [next(chunks) for _ in range(3)]

    assert [item["batch"].iq_h0.shape[0] for item in first] == [16, 16, 16]
    assert [item["spec"]["seed"] for item in first] == [768_000_000, 768_001_009, 768_002_018]
    assert all(np.all(item["metadata"]["course_level"] == "C3") for item in first)
    assert len(generated) == 3


def test_scene_generation_stays_blocked_while_review_is_pending(monkeypatch, config: dict):
    def forbidden_generate(*args, **kwargs):
        pytest.fail("draft R5-C teaching config must not generate evaluation scenes")

    monkeypatch.setattr(
        "radardistill.evaluation.d2a_r5c_teaching_pilot.generate_d2a_paired_batch",
        forbidden_generate,
    )
    with pytest.raises(RuntimeError, match="independent R5-C review"):
        next(iter_generated_split_chunks(config, {}, "h0_calibration"))


def test_prediction_rejects_silent_cpu_fallback():
    with pytest.raises(RuntimeError, match="requires the visible CUDA device"):
        predict_task_student_chunk(
            model=None,
            iq=np.zeros((1, 3, 130, 32), dtype=np.complex64),
            device=torch.device("cpu"),
        )
