from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_d2a_r7_unified_performance_matrix",
    ROOT / "scripts/build_d2a_r7_unified_performance_matrix.py",
)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(matrix)


def test_historical_cpi_false_alarm_is_not_converted_to_cell_pfa():
    row = matrix.normalized_row(
        {"pfa": 0.02, "pd": 0.8, "auc": 0.9},
        stage="D2-A-R6",
        experiment="R6-A",
        path=ROOT / "experiments/D2-A-R6/R6-A/metrics.json",
        pointer="/evaluation/by_snr/0",
        context={"snr_db": 0.0},
        method_id="student",
    )

    assert row["false_alarm_unit"] == "Pfa_CPI"
    assert row["pfa_cpi"] == 0.02
    assert row["pfa_cell"] == "NOT_REPORTED_FOR_THIS_SOURCE_RECORD"
    assert row["pfa_cell_verification_status"] == "NOT_VERIFIED_NO_DENSE_CELL_METRIC"
    assert row["comparability_status"] == "CROSS_STAGE_NOT_DIRECTLY_RANKED"


def test_checkpoint_selection_stage_is_preserved_from_metric_pointer():
    row = matrix.normalized_row(
        {"auc": 0.8},
        stage="D1-E001",
        experiment="D1-E001",
        path=ROOT / "experiments/D1-E001/formal/metrics.json",
        pointer="/seed_summaries/0/strategy_summaries/0/initial_validation/by_snr/-20",
        context={"seed": "seed_1", "snr_db": -20},
        method_id="A_truth_only",
    )

    assert row["seed"] == "seed_1"
    assert row["checkpoint_role"] == "initial_validation"
    assert row["source_pointer"].endswith("initial_validation/by_snr/-20")


def test_not_applicable_and_missing_are_preserved_as_status_values():
    row = matrix.normalized_row(
        {"pfa": "NOT_APPLICABLE", "range_rmse_cells": "NOT_RECORDED"},
        stage="D2-A-R7",
        experiment="R7-D",
        path=ROOT / "experiments/D2-A-R7/R7-D/metrics.json",
        pointer="/classical/profile_ML",
        context={"task_scope": "parameter_only"},
        method_id="profile_ML",
    )

    assert row["pfa_cpi"] == "NOT_APPLICABLE"
    assert row["range_rmse_cells"] == "NOT_RECORDED"
    assert row["metric_payload_json"] == '{"pfa":"NOT_APPLICABLE","range_rmse_cells":"NOT_RECORDED"}'


def test_teacher_soft_response_is_not_labeled_as_a_threshold_detector():
    row = matrix.normalized_row(
        {"target_top_cell_localization_rate": 0.7, "response_lift": 0.2},
        stage="D2-A-R5",
        experiment="R5-C",
        path=ROOT / "experiments/D2-A-R5/R5-C/metrics.json",
        pointer="/teacher_audit/candidate_summaries/K3",
        context={},
        method_id="K3",
        metric_group="TEACHER_SOFT_RESPONSE_QUALITY",
    )

    assert row["false_alarm_unit"] == "NOT_APPLICABLE_SOFT_TEACHER_RESPONSE"
    assert row["pfa_cpi"] == ""


@pytest.mark.parametrize(
    "value,expected",
    [
        ([{"method": "a"}, 3, {"method": "b"}], 2),
        ({"method": "a"}, 1),
        (None, 0),
    ],
)
def test_classical_result_collection_normalization(value, expected):
    assert len(matrix.as_items(value)) == expected


def test_unrun_advanced_references_are_explicit_without_fake_performance():
    rows = []
    matrix.add_r7e_nonrun_rows(rows)

    assert len(rows) == 9
    stap = next(row for row in rows if row["method_id"] == "full_STAP_SMI_390D")
    cfar = next(row for row in rows if row["method_id"] == "OS_CFAR_dense_cell")
    assert stap["metric_status"] == "NOT_APPLICABLE"
    assert stap["applicability_status"] == "NOT_APPLICABLE_CURRENT_TRAINING_SUPPORT"
    assert cfar["metric_status"] == "NOT_RUN"
    assert cfar["pd"] == "NOT_RUN"
    assert cfar["pfa_cell"] == "NOT_RUN"


def test_d1_r1_strategy_and_seed_are_both_preserved():
    rows = []
    matrix.add_json_metric_file(rows, ROOT / "experiments/D1-R1/metrics.json")

    labeled = [row for row in rows if row["method_id"] == "A_truth_only"]
    assert labeled
    assert {row["seed"] for row in labeled} == {"1", "2", "3"}
    assert all(row["method_name"] == "A_truth_only" for row in labeled)


def test_r7d_model_and_condition_statuses_can_be_written_together(tmp_path, monkeypatch):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "status": "COMPLETE",
        "models": [{
            "experiment_id": "D2-A-R7",
            "method_id": "r7d_test_model",
            "model_label": "R7-D test model",
            "task_scope": "joint",
            "status": "COMPLETE",
            "datasets": {
                "L0_TARGET_ONLY": {"detection": {"status": "L0_DIAGNOSTIC"}, "parameters": None},
                "L1_AWGN_NO_CLUTTER": [{
                    "snr_db": 0,
                    "detection": {"status": "EVALUATED", "h1_pd": 0.8},
                    "parameters": None,
                }],
            },
        }],
        "classical": [],
    }))
    monkeypatch.setattr(matrix, "R7_METRICS", metrics_path)

    rows = []
    matrix.add_r7_rows(rows)

    assert [row["metric_status"] for row in rows] == ["L0_DIAGNOSTIC", "EVALUATED"]
    assert {row["experiment"] for row in rows} == {"R7-D"}
    assert rows[1]["pd"] == 0.8
    assert rows[1]["doppler_rmse_hz"] == ""
