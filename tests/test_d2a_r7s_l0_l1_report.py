from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_d2a_r7s_l0_l1_report",
    ROOT / "scripts/build_d2a_r7s_l0_l1_report.py",
)
assert SPEC is not None and SPEC.loader is not None
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_capability_rows_keep_l0_diagnostic_distinct_from_l1_detection():
    metrics = {
        "source_commit": "abc123",
        "models": [{
            "method_id": "model_1", "experiment_id": "D1-E001", "model_label": "student",
            "task_scope": "joint", "seed_id": "seed_1", "parameter_count": 17,
            "checkpoint_path": "checkpoints/student.pt", "checkpoint_sha256": "deadbeef",
            "checkpoint_update": 100, "status": "COMPLETE",
            "datasets": {
                "L0_TARGET_ONLY": {
                    "detection": {"status": "UNTHRESHOLDED_ZERO_H0_DIAGNOSTIC", "auc": 1.0},
                    "parameters": {"all_h1": {"n": 2, "range_rmse_cells": 0.1,
                        "doppler_rmse_hz": 3.0, "range_bootstrap_ci95": {"rmse_ci95": [0.08, 0.12]},
                        "doppler_bootstrap_ci95": {"rmse_ci95": [2.0, 4.0]}}},
                },
                "L1_AWGN_NO_CLUTTER": [{
                    "snr_db": 0, "h0_calibration_n": 5, "h0_verification_n": 5, "h1_n": 2,
                    "detection": {"status": "EVALUATED", "verification_pfa": 0.2,
                        "verification_false_alarms": 1, "verification_pfa_ci95": [0.01, 0.5],
                        "h1_pd": 0.5, "h1_pd_ci95": [0.01, 0.9], "auc": 0.7,
                        "target_pfa_cpi": 0.1, "false_alarm_unit": "Pfa_CPI"},
                    "parameters": {"all_h1": {"n": 2, "range_rmse_cells": 0.2,
                        "doppler_rmse_hz": 4.0}, "detected_h1": {"n": 1,
                        "range_rmse_cells": 0.3, "doppler_rmse_hz": 5.0},
                        "joint_success": 0.5, "joint_success_count": 1,
                        "joint_success_ci95": [0.01, 0.9]},
                }],
            },
        }],
        "classical": [],
    }

    rows = report.build_capability_rows(metrics)

    assert [row["dataset"] for row in rows] == ["L0_TARGET_ONLY", "L1_AWGN_NO_CLUTTER"]
    assert rows[0]["pd"] == "NOT_APPLICABLE_NO_THRESHOLD"
    assert rows[0]["doppler_rmse_hz_all_h1"] == 3.0
    assert rows[1]["pfa_cpi"] == 0.2
    assert rows[1]["n_h0_verification"] == 5
    assert rows[1]["doppler_rmse_hz_all_h1"] == 4.0
    assert rows[1]["doppler_rmse_hz_detected_h1"] == 5.0
    assert rows[1]["joint_success"] == 0.5


def test_identity_builder_preserves_unknown_metadata_and_teacher_control_labels():
    metrics = {"models": [{
        "method_id": "R6-G-T3", "experiment_id": "D2-A-R6", "arm": "T3",
        "model_label": "R6-G T3 CE-TCN", "task_scope": "joint", "seed_id": "707",
        "checkpoint_update": 46400, "parameter_count": 97229,
        "checkpoint_path": "x.pt", "checkpoint_sha256": "sha", "status": "COMPLETE",
    }]}
    inventory = [{
        "method_id": "R6-G-T3", "model_name": "CE-TCN", "task_scope": "joint",
        "checkpoint_path": "x.pt", "checkpoint_sha256": "sha", "checkpoint_update": "46400",
        "seed_id": "707", "arm": "T3",
    }]

    rows = report.build_identity_rows(metrics, inventory)

    assert len(rows) == 1
    assert rows[0]["teacher_knowledge"] == "TRUTH_ASSISTED_CONTROL_NO_TEACHER"
    assert rows[0]["training_scene_exposure"] == "NOT_RECORDED"


def test_profile_ml_missing_l0_is_not_fabricated():
    metrics = {"models": [], "classical": [{
        "method_id": "profile_ML", "method_label": "profile ML", "method_role": "parameter_only",
        "datasets": {"L1_AWGN_NO_CLUTTER": []},
    }]}

    rows = report.build_capability_rows(metrics)

    assert rows == []


def test_actual_r7d_table_covers_all_available_checkpoints_and_classical_results():
    metrics = json.loads((ROOT / "experiments/D2-A-R7/R7-D/metrics.json").read_text())
    rows = report.build_capability_rows(metrics)

    assert len(rows) == 485
    assert sum(row["dataset"] == "L0_TARGET_ONLY" for row in rows) == 53
    assert sum(row["dataset"] == "L1_AWGN_NO_CLUTTER" for row in rows) == 432
    assert len({row["method_id"] for row in rows if row["checkpoint_sha256"] not in ("", "NOT_APPLICABLE_NONLEARNED")}) == 47
    profile_ml = [row for row in rows if row["method_id"] == "profile_ML_finite_grid_256_H1"]
    assert len(profile_ml) == 8
    assert {row["n_h1"] for row in profile_ml} == {256}
    assert all(row["pd"] == "NOT_APPLICABLE_PARAMETER_ONLY" for row in profile_ml)
    assert all(row["range_rmse_cells_all_h1"] != "" for row in profile_ml)
