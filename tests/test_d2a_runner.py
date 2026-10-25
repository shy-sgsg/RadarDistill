from scripts.run_d2a_e001 import _format_estimate_ci, _write_analysis


def test_estimate_ci_formatter_keeps_point_and_interval_in_same_units():
    assert _format_estimate_ci(2.0, [1.25, 2.75]) == "2.000 [1.250, 2.750]"
    assert _format_estimate_ci(None, None) == "n/a"


def test_analysis_report_includes_range_and_doppler_rmse_intervals(tmp_path):
    row = {
        "group": "primary",
        "scenario_id": "separated",
        "method": "B3",
        "snr_db_input": 10.0,
        "cnr_db_requested": 10.0,
        "pd": 1.0,
        "pd_ci95": [1.0, 1.0],
        "pfa_cpi": 0.05,
        "pfa_cpi_ci95": [0.02, 0.08],
        "roc_auc": 1.0,
        "roc_auc_ci95": [0.99, 1.0],
        "range_rmse_cells_detected": 2.0,
        "range_rmse_cells_detected_ci95": [1.25, 2.75],
        "doppler_rmse_hz_detected": 3.0,
        "doppler_rmse_hz_detected_ci95": [2.0, 4.0],
        "joint_success": 1.0,
    }
    metrics = {
        "condition_results": [row],
        "elapsed_seconds": 1.0,
        "dpca_diagnostics": {
            "B3": {
                "eligible_under_preregistered_gate": True,
                "median_separated_clutter_suppression_db": 15.0,
                "median_separated_clutter_suppression_db_ci95": [14.0, 16.0],
                "median_separated_target_retention": 2.0,
                "median_separated_target_retention_ci95": [1.5, 2.5],
            }
        },
    }

    _write_analysis(metrics, tmp_path)

    report = (tmp_path / "analysis.md").read_text(encoding="utf-8")
    assert "Range RMSE (cells, 95% CI)" in report
    assert "2.000 [1.250, 2.750]" in report
    assert "3.000 [2.000, 4.000]" in report
