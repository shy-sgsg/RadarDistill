import yaml
import numpy as np
from pathlib import Path
import pytest
import subprocess
import sys

from radardistill.evaluation.d1_evaluation import (
    binary_detection_metrics,
    doppler_crlb_hz,
    evaluation_chunk_seed,
    estimate_fft_baselines,
    estimate_phase_difference_multichannel,
    estimate_profile_ml_reference,
    _hierarchical_joint_comparisons,
    _assert_frozen_config,
    _compute_stage_gates,
    _conditional_detection_estimation,
    _load_student_checkpoint,
    _analysis_replay_changed_paths,
    _require_completed_blind_predictions,
    parameter_bootstrap_intervals,
    parameter_error_metrics,
    paired_rmse_delta_interval,
    wilson_interval,
)
from radardistill.simulation.d0 import synthesize_target_echo


def _d0_config():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "experiments/D0-E001/config.yaml").read_text())


def test_multichannel_fft_baselines_recover_a_grid_tone_and_fractional_range():
    config = _d0_config()
    iq = synthesize_target_echo(
        range_cell=12.25,
        doppler_hz=120.0,
        initial_phase_rad=0.41,
        amplitude=1.7,
        azimuth_deg=2.0,
        elevation_deg=0.5,
        config=config,
    )[None]

    estimates = estimate_fft_baselines(iq, prf_hz=1300.0)

    assert set(estimates) == {
        "native_fft_peak",
        "native_fft_log_parabolic",
        "fourfold_fft_parabolic",
    }
    for result in estimates.values():
        assert abs(float(result["range_cell"][0]) - 12.25) < 0.2
        assert abs(float(result["doppler_hz"][0]) - 120.0) < 0.05


def test_multichannel_phase_difference_uses_all_channels_and_range_bins():
    config = _d0_config()
    iq = synthesize_target_echo(
        range_cell=12.4,
        doppler_hz=-213.0,
        initial_phase_rad=1.2,
        amplitude=0.8,
        azimuth_deg=-1.0,
        elevation_deg=0.0,
        config=config,
    )[None]

    result = estimate_phase_difference_multichannel(iq, prf_hz=1300.0)

    assert abs(float(result[0]) + 213.0) < 2e-5


def test_profile_likelihood_reference_recovers_off_grid_range_and_doppler():
    config = _d0_config()
    iq = synthesize_target_echo(
        range_cell=12.37,
        doppler_hz=127.125,
        initial_phase_rad=0.73,
        amplitude=1.2,
        azimuth_deg=2.5,
        elevation_deg=0.3,
        config=config,
    )[None]

    estimate = estimate_profile_ml_reference(
        iq,
        prf_hz=1300.0,
        range_interval=(4.0, 28.0),
        range_grid_step=0.25,
        frequency_fft_factor=16,
    )

    assert abs(float(estimate["range_cell"][0]) - 12.37) < 0.08
    assert abs(float(estimate["doppler_hz"][0]) - 127.125) < 0.15


def test_wilson_interval_handles_all_success_and_all_failure():
    assert wilson_interval(0, 100)[0] == 0.0
    assert wilson_interval(100, 100)[1] == 1.0
    low, high = wilson_interval(50, 100)
    assert 0.40 < low < 0.42
    assert 0.58 < high < 0.60


def test_binary_metrics_use_strict_threshold_and_report_auc_and_intervals():
    labels = np.asarray([False, False, True, True])
    scores = np.asarray([0.0, 1.0, 2.0, 3.0])
    snr_db = np.asarray([0, 0, 0, 0])

    summary = binary_detection_metrics(
        scores,
        labels,
        snr_db,
        thresholds={"0.5": 1.0},
    )

    overall = summary["overall"]["thresholds"]["0.5"]
    assert summary["overall"]["auc"] == 1.0
    assert overall["tp"] == 2 and overall["fp"] == 0
    assert overall["pd"] == 1.0 and overall["pfa"] == 0.0
    assert 0.342 < overall["pd_wilson_95"][0] < 0.343
    assert overall["pd_wilson_95"][1] == 1.0
    assert overall["pfa_wilson_95"][0] == 0.0


def test_parameter_error_metrics_wrap_doppler_and_report_outliers():
    metrics = parameter_error_metrics(
        np.asarray([10.0, 12.0]),
        np.asarray([10.0, 12.0]),
        np.asarray([649.0, 0.0]),
        np.asarray([-649.0, 100.0]),
        prf_hz=1300.0,
        outlier_threshold_hz=50.0,
    )

    assert metrics["range_rmse_cells"] == 0.0
    assert metrics["doppler_errors_hz"] == [2.0, 100.0]
    assert metrics["doppler_rmse_hz"] > 70.0
    assert metrics["doppler_outlier_rate"] == 0.5


def test_evaluation_chunk_seeds_are_unique_across_snr_and_chunks():
    seeds = [
        evaluation_chunk_seed(100_000, snr_index, chunk_index, snr_stride=1009, snr_count=8)
        for snr_index in range(8)
        for chunk_index in range(5)
    ]

    assert len(seeds) == len(set(seeds))
    assert seeds[0] == 100_000
    assert seeds[5] == 100_000 + 1009


def test_parameter_bootstrap_intervals_are_reproducible_and_stratified():
    range_errors = np.asarray([0.1, 0.2, 0.3, 1.0, 1.1, 1.2])
    doppler_errors = np.asarray([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    snr_db = np.asarray([-10, -10, -10, 10, 10, 10])

    first = parameter_bootstrap_intervals(
        range_errors, doppler_errors, snr_db, replicates=200, seed=33
    )
    second = parameter_bootstrap_intervals(
        range_errors, doppler_errors, snr_db, replicates=200, seed=33
    )

    assert first == second
    assert set(first["by_snr_db"]) == {"-10", "10"}
    assert first["overall"]["range_rmse_cells_95_ci"][0] <= first["overall"]["range_rmse_cells"]
    assert first["overall"]["range_rmse_cells_95_ci"][1] >= first["overall"]["range_rmse_cells"]


def test_paired_rmse_delta_interval_stays_negative_when_student_is_better():
    interval = paired_rmse_delta_interval(
        np.asarray([0.5, 1.0, 1.5]),
        np.asarray([2.0, 2.0, 2.0]),
        replicates=200,
        seed=17,
    )

    assert interval["upper_95"] < 0.0


def test_doppler_crlb_matches_three_channel_sinc_range_information():
    actual = doppler_crlb_hz(
        np.asarray([12.0]),
        np.asarray([10.0]),
        prf_hz=1300.0,
        pulse_count=130,
        channel_count=3,
    )[0]
    centered_pulse_energy = 130.0 * (130.0**2 - 1.0) / 12.0
    expected = 1300.0 / np.sqrt(8.0 * np.pi**2 * 10.0 * 3.0 * centered_pulse_energy)

    assert np.isclose(actual, expected, rtol=1e-14)


def test_evaluation_requires_the_committed_and_effective_configs_to_match(tmp_path):
    root = tmp_path
    output = root / "run"
    (root / "experiments/D1-E001").mkdir(parents=True)
    output.mkdir()
    config = {"radar": {"prf_hz": 1300.0}}
    encoded = yaml.safe_dump(config)
    (root / "experiments/D1-E001/config.yaml").write_text(encoded)
    (output / "config_used.yaml").write_text(encoded)

    _assert_frozen_config(config, root, output)

    (output / "config_used.yaml").write_text(yaml.safe_dump({"radar": {"prf_hz": 1200.0}}))
    with pytest.raises(ValueError, match="must match"):
        _assert_frozen_config(config, root, output)


def test_student_checkpoint_loader_unwraps_training_checkpoint_envelope(tmp_path):
    import torch

    from radardistill.students.task_student import TaskStudent

    model_config = {"width": 32, "temporal_dilations": [1, 2, 4, 8, 16, 32, 64]}
    model = TaskStudent(width=model_config["width"], temporal_dilations=model_config["temporal_dilations"], include_spectrum_head=False)
    checkpoint_path = tmp_path / "best_inference.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "student": model_config,
            "method": "A_truth_only",
            "seed_id": "seed_1",
            "update": 1600,
            "validation": {"task_loss": 1.0},
        },
        checkpoint_path,
    )
    config = {"student": model_config}

    loaded = _load_student_checkpoint(checkpoint_path, config, torch.device("cpu"))

    assert loaded.spectrum_head is None
    assert all(
        torch.equal(model.state_dict()[key], loaded.state_dict()[key])
        for key in model.state_dict()
    )


def test_hierarchical_joint_bootstrap_preserves_paired_strategy_direction():
    predictions = {}
    for seed_id in ("seed_1", "seed_2", "seed_3"):
        for strategy, range_error, doppler_error in (
            ("A_truth_only", 1.0, 10.0),
            ("B_teacher_pretrain_then_truth", 0.5, 5.0),
            ("C_joint_truth_and_teacher", 1.5, 20.0),
            ("D_progressive_distillation", 1.0, 10.0),
        ):
            by_snr = {}
            for snr in ("0", "5", "10", "20"):
                by_snr[snr] = {
                    "pred_range_cell": np.full(16, range_error),
                    "truth_range_cell": np.zeros(16),
                    "pred_doppler_hz": np.full(16, doppler_error),
                    "truth_doppler_hz": np.zeros(16),
                }
            predictions[f"{seed_id}/{strategy}"] = by_snr

    result = _hierarchical_joint_comparisons(
        predictions,
        ["seed_1", "seed_2", "seed_3"],
        ["A_truth_only", "B_teacher_pretrain_then_truth", "C_joint_truth_and_teacher", "D_progressive_distillation"],
        ["0", "5", "10", "20"],
        prf_hz=1300.0,
        replicates=100,
        seed=5,
    )

    assert result["B_teacher_pretrain_then_truth"]["paired_hierarchical_bootstrap_95_ci"][1] < 0
    assert result["B_teacher_pretrain_then_truth"]["seeds_improved"] == 3
    assert result["C_joint_truth_and_teacher"]["paired_hierarchical_bootstrap_95_ci"][0] > 0
    assert result["D_progressive_distillation"]["mean_delta_joint_endpoint"] == 0.0


def test_stage_gate_summary_evaluates_registered_three_seed_rules(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "experiments/D1-E001/config.yaml").read_text())
    config["evaluation"]["uncertainty"]["bootstrap_replicates"] = 50
    seed_ids = [row["id"] for row in config["splits"]["formal"]["training_seeds"]]
    strategies = list(config["training"]["methods"])
    snrs = [str(value) for value in config["data"]["evaluation_snr_db"]]
    truth_range = np.linspace(4.1, 27.9, 128)
    truth_doppler = np.linspace(-610.0, 610.0, 128)
    truth_amplitude = np.linspace(0.5, 1.5, 128)
    student_predictions = {}
    detection = {}
    estimation = {}
    formal_seed_summaries = []

    for seed_id in seed_ids:
        seed_methods = []
        for strategy in strategies:
            method_key = f"{seed_id}/{strategy}"
            range_delta = 0.005 if strategy == "B_teacher_pretrain_then_truth" else 0.01
            doppler_delta = 0.05 if strategy == "B_teacher_pretrain_then_truth" else 0.1
            by_snr = {}
            detection_by_snr = {}
            estimation_by_snr = {}
            for snr in snrs:
                by_snr[snr] = {
                    "h0_score": np.zeros(128),
                    "h1_score": np.ones(128),
                    "truth_range_cell": truth_range,
                    "truth_doppler_hz": truth_doppler,
                    "truth_target_amplitude": truth_amplitude,
                    "pred_range_cell": truth_range + range_delta,
                    "pred_doppler_hz": truth_doppler + doppler_delta,
                }
                detection_by_snr[snr] = {
                    "auc": 1.0,
                    "thresholds": {"0.01": {"pd": 1.0}},
                }
                estimation_by_snr[snr] = {
                    "range_rmse_cells": range_delta,
                    "doppler_rmse_hz": doppler_delta,
                }
            student_predictions[method_key] = by_snr
            detection[method_key] = {"by_snr_db": detection_by_snr}
            estimation[method_key] = {"by_snr_db": estimation_by_snr}
            seed_methods.append(
                {
                    "method": strategy,
                    "best_update": 1600,
                    "best_validation": {"task_loss": 0.5 if strategy == "A_truth_only" else 0.4},
                }
            )
            history_path = tmp_path / "formal" / seed_id / strategy / "training_history.csv"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            first = 1600 if strategy == "A_truth_only" else 500
            history_path.write_text(
                "update,validation_task_loss\n0,1.0\n500,0.4\n1600,0.5\n"
                if first == 500
                else "update,validation_task_loss\n0,1.0\n1600,0.5\n"
            )
        formal_seed_summaries.append({"seed_id": seed_id, "strategy_summaries": seed_methods})

    teacher_by_snr = {}
    profile_by_snr = {}
    teacher_detection = {"by_snr_db": {}}
    for snr in snrs:
        teacher_by_snr[snr] = {
            "h0_score": np.zeros(128),
            "h1_score": np.ones(128),
            "truth_range_cell": truth_range,
            "truth_doppler_hz": truth_doppler,
            "truth_target_amplitude": truth_amplitude,
            "pred_range_cell": truth_range + 0.1,
            "pred_doppler_hz": truth_doppler + 0.2,
        }
        profile_by_snr[snr] = {
            "truth_range_cell": truth_range,
            "truth_doppler_hz": truth_doppler,
            "truth_target_amplitude": truth_amplitude,
            "pred_range_cell": truth_range + 0.2,
            "pred_doppler_hz": truth_doppler + 0.3,
        }
        teacher_detection["by_snr_db"][snr] = {"thresholds": {"0.01": {"pd": 1.0}}}
    classical_predictions = {"APX15_teacher": teacher_by_snr, "profile_ml": profile_by_snr}
    detection["APX15_teacher"] = teacher_detection
    metrics = {"detection": detection, "estimation": estimation}
    formal_metrics = {"seed_summaries": formal_seed_summaries}

    gates = _compute_stage_gates(
        metrics=metrics,
        student_predictions=student_predictions,
        classical_predictions=classical_predictions,
        formal_metrics=formal_metrics,
        config=config,
        output=tmp_path,
    )

    assert gates["G1_truth_supervised_task_learning"]["passed"] is True
    assert gates["G2_distillation_value"]["passed"] is True
    assert gates["G2_distillation_value"]["strategies"]["B_teacher_pretrain_then_truth"]["final_metric_rule_passed"] is True
    assert gates["G3_traditional_competitiveness"]["passed"] is True
    assert gates["stage_decision"] == "GO"


def test_conditional_estimation_uses_frozen_detection_threshold_and_handles_zero_detections():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "experiments/D1-E001/config.yaml").read_text())
    config["data"]["evaluation_snr_db"] = [-20, 20]
    config["evaluation"]["uncertainty"]["bootstrap_replicates"] = 50
    by_snr = {}
    for snr, score in (("-20", np.zeros(4)), ("20", np.asarray([0.5, 1.0, 1.5, 2.0]))):
        by_snr[snr] = {
            "h1_score": score,
            "truth_range_cell": np.asarray([8.0, 12.0, 16.0, 20.0]),
            "truth_doppler_hz": np.asarray([-200.0, -100.0, 100.0, 200.0]),
            "truth_target_amplitude": np.ones(4),
            "pred_range_cell": np.asarray([8.1, 12.1, 16.1, 20.1]),
            "pred_doppler_hz": np.asarray([-199.0, -99.0, 101.0, 201.0]),
        }

    summary = _conditional_detection_estimation(
        by_snr,
        thresholds={"0.01": {"threshold": 1.0}},
        config=config,
        bootstrap_seed=23,
    )

    assert summary["counts_by_snr_db"]["-20"] == {"n_h1": 4, "detected": 0, "missed": 4}
    assert summary["by_snr_db"]["-20"]["n"] == 0
    assert summary["by_snr_db"]["-20"]["range_rmse_cells"] is None
    assert summary["counts_by_snr_db"]["20"] == {"n_h1": 4, "detected": 2, "missed": 2}
    assert summary["overall"]["n"] == 2
    assert summary["overall"]["range_rmse_cells"] == pytest.approx(0.1)


def test_evaluation_cli_exposes_an_explicit_saved_prediction_replay_mode():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(project_root / "scripts/evaluate_d1_e001.py"), "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--replay-existing-predictions" in result.stdout


def test_analysis_replay_only_allows_the_evaluator_module_to_change():
    frozen = {
        "code_commit": "old-code",
        "blob_oids": {
            "src/radardistill/evaluation/d1_evaluation.py": "old-evaluator",
            "scripts/evaluate_d1_e001.py": "old-cli",
            "experiments/D1-E001/config.yaml": "same-config",
        },
    }
    current = {
        "code_commit": "fixed-code",
        "blob_oids": {
            "src/radardistill/evaluation/d1_evaluation.py": "fixed-evaluator",
            "scripts/evaluate_d1_e001.py": "fixed-cli",
            "experiments/D1-E001/config.yaml": "same-config",
        },
    }

    assert _analysis_replay_changed_paths(frozen, current) == sorted(
        [
            "src/radardistill/evaluation/d1_evaluation.py",
            "scripts/evaluate_d1_e001.py",
        ]
    )

    current["blob_oids"]["experiments/D1-E001/config.yaml"] = "changed-config"
    with pytest.raises(RuntimeError, match="changed config"):
        _analysis_replay_changed_paths(frozen, current)


def test_analysis_replay_requires_every_saved_student_and_classical_prediction(tmp_path):
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "experiments/D1-E001/config.yaml").read_text()
    )
    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    expected = sorted(
        f"{seed['id']}/{method}"
        for seed in config["splits"]["formal"]["training_seeds"]
        for method in config["training"]["methods"]
    )
    for method in expected:
        (prediction_dir / f"{method.replace('/', '__')}.npz").touch()
    (prediction_dir / "classical_baselines.npz").touch()
    partial = {
        "phase": "final_blind",
        "status": "RUNNING",
        "completed_student_methods": expected,
        "classical_baselines_completed": True,
    }

    assert _require_completed_blind_predictions(partial, config, prediction_dir) == expected

    partial["completed_student_methods"].pop()
    with pytest.raises(RuntimeError, match="every prediction completed"):
        _require_completed_blind_predictions(partial, config, prediction_dir)


