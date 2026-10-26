from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_d2a_t001 as runner

from radardistill.teachers.d2a_t001 import CovarianceBank, post_doppler_ace_amf


def read_yaml(path: str):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_course_seed_subblocks_are_disjoint_for_full_development_size():
    seeds = [
        runner._seed(34_000_000, chunk, 1009, course, course_stride=250_000)
        for course in range(4)
        for chunk in range(188)
    ]
    assert len(seeds) == len(set(seeds))
    assert min(seeds) == 34_000_000
    assert max(seeds) < 35_000_000


def test_g3_is_reported_as_not_evaluated_when_g2_fails():
    result = runner._adjudicate_g3(
        g2_passed=False,
        soft_summaries={},
        gates={},
    )

    assert result == {
        "evaluated": False,
        "passed": False,
        "reason": "G2 failed; the preregistered protocol stops before G3 adjudication",
    }


def test_g3_applies_frozen_limits_after_g2_passes():
    config = read_yaml("experiments/D2-A-T001/config.yaml")
    soft_summaries = {
        "ACE_AMF": {
            "hard_condition": {
                "target_window_mass_lift_ci95": [0.01, 0.03],
                "teacher_top_cell_localization_rate": 0.25,
            },
            "pooled_C3": {"target_window_mass_lift_ci95": [0.01, 0.03]},
        }
    }

    result = runner._adjudicate_g3(
        g2_passed=True,
        soft_summaries=soft_summaries,
        gates=config["metrics"]["teacher_gates"],
    )

    assert result["evaluated"] is True
    assert result["passed"] is True
    assert result["hard_target_mass_lift_ci_lower_positive"] is True
    assert result["hard_top_cell_localization_minimum"] is True
    assert result["pooled_C3_mass_lift_ci_lower_positive"] is True


def test_h0_runner_streams_scores_and_preserves_scene_ids():
    config = read_yaml("experiments/D2-A-T001/config.yaml")
    physics = read_yaml("experiments/D2-A-E001/config.yaml")
    r3 = read_yaml("experiments/D2-A-R3/config.yaml")
    bins = 130 * 4
    identity = np.broadcast_to(np.eye(3, dtype=np.complex128), (bins, 3, 3)).copy()
    covariance = CovarianceBank(
        covariance=identity,
        inverse=identity,
        eigenvalues=np.ones((bins, 3)),
        condition_number=np.ones(bins),
        eigenvalue_floor_fraction=np.zeros(bins),
        scene_count=0,
        pulse_count=130,
        channel_count=3,
        oversampling_factor=4,
    )
    split = {
        "scenes": 4,
        "seed_base": 9900100,
        "mixture_seed": 901,
    }
    rows, ids, timing_iq = runner._score_h0_split(
        "test",
        split,
        config,
        physics,
        covariance,
        r3["detection"]["t1_global_robust_background"],
    )
    assert len(rows) == 4
    assert len(ids) == 4
    assert timing_iq.shape == (4, 3, 130, 32)
    assert all(np.isfinite(row[f"score_{method}"]) for row in rows for method in runner.METHODS)
    assert all("iq" not in row for row in rows)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runner test requires an available GPU")
def test_method_runner_uses_requested_cuda_backend():
    config = read_yaml("experiments/D2-A-T001/config.yaml")
    physics = read_yaml("experiments/D2-A-E001/config.yaml")
    r3 = read_yaml("experiments/D2-A-R3/config.yaml")
    paired = runner.generate_d2a_paired_batch(2, 9900200, 10.0, 10.0, physics)
    bins = 130 * 4
    identity = np.broadcast_to(np.eye(3, dtype=np.complex128), (bins, 3, 3)).copy()
    covariance = CovarianceBank(
        covariance=identity,
        inverse=identity,
        eigenvalues=np.ones((bins, 3)),
        condition_number=np.ones(bins),
        eigenvalue_floor_fraction=np.zeros(bins),
        scene_count=0,
        pulse_count=130,
        channel_count=3,
        oversampling_factor=4,
    )

    outputs = runner._method_outputs(
        paired.iq_h1,
        covariance,
        physics,
        config,
        r3["detection"]["t1_global_robust_background"],
        device="cuda:0",
    )
    expected = post_doppler_ace_amf(
        paired.iq_h1, covariance, physics, config["teacher"], device="cuda:0"
    )

    np.testing.assert_allclose(outputs["ACE_AMF"]["score"], expected.score, rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(outputs["ACE_AMF"]["soft_response"]))


def test_preflight_records_intermediate_size_above_final_cap_without_blocking(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    config_path = root / "config.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    config = {
        "experiment": {"id": "D2-A-T001", "protocol_version": 1.2, "final_blind": False},
        "physics": {"source_config": "physics.yaml"},
        "teacher": {"amf_score_clip": None, "training_batch_size": 16},
        "data": {"generation_batch_size": 16},
        "execution": {
            "teacher_device": "cuda:0",
            "minimum_free_cuda_memory_bytes": 0,
            "teacher_batch_size": 16,
        },
        "outputs": {"directory": "experiments/D2-A-T001"},
    }
    resources = {
        "physics.yaml": {"radar": {"channel_count": 3, "pulse_count": 130, "prf_hz": 1300.0}},
        "config.yaml": config,
        "E002/config.yaml": {"experiment": {"protocol_version": 2.1}},
        "R3/config.yaml": {},
    }

    def fake_read_yaml(path):
        path = Path(path)
        key = "E002/config.yaml" if "E002" in str(path) else "R3/config.yaml" if "R3" in str(path) else path.name
        return resources[key]

    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(runner, "_read_yaml", fake_read_yaml)
    monkeypatch.setattr(runner, "_require_frozen_commit", lambda _: "frozen-sha")
    monkeypatch.setattr(runner, "_cuda_device_status", lambda *_: {"device": "cuda:0"})
    monkeypatch.setattr(runner, "_repo_size_bytes", lambda: 21 * 1024**3)
    monkeypatch.setattr(runner, "_environment", lambda *_: {})
    monkeypatch.setattr(runner.os, "statvfs", lambda _: SimpleNamespace(f_bavail=4 * 1024**3, f_frsize=1))

    result = runner._preflight(config_path, "cuda:0")

    assert result["passed"] is True
    assert result["project_size_bytes_before_run"] == 21 * 1024**3
    assert result["final_project_size_cap_bytes"] == 20 * 1024**3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA benchmark test requires an available GPU")
def test_candidate_benchmark_reports_cpu_and_cuda_throughput():
    config = read_yaml("experiments/D2-A-T001/config.yaml")
    physics = read_yaml("experiments/D2-A-E001/config.yaml")
    bins = 130 * 4
    identity = np.broadcast_to(np.eye(3, dtype=np.complex128), (bins, 3, 3)).copy()
    covariance = CovarianceBank(
        covariance=identity,
        inverse=identity,
        eigenvalues=np.ones((bins, 3)),
        condition_number=np.ones(bins),
        eigenvalue_floor_fraction=np.zeros(bins),
        scene_count=0,
        pulse_count=130,
        channel_count=3,
        oversampling_factor=4,
    )
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex64)

    result = runner._benchmark_candidate(
        iq, config, physics, covariance, "cuda:0", repetitions=1
    )

    assert set(result["backends"]) == {"cpu", "cuda:0"}
    assert result["backends"]["cpu"]["batch_1_cpi_count"] == 1
    assert result["backends"]["cuda:0"]["batch_1_cpi_count"] == 1
    assert result["backends"]["cpu"]["batch_16_cpi_count"] == 16
    assert result["backends"]["cuda:0"]["batch_16_cpi_count"] == 16


def test_analysis_does_not_report_g3_soft_quality_when_g2_fails(tmp_path):
    method_metrics = {
        "global": {
            "all": {
                "pd": 0.5,
                "auc": 0.6,
                "range_rmse_all_h1_cells": 1.0,
                "doppler_rmse_all_h1_hz": 2.0,
                "joint_success": 0.4,
            }
        }
    }
    metrics = {
        "gates": {
            "G1": {"passed": False, "scale_invariance_gate": False},
            "G2": {
                "passed": False,
                "h0": {
                    "large_subgroups_upper_at_most_0_10": False,
                    "large_subgroup_count": 12,
                },
                "primary_hard_condition": {
                    "ACE_AMF_minus_best_old_joint_success": {"candidate_minus_reference": 0.0, "ci95": [-0.1, 0.1]}
                },
                "identifiable_protection_vs_T1": {
                    "pd": {"candidate_minus_reference": -0.2, "ci95": [-0.3, -0.1]},
                    "joint_success": {"candidate_minus_reference": -0.2, "ci95": [-0.3, -0.1]},
                    "range_rmse_cells": {"candidate_minus_T1": 3.0, "ci95": [2.0, 4.0]},
                    "doppler_rmse_hz": {"candidate_minus_T1": 90.0, "ci95": [80.0, 100.0]},
                },
                "protection_gates": {"pd_ci_lower": False},
            },
            "G3": {"passed": False, "evaluated": False},
            "student_pilot_eligible": False,
        },
        "development": {method: method_metrics for method in runner.METHODS},
        "h0_verification": {
            method: {"pfa_ge": 0.05, "pfa_ci95_exact_upper": 0.06}
            for method in runner.METHODS
        },
        "primary_hard_condition": "C3_partial_overlap_snr_0_cnr_20",
        "best_old_baseline_in_hard_condition": "B3",
        "compute": {
            "teacher_device": "cuda:0",
            "comparison_baselines_device": "CPU NumPy",
            "teacher_stage_wall_seconds": 3539.7,
            "covariance_fit_wall_seconds": 47.95,
            "covariance_fit_scenes_per_second": 85.42,
            "cuda_peak_allocated_bytes": 1293192704,
            "cuda_peak_reserved_bytes": 1658847232,
            "candidate_inference_timing": {
                "backends": {
                    "cpu": {
                        "batch_1_seconds_per_cpi": 0.01693,
                        "batch_16_seconds_per_cpi": 0.01901,
                    },
                    "cuda:0": {
                        "batch_1_seconds_per_cpi": 0.002288,
                        "batch_16_seconds_per_cpi": 0.001217,
                    },
                }
            },
        },
        "physical_diagnostics": {
            "common_phase_rotation": {"max_abs_score_difference": 0.0, "max_abs_map_difference": 0.0},
            "common_complex_scale": {
                "(0.6+1.2j)": {
                    "max_abs_score_difference": 4.57763671875e-05,
                    "max_abs_map_difference": 4.57763671875e-05,
                }
            },
            "target_only_doppler_shift": {
                "10.0": {
                    "mean_circular_shift_residual_hz": -10.00008,
                    "within_5_hz_fraction": 0.0,
                    "shifted_detection_fraction_at_original_threshold": 0.1328,
                }
            },
        },
        "soft_response": {"evaluated": False},
    }

    runner._write_analysis(metrics, tmp_path, "test-sha")
    analysis = (tmp_path / "analysis.md").read_text(encoding="utf-8")

    assert "G3：NOT EVALUATED" in analysis
    assert "G2 未通过；按预注册协议不评价 soft response 质量" in analysis
    assert "target-mass lift" not in analysis
    assert "候选设备：** `cuda:0`" in analysis
    assert "公共复尺度的 score/map 最大绝对差为 `4.578e-05`" in analysis
    assert "频移 +10.0 Hz" in analysis
    assert "教师全阶段 `3539.7 s`" in analysis
    assert "batch=16" in analysis
    assert "实时" in analysis
