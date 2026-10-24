import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.training.d1_runner import (
    _teacher_metrics,
    _accumulation_scaled_loss,
    _grouped_quantile_summary,
    _pilot_gate,
    _formal_seed_records,
    _save_checkpoint,
    _write_analysis,
    _write_split_cache,
    calibrate_score_threshold,
    require_cuda,
    run_d1_e001,
    strategy_update_counts,
    validation_seeds_by_snr,
)
from radardistill.teachers.apx15_doppler import TeacherBatch


def test_runner_requires_cuda_and_rejects_cpu_fallback():
    with pytest.raises(RuntimeError, match="CUDA is required"):
        require_cuda("cuda", cuda_available=False, device_count=0)
    with pytest.raises(ValueError, match="must be cuda"):
        require_cuda("cpu", cuda_available=True, device_count=1)


def test_require_cuda_selects_device_zero_after_availability_check(monkeypatch):
    import torch

    selected = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: selected.append(device))

    device = require_cuda("cuda", cuda_available=True, device_count=1)

    assert device == torch.device("cuda:0")
    assert selected == [torch.device("cuda:0")]


def test_run_fails_fast_without_cuda_before_creating_experiment_outputs(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    output = tmp_path / "not_started"

    with pytest.raises(RuntimeError, match="no model training was started"):
        run_d1_e001({}, output, root=tmp_path, phase="pilot", device="cuda")

    assert not output.exists()


def test_all_strategy_schedules_have_exactly_the_frozen_update_budget():
    methods = {
        "A_truth_only": {"stages": [{"updates": 1600}]},
        "B_teacher_pretrain_then_truth": {"stages": [{"updates": 400}, {"updates": 1200}]},
        "C_joint_truth_and_teacher": {"stages": [{"updates": 1600}]},
        "D_progressive_distillation": {"stages": [{"updates": 320}, {"updates": 480}, {"updates": 480}, {"updates": 320}]},
    }

    counts = strategy_update_counts(methods)

    assert counts == {name: 1600 for name in methods}


def test_validation_seed_derivation_is_deterministic_and_disjoint():
    first = validation_seeds_by_snr(2026120102, 1009, 8)
    second = validation_seeds_by_snr(2026120102, 1009, 8)

    assert first == second
    assert len(first) == 8
    assert len(set(first)) == 8
    assert first[0] == 2026120102
    assert first[-1] == 2026120102 + 7 * 1009


def test_validation_cache_allocates_every_snr_and_keeps_truth_separate(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    d0_config = yaml.safe_load(
        (project_root / "experiments" / "D0-E001" / "config.yaml").read_text()
    )

    class FakeTeacher:
        def predict(self, iq):
            count = iq.shape[0]
            return TeacherBatch(
                teacher_response=np.full((count, 130, 32), 1.0 / (130 * 32), dtype=np.float32),
                teacher_detection_score=np.zeros(count, dtype=np.float32),
                teacher_range_cell=np.full(count, np.nan, dtype=np.float32),
                teacher_doppler_hz=np.full(count, np.nan, dtype=np.float32),
                teacher_confidence=np.zeros(count, dtype=np.float32),
            )

    arrays = _write_split_cache(
        cache_root=tmp_path,
        tag="tiny_validation",
        base_seed=99101,
        examples_per_class=2,
        snr_spec=[-10, 10],
        validation_snr=[-10, 10],
        validation_seed_stride=1009,
        d0_config=d0_config,
        teacher=FakeTeacher(),
    )

    assert arrays["iq"].shape == (8, 3, 130, 32)
    np.testing.assert_array_equal(arrays["truth_presence"], [False, False, True, True] * 2)
    assert np.isnan(arrays["truth_range_cell"][:2]).all()
    assert np.isfinite(arrays["truth_range_cell"][2:4]).all()
    assert arrays["teacher_response"].shape == (8, 130, 32)


def test_teacher_metrics_report_detection_and_parameter_baselines_by_snr():
    arrays = {
        "truth_presence": np.array([False, True, False, True]),
        "teacher_detection_score": np.array([1.0, 4.0, 2.0, 3.0]),
        "truth_range_cell": np.array([np.nan, 4.0, np.nan, 10.0]),
        "teacher_range_cell": np.array([2.0, 5.0, 8.0, 12.0]),
        "truth_doppler_hz": np.array([np.nan, 640.0, np.nan, -20.0]),
        "teacher_doppler_hz": np.array([0.0, -648.0, 0.0, -10.0]),
        "snr_db": np.array([-10.0, -10.0, 10.0, 10.0]),
    }

    metrics = _teacher_metrics(arrays)

    assert metrics["presence_auc"] == 1.0
    assert metrics["by_snr"]["-10"]["n_h1"] == 1
    assert metrics["by_snr"]["-10"]["range_rmse_cells"] == 1.0
    assert metrics["by_snr"]["-10"]["doppler_rmse_hz"] == 12.0
    assert metrics["by_snr"]["10"]["range_rmse_cells"] == 2.0
    assert metrics["by_snr"]["10"]["doppler_rmse_hz"] == 10.0


def test_microbatch_accumulation_matches_full_batch_gradient_with_uneven_h1_counts():
    import torch

    from radardistill.distillation.task_losses import task_loss

    torch.manual_seed(31)
    config = {
        "radar": {"prf_hz": 1300.0},
        "training": {
            "loss": {
                "presence_bce_weight": 1.0,
                "range_cell_cross_entropy_weight": 0.8,
                "range_offset_smooth_l1_weight": 0.7,
                "range_offset_smooth_l1_beta": 0.1,
                "doppler_circular_cosine_weight": 1.2,
            }
        },
    }
    labels = torch.tensor([True, False, True, True, False])
    truth = {
        "truth_presence": labels,
        "truth_range_cell": torch.tensor([4.5, float("nan"), 12.2, 8.1, float("nan")]),
        "truth_doppler_hz": torch.tensor([12.0, float("nan"), 330.0, -620.0, float("nan")]),
        "teacher_response": torch.softmax(torch.randn(5, 130 * 32), dim=-1).reshape(5, 130, 32),
    }
    base_outputs = {
        "presence_logit": torch.randn(5),
        "range_logits": torch.randn(5, 32),
        "range_offset": torch.sigmoid(torch.randn(5, 32)),
        "doppler_vector": torch.nn.functional.normalize(torch.randn(5, 32, 2), dim=-1),
        "spectrum_logits": torch.randn(5, 130, 32),
    }
    full_outputs = {key: value.clone().requires_grad_() for key, value in base_outputs.items()}
    full_loss = task_loss(full_outputs, truth, config, task_weight=0.6, teacher_weight=0.05)["total"]
    full_loss.backward()

    accumulated_outputs = {key: value.clone().requires_grad_() for key, value in base_outputs.items()}
    effective_h1 = int(labels.sum())
    for start, end in ((0, 2), (2, 5)):
        part_labels = labels[start:end]
        part_truth = {
            key: value[start:end]
            for key, value in truth.items()
        }
        part_outputs = {key: value[start:end] for key, value in accumulated_outputs.items()}
        losses = task_loss(part_outputs, part_truth, config, task_weight=0.6, teacher_weight=0.05)
        scaled = _accumulation_scaled_loss(
            losses,
            config,
            microbatch_size=end - start,
            effective_batch_size=len(labels),
            microbatch_h1=int(part_labels.sum()),
            effective_h1=effective_h1,
            task_weight=0.6,
            teacher_weight=0.05,
        )
        scaled.backward()

    for key in full_outputs:
        torch.testing.assert_close(accumulated_outputs[key].grad, full_outputs[key].grad)


def test_checkpoint_writer_persists_update_zero_inference_state(tmp_path):
    import torch

    from radardistill.students.task_student import TaskStudent

    model = TaskStudent(width=8, temporal_dilations=[1])
    path = tmp_path / "initial_inference.pt"
    config = {"student": {"width": 8, "temporal_dilations": [1]}}

    _save_checkpoint(
        path,
        model,
        config=config,
        method_name="A_truth_only",
        seed_id="pilot",
        update=0,
        validation={"task_loss": 1.25},
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["update"] == 0
    assert checkpoint["method"] == "A_truth_only"
    assert checkpoint["seed_id"] == "pilot"
    assert checkpoint["validation"]["task_loss"] == 1.25
    assert not any(key.startswith("spectrum_head.") for key in checkpoint["state_dict"])


def test_physical_consistency_summary_masks_h0_parameters_and_groups_by_snr():
    values = np.array([500.0, 1.0, 3.0, 8.0])
    h1 = np.array([False, True, True, True])
    snr = np.array([-10.0, -10.0, 0.0, 0.0])

    summary = _grouped_quantile_summary(values, h1, snr)

    assert summary["n"] == 3
    assert summary["mean_abs"] == 4.0
    assert summary["by_snr_db"]["-10"]["n"] == 1
    assert summary["by_snr_db"]["-10"]["mean_abs"] == 1.0
    assert summary["by_snr_db"]["0"]["p95_abs"] == pytest.approx(7.75)


def test_analysis_writes_student_per_snr_metrics_and_physical_groups(tmp_path):
    summary = {
        "method": "A_truth_only",
        "seed_id": "pilot",
        "optimizer_updates": 1600,
        "best_update": 1600,
        "train_seconds": 1.0,
        "cuda_peak_allocated_bytes": 1024,
        "best_validation": {
            "presence_auc": 0.9,
            "range_rmse_cells": 1.0,
            "doppler_rmse_hz": 2.0,
            "by_snr": {
                "20": {
                    "presence_auc": 1.0,
                    "range_rmse_cells": 0.1,
                    "doppler_rmse_hz": 3.0,
                    "n_h0": 4,
                    "n_h1": 5,
                }
            },
        },
    }
    grouped = {"n": 1, "mean_abs": 1.0, "p95_abs": 2.0, "max_abs": 3.0, "by_snr_db": {"20": {"n": 1, "p95_abs": 2.0}}}
    physical = {
        "n_validation_samples": 2,
        "common_phase_rotation": {"0.37": {"presence_probability_delta": grouped, "range_cell_delta": grouped, "doppler_hz_delta": grouped}},
        "frequency_shift_equivariance": {"5.0": {"presence_probability_delta": grouped, "range_cell_delta": grouped, "doppler_equivariance_residual_hz": grouped}},
    }
    teacher = {"by_snr": {"20": {"presence_auc": 1.0, "range_rmse_cells": 0.2, "doppler_rmse_hz": 0.1, "n_h0": 4, "n_h1": 5}}}
    output = tmp_path / "analysis.md"

    _write_analysis(output, [summary], "pilot", teacher, physical)

    text = output.read_text()
    assert "学生逐 SNR 验证指标" in text
    assert "| A_truth_only | 20 | 1.0000 | 0.1000 | 3.000 | 4 | 5 |" in text
    assert "H1 参数输出分 SNR 变化" in text
    assert "公共相位 0.37 rad | 20 | 1 |" in text


def test_pilot_gate_requires_frozen_high_snr_learning_screen():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "experiments/D1-E001/config.yaml").read_text())
    methods = list(config["training"]["methods"])
    summaries = []
    for name in methods:
        summaries.append(
            {
                "method": name,
                "optimizer_updates": 1600,
                "initial_validation": {
                    "task_loss": 5.0,
                    "presence_auc": 0.5,
                    "range_rmse_cells": 8.0,
                    "doppler_rmse_hz": 300.0,
                    "doppler_bias_hz": 0.0,
                    "by_snr": {},
                },
                "best_validation": {
                    "task_loss": 2.0,
                    "presence_auc": 0.99,
                    "range_rmse_cells": 1.0,
                    "doppler_rmse_hz": 20.0,
                    "doppler_bias_hz": 0.0,
                    "by_snr": {
                        "20": {
                            "presence_auc": 0.99,
                            "range_rmse_cells": 1.0,
                            "doppler_rmse_hz": 20.0,
                            "n_h0": 128,
                            "n_h1": 128,
                        }
                    },
                },
            }
        )

    passing = _pilot_gate(summaries, config)
    assert passing["technical_feasibility_passed"] is True
    assert passing["learning_screen_passed"] is True
    assert passing["passed"] is True

    summaries[0]["best_validation"]["by_snr"]["20"]["doppler_rmse_hz"] = 60.0
    failing = _pilot_gate(summaries, config)
    assert failing["technical_feasibility_passed"] is True
    assert failing["learning_screen_passed"] is False
    assert failing["passed"] is False


def test_formal_seed_records_require_exactly_three_disjoint_records():
    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "experiments/D1-E001/config.yaml").read_text())

    records = _formal_seed_records(config)

    assert len(records) == 3
    all_seeds = [
        record[key]
        for record in records
        for key in ("train", "validation", "initialization")
    ]
    assert len(set(all_seeds)) == 9

    config["splits"]["formal"]["training_seeds"][1]["train"] = records[0]["train"]
    with pytest.raises(ValueError, match="must be unique"):
        _formal_seed_records(config)

    config = yaml.safe_load((project_root / "experiments/D1-E001/config.yaml").read_text())
    config["splits"]["formal"]["training_seeds"].append(
        {"id": "seed_4", "train": 41, "validation": 42, "initialization": 43}
    )
    with pytest.raises(ValueError, match="exactly three"):
        _formal_seed_records(config)


def test_formal_runner_dispatches_three_seeds_sequentially(tmp_path, monkeypatch):
    import torch
    import radardistill.training.d1_runner as runner

    config = {
        "splits": {
            "formal": {
                "training_seeds": [
                    {"id": "seed_1", "train": 11, "validation": 12, "initialization": 13},
                    {"id": "seed_2", "train": 21, "validation": 22, "initialization": 23},
                    {"id": "seed_3", "train": 31, "validation": 32, "initialization": 33},
                ]
            }
        }
    }
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    (pilot_dir / "metrics.json").write_text('{"pilot_gate":{"passed":true}}')
    calls = []

    def fake_run_phase(**kwargs):
        calls.append(kwargs["seed_record"]["id"])
        return {"seed_id": kwargs["seed_record"]["id"], "phase": "formal", "status": "COMPLETED"}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "_pilot_gate", lambda summaries, config: {"passed": True})
    monkeypatch.setattr(runner, "_run_phase", fake_run_phase)

    result = run_d1_e001(config, tmp_path, root=tmp_path, phase="formal")

    assert calls == ["seed_1", "seed_2", "seed_3"]
    assert result["completed_seed_ids"] == calls
    assert result["status"] == "COMPLETED"
    assert (tmp_path / "formal" / "metrics.json").is_file()


def test_formal_runner_recomputes_pilot_gate_from_saved_strategy_metrics(tmp_path, monkeypatch):
    import json
    import torch
    import radardistill.training.d1_runner as runner

    project_root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "experiments/D1-E001/config.yaml").read_text())
    pilot_dir = tmp_path / "pilot"
    pilot_dir.mkdir()
    summaries = []
    for name in config["training"]["methods"]:
        h1_metrics = {
            "presence_auc": 0.50,
            "range_rmse_cells": 8.0,
            "doppler_rmse_hz": 300.0,
            "n_h0": 128,
            "n_h1": 128,
        }
        summaries.append(
            {
                "method": name,
                "optimizer_updates": 1600,
                "initial_validation": {
                    "task_loss": 5.0,
                    "presence_auc": 0.5,
                    "range_rmse_cells": 8.0,
                    "doppler_rmse_hz": 300.0,
                    "doppler_bias_hz": 0.0,
                    "by_snr": {},
                },
                "best_validation": {
                    "task_loss": 2.0,
                    "presence_auc": 0.5,
                    "range_rmse_cells": 8.0,
                    "doppler_rmse_hz": 300.0,
                    "doppler_bias_hz": 0.0,
                    "by_snr": {"20": h1_metrics},
                },
            }
        )
    (pilot_dir / "metrics.json").write_text(
        json.dumps({"pilot_gate": {"passed": True}, "strategy_summaries": summaries})
    )
    calls = []

    def fake_run_phase(**kwargs):
        calls.append(kwargs["seed_record"]["id"])
        return {"seed_id": kwargs["seed_record"]["id"], "phase": "formal", "status": "COMPLETED"}

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner, "_run_phase", fake_run_phase)

    with pytest.raises(RuntimeError, match="recomputed pilot gate failed"):
        run_d1_e001(config, tmp_path, root=tmp_path, phase="formal")

    assert calls == []


@pytest.mark.parametrize(("pfa", "expected_exceedances"), [(0.01, 499), (0.001, 49)])
def test_calibration_uses_finite_sample_order_statistic_and_strict_greater(pfa, expected_exceedances):
    scores = [float(value) for value in range(1, 50_001)]

    threshold = calibrate_score_threshold(scores, pfa)
    observed = sum(score > threshold for score in scores)

    assert math.isfinite(threshold)
    assert observed == expected_exceedances
