from __future__ import annotations

import numpy as np
import pytest


def test_scaled_task_students_keep_topology_and_expected_parameters():
    import torch

    from radardistill.students.r6_scaled_cnn import (
        R6C_MODEL_SPECS,
        build_r6c_student,
        count_r6c_parameters,
    )

    for name, spec in R6C_MODEL_SPECS.items():
        model = build_r6c_student(name)
        counts = count_r6c_parameters(model)
        assert model.width == spec["width"]
        assert model.temporal_dilations == (1, 2, 4, 8, 16, 32, 64)
        assert model.spectrum_head is None
        assert counts == {
            "trainable_parameters": spec["parameters"],
            "inference_parameters": spec["parameters"],
        }

    model = build_r6c_student("Medium")
    outputs = model(torch.randn(1, 6, 130, 32), compute_spectrum=False)
    assert outputs["presence_logit"].shape == (1,)
    assert outputs["range_logits"].shape == (1, 32)
    assert outputs["range_offset"].shape == (1, 32)
    assert outputs["doppler_vector"].shape == (1, 32, 2)
    sum(value.square().mean() for value in outputs.values()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_r6c_split_plans_are_seed_disjoint_and_core_is_exactly_nested():
    import yaml

    from radardistill.training.d2a_r6c_scaling import build_r6c_paired_split_plan

    with open("experiments/D2-A-R6/R6-C/config.yaml", encoding="utf-8") as handle:
        r6c = yaml.safe_load(handle)
    with open("experiments/D2-A-R6/R6-A/config.yaml", encoding="utf-8") as handle:
        r6a = yaml.safe_load(handle)
    conditions = r6a["radar"]["conditions"]

    core = build_r6c_paired_split_plan(r6c["data"], conditions, "train_core")
    extended = build_r6c_paired_split_plan(r6c["data"], conditions, "train_extended")
    validation = build_r6c_paired_split_plan(r6c["data"], conditions, "validation")

    assert sum(row["count"] for row in core) == 51_200
    assert sum(row["count"] for row in extended) == 102_400
    assert sum(row["count"] for row in validation) == 5_000
    core_chunks = {(row["condition_index"], row["seed"], row["count"]) for row in core}
    extended_chunks = {(row["condition_index"], row["seed"], row["count"]) for row in extended}
    assert core_chunks <= extended_chunks
    assert all(row["count"] == 16 for row in core + extended)

    split_seeds = {
        name: {row["seed"] for row in plan}
        for name, plan in (("core", core), ("extended", extended), ("validation", validation))
    }
    for name in ("core", "extended", "validation"):
        plan = {"core": core, "extended": extended, "validation": validation}[name]
        assert len(split_seeds[name]) == len(plan)
    assert split_seeds["core"].isdisjoint(split_seeds["validation"])
    assert split_seeds["extended"].isdisjoint(split_seeds["validation"])
    assert {row["seed"] for row in core} <= {row["seed"] for row in extended}


def test_registered_r6c_evaluation_splits_are_mutually_disjoint():
    from scripts.run_d2a_r6c_scaling import _all_split_plans, _check_nested_scene_ids, _configs

    r6c, r6a, _, _ = _configs()
    plans = _all_split_plans(r6c, r6a)
    ids = _check_nested_scene_ids(plans)
    assert {name: len(value) for name, value in ids.items()} == {
        "train_extended": 102_400,
        "validation": 5_000,
        "h0_calibration": 120_000,
        "h0_verification": 120_000,
        "h1_development": 6_144,
    }


def test_r6c_evaluation_plan_matches_update_budgets_and_reuses_core_arm(tmp_path, monkeypatch):
    from scripts import run_d2a_r6c_scaling as runner

    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs" / "seed_404")
    scale_training = {
        "extension_pair": ["Medium", "XLarge"],
        "models": {
            "Medium": {"updates_completed": 25_600},
            "Large": {"updates_completed": 12_800},
            "XLarge": {"updates_completed": 25_600},
        },
    }
    data_scale_training = {
        "core_arm": {"updates_completed": 12_800},
        "extended_arm": {"updates_completed": 12_800},
    }

    arms = runner._evaluation_arm_plan(scale_training, data_scale_training)

    assert arms["scale_Medium"] == (
        "Medium", "train_core", tmp_path / "runs/seed_404/train_core/Medium/best_checkpoint_12800.pt"
    )
    assert arms["scale_Large"] == (
        "Large", "train_core", tmp_path / "runs/seed_404/train_core/Large/best_checkpoint_12800.pt"
    )
    assert arms["scale_XLarge"] == (
        "XLarge", "train_core", tmp_path / "runs/seed_404/train_core/XLarge/best_checkpoint_12800.pt"
    )
    assert arms["scale_extended_Medium"] == (
        "Medium", "train_core", tmp_path / "runs/seed_404/train_core/Medium/best_checkpoint_25600.pt"
    )
    assert arms["scale_extended_XLarge"] == (
        "XLarge", "train_core", tmp_path / "runs/seed_404/train_core/XLarge/best_checkpoint_25600.pt"
    )
    assert "data_core_Medium" not in arms
    assert arms["data_extended_Medium"] == (
        "Medium", "train_extended", tmp_path / "runs/seed_404/train_extended/Medium/best_checkpoint_12800.pt"
    )

    scale_training["models"]["XLarge"]["updates_completed"] = 12_800
    with pytest.raises(RuntimeError, match="different completed update budgets"):
        runner._evaluation_arm_plan(scale_training, data_scale_training)


def test_r6c_evaluation_plan_omits_extension_arms_when_gate_stops(tmp_path, monkeypatch):
    from scripts import run_d2a_r6c_scaling as runner

    monkeypatch.setattr(runner, "RUNS", tmp_path / "runs" / "seed_404")
    scale_training = {
        "extension_pair": ["Medium", "Large"],
        "models": {name: {"updates_completed": 12_800} for name in ("Medium", "Large", "XLarge")},
    }
    data_scale_training = {
        "core_arm": {"updates_completed": 12_800},
        "extended_arm": {"updates_completed": 12_800},
    }

    arms = runner._evaluation_arm_plan(scale_training, data_scale_training)

    assert set(arms) == {"scale_Medium", "scale_Large", "scale_XLarge", "data_extended_Medium"}


def test_r6c_disk_usage_reads_free_and_total_fields(monkeypatch, tmp_path):
    from scripts import run_d2a_r6c_scaling as runner

    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: type("Usage", (), {"total": 1000, "used": 700, "free": 300})(),
    )
    assert runner._disk_usage_bytes(tmp_path) == (300, 1000)


def test_r6c_technical_batch_loss_chunks_cuda_probe_without_changing_loss():
    import torch

    from scripts.run_d2a_r6c_scaling import _configs, _technical_batch_task_loss, _task_config
    from radardistill.distillation.task_losses import task_loss

    class BatchRecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.7))
            self.max_batch = 0

        def forward(self, x, compute_spectrum=False):
            self.max_batch = max(self.max_batch, int(x.shape[0]))
            signal = x.mean(dim=(1, 2, 3)) * self.scale
            batch = len(signal)
            return {
                "presence_logit": signal,
                "range_logits": signal[:, None].expand(batch, 32),
                "range_offset": signal[:, None].expand(batch, 32),
                "doppler_vector": torch.stack((signal + 1.0, signal + 0.5), dim=-1)[:, None, :].expand(batch, 32, 2),
            }

    _, _, _, e003 = _configs()
    config = _task_config(e003)
    x = torch.arange(32 * 6 * 2 * 2, dtype=torch.float32).reshape(32, 6, 2, 2) / 100.0
    truth = {
        "truth_presence": torch.tensor([0.0, 1.0] * 16),
        "truth_range_cell": torch.tensor([0.0, 1.25] * 16),
        "truth_doppler_hz": torch.tensor([0.0, 113.0] * 16),
    }
    full_model = BatchRecordingModel()
    with torch.inference_mode():
        expected = task_loss(full_model(x, compute_spectrum=False), truth, config, teacher_weight=0.0)["task_total"]

    chunked_model = BatchRecordingModel()
    actual = _technical_batch_task_loss(
        chunked_model, x, truth, device=torch.device("cpu"), microbatch=16, loss_config=config
    )

    assert actual == pytest.approx(float(expected), rel=1e-6, abs=1e-7)
    assert chunked_model.max_batch == 16


def test_r6c_h1_exposure_plan_uses_all_scenes_each_epoch():
    from radardistill.training.d2a_r6c_scaling import build_r6c_h1_plan

    conditions = np.repeat(np.arange(4, dtype=np.int16), 16)
    plan = build_r6c_h1_plan(
        conditions, updates=8, h1_per_update=32, seed=44
    )
    assert plan.shape == (8, 32)
    np.testing.assert_array_equal(np.sort(plan.reshape(-1)), np.repeat(np.arange(64), 4))


def _history(task_losses: list[float]) -> list[dict[str, float]]:
    updates = (11_200, 11_600, 12_000, 12_400, 12_800)
    return [
        {
            "update": update,
            "validation_task_loss": loss,
            "interval_train_task_loss": loss - 0.02,
            "validation_presence_loss": 0.2,
            "validation_range_cell_loss": 0.3,
            "validation_range_offset_loss": 0.1,
            "validation_doppler_loss": 0.2,
            "validation_range_rmse_cells": 1.2,
            "validation_doppler_rmse_hz": 25.0,
            "validation_auc": 0.9,
            "validation_range_cell_mismatch_rate": 0.1,
        }
        for update, loss in zip(updates, task_losses, strict=True)
    ]


def test_extension_gate_requires_trend_and_protects_all_metrics():
    from radardistill.training.d2a_r6c_scaling import r6c_extension_gate

    improving = _history([1.0, 0.994, 0.988, 0.982, 0.985])
    result = r6c_extension_gate(improving, start_update=11_200, end_update=12_800)
    assert result["eligible"] is True
    assert result["relative_task_loss_drop"] >= 0.01
    assert result["nonworsening_intervals"] >= 3

    overfit = _history([1.0, 0.994, 0.988, 0.982, 0.985])
    overfit[-1]["interval_train_task_loss"] = 0.5
    assert not r6c_extension_gate(overfit, start_update=11_200, end_update=12_800)["eligible"]

    task_plateau = _history([1.0, 1.001, 1.002, 1.001, 1.001])
    assert not r6c_extension_gate(task_plateau, start_update=11_200, end_update=12_800)["eligible"]


def test_extension_pair_stops_or_continues_together():
    from radardistill.training.d2a_r6c_scaling import (
        r6c_extension_pair_decision,
        r6c_select_extension_candidate,
    )

    assert r6c_select_extension_candidate({
        "Large": {"task_loss": 0.3}, "XLarge": {"task_loss": 0.2}
    }) == "XLarge"
    assert r6c_select_extension_candidate({
        "Large": {"task_loss": 0.2}, "XLarge": {"task_loss": 0.2}
    }) == "Large"
    good = {"eligible": True}
    bad = {"eligible": False}
    assert r6c_extension_pair_decision(good, good)["continue_both"] is True
    assert r6c_extension_pair_decision(good, bad)["continue_both"] is False


def test_r6c_learning_rate_has_registered_segment_boundaries():
    from radardistill.training.d2a_r6c_scaling import r6c_learning_rate

    assert r6c_learning_rate(1) == pytest.approx(1e-3)
    assert r6c_learning_rate(6400) == pytest.approx(1e-4)
    assert r6c_learning_rate(6401) == pytest.approx(1e-4)
    assert r6c_learning_rate(12800) == pytest.approx(1e-5)
    assert r6c_learning_rate(12801) == pytest.approx(1e-5)
    assert r6c_learning_rate(25600) == pytest.approx(1e-6)
    assert r6c_learning_rate(25601) == pytest.approx(1e-6)
    assert r6c_learning_rate(51200) == pytest.approx(1e-7)
    with pytest.raises(ValueError):
        r6c_learning_rate(0)
