from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from radardistill.training.d2a_e002 import (
    _cpu_state_dict,
    _paired_validation,
    _train_arm_attempt,
    make_paired_examples,
    run_micro_overfit_sanity,
    scale_d2a_e002_microbatch_loss,
    train_d2a_e002_pilot,
)
from radardistill.students.task_student import TaskStudent


@pytest.fixture
def e002_config():
    return yaml.safe_load(Path("experiments/D2-A-E002/config.yaml").read_text())


def test_pair_examples_interleave_h0_h1_and_keep_teacher_map_on_h1_only():
    iq_h0 = np.full((2, 3, 130, 32), 2.0 + 3.0j, dtype=np.complex64)
    iq_h1 = np.full((2, 3, 130, 32), 5.0 + 7.0j, dtype=np.complex64)
    teacher = np.empty((2, 130, 32), dtype=np.float32)
    teacher[0] = 0.25
    teacher[1] = 0.75
    arrays = {
        "iq_h0": iq_h0,
        "iq_h1": iq_h1,
        "truth_range_cell": np.asarray([4.5, 12.25], dtype=np.float32),
        "truth_doppler_hz": np.asarray([20.0, -30.0], dtype=np.float32),
        "teacher_response": teacher,
    }
    examples = make_paired_examples(arrays, np.asarray([1, 0]), include_teacher=True)
    np.testing.assert_array_equal(examples["truth_presence"], [False, True, False, True])
    np.testing.assert_array_equal(examples["iq_ri"][0, :3, 0, 0], [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(examples["iq_ri"][0, 3:, 0, 0], [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(examples["iq_ri"][1, :3, 0, 0], [5.0, 5.0, 5.0])
    np.testing.assert_array_equal(examples["truth_range_cell"], [0.0, 12.25, 0.0, 4.5])
    np.testing.assert_array_equal(examples["truth_doppler_hz"], [0.0, -30.0, 0.0, 20.0])
    assert np.all(examples["teacher_response"][::2] == 0.0)
    np.testing.assert_array_equal(examples["teacher_response"][1::2], teacher[[1, 0]])


def test_gradient_accumulation_matches_one_effective_paired_batch(e002_config):
    weights = e002_config["training"]["losses"]
    teacher_weight = 0.05
    full_terms = {
        name: torch.tensor(value)
        for name, value in {
            "presence": 0.4,
            "range_cell": 0.3,
            "range_offset": 0.2,
            "doppler": 0.1,
            "teacher_kl": 1.5,
        }.items()
    }
    expected = (
        weights["presence_bce_weight"] * full_terms["presence"]
        + weights["range_cell_cross_entropy_weight"] * full_terms["range_cell"]
        + weights["range_offset_smooth_l1_weight"] * full_terms["range_offset"]
        + weights["doppler_circular_cosine_weight"] * full_terms["doppler"]
        + teacher_weight * full_terms["teacher_kl"]
    )
    for microbatch in (16, 8, 4):
        micro_h1 = microbatch // 2
        accumulation = 64 // microbatch
        total = sum(
            scale_d2a_e002_microbatch_loss(
                full_terms,
                e002_config,
                microbatch_size=microbatch,
                effective_batch_size=64,
                microbatch_h1=micro_h1,
                effective_h1=32,
                teacher_weight=teacher_weight,
            )
            for _ in range(accumulation)
        )
        torch.testing.assert_close(total, expected)


def test_training_entry_rejects_cpu_before_accessing_data(e002_config, tmp_path):
    with pytest.raises(ValueError, match="cuda:0|CPU fallback"):
        train_d2a_e002_pilot(
            {}, {}, e002_config, tmp_path, device_name="cpu", arms=("A",)
        )


def test_validation_path_reports_task_auc_and_true_cell_doppler(e002_config):
    rng = np.random.default_rng(83)
    iq_h0 = (rng.standard_normal((2, 3, 130, 32)) + 1j * rng.standard_normal((2, 3, 130, 32))).astype(np.complex64)
    iq_h1 = (rng.standard_normal((2, 3, 130, 32)) + 1j * rng.standard_normal((2, 3, 130, 32))).astype(np.complex64)
    arrays = {
        "iq_h0": iq_h0,
        "iq_h1": iq_h1,
        "base_scene_id": np.asarray(["v0", "v1"]),
        "truth_range_cell": np.asarray([5.25, 19.75], dtype=np.float32),
        "truth_doppler_hz": np.asarray([25.0, -60.0], dtype=np.float32),
    }
    e002_config["student"]["width"] = 8
    e002_config["student"]["temporal_dilations"] = [1]
    e002_config["training"]["paired_base_scenes_per_update"] = 2
    model = TaskStudent(width=8, temporal_dilations=[1], include_spectrum_head=True)
    result = _paired_validation(model, arrays, e002_config, torch.device("cpu"), 4)
    assert np.isfinite(result["task_loss"])
    assert 0.0 <= result["auc"] <= 1.0
    assert np.isfinite(result["true_cell_doppler_rmse_hz"])
    assert np.isfinite(result["predicted_range_cell_mismatch_rate"])


def test_training_arm_initializes_validation_before_checkpointing(e002_config, tmp_path, monkeypatch):
    e002_config["student"]["width"] = 8
    e002_config["student"]["temporal_dilations"] = [1]
    e002_config["training"]["effective_batch_size"] = 2
    e002_config["training"]["paired_base_scenes_per_update"] = 1
    e002_config["training"]["optimizer_updates_per_arm"] = 1
    e002_config["training"]["validation_interval_updates"] = 1
    e002_config["training"]["optimizer"]["learning_rate_initial"] = 1e-3
    e002_config["training"]["optimizer"]["learning_rate_final"] = 1e-4

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device=None: 0)

    rng = np.random.default_rng(91)
    arrays = {
        "iq_h0": (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32))).astype(np.complex64),
        "iq_h1": (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32))).astype(np.complex64),
        "base_scene_id": np.asarray(["tiny-scene"]),
        "truth_range_cell": np.asarray([6.25], dtype=np.float32),
        "truth_doppler_hz": np.asarray([35.0], dtype=np.float32),
    }
    initial_model = TaskStudent(width=8, temporal_dilations=[1], include_spectrum_head=True)
    result = _train_arm_attempt(
        arm="A",
        pair_plan=np.asarray([[0]], dtype=np.int64),
        train_arrays=arrays,
        validation_arrays=arrays,
        config=e002_config,
        device=torch.device("cpu"),
        initial_state=_cpu_state_dict(initial_model),
        out_dir=tmp_path / "arm_a",
        microbatch_size=2,
        a_best_validation_loss=None,
    )
    assert result["optimizer_updates"] == 1
    assert (tmp_path / "arm_a" / "initial_checkpoint.pt").is_file()
    assert (tmp_path / "arm_a" / "best_inference.pt").is_file()


def test_micro_overfit_sanity_reports_truth_and_k2_gradient_checks(e002_config):
    e002_config["student"]["width"] = 8
    e002_config["student"]["temporal_dilations"] = [1]
    e002_config["training"]["micro_overfit_sanity"] = {
        "c0_pairs": 1,
        "optimizer_updates": 1,
        "effective_batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "final_truth_task_loss_ratio_max": 1.0,
    }
    rng = np.random.default_rng(92)
    teacher = np.full((1, 130, 32), 1.0 / (130 * 32), dtype=np.float32)
    arrays = {
        "iq_h0": (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32))).astype(np.complex64),
        "iq_h1": (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32))).astype(np.complex64),
        "base_scene_id": np.asarray(["sanity-scene"]),
        "course_level": np.asarray(["C0"]),
        "truth_range_cell": np.asarray([6.25], dtype=np.float32),
        "truth_doppler_hz": np.asarray([35.0], dtype=np.float32),
        "teacher_response": teacher,
    }
    result = run_micro_overfit_sanity(arrays, e002_config, device=torch.device("cpu"))
    assert result["n_paired_scenes"] == 1
    assert result["truth_task_final_to_initial_loss_ratio"] >= 0.0
    assert set(result["truth_gradient_norms"]) == {
        "stem", "presence_head", "range_head", "range_offset_head", "doppler_head"
    }
    assert all(np.isfinite(value) and value > 0.0 for value in result["truth_gradient_norms"].values())
    assert set(result["initial_truth_loss_components"]) == {
        "presence", "range_cell", "range_offset", "doppler", "teacher_kl"
    }
    assert set(result["single_task_gradient_norms"]) == {
        "presence", "range_cell", "range_offset", "doppler"
    }
    assert all(
        np.isfinite(value) and value > 0.0
        for task_groups in result["single_task_gradient_norms"].values()
        for value in task_groups.values()
    )
    assert result["k2_gradient_norm"] > 0.0


def test_micro_overfit_sanity_fails_when_one_task_lacks_stem_gradient(e002_config, monkeypatch):
    e002_config["student"]["width"] = 8
    e002_config["student"]["temporal_dilations"] = [1]
    e002_config["training"]["micro_overfit_sanity"] = {
        "c0_pairs": 1,
        "optimizer_updates": 1,
        "effective_batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "final_truth_task_loss_ratio_max": 10.0,
    }
    rng = np.random.default_rng(93)
    arrays = {
        "iq_h0": (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32))).astype(np.complex64),
        "iq_h1": (rng.standard_normal((1, 3, 130, 32)) + 1j * rng.standard_normal((1, 3, 130, 32))).astype(np.complex64),
        "base_scene_id": np.asarray(["gradient-scene"]),
        "course_level": np.asarray(["C0"]),
        "truth_range_cell": np.asarray([6.25], dtype=np.float32),
        "truth_doppler_hz": np.asarray([35.0], dtype=np.float32),
    }
    original_grad = torch.autograd.grad
    call_count = 0

    def zero_one_task_gradient(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = original_grad(*args, **kwargs)
        if call_count == 6:
            return tuple(None for _ in result)
        return result

    monkeypatch.setattr(torch.autograd, "grad", zero_one_task_gradient)
    result = run_micro_overfit_sanity(arrays, e002_config, device=torch.device("cpu"))
    assert result["single_task_gradient_norms"]["presence"]["stem"] == 0.0
    assert result["truth_gradient_groups_passed"] is False
    assert result["passed"] is False
