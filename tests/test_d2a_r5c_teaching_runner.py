from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from radardistill.distillation.r5c_teaching import response_kl
from radardistill.training import d2a_r5c_teaching_pilot as pilot
from radardistill.training.d2a_r5c_teaching_pilot import (
    assert_split_seed_disjoint,
    build_arm_model,
    build_initial_state,
    build_pilot_schedule,
    build_split_seed_specs,
    compute_h1_teacher_maps,
    compute_training_microbatch_loss,
    h1_response_loss,
    make_paired_examples,
    scale_pilot_microbatch_loss,
    train_arm,
    train_arms_sequentially,
    validate_r5c_teaching_config,
    _task_head_gradient_norms,
    _truncate_history_to_update,
    _validate_pair_plan,
)
from scripts import run_d2a_r5c_teaching_pilot as runner


CONFIG_PATH = Path("experiments/D2-A-R5/R5-C-teaching/config.yaml")


@pytest.fixture
def config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_draft_config_freezes_four_matched_arms_and_resource_budget(config: dict) -> None:
    validate_r5c_teaching_config(config)

    assert config["experiment"]["status"] == "draft"
    assert config["experiment"]["run_authorized"] is False
    assert config["training"]["arms"] == ["T0", "T1", "T2", "T3"]
    assert config["training"]["updates"] == 12800
    assert config["training"]["effective_batch_size_cpi"] == 64
    assert config["resources"]["device"] == "cuda:0"


def test_validator_accepts_review_supported_preregistration_record(config: dict) -> None:
    approved = deepcopy(config)
    approved["experiment"].update(
        status="preregistered",
        protocol_version=1.0,
        independent_chatgpt_review={
            "status": "returned",
            "target_commit": "a" * 40,
            "decision": "CONTINUE",
            "supports_pilot": True,
        },
        run_authorized=True,
    )
    approved["training"]["execution_authorized"] = True

    validate_r5c_teaching_config(approved)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "arms", ["T0", "T1"]),
        ("training", "updates", 6400),
        ("training", "teacher_loss_weight", 0.1),
        ("resources", "cpu_fallback", "allowed"),
        ("data", "condition_seed_stride", 100),
    ],
)
def test_draft_validator_rejects_changes_to_frozen_design(
    config: dict, section: str, key: str, value
) -> None:
    invalid = deepcopy(config)
    invalid[section][key] = value

    with pytest.raises(ValueError):
        validate_r5c_teaching_config(invalid)


def test_draft_validator_rejects_changed_false_alarm_and_overfit_gates(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["evaluation"]["c3_h0_pfa_point_limit"] = 0.20
    with pytest.raises(ValueError):
        validate_r5c_teaching_config(invalid)

    invalid = deepcopy(config)
    invalid["gates"]["g1_overfit"]["updates"] = 100
    with pytest.raises(ValueError):
        validate_r5c_teaching_config(invalid)


def test_draft_validator_rejects_unregistered_extra_fields(config: dict) -> None:
    invalid = deepcopy(config)
    invalid["training"]["unregistered_loss_search"] = True

    with pytest.raises(ValueError, match="fields differ"):
        validate_r5c_teaching_config(invalid)


def test_frozen_input_manifest_covers_evaluation_and_scene_generation_dependencies():
    required_inputs = {
        "src/radardistill/baselines.py",
        "src/radardistill/evaluation/d2a_e002.py",
        "src/radardistill/simulation/d0.py",
        "src/radardistill/simulation/d2a.py",
        "scripts/run_d2a_r5c_teacher_audit.py",
    }

    assert required_inputs <= set(pilot.FROZEN_INPUTS)


def test_seed_specs_have_exact_scene_counts_and_blocked_chunks(config: dict) -> None:
    validate_r5c_teaching_config(config)
    specs = build_split_seed_specs(config)

    assert len(specs) == 1564
    assert sum(spec["scene_count"] for spec in specs) == 24832
    assert {spec["scene_count"] for spec in specs} <= set(range(1, 17))
    assert len({spec["seed"] for spec in specs}) == len(specs)
    split_scenes = {}
    for spec in specs:
        split_scenes[spec["split"]] = split_scenes.get(spec["split"], 0) + spec["scene_count"]
    assert split_scenes == {
        "h0_calibration": 8192,
        "h0_verification": 8192,
        "c3_h0_verification": 4096,
        "h1_development": 4352,
    }


def test_seed_formula_uses_local_condition_indices_and_fixed_strides(config: dict) -> None:
    specs = build_split_seed_specs(config)
    calibration = [spec for spec in specs if spec["split"] == "h0_calibration"]
    c3_verification = [spec for spec in specs if spec["split"] == "c3_h0_verification"]

    assert calibration[0]["seed"] == 728_000_000
    assert calibration[1]["seed"] == 728_001_009
    assert calibration[128]["condition_index"] == 1
    assert calibration[128]["seed"] == 729_000_000
    assert c3_verification[0]["condition_index"] == 0
    assert c3_verification[0]["seed"] == 768_000_000
    assert c3_verification[43]["condition_index"] == 1
    assert c3_verification[43]["seed"] == 769_000_000


def test_split_seed_disjointness_rejects_any_reserved_block_collision(config: dict) -> None:
    specs = build_split_seed_specs(config)

    assert_split_seed_disjoint(specs, [727_999_999, 729_000_000_000])
    with pytest.raises(ValueError, match="collision"):
        assert_split_seed_disjoint(specs, [728_000_123])
    with pytest.raises(ValueError, match="duplicate"):
        assert_split_seed_disjoint(specs, [727_999_999, 727_999_999])


def test_split_seed_disjointness_does_not_auto_renumber_collisions(config: dict) -> None:
    specs = build_split_seed_specs(config)
    collision = specs[0]["seed"]

    with pytest.raises(ValueError, match="collision"):
        assert_split_seed_disjoint(specs, [collision])


def test_all_arms_load_the_same_seeded_initial_state_bit_for_bit():
    state = build_initial_state(303)
    models = [build_arm_model(arm, state) for arm in ("T0", "T1", "T2", "T3")]

    assert sum(parameter.numel() for parameter in models[0].parameters()) == 120717
    assert all(model.spectrum_frequency_bins == 520 for model in models)
    for model in models:
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    for model in models:
        del model


def test_paired_examples_preserve_shared_h0_h1_interleave():
    arrays = {
        "iq_h0": np.zeros((2, 3, 130, 32), dtype=np.complex64),
        "iq_h1": np.ones((2, 3, 130, 32), dtype=np.complex64),
        "truth_range_cell": np.asarray([4.5, 7.5], dtype=np.float32),
        "truth_doppler_hz": np.asarray([20.0, -30.0], dtype=np.float32),
    }

    batch = make_paired_examples(arrays, np.asarray([1, 0]), include_teacher=False)

    assert batch["truth_presence"].tolist() == [False, True, False, True]
    assert batch["truth_range_cell"].tolist() == [0.0, 7.5, 0.0, 4.5]
    assert np.all(batch["iq_ri"][0] == 0.0)
    assert np.all(batch["iq_ri"][1, :3] == 1.0)
    assert np.all(batch["iq_ri"][1, 3:] == 0.0)


def test_response_loss_receives_only_h1_rows_and_has_no_h0_teacher_loss():
    logits = torch.randn(4, 520, 32, requires_grad=True)
    truth = {"truth_presence": torch.tensor([False, True, False, True])}
    teacher = torch.rand(2, 520, 32)
    teacher = teacher / teacher.sum(dim=(1, 2), keepdim=True)
    outputs = {"spectrum_logits": logits}

    loss = h1_response_loss(outputs, truth, "T1", teacher_map=teacher)
    expected = response_kl(logits[1::2], "T1", teacher_map=teacher)
    torch.testing.assert_close(loss, expected)

    h0_truth = {"truth_presence": torch.zeros(4, dtype=torch.bool)}
    h0_loss = h1_response_loss(outputs, h0_truth, "T1", teacher_map=None)
    assert h0_loss.item() == 0.0


def test_k3_teacher_map_helper_uses_only_explicit_h1_cuda_input(monkeypatch):
    calls = []

    def require_cuda(device):
        calls.append(("device", device))
        return torch.device("cpu")

    def fake_teacher(iq, config, *, device, oversampling_factor):
        calls.append(("teacher", np.asarray(iq).shape, device, oversampling_factor))
        return np.full((len(iq), 520, 32), 1.0 / (520 * 32), dtype=np.float32)

    monkeypatch.setattr(pilot, "_require_training_cuda", require_cuda)
    monkeypatch.setattr(pilot, "k2_response_cuda", fake_teacher)
    iq_h1 = np.ones((3, 3, 130, 32), dtype=np.complex64)

    result = compute_h1_teacher_maps(iq_h1, {"frozen": True}, "cuda:0")

    assert calls == [
        ("device", "cuda:0"),
        ("teacher", (3, 3, 130, 32), "cuda:0", 4),
    ]
    assert result.shape == (3, 520, 32)
    assert result.device.type == "cpu"


def test_training_loss_invokes_teacher_only_for_h1_rows(config: dict, monkeypatch):
    class TinyTaskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.2))

        def forward(self, iq_ri, *, compute_spectrum=True):
            batch_size = iq_ri.shape[0]
            zeros = self.scale * torch.zeros(batch_size, device=iq_ri.device)
            base = self.scale * torch.ones(batch_size, 32, device=iq_ri.device)
            vector = torch.stack((base, base), dim=-1)
            result = {
                "presence_logit": zeros,
                "range_logits": base,
                "range_offset": torch.sigmoid(base),
                "doppler_vector": torch.nn.functional.normalize(vector, dim=-1),
            }
            if compute_spectrum:
                result["spectrum_logits"] = self.scale * torch.ones(
                    batch_size, 520, 32, device=iq_ri.device
                )
            return result

    calls = []

    def fake_h1_maps(iq_h1, physics_config, device):
        calls.append(np.asarray(iq_h1).copy())
        return torch.full(
            (len(iq_h1), 520, 32), 1.0 / (520 * 32), dtype=torch.float32, device=device
        )

    monkeypatch.setattr(pilot, "compute_h1_teacher_maps", fake_h1_maps)
    batch = {
        "iq_ri": np.zeros((4, 6, 130, 32), dtype=np.float32),
        "truth_presence": np.asarray([False, True, False, True]),
        "truth_range_cell": np.asarray([0.0, 5.5, 0.0, 10.2], dtype=np.float32),
        "truth_doppler_hz": np.asarray([0.0, 12.0, 0.0, -44.0], dtype=np.float32),
    }
    h1_iq = np.ones((2, 3, 130, 32), dtype=np.complex64)
    model = TinyTaskModel()

    teacher_loss, _ = compute_training_microbatch_loss(
        model, batch, "T1", config, "cpu", physics_config={}, h1_iq=h1_iq
    )
    local_teacher_loss, _ = compute_training_microbatch_loss(
        model, batch, "T2", config, "cpu", physics_config={}, h1_iq=h1_iq
    )
    truth_loss, _ = compute_training_microbatch_loss(
        model, batch, "T0", config, "cpu", physics_config={}, h1_iq=None
    )
    truth_assisted_loss, _ = compute_training_microbatch_loss(
        model, batch, "T3", config, "cpu", physics_config={}, h1_iq=None
    )

    assert len(calls) == 2
    assert calls[0].shape == (2, 3, 130, 32)
    assert calls[1].shape == (2, 3, 130, 32)
    assert torch.isfinite(teacher_loss)
    assert torch.isfinite(local_teacher_loss)
    assert torch.isfinite(truth_loss)
    assert torch.isfinite(truth_assisted_loss)


def test_pilot_microbatch_scaling_matches_one_effective_batch(config: dict):
    losses_1 = {
        "presence": torch.tensor(0.4), "range_cell": torch.tensor(0.8),
        "range_offset": torch.tensor(0.2), "doppler": torch.tensor(0.1),
        "teacher_kl": torch.tensor(2.0),
    }
    losses_2 = {
        "presence": torch.tensor(0.6), "range_cell": torch.tensor(0.4),
        "range_offset": torch.tensor(0.6), "doppler": torch.tensor(0.3),
        "teacher_kl": torch.tensor(1.0),
    }

    scaled = scale_pilot_microbatch_loss(
        losses_1, config, arm="T1", microbatch_size=2, effective_batch_size=4,
        microbatch_h1=1, effective_h1=2,
    ) + scale_pilot_microbatch_loss(
        losses_2, config, arm="T1", microbatch_size=2, effective_batch_size=4,
        microbatch_h1=1, effective_h1=2,
    )

    assert scaled.item() == pytest.approx(0.5 + 0.6 + 0.4 + 0.2 + 0.075)


def test_pilot_schedule_matches_frozen_two_segment_r5_boundaries(config: dict):
    schedule = build_pilot_schedule(config)

    assert schedule.lr_for_update(1) == pytest.approx(0.001)
    assert schedule.lr_for_update(6400) == pytest.approx(0.0001)
    assert schedule.lr_for_update(6401) == pytest.approx(0.0001)
    assert schedule.lr_for_update(12800) == pytest.approx(0.00001)


def test_pair_plan_validator_requires_eight_full_fixed_cohort_passes():
    epoch = np.arange(51200, dtype=np.int64).reshape(1600, 32)
    valid_plan = np.tile(epoch, (8, 1))

    assert _validate_pair_plan(valid_plan, 51200).shape == (12800, 32)
    invalid = valid_plan.copy()
    invalid[0, 0], invalid[0, 1] = invalid[0, 1], invalid[0, 0]
    invalid[1, 0] = invalid[0, 0]
    with pytest.raises(ValueError, match="every fixed training scene"):
        _validate_pair_plan(invalid, 51200)


def test_training_arms_run_sequentially_in_preregistered_order(config: dict, monkeypatch):
    events = []

    def fake_train(arm, **kwargs):
        events.append(("start", arm))
        events.append(("finish", arm))
        return {"arm": arm}

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append(("empty", None)))
    results = train_arms_sequentially(
        initial_state={}, train_arrays={}, validation_arrays={}, pair_plan=np.zeros((1, 1)),
        config=config, device=torch.device("cpu"), microbatch=16, arm_root=Path("unused"),
        physics_config={}, trainer=fake_train,
    )

    assert [item["arm"] for item in results] == ["T0", "T1", "T2", "T3"]
    assert events == [
        ("start", "T0"), ("finish", "T0"), ("empty", None),
        ("start", "T1"), ("finish", "T1"), ("empty", None),
        ("start", "T2"), ("finish", "T2"), ("empty", None),
        ("start", "T3"), ("finish", "T3"), ("empty", None),
    ]


def test_draft_config_blocks_training_before_cuda_or_data_access(config: dict):
    with pytest.raises(RuntimeError, match="independent R5-C review"):
        train_arm(
            "T0", {}, {}, np.empty((0, 0), dtype=np.int64), config,
            torch.device("cpu"), 16, validation_arrays={}, arm_root=Path("unused"),
            physics_config={},
        )


def test_draft_config_blocks_preflight_before_git_cuda_or_data_access(config: dict, monkeypatch):
    monkeypatch.setattr(
        runner,
        "_require_committed_main",
        lambda: pytest.fail("preflight reached git checks before review authorization"),
    )
    monkeypatch.setattr(
        runner,
        "_require_training_cuda",
        lambda *_: pytest.fail("preflight reached CUDA checks before review authorization"),
    )

    with pytest.raises(RuntimeError, match="independent R5-C review"):
        runner.run_preflight(config)


def test_preflight_freezes_the_k2_teachers_baseline_dependency():
    assert {
        "src/radardistill/d2a_baselines.py",
        "src/radardistill/distillation/task_losses.py",
        "src/radardistill/evaluation/d2a_e001.py",
    } <= set(runner.FROZEN_INPUTS)


def test_preflight_rejects_authorized_config_not_equal_to_committed_config(config: dict, monkeypatch):
    authorized = deepcopy(config)
    authorized["experiment"].update(
        status="preregistered",
        protocol_version=1.0,
        independent_chatgpt_review={
            "status": "returned",
            "target_commit": "a" * 40,
            "decision": "CONTINUE",
            "supports_pilot": True,
        },
        run_authorized=True,
    )
    authorized["training"]["execution_authorized"] = True
    monkeypatch.setattr(runner, "_require_committed_main", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_read_yaml", lambda _path: config)
    monkeypatch.setattr(
        runner,
        "_require_training_cuda",
        lambda *_: pytest.fail("preflight reached CUDA before checking committed config"),
    )

    with pytest.raises(RuntimeError, match="must match the committed config"):
        runner.run_preflight(authorized)


def test_low_level_training_requires_preflight_and_passing_g1(config: dict, monkeypatch, tmp_path):
    authorized = deepcopy(config)
    authorized["experiment"].update(
        status="preregistered",
        protocol_version=1.0,
        independent_chatgpt_review={
            "status": "returned",
            "target_commit": "a" * 40,
            "decision": "CONTINUE",
            "supports_pilot": True,
        },
        run_authorized=True,
    )
    authorized["training"]["execution_authorized"] = True
    monkeypatch.setattr(pilot, "_STAGE_ROOT", tmp_path)
    monkeypatch.setattr(pilot, "_require_training_main_head", lambda: "b" * 40)
    monkeypatch.setattr(
        pilot,
        "_require_training_cuda",
        lambda *_: pytest.fail("training reached CUDA before checking preflight/G1"),
    )

    with pytest.raises(RuntimeError, match="preflight"):
        train_arm(
            "T0", {}, {}, np.empty((0, 0), dtype=np.int64), authorized,
            torch.device("cpu"), 16, validation_arrays={}, arm_root=tmp_path / "T0",
            physics_config={},
        )

    (tmp_path / "preflight.json").write_text(
        json.dumps(
            {"git_sha": "b" * 40, "passed": True, "config": authorized}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="G1"):
        train_arm(
            "T0", {}, {}, np.empty((0, 0), dtype=np.int64), authorized,
            torch.device("cpu"), 16, validation_arrays={}, arm_root=tmp_path / "T0",
            physics_config={},
        )

    passing_arms = {arm: {"passed": True} for arm in authorized["training"]["arms"]}
    g1_path = tmp_path / "g1_overfit_sanity.json"
    stale = {"git_sha": "c" * 40, "config": authorized, "passed": True, "arms": passing_arms}
    g1_path.write_text(json.dumps(stale), encoding="utf-8")
    with pytest.raises(RuntimeError, match="G1"):
        train_arm(
            "T0", {}, {}, np.empty((0, 0), dtype=np.int64), authorized,
            torch.device("cpu"), 16, validation_arrays={}, arm_root=tmp_path / "T0",
            physics_config={},
        )

    current = {**stale, "git_sha": "b" * 40}
    g1_path.write_text(json.dumps(current), encoding="utf-8")
    pilot._require_training_preflight(authorized, require_g1=True)


def test_trainer_boundary_rejects_dirty_frozen_files(monkeypatch, tmp_path):
    monkeypatch.setattr(pilot, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(pilot, "FROZEN_INPUTS", ("AGENTS.md",))

    def fake_run(args, **_kwargs):
        command = tuple(args)
        if command == ("git", "branch", "--show-current"):
            return SimpleNamespace(returncode=0, stdout="main\n")
        if command == ("git", "rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        if command == ("git", "rev-parse", "origin/main"):
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n")
        if command == ("git", "ls-files", "--error-unmatch", "AGENTS.md"):
            return SimpleNamespace(returncode=0, stdout="AGENTS.md\n")
        if command == ("git", "diff", "--quiet", "HEAD", "--", "AGENTS.md"):
            return SimpleNamespace(returncode=1, stdout="")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="has local changes"):
        pilot._require_training_main_head()


def test_preflight_model_parameter_counts_include_training_and_inference(config: dict):
    training, inference = runner._model_parameter_counts()

    assert training == config["model"]["training_parameter_count_expected"] == 120717
    assert inference == config["model"]["inference_parameter_count_expected"] == 70277


def test_resume_history_trims_uncheckpointed_rows_even_at_cursor_zero(tmp_path):
    history = tmp_path / "training_history.csv"
    history.write_text("update,arm\n0,T0\n1,T0\n2,T0\n", encoding="utf-8")

    _truncate_history_to_update(history, 0)

    assert history.read_text(encoding="utf-8") == "update,arm\n0,T0\n"


def test_g1_checks_gradient_norms_for_all_four_task_heads():
    class Heads(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.presence_per_range = torch.nn.Linear(1, 1, bias=False)
            self.range_head = torch.nn.Linear(1, 1, bias=False)
            self.range_offset_head = torch.nn.Linear(1, 1, bias=False)
            self.doppler_head = torch.nn.Linear(1, 1, bias=False)

    model = Heads()
    x = torch.ones(1, 1)
    terms = {
        "presence": model.presence_per_range(x).sum(),
        "range_cell": model.range_head(x).sum(),
        "range_offset": model.range_offset_head(x).sum() * 0.0,
        "doppler": model.doppler_head(x).sum(),
    }

    norms = _task_head_gradient_norms(model, terms)

    assert set(norms) == {"presence", "range", "range_offset", "doppler"}
    assert norms["presence"] > 0.0
    assert norms["range"] > 0.0
    assert norms["range_offset"] == 0.0
    assert norms["doppler"] > 0.0


def test_overfit_retries_cuda_oom_with_smaller_microbatch(config: dict, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "_require_preflight_complete", lambda _config: {})
    monkeypatch.setattr(runner, "_require_committed_main", lambda: "a" * 40)
    monkeypatch.setattr(runner, "_load_training_inputs", lambda _config: ({}, {}, {}, {}))
    monkeypatch.setattr(runner, "build_initial_state", lambda _seed: {})
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)

    def fake_overfit_sanity(**kwargs):
        calls.append(kwargs["microbatch"])
        if len(calls) == 1:
            raise torch.cuda.OutOfMemoryError("mock G1 OOM")
        return {"passed": True}

    monkeypatch.setattr(runner, "run_overfit_sanity", fake_overfit_sanity)

    result = runner.run_overfit(config)

    assert calls == [16, 8]
    assert result["microbatch_attempts"] == [16, 8]


def test_evaluation_uses_and_persists_only_requested_presence_side(
    monkeypatch, tmp_path, config: dict
):
    evaluation_config = deepcopy(config)
    for split in evaluation_config["data"]["splits"].values():
        split["scene_count"] = 1
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(runner, "_require_training_authorization", lambda _: None)
    monkeypatch.setattr(
        runner,
        "_require_preflight_complete",
        lambda *args, **kwargs: {"git_sha": "a" * 40},
    )
    monkeypatch.setattr(runner, "_require_committed_main", lambda: "a" * 40)
    monkeypatch.setattr(
        runner,
        "_load_evaluation_models",
        lambda *_: {arm: torch.nn.Identity() for arm in evaluation_config["training"]["arms"]},
    )
    monkeypatch.setattr(runner, "_require_training_cuda", lambda _: torch.device("cpu"))
    monkeypatch.setattr(runner, "_read_yaml", lambda _: {})
    monkeypatch.setattr(runner, "summarize_pilot_arm", lambda *args: {})
    monkeypatch.setattr(runner, "paired_teaching_contrasts", lambda *args: {})
    saved_payloads = {}
    monkeypatch.setattr(
        runner,
        "_save_arm_predictions",
        lambda arm, payload: saved_payloads.setdefault(arm, payload),
    )
    monkeypatch.setattr(runner.torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_reserved", lambda *_: 0)
    monkeypatch.setattr(runner.torch.cuda, "get_device_name", lambda *_: "mock CUDA")

    split_names = (
        "h0_calibration",
        "h0_verification",
        "c3_h0_verification",
        "h1_development",
    )

    def fake_chunks(_, __, split_name):
        course = "C3" if split_name == "c3_h0_verification" else "C0"
        metadata = {
            "base_scene_ids": np.asarray([f"{split_name}-scene"]),
            "course_level": np.asarray([course]),
            "scenario_id": np.asarray(["separated"]),
            "snr_db": np.asarray([0.0]),
            "requested_cnr_db": np.asarray([20.0]),
            "condition_index": np.asarray([0]),
        }
        batch = SimpleNamespace(
            iq_h0=np.zeros((1, 3, 130, 32), dtype=np.complex64),
            iq_h1=np.ones((1, 3, 130, 32), dtype=np.complex64),
            truth_target_range_cell=np.asarray([1.5]),
            truth_target_doppler_hz=np.asarray([12.0]),
        )
        yield {"spec": {"split": split_name, "scene_count": 1}, "batch": batch, "metadata": metadata}

    monkeypatch.setattr(runner, "iter_generated_split_chunks", fake_chunks)
    calls = []

    def fake_predict(_model, iq, _device, *, truth_range_cell=None, prf_hz=1300.0):
        marker = float(np.asarray(iq).real.flat[0])
        calls.append((marker, truth_range_cell is not None))
        prediction = {
            "score": np.asarray([0.5]),
            "predicted_range_cell": np.asarray([1.5]),
            "predicted_range_bin": np.asarray([1]),
            "predicted_doppler_hz": np.asarray([12.0]),
        }
        if truth_range_cell is not None:
            prediction["true_range_gated_doppler_hz"] = np.asarray([12.0])
        return prediction

    monkeypatch.setattr(runner, "predict_task_student_chunk", fake_predict)

    result = runner.run_evaluation(evaluation_config)

    assert result["split_scene_counts"] == {name: 1 for name in split_names}
    assert len(calls) == 16
    assert calls.count((1.0, True)) == 4
    assert all("h0_score" not in payload["development"] for payload in saved_payloads.values())
