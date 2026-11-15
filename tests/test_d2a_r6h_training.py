from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from radardistill.training.d2a_r6h import (
    _registered_prior_scene_ids,
    build_r6h_split_plans,
    ensure_r6h_owned_directory,
    r6h_scene_plan_audit,
    validate_r6h_training_complete,
    validate_r6h_config,
)
from radardistill.training import d2a_r6h as r6h_training


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config(stage: str) -> dict:
    return _load_yaml(ROOT / f"experiments/D2-A-R6/R6-{stage}/config.yaml")


def test_frozen_config_is_execution_ready_but_final_blind_remains_forbidden() -> None:
    config = _config("H")

    summary = validate_r6h_config(config, require_execution=True)

    assert summary["execution_ready"] is True
    assert summary["final_blind_authorized"] is False


def test_r6h_prior_scene_audit_includes_registered_d_to_g_plans() -> None:
    r6a = _config("A")
    h_config = _config("H")
    prior_ids_by_stage = _registered_prior_scene_ids(
        r6a, h_config["data"]["prior_registered_plan_stages"]
    )
    h_plans = build_r6h_split_plans(h_config, r6a)
    h_audit = r6h_scene_plan_audit(h_plans, prior_scene_ids=set())
    h_ids = set().union(*h_audit["scene_ids"].values())

    assert set(prior_ids_by_stage) == {"D", "E", "F", "G"}
    assert all(prior_ids_by_stage[stage] for stage in prior_ids_by_stage)
    assert all(h_ids.isdisjoint(ids) for ids in prior_ids_by_stage.values())


def test_preflight_uses_current_head_for_owned_cache_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    config, condition_config, physics = r6h_training.r6h_configs()
    head = "f" * 40
    counts = {
        name: int(config["data"]["splits"][name].get(
            "paired_scenes", config["data"]["splits"][name].get("scene_count", 0)
        ))
        for name in ("train", "validation", "h0_calibration", "h0_verification", "h1_development")
    }
    counts["g1_test"] = 16
    plans = {
        name: [{"seed": 1_000_000_000 + index, "count": count}]
        for index, (name, count) in enumerate(counts.items())
    }
    cache_sha: list[str] = []

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "main"
        if args == ("status", "--porcelain", "--untracked-files=no"):
            return ""
        return head

    class OneParameter:
        @staticmethod
        def numel() -> int:
            return 1

    class TinyModel:
        @staticmethod
        def parameters():
            return iter((OneParameter(),))

    monkeypatch.setattr(r6h_training, "r6h_configs", lambda: (config, condition_config, physics))
    monkeypatch.setattr(r6h_training, "_git", fake_git)
    monkeypatch.setattr(r6h_training, "build_r6h_split_plans", lambda *_args: plans)
    monkeypatch.setattr(r6h_training, "_prior_scene_ids", lambda: set())
    monkeypatch.setattr(
        r6h_training,
        "_registered_prior_scene_ids",
        lambda *_args: {stage: set() for stage in ("D", "E", "F", "G")},
    )
    monkeypatch.setattr(
        r6h_training,
        "r6h_scene_plan_audit",
        lambda *_args, **_kwargs: {
            "scene_counts": counts,
            "prior_overlap_count": 0,
            "overlap_count": 0,
        },
    )
    monkeypatch.setattr(r6h_training, "build_r6h_student", lambda *_args, **_kwargs: TinyModel())
    monkeypatch.setattr(
        r6h_training,
        "_remaining_owned_cache_bytes",
        lambda _plans, git_sha: (cache_sha.append(git_sha) or 0),
    )
    monkeypatch.setattr(
        r6h_training,
        "shutil",
        SimpleNamespace(disk_usage=lambda _path: SimpleNamespace(
            free=40 * 1024**3, total=100 * 1024**3
        )),
    )
    monkeypatch.setattr(
        r6h_training,
        "_require_cuda",
        lambda: (r6h_training.torch.device("cuda:0"), r6h_training.DEVICE_NAME, 2 * 1024**3, 4 * 1024**3),
    )
    monkeypatch.setattr(r6h_training, "_source_sha256", lambda: {})
    monkeypatch.setattr(r6h_training, "_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(r6h_training.subprocess, "check_output", lambda *_args, **_kwargs: "580.178.04\n")
    written: dict = {}
    monkeypatch.setattr(r6h_training, "_write_json", lambda _path, record: written.update(record))

    result = r6h_training.r6h_preflight()

    assert result["git_sha"] == head
    assert cache_sha == [head]
    assert written["status"] == "G0_PASS"


def test_r6h_cache_owner_cannot_rebind_a_nonempty_directory(tmp_path: Path) -> None:
    directory = tmp_path / "cache" / "train"
    owner = {"stage": "D2-A-R6-H", "split": "train", "git_sha": "a" * 40}

    ensure_r6h_owned_directory(directory, owner)
    (directory / "payload.bin").write_bytes(b"owned data")
    ensure_r6h_owned_directory(directory, owner)

    with pytest.raises(RuntimeError, match="refusing to rebind"):
        ensure_r6h_owned_directory(directory, {**owner, "git_sha": "b" * 40})


def test_r6h_cache_owner_refuses_unowned_payload(tmp_path: Path) -> None:
    directory = tmp_path / "cache" / "validation"
    directory.mkdir(parents=True)
    (directory / "payload.bin").write_bytes(b"user data")

    with pytest.raises(RuntimeError, match="unowned"):
        ensure_r6h_owned_directory(
            directory, {"stage": "D2-A-R6-H", "split": "validation"}
        )


def test_cache_cleanup_requires_all_three_training_arms_complete() -> None:
    with pytest.raises(RuntimeError, match="all registered arms must finish"):
        validate_r6h_training_complete({"status": "complete", "arms": {"ce_tcn": {"status": "complete"}}})

    summary = {
        "status": "complete",
        "arms": {arm: {"status": "complete"} for arm in ("ce_tcn", "complex_ssm", "cnn_ssm")},
    }
    validate_r6h_training_complete(summary)


def test_train_all_records_in_progress_before_training_each_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(r6h_training, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(r6h_training, "_require_preflight", lambda **kwargs: (None, None))

    def train_arm(arm: str) -> dict:
        if arm == "complex_ssm":
            raise RuntimeError("simulated interrupted arm")
        return {"status": "complete", "arm": arm}

    monkeypatch.setattr(r6h_training, "r6h_train_arm", train_arm)
    with pytest.raises(RuntimeError, match="simulated interrupted"):
        r6h_training.r6h_train_all()

    summary = json.loads((tmp_path / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "in_progress"
    assert summary["arms"]["ce_tcn"]["status"] == "complete"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["training"].update(device="cpu"), "cuda:0"),
        (lambda c: c["training"].update(cpu_fallback="allowed"), "CPU fallback"),
        (lambda c: c["training"].update(per_update_examples={"H0": 16, "H1": 32}), "effective batch"),
        (lambda c: c["training"].update(fixed_updates_per_model=25_000), "25,600"),
        (lambda c: c["data"].update(prior_registered_plan_stages=["R6-E"]), "prior registered plan stages"),
        (lambda c: c["models"].update(arms=["complex_ssm", "cnn_ssm"]), "three registered arms"),
    ],
)
def test_config_rejects_deviations_from_frozen_protocol_shape(mutate, message: str) -> None:
    config = deepcopy(_config("H"))
    config["experiment"].update(status="preregistered", execution_authorized=True)
    config["prerequisites"] = {
        "r6g_training_complete": True,
        "r6g_gpu_released": True,
        "r6g_completion_sha": "a" * 40,
    }
    mutate(config)

    with pytest.raises((ValueError, RuntimeError), match=message):
        validate_r6h_config(config, require_execution=True)


def test_r6h_split_plans_match_registered_sizes_and_have_no_internal_or_e_g_overlap() -> None:
    r6a = _config("A")
    configs = {stage: _config(stage) for stage in ("E", "G", "H")}
    plans = {
        stage: build_r6h_split_plans(config, r6a)
        for stage, config in configs.items()
    }
    audit = r6h_scene_plan_audit(plans["H"], prior_scene_ids=set())

    assert audit["overlap_count"] == 0
    assert audit["prior_overlap_count"] == 0
    assert audit["scene_counts"] == {
        "train": 51_200,
        "validation": 5_000,
        "h0_calibration": 120_000,
        "h0_verification": 120_000,
        "h1_development": 6_144,
        "g1_test": 16,
    }
    h_ids = set().union(*audit["scene_ids"].values())
    for stage in ("E", "G"):
        other = r6h_scene_plan_audit(plans[stage], prior_scene_ids=set())
        other_ids = set().union(*other["scene_ids"].values())
        assert h_ids.isdisjoint(other_ids)
