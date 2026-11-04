from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from radardistill.training.d2a_r6h import (
    build_r6h_split_plans,
    ensure_r6h_owned_directory,
    r6h_scene_plan_audit,
    validate_r6h_training_complete,
    validate_r6h_config,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _config(stage: str) -> dict:
    return _load_yaml(ROOT / f"experiments/D2-A-R6/R6-{stage}/config.yaml")


def test_draft_config_is_auditable_but_cannot_start_preflight_or_data_generation() -> None:
    config = _config("H")

    summary = validate_r6h_config(config, require_execution=False)

    assert summary["execution_ready"] is False
    assert summary["final_blind_authorized"] is False
    with pytest.raises(RuntimeError, match="not frozen and authorized"):
        validate_r6h_config(config, require_execution=True)


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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c["training"].update(device="cpu"), "cuda:0"),
        (lambda c: c["training"].update(cpu_fallback="allowed"), "CPU fallback"),
        (lambda c: c["training"].update(per_update_examples={"H0": 16, "H1": 32}), "effective batch"),
        (lambda c: c["training"].update(fixed_updates_per_model=25_000), "25,600"),
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
