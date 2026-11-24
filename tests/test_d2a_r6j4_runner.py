from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import numpy as np
import torch
import yaml

import radardistill.training.d2a_r6j4 as r6j4_training
from radardistill.evaluation.d2a_r6j4 import ensure_r6j4_gates_passed
from radardistill.training.d2a_r6j4 import (
    audit_r6j4_scene_plans,
    _accumulate_validation_losses,
    build_r6j4_scene_plans,
    checkpoint_owner_matches,
    _paired_batch,
    r6j4_learning_rate,
    r6j4_presence_labels,
    validate_r6j4_cuda_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-J4/config.yaml").read_text())
R6A = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-A/config.yaml").read_text())


def test_registered_scene_plans_are_deterministic_balanced_and_disjoint() -> None:
    first = build_r6j4_scene_plans(CONFIG, R6A["radar"]["conditions"])
    second = build_r6j4_scene_plans(copy.deepcopy(CONFIG), R6A["radar"]["conditions"])
    assert first == second
    expected = {
        "train": 51_200,
        "validation": 5_000,
        "h0_calibration": 120_000,
        "h0_verification": 120_000,
        "h1_development": 6_144,
        "g1": 16,
    }
    result = audit_r6j4_scene_plans(first, expected_counts=expected, prior_scene_ids=set())
    assert result["prior_overlap_count"] == 0
    assert result["split_overlap_count"] == 0
    assert result["scene_counts"] == expected


def test_cuda_contract_refuses_unavailable_or_wrong_device() -> None:
    kwargs = {
        "required_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
        "required_index": 0,
        "minimum_free_bytes": 1_000,
    }
    with pytest.raises(RuntimeError, match="CUDA"):
        validate_r6j4_cuda_contract(
            available=False, device_name=None, device_index=None, free_bytes=0, **kwargs
        )
    with pytest.raises(RuntimeError, match="device"):
        validate_r6j4_cuda_contract(
            available=True, device_name="different GPU", device_index=0, free_bytes=10_000, **kwargs
        )
    with pytest.raises(RuntimeError, match="free VRAM"):
        validate_r6j4_cuda_contract(
            available=True,
            device_name=kwargs["required_name"],
            device_index=0,
            free_bytes=999,
            **{key: value for key, value in kwargs.items() if key != "required_name"},
        )


def test_each_effective_batch_contains_exactly_32_h0_and_32_h1() -> None:
    labels = r6j4_presence_labels(64)
    assert labels.shape == (64,)
    assert int((labels == 0).sum()) == 32
    assert int((labels == 1).sum()) == 32
    with pytest.raises(ValueError, match="even"):
        r6j4_presence_labels(63)


def test_paired_batch_interleaves_h0_h1_so_microbatches_keep_the_same_objective() -> None:
    iq_h0 = np.zeros((32, 3, 130, 32), dtype=np.complex64)
    iq_h1 = np.ones((32, 3, 130, 32), dtype=np.complex64)
    data = {
        "iq_h0": iq_h0,
        "iq_h1": iq_h1,
        "truth_range_cell": np.full(32, 4.5, dtype=np.float32),
        "truth_doppler_hz": np.full(32, 12.0, dtype=np.float32),
    }
    _, truth = _paired_batch(data, np.arange(32))
    for start in range(0, 64, 16):
        labels = truth["truth_presence"][start : start + 16]
        assert int((labels == 0).sum()) == int((labels == 1).sum()) == 8


def test_cosine_schedule_has_frozen_endpoints_and_is_bounded() -> None:
    assert r6j4_learning_rate(1) == pytest.approx(1e-3)
    assert r6j4_learning_rate(25_600) == pytest.approx(1e-5)
    assert 1e-5 < r6j4_learning_rate(12_800) < 1e-3
    with pytest.raises(ValueError):
        r6j4_learning_rate(25_601)


def test_validation_accumulator_supports_both_arms_distinct_loss_names() -> None:
    j1_totals: dict[str, float] = {}
    _accumulate_validation_losses(
        j1_totals,
        {"presence": 1.0, "range_cell": 2.0, "range_offset": 3.0, "doppler": 4.0,
         "candidate": 5.0, "total": 15.0},
        0.25,
    )
    _accumulate_validation_losses(
        j1_totals,
        {"presence": 3.0, "range_cell": 4.0, "range_offset": 5.0, "doppler": 6.0,
         "candidate": 7.0, "total": 25.0},
        0.75,
    )
    assert j1_totals == {
        "presence": 2.5, "range_cell": 3.5, "range_offset": 4.5,
        "doppler": 5.5, "candidate": 6.5, "total": 22.5,
    }

    j4_totals: dict[str, float] = {}
    _accumulate_validation_losses(
        j4_totals,
        {"presence": 1.0, "candidate": 2.0, "range": 3.0, "doppler": 4.0, "total": 10.0},
        1.0,
    )
    assert j4_totals == {"presence": 1.0, "candidate": 2.0, "range": 3.0, "doppler": 4.0, "total": 10.0}


def test_training_progress_is_json_and_flushes_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlushTrackingWriter:
        def __init__(self) -> None:
            self.value = ""
            self.flush_count = 0

        def write(self, value: str) -> int:
            self.value += value
            return len(value)

        def flush(self) -> None:
            self.flush_count += 1

    output = FlushTrackingWriter()
    monkeypatch.setattr("sys.stdout", output)
    payload = {"stage": "train", "arm": "j1_joint", "update": 400, "validation_total": 0.5}

    r6j4_training._emit_r6j4_training_progress(payload)

    assert json.loads(output.value) == payload
    assert output.flush_count == 1


def test_checkpoint_payload_must_match_all_owner_fields() -> None:
    owner = {
        "owner": "D2-A-R6-J4-training-run",
        "arm": "j4_joint",
        "git_sha": "abc123",
        "protocol_sha256": "protocol",
        "config_sha256": "config",
    }
    payload = {"owner": dict(owner), "update": 800, "model_state": {}}
    assert checkpoint_owner_matches(payload, owner)
    altered = copy.deepcopy(payload)
    altered["owner"]["arm"] = "j1_joint"
    assert not checkpoint_owner_matches(altered, owner)


def test_evaluation_refuses_to_run_without_passing_g0_and_g1() -> None:
    with pytest.raises(RuntimeError, match="G0"):
        ensure_r6j4_gates_passed({"status": "FAIL"}, {"status": "PASS"})
    with pytest.raises(RuntimeError, match="G1"):
        ensure_r6j4_gates_passed({"status": "G0_PASS"}, {"status": "FAIL"})
    ensure_r6j4_gates_passed({"status": "G0_PASS"}, {"status": "G1_PASS"})
