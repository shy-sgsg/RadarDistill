from __future__ import annotations

import csv
from types import SimpleNamespace

import numpy as np

from radardistill.training import d2a_r6e


def _fake_generated(count: int = 2) -> SimpleNamespace:
    iq_h0 = np.zeros((count, 3, 130, 32), dtype=np.complex64)
    iq_h1 = np.zeros_like(iq_h0)
    iq_h0.real = 1.0
    iq_h0.imag = 2.0
    iq_h1.real = 3.0
    iq_h1.imag = 4.0
    return SimpleNamespace(
        base_scene_id=np.asarray([f"d2a-scene-2300000000-{i}" for i in range(count)]),
        iq_h0=iq_h0,
        iq_h1=iq_h1,
        truth_target_range_cell=np.arange(count, dtype=np.float32) + 10.25,
        truth_target_doppler_hz=np.arange(count, dtype=np.float32) - 100.0,
    )


def test_all_r6e_split_plans_match_registered_scene_counts() -> None:
    config, r6a, _ = d2a_r6e._configs()
    plans = d2a_r6e._all_split_plans(config, r6a)
    assert d2a_r6e._scene_counts(plans) == {
        "train": 51_200,
        "validation": 5_000,
        "h0_calibration": 120_000,
        "h0_verification": 120_000,
        "h1_development": 6_144,
        "g1_test": 16,
    }


def test_g1_data_uses_paired_cache_contract_and_interleaves_h0_h1() -> None:
    data = d2a_r6e._paired_iq_and_truth(_fake_generated())
    batch = d2a_r6e._g1_batch(data, np.asarray([1, 0]), np.asarray([0, 1]))

    assert batch["iq_ri"].shape == (4, 6, 130, 32)
    assert batch["truth_presence"].tolist() == [False, True, False, True]
    np.testing.assert_array_equal(batch["iq_ri"][0, :3], 1.0)
    np.testing.assert_array_equal(batch["iq_ri"][0, 3:], 2.0)
    np.testing.assert_array_equal(batch["iq_ri"][1, :3], 3.0)
    np.testing.assert_array_equal(batch["iq_ri"][1, 3:], 4.0)
    assert batch["truth_range_cell"].tolist() == [0.0, 10.25, 0.0, 11.25]
    assert batch["truth_doppler_hz"].tolist() == [0.0, -100.0, 0.0, -99.0]


def test_cache_manifest_must_match_frozen_plan_ids(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(d2a_r6e, "STAGE", tmp_path)
    plan = [{"seed": 17, "count": 2}]
    manifest = tmp_path / "scene_manifest_train.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("scene_id",))
        writer.writerow(("unexpected-0",))
        writer.writerow(("unexpected-1",))

    assert not d2a_r6e._cache_manifest_matches(
        "train", plan, {"scene_id": np.asarray(["unexpected-0", "unexpected-1"])}
    )


def test_atomic_checkpoint_writer_accepts_payload_then_path(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    d2a_r6e._atomic_torch_save({"update": 0}, path)

    import torch

    assert torch.load(path, map_location="cpu", weights_only=False) == {"update": 0}


def test_cache_owner_allows_training_only_source_changes_but_checks_data_sources() -> None:
    expected = {
        "owner": "D2-A-R6-E-paired-cache",
        "git_sha": "new-training-code",
        "source_sha256": {"simulator": "sim-v1", "paired_plan": "plan-v1"},
    }
    prior_cache_owner = {
        "owner": "D2-A-R6-E-paired-cache",
        "git_sha": "old-training-code",
        "source_sha256": {
            "trainer": "old-trainer-v1",
            "simulator": "sim-v1",
            "paired_plan": "plan-v1",
        },
    }
    assert d2a_r6e._owner_matches(expected, prior_cache_owner)
    prior_cache_owner["source_sha256"]["simulator"] = "sim-v2"
    assert not d2a_r6e._owner_matches(expected, prior_cache_owner)
