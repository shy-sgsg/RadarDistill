from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.simulation.d0 import generate_d0_batch
from radardistill.training.d1_e002 import (
    _open_or_generate_split,
    balanced_batch_orders,
    d1_anchor_checkpoint,
    require_cuda,
    select_formal_candidates,
    validate_e002_config,
    validate_e002_seed_disjointness,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / "experiments/D1-E002/config.yaml").read_text())


def _metrics(rmse: float, *, auc: float = 0.7, range_rmse: float = 0.2) -> dict:
    return {
        "high_snr_mean_doppler_rmse_hz": rmse,
        "doppler_rmse_by_snr_db": {"0": rmse, "5": rmse, "10": rmse, "20": rmse},
        "low_snr_mean_auc": auc,
        "high_snr_mean_range_rmse_cells": range_rmse,
        "finite_outputs": True,
        "wrapped_frequency_decode_valid": True,
        "completed_updates": 1600,
    }


def test_balanced_batch_orders_are_reproducible_and_class_balanced() -> None:
    first = balanced_batch_orders(4096, 64, 1600, 2026210104)
    second = balanced_batch_orders(4096, 64, 1600, 2026210104)
    assert first.shape == (1600, 64)
    assert (first == second).all()
    assert (first < 4096).sum(axis=1).tolist() == [32] * 1600
    assert (first >= 4096).sum(axis=1).tolist() == [32] * 1600
    assert first.min() >= 0 and first.max() < 8192


def test_cuda_preflight_rejects_cpu_and_missing_device() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        require_cuda("cpu", cuda_available=True, device_count=1)
    with pytest.raises(RuntimeError, match="required"):
        require_cuda("cuda:0", cuda_available=False, device_count=0)


def test_candidate_selection_obeys_frozen_pilot_gates() -> None:
    config = _config()
    pilot = {
        "S1": _metrics(31.0),
        "S2": _metrics(24.0),
        "S3": _metrics(8.0, auc=0.60),
    }
    s0 = _metrics(30.0)
    matched_fft = _metrics(9.0)
    selected = select_formal_candidates(pilot, s0, matched_fft, config)
    assert selected["formal_candidates"] == ["S2"]
    assert selected["candidates"]["S1"]["eligible"] is False
    assert selected["candidates"]["S2"]["eligible"] is True
    assert selected["candidates"]["S3"]["eligible"] is False


def test_seed_validation_rejects_reuse_of_existing_dataset_seed() -> None:
    config = _config()
    with pytest.raises(ValueError, match="overlap"):
        validate_e002_seed_disjointness(config, existing_seed_values={2026210101})


def test_config_validator_preserves_cuda_truth_only_matched_budget() -> None:
    config = _config()
    validate_e002_config(config)
    config["training"]["cpu_fallback"] = True
    with pytest.raises(ValueError, match="CUDA FP32 truth-only"):
        validate_e002_config(config)


def test_anchor_checkpoint_resolution_pairs_pilot_and_formal_seeds() -> None:
    config = _config()
    assert d1_anchor_checkpoint(ROOT, "pilot_seed_1", config) == (
        "experiments/D1-E001/pilot/A_truth_only/best_inference.pt"
    )
    assert d1_anchor_checkpoint(ROOT, "seed_2", config) == config["models"]["S0"]["checkpoints"][1]
    with pytest.raises(KeyError, match="no frozen D1-A anchor"):
        d1_anchor_checkpoint(ROOT, "seed_4", config)


def test_training_cache_is_reproducible_and_keeps_counterfactual_pairing(tmp_path: Path) -> None:
    config = _config()
    split = {
        "id": "cache_test",
        "train_seed": 45001,
        "train_examples_per_class": 4,
        "train_examples_per_chunk": 4,
    }
    cache_path = tmp_path / "train"
    first = _open_or_generate_split(
        root=ROOT,
        config=config,
        split=split,
        split_kind="train",
        cache_path=cache_path,
    )
    second = _open_or_generate_split(
        root=ROOT,
        config=config,
        split=split,
        split_kind="train",
        cache_path=cache_path,
    )
    d0_config = yaml.safe_load((ROOT / "experiments/D0-E001/config.yaml").read_text())
    interval = tuple(config["data"]["training_snr_interval_db"])
    expected_h0 = generate_d0_batch(4, 45001, interval, 0.0, d0_config)
    expected_h1 = generate_d0_batch(4, 45001, interval, 1.0, d0_config)
    np.testing.assert_array_equal(first["iq"][:4], expected_h0.iq)
    np.testing.assert_array_equal(first["iq"][4:], expected_h1.iq)
    np.testing.assert_array_equal(second["iq"], first["iq"])
    assert np.isnan(first["truth_doppler_hz"][:4]).all()
    assert np.isfinite(first["truth_doppler_hz"][4:]).all()
