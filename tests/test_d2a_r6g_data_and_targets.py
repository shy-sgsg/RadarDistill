from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.training import d2a_r6g


ROOT = Path(__file__).resolve().parents[1]


def _configs() -> tuple[dict, dict]:
    config = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-G/config.yaml").read_text())
    r6a = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-A/config.yaml").read_text())
    return config, r6a


def test_r6g_plans_have_expected_sizes_and_no_cross_split_ids() -> None:
    config, r6a = _configs()
    plans = d2a_r6g.build_r6g_plans(config, r6a)
    expected = {
        "train": 51_200,
        "validation": 5_000,
        "h0_calibration": 120_000,
        "h0_verification": 120_000,
        "h1_development": 6_144,
        "g1_test": 16,
    }
    audit = d2a_r6g.audit_r6g_plans(plans, expected_counts=expected, prior_scene_ids=set())
    assert audit["scene_counts"] == expected
    assert audit["cross_split_overlap_count"] == 0
    g1 = plans["g1_test"][0]
    assert g1["condition"]["course"] == "C3"
    assert g1["condition"]["scenario_id"] == "partial_overlap"
    assert g1["condition"]["snr_db"] == 0.0


def test_r6g_plan_audit_rejects_historical_scene_reuse() -> None:
    config, r6a = _configs()
    plans = d2a_r6g.build_r6g_plans(config, r6a)
    historical = {f"d2a-scene-{plans['train'][0]['seed']}-0"}
    with pytest.raises(RuntimeError, match="prior scene"):
        d2a_r6g.audit_r6g_plans(
            {"train": plans["train"][:1]},
            expected_counts={"train": plans["train"][0]["count"]},
            prior_scene_ids=historical,
        )


def test_teacher_targets_require_finite_normalized_rd_maps() -> None:
    valid = np.full((2, 520, 32), 1.0 / (520 * 32), dtype=np.float32)
    d2a_r6g.validate_r6g_target_maps(valid, expected_count=2)
    invalid = valid.copy()
    invalid[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        d2a_r6g.validate_r6g_target_maps(invalid, expected_count=2)
    with pytest.raises(ValueError, match="normalized"):
        d2a_r6g.validate_r6g_target_maps(valid * 2, expected_count=2)


def test_t3_target_maps_embed_finite_cpi_response_at_truth_gate() -> None:
    import torch

    range_cell = torch.tensor([7.25, 31.75], dtype=torch.float32)
    doppler_hz = torch.tensor([100.0, -650.0], dtype=torch.float32)
    maps = d2a_r6g.build_r6g_t3_maps(range_cell, doppler_hz, device=torch.device("cpu"))
    assert maps.shape == (2, 520, 32)
    assert torch.isfinite(maps).all()
    assert torch.all(maps >= 0)
    assert torch.allclose(maps.sum(dim=(1, 2)), torch.ones(2), atol=1e-6)
    assert torch.count_nonzero(maps[0, :, :7]) == 0
    assert torch.count_nonzero(maps[0, :, 8:]) == 0
    assert torch.count_nonzero(maps[1, :, :31]) == 0
    assert torch.count_nonzero(maps[1, :, 32:]) == 0


def test_cached_response_map_kl_has_finite_nonzero_gradients() -> None:
    import torch

    logits = torch.zeros((2, 520, 32), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 260, 7] = 1.0
    target[1, 300, 11] = 1.0
    loss = d2a_r6g.r6g_response_map_kl(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_cached_response_map_kl_rejects_invalid_targets() -> None:
    import torch

    logits = torch.zeros((1, 520, 32))
    bad = torch.zeros_like(logits)
    with pytest.raises(ValueError, match="positive total mass"):
        d2a_r6g.r6g_response_map_kl(logits, bad)
