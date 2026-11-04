from __future__ import annotations

import numpy as np
import pytest
import torch

from radardistill.training import d2a_r6g


def test_matched_sampler_repeats_each_h1_scene_equally_and_matches_h0_condition() -> None:
    conditions = np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
    h1_plan, h0_plan, audit = d2a_r6g.build_r6g_sampler_plans(
        conditions, updates=16, effective_batch=4, seed=708
    )
    assert h1_plan.shape == h0_plan.shape == (16, 2)
    visits = np.bincount(h1_plan.reshape(-1), minlength=len(conditions))
    assert np.array_equal(visits, np.full(8, 4))
    assert np.array_equal(conditions[h1_plan], conditions[h0_plan])
    assert audit["h1_scene_visits_per_presence"] == 4
    assert audit["h0_condition_matched"] is True


@pytest.mark.parametrize("effective_batch", [0, 3, 5])
def test_sampler_rejects_invalid_effective_batch(effective_batch: int) -> None:
    conditions = np.asarray([0, 0, 1, 1], dtype=np.uint8)
    with pytest.raises(ValueError):
        d2a_r6g.build_r6g_sampler_plans(
            conditions, updates=4, effective_batch=effective_batch, seed=708
        )


@pytest.mark.parametrize("arm", ["T0", "T1"])
def test_training_update_keeps_h0_out_of_auxiliary_targets(arm: str) -> None:
    from radardistill.students.r6g_student import R6GCEStudent

    torch.set_num_threads(1)
    model = R6GCEStudent(width=64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(9)
    batch = {
        "iq_ri": rng.normal(size=(4, 6, 130, 32)).astype(np.float32),
        "truth_presence": np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
        "truth_range_cell": np.asarray([0.0, 7.25, 0.0, 14.5], dtype=np.float32),
        "truth_doppler_hz": np.asarray([0.0, 100.0, 0.0, -650.0], dtype=np.float32),
    }
    target_maps = None
    if arm == "T1":
        values = np.zeros((2, 520, 32), dtype=np.float16)
        values[0, 260, 7] = 1.0
        values[1, 300, 14] = 1.0
        target_maps = {"k3": values, "t3": values}
    config = {"radar": {"prf_hz": 1300.0}, "training": {"teacher_loss_weight": 0.05, "gradient_clip_norm": 1.0}}
    terms = d2a_r6g._r6g_training_update(
        model, optimizer, batch, np.asarray([0, 1]), arm=arm,
        target_maps=target_maps, microbatch=2, effective_batch=4,
        device=torch.device("cpu"), config=config,
    )
    assert all(np.isfinite(value) for value in terms.values())
    assert terms["auxiliary"] == 0.0 if arm == "T0" else terms["auxiliary"] > 0.0
    assert terms["total"] == pytest.approx(terms["truth"] + 0.05 * terms["auxiliary"], rel=1e-5)
