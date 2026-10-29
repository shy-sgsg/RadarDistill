from __future__ import annotations

import numpy as np
import pytest

from radardistill.training.d2a_r6b_hard_negative import (
    assemble_independent_h0_h1_batch,
    build_h0_sample_plan,
    build_hard_negative_pools,
    build_r6b_paired_split_plan,
    build_repeated_h1_plan,
    clopper_pearson_upper,
    required_zero_event_sample_size,
)


def test_zero_event_sample_size_and_exact_upper_bound():
    assert [required_zero_event_sample_size(p) for p in (0.05, 0.01, 0.001, 0.0001, 1e-6)] == [59, 299, 2995, 29956, 2995731]
    assert clopper_pearson_upper(0, 59) == pytest.approx(1 - 0.05 ** (1 / 59))
    assert 0.0 < clopper_pearson_upper(1, 100) < 0.1
    with pytest.raises(ValueError):
        clopper_pearson_upper(2, 1)


def test_balanced_train_validation_plan_has_disjoint_new_scene_seeds():
    import yaml

    r6b = yaml.safe_load(open("experiments/D2-A-R6/R6-B/config.yaml", encoding="utf-8"))
    r6a = yaml.safe_load(open("experiments/D2-A-R6/R6-A/config.yaml", encoding="utf-8"))
    train = build_r6b_paired_split_plan(r6b["data"], r6a["radar"]["conditions"], "train")
    validation = build_r6b_paired_split_plan(r6b["data"], r6a["radar"]["conditions"], "validation")
    assert sum(row["count"] for row in train) == 51_200
    assert sum(row["count"] for row in validation) == 5_000
    train_seeds = {row["seed"] for row in train}
    validation_seeds = {row["seed"] for row in validation}
    assert len(train_seeds) == len(train)
    assert len(validation_seeds) == len(validation)
    assert train_seeds.isdisjoint(validation_seeds)
    for plan, expected_per_course in ((train, 12_800), (validation, 1_250)):
        for course in ("C0", "C1", "C2", "C3"):
            assert sum(row["count"] for row in plan if row["condition"]["course"] == course) == expected_per_course
    for plan in (train, validation):
        for course in ("C2", "C3"):
            counts = [sum(row["count"] for row in plan if row["condition_index"] == i)
                      for i, condition in enumerate(r6a["radar"]["conditions"])
                      if condition["course"] == course]
            assert max(counts) - min(counts) <= 1


def test_all_r6b_split_plans_are_unique_and_match_registered_scene_counts():
    from scripts.run_d2a_r6b_low_pfa import _all_split_plans, _configs, _planned_ids

    config, r6a, _, _ = _configs()
    plans = _all_split_plans(config, r6a)
    expected = {"train": 51200, "validation": 5000, "h0_calibration": 120000,
                "h0_verification": 120000, "h1_development": 6144}
    assert {name: sum(row["count"] for row in plan) for name, plan in plans.items()} == expected
    seeds = [row["seed"] for plan in plans.values() for row in plan]
    assert len(seeds) == len(set(seeds))
    assert len(_planned_ids(plans)) == sum(expected.values())


def test_r6b_optimizer_schedule_inherits_frozen_two_segment_r5_schedule():
    from scripts.run_d2a_r6b_low_pfa import _r6b_learning_rate

    assert _r6b_learning_rate(1) == pytest.approx(0.001)
    assert _r6b_learning_rate(6400) == pytest.approx(0.0001)
    assert _r6b_learning_rate(6401) == pytest.approx(0.0001)
    assert _r6b_learning_rate(12800) == pytest.approx(0.00001)


def test_seed303_medium_state_loads_identically_into_both_matched_arms():
    import torch
    from scripts.run_d2a_r6b_low_pfa import _initial_medium_state, _task_model

    state = _initial_medium_state(303)
    first = _task_model(state, device=torch.device("cpu"))
    second = _task_model(state, device=torch.device("cpu"))
    assert sum(parameter.numel() for parameter in first.parameters()) == 82887
    assert all(torch.equal(state[key], value) for key, value in first.state_dict().items())
    assert all(torch.equal(first.state_dict()[key], value) for key, value in second.state_dict().items())


def test_r6b_evaluation_loader_uses_frozen_checkpoint_from_r6a_config(monkeypatch):
    import torch

    from radardistill.students.task_student import TaskStudent
    from scripts import run_d2a_r6b_low_pfa as runner

    frozen_paths = []

    def fake_load_frozen_medium(checkpoint_source, checkpoint_record):
        frozen_paths.append((checkpoint_source, checkpoint_record))
        return TaskStudent(width=96, include_spectrum_head=False)

    training_model = TaskStudent(width=96, include_spectrum_head=True)
    training_state = training_model.state_dict()
    monkeypatch.setattr(runner, "load_frozen_medium", fake_load_frozen_medium)
    monkeypatch.setattr(
        torch,
        "load",
        lambda *_args, **_kwargs: {"model_state_dict": training_state},
    )

    states = runner._load_evaluation_states({
        "truth_only": {"output_dir": "unused/truth_only"},
        "hard_negative": {"output_dir": "unused/hard_negative"},
    })

    assert frozen_paths == [(
        str(runner.ROOT / "experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/best_checkpoint.pt"),
        str(runner.ROOT / "experiments/D2-A-R5/R5-A/capacity/runs/seed_101/Medium/training_record.json"),
    )]
    assert set(states) == {"Medium_Frozen", "Medium_TruthOnly", "Medium_HardNegative"}
    assert sum(value.numel() for value in states["Medium_TruthOnly"].values()) == 70277


def test_r6b_review_request_writes_literal_protocol_config_path(tmp_path, monkeypatch):
    import json

    from scripts import run_d2a_r6b_low_pfa as runner

    stage = tmp_path / "R6-B"
    runs = tmp_path / "runs" / "seed_303"
    stage.mkdir(parents=True)
    runs.mkdir(parents=True)
    (runs / "training.json").write_text(json.dumps({}), encoding="utf-8")
    monkeypatch.setattr(runner, "STAGE", stage)
    monkeypatch.setattr(runner, "RUNS", runs)

    runner._write_chatgpt_review_request({"git_sha": "test-commit"})

    review = (stage / "review_request.md").read_text(encoding="utf-8")
    assert "experiments/D2-A-R6/R6-B/{protocol.md,config.yaml}" in review


def test_hard_pools_are_top_quartile_with_deterministic_tie_breaking():
    scores = np.asarray([1, 9, 9, 2, 3, 3, 0, 0], dtype=float)
    conditions = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    pools = build_hard_negative_pools(scores, conditions, top_fraction=0.25)
    np.testing.assert_array_equal(pools[0], [1])
    np.testing.assert_array_equal(pools[1], [4])
    with pytest.raises(ValueError):
        build_hard_negative_pools(scores, conditions, top_fraction=0)


def test_h0_sampler_keeps_h1_order_and_condition_quotas_matched():
    h1_condition = np.asarray([0, 0, 1, 1])
    h1_plan = np.asarray([[0, 2], [1, 3]])
    h0_condition = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    hard = {0: np.asarray([0]), 1: np.asarray([4])}
    baseline, base_audit = build_h0_sample_plan(
        h1_plan, h1_condition, h0_condition, hard, mode="uniform", seed=31, hard_fraction=0.5
    )
    intervention, intervention_audit = build_h0_sample_plan(
        h1_plan, h1_condition, h0_condition, hard, mode="hard_negative", seed=47, hard_fraction=0.5
    )
    assert h1_plan.shape == baseline.shape == intervention.shape
    assert not np.array_equal(baseline, intervention)
    np.testing.assert_array_equal(h0_condition[baseline], h1_condition[h1_plan])
    np.testing.assert_array_equal(h0_condition[intervention], h1_condition[h1_plan])
    assert base_audit["mode"] == "uniform"
    assert intervention_audit["hard_fraction_by_condition"] == {"0": 0.5, "1": 0.5}
    assert intervention_audit["h0_slots_by_condition"] == {"0": 2, "1": 2}
    with pytest.raises(ValueError, match="hard pool"):
        build_h0_sample_plan(
            h1_plan, h1_condition, h0_condition, {}, mode="hard_negative", seed=47, hard_fraction=0.5
        )


def test_repeated_h1_plan_covers_each_train_scene_each_epoch_once():
    conditions = np.repeat(np.arange(4), 4)
    plan = build_repeated_h1_plan(conditions, updates=8, h1_per_update=4, seed=82)
    assert plan.shape == (8, 4)
    epochs = plan.reshape(2, 4, 4)
    for epoch in epochs:
        indices = epoch.reshape(-1)
        np.testing.assert_array_equal(np.sort(indices), np.arange(16))


def test_independent_h0_h1_batch_uses_truth_only_from_h1_indices():
    iq_h0 = np.full((4, 3, 130, 32), 2 + 3j, dtype=np.complex64)
    iq_h1 = np.full((4, 3, 130, 32), 5 + 7j, dtype=np.complex64)
    arrays = {
        "iq_h0": iq_h0,
        "iq_h1": iq_h1,
        "truth_range_cell": np.asarray([1, 2, 3, 4], dtype=np.float32),
        "truth_doppler_hz": np.asarray([10, 20, 30, 40], dtype=np.float32),
    }
    batch = assemble_independent_h0_h1_batch(arrays, np.asarray([0, 1]), np.asarray([3, 2]))
    np.testing.assert_array_equal(batch["truth_presence"], [False, True, False, True])
    np.testing.assert_array_equal(batch["truth_range_cell"], [0, 4, 0, 3])
    np.testing.assert_array_equal(batch["truth_doppler_hz"], [0, 40, 0, 30])
    assert batch["iq_ri"].shape == (4, 6, 130, 32)
    assert np.all(batch["iq_ri"][0, :3] == 2)
    assert np.all(batch["iq_ri"][1, :3] == 5)
