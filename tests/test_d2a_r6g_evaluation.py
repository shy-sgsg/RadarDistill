from __future__ import annotations

import csv
import json
import numpy as np
import pytest
import torch

from radardistill.evaluation.d2a_r6g import (
    capture_validation_checkpoint,
    evaluate_r6g,
    ensure_initial_validation_checkpoint,
    paired_r6g_contrast,
    predict_r6g_eval_split,
    summarize_validation_history,
    teacher_correctness_mask,
    teacher_correctness_strata,
)


def _conditions() -> list[dict[str, object]]:
    return [
        {"course": "C1", "scenario_id": "separated", "snr_db": 0.0, "cnr_db": 20.0},
        {"course": "C2", "scenario_id": "separated", "snr_db": 0.0, "cnr_db": 20.0},
        {"course": "C3", "scenario_id": "partial_overlap", "snr_db": 0.0, "cnr_db": 20.0},
    ]


def _predictions(prefix: str, score: list[float], doppler: list[float]) -> dict[str, np.ndarray]:
    count = len(score)
    return {
        "scene_id": np.asarray([f"{prefix}-{index}" for index in range(count)]),
        "condition_index": np.asarray([0, 1, 1, 2, 2, 2][:count], dtype=np.uint8),
        "score": np.asarray(score, dtype=np.float32),
        "predicted_range_cell": np.asarray([4.0, 5.0, 5.0, 6.0, 6.0, 6.0][:count]),
        "truth_range_cell": np.asarray([4.0, 5.0, 5.0, 6.0, 6.0, 6.0][:count]),
        "predicted_doppler_hz": np.asarray(doppler, dtype=np.float32),
        "truth_doppler_hz": np.zeros(count, dtype=np.float32),
    }


def test_teacher_correctness_uses_range_tolerance_and_wrapped_doppler():
    actual = teacher_correctness_mask(
        predicted_range_bin=np.asarray([5, 6, 6, 6]),
        predicted_doppler_hz=np.asarray([649.0, -649.0, 20.01, 20.1]),
        truth_range_cell=np.asarray([5.5, 5.0, 5.0, 5.0]),
        truth_doppler_hz=np.asarray([-649.0, 649.0, 0.0, 0.0]),
        prf_hz=1300.0,
    )

    assert actual.tolist() == [True, True, False, False]


def test_paired_contrast_requires_same_scene_order_and_reports_frozen_primary_scope():
    baseline = _predictions("scene", [0.1, 0.9, 0.1, 0.1, 0.9, 0.1], [0, 0, 0, 0, 0, 0])
    candidate = _predictions("scene", [0.1, 0.9, 0.9, 0.9, 0.9, 0.9], [0, 0, 0, 0, 0, 0])
    baseline["scene_id"] = np.asarray([f"pair-{index}" for index in range(6)])
    candidate["scene_id"] = baseline["scene_id"].copy()

    result = paired_r6g_contrast(
        baseline,
        candidate,
        threshold_baseline=0.5,
        threshold_candidate=0.5,
        conditions=_conditions(),
        replicates=500,
        seed=2026110401,
        prf_hz=1300.0,
    )

    assert result["primary_scope"] == "identifiable_C2_C3"
    assert result["n_primary_scenes"] == 5
    assert result["joint_success_difference"] == pytest.approx(3 / 5)
    assert result["joint_success_ci95"][0] <= result["joint_success_difference"]
    assert result["joint_success_ci95"][1] >= result["joint_success_difference"]

    candidate["scene_id"] = np.asarray([f"wrong-{index}" for index in range(6)])
    with pytest.raises(ValueError, match="paired scene IDs"):
        paired_r6g_contrast(
            baseline,
            candidate,
            threshold_baseline=0.5,
            threshold_candidate=0.5,
            conditions=_conditions(),
            replicates=10,
            seed=1,
            prf_hz=1300.0,
        )


def test_teacher_correctness_strata_separates_hard_c3_failures():
    predictions = _predictions("pair", [0.9] * 6, [0.0] * 6)
    predictions["condition_index"] = np.asarray([0, 1, 1, 2, 2, 2], dtype=np.uint8)
    predictions["teacher_correct"] = np.asarray([True, True, False, True, False, False])

    result = teacher_correctness_strata(
        predictions,
        threshold=0.5,
        conditions=_conditions(),
        prf_hz=1300.0,
    )

    assert result["k3_correct_rate"]["all_H1"] == pytest.approx(3 / 6)
    assert result["k3_correct_rate"]["C3_partial_overlap_snr0_cnr20"] == pytest.approx(1 / 3)
    assert result["k3_correct_rate_by_condition"]["2"] == pytest.approx(1 / 3)
    assert result["student_by_teacher_correctness"]["teacher_incorrect"][
        "C3_partial_overlap_snr0_cnr20"
    ]["n"] == 2
    assert result["student_by_teacher_correctness"]["teacher_incorrect"][
        "C3_partial_overlap_snr0_cnr20"
    ]["pd"] == pytest.approx(1.0)


def test_validation_history_requires_fixed_curve_points_and_reports_normalized_area(tmp_path):
    path = tmp_path / "training_history.csv"
    rows = [
        {"update": 0, "validation_total": 8.0},
        {"update": 12800, "validation_total": 6.0},
        {"update": 25600, "validation_total": 4.0},
        {"update": 51200, "validation_total": 2.0},
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["update", "validation_total"])
        writer.writeheader()
        writer.writerows(rows)

    result = summarize_validation_history(path)

    assert result["validation_truth_loss_points"]["25600"]["validation_total"] == 4.0
    assert result["normalized_validation_truth_loss_area"] == pytest.approx(4.5)


def test_curve_checkpoint_capture_copies_only_registered_updates(tmp_path):
    run_dir = tmp_path / "T1"
    run_dir.mkdir()
    source = run_dir / "last_checkpoint.pt"
    owner = {"owner": "D2-A-R6-G-training-run", "arm": "T1"}
    torch.save({"owner": owner, "update": 400, "model_state_dict": {}}, source)

    assert capture_validation_checkpoint(source, arm="T1", target_updates=(0, 12800)) is None
    torch.save({"owner": owner, "update": 12800, "model_state_dict": {"w": torch.ones(1)}}, source)
    assert capture_validation_checkpoint(source, arm="T1", target_updates=(0, 12800)) == 12800
    captured = torch.load(run_dir / "validation_checkpoints/update_12800.pt", map_location="cpu", weights_only=False)
    assert captured["update"] == 12800
    assert torch.equal(captured["model_state_dict"]["w"], torch.ones(1))


def test_capture_audit_recognizes_preexisting_registered_snapshots(tmp_path):
    from scripts.run_d2a_r6g_evaluation import existing_validation_updates

    run_dir = tmp_path / "T0"
    snapshot_dir = run_dir / "validation_checkpoints"
    snapshot_dir.mkdir(parents=True)
    owner = {"owner": "D2-A-R6-G-training-run", "arm": "T0"}
    for update in (0, 12800, 25600):
        torch.save(
            {"owner": owner, "update": update, "model_state_dict": {"w": torch.ones(1)}},
            snapshot_dir / f"update_{update:05d}.pt",
        )

    assert existing_validation_updates("T0", run_dir) == {0, 12800, 25600}


def test_initial_curve_checkpoint_comes_from_shared_preupdate_state(tmp_path, monkeypatch):
    from radardistill.evaluation import d2a_r6g

    monkeypatch.setattr(d2a_r6g, "RUNS", tmp_path)
    run_dir = tmp_path / "T0"
    run_dir.mkdir()
    owner = {
        "owner": "D2-A-R6-G-training-run",
        "git_sha": "abc123",
        "arm": "T0",
        "config_sha256": "cfg",
        "initialization_seed": 707,
    }
    (run_dir / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
    torch.save(
        {
            "owner": {"git_sha": "abc123", "config_sha256": "cfg", "seed": 707},
            "state_dict": {"w": torch.tensor([1.0, 2.0])},
        },
        tmp_path / "shared_initial_state.pt",
    )

    assert ensure_initial_validation_checkpoint("T0")
    path = run_dir / "validation_checkpoints/update_00000.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["owner"] == owner
    assert saved["update"] == 0
    assert torch.equal(saved["model_state_dict"]["w"], torch.tensor([1.0, 2.0]))


def test_formal_eval_rejects_cpu_instead_of_silent_fallback():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        predict_r6g_eval_split(
            torch.nn.Identity(),
            "h0_calibration",
            [],
            {},
            device=torch.device("cpu"),
            microbatch=4,
            prf_hz=1300.0,
        )


def test_evaluate_r6g_rejects_incomplete_arm_before_predicting_any_split(tmp_path, monkeypatch):
    from radardistill.evaluation import d2a_r6g as evaluation
    from radardistill.training import d2a_r6g as training

    run_root = tmp_path / "runs"
    stage = tmp_path / "stage"
    config = {
        "training": {"fixed_updates_per_arm": 51200},
        "resources": {"required_gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU"},
        "student": {"width": 64},
        "evaluation": {"target_pfa_cpi": 0.05},
        "radar": {"prf_hz": 1300.0},
    }
    r6a = {"radar": {"conditions": []}}
    monkeypatch.setattr(evaluation, "RUNS", run_root)
    monkeypatch.setattr(evaluation, "STAGE", stage)
    monkeypatch.setattr(evaluation, "ROOT", tmp_path)
    monkeypatch.setattr(training, "_require_g0", lambda: ({"git_sha": "g0"}, "g0"))
    monkeypatch.setattr(training, "_require_matching_g1", lambda _g0: {"formal_microbatch": 4})
    monkeypatch.setattr(training, "_configs", lambda: (config, r6a, {}))
    monkeypatch.setattr(
        training,
        "build_r6g_plans",
        lambda *_args: {name: [] for name in ("h0_calibration", "h0_verification", "h1_development")},
    )
    monkeypatch.setattr(evaluation.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        evaluation.torch.cuda,
        "get_device_name",
        lambda _index: "NVIDIA GeForce RTX 3050 Laptop GPU",
    )

    class FakeModel:
        def to(self, _device):
            return self

        def load_state_dict(self, _state_dict, strict=True):
            assert strict

        def eval(self):
            return self

    monkeypatch.setattr(evaluation, "R6ECEStudent", lambda **_kwargs: FakeModel())

    def fail_if_prediction_starts(*_args, **_kwargs):
        pytest.fail("R6-G evaluation predicted a split before every arm was complete")

    monkeypatch.setattr(evaluation, "predict_r6g_eval_split", fail_if_prediction_starts)

    for arm in ("T0", "T1", "T3"):
        run_dir = run_root / arm
        run_dir.mkdir(parents=True)
        owner = {"owner": "D2-A-R6-G-training-run", "arm": arm}
        record = {
            "status": "complete" if arm != "T3" else "running",
            "arm": arm,
            "updates": 51200 if arm != "T3" else 12800,
            "best_update": 400,
            "owner": owner,
            "inference_checkpoint": f"artifacts/{arm}.pt",
        }
        (run_dir / "training_record.json").write_text(json.dumps(record), encoding="utf-8")
        if arm != "T3":
            checkpoint = tmp_path / record["inference_checkpoint"]
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"owner": owner, "update": 400, "model_state_dict": {}}, checkpoint)

    with pytest.raises(RuntimeError, match="T3.*complete"):
        evaluate_r6g()
