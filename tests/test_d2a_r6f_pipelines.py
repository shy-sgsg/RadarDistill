import numpy as np
import pytest
import torch

from radardistill.evaluation.d2a_r6f import (
    audit_r6f_scene_ids,
    build_ce_system_predictions,
    build_r6f_split_plans,
    check_batch_consistency,
    candidate_features_at_bins,
    compose_system_scores,
    decode_model_at_bins,
    decode_model_grid,
    decision_outputs,
    normalized_fit_fraction,
    paired_r6f_contrast,
    summarize_candidate_agreement,
    summarize_candidate_correctness,
    run_cost_system_batch,
)
from radardistill.evaluation import d2a_r6f


def _grid_outputs() -> dict[str, torch.Tensor]:
    presence = torch.tensor([0.0, np.log(3.0)], dtype=torch.float32)
    range_logits = torch.zeros((2, 32), dtype=torch.float32)
    range_logits[0, 3] = 2.0
    range_logits[1, 7] = 2.0
    offsets = torch.zeros((2, 32), dtype=torch.float32)
    offsets[0, 3] = 0.25
    offsets[0, 2] = 0.50
    offsets[1, 7] = 0.75
    offsets[1, 4] = 0.10
    vectors = torch.zeros((2, 32, 2), dtype=torch.float32)
    vectors[0, 3] = torch.tensor([0.0, 1.0])
    vectors[0, 2] = torch.tensor([1.0, 0.0])
    vectors[1, 7] = torch.tensor([0.0, -1.0])
    vectors[1, 4] = torch.tensor([-1.0, 0.0])
    return {
        "presence_logit": presence,
        "range_logits": range_logits,
        "range_offset": offsets,
        "doppler_vector": vectors,
    }


def test_decode_model_grid_uses_argmax_range_and_circular_doppler():
    decoded = decode_model_grid(_grid_outputs(), prf_hz=1300.0)

    np.testing.assert_allclose(decoded["presence_probability"], [0.5, 0.75], atol=1e-7)
    np.testing.assert_array_equal(decoded["candidate_range_bin"], [3, 7])
    np.testing.assert_allclose(decoded["range_cell"], [3.25, 7.75], atol=1e-7)
    np.testing.assert_allclose(decoded["doppler_hz"], [325.0, -325.0], atol=1e-4)


def test_decode_model_at_bins_propagates_external_candidate_gate():
    decoded = decode_model_at_bins(_grid_outputs(), np.array([2, 4]), prf_hz=1300.0)

    np.testing.assert_array_equal(decoded["candidate_range_bin"], [2, 4])
    np.testing.assert_allclose(decoded["range_cell"], [2.5, 4.1], atol=1e-7)
    np.testing.assert_allclose(decoded["doppler_hz"], [0.0, -650.0], atol=1e-4)


def test_normalized_fit_fraction_is_bounded_and_scale_invariant():
    n = 130
    prf_hz = 1300.0
    pulses = np.arange(n)
    tone = np.exp(2j * np.pi * 100.0 * pulses / prf_hz)
    fit = normalized_fit_fraction(tone[None, :], prf_hz=prf_hz)
    scaled_fit = normalized_fit_fraction(((-2.0 + 3.0j) * tone)[None, :], prf_hz=prf_hz)

    assert fit.shape == (1,)
    assert 0.99 < fit[0] <= 1.0
    np.testing.assert_allclose(fit, scaled_fit, rtol=1e-12, atol=1e-12)


def test_composite_scores_keep_each_complete_chain_score_explicit():
    scores = compose_system_scores(
        b3_score=np.array([10.0, 20.0]),
        ce_probability=np.array([0.5, 0.8]),
        j1_probability=np.array([0.2, 0.7]),
        fit_fraction=np.array([0.4, 0.25]),
    )

    np.testing.assert_array_equal(scores["A"], [10.0, 20.0])
    np.testing.assert_array_equal(scores["B"], [0.5, 0.8])
    np.testing.assert_array_equal(scores["C"], [0.5, 0.8])
    np.testing.assert_array_equal(scores["D"], [10.0, 20.0])
    np.testing.assert_array_equal(scores["E"], [0.2, 0.7])
    np.testing.assert_allclose(scores["F"], [0.2, 0.2], atol=1e-7)


def test_calibration_target_is_read_from_the_registered_system_section():
    config, _, _ = d2a_r6f._configs()

    assert d2a_r6f.calibration_target_pfa(config) == 0.05


def test_cached_prediction_loader_reuses_only_complete_disjoint_splits(tmp_path, monkeypatch):
    monkeypatch.setattr(d2a_r6f, "PREDICTIONS", tmp_path)
    conditions = [{"course": "C0"}]
    plans = {
        split: [{"condition_index": 0, "count": 2}]
        for split in ("h0_calibration", "h0_verification", "h1_development")
    }
    for split, prefix in (("h0_calibration", "cal"), ("h0_verification", "ver"), ("h1_development", "dev")):
        arrays = {
            "scene_id": np.array([f"{prefix}-0", f"{prefix}-1"]),
            "condition_index": np.zeros(2, dtype=np.uint8),
            **{f"{name}__score": np.array([0.1, 0.2]) for name in d2a_r6f.SYSTEM_NAMES},
        }
        if split == "h1_development":
            arrays.update({
                "truth_range_cell": np.array([2.25, 3.25]),
                "truth_doppler_hz": np.array([100.0, 200.0]),
                "oracle_fft4_doppler_hz": np.array([100.0, 200.0]),
                "oracle_profile_ml_doppler_hz": np.array([100.0, 200.0]),
                "oracle_ce_range_cell": np.array([2.25, 3.25]),
                "oracle_ce_doppler_hz": np.array([100.0, 200.0]),
                "oracle_j1_range_cell": np.array([2.25, 3.25]),
                "oracle_j1_doppler_hz": np.array([100.0, 200.0]),
            })
            for name in d2a_r6f.SYSTEM_NAMES:
                arrays[f"{name}__candidate_range_bin"] = np.array([2, 3])
                arrays[f"{name}__predicted_range_cell"] = np.array([2.25, 3.25])
                arrays[f"{name}__predicted_doppler_hz"] = np.array([100.0, 200.0])
            arrays["D__detection_candidate_range_bin"] = np.array([2, 3])
        d2a_r6f._write_npz(tmp_path / f"{split}.npz", arrays)

    loaded, counts = d2a_r6f._load_finalized_prediction_cache(plans, conditions)

    assert counts == {"h0_calibration": 2, "h0_verification": 2, "h1_development": 2}
    assert loaded["h1_development"]["scene_id"].tolist() == ["dev-0", "dev-1"]


def test_detection_decision_abstains_on_below_threshold_cpis():
    decision = decision_outputs(
        np.array([0.49, 0.50, 0.90]),
        np.array([1.25, 2.25, 3.25]),
        np.array([10.0, 20.0, 30.0]),
        threshold=0.50,
    )

    np.testing.assert_array_equal(decision["detected"], [False, True, True])
    assert np.isnan(decision["output_range_cell"][0])
    assert np.isnan(decision["output_doppler_hz"][0])
    np.testing.assert_allclose(decision["output_range_cell"][1:], [2.25, 3.25])
    np.testing.assert_allclose(decision["output_doppler_hz"][1:], [20.0, 30.0])


def test_candidate_estimators_use_the_requested_dpca_range_gate():
    prf_hz = 1300.0
    pulses = np.arange(130)
    iq = np.zeros((2, 3, 130, 32), dtype=np.complex128)
    iq[0, 0, :, 5] = np.exp(2j * np.pi * 100.0 * pulses / prf_hz)
    iq[1, 0, :, 9] = np.exp(2j * np.pi * 200.0 * pulses / prf_hz)
    physics = {
        "radar": {"prf_hz": prf_hz},
        "channels": {"positions_xyz_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        "platform": {"velocity_xyz_mps": [1, 0, 0]},
    }

    estimates = candidate_features_at_bins(
        iq, np.array([5, 9]), physics, prf_hz=prf_hz, include_profile_ml=True
    )

    np.testing.assert_allclose(estimates["fft4_doppler_hz"], [100.0, 200.0], atol=1e-9)
    np.testing.assert_allclose(estimates["profile_ml_doppler_hz"], [100.0, 200.0], atol=0.05)
    assert np.all(estimates["fit_fraction"] > 0.99)


def test_ce_systems_keep_classical_and_neural_candidate_gates_separate():
    prf_hz = 1300.0
    pulses = np.arange(130)
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex128)
    iq[0, 0, :, 5] = np.exp(2j * np.pi * 100.0 * pulses / prf_hz)
    iq[0, 0, :, 7] = 0.8 * np.exp(2j * np.pi * 200.0 * pulses / prf_hz)
    physics = {
        "radar": {"prf_hz": prf_hz},
        "channels": {"positions_xyz_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        "platform": {"velocity_xyz_mps": [1, 0, 0]},
    }
    outputs = _grid_outputs()
    outputs = {key: value[:1].clone() for key, value in outputs.items()}
    outputs["range_logits"].fill_(-3.0)
    outputs["range_logits"][0, 7] = 3.0
    outputs["range_offset"][0, 5] = 0.2
    outputs["range_offset"][0, 7] = 0.75
    for gate, frequency in ((5, -100.0), (7, 200.0)):
        phase = 2.0 * np.pi * frequency / prf_hz
        outputs["doppler_vector"][0, gate] = torch.tensor([np.cos(phase), np.sin(phase)])

    result = build_ce_system_predictions(
        iq, outputs, physics, prf_hz=prf_hz, include_parameter_estimates=True
    )

    assert result["A"]["candidate_range_bin"].tolist() == [5]
    assert result["B"]["candidate_range_bin"].tolist() == [7]
    assert result["D"]["candidate_range_bin"].tolist() == [7]
    assert result["D"]["detection_candidate_range_bin"].tolist() == [5]
    assert result["C"]["candidate_range_bin"].tolist() == [7]
    assert result["F"]["candidate_range_bin"].tolist() == [7]
    np.testing.assert_allclose(result["A"]["predicted_doppler_hz"], [100.0], atol=1e-9)
    np.testing.assert_allclose(result["B"]["predicted_doppler_hz"], [200.0], atol=1e-4)
    np.testing.assert_allclose(result["C"]["predicted_doppler_hz"], [200.0], atol=0.05)
    np.testing.assert_allclose(result["D"]["predicted_range_cell"], [7.75], atol=1e-6)
    np.testing.assert_allclose(result["D"]["predicted_doppler_hz"], [200.0], atol=1e-4)
    np.testing.assert_allclose(result["F"]["score"], result["B"]["score"] * result["F"]["fit_fraction"])


def test_hybrid_candidate_agreement_is_reported_by_registered_group():
    conditions = [
        {"course": "C0"}, {"course": "C2"}, {"course": "C3"},
    ]
    summary = summarize_candidate_agreement(
        detection_bins=np.array([5, 7, 4]),
        parameter_bins=np.array([5, 8, 4]),
        condition_index=np.array([0, 1, 2]),
        conditions=conditions,
    )

    assert summary["overall"] == {"n": 3, "agree": 2, "agreement_rate": pytest.approx(2 / 3)}
    assert summary["by_group"]["C2_C3"] == {
        "n": 2, "agree": 1, "agreement_rate": pytest.approx(0.5)
    }
    assert summary["by_group"]["C3"]["agreement_rate"] == 1.0


def test_h0_ce_pipeline_returns_scores_without_parameter_labels_or_outputs():
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex128)
    physics = {
        "radar": {"prf_hz": 1300.0},
        "channels": {"positions_xyz_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        "platform": {"velocity_xyz_mps": [1, 0, 0]},
    }
    outputs = {key: value[:1].clone() for key, value in _grid_outputs().items()}

    result = build_ce_system_predictions(
        iq, outputs, physics, prf_hz=1300.0, include_parameter_estimates=False
    )

    assert set(result) == {"A", "B", "C", "D", "F"}
    assert all("predicted_range_cell" not in item for item in result.values())
    assert all(np.isfinite(item["score"]).all() for item in result.values())


def test_scene_audit_rejects_overlap_with_prior_registered_scenes():
    plans = {
        "h0_calibration": [{"seed": 100, "count": 2}],
        "h0_verification": [{"seed": 200, "count": 1}],
    }
    audit = audit_r6f_scene_ids(plans, g1_seed=50, g1_count=1, prior_scene_ids=set())
    assert audit["planned_scene_ids"] == 3
    assert audit["g1_scene_ids"] == 1
    assert audit["overlap"] == 0
    with pytest.raises(ValueError, match="prior scene ID overlap"):
        audit_r6f_scene_ids(
            plans,
            g1_seed=50,
            g1_count=1,
            prior_scene_ids={"d2a-scene-200-0"},
        )


def test_split_plans_keep_registered_condition_counts_and_seed_blocks():
    conditions = [{"course": f"C{i // 3}"} for i in range(12)]
    config = {
        "data": {
            "condition_seed_stride": 1000,
            "chunk_size": 2,
            "chunk_seed_stride": 20,
            "splits": {
                "h0_calibration": {"presence": "H0", "scene_count": 24, "count_per_condition": 2, "seed_base": 10000},
                "h0_verification": {"presence": "H0", "scene_count": 24, "count_per_condition": 2, "seed_base": 30000},
                "h1_development": {"presence": "H1", "scene_count": 24, "count_per_condition": 2, "seed_base": 50000},
            },
        }
    }

    plans = build_r6f_split_plans(config, conditions)

    assert {name: sum(row["count"] for row in rows) for name, rows in plans.items()} == {
        "h0_calibration": 24,
        "h0_verification": 24,
        "h1_development": 24,
    }
    assert plans["h0_calibration"][0]["seed"] == 10000
    assert plans["h0_calibration"][1]["seed"] == 11000
    assert plans["h0_verification"][0]["seed"] == 30000
    assert plans["h1_development"][0]["presence"] == "H1"


def test_batch_consistency_rejects_a_candidate_dependent_on_batch_size():
    large_batch = {"score": np.array([0.2, 0.8]), "range": np.array([3.0, 8.0])}
    same_single_calls = {"score": np.array([0.2, 0.8]), "range": np.array([3.0, 8.0])}
    shifted_single_calls = {"score": np.array([0.2, 0.801]), "range": np.array([3.0, 8.0])}

    assert check_batch_consistency(large_batch, same_single_calls, atol=1e-4)["max_abs_difference"] == 0.0
    with pytest.raises(ValueError, match="batch-size consistency"):
        check_batch_consistency(large_batch, shifted_single_calls, atol=1e-4)


def test_cost_chain_a_includes_b3_decision_and_candidate_fft_estimation():
    prf_hz = 1300.0
    pulses = np.arange(130)
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex128)
    iq[0, 0, :, 5] = np.exp(2j * np.pi * 100.0 * pulses / prf_hz)
    physics = {
        "radar": {"prf_hz": prf_hz},
        "channels": {"positions_xyz_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        "platform": {"velocity_xyz_mps": [1, 0, 0]},
    }

    output = run_cost_system_batch(
        "A", iq, physics, prf_hz=prf_hz, threshold=0.0, model=None, device=torch.device("cpu")
    )

    assert output["detected_count"] == 1
    assert output["estimator_invocations"] == 1
    assert output["decision"]["detected"].tolist() == [True]
    np.testing.assert_allclose(output["decision"]["output_doppler_hz"], [100.0], atol=1e-9)


def test_cost_chain_d_uses_b3_gate_for_detection_and_native_ce_estimator(monkeypatch):
    prf_hz = 1300.0
    pulses = np.arange(130)
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex128)
    iq[0, 0, :, 5] = np.exp(2j * np.pi * 100.0 * pulses / prf_hz)
    physics = {
        "radar": {"prf_hz": prf_hz},
        "channels": {"positions_xyz_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
        "platform": {"velocity_xyz_mps": [1, 0, 0]},
    }
    outputs = _grid_outputs()
    outputs = {key: value[1:2].clone() for key, value in outputs.items()}
    monkeypatch.setattr(d2a_r6f, "_model_outputs", lambda model, batch, device: outputs)

    result = run_cost_system_batch(
        "D", iq, physics, prf_hz=prf_hz, threshold=0.0,
        model=object(), device=torch.device("cpu"),
    )

    assert result["detected_count"] == 1
    assert result["estimator_invocations"] == 1
    assert result["detection_candidate_range_bin"].tolist() == [5]
    assert result["candidate_range_bin"].tolist() == [7]
    np.testing.assert_allclose(result["decision"]["output_range_cell"], [7.75], atol=1e-6)
    np.testing.assert_allclose(result["decision"]["output_doppler_hz"], [-325.0], atol=1e-4)


def test_paired_contrast_counts_misses_as_joint_failures():
    conditions = [
        {"course": "C0"}, {"course": "C1"}, {"course": "C1"},
        {"course": "C2"}, {"course": "C2"}, {"course": "C2"},
        {"course": "C3"}, {"course": "C3"}, {"course": "C3"},
        {"course": "C3"}, {"course": "C3"}, {"course": "C3"},
    ]
    condition_index = np.repeat(np.arange(12), 2)
    truth_range = np.full(len(condition_index), 10.25)
    truth_doppler = np.full(len(condition_index), 100.0)
    baseline = {
        "scene_id": np.array([f"scene-{i}" for i in range(len(condition_index))]),
        "condition_index": condition_index,
        "score": np.zeros(len(condition_index)),
        "predicted_range_cell": truth_range + 2.0,
        "predicted_doppler_hz": truth_doppler + 40.0,
        "truth_range_cell": truth_range,
        "truth_doppler_hz": truth_doppler,
    }
    candidate = {
        **baseline,
        "score": np.ones(len(condition_index)),
        "predicted_range_cell": truth_range.copy(),
        "predicted_doppler_hz": truth_doppler.copy(),
    }

    contrast = paired_r6f_contrast(
        baseline,
        candidate,
        threshold_baseline=0.5,
        threshold_candidate=0.5,
        conditions=conditions,
        replicates=100,
        seed=99,
    )

    assert contrast["joint_success_difference"] == 1.0
    assert contrast["joint_success_ci99"] == [1.0, 1.0]
    assert contrast["protection"]["pd_difference"] == 1.0
    assert contrast["candidate_screening_pass"] is True


def test_candidate_correctness_uses_nearest_truth_gate_and_reports_c3():
    conditions = [
        {"course": "C0"}, {"course": "C1"}, {"course": "C1"},
        {"course": "C2"}, {"course": "C2"}, {"course": "C2"},
        {"course": "C3"}, {"course": "C3"}, {"course": "C3"},
        {"course": "C3"}, {"course": "C3"}, {"course": "C3"},
    ]
    summary = summarize_candidate_correctness(
        np.array([2, 3, 19, 18]),
        np.array([2.2, 3.1, 18.8, 19.9]),
        np.array([0, 3, 10, 11]),
        conditions,
    )

    assert summary["overall"] == {"n": 4, "correct": 3, "accuracy": 0.75}
    assert summary["by_group"]["C2_C3"] == {"n": 3, "correct": 2, "accuracy": 2 / 3}
    assert summary["by_group"]["C3"] == {"n": 2, "correct": 1, "accuracy": 0.5}


@pytest.mark.parametrize("bad_bins", [np.array([-1]), np.array([32]), np.array([1.5])])
def test_decode_model_at_bins_rejects_invalid_candidate_gates(bad_bins):
    with pytest.raises(ValueError):
        decode_model_at_bins(_grid_outputs(), bad_bins, prf_hz=1300.0)
