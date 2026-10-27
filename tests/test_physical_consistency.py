from __future__ import annotations

import numpy as np
import pytest
import torch

from radardistill.evaluation.physical_consistency import (
    select_scene_ids_by_stratum,
    summarize_teacher_student_error_matrix,
    summarize_prediction_intervention,
    wrap_periodic_difference,
)
from radardistill.students.physical_coherence import FixedCoherenceFeatures
from radardistill.students.physics_guided_student import PhysicsGuidedTaskStudent


def test_stratified_scene_selection_is_invariant_to_row_permutation() -> None:
    scene_ids = np.asarray([f"scene-{index:03d}" for index in range(60)])
    strata = np.repeat(np.asarray(["C0", "C1", "C2"]), 20)
    quotas = {"C0": 7, "C1": 5, "C2": 9}

    selected = select_scene_ids_by_stratum(scene_ids, strata, quotas, seed=101)
    permutation = np.random.default_rng(13).permutation(len(scene_ids))
    permuted = select_scene_ids_by_stratum(
        scene_ids[permutation], strata[permutation], quotas, seed=101
    )

    assert selected.keys() == permuted.keys()
    for stratum in quotas:
        assert selected[stratum].tolist() == permuted[stratum].tolist()
        assert len(selected[stratum]) == quotas[stratum]


def test_stratified_scene_selection_rejects_duplicate_ids_and_missing_quota() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_scene_ids_by_stratum(
            np.asarray(["same", "same"]), np.asarray(["C0", "C0"]), {"C0": 1}, seed=1
        )
    with pytest.raises(ValueError, match="quota"):
        select_scene_ids_by_stratum(
            np.asarray(["a", "b"]), np.asarray(["C0", "C0"]), {"C0": 3}, seed=1
        )


def test_stratified_scene_selection_rejects_unregistered_strata() -> None:
    with pytest.raises(ValueError, match="strata"):
        select_scene_ids_by_stratum(
            np.asarray(["a", "b"]), np.asarray(["C0", "C1"]), {"C0": 1}, seed=1
        )


def test_periodic_difference_wraps_to_half_open_interval() -> None:
    result = wrap_periodic_difference(np.asarray([650.0, -650.0, 1301.0]), period=1300.0)
    np.testing.assert_allclose(result, np.asarray([-650.0, -650.0, 1.0]))


def test_prediction_intervention_reports_h0_h1_and_condition_transitions() -> None:
    baseline = {
        "score": np.asarray([0.1, 0.1, 0.9, 0.1]),
        "range_cell": np.asarray([0.0, 0.0, 10.2, 4.0]),
        "doppler_hz": np.asarray([0.0, 0.0, 100.0, -50.0]),
    }
    intervention = {
        "score": np.asarray([0.8, 0.1, 0.9, 0.9]),
        "range_cell": np.asarray([0.0, 0.0, 10.2, 4.0]),
        "doppler_hz": np.asarray([0.0, 0.0, 130.0, -50.0]),
    }
    summary = summarize_prediction_intervention(
        baseline,
        intervention,
        truth_presence=np.asarray([False, False, True, True]),
        condition=np.asarray(["C0", "C1", "C0", "C1"]),
        threshold=0.5,
        truth_range_cell=np.asarray([0.0, 0.0, 10.0, 4.0]),
        truth_doppler_hz=np.asarray([0.0, 0.0, 100.0, -50.0]),
        prf_hz=1300.0,
    )

    assert summary["H0"]["n"] == 2
    assert summary["H0"]["baseline_h0_false_alarm_rate"] == 0.0
    assert summary["H0"]["intervention_h0_false_alarm_rate"] == 0.5
    assert summary["H1"]["n"] == 2
    assert summary["H1"]["baseline_joint_success_rate"] == 0.5
    assert summary["H1"]["intervention_joint_success_rate"] == 0.5
    assert summary["condition:C1/H1"]["baseline_detected_count"] == 0
    assert summary["condition:C1/H1"]["intervention_detected_count"] == 1


def test_doppler_shift_intervention_uses_shifted_truth_and_periodic_response() -> None:
    scene_ids = np.asarray(
        ["scene-a", "scene-a", "scene-b", "scene-b", "scene-c", "scene-c", "scene-d", "scene-d"]
    )
    truth_presence = np.asarray([False, True] * 4)
    baseline = {
        "score": np.asarray([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]),
        "range_cell": np.asarray([np.nan, 10.0] * 4),
        "doppler_hz": np.asarray([np.nan, 649.0, np.nan, -649.0, np.nan, 50.0, np.nan, 200.0]),
    }
    shifted = {
        "score": np.asarray([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.8, 0.9]),
        "range_cell": np.asarray([np.nan, 10.0] * 4),
        # Three exact +5 Hz responses, including both sides of the Nyquist edge;
        # the last scene has a +11 Hz response and therefore a +6 Hz residual.
        "doppler_hz": np.asarray([np.nan, -646.0, np.nan, -644.0, np.nan, 55.0, np.nan, 211.0]),
    }

    summary = summarize_prediction_intervention(
        baseline,
        shifted,
        scene_ids=scene_ids,
        expected_doppler_shift_hz=5.0,
        truth_presence=truth_presence,
        condition=np.asarray(["C3"] * 8),
        threshold=0.5,
        truth_range_cell=np.asarray([np.nan, 10.0] * 4),
        truth_doppler_hz=np.asarray([np.nan, 649.0, np.nan, -649.0, np.nan, 50.0, np.nan, 200.0]),
        prf_hz=1300.0,
        doppler_gate_hz=3.0,
        bootstrap_replicates=200,
        bootstrap_seed=20261028,
    )

    response = summary["H1"]["doppler_response"]
    assert summary["H1"]["baseline_parameter_localization_rate"] == 1.0
    assert summary["H1"]["intervention_parameter_localization_rate"] == 0.75
    assert response["n_base_scenes"] == 4
    assert response["mean_signed_residual_hz"] == pytest.approx(1.5)
    assert response["mean_absolute_residual_hz"] == pytest.approx(1.5)
    assert response["mean_signed_residual_ci95_hz"][0] <= 1.5
    assert response["mean_signed_residual_ci95_hz"][1] >= 1.5
    assert response["mean_absolute_residual_ci95_hz"][0] <= 1.5
    assert response["mean_absolute_residual_ci95_hz"][1] >= 1.5
    assert summary["H0"]["intervention_h0_false_alarm_rate"] == 0.25
    assert summary["doppler_intervention"]["expected_shift_hz"] == 5.0


def test_teacher_student_error_matrix_separates_task_errors_and_bootstraps_scenes() -> None:
    scene_ids = np.asarray([f"scene-{index}" for index in range(8)])
    truth_presence = np.asarray([False] * 4 + [True] * 4)
    conditions = np.asarray(["C2", "C2", "C3", "C3", "C2", "C2", "C3", "C3"])
    teacher = {
        "score": np.asarray([0.1, 0.8, 0.1, 0.9, 0.9, 0.1, 0.9, 0.1]),
        "range_cell": np.asarray([np.nan] * 4 + [10.2, 15.0, 12.2, 20.0]),
        "doppler_hz": np.asarray([np.nan] * 4 + [649.0, -649.0, 100.0, 400.0]),
    }
    student = {
        "score": np.asarray([0.1, 0.1, 0.8, 0.9, 0.9, 0.9, 0.1, 0.1]),
        "range_cell": np.asarray([np.nan] * 4 + [10.2, 11.2, 30.0, 13.2]),
        "doppler_hz": np.asarray([np.nan] * 4 + [649.0, 651.0, 100.0, -10.0]),
    }

    result = summarize_teacher_student_error_matrix(
        teacher,
        student,
        scene_ids=scene_ids,
        truth_presence=truth_presence,
        condition=conditions,
        teacher_threshold=0.5,
        student_threshold=0.5,
        truth_range_cell=np.asarray([np.nan] * 4 + [10.0, 11.0, 12.0, 13.0]),
        truth_doppler_hz=np.asarray([np.nan] * 4 + [649.0, -649.0, 10.0, -10.0]),
        prf_hz=1300.0,
        bootstrap_replicates=200,
        bootstrap_seed=17,
    )

    h0 = result["groups"]["H0"]["tasks"]["presence"]
    assert h0["counts"] == {
        "both_correct": 1,
        "teacher_only_correct": 1,
        "student_only_correct": 1,
        "both_wrong": 1,
    }
    assert h0["student_rescue_given_teacher_wrong"] == pytest.approx(0.5)
    assert h0["student_regression_given_teacher_correct"] == pytest.approx(0.5)

    h1_tasks = result["groups"]["H1"]["tasks"]
    assert h1_tasks["presence"]["counts"]["teacher_only_correct"] == 1
    assert h1_tasks["presence"]["counts"]["student_only_correct"] == 1
    assert h1_tasks["range"]["counts"] == {
        "both_correct": 1,
        "teacher_only_correct": 1,
        "student_only_correct": 2,
        "both_wrong": 0,
    }
    assert h1_tasks["doppler"]["counts"] == {
        "both_correct": 2,
        "teacher_only_correct": 0,
        "student_only_correct": 1,
        "both_wrong": 1,
    }
    assert h1_tasks["joint"]["counts"] == {
        "both_correct": 1,
        "teacher_only_correct": 0,
        "student_only_correct": 1,
        "both_wrong": 2,
    }
    assert h1_tasks["doppler"]["student_minus_teacher_accuracy"] == pytest.approx(0.25)
    assert h1_tasks["doppler"]["scene_bootstrap_95_ci"][0] <= 0.25
    assert h1_tasks["doppler"]["scene_bootstrap_95_ci"][1] >= 0.25
    assert result["bootstrap"]["unit"] == "base_scene_id"
    assert result["uncertainty_is_training_seed_variance"] is False
    assert result["groups"]["condition:C3/H1"]["n"] == 2


def test_teacher_student_error_matrix_bootstraps_paired_h0_h1_rows_by_base_scene() -> None:
    teacher = {
        "score": np.asarray([0.1, 0.9]),
        "range_cell": np.asarray([0.0, 1.0]),
        "doppler_hz": np.asarray([0.0, 20.0]),
    }
    student = {
        "score": np.asarray([0.9, 0.9]),
        "range_cell": np.asarray([0.0, 1.0]),
        "doppler_hz": np.asarray([0.0, 20.0]),
    }

    result = summarize_teacher_student_error_matrix(
        teacher,
        student,
        scene_ids=np.asarray(["paired-scene", "paired-scene"]),
        truth_presence=np.asarray([False, True]),
        condition=np.asarray(["C2", "C2"]),
        teacher_threshold=0.5,
        student_threshold=0.5,
        truth_range_cell=np.asarray([np.nan, 1.0]),
        truth_doppler_hz=np.asarray([np.nan, 20.0]),
        prf_hz=1300.0,
        bootstrap_replicates=100,
        bootstrap_seed=19,
    )

    overall = result["groups"]["all"]
    presence = overall["tasks"]["presence"]
    assert overall["n"] == 2
    assert overall["n_base_scenes"] == 1
    assert presence["n_base_scenes"] == 1
    assert presence["counts"]["both_correct"] == 1
    assert presence["counts"]["teacher_only_correct"] == 1
    assert presence["student_minus_teacher_accuracy"] == pytest.approx(-0.5)
    assert presence["scene_bootstrap_95_ci"] == pytest.approx([-0.5, -0.5])


def _tone(frequency_hz: float, phases: tuple[float, float, float] = (0.0, 0.0, 0.0), *, dtype: torch.dtype = torch.complex128) -> torch.Tensor:
    pulse = torch.arange(130, dtype=torch.float64)
    channels = [
        torch.exp(1j * (2.0 * torch.pi * frequency_hz * pulse / 1300.0 + phase))
        for phase in phases
    ]
    value = torch.stack(channels).to(dtype=dtype)
    return value[None, :, :, None].expand(1, 3, 130, 32).contiguous()


def test_fixed_features_map_lag_phase_and_aliases_as_registered() -> None:
    frontend = FixedCoherenceFeatures().double()
    first = frontend(_tone(87.5))
    alias_lag4 = frontend(_tone(87.5 + 1300.0 / 4.0))
    alias_lag16 = frontend(_tone(87.5 + 1300.0 / 16.0))

    for feature_channel, lag in enumerate((1, 4, 16)):
        observed = torch.complex(first[0, feature_channel, 0, 0], first[0, 12 + feature_channel, 0, 0])
        expected = torch.exp(
            torch.tensor(1j * 2.0 * torch.pi * 87.5 * lag / 1300.0, dtype=torch.complex128)
        )
        torch.testing.assert_close(observed, expected, atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(first[:, 1], alias_lag4[:, 1], atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(first[:, 2], alias_lag16[:, 2], atol=1e-10, rtol=1e-8)
    assert not torch.allclose(first[:, 0], alias_lag4[:, 0], atol=1e-4, rtol=1e-4)


def test_public_phase_invariance_and_cross_channel_relative_phase() -> None:
    frontend = FixedCoherenceFeatures().double()
    phases = (0.2, -0.7, 1.1)
    iq = _tone(127.0, phases)
    base = frontend(iq)
    for rotation in (0.13, 2.4, 5.9):
        actual = frontend(iq * torch.exp(torch.tensor(1j * rotation, dtype=torch.complex128)))
        torch.testing.assert_close(actual, base, atol=1e-10, rtol=1e-8)
    cross_01 = torch.complex(base[0, 9, 0, 0], base[0, 21, 0, 0])
    torch.testing.assert_close(
        cross_01,
        torch.exp(torch.tensor(1j * (phases[0] - phases[1]), dtype=torch.complex128)),
        atol=1e-10,
        rtol=1e-8,
    )


def test_features_preserve_range_locality_and_scale_semantics() -> None:
    frontend = FixedCoherenceFeatures().double()
    iq = torch.zeros((1, 3, 130, 32), dtype=torch.complex128)
    tone = _tone(73.0)[0, :, :, 0]
    iq[0, :, :, 11] = tone
    base = frontend(iq)
    assert torch.count_nonzero(base[:, :27, :, :11]) == 0
    assert torch.count_nonzero(base[:, :27, :, 12:]) == 0

    scale = 2.5 * torch.exp(torch.tensor(0.8j, dtype=torch.complex128))
    scaled = frontend(iq * scale)
    torch.testing.assert_close(scaled[:, :27], base[:, :27], atol=1e-10, rtol=1e-8)
    torch.testing.assert_close(
        scaled[:, 27],
        base[:, 27] + 2.0 * torch.log(torch.tensor(2.5, dtype=torch.float64)),
        atol=1e-10,
        rtol=1e-8,
    )


def test_feature_frontend_is_finite_for_zero_tiny_and_large_inputs() -> None:
    frontend = FixedCoherenceFeatures()
    for amplitude in (0.0, 1e-25, 1e30):
        iq = _tone(41.0, dtype=torch.complex64) * amplitude
        features = frontend(iq)
        assert features.shape == (1, 28, 130, 32)
        assert torch.isfinite(features).all()
    zero = frontend(torch.zeros((1, 3, 130, 32), dtype=torch.complex64))
    assert torch.count_nonzero(zero[:, :27]) == 0
    torch.testing.assert_close(
        zero[:, 27],
        torch.full_like(zero[:, 27], float(np.log(1e-12))),
        atol=1e-6,
        rtol=1e-7,
    )


def test_complex64_features_match_complex128_reference_with_relative_error() -> None:
    rng = np.random.default_rng(20261026)
    real = rng.standard_normal((2, 3, 130, 32))
    imag = rng.standard_normal((2, 3, 130, 32))
    reference = FixedCoherenceFeatures()(torch.from_numpy(real + 1j * imag).to(torch.complex128))
    candidate = FixedCoherenceFeatures()(torch.from_numpy((real + 1j * imag).astype(np.complex64)))
    relative_max_error = (candidate.double() - reference).abs().max() / max(1.0, reference.abs().max().item())
    assert relative_max_error.item() <= 1e-4


def test_physics_guided_student_adds_only_small_projection_and_backpropagates() -> None:
    torch.manual_seed(101)
    raw = PhysicsGuidedTaskStudent()
    torch.manual_seed(101)
    baseline = PhysicsGuidedTaskStudent.make_raw_student()
    assert raw.shared_parameter_count == sum(parameter.numel() for parameter in baseline.parameters())
    assert raw.projection_parameter_count == 174
    assert raw.total_parameter_count <= raw.shared_parameter_count * 1.05

    iq_ri = torch.randn(2, 6, 130, 32)
    outputs = raw(iq_ri, compute_spectrum=False)
    loss = outputs["presence_logit"].mean() + outputs["range_logits"].mean() + outputs["doppler_vector"].mean()
    loss.backward()
    assert raw.projection.weight.grad is not None
    assert torch.isfinite(raw.projection.weight.grad).all()
    assert raw.projection.weight.grad.abs().sum() > 0


def test_physics_guided_task_outputs_are_public_phase_invariant() -> None:
    torch.manual_seed(808)
    model = PhysicsGuidedTaskStudent().eval()
    real = torch.randn(2, 3, 130, 32)
    imag = torch.randn(2, 3, 130, 32)
    iq_ri = torch.cat((real, imag), dim=1)
    phase = 1.73
    cosine, sine = np.cos(phase), np.sin(phase)
    rotated_real = real * cosine - imag * sine
    rotated_imag = real * sine + imag * cosine
    rotated_ri = torch.cat((rotated_real, rotated_imag), dim=1)

    with torch.inference_mode():
        baseline = model(iq_ri, compute_spectrum=False)
        rotated = model(rotated_ri, compute_spectrum=False)
    for name in ("presence_logit", "range_logits", "range_offset", "doppler_vector"):
        torch.testing.assert_close(rotated[name], baseline[name], atol=2e-5, rtol=1e-4)

