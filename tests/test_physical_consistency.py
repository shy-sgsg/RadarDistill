from __future__ import annotations

import numpy as np
import pytest
import torch

from radardistill.evaluation.physical_consistency import (
    select_scene_ids_by_stratum,
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

