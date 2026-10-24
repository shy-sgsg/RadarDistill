from __future__ import annotations

import inspect

import pytest
import torch

from radardistill.students.d1_e002 import (
    D1E002Student,
    lag_one_complex_coherence,
    wrap_frequency_hz,
)


@pytest.mark.parametrize("variant", ["S1", "S2", "S3"])
def test_candidate_outputs_are_finite_shaped_and_differentiable(variant: str) -> None:
    model = D1E002Student(variant=variant, width=32, temporal_dilations=(1, 2, 4, 8, 16, 32, 64))
    inputs = torch.randn(2, 6, 130, 32, requires_grad=True)
    outputs = model(inputs)

    assert outputs["presence_logit"].shape == (2,)
    assert outputs["range_logits"].shape == (2, 32)
    assert outputs["range_offset"].shape == (2, 32)
    assert outputs["doppler_vector"].shape == (2, 32, 2)
    assert torch.isfinite(outputs["presence_logit"]).all()
    assert torch.isfinite(outputs["range_logits"]).all()
    assert torch.isfinite(outputs["range_offset"]).all()
    assert torch.isfinite(outputs["doppler_vector"]).all()
    torch.linalg.vector_norm(outputs["doppler_vector"], dim=-1).mean().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_s2_lag_one_feature_tracks_phase_increment_and_ignores_common_phase() -> None:
    prf_hz = 1300.0
    frequency_hz = 137.25
    phase = torch.tensor([0.37], dtype=torch.float64)
    time = torch.arange(130, dtype=torch.float64)
    tone = torch.exp(1j * (2.0 * torch.pi * frequency_hz * time / prf_hz + phase[0]))
    iq = tone.view(1, 1, 130, 1).expand(1, 3, 130, 32).clone()
    rotated = iq * torch.exp(1j * torch.tensor(1.41, dtype=torch.float64))

    features = lag_one_complex_coherence(iq, epsilon=1.0e-8)
    rotated_features = lag_one_complex_coherence(rotated, epsilon=1.0e-8)
    assert features.shape == (1, 3, 32, 2)
    assert torch.allclose(features, rotated_features, atol=1e-10, rtol=1e-10)
    angle = torch.atan2(features[..., 1], features[..., 0])
    assert torch.allclose(angle, torch.full_like(angle, 2.0 * torch.pi * frequency_hz / prf_hz), atol=1e-10)


def test_wrapped_frequency_uses_half_open_prf_interval() -> None:
    actual = wrap_frequency_hz(torch.tensor([650.0, 655.0, -655.0, -650.0]), 1300.0)
    expected = torch.tensor([-650.0, -645.0, 645.0, -650.0])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_s3_is_explicitly_hybrid_and_forward_has_no_truth_inputs() -> None:
    model = D1E002Student(variant="S3")
    assert model.method_kind == "fft_assisted_hybrid"
    assert model.is_hybrid is True
    assert list(inspect.signature(model.forward).parameters) == ["iq_ri"]
    outputs = model(torch.randn(1, 6, 130, 32))
    assert outputs["doppler_coarse_hz"].shape == (1, 32)
    assert outputs["doppler_delta_bins"].shape == (1, 32)


def test_variant_name_rejects_unregistered_architectures() -> None:
    with pytest.raises(ValueError, match="S1, S2, or S3"):
        D1E002Student(variant="S4")
