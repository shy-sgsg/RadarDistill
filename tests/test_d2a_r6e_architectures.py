from __future__ import annotations

import math

import pytest
import torch

from radardistill.students.r6_architectures import (
    R6ECEStudent,
    R6ERAXTransformer,
    count_r6e_parameters,
)


def _rotate_common_phase(iq_ri: torch.Tensor, phase_rad: float) -> torch.Tensor:
    real, imag = iq_ri[:, :3], iq_ri[:, 3:]
    cosine, sine = math.cos(phase_rad), math.sin(phase_rad)
    return torch.cat((real * cosine - imag * sine, real * sine + imag * cosine), dim=1)


@pytest.mark.parametrize("model_type", [R6ECEStudent, R6ERAXTransformer])
def test_r6e_models_share_task_output_contract_and_finite_gradients(model_type):
    model = model_type()
    iq = torch.randn(1, 6, 130, 32)
    output = model(iq)

    assert output["presence_logit"].shape == (1,)
    assert output["range_logits"].shape == (1, 32)
    assert output["range_offset"].shape == (1, 32)
    assert output["doppler_vector"].shape == (1, 32, 2)
    assert all(torch.isfinite(value).all() for value in output.values())
    torch.testing.assert_close(
        output["doppler_vector"].norm(dim=-1), torch.ones(1, 32), rtol=1e-5, atol=1e-5
    )

    loss = sum(value.square().mean() for value in output.values())
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_complex_tcn_is_common_phase_equivariant_and_task_invariant():
    model = R6ECEStudent().eval()
    iq = torch.randn(1, 6, 130, 32)
    phase = 1.137

    with torch.inference_mode():
        features = model.forward_features(iq)
        rotated_features = model.forward_features(_rotate_common_phase(iq, phase))
        original_output = model(iq)
        rotated_output = model(_rotate_common_phase(iq, phase))

    expected = features * torch.polar(
        torch.ones((), dtype=features.real.dtype),
        torch.tensor(phase, dtype=features.real.dtype),
    )
    torch.testing.assert_close(rotated_features, expected, rtol=2e-4, atol=2e-5)
    for key in ("presence_logit", "range_logits", "range_offset", "doppler_vector"):
        torch.testing.assert_close(rotated_output[key], original_output[key], rtol=2e-4, atol=2e-5)


def test_rax_retains_full_time_range_grid_until_axial_blocks_finish():
    model = R6ERAXTransformer().eval()
    with torch.inference_mode():
        grid = model.encode_grid(torch.randn(1, 6, 130, 32))
    assert grid.shape == (1, 130, 32, 48)
    assert len(model.blocks) == 2
    assert model.blocks[0].time_attention.relative_bias.shape == (4, 259)
    assert model.blocks[0].range_attention.relative_bias.shape == (4, 63)


@pytest.mark.parametrize("model_type", [R6ECEStudent, R6ERAXTransformer])
def test_r6e_candidate_parameter_counts_are_matched_to_j1(model_type):
    model = model_type()
    counts = count_r6e_parameters(model)
    assert counts["parameters"] == counts["trainable_parameters"]
    assert 56_222 <= counts["parameters"] <= 84_332


@pytest.mark.parametrize("model_type", [R6ECEStudent, R6ERAXTransformer])
def test_r6e_models_reject_incompatible_iq_layout(model_type):
    model = model_type()
    with pytest.raises(ValueError, match=r"\[B,6,130,32\]"):
        model(torch.randn(2, 6, 128, 32))
