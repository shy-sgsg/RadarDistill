from __future__ import annotations

import math

import pytest
import torch

from radardistill.students.r6_ssm import (
    R6ComplexCNNSSMStudent,
    R6ComplexDiagonalSSMBlock,
    R6ComplexSSMStudent,
    _complex_linear_conv_fft,
    count_r6h_parameters,
)


def _direct_causal_convolution(values: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    batch, channels, time, ranges = values.shape
    output = torch.zeros_like(values)
    for t in range(time):
        for lag in range(min(t + 1, kernel.shape[-1])):
            output[:, :, t, :] += values[:, :, t - lag, :] * kernel[:, lag].reshape(
                1, channels, 1
            )
    return output


def _rotate_common_phase(iq_ri: torch.Tensor, phase: float) -> torch.Tensor:
    cosine, sine = math.cos(phase), math.sin(phase)
    real, imag = iq_ri[:, :3], iq_ri[:, 3:]
    return torch.cat((real * cosine - imag * sine, real * sine + imag * cosine), dim=1)


def test_fft_linear_convolution_matches_direct_reference() -> None:
    generator = torch.Generator().manual_seed(6021)
    values = torch.randn(2, 3, 9, 2, generator=generator, dtype=torch.float64)
    values = torch.complex(values, torch.randn(2, 3, 9, 2, generator=generator, dtype=torch.float64))
    kernel = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    kernel = torch.complex(kernel, torch.randn(3, 4, generator=generator, dtype=torch.float64))

    actual = _complex_linear_conv_fft(values, kernel, time_dim=2)
    expected = _direct_causal_convolution(values, kernel)

    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_fft_linear_convolution_does_not_wrap_cpi_boundary() -> None:
    values = torch.zeros(1, 1, 8, 1, dtype=torch.complex64)
    values[0, 0, -1, 0] = 1.0 + 0.0j
    kernel = torch.tensor([[0.0 + 0.0j, 1.0 + 0.0j]], dtype=torch.complex64)

    actual = _complex_linear_conv_fft(values, kernel, time_dim=2)

    assert actual[0, 0, 0, 0].abs().item() < 1e-6
    assert actual[0, 0, -1, 0].abs().item() < 1e-6


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_fft_convolution_supports_complex_dtypes(dtype: torch.dtype) -> None:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    values = torch.ones(1, 2, 5, 3, dtype=dtype)
    kernel = torch.ones(2, 3, dtype=dtype)

    actual = _complex_linear_conv_fft(values, kernel, time_dim=2)

    assert actual.dtype == dtype
    assert actual.real.dtype == real_dtype
    assert torch.isfinite(actual).all()


def test_ssm_block_preserves_shape_dtype_phase_and_gradients() -> None:
    torch.manual_seed(6022)
    block = R6ComplexDiagonalSSMBlock(width=8, modes=3, sequence_length=16)
    values = torch.randn(2, 8, 16, 4, dtype=torch.complex64, requires_grad=True)
    rotated = values * torch.tensor(complex(math.cos(0.61), math.sin(0.61)))

    output = block(values)
    rotated_output = block(rotated)
    phase = torch.tensor(complex(math.cos(0.61), math.sin(0.61)))
    torch.testing.assert_close(rotated_output, output * phase, rtol=2e-4, atol=2e-4)
    assert output.shape == values.shape
    assert output.dtype == values.dtype
    output.abs().square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert block.forward_b_real.grad is not None
    assert torch.isfinite(block.forward_b_real.grad).all()
    assert block.forward_b_real.grad.abs().sum().item() > 0.0


def test_ssm_convolution_does_not_mix_range_cells() -> None:
    values = torch.zeros(1, 2, 7, 3, dtype=torch.complex64)
    values[:, :, :, 1] = torch.randn(1, 2, 7, dtype=torch.complex64)
    kernel = torch.randn(2, 5, dtype=torch.complex64)

    output = _complex_linear_conv_fft(values, kernel, time_dim=2)

    assert output[:, :, :, 0].abs().max().item() == 0.0
    assert output[:, :, :, 2].abs().max().item() == 0.0


def test_ssm_block_is_finite_for_bin_off_bin_and_nyquist_neighbor_tones() -> None:
    torch.manual_seed(6024)
    block = R6ComplexDiagonalSSMBlock(width=8, modes=3, sequence_length=130)
    pulse = torch.arange(130, dtype=torch.float32)
    channel_scale = torch.linspace(0.5, 1.5, 8).reshape(1, 8, 1, 1)
    range_scale = torch.linspace(0.8, 1.2, 4).reshape(1, 1, 1, 4)

    for frequency in (0.0, 7.0 / 130.0, 7.31 / 130.0, 0.49):
        tone = torch.exp(2j * math.pi * frequency * pulse).to(torch.complex64)
        values = tone.reshape(1, 1, 130, 1) * channel_scale * range_scale
        output = block(values)
        assert torch.isfinite(output.real).all()
        assert torch.isfinite(output.imag).all()


@pytest.mark.parametrize("model_type", [R6ComplexSSMStudent, R6ComplexCNNSSMStudent])
def test_ssm_students_match_output_contract_and_common_phase_invariance(model_type) -> None:
    torch.manual_seed(6023)
    model = model_type(width=8, modes=2, blocks=1).eval()
    iq = torch.randn(2, 6, 130, 32)

    output = model(iq)
    rotated_output = model(_rotate_common_phase(iq, 0.83))

    assert output["presence_logit"].shape == (2,)
    assert output["range_logits"].shape == (2, 32)
    assert output["range_offset"].shape == (2, 32)
    assert output["doppler_vector"].shape == (2, 32, 2)
    for key in output:
        torch.testing.assert_close(output[key], rotated_output[key], rtol=2e-4, atol=2e-4)


def test_default_candidate_parameter_counts_match_ce_tcn_range() -> None:
    from radardistill.students.r6_architectures import R6ECEStudent

    reference = count_r6h_parameters(R6ECEStudent())
    candidates = [
        count_r6h_parameters(R6ComplexSSMStudent()),
        count_r6h_parameters(R6ComplexCNNSSMStudent()),
    ]

    assert reference["parameters"] == 63_429
    for result in candidates:
        assert 0.8 * reference["parameters"] <= result["parameters"] <= 1.2 * reference["parameters"]
