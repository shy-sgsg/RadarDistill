import numpy as np
import pytest
import torch

from radardistill.p1_a_e001 import (
    P1PhaseMLP,
    circular_error,
    generate_phase_batch,
    normalized_cross_coherence,
    phase_crlb_std,
    phase_fisher_information,
    phase_profile_ml,
    phase_reference_correlation,
    phase_sample_circular_mean,
    require_cuda,
)
from radardistill.p1_a_e001_runner import _paired_bootstrap_rmse


def test_phase_batch_is_seeded_and_uses_shared_tone_and_recordwise_snr():
    first = generate_phase_batch(64, seed=1201, snr_db=5.0)
    second = generate_phase_batch(64, seed=1201, snr_db=5.0)

    np.testing.assert_array_equal(first.iq, second.iq)
    np.testing.assert_array_equal(first.delta_rad, second.delta_rad)
    assert first.iq.shape == (64, 3, 130)
    assert first.delta_rad.shape == (64, 2)
    assert first.channel_snr_db.shape == (64, 3)
    np.testing.assert_allclose(
        first.channel_snr_db,
        10.0 * np.log10(first.amplitude**2 / first.noise_power[:, None]),
        atol=1e-10,
    )


def test_noiseless_generators_and_three_analytic_estimators_recover_wrapped_phase():
    batch = generate_phase_batch(128, seed=773, snr_db=np.inf)
    correlation = phase_reference_correlation(batch.iq)
    sample_mean = phase_sample_circular_mean(batch.iq)
    profile_ml = phase_profile_ml(batch.iq)

    assert np.max(np.abs(circular_error(correlation, batch.delta_rad))) < 2e-6
    assert np.max(np.abs(circular_error(sample_mean, batch.delta_rad))) < 2e-6
    assert np.max(np.abs(circular_error(profile_ml, batch.delta_rad))) < 2e-5


def test_reference_channel_and_conjugate_direction_match_phase_label_convention():
    n = np.arange(37, dtype=np.float64)
    waveform = np.exp(1j * (0.31 * n + 0.8))
    truth = np.asarray([[np.pi - 1e-5, -np.pi + 2e-5]], dtype=np.float64)
    phase = np.asarray([[0.6, 0.6 + truth[0, 0], 0.6 + truth[0, 1]]])
    iq = np.asarray(
        [2.0, 0.7, 1.3], dtype=np.float64
    )[None, :, None] * np.exp(1j * phase[:, :, None]) * waveform[None, None, :]

    np.testing.assert_allclose(phase_reference_correlation(iq), truth, atol=3e-5)
    features = normalized_cross_coherence(torch.as_tensor(iq.astype(np.complex64)))
    decoded_features = torch.stack(
        (torch.atan2(features[:, 1], features[:, 0]), torch.atan2(features[:, 3], features[:, 2])),
        dim=-1,
    )
    torch.testing.assert_close(
        decoded_features,
        torch.as_tensor(truth.astype(np.float32)),
        atol=3e-5,
        rtol=0,
    )


def test_cross_coherence_is_invariant_to_common_phase_scale_and_frequency_modulation():
    batch = generate_phase_batch(12, seed=113, snr_db=20.0)
    iq = torch.as_tensor(batch.iq)
    base = normalized_cross_coherence(iq)
    common_rotation = iq * torch.exp(torch.tensor(0.79j))
    global_scale = iq * torch.tensor(0.8 - 1.7j)
    n = torch.arange(iq.shape[-1], dtype=torch.float32)
    frequency_shift = iq * torch.exp(2j * torch.pi * 43.25 * n / 1300.0)

    torch.testing.assert_close(normalized_cross_coherence(common_rotation), base)
    torch.testing.assert_close(normalized_cross_coherence(global_scale), base)
    torch.testing.assert_close(normalized_cross_coherence(frequency_shift), base)
    assert base.shape == (12, 4)


def test_phase_mlp_has_small_parameter_count_and_finite_gradients():
    torch.manual_seed(41)
    model = P1PhaseMLP()
    inputs = torch.randn(16, 4)
    targets = torch.randn(16, 4)
    output = model(inputs)
    (output - targets).square().mean().backward()

    assert output.shape == (16, 4)
    assert sum(parameter.numel() for parameter in model.parameters()) < 10_000
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_relative_phase_crlb_matches_finite_difference_fisher_with_waveform_nuisance():
    amplitudes = np.asarray([0.8, 1.1, 1.4], dtype=np.float64)
    sigma2 = 0.17
    sample_count = 7
    phase = np.asarray([0.0, 0.53, -1.07], dtype=np.float64)
    waveform = np.exp(1j * np.linspace(-0.8, 1.3, sample_count))
    analytic = phase_fisher_information(amplitudes, sigma2, waveform)

    # Numerically differentiate mean samples in parameters [delta21, delta31, Re(s[n]), Im(s[n])]
    # and eliminate the independently unknown complex waveform at every sample by a Schur complement.
    h = amplitudes * np.exp(1j * phase)
    numerical = np.zeros((2, 2), dtype=np.float64)
    epsilon = 1e-6
    for sample in waveform:
        def mean(delta21, delta31, real_s, imag_s):
            coefficients = amplitudes * np.exp(
                1j * np.asarray([0.0, delta21, delta31], dtype=np.float64)
            )
            return coefficients * (real_s + 1j * imag_s)

        base_params = [phase[1], phase[2], sample.real, sample.imag]
        jacobian = np.empty((3, 4), dtype=np.complex128)
        for column in range(4):
            plus = base_params.copy()
            minus = base_params.copy()
            plus[column] += epsilon
            minus[column] -= epsilon
            jacobian[:, column] = (mean(*plus) - mean(*minus)) / (2 * epsilon)
        full = (2 / sigma2) * np.real(jacobian.conj().T @ jacobian)
        target = full[:2, :2]
        nuisance = full[2:, 2:]
        cross = full[:2, 2:]
        numerical += target - cross @ np.linalg.solve(nuisance, cross.T)

    np.testing.assert_allclose(analytic, numerical, rtol=2e-7, atol=2e-7)
    assert np.all(phase_crlb_std(amplitudes, sigma2, waveform) > 0)


def test_circular_error_wraps_at_plus_minus_pi():
    errors = circular_error(np.asarray([-np.pi + 0.01, np.pi - 0.01]), np.asarray([np.pi - 0.01, -np.pi + 0.01]))
    np.testing.assert_allclose(errors, np.asarray([0.02, -0.02]), atol=1e-12)


def test_training_device_requires_visible_cuda_and_never_falls_back():
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda(cuda_available=False, device_count=0)
    with pytest.raises(ValueError, match="cuda"):
        require_cuda(training_device="cpu", cuda_available=True, device_count=1)


def test_paired_bootstrap_reports_negative_student_minus_baseline_difference():
    baseline = np.asarray([0.4, -0.3, 0.2, 0.5, -0.4])
    student = baseline * 0.25
    result = _paired_bootstrap_rmse(student, baseline, samples=200, seed=81)

    assert result["mean_rmse_difference_student_minus_baseline_rad"] < 0
    assert result["ci95_upper_rad"] < 0
    assert result["n"] == len(student)
