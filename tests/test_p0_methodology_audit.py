import numpy as np
import pytest
import torch

from radardistill.baselines import (
    estimate_fft_peak,
    estimate_mle_global_grid,
    estimate_mle_local,
    local_teacher_profile_likelihood,
    wrapped_error,
)
from radardistill.cnn import TinyCircularCNN, frequency_to_circle
from radardistill.experiment import _select_validation_target, _test_subsets
from radardistill.signal import SignalBatch, generate_batch, generate_single_tone
from radardistill.torch_models import (
    CorrelationMLP,
    FrequencyResponseMLP,
    FrozenE005CNN,
    correlation_features,
    frequency_response_features,
)
from radardistill.torch_experiment import (
    _evaluate_predictions,
    _g3_gate_windows,
    _paired_rmse_delta_ci,
    _prediction_replicates,
)


def test_global_grid_profile_mle_refines_off_grid_and_nyquist_tones():
    prf_hz = 1500.0
    n_samples = 97
    truths = np.asarray([-prf_hz / 2.0 + 0.001, -221.37, 338.42, prf_hz / 2.0 - 0.002])
    signals = np.stack(
        [
            generate_single_tone(f, 0.31, 0.73, np.inf, prf_hz, n_samples, 12 + i)
            for i, f in enumerate(truths)
        ]
    )

    estimate = estimate_mle_global_grid(signals, prf_hz, grid_size=4096)
    np.testing.assert_array_less(np.abs(wrapped_error(estimate, truths, prf_hz)), 1e-4)


def test_local_and_global_profile_mle_match_for_noiseless_off_grid_tone():
    prf_hz, n_samples, truth = 1300.0, 130, 203.275
    iq = generate_single_tone(truth, -1.2, 1.1, np.inf, prf_hz, n_samples, 9)
    local = estimate_mle_local(iq, prf_hz)
    global_grid = estimate_mle_global_grid(iq, prf_hz, grid_size=4096)
    assert abs(float(wrapped_error(local, global_grid, prf_hz))) < 1e-4
    assert abs(float(wrapped_error(global_grid, truth, prf_hz))) < 1e-4


def test_low_snr_fft_wrong_peak_can_trap_local_mle_while_global_grid_finds_other_peak():
    from radardistill.baselines import estimate_fft_peak
    from radardistill.signal import generate_batch

    prf_hz = 1300.0
    batch = generate_batch(128, seed=61000, prf_hz=prf_hz, n_samples=130, snr_db=-15.0)
    index = 5
    fft = estimate_fft_peak(batch.iq[index], prf_hz)
    local = estimate_mle_local(batch.iq[index], prf_hz)
    global_grid = estimate_mle_global_grid(batch.iq[index], prf_hz, grid_size=1024)
    fft_error = float(wrapped_error(fft, batch.frequency_hz[index], prf_hz))
    local_error = float(wrapped_error(local, batch.frequency_hz[index], prf_hz))
    global_error = float(wrapped_error(global_grid, batch.frequency_hz[index], prf_hz))
    assert abs(fft_error) > prf_hz / 4
    assert abs(local_error) > prf_hz / 4
    assert abs(global_error) < 2.0


def test_simulator_and_local_mle_follow_phase_scale_shift_and_conjugation_relations():
    prf_hz, n_samples, truth = 1300.0, 130, 217.3
    clean = generate_single_tone(truth, 0.42, 0.8, np.inf, prf_hz, n_samples, 7)
    rotation = np.exp(0.73j)
    scale = 1.6 - 0.4j
    delta_hz = 3.0 * prf_hz / n_samples
    n = np.arange(n_samples)
    shifted = clean * np.exp(2j * np.pi * delta_hz * n / prf_hz)
    rotated_signal = generate_single_tone(truth, 0.42 + 0.73, 0.8, np.inf, prf_hz, n_samples, 8)
    scaled_signal = generate_single_tone(truth, 0.42, abs(scale) * 0.8, np.inf, prf_hz, n_samples, 9)
    shifted_signal = generate_single_tone(truth + delta_hz, 0.42, 0.8, np.inf, prf_hz, n_samples, 10)
    conjugate_signal = generate_single_tone(-truth, -0.42, 0.8, np.inf, prf_hz, n_samples, 11)
    np.testing.assert_allclose(rotated_signal, clean * rotation, atol=2e-6)
    np.testing.assert_allclose(scaled_signal, clean * abs(scale), atol=2e-6)
    np.testing.assert_allclose(shifted_signal, shifted, atol=2e-6)
    np.testing.assert_allclose(conjugate_signal, clean.conj(), atol=2e-6)

    observed = generate_single_tone(truth, 0.42, 0.8, 20.0, prf_hz, n_samples, 12)
    base = estimate_mle_local(observed, prf_hz)
    assert abs(float(wrapped_error(estimate_mle_local(observed * rotation, prf_hz), base, prf_hz))) < 2e-3
    assert abs(float(wrapped_error(estimate_mle_local(observed * scale, prf_hz), base, prf_hz))) < 2e-3
    shifted_estimate = estimate_mle_local(observed * np.exp(2j * np.pi * delta_hz * n / prf_hz), prf_hz)
    assert abs(float(wrapped_error(shifted_estimate, base + delta_hz, prf_hz))) < 2e-3
    conjugated_estimate = estimate_mle_local(observed.conj(), prf_hz)
    assert abs(float(wrapped_error(conjugated_estimate, -base, prf_hz))) < 2e-3
    assert abs(float(wrapped_error(estimate_fft_peak(clean, prf_hz), truth, prf_hz))) <= prf_hz / (2 * n_samples)


def test_estimator_rejects_empty_and_invalid_batch_shapes():
    from radardistill.baselines import estimate_fft_peak

    for values in (np.asarray([], dtype=np.complex64), np.empty((2, 0), dtype=np.complex64)):
        with pytest.raises(ValueError):
            estimate_fft_peak(values, 1300.0)


def test_test_subset_fft_offset_uses_actual_sample_count_and_prf():
    prf_hz, n_samples = 900.0, 50
    frequencies = np.asarray([-211.0, -83.0, 0.3, 117.0, 244.0])
    batch = SignalBatch(
        iq=np.ones((len(frequencies), n_samples), dtype=np.complex64),
        frequency_hz=frequencies,
        phase_rad=np.zeros(len(frequencies)),
        amplitude=np.ones(len(frequencies)),
        snr_db=np.zeros(len(frequencies)),
        seed=1,
    )

    subsets = _test_subsets(batch, prf_hz)
    offsets = (frequencies / (prf_hz / n_samples) + 0.5) % 1.0 - 0.5
    expected = np.clip(np.floor((offsets + 0.5) * 4.0).astype(int), 0, 3)
    observed = np.asarray(
        [next(i for i in range(4) if subsets[f"fft_bin_offset_quartile_{i}"][j]) for j in range(len(frequencies))]
    )
    np.testing.assert_array_equal(observed, expected)


def test_teacher_profile_returns_normalized_local_likelihood_vector():
    prf_hz, n_samples, truth, snr_db = 1300.0, 130, 649.8, 15.0
    iq = generate_single_tone(truth, 0.7, 0.9, snr_db, prf_hz, n_samples, 23)
    from radardistill.baselines import estimate_mle

    teacher = estimate_mle(iq, prf_hz)
    noise_power = 0.9**2 * 10 ** (-snr_db / 10.0)
    candidates, probability = local_teacher_profile_likelihood(
        iq, teacher, noise_power, prf_hz, candidate_count=65, half_width_bins=2.0
    )
    assert candidates.shape == probability.shape == (65,)
    assert np.all(np.isfinite(probability))
    np.testing.assert_allclose(np.sum(probability), 1.0, atol=1e-12)
    assert abs(float(wrapped_error(candidates[np.argmax(probability)], teacher, prf_hz))) <= prf_hz / n_samples


def test_e2_validation_selection_is_explicit_and_separates_teacher_from_truth():
    truth = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    teacher = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    np.testing.assert_array_equal(_select_validation_target("teacher", truth, teacher), teacher)
    np.testing.assert_array_equal(_select_validation_target("truth", truth, teacher), truth)
    with pytest.raises(ValueError):
        _select_validation_target("implicit", truth, teacher)


@pytest.mark.parametrize(
    "parameter_name,index",
    [
        ("conv_weight", (0, 0, 0)),
        ("conv_bias", (0,)),
        ("hidden_weight", (0, 0)),
        ("hidden_bias", (0,)),
        ("head_weight", (0, 0)),
        ("head_bias", (0,)),
    ],
)
def test_numpy_cnn_gradients_match_finite_differences_with_distillation(parameter_name, index):
    rng = np.random.default_rng(9)
    iq = (rng.normal(size=(7, 16)) + 1j * rng.normal(size=(7, 16))).astype(np.complex64)
    target = frequency_to_circle(np.linspace(-400.0, 450.0, 7), 1300.0)
    teacher = frequency_to_circle(np.linspace(300.0, -350.0, 7), 1300.0)
    model = TinyCircularCNN(
        n_samples=16, seed=4, channels=2, kernel_size=3, pooling="flatten", hidden_units=3
    )
    loss, gradients = model.loss_and_gradients(
        iq, target, teacher_targets=teacher, distillation_weight=0.6
    )
    parameter = getattr(model, parameter_name)
    original = float(parameter[index])
    step = 1e-2
    parameter[index] = original + step
    plus = model.loss_value(iq, target, teacher_targets=teacher, distillation_weight=0.6)
    parameter[index] = original - step
    minus = model.loss_value(iq, target, teacher_targets=teacher, distillation_weight=0.6)
    parameter[index] = original
    numerical = (plus - minus) / (2.0 * step)
    analytic = float(gradients[parameter_name][index])
    assert np.isfinite(loss)
    assert analytic == pytest.approx(numerical, rel=2e-2, abs=1e-3)


def test_complex_correlation_features_are_phase_and_full_scale_invariant():
    rng = np.random.default_rng(8)
    iq = torch.as_tensor(
        (rng.normal(size=(5, 130)) + 1j * rng.normal(size=(5, 130))).astype(np.complex64)
    )
    baseline = correlation_features(iq, lags=(1, 2, 4, 8))
    rotated = correlation_features(iq * torch.exp(torch.tensor(0.73j)), lags=(1, 2, 4, 8))
    scaled = correlation_features(iq * (2.4 - 0.5j), lags=(1, 2, 4, 8))
    torch.testing.assert_close(rotated, baseline, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(scaled, baseline, rtol=1e-5, atol=1e-6)


def test_spectrum_features_preserve_frequency_shift_and_ignore_common_phase_scale():
    rng = np.random.default_rng(12)
    iq = torch.as_tensor(
        (rng.normal(size=(3, 130)) + 1j * rng.normal(size=(3, 130))).astype(np.complex64)
    )
    base = frequency_response_features(iq, n_fft=520)
    transformed = frequency_response_features(iq * torch.exp(torch.tensor(0.91j)) * 3.0, n_fft=520)
    torch.testing.assert_close(transformed, base, rtol=1e-5, atol=1e-6)

    frequency_shift_bins = 7
    n = torch.arange(iq.shape[-1], dtype=torch.float32)
    shifted_iq = iq * torch.exp(2j * torch.pi * frequency_shift_bins * n / 520)
    shifted = frequency_response_features(shifted_iq, n_fft=520)
    expected = torch.roll(base, shifts=frequency_shift_bins, dims=-1)
    torch.testing.assert_close(shifted, expected, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize(
    "model,input_size",
    [(CorrelationMLP(), 8), (FrequencyResponseMLP(n_fft=520), 520)],
)
def test_torch_student_forward_and_parameter_gradients(model, input_size):
    generator = torch.Generator().manual_seed(21)
    inputs = torch.randn((4, input_size), generator=generator, dtype=torch.float32)
    prediction = model(inputs)
    target = torch.tensor([[1.0, 0.0]] * 4)
    loss = (prediction[:, :2] - target).square().mean() + 0.01 * prediction[:, 2:].square().mean()
    loss.backward()
    parameter = next(model.parameters())
    assert prediction.shape == (4, 67)
    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    index = (0, 0)
    original = float(parameter[index].detach())
    step = 1e-3
    with torch.no_grad():
        parameter[index] = original + step
    plus = (model(inputs)[:, :2] - target).square().mean() + 0.01 * model(inputs)[:, 2:].square().mean()
    with torch.no_grad():
        parameter[index] = original - step
    minus = (model(inputs)[:, :2] - target).square().mean() + 0.01 * model(inputs)[:, 2:].square().mean()
    with torch.no_grad():
        parameter[index] = original
    numerical = float(((plus - minus) / (2.0 * step)).detach())
    analytic = float(parameter.grad[index])
    assert analytic == pytest.approx(numerical, rel=4e-2, abs=2e-3)


def test_frozen_e005_numpy_checkpoint_torch_port_matches_predictions(tmp_path):
    rng = np.random.default_rng(55)
    increments = (
        rng.normal(size=(8, 129)) + 1j * rng.normal(size=(8, 129))
    ).astype(np.complex64)
    model = TinyCircularCNN(n_samples=129, seed=5, channels=6, kernel_size=9)
    expected = model.predict(increments)
    checkpoint = tmp_path / "e005.npz"
    model.save(str(checkpoint))
    port = FrozenE005CNN(channels=6, kernel_size=9)
    port.load_numpy_checkpoint(str(checkpoint))
    features = torch.stack((torch.as_tensor(increments.real), torch.as_tensor(increments.imag)), dim=1)
    observed = port(features).detach().numpy()
    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-6)


def test_paired_bootstrap_supports_seed_matched_candidate_and_reference():
    candidate = np.asarray([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]])
    reference = np.asarray([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]])
    result = _paired_rmse_delta_ci(candidate, reference, 64, np.random.default_rng(4))
    assert result["delta_rmse_hz"] > 0.0
    assert result["ci95_low_hz"] <= result["delta_rmse_hz"] <= result["ci95_high_hz"]
    assert result["candidate_rmse_ci95_high_hz"] > result["candidate_rmse_ci95_low_hz"]


def test_g3_requires_negative_paired_ci_and_two_of_three_seed_directions():
    snr_db = np.asarray([0.0, 0.0, 5.0, 5.0])
    reference = np.ones((3, 4), dtype=np.float64)
    candidate = np.tile(np.asarray([[0.1], [1.2], [1.3]]), (1, 4))
    errors = {
        "final_blind__S1/E1_truth": reference,
        "final_blind__S1/E3_T2_truth_plus_soft": candidate,
    }
    paired = {
        "S1/E3_T2_truth_plus_soft": {
            str(snr): {"ci95_high_hz": -0.01} for snr in (0.0, 5.0)
        }
    }
    windows, details = _g3_gate_windows(paired, errors, snr_db, [0.0, 5.0], 2)
    assert windows["S1/E3_T2_truth_plus_soft"] == []
    assert details["S1/E3_T2_truth_plus_soft"]["0.0"]["seeds_improved"] == 1

    candidate[:] = np.asarray([[0.1], [0.2], [1.3]])
    windows, details = _g3_gate_windows(paired, errors, snr_db, [0.0, 5.0], 2)
    assert windows["S1/E3_T2_truth_plus_soft"] == [[0.0, 5.0]]
    assert details["S1/E3_T2_truth_plus_soft"]["5.0"]["seeds_improved"] == 2


def test_vector_baseline_predictions_are_one_replicate_not_one_per_sample():
    batch = generate_batch(12, seed=45, prf_hz=1300.0, n_samples=16, snr_db=10.0)
    predictions = {"baseline": batch.frequency_hz + 3.0}
    config = {
        "prf_hz": 1300.0,
        "n_samples": 16,
        "data": {"test_snr_levels_db": [10.0]},
    }
    replicates = _prediction_replicates(predictions["baseline"], (12,), "baseline")
    assert len(replicates) == 1
    np.testing.assert_array_equal(replicates[0], predictions["baseline"])
    rows, summary, errors = _evaluate_predictions("test", batch, predictions, config)
    assert errors["test__baseline"].shape == (1, 12)
    assert summary["baseline"]["10.0"]["n_per_seed"] == 12
    assert summary["baseline"]["10.0"]["rmse_mean_seed_hz"] == pytest.approx(3.0)
    assert len(rows) == 18
