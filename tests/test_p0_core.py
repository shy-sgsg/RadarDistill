import numpy as np

from radardistill.baselines import (
    estimate_mle,
    estimate_phase_difference,
    frequency_crlb_hz,
    local_teacher_soft_target,
    wrapped_error,
)
from radardistill.cnn import TinyCircularCNN, frequency_to_circle
from radardistill.experiment import _metrics, run_p0
from radardistill.signal import generate_batch, generate_single_tone


def test_noiseless_tone_has_the_requested_phase_increment():
    frequency_hz = 137.35
    phase_rad = 0.71
    prf_hz = 1300.0
    iq = generate_single_tone(
        frequency_hz,
        phase_rad,
        amplitude=1.2,
        snr_db=np.inf,
        prf_hz=prf_hz,
        n_samples=130,
        seed=5,
    )

    measured_increment = np.angle(np.mean(iq[1:] * np.conj(iq[:-1])))
    measured_frequency = measured_increment * prf_hz / (2.0 * np.pi)
    assert abs(measured_frequency - frequency_hz) < 2e-5


def test_batch_generation_is_seeded_and_uses_off_grid_frequencies():
    first = generate_batch(count=64, seed=18, prf_hz=1300.0, n_samples=130)
    repeated = generate_batch(count=64, seed=18, prf_hz=1300.0, n_samples=130)

    np.testing.assert_array_equal(first.iq, repeated.iq)
    np.testing.assert_array_equal(first.frequency_hz, repeated.frequency_hz)
    assert np.any(np.abs(first.frequency_hz / 10.0 - np.round(first.frequency_hz / 10.0)) > 1e-3)
    assert first.iq.shape == (64, 130)
    assert np.iscomplexobj(first.iq)


def test_batch_noise_power_matches_the_requested_snr():
    batch = generate_batch(count=256, seed=6, prf_hz=1300.0, n_samples=130, snr_db=0.0)
    n = np.arange(130, dtype=np.float64)
    clean = batch.amplitude[:, None] * np.exp(
        1j * (2.0 * np.pi * batch.frequency_hz[:, None] * n / 1300.0 + batch.phase_rad[:, None])
    )
    measured_noise_power = float(np.mean(np.abs(batch.iq - clean) ** 2))
    expected_noise_power = float(np.mean(batch.amplitude**2))
    assert abs(measured_noise_power / expected_noise_power - 1.0) < 0.02


def test_frequency_error_wraps_at_the_unambiguous_interval_boundary():
    error = wrapped_error(-649.9, 649.9, prf_hz=1300.0)
    assert abs(float(error) - 0.2) < 1e-9


def test_continuous_mle_recovers_a_noiseless_sub_bin_tone():
    truth_hz = 137.35
    iq = generate_single_tone(
        truth_hz,
        phase_rad=-1.2,
        amplitude=0.8,
        snr_db=np.inf,
        prf_hz=1300.0,
        n_samples=130,
        seed=4,
    )

    estimate_hz = estimate_mle(iq, prf_hz=1300.0)
    assert abs(float(wrapped_error(estimate_hz, truth_hz, 1300.0))) < 1e-4


def test_phase_difference_estimator_recovers_a_clean_tone():
    truth_hz = -226.4
    iq = generate_single_tone(
        truth_hz,
        phase_rad=0.2,
        amplitude=1.0,
        snr_db=np.inf,
        prf_hz=1300.0,
        n_samples=130,
        seed=10,
    )
    estimate_hz = estimate_phase_difference(iq, prf_hz=1300.0)
    assert abs(float(wrapped_error(estimate_hz, truth_hz, 1300.0))) < 1e-5


def test_crlb_reports_the_complex_tone_standard_deviation():
    observed = frequency_crlb_hz(prf_hz=1300.0, n_samples=130, snr_db=10.0)
    centered_time_energy = 130.0 * (130.0**2 - 1.0) / 12.0
    expected_variance = 1300.0**2 / (8.0 * np.pi**2 * 10.0 * centered_time_energy)
    assert abs(observed**2 - expected_variance) < 1e-12


def test_teacher_local_likelihood_is_a_finite_circular_soft_target():
    truth_hz = 89.2
    iq = generate_single_tone(
        truth_hz,
        phase_rad=0.4,
        amplitude=1.0,
        snr_db=10.0,
        prf_hz=1300.0,
        n_samples=130,
        seed=32,
    )
    teacher_hz = estimate_mle(iq, prf_hz=1300.0)
    target = local_teacher_soft_target(
        iq[None, :],
        teacher_frequency_hz=np.asarray([teacher_hz]),
        noise_power=np.asarray([0.1]),
        prf_hz=1300.0,
    )
    assert target.shape == (1, 2)
    assert np.isfinite(target).all()
    assert float(np.linalg.norm(target[0])) <= 1.0 + 1e-8


def test_outliers_are_counted_against_half_an_fft_bin():
    summary = _metrics(
        np.asarray([0.0, 3.0, 6.0]),
        np.random.default_rng(1),
        bootstrap_samples=20,
        outlier_threshold_hz=5.0,
    )
    assert summary["outlier_rate"] == 1.0 / 3.0


def test_tiny_cnn_predicts_circular_outputs_and_training_updates_it():
    batch = generate_batch(
        count=128,
        seed=21,
        prf_hz=1300.0,
        n_samples=130,
        snr_db=20.0,
    )
    targets = frequency_to_circle(batch.frequency_hz, prf_hz=1300.0)
    model = TinyCircularCNN(n_samples=130, seed=3)

    initial = model.predict(batch.iq[:8])
    losses = model.fit(batch.iq, targets, epochs=4, batch_size=32, learning_rate=0.002, seed=12)
    predicted = model.predict(batch.iq[:8])

    assert initial.shape == (8, 2)
    assert predicted.shape == (8, 2)
    assert np.isfinite(predicted).all()
    assert len(losses) == 4
    assert np.isfinite(losses).all()
    assert not np.array_equal(initial, predicted)


def test_flatten_cnn_keeps_temporal_positions_in_its_output():
    batch = generate_batch(
        count=16,
        seed=24,
        prf_hz=1300.0,
        n_samples=130,
        snr_db=10.0,
    )
    model = TinyCircularCNN(n_samples=130, seed=2, channels=4, kernel_size=5, pooling="flatten", hidden_units=8)
    initial = model.predict(batch.iq)
    history = model.fit(
        batch.iq,
        frequency_to_circle(batch.frequency_hz, prf_hz=1300.0),
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        seed=7,
    )
    output = model.predict(batch.iq)
    assert output.shape == (16, 2)
    assert np.isfinite(output).all()
    assert len(history) == 2
    assert not np.array_equal(initial, output)


def test_flatten_cnn_weights_round_trip_without_changing_predictions(tmp_path):
    batch = generate_batch(count=8, seed=25, prf_hz=1300.0, n_samples=130, snr_db=10.0)
    model = TinyCircularCNN(n_samples=130, seed=2, channels=4, kernel_size=5, pooling="flatten", hidden_units=8)
    expected = model.predict(batch.iq)
    path = tmp_path / "flatten_model.npz"
    model.save(str(path))

    restored = TinyCircularCNN(n_samples=130, seed=9, channels=4, kernel_size=5, pooling="flatten", hidden_units=8)
    restored.load(str(path))
    np.testing.assert_allclose(restored.predict(batch.iq), expected, rtol=0.0, atol=1e-7)


def test_p0_runner_writes_replayable_experiment_artifacts(tmp_path):
    config = {
        "experiment_id": "P0-E-SMOKE",
        "prf_hz": 1300.0,
        "n_samples": 130,
        "train_count": 24,
        "validation_count": 12,
        "test_per_snr": 8,
        "train_snr_levels_db": [0.0, 5.0],
        "test_snr_levels_db": [0.0, 5.0],
        "train_seeds": [42],
        "validation_seed": 70,
        "test_seed": 90,
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": 2,
        "teacher_weight": 0.5,
        "channels": 4,
        "kernel_size": 5,
        "pooling": "flatten",
        "hidden_units": 8,
        "bootstrap_samples": 20,
    }
    run_p0(config, tmp_path)

    assert (tmp_path / "config.yaml").is_file()
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "result.csv").is_file()
    assert (tmp_path / "analysis.md").is_file()
    assert (tmp_path / "datasets" / "test.npz").is_file()
    assert (tmp_path / "models" / "seed_42_E1_supervised.npz").is_file()
    import json

    assert json.loads((tmp_path / "metrics.json").read_text())["student"]["pooling"] == "flatten"
