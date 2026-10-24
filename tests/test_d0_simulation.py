import numpy as np
import pytest
import yaml

from radardistill.simulation.d0 import (
    channel_steering_vector,
    generate_d0_batch,
    verify_lfm_range_psf,
)


@pytest.fixture(scope="module")
def d0_config():
    with open("experiments/D0-E001/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_d0_batch_shape_dtype_and_continuous_truth(d0_config):
    batch = generate_d0_batch(
        count=64,
        seed=13001,
        snr_db=(-5.0, 10.0),
        presence_ratio=0.5,
        config=d0_config,
    )

    assert batch.iq.shape == (64, 3, 130, 32)
    assert batch.iq.dtype == np.complex64
    assert batch.noise_iq.shape == batch.iq.shape
    assert batch.signal_iq.shape == batch.iq.shape
    assert np.iscomplexobj(batch.noise_iq)
    assert np.iscomplexobj(batch.signal_iq)
    assert np.any(batch.truth_presence)
    assert np.any(~batch.truth_presence)
    h1 = batch.truth_presence
    h0 = ~h1
    assert np.all(np.isfinite(batch.truth_range_cell[h1]))
    assert np.all(np.isfinite(batch.truth_doppler_hz[h1]))
    assert np.all(np.isnan(batch.truth_range_cell[h0]))
    assert np.all(np.isnan(batch.truth_doppler_hz[h0]))
    assert np.all((batch.truth_range_cell[h1] >= 4.0) & (batch.truth_range_cell[h1] < 28.0))
    assert np.all((batch.truth_doppler_hz[h1] >= -650.0) & (batch.truth_doppler_hz[h1] < 650.0))
    assert np.all((batch.snr_db >= -5.0) & (batch.snr_db <= 10.0))
    assert np.allclose(batch.iq, batch.signal_iq + batch.noise_iq, rtol=1e-6, atol=1e-6)


def test_seed_reproduces_iq_noise_and_all_labels(d0_config):
    first = generate_d0_batch(24, 13002, 3.0, 0.5, d0_config)
    repeated = generate_d0_batch(24, 13002, 3.0, 0.5, d0_config)

    for name in (
        "iq",
        "signal_iq",
        "noise_iq",
        "truth_presence",
        "truth_range_cell",
        "truth_doppler_hz",
        "truth_initial_phase_rad",
        "nuisance_initial_phase_rad",
        "truth_target_amplitude",
        "reference_amplitude",
        "noise_power",
        "snr_db",
        "truth_azimuth_deg",
        "truth_elevation_deg",
        "nuisance_azimuth_deg",
        "nuisance_elevation_deg",
    ):
        np.testing.assert_allclose(
            getattr(first, name), getattr(repeated, name), rtol=0.0, atol=0.0, equal_nan=True
        )


def test_h0_and_h1_counterfactuals_share_noise_and_nuisance_draws(d0_config):
    h0 = generate_d0_batch(32, 13003, 0.0, 0.0, d0_config)
    h1 = generate_d0_batch(32, 13003, 0.0, 1.0, d0_config)

    np.testing.assert_array_equal(h0.noise_iq, h1.noise_iq)
    np.testing.assert_array_equal(h0.noise_power, h1.noise_power)
    np.testing.assert_array_equal(h0.reference_amplitude, h1.reference_amplitude)
    np.testing.assert_array_equal(h0.nuisance_initial_phase_rad, h1.nuisance_initial_phase_rad)
    np.testing.assert_array_equal(h0.nuisance_azimuth_deg, h1.nuisance_azimuth_deg)
    np.testing.assert_array_equal(h0.nuisance_elevation_deg, h1.nuisance_elevation_deg)
    np.testing.assert_array_equal(h0.signal_iq, np.zeros_like(h0.signal_iq))
    np.testing.assert_allclose(h1.iq - h0.iq, h1.signal_iq, rtol=1e-6, atol=1e-6)
    assert np.all(~h0.truth_presence)
    assert np.all(h1.truth_presence)


def test_noise_power_matches_each_sample_snr(d0_config):
    batch = generate_d0_batch(64, 13004, 0.0, 0.5, d0_config)
    expected = batch.reference_amplitude**2 / batch.noise_power
    expected_db = 10.0 * np.log10(expected)

    np.testing.assert_allclose(batch.snr_db, expected_db, atol=1e-6, rtol=0.0)
    normalized_power = np.abs(batch.noise_iq) ** 2 / batch.noise_power[:, None, None, None]
    assert abs(float(normalized_power.mean()) - 1.0) < 0.005


def test_fractional_range_target_uses_the_frozen_lfm_psf(d0_config):
    result = verify_lfm_range_psf(d0_config)
    assert result["offset_count"] == 191
    assert result["max_normalized_magnitude_error"] <= 2e-4

    batch = generate_d0_batch(1, 13005, np.inf, 1.0, d0_config)
    r0 = float(batch.truth_range_cell[0])
    assert abs(r0 - round(r0)) > 1e-5
    assert np.count_nonzero(np.abs(batch.signal_iq[0, 0, 0]) > 1e-5) > 1


def test_three_channel_steering_matches_apx15_relative_path_phase(d0_config):
    azimuth_deg = 23.0
    elevation_deg = -4.1
    steering = channel_steering_vector(azimuth_deg, elevation_deg, d0_config)
    positions = np.asarray(d0_config["channels"]["positions_xyz_m"], dtype=np.float64)
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    look = np.asarray([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    wavelength = 299792458.0 / d0_config["waveform"]["carrier_hz"]
    expected = np.exp(-2j * np.pi * ((positions - positions.mean(axis=0)) @ look) / wavelength)

    np.testing.assert_allclose(steering, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(steering), 1.0, rtol=0.0, atol=1e-12)
