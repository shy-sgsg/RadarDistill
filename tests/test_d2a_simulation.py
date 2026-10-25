from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from radardistill.simulation.d0 import generate_d0_batch
from radardistill.simulation.d2a import (
    compute_scene_power_metrics,
    generate_d0_compatible_no_clutter,
    generate_d2a_paired_batch,
    synthesize_scatterer_echoes,
)


@pytest.fixture(scope="module")
def d2a_config():
    with open("experiments/D2-A-E001/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _single_scatterer(config, *, azimuth_deg=90.0, platform_speed_mps=60.0, target_cell=12.25):
    platform = np.asarray(config["platform"]["initial_position_xyz_m"], dtype=np.float64)
    altitude = platform[2]
    slant_range = (
        config["range_grid"]["axis_start_m"]
        + target_cell * config["waveform"]["range_cell_m"]
    )
    horizontal_range = np.sqrt(slant_range**2 - altitude**2)
    azimuth = np.deg2rad(azimuth_deg)
    position = platform + np.asarray(
        [horizontal_range * np.cos(azimuth), horizontal_range * np.sin(azimuth), -altitude]
    )
    return np.asarray([position]), np.asarray([0.7 - 0.2j]), platform, np.asarray(
        [platform_speed_mps, 0.0, 0.0]
    )


def _response(config, *, azimuth_deg=90.0, platform_speed_mps=60.0, beam_center_deg=90.0):
    positions, reflectivity, platform, velocity = _single_scatterer(
        config, azimuth_deg=azimuth_deg, platform_speed_mps=platform_speed_mps
    )
    return synthesize_scatterer_echoes(
        positions,
        reflectivity,
        np.arange(config["radar"]["pulse_count"]) / config["radar"]["prf_hz"],
        config,
        platform_position_xyz_m=platform,
        platform_velocity_xyz_mps=velocity,
        beam_center_azimuth_deg=beam_center_deg,
    )


def test_t1_zero_clutter_path_regresses_to_frozen_d0(d2a_config):
    with open("experiments/D0-E001/config.yaml", encoding="utf-8") as stream:
        d0_config = yaml.safe_load(stream)
    expected = generate_d0_batch(8, 81001, 3.0, 0.5, d0_config)
    actual = generate_d0_compatible_no_clutter(8, 81001, 3.0, 0.5, d0_config)
    for name in expected.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_t2_noiseless_static_scatterer_preserves_range_phase_doppler_and_coherence(d2a_config):
    response = _response(d2a_config)
    channel = 1
    pulse = 25
    scatterer = 0
    range_bin = int(round(float(response.range_cell[channel, pulse, scatterer])))
    path = float(response.path_length_m[channel, pulse, scatterer])
    wavelength = d2a_config["waveform"]["propagation_speed_mps"] / d2a_config["waveform"]["carrier_hz"]
    phase = np.exp(-2j * np.pi * path / wavelength)
    expected = response.reflectivity[scatterer] * response.beam_gain[pulse, scatterer] * phase
    expected *= np.sinc(response.range_cell[channel, pulse, scatterer] - range_bin)
    np.testing.assert_allclose(response.iq[channel, pulse, range_bin], expected, rtol=1e-6, atol=1e-6)

    for ch in range(d2a_config["radar"]["channel_count"]):
        phase_step = np.angle(response.iq[ch, 10, range_bin] * np.conj(response.iq[ch, 9, range_bin]))
        expected_step = -2.0 * np.pi * (
            response.path_length_m[ch, 10, scatterer] - response.path_length_m[ch, 9, scatterer]
        ) / wavelength
        assert abs(np.angle(np.exp(1j * (phase_step - expected_step)))) < 2e-4


def test_t3_stationary_scatterer_doppler_tracks_platform_speed(d2a_config):
    slow = _response(d2a_config, azimuth_deg=88.9, platform_speed_mps=20.0)
    fast = _response(d2a_config, azimuth_deg=88.9, platform_speed_mps=60.0)
    ch, n, s = 1, 20, 0
    wavelength = d2a_config["waveform"]["propagation_speed_mps"] / d2a_config["waveform"]["carrier_hz"]
    prf = d2a_config["radar"]["prf_hz"]
    f_slow = -(slow.path_length_m[ch, n + 1, s] - slow.path_length_m[ch, n, s]) * prf / wavelength
    f_fast = -(fast.path_length_m[ch, n + 1, s] - fast.path_length_m[ch, n, s]) * prf / wavelength
    assert f_fast > f_slow > 0.0
    assert f_fast / f_slow == pytest.approx(3.0, rel=0.01)


def test_t4_beam_pointing_changes_scatterer_gain_in_the_expected_direction(d2a_config):
    on_axis = _response(d2a_config, azimuth_deg=90.8, beam_center_deg=90.8)
    off_axis = _response(d2a_config, azimuth_deg=90.8, beam_center_deg=90.0)
    assert on_axis.beam_gain[0, 0] > off_axis.beam_gain[0, 0]
    assert np.max(np.abs(on_axis.iq)) > np.max(np.abs(off_axis.iq))


def test_t5_phase_center_geometry_matches_explicit_tx_rx_paths(d2a_config):
    response = _response(d2a_config)
    t = 7 / d2a_config["radar"]["prf_hz"]
    platform = np.asarray(d2a_config["platform"]["initial_position_xyz_m"]) + np.asarray(
        d2a_config["platform"]["velocity_xyz_mps"]
    ) * t
    tx = platform + np.asarray(d2a_config["transmitter"]["phase_center_offset_xyz_m"])
    rx_positions = np.asarray(d2a_config["channels"]["positions_xyz_m"])
    scatterer = response.scatterer_positions_xyz_m[0]
    for ch in range(3):
        expected = np.linalg.norm(scatterer - tx) + np.linalg.norm(scatterer - platform - rx_positions[ch])
        assert response.path_length_m[ch, 7, 0] == pytest.approx(expected, abs=1e-9)


def test_t6_paired_residual_is_only_target_signal(d2a_config):
    batch = generate_d2a_paired_batch(12, 81006, 10.0, 0.0, d2a_config, scenario_id="separated")
    np.testing.assert_allclose(batch.iq_h1 - batch.iq_h0, batch.target_signal_iq, rtol=1e-6, atol=1e-6)


def test_t7_power_metrics_use_explicit_snr_cnr_and_scnr_definitions():
    target = np.full((2, 3, 4), 2.0 + 0j, dtype=np.complex64)
    clutter = np.full_like(target, 1.0 + 0j)
    noise = np.full_like(target, 0.5 + 0j)
    metrics = compute_scene_power_metrics(target, clutter, noise)
    assert metrics["snr_linear"] == pytest.approx(16.0)
    assert metrics["cnr_linear"] == pytest.approx(4.0)
    assert metrics["scnr_linear"] == pytest.approx(16.0 / 5.0)
    assert metrics["scnr_linear"] == pytest.approx(
        metrics["snr_linear"] / (1.0 + metrics["cnr_linear"])
    )


def test_t8_seed_reproduces_scene_noise_and_ids(d2a_config):
    first = generate_d2a_paired_batch(3, 81008, 5.0, 0.0, d2a_config)
    second = generate_d2a_paired_batch(3, 81008, 5.0, 0.0, d2a_config)
    for name in ("iq_h0", "iq_h1", "clutter_iq", "noise_iq", "target_signal_iq", "base_scene_id", "background_id", "target_id", "realization_id"):
        np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_t9_target_truth_does_not_change_h0_background_or_noise(d2a_config):
    first = generate_d2a_paired_batch(
        4, 81009, 5.0, 0.0, d2a_config,
        target_range_cell=8.5, target_azimuth_deg=89.0, target_radial_velocity_mps=1.0,
    )
    second = generate_d2a_paired_batch(
        4, 81009, 5.0, 0.0, d2a_config,
        target_range_cell=24.5, target_azimuth_deg=91.0, target_radial_velocity_mps=4.0,
    )
    np.testing.assert_array_equal(first.iq_h0, second.iq_h0)
    np.testing.assert_array_equal(first.clutter_iq, second.clutter_iq)
    np.testing.assert_array_equal(first.noise_iq, second.noise_iq)
    assert not np.array_equal(first.target_signal_iq, second.target_signal_iq)


def test_t10_coordinate_translation_and_common_phase_rotation_are_symmetries(d2a_config):
    response = _response(d2a_config)
    translation = np.asarray([13.0, -8.0, 2.0])
    positions, reflectivity, platform, velocity = _single_scatterer(d2a_config)
    translated_config = copy.deepcopy(d2a_config)
    translated = synthesize_scatterer_echoes(
        positions + translation,
        reflectivity,
        np.arange(130) / 1300.0,
        translated_config,
        platform_position_xyz_m=platform + translation,
        platform_velocity_xyz_mps=velocity,
    )
    np.testing.assert_allclose(translated.iq, response.iq, rtol=1e-6, atol=1e-6)

    rotated = synthesize_scatterer_echoes(
        positions,
        reflectivity * np.exp(0.73j),
        np.arange(130) / 1300.0,
        d2a_config,
        platform_position_xyz_m=platform,
        platform_velocity_xyz_mps=velocity,
    )
    np.testing.assert_allclose(rotated.iq, response.iq * np.exp(0.73j), rtol=1e-6, atol=1e-6)


def test_t11_platform_motion_produces_physical_range_migration(d2a_config):
    response = _response(d2a_config, azimuth_deg=0.0, platform_speed_mps=60.0)
    cell = response.range_cell[1, :, 0]
    expected_migration = -(60.0 * (129 / 1300.0)) / d2a_config["waveform"]["range_cell_m"]
    assert cell[-1] - cell[0] == pytest.approx(expected_migration, rel=0.02)
    assert abs(cell[-1] - cell[0]) > 1.0


def test_t12_h0_and_h1_are_same_shape_and_share_real_background_not_shortcuts(d2a_config):
    batch = generate_d2a_paired_batch(2, 81012, 10.0, 5.0, d2a_config)
    assert batch.iq_h0.shape == batch.iq_h1.shape == (2, 3, 130, 32)
    assert np.any(batch.clutter_iq != 0)
    assert np.any(batch.noise_iq != 0)
    assert np.all(batch.truth_presence_h0 == 0)
    assert np.all(batch.truth_presence_h1 == 1)
    np.testing.assert_allclose(batch.iq_h0, batch.clutter_iq + batch.noise_iq, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(batch.iq_h1, batch.iq_h0 + batch.target_signal_iq, rtol=1e-6, atol=1e-6)

    exact = generate_d2a_paired_batch(1, 81013, 10.0, 5.0, d2a_config, scenario_id="exact_overlap")
    assert exact.truth_target_radial_velocity_mps[0] == 0.0
    expected_target = synthesize_scatterer_echoes(
        exact.scatterer_positions_xyz_m[0, :1],
        np.asarray([exact.truth_target_amplitude[0] * np.exp(1j * exact.truth_target_phase_rad[0])]),
        np.arange(130) / 1300.0,
        d2a_config,
        platform_position_xyz_m=np.asarray(d2a_config["platform"]["initial_position_xyz_m"]),
        platform_velocity_xyz_mps=np.asarray(d2a_config["platform"]["velocity_xyz_mps"]),
    )
    np.testing.assert_allclose(exact.target_signal_iq[0], expected_target.iq, rtol=1e-6, atol=1e-6)
    assert np.isfinite(exact.truth_target_range_cell[0])
    assert 0.0 <= exact.truth_target_azimuth_deg[0] <= 180.0
