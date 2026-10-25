from __future__ import annotations

import numpy as np
import pytest
import yaml

from radardistill.d2a_baselines import (
    dpca_difference,
    run_b0,
    run_b1,
    run_b2,
    run_b3,
    steering_vector,
)
from radardistill.simulation.d2a import synthesize_scatterer_echoes


@pytest.fixture(scope="module")
def d2a_config():
    with open("experiments/D2-A-E001/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _edge_clutter(d2a_config):
    platform = np.asarray(d2a_config["platform"]["initial_position_xyz_m"], dtype=np.float64)
    slant_range = (
        d2a_config["range_grid"]["axis_start_m"]
        + 12.25 * d2a_config["waveform"]["range_cell_m"]
    )
    horizontal_range = np.sqrt(slant_range**2 - platform[2] ** 2)
    azimuth = np.deg2rad(88.86)
    position = platform + np.asarray(
        [horizontal_range * np.cos(azimuth), horizontal_range * np.sin(azimuth), -platform[2]]
    )
    return np.asarray([position]), np.asarray([0.7 - 0.2j]), platform, np.asarray([60.0, 0.0, 0.0])


def test_b0_recovers_doppler_and_range_peak(d2a_config):
    prf = d2a_config["radar"]["prf_hz"]
    n = d2a_config["radar"]["pulse_count"]
    f = 120.0
    iq = np.zeros((3, n, 32), dtype=np.complex64)
    iq[0, :, 11] = np.exp(2j * np.pi * f * np.arange(n) / prf)
    result = run_b0(iq, prf)
    assert result.doppler_hz == pytest.approx(f)
    assert result.range_cell == 11
    assert result.score > 0.0


def test_b1_is_invariant_to_independent_channel_phase_rotations(d2a_config):
    prf = d2a_config["radar"]["prf_hz"]
    rng = np.random.default_rng(82002)
    iq = (rng.standard_normal((3, 130, 32)) + 1j * rng.standard_normal((3, 130, 32))).astype(np.complex64)
    phases = np.exp(1j * np.asarray([0.1, 1.3, -2.4]))
    first = run_b1(iq, prf)
    second = run_b1(iq * phases[:, None, None], prf)
    assert second.score == pytest.approx(first.score, rel=1e-6)


def test_b2_uses_geometry_steering_and_recovers_look_direction(d2a_config):
    prf = d2a_config["radar"]["prf_hz"]
    azimuth = 90.7
    elevation = d2a_config["beam"]["center_elevation_deg"]
    steering = steering_vector(azimuth, elevation, d2a_config)
    n = d2a_config["radar"]["pulse_count"]
    pulse = np.arange(n)
    iq = np.zeros((3, n, 32), dtype=np.complex64)
    iq[:, :, 14] = steering[:, None] * np.exp(2j * np.pi * 80.0 * pulse / prf)[None, :]
    result = run_b2(iq, prf, d2a_config)
    assert result.angle_deg == pytest.approx(azimuth, abs=0.15)
    assert result.doppler_hz == pytest.approx(80.0)
    assert result.range_cell == 14


def test_b3_dpca_suppresses_static_clutter_and_retains_separated_target(d2a_config):
    pulse_times = np.arange(130) / d2a_config["radar"]["prf_hz"]
    positions, reflectivity, platform, velocity = _edge_clutter(d2a_config)
    clutter = synthesize_scatterer_echoes(
        positions, reflectivity, pulse_times, d2a_config,
        platform_position_xyz_m=platform, platform_velocity_xyz_mps=velocity,
    ).iq
    clutter_diff = dpca_difference(clutter, d2a_config)
    clutter_retention = np.mean(np.abs(clutter_diff) ** 2) / np.mean(np.abs(clutter[0]) ** 2)
    assert clutter_retention < 0.25

    target_position = positions[0]
    slant_range = d2a_config["range_grid"]["axis_start_m"] + 12.25 * d2a_config["waveform"]["range_cell_m"]
    horizontal_range = np.linalg.norm(target_position[:2] - platform[:2])
    target_velocity = np.asarray([0.0, 4.0 * slant_range / horizontal_range, 0.0])
    target = synthesize_scatterer_echoes(
        positions, reflectivity, pulse_times, d2a_config,
        platform_position_xyz_m=platform, platform_velocity_xyz_mps=velocity,
        scatterer_velocity_xyz_mps=target_velocity[None, :],
    ).iq
    target_diff = dpca_difference(target, d2a_config)
    target_retention = np.mean(np.abs(target_diff) ** 2) / np.mean(np.abs(target[0]) ** 2)
    assert target_retention > 0.5
    assert run_b3(target, d2a_config["radar"]["prf_hz"], d2a_config).score > 0.0


def test_dpca_rejects_geometry_without_along_track_baseline(d2a_config):
    invalid = {**d2a_config, "channels": {**d2a_config["channels"], "positions_xyz_m": [[0, 0, 0]] * 3}}
    with pytest.raises(ValueError, match="along-track"):
        dpca_difference(np.ones((3, 130, 32), dtype=np.complex64), invalid)
