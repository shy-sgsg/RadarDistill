from __future__ import annotations

import numpy as np
import pytest
import yaml

from radardistill.d2a_baselines import dpca_difference, dpca_time_shift_s
from radardistill.evaluation.d2a_r7_teacher_diagnostics import (
    dpca_component_diagnostics,
    summarize_component_diagnostics,
)
from radardistill.simulation.d2a import synthesize_scatterer_echoes


@pytest.fixture(scope="module")
def physics():
    with open("experiments/D2-A-E001/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _scatterer_geometry(config, count=1):
    platform = np.asarray(config["platform"]["initial_position_xyz_m"], dtype=np.float64)
    range_start = float(config["range_grid"]["axis_start_m"])
    cell_m = float(config["waveform"]["range_cell_m"])
    height = float(platform[2])
    cells = np.linspace(8.25, 24.25, count)
    angles = np.linspace(88.9, 91.1, count)
    positions = []
    for cell, angle in zip(cells, angles):
        slant = range_start + cell * cell_m
        horizontal = np.sqrt(slant**2 - height**2)
        theta = np.deg2rad(angle)
        positions.append(platform + [horizontal * np.cos(theta), horizontal * np.sin(theta), -height])
    return np.asarray(positions), np.ones(count, dtype=np.complex128)


def _noise(seed=9032):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal((3, 130, 32))
    imag = rng.standard_normal((3, 130, 32))
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def test_dpca_shift_matches_geometry_and_channel_swap_reverses_sign(physics):
    delay = dpca_time_shift_s(physics, (0, 1))
    baseline = np.asarray(physics["channels"]["positions_xyz_m"][1]) - np.asarray(
        physics["channels"]["positions_xyz_m"][0]
    )
    velocity = np.asarray(physics["platform"]["velocity_xyz_mps"])
    expected = -float(np.dot(baseline, velocity)) / (2.0 * float(np.dot(velocity, velocity)))
    assert delay == pytest.approx(expected)
    assert dpca_time_shift_s(physics, (1, 0)) == pytest.approx(-expected)


def test_single_static_scatterer_is_better_aligned_by_frozen_dpca_shift(physics):
    positions, reflectivity = _scatterer_geometry(physics)
    pulse_times = np.arange(130, dtype=np.float64) / float(physics["radar"]["prf_hz"])
    iq = synthesize_scatterer_echoes(positions, reflectivity, pulse_times, physics).iq
    result = dpca_component_diagnostics(iq, physics)
    assert result["rho_before"].shape == (1,)
    assert result["rho_after"].shape == (1,)
    assert abs(result["phase_residual_after_rad"][0]) < abs(
        result["phase_residual_before_rad"][0]
    )
    assert result["residual_power_ratio_aligned"][0] < result["residual_power_ratio_unaligned"][0]


def test_multiple_static_scatterers_preserve_finite_per_scene_diagnostics(physics):
    positions, reflectivity = _scatterer_geometry(physics, count=4)
    pulse_times = np.arange(130, dtype=np.float64) / float(physics["radar"]["prf_hz"])
    iq = synthesize_scatterer_echoes(positions, reflectivity, pulse_times, physics).iq
    result = dpca_component_diagnostics(iq, physics)
    summary = summarize_component_diagnostics(result, bootstrap_replicates=100, seed=12)
    assert summary["n_scenes"] == 1
    assert np.isfinite(summary["coherence_after_magnitude_median"])
    assert np.isfinite(summary["aligned_residual_power_ratio_median"])


def test_noise_only_residual_power_has_expected_two_channel_sum_scale(physics):
    iq = _noise()
    result = dpca_component_diagnostics(iq, physics)
    assert result["residual_power_ratio_aligned"][0] == pytest.approx(2.0, abs=0.1)
    assert np.isfinite(result["phase_residual_after_rad"][0])


def test_target_only_and_mixed_finite_cpi_are_measured_without_hard_gate(physics):
    positions, reflectivity = _scatterer_geometry(physics)
    pulse_times = np.arange(130, dtype=np.float64) / float(physics["radar"]["prf_hz"])
    target_velocity = np.asarray([[0.0, 4.0, 0.0]])
    target = synthesize_scatterer_echoes(
        positions,
        reflectivity,
        pulse_times,
        physics,
        scatterer_velocity_xyz_mps=target_velocity,
    ).iq
    noise = _noise(seed=9077)
    for signal in (target, target + noise):
        result = dpca_component_diagnostics(signal, physics)
        assert np.all(np.isfinite(result["residual_power_ratio_aligned"]))
        assert np.all(np.isfinite(result["rho_before"]))
        assert np.all(np.isfinite(result["rho_after"]))
        assert result["residual_power_ratio_aligned"].shape == (1,)


def test_component_diagnostics_match_frozen_difference_function(physics):
    iq = _noise(seed=9133)
    result = dpca_component_diagnostics(iq, physics)
    expected = dpca_difference(iq, physics)
    assert result["residual_power_ratio_aligned"][0] == pytest.approx(
        np.mean(np.abs(expected) ** 2) / np.mean(np.abs(iq[0]) ** 2)
    )
