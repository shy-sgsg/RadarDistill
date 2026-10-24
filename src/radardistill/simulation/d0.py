"""D0 three-channel range-compressed slow-time simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class D0Batch:
    iq: np.ndarray
    signal_iq: np.ndarray
    noise_iq: np.ndarray
    truth_presence: np.ndarray
    truth_range_cell: np.ndarray
    truth_doppler_hz: np.ndarray
    truth_initial_phase_rad: np.ndarray
    nuisance_initial_phase_rad: np.ndarray
    truth_target_amplitude: np.ndarray
    reference_amplitude: np.ndarray
    noise_power: np.ndarray
    snr_db: np.ndarray
    truth_azimuth_deg: np.ndarray
    truth_elevation_deg: np.ndarray
    nuisance_azimuth_deg: np.ndarray
    nuisance_elevation_deg: np.ndarray


def _config_value(config: dict[str, Any], section: str, key: str) -> Any:
    try:
        return config[section][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing D0 configuration value {section}.{key}") from exc


def _snr_values(count: int, snr_db: Any, rng: np.random.Generator) -> np.ndarray:
    if isinstance(snr_db, tuple):
        if len(snr_db) != 2:
            raise ValueError("an SNR interval must be a two-value tuple")
        low, high = (float(v) for v in snr_db)
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError("SNR interval endpoints must be finite and ordered")
        values = rng.uniform(low, high, size=count)
        return np.minimum(values, np.nextafter(high, low))

    if np.isscalar(snr_db):
        value = float(snr_db)
        if np.isnan(value) or value == -np.inf:
            raise ValueError("SNR must be finite or positive infinity for a noiseless sample")
        return np.full(count, value, dtype=np.float64)

    values = np.asarray(snr_db, dtype=np.float64)
    if values.shape != (count,):
        raise ValueError(f"per-sample SNR must have shape ({count},)")
    if np.any(np.isnan(values)) or np.any(np.isneginf(values)):
        raise ValueError("SNR values must be finite or positive infinity")
    return values


def _steering_batch(
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    positions = np.asarray(config["channels"]["positions_xyz_m"], dtype=np.float64)
    if positions.shape != (3, 3):
        raise ValueError("D0 requires exactly three configured receive phase centers")
    azimuth = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    elevation = np.deg2rad(np.asarray(elevation_deg, dtype=np.float64))
    look = np.column_stack(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    )
    wavelength = float(config["waveform"]["propagation_speed_mps"]) / float(
        config["waveform"]["carrier_hz"]
    )
    relative_positions = positions - positions.mean(axis=0, keepdims=True)
    path_projection = look @ relative_positions.T
    return np.exp(-2j * np.pi * path_projection / wavelength)


def channel_steering_vector(
    azimuth_deg: float,
    elevation_deg: float,
    config: dict[str, Any],
) -> np.ndarray:
    """Return the unit-amplitude APX15-derived three-channel far-field response."""
    az_range = config["channels"]["azimuth_deg_interval"]
    el_range = config["channels"]["elevation_deg_interval"]
    if not (float(az_range[0]) <= azimuth_deg <= float(az_range[1])):
        raise ValueError("azimuth is outside the frozen D0 nuisance interval")
    if not (float(el_range[0]) <= elevation_deg <= float(el_range[1])):
        raise ValueError("elevation is outside the frozen D0 nuisance interval")
    return _steering_batch(
        np.asarray([azimuth_deg]), np.asarray([elevation_deg]), config
    )[0]


def _target_signal_batch(
    range_cell: np.ndarray,
    doppler_hz: np.ndarray,
    initial_phase_rad: np.ndarray,
    amplitude: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    pulse_count = int(_config_value(config, "radar", "pulse_count"))
    range_count = int(_config_value(config, "radar", "range_bin_count"))
    prf_hz = float(_config_value(config, "radar", "prf_hz"))
    pulse_times = np.arange(pulse_count, dtype=np.float64) / prf_hz

    spatial = _steering_batch(azimuth_deg, elevation_deg, config)
    temporal = np.exp(
        2j * np.pi * np.asarray(doppler_hz, dtype=np.float64)[:, None] * pulse_times[None, :]
    )
    range_axis = np.arange(range_count, dtype=np.float64)
    range_psf = np.sinc(range_axis[None, :] - np.asarray(range_cell)[:, None])
    common = np.asarray(amplitude, dtype=np.float64) * np.exp(
        1j * np.asarray(initial_phase_rad, dtype=np.float64)
    )

    signal = (
        common[:, None, None, None]
        * spatial[:, :, None, None]
        * temporal[:, None, :, None]
        * range_psf[:, None, None, :]
    )
    return np.asarray(signal, dtype=np.complex64)


def synthesize_target_echo(
    range_cell: float,
    doppler_hz: float,
    initial_phase_rad: float,
    amplitude: float,
    azimuth_deg: float,
    elevation_deg: float,
    config: dict[str, Any],
) -> np.ndarray:
    """Synthesize one noiseless target term for fixed-boundary correctness checks."""
    range_interval = config["waveform"]["matched_filter_psf"]["target_range_cell_interval"]
    doppler_interval = config["radar"]["doppler_interval_hz"]
    if not (float(range_interval[0]) <= range_cell < float(range_interval[1])):
        raise ValueError("range_cell is outside the frozen D0 monitoring interval")
    if not (float(doppler_interval[0]) <= doppler_hz < float(doppler_interval[1])):
        raise ValueError("doppler_hz is outside the half-open D0 interval")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("amplitude must be positive and finite")
    return _target_signal_batch(
        np.asarray([range_cell]),
        np.asarray([doppler_hz]),
        np.asarray([initial_phase_rad]),
        np.asarray([amplitude]),
        np.asarray([azimuth_deg]),
        np.asarray([elevation_deg]),
        config,
    )[0]


def generate_complex_awgn(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    noise_power: float | np.ndarray,
) -> np.ndarray:
    """Draw circular complex Gaussian noise with E|w|^2 equal to noise_power."""
    if not shape or any(int(dim) <= 0 for dim in shape):
        raise ValueError("noise shape must contain positive dimensions")
    power = np.asarray(noise_power, dtype=np.float64)
    if power.ndim == 1 and len(shape) > 1 and power.shape == (shape[0],):
        power = power.reshape((shape[0],) + (1,) * (len(shape) - 1))
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise ValueError("noise power must be finite and non-negative")
    real = rng.standard_normal(shape, dtype=np.float32)
    imag = rng.standard_normal(shape, dtype=np.float32)
    return np.asarray((real + 1j * imag) * np.sqrt(power / 2.0), dtype=np.complex64)


def generate_d0_batch(
    count: int,
    seed: int,
    snr_db: float | tuple[float, float] | np.ndarray,
    presence_ratio: float,
    config: dict[str, Any],
) -> D0Batch:
    """Generate fixed-window H0/H1 IQ; all latent draws precede target gating."""
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0.0 <= presence_ratio <= 1.0:
        raise ValueError("presence_ratio must be between zero and one")

    rng = np.random.default_rng(int(seed))
    presence_draw = rng.random(count)
    range_interval = config["waveform"]["matched_filter_psf"]["target_range_cell_interval"]
    doppler_interval = config["radar"]["doppler_interval_hz"]
    azimuth_interval = config["channels"]["azimuth_deg_interval"]
    elevation_interval = config["channels"]["elevation_deg_interval"]

    latent_range = rng.uniform(*range_interval, size=count)
    latent_range = np.minimum(latent_range, np.nextafter(range_interval[1], range_interval[0]))
    latent_doppler = rng.uniform(*doppler_interval, size=count)
    latent_doppler = np.minimum(latent_doppler, np.nextafter(doppler_interval[1], doppler_interval[0]))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=count)
    amplitude_db = rng.uniform(*config["target_and_noise"]["target_amplitude_db_interval"], size=count)
    reference_amplitude = np.power(10.0, amplitude_db / 20.0)
    azimuth = rng.uniform(*azimuth_interval, size=count)
    elevation = rng.uniform(*elevation_interval, size=count)
    sampled_snr_db = _snr_values(count, snr_db, rng)
    noise_power = reference_amplitude**2 * np.power(10.0, -sampled_snr_db / 10.0)

    truth_presence = presence_draw < float(presence_ratio)
    signal = _target_signal_batch(
        latent_range,
        latent_doppler,
        phase,
        reference_amplitude,
        azimuth,
        elevation,
        config,
    )
    signal *= truth_presence[:, None, None, None]
    noise = generate_complex_awgn(
        rng,
        (
            count,
            int(config["radar"]["channel_count"]),
            int(config["radar"]["pulse_count"]),
            int(config["radar"]["range_bin_count"]),
        ),
        noise_power,
    )
    iq = np.asarray(signal + noise, dtype=np.complex64)

    def masked_truth(values: np.ndarray) -> np.ndarray:
        return np.where(truth_presence, values, np.nan).astype(np.float64)

    return D0Batch(
        iq=iq,
        signal_iq=signal,
        noise_iq=noise,
        truth_presence=truth_presence.astype(np.bool_),
        truth_range_cell=masked_truth(latent_range),
        truth_doppler_hz=masked_truth(latent_doppler),
        truth_initial_phase_rad=masked_truth(phase),
        nuisance_initial_phase_rad=phase.astype(np.float64),
        truth_target_amplitude=masked_truth(reference_amplitude),
        reference_amplitude=reference_amplitude.astype(np.float64),
        noise_power=noise_power.astype(np.float64),
        snr_db=sampled_snr_db.astype(np.float64),
        truth_azimuth_deg=masked_truth(azimuth),
        truth_elevation_deg=masked_truth(elevation),
        nuisance_azimuth_deg=azimuth.astype(np.float64),
        nuisance_elevation_deg=elevation.astype(np.float64),
    )


def verify_lfm_range_psf(config: dict[str, Any]) -> dict[str, float | int]:
    """Compare the finite sampled LFM matched-filter response with sinc magnitude."""
    waveform = config["waveform"]
    bandwidth_hz = float(waveform["bandwidth_hz"])
    sample_rate_hz = float(waveform["sample_rate_hz"])
    pulse_width_s = float(waveform["pulse_width_s"])
    sample_count = int(waveform["pulse_sample_count"])
    verifier = waveform["verifier"]
    offset_count = int(verifier["fractional_range_offsets_cells"])
    low, high = (float(v) for v in verifier["offset_interval_cells"])
    offsets = np.linspace(low, high, offset_count, dtype=np.float64)

    times = (np.arange(sample_count, dtype=np.float64) + 0.5) / sample_rate_hz
    chirp_rate_hz_per_s = bandwidth_hz / pulse_width_s
    reference = np.exp(1j * np.pi * chirp_rate_hz_per_s * (times - pulse_width_s / 2.0) ** 2)
    reference_energy = float(np.vdot(reference, reference).real)
    errors = np.empty(offset_count, dtype=np.float64)
    for index, offset_cells in enumerate(offsets):
        delay_s = float(offset_cells) / bandwidth_hz
        delayed_times = times - delay_s
        active = (delayed_times >= 0.0) & (delayed_times < pulse_width_s)
        delayed = np.zeros(sample_count, dtype=np.complex128)
        delayed[active] = np.exp(
            1j
            * np.pi
            * chirp_rate_hz_per_s
            * (delayed_times[active] - pulse_width_s / 2.0) ** 2
        )
        matched = np.vdot(reference, delayed) / reference_energy
        ideal_magnitude = abs(float(np.sinc(offset_cells)))
        errors[index] = abs(abs(matched) - ideal_magnitude)

    return {
        "offset_count": int(offset_count),
        "max_normalized_magnitude_error": float(np.max(errors)),
        "rmse_normalized_magnitude_error": float(np.sqrt(np.mean(errors**2))),
        "max_error_offset_cells": float(offsets[int(np.argmax(errors))]),
    }
