"""Classical range-Doppler detection baselines for D2-A."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DetectionOutput:
    score: np.ndarray | float
    doppler_hz: np.ndarray | float
    range_cell: np.ndarray | float
    angle_deg: np.ndarray | float | None
    rd_power: np.ndarray


def _as_batch(iq: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(iq)
    scalar = values.ndim == 3
    if scalar:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[0] == 0 or values.shape[1] < 1 or values.shape[2] < 2 or values.shape[3] < 1:
        raise ValueError("iq must have shape (C,N,R) or (B,C,N,R) with nonempty dimensions")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    return values.astype(np.complex128, copy=False), scalar


def _frequency_axis(pulse_count: int, prf_hz: float) -> np.ndarray:
    if not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    return np.fft.fftshift(np.fft.fftfreq(pulse_count, d=1.0 / prf_hz))


def _output_from_rd(
    rd_power: np.ndarray,
    scalar: bool,
    prf_hz: float,
    angle_power: np.ndarray | None = None,
    angle_grid_deg: np.ndarray | None = None,
) -> DetectionOutput:
    values = np.asarray(rd_power, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("rd_power must have shape (B,N,R)")
    flat_peak = np.argmax(values.reshape(values.shape[0], -1), axis=1)
    range_count = values.shape[2]
    doppler_index = flat_peak // range_count
    range_index = flat_peak % range_count
    score = values[np.arange(values.shape[0]), doppler_index, range_index]
    frequencies = _frequency_axis(values.shape[1], prf_hz)
    doppler = frequencies[doppler_index]

    angle: np.ndarray | None = None
    if angle_power is not None:
        if angle_grid_deg is None:
            raise ValueError("angle_grid_deg is required with angle_power")
        full_peak = np.argmax(angle_power.reshape(angle_power.shape[0], -1), axis=1)
        per_angle = angle_power.shape[1]
        cells_per_angle = angle_power.shape[2] * angle_power.shape[3]
        angle_index = full_peak // cells_per_angle
        angle = np.asarray(angle_grid_deg, dtype=np.float64)[angle_index]
        per_angle_peak = np.max(angle_power, axis=(2, 3))
        score = per_angle_peak[np.arange(values.shape[0]), angle_index]

    if scalar:
        return DetectionOutput(
            score=float(score[0]),
            doppler_hz=float(doppler[0]),
            range_cell=float(range_index[0]),
            angle_deg=None if angle is None else float(angle[0]),
            rd_power=values[0],
        )
    return DetectionOutput(
        score=score,
        doppler_hz=doppler,
        range_cell=range_index.astype(np.float64),
        angle_deg=angle,
        rd_power=values,
    )


def _range_doppler_power(iq: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(iq, axis=2), axes=2)
    return np.abs(spectrum) ** 2


def run_b0(iq: np.ndarray, prf_hz: float) -> DetectionOutput:
    """Single-channel rectangular slow-time FFT maximum-power detector."""
    values, scalar = _as_batch(iq)
    power = _range_doppler_power(values[:, :1])[:, 0]
    _frequency_axis(power.shape[1], prf_hz)
    return _output_from_rd(power, scalar, prf_hz)


def run_b1(iq: np.ndarray, prf_hz: float) -> DetectionOutput:
    """Three-channel noncoherent sum-power range-Doppler detector."""
    values, scalar = _as_batch(iq)
    power = np.sum(_range_doppler_power(values), axis=1)
    _frequency_axis(power.shape[1], prf_hz)
    return _output_from_rd(power, scalar, prf_hz)


def steering_vector(azimuth_deg: float, elevation_deg: float, config: dict[str, Any]) -> np.ndarray:
    """Return the far-field receive-channel manifold under D2-A's path sign."""
    positions = np.asarray(config["channels"]["positions_xyz_m"], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("channels.positions_xyz_m must have shape (C,3)")
    wavelength = float(config["waveform"]["propagation_speed_mps"]) / float(
        config["waveform"]["carrier_hz"]
    )
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    look = np.asarray(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)]
    )
    relative_positions = positions - positions.mean(axis=0, keepdims=True)
    # For R_rx ~= R - b dot look, exp(-j 2 pi R_rx/lambda) has +j spatial phase.
    return np.exp(2j * np.pi * (relative_positions @ look) / wavelength)


def run_b2(
    iq: np.ndarray,
    prf_hz: float,
    config: dict[str, Any],
    azimuth_grid_deg: np.ndarray | None = None,
) -> DetectionOutput:
    """Geometry-steered coherent beam search followed by RD maximum power."""
    values, scalar = _as_batch(iq)
    if values.shape[1] != int(config["radar"]["channel_count"]):
        raise ValueError("B2 input channel count does not match its geometry")
    if azimuth_grid_deg is None:
        center = float(config["beam"]["center_azimuth_deg"])
        width = float(config["beam"]["azimuth_beamwidth_deg"])
        count = int(config["baselines"]["B2_azimuth_grid_count"])
        azimuth_grid = np.linspace(center - width / 2.0, center + width / 2.0, count)
    else:
        azimuth_grid = np.asarray(azimuth_grid_deg, dtype=np.float64)
    if azimuth_grid.ndim != 1 or azimuth_grid.size == 0 or np.any(~np.isfinite(azimuth_grid)):
        raise ValueError("azimuth_grid_deg must be a nonempty finite vector")

    elevation = float(config["beam"]["center_elevation_deg"])
    manifolds = np.stack(
        [steering_vector(az, elevation, config) for az in azimuth_grid], axis=0
    )
    beamformed = np.einsum("ac,bcnr->banr", np.conj(manifolds), values, optimize=True)
    beamformed /= values.shape[1]
    rd = np.fft.fftshift(np.fft.fft(beamformed, axis=2), axes=2)
    angle_power = np.abs(rd) ** 2
    rd_power = np.max(angle_power, axis=1)
    return _output_from_rd(rd_power, scalar, prf_hz, angle_power, azimuth_grid)


def dpca_time_shift_s(config: dict[str, Any], channel_pair: tuple[int, int] = (0, 1)) -> float:
    """Return continuous delay aligning the two receive effective phase centers."""
    positions = np.asarray(config["channels"]["positions_xyz_m"], dtype=np.float64)
    velocity = np.asarray(config["platform"]["velocity_xyz_mps"], dtype=np.float64)
    first, second = (int(index) for index in channel_pair)
    if min(first, second) < 0 or max(first, second) >= len(positions) or first == second:
        raise ValueError("channel_pair must contain two distinct configured channel indices")
    speed_squared = float(np.dot(velocity, velocity))
    baseline = positions[second] - positions[first]
    along_track = float(np.dot(baseline, velocity))
    if speed_squared <= 0.0 or abs(along_track) <= 1e-12:
        raise ValueError("DPCA geometry requires a nonzero along-track baseline and platform speed")
    # With Tx at a common platform phase center, the effective receive phase
    # center is halfway from Tx to Rx, hence the factor 1/2.
    return -along_track / (2.0 * speed_squared)


def dpca_difference(
    iq: np.ndarray,
    config: dict[str, Any],
    channel_pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Apply geometry-derived fractional slow-time shift and channel difference."""
    values, scalar = _as_batch(iq)
    first, second = (int(index) for index in channel_pair)
    if max(first, second) >= values.shape[1] or min(first, second) < 0:
        raise ValueError("channel_pair is outside the IQ channel dimension")
    pulse_count = values.shape[2]
    prf_hz = float(config["radar"]["prf_hz"])
    shift_samples = dpca_time_shift_s(config, channel_pair) * prf_hz
    frequencies = np.fft.fftfreq(pulse_count)
    phase = np.exp(2j * np.pi * frequencies * shift_samples)
    second_spectrum = np.fft.fft(values[:, second], axis=1)
    shifted_second = np.fft.ifft(second_spectrum * phase[None, :, None], axis=1)
    difference = values[:, first] - shifted_second
    return difference[0] if scalar else difference


def run_b3(iq: np.ndarray, prf_hz: float, config: dict[str, Any]) -> DetectionOutput:
    """Candidate along-track DPCA detector, subject to the R1 validation gate."""
    values, scalar = _as_batch(iq)
    difference = dpca_difference(values, config)
    spectrum = np.fft.fftshift(np.fft.fft(difference, axis=1), axes=1)
    power = np.abs(spectrum) ** 2
    return _output_from_rd(power, scalar, prf_hz)
