"""Post-Doppler three-channel ACE/AMF teacher for D2-A-T001.

The covariance bank is fit from H0 observable IQ only. Runtime inference takes
IQ, that bank, and the public frozen geometry/configuration; it has no labels or
scene metadata in its API.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from radardistill.d2a_baselines import steering_vector


@dataclass(frozen=True)
class CovarianceBank:
    covariance: np.ndarray
    inverse: np.ndarray
    eigenvalues: np.ndarray
    condition_number: np.ndarray
    eigenvalue_floor_fraction: np.ndarray
    scene_count: int
    pulse_count: int
    channel_count: int
    oversampling_factor: int

    def diagnostics(self) -> dict[str, Any]:
        return {
            "scene_count": int(self.scene_count),
            "pulse_count": int(self.pulse_count),
            "channel_count": int(self.channel_count),
            "oversampling_factor": int(self.oversampling_factor),
            "eigenvalue_min_by_bin": np.min(self.eigenvalues, axis=1).tolist(),
            "eigenvalue_max_by_bin": np.max(self.eigenvalues, axis=1).tolist(),
            "condition_number_by_bin": self.condition_number.tolist(),
            "eigenvalue_floor_fraction_by_bin": self.eigenvalue_floor_fraction.tolist(),
        }


@dataclass(frozen=True)
class ACEAMFOutput:
    score: np.ndarray | float
    doppler_hz: np.ndarray | float
    range_cell: np.ndarray | float
    angle_deg: np.ndarray | float
    score_map: np.ndarray
    spatial_coherence_map: np.ndarray
    soft_response: np.ndarray


def _as_iq_batch(iq: np.ndarray, *, expected_channels: int | None = None) -> tuple[np.ndarray, bool]:
    values = np.asarray(iq)
    scalar = values.ndim == 3
    if scalar:
        values = values[None, ...]
    if values.ndim != 4 or min(values.shape) <= 0:
        raise ValueError("iq must have shape (C,N,R) or (B,C,N,R) with nonempty dimensions")
    if expected_channels is not None and values.shape[1] != expected_channels:
        raise ValueError("IQ channel count does not match the covariance bank")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    return values.astype(np.complex128, copy=False), scalar


def fit_post_doppler_covariance(
    h0_iq_batches: Iterable[np.ndarray],
    *,
    oversampling_factor: int = 4,
    eigenvalue_floor_relative_trace: float = 1e-8,
) -> CovarianceBank:
    """Fit a per-Doppler 3-channel covariance from streamed H0 IQ batches.

    Each CPI contributes equally after its observable mean power normalization;
    all range cells contribute equally within that CPI. No truth or simulation
    sidecar is accepted by this function.
    """
    if oversampling_factor <= 0:
        raise ValueError("oversampling_factor must be positive")
    if not np.isfinite(eigenvalue_floor_relative_trace) or eigenvalue_floor_relative_trace <= 0.0:
        raise ValueError("eigenvalue_floor_relative_trace must be positive and finite")

    covariance_sum: np.ndarray | None = None
    scene_count = 0
    expected_shape: tuple[int, int, int] | None = None
    for raw_batch in h0_iq_batches:
        values, _ = _as_iq_batch(raw_batch, expected_channels=3)
        _, channels, pulse_count, range_count = values.shape
        if expected_shape is None:
            expected_shape = (channels, pulse_count, range_count)
            covariance_sum = np.zeros(
                (pulse_count * oversampling_factor, channels, channels), dtype=np.complex128
            )
        elif expected_shape != (channels, pulse_count, range_count):
            raise ValueError("all H0 IQ batches must have the same (C,N,R) shape")

        scene_energy = np.mean(np.abs(values) ** 2, axis=(1, 2, 3))
        if np.any(~np.isfinite(scene_energy)) or np.any(scene_energy <= 0.0):
            raise ValueError("each covariance-training CPI must have positive finite observable power")
        spectrum = np.fft.fftshift(
            np.fft.fft(values, n=pulse_count * oversampling_factor, axis=2), axes=2
        )
        normalized = spectrum / np.sqrt(scene_energy)[:, None, None, None]
        assert covariance_sum is not None
        covariance_sum += np.einsum(
            "bckr,bdkr->kcd", normalized, normalized.conj(), optimize=True
        ) / range_count
        scene_count += values.shape[0]

    if scene_count == 0 or expected_shape is None or covariance_sum is None:
        raise ValueError("at least one nonempty H0 IQ batch is required")

    channels, pulse_count, _ = expected_shape
    covariance = covariance_sum / scene_count
    covariance = 0.5 * (covariance + covariance.conj().transpose(0, 2, 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    trace = np.real(np.trace(covariance, axis1=1, axis2=2))
    if np.any(~np.isfinite(trace)) or np.any(trace <= 0.0):
        raise ValueError("every Doppler covariance bin must have positive finite trace")
    floor = eigenvalue_floor_relative_trace * trace / channels
    floored_values = np.maximum(eigenvalues, floor[:, None])
    floor_fraction = np.mean(eigenvalues < floor[:, None], axis=1)
    covariance_regularized = np.einsum(
        "kci,ki,kdi->kcd", eigenvectors, floored_values, eigenvectors.conj(), optimize=True
    )
    inverse = np.einsum(
        "kci,ki,kdi->kcd", eigenvectors, 1.0 / floored_values, eigenvectors.conj(), optimize=True
    )
    covariance_regularized = 0.5 * (
        covariance_regularized + covariance_regularized.conj().transpose(0, 2, 1)
    )
    inverse = 0.5 * (inverse + inverse.conj().transpose(0, 2, 1))
    condition_number = np.max(floored_values, axis=1) / np.min(floored_values, axis=1)
    if any(np.any(~np.isfinite(item)) for item in (covariance_regularized, inverse, floored_values, condition_number)):
        raise FloatingPointError("covariance fitting produced a non-finite matrix or diagnostic")
    return CovarianceBank(
        covariance=covariance_regularized,
        inverse=inverse,
        eigenvalues=floored_values,
        condition_number=condition_number,
        eigenvalue_floor_fraction=floor_fraction,
        scene_count=scene_count,
        pulse_count=pulse_count,
        channel_count=channels,
        oversampling_factor=oversampling_factor,
    )


def _parabolic_offset(left: float, center: float, right: float) -> float:
    denominator = left - 2.0 * center + right
    if not np.isfinite(denominator) or denominator >= -1e-15:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))


def _parameter_estimates(
    score_maps: np.ndarray,
    angle_scores: np.ndarray,
    angle_grid_deg: np.ndarray,
    prf_hz: float,
    oversampling_factor: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_count, frequency_count, range_count = score_maps.shape
    flat_peak = np.argmax(score_maps.reshape(batch_count, -1), axis=1)
    frequency_index = flat_peak // range_count
    range_index = flat_peak % range_count
    row = np.arange(batch_count)
    peak_score = score_maps[row, frequency_index, range_index]
    angle_index = np.argmax(angle_scores[row, :, frequency_index, range_index], axis=1)
    angle = angle_grid_deg[angle_index]

    frequency_values = np.fft.fftshift(
        np.fft.fftfreq(frequency_count, d=1.0 / prf_hz)
    )
    doppler_delta = np.zeros(batch_count, dtype=np.float64)
    for index in range(batch_count):
        k = int(frequency_index[index])
        doppler_delta[index] = _parabolic_offset(
            float(score_maps[index, (k - 1) % frequency_count, range_index[index]]),
            float(score_maps[index, k, range_index[index]]),
            float(score_maps[index, (k + 1) % frequency_count, range_index[index]]),
        )
    doppler = frequency_values[frequency_index] + doppler_delta * prf_hz / frequency_count
    doppler = np.remainder(doppler + prf_hz / 2.0, prf_hz) - prf_hz / 2.0

    range_values = range_index.astype(np.float64)
    for index in range(batch_count):
        r = int(range_index[index])
        if 0 < r < range_count - 1:
            range_values[index] += _parabolic_offset(
                float(score_maps[index, frequency_index[index], r - 1]),
                float(score_maps[index, frequency_index[index], r]),
                float(score_maps[index, frequency_index[index], r + 1]),
            )
    return peak_score, doppler, range_values, angle


def post_doppler_ace_amf(
    iq: np.ndarray,
    covariance_bank: CovarianceBank,
    physics_config: dict[str, Any],
    teacher_config: dict[str, Any],
) -> ACEAMFOutput:
    """Return scale-normalized post-Doppler ACE×AMF score and K2-shaped map."""
    values, scalar = _as_iq_batch(iq, expected_channels=covariance_bank.channel_count)
    batch_count, channels, pulse_count, range_count = values.shape
    oversampling_factor = int(teacher_config["frequency_oversampling_factor"])
    frequency_count = pulse_count * oversampling_factor
    if (
        covariance_bank.pulse_count != pulse_count
        or covariance_bank.oversampling_factor != oversampling_factor
        or covariance_bank.covariance.shape != (frequency_count, channels, channels)
        or covariance_bank.inverse.shape != (frequency_count, channels, channels)
    ):
        raise ValueError("covariance bank dimensions do not match IQ and frozen frequency grid")
    prf_hz = float(physics_config["radar"]["prf_hz"])
    if not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("physics_config radar PRF must be positive and finite")
    angle_grid = np.asarray(teacher_config["angle_grid_deg"], dtype=np.float64)
    expected_angles = int(teacher_config["angle_grid_count"])
    if angle_grid.shape != (expected_angles,) or np.any(~np.isfinite(angle_grid)):
        raise ValueError("teacher angle grid must match its frozen finite grid count")
    if np.any(~np.isfinite(covariance_bank.inverse)):
        raise ValueError("covariance bank inverse must be finite")

    observable_power = np.mean(np.abs(values) ** 2, axis=(1, 2, 3))
    spectrum = np.fft.fftshift(
        np.fft.fft(values, n=frequency_count, axis=2), axes=2
    ).transpose(0, 2, 3, 1)
    inverse_y = np.einsum(
        "kcd,bkrd->bkrc", covariance_bank.inverse, spectrum, optimize=True
    )
    whitened_energy = np.real(np.einsum("bkrc,bkrc->bkr", spectrum.conj(), inverse_y, optimize=True))
    whitened_energy = np.maximum(whitened_energy, 0.0)
    manifolds = np.stack(
        [
            steering_vector(
                float(azimuth),
                float(physics_config["beam"]["center_elevation_deg"]),
                physics_config,
            )
            for azimuth in angle_grid
        ],
        axis=0,
    )
    numerator = np.einsum(
        "ac,kcd,bkrd->bakr",
        manifolds.conj(),
        covariance_bank.inverse,
        spectrum,
        optimize=True,
    )
    steering_energy = np.real(
        np.einsum(
            "ac,kcd,ad->ak", manifolds.conj(), covariance_bank.inverse, manifolds, optimize=True
        )
    )
    epsilon = float(teacher_config["ace_denominator_epsilon"])
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("ace_denominator_epsilon must be positive and finite")
    numerator_power = np.abs(numerator) ** 2
    valid_steering = steering_energy > epsilon
    valid_energy = whitened_energy > epsilon
    ace_denominator = steering_energy[None, :, :, None] * whitened_energy[:, None, :, :]
    spatial_ace = np.divide(
        numerator_power,
        ace_denominator,
        out=np.zeros_like(numerator_power, dtype=np.float64),
        where=valid_steering[None, :, :, None] & valid_energy[:, None, :, :],
    )
    diagnostic_clip = teacher_config.get("ace_diagnostic_clip", [0.0, 1.0])
    if len(diagnostic_clip) != 2:
        raise ValueError("ace_diagnostic_clip must contain lower and upper limits")
    spatial_ace = np.clip(spatial_ace, float(diagnostic_clip[0]), float(diagnostic_clip[1]))

    matched_energy = np.divide(
        numerator_power,
        steering_energy[None, :, :, None],
        out=np.zeros_like(numerator_power, dtype=np.float64),
        where=valid_steering[None, :, :, None],
    )
    normalized_energy = np.divide(
        matched_energy,
        observable_power[:, None, None, None],
        out=np.zeros_like(matched_energy, dtype=np.float64),
        where=observable_power[:, None, None, None] > 0.0,
    )
    # Numerically, normalized_energy is ACE * whitened_energy / e_CPI.
    # Keeping the algebraically equivalent AMF form avoids a 0/0 at empty bins.
    score_by_angle = np.maximum(normalized_energy, 0.0)
    score_maps = np.max(score_by_angle, axis=1)
    coherence_maps = np.max(spatial_ace, axis=1)
    peak_score, doppler, range_cell, angle = _parameter_estimates(
        score_maps, score_by_angle, angle_grid, prf_hz, oversampling_factor
    )

    native_map = score_maps[:, ::oversampling_factor, :].copy()
    totals = np.sum(native_map, axis=(1, 2), keepdims=True)
    soft_response = np.divide(
        native_map,
        totals,
        out=np.zeros_like(native_map),
        where=totals > 0.0,
    )
    if any(np.any(~np.isfinite(item)) for item in (score_maps, coherence_maps, soft_response, peak_score, doppler, range_cell, angle)):
        raise FloatingPointError("post-Doppler ACE/AMF produced non-finite outputs")

    if scalar:
        return ACEAMFOutput(
            score=float(peak_score[0]),
            doppler_hz=float(doppler[0]),
            range_cell=float(range_cell[0]),
            angle_deg=float(angle[0]),
            score_map=score_maps[0],
            spatial_coherence_map=coherence_maps[0],
            soft_response=soft_response[0],
        )
    return ACEAMFOutput(
        score=peak_score,
        doppler_hz=doppler,
        range_cell=range_cell,
        angle_deg=angle,
        score_map=score_maps,
        spatial_coherence_map=coherence_maps,
        soft_response=soft_response,
    )
