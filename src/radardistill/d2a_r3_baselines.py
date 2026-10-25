"""Predeclared scale-normalized B3 statistics for D2-A-R3 only."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from radardistill.d2a_baselines import dpca_difference


@dataclass(frozen=True)
class R3DetectionOutput:
    """Keep dimensionless detection statistics separate from raw RD power."""

    score: np.ndarray | float
    score_map: np.ndarray
    raw_rd_power: np.ndarray
    background_estimate: np.ndarray
    doppler_hz: np.ndarray | float
    range_cell: np.ndarray | float


def _as_batch(iq: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(iq)
    scalar = values.ndim == 3
    if scalar:
        values = values[None, ...]
    if (
        values.ndim != 4
        or values.shape[0] == 0
        or values.shape[1] < 2
        or values.shape[2] < 2
        or values.shape[3] < 1
    ):
        raise ValueError("iq must have shape (C,N,R) or (B,C,N,R) with nonempty dimensions")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    return values.astype(np.complex128, copy=False), scalar


def _b3_power_map(iq: np.ndarray, config: dict[str, Any]) -> tuple[np.ndarray, bool]:
    values, scalar = _as_batch(iq)
    residual = dpca_difference(values, config)
    spectrum = np.fft.fftshift(np.fft.fft(residual, axis=1), axes=1)
    power = np.abs(spectrum) ** 2
    if np.any(~np.isfinite(power)):
        raise ValueError("B3 RD power must be finite")
    return power.astype(np.float64, copy=False), scalar


def _output_from_map(
    score_map: np.ndarray,
    raw_rd_power: np.ndarray,
    background_estimate: np.ndarray,
    scalar: bool,
    prf_hz: float,
) -> R3DetectionOutput:
    if not np.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    if score_map.shape != raw_rd_power.shape or score_map.shape != background_estimate.shape:
        raise ValueError("score, raw-power, and background maps must have equal shapes")
    if score_map.ndim != 3 or np.any(~np.isfinite(score_map)):
        raise ValueError("score_map must be a finite (B,N,R) array")
    flat_peak = np.argmax(score_map.reshape(score_map.shape[0], -1), axis=1)
    range_count = score_map.shape[2]
    doppler_index = flat_peak // range_count
    range_index = flat_peak % range_count
    score = score_map[np.arange(score_map.shape[0]), doppler_index, range_index]
    frequencies = np.fft.fftshift(np.fft.fftfreq(score_map.shape[1], d=1.0 / prf_hz))
    doppler = frequencies[doppler_index]
    if scalar:
        return R3DetectionOutput(
            score=float(score[0]),
            score_map=score_map[0],
            raw_rd_power=raw_rd_power[0],
            background_estimate=background_estimate[0],
            doppler_hz=float(doppler[0]),
            range_cell=float(range_index[0]),
        )
    return R3DetectionOutput(
        score=score,
        score_map=score_map,
        raw_rd_power=raw_rd_power,
        background_estimate=background_estimate,
        doppler_hz=doppler,
        range_cell=range_index.astype(np.float64),
    )


def _t1_background(power: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    doppler_radius = int(options["peak_guard_doppler_radius_cells"])
    range_radius = int(options["peak_guard_range_radius_cells"])
    if doppler_radius < 0 or range_radius < 0:
        raise ValueError("T1 guard radii must be nonnegative")
    if options["doppler_boundary"] != "circular" or options["range_boundary"] != "clipped":
        raise ValueError("T1 requires circular Doppler and clipped range boundaries")
    backgrounds = np.empty(power.shape[0], dtype=np.float64)
    doppler_count, range_count = power.shape[1:]
    for scene_index, scene_power in enumerate(power):
        peak_doppler, peak_range = np.unravel_index(np.argmax(scene_power), scene_power.shape)
        mask = np.ones(scene_power.shape, dtype=bool)
        doppler_indices = (
            peak_doppler + np.arange(-doppler_radius, doppler_radius + 1)
        ) % doppler_count
        range_indices = np.arange(
            max(0, peak_range - range_radius), min(range_count, peak_range + range_radius + 1)
        )
        mask[np.ix_(doppler_indices, range_indices)] = False
        backgrounds[scene_index] = np.median(scene_power[mask])
    if np.any(~np.isfinite(backgrounds)) or np.any(backgrounds <= 0.0):
        raise ValueError("T1 robust background power must be positive and finite")
    return backgrounds


def run_t1_global_normalized(
    iq: np.ndarray,
    prf_hz: float,
    config: dict[str, Any],
    options: dict[str, Any],
) -> R3DetectionOutput:
    """Normalize every B3 cell by a robust scene background median."""
    power, scalar = _b3_power_map(iq, config)
    background = _t1_background(power, options)
    background_map = np.broadcast_to(background[:, None, None], power.shape).copy()
    score_map = power / background_map
    return _output_from_map(score_map, power, background_map, scalar, prf_hz)


def _local_median_background(power: np.ndarray, options: dict[str, Any]) -> np.ndarray:
    d_width = int(options["doppler_window_width_cells"])
    r_width = int(options["range_window_width_cells"])
    d_guard = int(options["doppler_guard_width_cells"])
    r_guard = int(options["range_guard_width_cells"])
    min_references = int(options["min_reference_cells"])
    if any(width <= 0 or width % 2 == 0 for width in (d_width, r_width, d_guard, r_guard)):
        raise ValueError("T2 window and guard widths must be positive odd integers")
    if d_guard >= d_width or r_guard >= r_width:
        raise ValueError("T2 guard must be narrower than the full reference window")
    if min_references < 1:
        raise ValueError("T2 min_reference_cells must be positive")
    if options["doppler_boundary"] != "circular" or options["range_boundary"] != "clipped":
        raise ValueError("T2 requires circular Doppler and clipped range boundaries")

    d_radius = d_width // 2
    r_radius = r_width // 2
    d_guard_radius = d_guard // 2
    r_guard_radius = r_guard // 2
    shifted_training: list[np.ndarray] = []
    for d_offset in range(-d_radius, d_radius + 1):
        for r_offset in range(-r_radius, r_radius + 1):
            if abs(d_offset) <= d_guard_radius and abs(r_offset) <= r_guard_radius:
                continue
            shifted = np.roll(power, shift=-d_offset, axis=1)
            shifted = np.roll(shifted, shift=-r_offset, axis=2)
            if r_offset > 0:
                shifted[:, :, -r_offset:] = np.nan
            elif r_offset < 0:
                shifted[:, :, :-r_offset] = np.nan
            shifted_training.append(shifted)
    training = np.stack(shifted_training, axis=-1)
    reference_count = np.sum(np.isfinite(training), axis=-1)
    if np.any(reference_count < min_references):
        raise ValueError("T2 has fewer than min_reference_cells at one or more range cells")
    with np.errstate(invalid="ignore"):
        background = np.nanmedian(training, axis=-1)
    if np.any(~np.isfinite(background)) or np.any(background <= 0.0):
        raise ValueError("T2 local reference median must be positive and finite")
    return background


def run_t2_local_cfar(
    iq: np.ndarray,
    prf_hz: float,
    config: dict[str, Any],
    options: dict[str, Any],
) -> R3DetectionOutput:
    """Apply the preregistered local median CFAR-like statistic to B3 power."""
    power, scalar = _b3_power_map(iq, config)
    background = _local_median_background(power, options)
    score_map = power / background
    return _output_from_map(score_map, power, background, scalar, prf_hz)
