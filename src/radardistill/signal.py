"""Single-target complex slow-time signal simulator for P0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SignalBatch:
    iq: np.ndarray
    frequency_hz: np.ndarray
    phase_rad: np.ndarray
    amplitude: np.ndarray
    snr_db: np.ndarray
    seed: int


def _validate_frequency(frequency_hz: float, prf_hz: float) -> None:
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if not np.isfinite(frequency_hz) or not -prf_hz / 2 <= frequency_hz < prf_hz / 2:
        raise ValueError("frequency_hz must lie in [-PRF/2, PRF/2)")


def generate_single_tone(
    frequency_hz: float,
    phase_rad: float,
    amplitude: float,
    snr_db: float,
    prf_hz: float,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Return one complex sinusoid plus circular complex Gaussian noise.

    SNR is the per-sample signal power divided by total complex noise power.
    ``snr_db=np.inf`` produces a noiseless signal.
    """
    _validate_frequency(frequency_hz, prf_hz)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not np.isfinite(amplitude) or amplitude < 0:
        raise ValueError("amplitude must be nonnegative and finite")
    if not np.isfinite(phase_rad):
        raise ValueError("phase_rad must be finite")
    if not (np.isfinite(snr_db) or snr_db == np.inf):
        raise ValueError("snr_db must be finite or positive infinity")

    n = np.arange(n_samples, dtype=np.float64)
    clean = amplitude * np.exp(1j * (2.0 * np.pi * frequency_hz * n / prf_hz + phase_rad))
    rng = np.random.default_rng(seed)
    if snr_db == np.inf or amplitude == 0:
        noise = np.zeros(n_samples, dtype=np.complex128)
    else:
        noise_power = amplitude**2 / (10.0 ** (snr_db / 10.0))
        component_sigma = np.sqrt(noise_power / 2.0)
        noise = component_sigma * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
    return np.asarray(clean + noise, dtype=np.complex64)


def generate_batch(
    count: int,
    seed: int,
    prf_hz: float,
    n_samples: int,
    snr_db: float | None = None,
    snr_levels_db: tuple[float, ...] = (-15, -5, 0, 10, 20),
    amplitude_range: tuple[float, float] = (0.5, 1.5),
) -> SignalBatch:
    """Generate a seeded batch with continuous frequency and random phase/amplitude.

    When ``snr_db`` is omitted, SNR values are sampled from ``snr_levels_db``.
    Every call owns its RNG, so train/validation/test sets can use independent seeds.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    low_amp, high_amp = amplitude_range
    if not (0 < low_amp <= high_amp and np.isfinite(high_amp)):
        raise ValueError("amplitude_range must contain positive finite bounds")
    if snr_db is not None and not np.isfinite(snr_db):
        raise ValueError("batch snr_db must be finite")
    if not snr_levels_db or not np.isfinite(snr_levels_db).all():
        raise ValueError("snr_levels_db must contain finite values")

    rng = np.random.default_rng(seed)
    frequency = rng.uniform(-prf_hz / 2.0, prf_hz / 2.0, size=count)
    phase = rng.uniform(-np.pi, np.pi, size=count)
    amplitude = rng.uniform(low_amp, high_amp, size=count)
    if snr_db is None:
        snr = rng.choice(np.asarray(snr_levels_db, dtype=np.float64), size=count)
    else:
        snr = np.full(count, float(snr_db), dtype=np.float64)

    n = np.arange(n_samples, dtype=np.float64)
    clean = amplitude[:, None] * np.exp(
        1j * (2.0 * np.pi * frequency[:, None] * n[None, :] / prf_hz + phase[:, None])
    )
    noise_power = amplitude**2 * np.power(10.0, -snr / 10.0)
    component_sigma = np.sqrt(noise_power / 2.0)
    noise = component_sigma[:, None] * (
        rng.standard_normal((count, n_samples)) + 1j * rng.standard_normal((count, n_samples))
    )
    return SignalBatch(
        iq=np.asarray(clean + noise, dtype=np.complex64),
        frequency_hz=frequency.astype(np.float64),
        phase_rad=phase.astype(np.float64),
        amplitude=amplitude.astype(np.float64),
        snr_db=snr.astype(np.float64),
        seed=int(seed),
    )
