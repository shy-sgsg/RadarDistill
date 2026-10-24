"""Classical single-tone frequency estimators and P0 bounds."""

from __future__ import annotations

import numpy as np


def _as_batch(iq: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(iq)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    elif values.ndim != 2:
        raise ValueError("iq must have shape (N,) or (batch, N)")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("iq must contain a nonempty batch of signals with at least two samples")
    return values, scalar


def wrap_frequency(frequency_hz: np.ndarray | float, prf_hz: float) -> np.ndarray | float:
    """Map frequencies to the half-open unambiguous interval [-PRF/2, PRF/2)."""
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    result = (np.asarray(frequency_hz) + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
    return float(result) if result.ndim == 0 else result


def wrapped_error(estimate_hz: np.ndarray | float, truth_hz: np.ndarray | float, prf_hz: float):
    """Return signed estimation error on the Doppler circle."""
    return wrap_frequency(np.asarray(estimate_hz) - np.asarray(truth_hz), prf_hz)


def estimate_fft_peak(iq: np.ndarray, prf_hz: float) -> np.ndarray | float:
    """N-point FFT maximum-magnitude bin, with no sub-bin correction."""
    x, scalar = _as_batch(iq)
    spectrum = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
    peak = np.argmax(np.abs(spectrum), axis=-1)
    frequencies = np.fft.fftshift(np.fft.fftfreq(x.shape[-1], d=1.0 / prf_hz))
    result = frequencies[peak]
    return float(result[0]) if scalar else result


def estimate_fft_interpolated(iq: np.ndarray, prf_hz: float) -> np.ndarray | float:
    """Parabolic interpolation of the log-magnitude FFT peak."""
    x, scalar = _as_batch(iq)
    n_samples = x.shape[-1]
    spectrum = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
    magnitude = np.maximum(np.abs(spectrum), np.finfo(np.float64).tiny)
    log_magnitude = np.log(magnitude)
    peak = np.argmax(magnitude, axis=-1)
    row = np.arange(x.shape[0])
    left = log_magnitude[row, (peak - 1) % n_samples]
    center = log_magnitude[row, peak]
    right = log_magnitude[row, (peak + 1) % n_samples]
    denominator = left - 2.0 * center + right
    offset = np.divide(
        0.5 * (left - right),
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > 1e-12,
    )
    result = (peak + offset - n_samples // 2) * prf_hz / n_samples
    result = wrap_frequency(result, prf_hz)
    return float(result[0]) if scalar else result


def estimate_phase_difference(iq: np.ndarray, prf_hz: float) -> np.ndarray | float:
    """One-lag phase-difference frequency estimate."""
    x, scalar = _as_batch(iq)
    correlation = np.sum(x[:, 1:] * np.conj(x[:, :-1]), axis=-1)
    result = wrap_frequency(np.angle(correlation) * prf_hz / (2.0 * np.pi), prf_hz)
    return float(result[0]) if scalar else result


def _periodogram_power(x: np.ndarray, frequency_hz: np.ndarray, prf_hz: float) -> np.ndarray:
    n = np.arange(x.shape[-1], dtype=np.float64)
    phasor = np.exp(-2j * np.pi * frequency_hz[:, None] * n[None, :] / prf_hz)
    return np.abs(np.sum(x * phasor, axis=-1)) ** 2


def _refine_periodogram_batch(
    x: np.ndarray, centers: np.ndarray, half_width_hz: float, prf_hz: float, iterations: int
) -> np.ndarray:
    left = centers - half_width_hz
    right = centers + half_width_hz
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    p1 = _periodogram_power(x, x1, prf_hz)
    p2 = _periodogram_power(x, x2, prf_hz)
    for _ in range(iterations):
        choose_left = p1 > p2
        next_right = np.where(choose_left, x2, right)
        next_left = np.where(choose_left, left, x1)
        next_x1 = next_right - golden * (next_right - next_left)
        next_x2 = next_left + golden * (next_right - next_left)
        next_p1 = _periodogram_power(x, next_x1, prf_hz)
        next_p2 = _periodogram_power(x, next_x2, prf_hz)
        left, right = next_left, next_right
        x1, x2, p1, p2 = next_x1, next_x2, next_p1, next_p2
    return (left + right) / 2.0


def _estimate_mle_batch(x: np.ndarray, prf_hz: float, iterations: int) -> np.ndarray:
    n_samples = x.shape[-1]
    spectrum = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
    peak = np.argmax(np.abs(spectrum), axis=-1)
    center = (peak - n_samples // 2) * prf_hz / n_samples
    refined = _refine_periodogram_batch(x, center, prf_hz / n_samples, prf_hz, iterations)
    return np.asarray(wrap_frequency(refined, prf_hz))


def estimate_mle_local(iq: np.ndarray, prf_hz: float, iterations: int = 28, chunk_size: int = 512):
    """FFT-initialized local profile-ML estimate for a single complex tone.

    With unknown complex amplitude, maximizing periodogram power is the profile
    likelihood. A coarse FFT peak initializes a bounded golden-section search
    over its neighboring FFT bins. This is a local estimator and can miss a
    higher off-grid maximum elsewhere in the unambiguous interval.
    """
    if not np.isfinite(prf_hz) or prf_hz <= 0 or iterations < 0 or chunk_size <= 0:
        raise ValueError("prf_hz and chunk_size must be positive; iterations >= 0")
    x, scalar = _as_batch(iq)
    pieces = [
        _estimate_mle_batch(x[start : start + chunk_size], prf_hz, iterations)
        for start in range(0, len(x), chunk_size)
    ]
    result = np.concatenate(pieces)
    return float(result[0]) if scalar else result


def estimate_mle(iq: np.ndarray, prf_hz: float, iterations: int = 28, chunk_size: int = 512):
    """Backward-compatible alias for :func:`estimate_mle_local`."""
    return estimate_mle_local(iq, prf_hz, iterations=iterations, chunk_size=chunk_size)


def estimate_mle_global_grid(
    iq: np.ndarray,
    prf_hz: float,
    grid_size: int = 4096,
    iterations: int = 28,
    chunk_size: int = 128,
):
    """Global profile-ML grid search over one PRF followed by peak refinement.

    The zero-padded FFT evaluates the periodogram over a fixed high-resolution
    grid. Only the winning cell is refined. A finite grid remains an approximation
    to the continuous global maximum.
    """
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if grid_size < 2 or grid_size % 2 or iterations < 0 or chunk_size <= 0:
        raise ValueError("grid_size must be even and >= 2; iterations >= 0; chunk_size > 0")
    x, scalar = _as_batch(iq)
    if grid_size < x.shape[-1]:
        raise ValueError("grid_size must be at least the number of signal samples")
    frequencies = (np.arange(grid_size) - grid_size // 2) * prf_hz / grid_size
    step_hz = prf_hz / grid_size
    pieces = []
    for start in range(0, len(x), chunk_size):
        chunk = x[start : start + chunk_size]
        spectrum = np.fft.fftshift(np.fft.fft(chunk, n=grid_size, axis=-1), axes=-1)
        peak = np.argmax(np.abs(spectrum), axis=-1)
        centers = frequencies[peak]
        refined = _refine_periodogram_batch(chunk, centers, step_hz, prf_hz, iterations)
        pieces.append(np.asarray(wrap_frequency(refined, prf_hz)))
    result = np.concatenate(pieces)
    return float(result[0]) if scalar else result


def estimate_multilag_correlation(
    iq: np.ndarray,
    prf_hz: float,
    lags: tuple[int, ...] = (1, 2, 4, 8),
    grid_size: int = 8192,
    iterations: int = 24,
    chunk_size: int = 128,
):
    """Global weighted phase-coherence estimate from normalized lag correlations.

    This is the nonlearning counterpart to a correlation-feature student. It is
    not the full likelihood estimator. The score sums real phase-aligned lag
    correlations, weighted by each lag's number of sample pairs.
    """
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if grid_size < 2 or grid_size % 2 or iterations < 0 or chunk_size <= 0:
        raise ValueError("grid_size must be even and >= 2; iterations >= 0; chunk_size > 0")
    if not lags or any(int(lag) <= 0 for lag in lags) or len(set(lags)) != len(lags):
        raise ValueError("lags must be distinct positive integers")
    x, scalar = _as_batch(iq)
    n_samples = x.shape[-1]
    if any(lag >= n_samples for lag in lags):
        raise ValueError("each lag must be smaller than the number of samples")
    lag_values = np.asarray(lags, dtype=np.float64)
    weights = (n_samples - lag_values)[None, :]
    frequencies = (np.arange(grid_size) - grid_size // 2) * prf_hz / grid_size
    phase_grid = 2.0 * np.pi * frequencies[:, None] * lag_values[None, :] / prf_hz
    cos_grid, sin_grid = np.cos(phase_grid), np.sin(phase_grid)
    pieces = []

    for start in range(0, len(x), chunk_size):
        chunk = x[start : start + chunk_size]
        power = np.mean(np.abs(chunk) ** 2, axis=-1)
        correlation = np.stack(
            [
                np.mean(chunk[:, lag:] * np.conj(chunk[:, :-lag]), axis=-1)
                / np.maximum(power, np.finfo(np.float64).tiny)
                for lag in lags
            ],
            axis=-1,
        )
        score = np.zeros((len(chunk), grid_size), dtype=np.float64)
        for index in range(len(lags)):
            score += weights[:, index, None] * (
                correlation[:, index].real[:, None] * cos_grid[None, :, index]
                + correlation[:, index].imag[:, None] * sin_grid[None, :, index]
            )
        centers = frequencies[np.argmax(score, axis=-1)]

        def coherence_score(candidate: np.ndarray) -> np.ndarray:
            phase = 2.0 * np.pi * candidate[:, None] * lag_values[None, :] / prf_hz
            terms = correlation.real * np.cos(phase) + correlation.imag * np.sin(phase)
            return np.sum(weights * terms, axis=-1)

        left, right = centers - prf_hz / grid_size, centers + prf_hz / grid_size
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        x1, x2 = right - golden * (right - left), left + golden * (right - left)
        p1, p2 = coherence_score(x1), coherence_score(x2)
        for _ in range(iterations):
            choose_left = p1 > p2
            next_right = np.where(choose_left, x2, right)
            next_left = np.where(choose_left, left, x1)
            next_x1 = next_right - golden * (next_right - next_left)
            next_x2 = next_left + golden * (next_right - next_left)
            next_p1, next_p2 = coherence_score(next_x1), coherence_score(next_x2)
            left, right, x1, x2, p1, p2 = next_left, next_right, next_x1, next_x2, next_p1, next_p2
        pieces.append(np.asarray(wrap_frequency((left + right) / 2.0, prf_hz)))

    result = np.concatenate(pieces)
    return float(result[0]) if scalar else result
def frequency_crlb_hz(prf_hz: float, n_samples: int, snr_db: np.ndarray | float):
    """Standard-deviation CRLB for unknown complex amplitude/phase and white noise.

    SNR is per sample, A^2 / E[|w|^2]. The result is a bound for unbiased
    estimators under the single-tone complex Gaussian model.
    """
    if prf_hz <= 0 or n_samples < 2:
        raise ValueError("prf_hz must be positive and n_samples at least two")
    snr_linear = np.power(10.0, np.asarray(snr_db, dtype=np.float64) / 10.0)
    centered_energy = n_samples * (n_samples**2 - 1.0) / 12.0
    variance = prf_hz**2 / (8.0 * np.pi**2 * snr_linear * centered_energy)
    result = np.sqrt(variance)
    return float(result) if result.ndim == 0 else result


def local_teacher_profile_likelihood(
    iq: np.ndarray,
    teacher_frequency_hz: np.ndarray,
    noise_power: np.ndarray,
    prf_hz: float,
    candidate_count: int = 65,
    half_width_bins: float = 2.0,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequencies and normalized full local profile likelihood.

    The finite candidate band is centered on the teacher estimate, has the
    explicit configured width and grid, and is not a global posterior. The
    temperature scales profile log-likelihood before normalization.
    """
    x, scalar = _as_batch(iq)
    teacher = np.asarray(teacher_frequency_hz, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise_power, dtype=np.float64).reshape(-1)
    if len(teacher) != len(x) or len(noise) != len(x):
        raise ValueError("teacher_frequency_hz and noise_power must match the batch")
    if (
        not np.isfinite(prf_hz)
        or prf_hz <= 0
        or np.any(~np.isfinite(noise))
        or np.any(noise <= 0)
        or candidate_count < 3
        or not np.isfinite(half_width_bins)
        or half_width_bins <= 0
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("invalid PRF, noise power, candidate count, half-width, or temperature")
    offsets = np.linspace(-half_width_bins, half_width_bins, candidate_count) * prf_hz / x.shape[-1]
    candidates = wrap_frequency(teacher[:, None] + offsets[None, :], prf_hz)
    n = np.arange(x.shape[-1], dtype=np.float64)
    probability = np.empty((len(x), candidate_count), dtype=np.float64)
    for start in range(0, len(x), 128):
        end = min(start + 128, len(x))
        phase = np.exp(-2j * np.pi * candidates[start:end, :, None] * n[None, None, :] / prf_hz)
        sums = np.sum(x[start:end, None, :] * phase, axis=-1)
        logits = np.abs(sums) ** 2 / (x.shape[-1] * noise[start:end, None] * temperature)
        logits -= np.max(logits, axis=-1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
        probability[start:end] = probabilities
    if scalar:
        return candidates[0], probability[0]
    return candidates, probability


def local_teacher_soft_target(
    iq: np.ndarray,
    teacher_frequency_hz: np.ndarray,
    noise_power: np.ndarray,
    prf_hz: float,
    candidate_count: int = 65,
    half_width_bins: float = 2.0,
    temperature: float = 1.0,
) -> np.ndarray:
    """Return the legacy 2-D circular mean of the local likelihood target."""
    candidates, probability = local_teacher_profile_likelihood(
        iq,
        teacher_frequency_hz,
        noise_power,
        prf_hz,
        candidate_count=candidate_count,
        half_width_bins=half_width_bins,
        temperature=temperature,
    )
    scalar = probability.ndim == 1
    if scalar:
        candidates = candidates[None, :]
        probability = probability[None, :]
    angles = 2.0 * np.pi * candidates / prf_hz
    output = np.stack(
        (
            np.sum(probability * np.cos(angles), axis=-1),
            np.sum(probability * np.sin(angles), axis=-1),
        ),
        axis=-1,
    )
    return output[0] if scalar else output
