"""Classical single-tone frequency estimators and P0 bounds."""

from __future__ import annotations

import numpy as np


def _as_batch(iq: np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(iq)
    if values.ndim == 1:
        return values[None, :], True
    if values.ndim != 2:
        raise ValueError("iq must have shape (N,) or (batch, N)")
    if values.shape[1] < 2:
        raise ValueError("each signal must contain at least two samples")
    return values, False


def wrap_frequency(frequency_hz: np.ndarray | float, prf_hz: float) -> np.ndarray | float:
    """Map frequencies to the half-open unambiguous interval [-PRF/2, PRF/2)."""
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


def _estimate_mle_batch(x: np.ndarray, prf_hz: float, iterations: int) -> np.ndarray:
    n_samples = x.shape[-1]
    spectrum = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
    peak = np.argmax(np.abs(spectrum), axis=-1)
    center = (peak - n_samples // 2) * prf_hz / n_samples
    half_width = prf_hz / n_samples

    left = center - half_width
    right = center + half_width
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
    return np.asarray(wrap_frequency((left + right) / 2.0, prf_hz))


def estimate_mle(iq: np.ndarray, prf_hz: float, iterations: int = 28, chunk_size: int = 512):
    """Continuous-frequency ML estimate for one complex tone in white Gaussian noise.

    With unknown complex amplitude, maximizing the periodogram magnitude is the
    profile maximum-likelihood estimate. A coarse FFT peak initializes a bounded
    golden-section search over its local main lobe.
    """
    x, scalar = _as_batch(iq)
    pieces = [_estimate_mle_batch(x[start : start + chunk_size], prf_hz, iterations)
              for start in range(0, len(x), chunk_size)]
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


def local_teacher_soft_target(
    iq: np.ndarray,
    teacher_frequency_hz: np.ndarray,
    noise_power: np.ndarray,
    prf_hz: float,
    candidate_count: int = 65,
    half_width_bins: float = 2.0,
) -> np.ndarray:
    """Build a circular soft target from the teacher's local profile likelihood.

    The likelihood is evaluated in a small band around the continuous ML estimate.
    This retains the teacher mode and represents local uncertainty; it is not a
    global posterior and is used only as a distillation target.
    """
    x, scalar = _as_batch(iq)
    teacher = np.asarray(teacher_frequency_hz, dtype=np.float64).reshape(-1)
    noise = np.asarray(noise_power, dtype=np.float64).reshape(-1)
    if len(teacher) != len(x) or len(noise) != len(x):
        raise ValueError("teacher_frequency_hz and noise_power must match the batch")
    if np.any(noise <= 0) or candidate_count < 3:
        raise ValueError("noise_power must be positive and candidate_count at least three")
    offsets = np.linspace(-half_width_bins, half_width_bins, candidate_count) * prf_hz / x.shape[-1]
    candidates = wrap_frequency(teacher[:, None] + offsets[None, :], prf_hz)
    n = np.arange(x.shape[-1], dtype=np.float64)
    outputs = np.empty((len(x), 2), dtype=np.float64)
    for start in range(0, len(x), 128):
        end = min(start + 128, len(x))
        phase = np.exp(-2j * np.pi * candidates[start:end, :, None] * n[None, None, :] / prf_hz)
        sums = np.sum(x[start:end, None, :] * phase, axis=-1)
        logits = np.abs(sums) ** 2 / (x.shape[-1] * noise[start:end, None])
        logits -= np.max(logits, axis=-1, keepdims=True)
        probability = np.exp(logits)
        probability /= np.sum(probability, axis=-1, keepdims=True)
        angles = 2.0 * np.pi * candidates[start:end] / prf_hz
        outputs[start:end, 0] = np.sum(probability * np.cos(angles), axis=-1)
        outputs[start:end, 1] = np.sum(probability * np.sin(angles), axis=-1)
    return outputs[0] if scalar else outputs
