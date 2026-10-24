"""Utilities for the preregistered ideal three-channel phase experiment."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PhaseBatch:
    iq: np.ndarray
    frequency_hz: np.ndarray
    common_phase_rad: np.ndarray
    amplitude: np.ndarray
    channel_phase_rad: np.ndarray
    delta_rad: np.ndarray
    snr_db: np.ndarray
    noise_power: np.ndarray
    channel_snr_db: np.ndarray
    seed: int


def generate_phase_batch(
    count: int,
    seed: int,
    snr_db: float | None,
    *,
    prf_hz: float = 1300.0,
    n_samples: int = 130,
    snr_levels_db: tuple[float, ...] = (-15.0, -5.0, 0.0, 10.0, 20.0),
    amplitude_range: tuple[float, float] = (0.5, 1.5),
) -> PhaseBatch:
    """Generate a shared complex tone, static channel phases and independent AWGN."""
    if count <= 0 or n_samples < 2:
        raise ValueError("count must be positive and n_samples at least two")
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    low, high = amplitude_range
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        raise ValueError("amplitude_range must have positive finite ordered bounds")
    if snr_db is None:
        if not snr_levels_db or not np.isfinite(snr_levels_db).all():
            raise ValueError("snr_levels_db must contain finite values")
    elif not (np.isfinite(snr_db) or snr_db == np.inf):
        raise ValueError("snr_db must be finite or positive infinity")

    rng = np.random.default_rng(int(seed))
    frequency = rng.uniform(-prf_hz / 2.0, prf_hz / 2.0, count)
    common_phase = rng.uniform(-np.pi, np.pi, count)
    amplitude = rng.uniform(low, high, size=(count, 3))
    delta = rng.uniform(-np.pi, np.pi, size=(count, 2))
    reference_phase = rng.uniform(-np.pi, np.pi, count)
    phase = np.column_stack(
        (
            reference_phase,
            np.remainder(reference_phase + delta[:, 0] + np.pi, 2.0 * np.pi) - np.pi,
            np.remainder(reference_phase + delta[:, 1] + np.pi, 2.0 * np.pi) - np.pi,
        )
    )
    snr = (
        rng.choice(np.asarray(snr_levels_db, dtype=np.float64), size=count)
        if snr_db is None
        else np.full(count, float(snr_db), dtype=np.float64)
    )
    mean_power = np.mean(amplitude**2, axis=1)
    noise_power = np.zeros(count, dtype=np.float64)
    noisy = np.isfinite(snr)
    noise_power[noisy] = mean_power[noisy] * np.power(10.0, -snr[noisy] / 10.0)

    n = np.arange(n_samples, dtype=np.float64)
    waveform = np.exp(
        1j
        * (
            2.0 * np.pi * frequency[:, None] * n[None, :] / prf_hz
            + common_phase[:, None]
        )
    )
    channel_gain = amplitude * np.exp(1j * phase)
    clean = channel_gain[:, :, None] * waveform[:, None, :]
    noise = np.zeros_like(clean, dtype=np.complex128)
    if np.any(noisy):
        component_std = np.sqrt(noise_power[noisy, None, None] / 2.0)
        noise[noisy] = component_std * (
            rng.standard_normal((int(noisy.sum()), 3, n_samples))
            + 1j * rng.standard_normal((int(noisy.sum()), 3, n_samples))
        )
    channel_snr = np.full_like(amplitude, np.inf, dtype=np.float64)
    channel_snr[noisy] = 10.0 * np.log10(
        amplitude[noisy] ** 2 / noise_power[noisy, None]
    )
    return PhaseBatch(
        iq=np.asarray(clean + noise, dtype=np.complex64),
        frequency_hz=frequency.astype(np.float64),
        common_phase_rad=common_phase.astype(np.float64),
        amplitude=amplitude.astype(np.float64),
        channel_phase_rad=phase.astype(np.float64),
        delta_rad=delta.astype(np.float64),
        snr_db=snr.astype(np.float64),
        noise_power=noise_power.astype(np.float64),
        channel_snr_db=channel_snr.astype(np.float64),
        seed=int(seed),
    )


def normalized_cross_coherence(iq: torch.Tensor) -> torch.Tensor:
    """Return [Re/Im(C21), Re/Im(C31)] with each correlation energy-normalized."""
    values = torch.as_tensor(iq)
    if values.ndim == 2:
        values = values.unsqueeze(0)
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[2] < 2:
        raise ValueError("iq must have shape (batch, 3, N) or (3, N), with N >= 2")
    if not torch.is_complex(values):
        raise ValueError("iq must be complex valued")
    values = values.to(torch.complex64)
    reference = values[:, 0]
    reference_energy = reference.abs().square().sum(dim=-1)
    outputs = []
    tiny = torch.finfo(torch.float32).tiny
    for channel in (1, 2):
        signal = values[:, channel]
        cross = (signal * reference.conj()).sum(dim=-1)
        denominator = torch.sqrt(
            signal.abs().square().sum(dim=-1) * reference_energy
        ).clamp_min(tiny)
        outputs.append(cross / denominator)
    stacked = torch.stack(outputs, dim=-1)
    return torch.stack((stacked.real, stacked.imag), dim=-1).flatten(start_dim=1)


def phase_target_vectors(delta_rad: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(delta_rad, dtype=torch.float32)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or values.shape[-1] != 2:
        raise ValueError("delta_rad must have shape (batch, 2) or (2,)")
    return torch.stack(
        (torch.cos(values[:, 0]), torch.sin(values[:, 0]), torch.cos(values[:, 1]), torch.sin(values[:, 1])),
        dim=-1,
    )


class P1PhaseMLP(nn.Module):
    """Compact readout from explicit pairwise complex coherence to two phases."""

    def __init__(self, hidden_units: tuple[int, int] = (64, 64)) -> None:
        super().__init__()
        if len(hidden_units) != 2 or any(width <= 0 for width in hidden_units):
            raise ValueError("hidden_units must contain two positive widths")
        self.network = nn.Sequential(
            nn.Linear(4, hidden_units[0]),
            nn.GELU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.GELU(),
            nn.Linear(hidden_units[1], 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != 4:
            raise ValueError("features must have shape (batch, 4)")
        return self.network(features.float())


def decode_phase_vectors(outputs: torch.Tensor) -> torch.Tensor:
    if outputs.ndim != 2 or outputs.shape[-1] != 4:
        raise ValueError("outputs must have shape (batch, 4)")
    values = outputs.float()
    return torch.stack(
        (torch.atan2(values[:, 1], values[:, 0]), torch.atan2(values[:, 3], values[:, 2])),
        dim=-1,
    )


def circular_error(estimate: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return np.angle(np.exp(1j * (np.asarray(estimate) - np.asarray(truth))))


def phase_reference_correlation(iq: np.ndarray) -> np.ndarray:
    values = np.asarray(iq, dtype=np.complex128)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[2] < 2:
        raise ValueError("iq must have shape (batch, 3, N) or (3, N)")
    return np.column_stack(
        tuple(np.angle(np.sum(values[:, channel] * values[:, 0].conj(), axis=-1)) for channel in (1, 2))
    )


def phase_sample_circular_mean(iq: np.ndarray) -> np.ndarray:
    values = np.asarray(iq, dtype=np.complex128)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[2] < 2:
        raise ValueError("iq must have shape (batch, 3, N) or (3, N)")
    estimates = []
    for channel in (1, 2):
        products = values[:, channel] * values[:, 0].conj()
        unit = products / np.maximum(np.abs(products), np.finfo(np.float64).tiny)
        estimates.append(np.angle(unit.sum(axis=-1)))
    return np.column_stack(estimates)


def phase_profile_ml(iq: np.ndarray) -> np.ndarray:
    """Rank-one profile-ML phase estimates for equal-variance complex AWGN."""
    values = np.asarray(iq, dtype=np.complex128)
    if values.ndim == 2:
        values = values[None, ...]
    if values.ndim != 3 or values.shape[1] != 3 or values.shape[2] < 2:
        raise ValueError("iq must have shape (batch, 3, N) or (3, N)")
    left_vectors = np.linalg.svd(values, full_matrices=False)[0][:, :, 0]
    return np.column_stack(
        tuple(np.angle(left_vectors[:, channel] * left_vectors[:, 0].conj()) for channel in (1, 2))
    )


def phase_fisher_information(
    amplitude: np.ndarray, sigma2: float, waveform: np.ndarray
) -> np.ndarray:
    """Fisher information for (delta21, delta31) after eliminating shared s[n]."""
    gains = np.asarray(amplitude, dtype=np.float64)
    samples = np.asarray(waveform, dtype=np.complex128)
    if gains.shape != (3,) or np.any(gains <= 0) or not np.all(np.isfinite(gains)):
        raise ValueError("amplitude must contain three positive finite values")
    if not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError("sigma2 must be positive and finite")
    if samples.ndim != 1 or samples.size == 0 or not np.all(np.isfinite(samples)):
        raise ValueError("waveform must be a nonempty finite vector")
    q1, q2, q3 = gains**2
    total_gain = q1 + q2 + q3
    energy = float(np.sum(np.abs(samples) ** 2))
    return (2.0 * energy / sigma2) * np.asarray(
        [
            [q2 * (1.0 - q2 / total_gain), -q2 * q3 / total_gain],
            [-q2 * q3 / total_gain, q3 * (1.0 - q3 / total_gain)],
        ],
        dtype=np.float64,
    )


def phase_crlb_std(amplitude: np.ndarray, sigma2: float, waveform: np.ndarray) -> np.ndarray:
    covariance_bound = np.linalg.inv(phase_fisher_information(amplitude, sigma2, waveform))
    return np.sqrt(np.diag(covariance_bound))


def require_cuda(
    training_device: str = "cuda",
    *,
    cuda_available: bool | None = None,
    device_count: int | None = None,
) -> torch.device:
    if training_device != "cuda":
        raise ValueError("P1-A-E001 training.device must be cuda")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    count = torch.cuda.device_count() if device_count is None else device_count
    if not available or count < 1:
        raise RuntimeError("CUDA is required for P1-A-E001; no model training was started")
    return torch.device("cuda:0")
