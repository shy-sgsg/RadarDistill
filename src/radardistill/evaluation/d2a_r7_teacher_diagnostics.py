"""Component-level physical diagnostics for the frozen D2-A DPCA teacher."""

from __future__ import annotations

from typing import Any

import numpy as np

from radardistill.d2a_baselines import dpca_time_shift_s


def dpca_component_diagnostics(
    iq: np.ndarray,
    config: dict[str, Any],
    *,
    channel_pair: tuple[int, int] = (0, 1),
) -> dict[str, np.ndarray | float]:
    """Measure pre/post alignment coherence, gain, phase, and residual power.

    The function accepts one component or a batch with shape ``(C,N,R)`` or
    ``(B,C,N,R)``. It applies the same circular Fourier fractional shift used
    by the frozen DPCA implementation. It reports the unshifted and aligned
    residuals so finite-CPI and boundary behavior remain visible.
    """
    values = np.asarray(iq)
    if values.ndim == 3:
        values = values[None, ...]
    if (
        values.ndim != 4
        or values.shape[0] == 0
        or values.shape[1] < 2
        or values.shape[2] < 2
        or values.shape[3] == 0
    ):
        raise ValueError("iq must have shape (C,N,R) or (B,C,N,R) with nonempty dimensions")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    first, second = (int(index) for index in channel_pair)
    if first == second or min(first, second) < 0 or max(first, second) >= values.shape[1]:
        raise ValueError("channel_pair must contain two valid distinct indices")

    samples = values.astype(np.complex128, copy=False)
    channel_a = samples[:, first]
    channel_b = samples[:, second]
    prf_hz = float(config["radar"]["prf_hz"])
    delay_s = dpca_time_shift_s(config, channel_pair)
    shift_samples = delay_s * prf_hz
    normalized_frequency = np.fft.fftfreq(samples.shape[2])
    shift_phase = np.exp(2j * np.pi * normalized_frequency * shift_samples)
    aligned_b = np.fft.ifft(
        np.fft.fft(channel_b, axis=1) * shift_phase[None, :, None], axis=1
    )

    def coherence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        numerator = np.sum(left * np.conj(right), axis=(1, 2))
        denominator = np.sqrt(
            np.sum(np.abs(left) ** 2, axis=(1, 2))
            * np.sum(np.abs(right) ** 2, axis=(1, 2))
        )
        if np.any(denominator <= 0.0) or np.any(~np.isfinite(denominator)):
            raise ValueError("coherence is undefined for zero or invalid channel power")
        return numerator / denominator

    power_a = np.mean(np.abs(channel_a) ** 2, axis=(1, 2))
    power_b = np.mean(np.abs(aligned_b) ** 2, axis=(1, 2))
    if np.any(power_a <= 0.0) or np.any(power_b <= 0.0):
        raise ValueError("channel amplitude ratio is undefined for zero power")
    residual_aligned = np.mean(np.abs(channel_a - aligned_b) ** 2, axis=(1, 2))
    residual_unaligned = np.mean(np.abs(channel_a - channel_b) ** 2, axis=(1, 2))
    aligned_ratio = residual_aligned / power_a
    with np.errstate(divide="ignore", invalid="ignore"):
        suppression_db = -10.0 * np.log10(np.maximum(aligned_ratio, np.finfo(np.float64).tiny))

    return {
        "shift_samples": float(shift_samples),
        "rho_before": coherence(channel_a, channel_b),
        "rho_after": coherence(channel_a, aligned_b),
        "phase_residual_before_rad": np.angle(coherence(channel_a, channel_b)),
        "phase_residual_after_rad": np.angle(coherence(channel_a, aligned_b)),
        "amplitude_ratio_channel_a_to_aligned_b": np.sqrt(power_a / power_b),
        "residual_power_ratio_unaligned": residual_unaligned / power_a,
        "residual_power_ratio_aligned": aligned_ratio,
        "suppression_db_relative_to_channel_a": suppression_db,
    }


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    replicates: int,
    confidence: float,
    seed: int,
) -> list[float] | None:
    if values.size < 2 or np.all(values == values[0]):
        return None
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        count = min(256, replicates - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        samples[start : start + count] = np.median(values[indices], axis=1)
    alpha = 1.0 - confidence
    return [
        float(np.quantile(samples, alpha / 2.0)),
        float(np.quantile(samples, 1.0 - alpha / 2.0)),
    ]


def summarize_component_diagnostics(
    diagnostics: dict[str, np.ndarray | float],
    *,
    bootstrap_replicates: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarize scene-level component metrics with bootstrap uncertainty."""
    if bootstrap_replicates <= 0 or seed < 0 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap settings must be positive and valid")
    rho_before = np.asarray(diagnostics["rho_before"], dtype=np.complex128).reshape(-1)
    rho_after = np.asarray(diagnostics["rho_after"], dtype=np.complex128).reshape(-1)
    amplitude = np.asarray(
        diagnostics["amplitude_ratio_channel_a_to_aligned_b"], dtype=np.float64
    ).reshape(-1)
    unaligned = np.asarray(diagnostics["residual_power_ratio_unaligned"], dtype=np.float64).reshape(-1)
    aligned = np.asarray(diagnostics["residual_power_ratio_aligned"], dtype=np.float64).reshape(-1)
    suppression = np.asarray(
        diagnostics["suppression_db_relative_to_channel_a"], dtype=np.float64
    ).reshape(-1)
    arrays = (rho_before, rho_after, amplitude, unaligned, aligned, suppression)
    if not rho_before.size or any(values.size != rho_before.size for values in arrays):
        raise ValueError("all diagnostics must contain the same nonzero number of scenes")
    if any(np.any(~np.isfinite(values)) for values in arrays):
        raise ValueError("diagnostics must be finite")

    before_magnitude = np.abs(rho_before)
    after_magnitude = np.abs(rho_after)
    before_phase = np.angle(rho_before)
    after_phase = np.angle(rho_after)

    def distribution(values: np.ndarray, label: str) -> dict[str, Any]:
        return {
            f"{label}_median": float(np.median(values)),
            f"{label}_p05": float(np.quantile(values, 0.05)),
            f"{label}_p95": float(np.quantile(values, 0.95)),
            f"{label}_median_bootstrap_ci95": _bootstrap_median_ci(
                values,
                replicates=bootstrap_replicates,
                confidence=confidence,
                seed=seed,
            ),
        }

    return {
        "n_scenes": int(rho_before.size),
        "shift_samples": float(diagnostics["shift_samples"]),
        **distribution(before_magnitude, "coherence_before_magnitude"),
        **distribution(after_magnitude, "coherence_after_magnitude"),
        "coherence_before_circular_mean_phase_rad": float(
            np.angle(np.mean(np.exp(1j * before_phase)))
        ),
        "coherence_after_circular_mean_phase_rad": float(
            np.angle(np.mean(np.exp(1j * after_phase)))
        ),
        "phase_residual_after_median_abs_rad": float(np.median(np.abs(after_phase))),
        "phase_residual_after_p95_abs_rad": float(np.quantile(np.abs(after_phase), 0.95)),
        **distribution(amplitude, "channel_amplitude_ratio_a_to_aligned_b"),
        **distribution(unaligned, "residual_power_ratio_unaligned"),
        **distribution(aligned, "aligned_residual_power_ratio"),
        **distribution(suppression, "suppression_db_relative_to_channel_a"),
    }
