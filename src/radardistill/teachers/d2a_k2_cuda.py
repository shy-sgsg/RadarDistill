"""Strict-CUDA B3 DPCA soft-response implementation for the R5-C audit.

This module is separate from the historic NumPy K2 function so R5-C can
measure the same frozen response on the GPU without changing old experiments.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from radardistill.d2a_baselines import dpca_time_shift_s


def _require_cuda_device(device: str):
    import torch

    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid CUDA device {device!r}") from error
    if selected.type != "cuda":
        raise ValueError("the R5-C teacher backend requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA teacher execution was requested but CUDA is unavailable")
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError(f"requested CUDA device {selected} is not visible")
    if selected.index is None:
        selected = torch.device("cuda:0")
    torch.cuda.set_device(selected)
    return torch, selected


def k2_response_cuda(
    iq: np.ndarray,
    config: dict[str, Any],
    *,
    device: str = "cuda:0",
    oversampling_factor: int = 1,
) -> np.ndarray:
    """Compute a per-CPI normalized B3 DPCA power map on an explicit CUDA GPU.

    Args:
        iq: Observable complex IQ with shape ``[B,3,130,32]``.
        config: Frozen E001 physical geometry/configuration only.
        device: CUDA device. CPU devices are rejected; no fallback is allowed.
        oversampling_factor: Positive integer zero-padding factor on slow time.

    Returns:
        Float32 support maps with shape ``[B,130*factor,32]``. The values are
        normalized relative power support, not calibrated probabilities.
    """
    if isinstance(oversampling_factor, bool) or not isinstance(oversampling_factor, int):
        raise ValueError("oversampling_factor must be a positive integer")
    if oversampling_factor <= 0:
        raise ValueError("oversampling_factor must be a positive integer")

    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[0] <= 0 or values.shape[1:] != (3, 130, 32):
        raise ValueError("iq must have shape [B,3,130,32] with a nonempty batch")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")

    torch, selected = _require_cuda_device(device)
    samples = torch.as_tensor(values, dtype=torch.complex128, device=selected)
    shifted_spectrum = torch.fft.fft(samples[:, 1], dim=1)
    normalized_frequency = torch.fft.fftfreq(
        130, d=1.0, dtype=torch.float64, device=selected
    )
    shift_samples = dpca_time_shift_s(config, (0, 1)) * float(config["radar"]["prf_hz"])
    phase = torch.exp((2.0j * torch.pi) * normalized_frequency * shift_samples)
    aligned_channel = torch.fft.ifft(
        shifted_spectrum * phase[None, :, None], dim=1
    )
    dpca_residual = samples[:, 0] - aligned_channel
    spectrum = torch.fft.fftshift(
        torch.fft.fft(
            dpca_residual,
            n=130 * oversampling_factor,
            dim=1,
        ),
        dim=1,
    )
    power = spectrum.abs().square()
    totals = power.sum(dim=(1, 2), keepdim=True)
    if not bool(torch.all(torch.isfinite(power))) or not bool(torch.all(totals > 0.0)):
        raise ValueError("K2 cannot normalize a zero or invalid B3 response")
    response = (power / totals).to(torch.float32)
    if not bool(torch.all(torch.isfinite(response))):
        raise FloatingPointError("CUDA B3 response contains non-finite values")
    torch.cuda.synchronize(selected)
    return response.cpu().numpy()
