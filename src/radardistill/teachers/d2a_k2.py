"""IQ-only K2 soft range-Doppler response for D2-A-E002."""

from __future__ import annotations

from typing import Any

import numpy as np

from radardistill.d2a_baselines import run_b3


def k2_response(iq: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Return the frozen B3 DPCA power map normalized to a per-CPI distribution.

    The function accepts only observable IQ and the frozen physical geometry.
    It deliberately has no arguments for labels, scenario IDs, target location,
    SNR/CNR, or other simulation metadata.
    """
    values = np.asarray(iq)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32):
        raise ValueError("iq must have shape [B,3,130,32] or [3,130,32]")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    prf_hz = float(config["radar"]["prf_hz"])
    result = run_b3(values, prf_hz, config)
    power = np.asarray(result.rd_power, dtype=np.float64)
    if power.shape != (values.shape[0], 130, 32):
        raise RuntimeError("frozen B3 returned an unexpected K2 map shape")
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise RuntimeError("frozen B3 returned invalid K2 power")
    totals = power.sum(axis=(1, 2), keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("K2 cannot normalize a zero or invalid B3 response")
    return (power / totals).astype(np.float32)
