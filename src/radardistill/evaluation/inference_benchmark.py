"""Helpers for recording D1 inference latency measurements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def summarize_latency_samples(
    elapsed_seconds: Sequence[float], *, batch_size: int
) -> dict[str, int | float]:
    """Summarize elapsed wall times for repeated fixed-size batches."""
    if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    values = np.asarray(list(elapsed_seconds), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("elapsed_seconds must contain at least one measurement")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("latency measurements must be positive and finite")

    mean_seconds = float(np.mean(values))
    return {
        "repeat_count": int(values.size),
        "batch_size": int(batch_size),
        "mean_latency_ms": mean_seconds * 1000.0,
        "p50_latency_ms": float(np.percentile(values, 50.0) * 1000.0),
        "p95_latency_ms": float(np.percentile(values, 95.0) * 1000.0),
        "mean_throughput_samples_per_second": int(batch_size) / mean_seconds,
    }
