"""Read-only helpers for development-only D1-R1 Doppler diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..distillation.task_losses import decode_task_outputs


def validate_diagnostic_cache_path(path: str | Path) -> Path:
    """Reject paths that identify a blind split or saved blind predictions."""
    cache_path = Path(path)
    parts = [part.lower() for part in cache_path.parts]
    if any(
        part in {"blind", "final_blind", "final-blind", "blind_predictions"}
        or "blind_predictions" in part
        for part in parts
    ):
        raise ValueError(f"D1-R1 diagnostics cannot read blind data: {cache_path}")
    return cache_path


def circular_doppler_error_hz(
    estimate_hz: np.ndarray, truth_hz: np.ndarray, *, prf_hz: float
) -> np.ndarray:
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    estimate = np.asarray(estimate_hz, dtype=np.float64)
    truth = np.asarray(truth_hz, dtype=np.float64)
    if estimate.shape != truth.shape:
        raise ValueError("estimate and truth must have identical shapes")
    if np.any(~np.isfinite(estimate)) or np.any(~np.isfinite(truth)):
        raise ValueError("Doppler estimates and truths must be finite")
    return np.remainder(estimate - truth + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def diagnostic_truth_range_bins(
    truth_range_cell: np.ndarray,
    truth_presence: np.ndarray,
    *,
    range_bin_count: int,
) -> np.ndarray:
    """Create oracle indices for H1 only; H0 range labels are intentionally masked."""
    values = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    presence = np.asarray(truth_presence, dtype=np.bool_).reshape(-1)
    if values.shape != presence.shape:
        raise ValueError("truth range and presence arrays must have matching shapes")
    if range_bin_count <= 0:
        raise ValueError("range_bin_count must be positive")
    h1_values = values[presence]
    if np.any(~np.isfinite(h1_values)) or np.any(h1_values < 0.0) or np.any(h1_values >= range_bin_count):
        raise ValueError("H1 truth ranges must be finite and inside the range-bin extent")
    result = np.zeros(values.shape, dtype=np.int64)
    result[presence] = np.floor(h1_values).astype(np.int64)
    return result


def fit_circular_ridge_probe(
    train_features: np.ndarray,
    train_frequency_hz: np.ndarray,
    eval_features: np.ndarray,
    eval_frequency_hz: np.ndarray,
    *,
    prf_hz: float,
    alpha: float,
) -> dict[str, np.ndarray | float]:
    """Fit a fixed-standardization dual ridge probe to circular frequency targets."""
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    x_train = np.asarray(train_features, dtype=np.float64)
    x_eval = np.asarray(eval_features, dtype=np.float64)
    train_hz = np.asarray(train_frequency_hz, dtype=np.float64).reshape(-1)
    eval_hz = np.asarray(eval_frequency_hz, dtype=np.float64).reshape(-1)
    if x_train.ndim != 2 or x_eval.ndim != 2 or x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("probe features must be 2D with matching feature dimensions")
    if x_train.shape[0] != train_hz.size or x_eval.shape[0] != eval_hz.size:
        raise ValueError("probe frequency labels must match feature counts")
    if x_train.shape[0] == 0 or x_eval.shape[0] == 0:
        raise ValueError("probe training and evaluation data must be nonempty")
    if (
        np.any(~np.isfinite(x_train))
        or np.any(~np.isfinite(x_eval))
        or np.any(~np.isfinite(train_hz))
        or np.any(~np.isfinite(eval_hz))
    ):
        raise ValueError("probe data must be finite")

    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z_train = (x_train - mean) / scale
    z_eval = (x_eval - mean) / scale
    train_phase = 2.0 * math.pi * train_hz / prf_hz
    target = np.column_stack((np.cos(train_phase), np.sin(train_phase)))
    target_mean = target.mean(axis=0)

    kernel = z_train @ z_train.T
    kernel.flat[:: kernel.shape[0] + 1] += alpha
    dual = np.linalg.solve(kernel, target - target_mean)
    predicted_vector = (z_eval @ z_train.T) @ dual + target_mean
    predicted_phase = np.arctan2(predicted_vector[:, 1], predicted_vector[:, 0])
    predicted_hz = np.remainder(
        predicted_phase * (prf_hz / (2.0 * math.pi)) + prf_hz / 2.0,
        prf_hz,
    ) - prf_hz / 2.0
    errors = circular_doppler_error_hz(predicted_hz, eval_hz, prf_hz=prf_hz)
    return {
        "predicted_hz": predicted_hz,
        "errors_hz": errors,
        "rmse_hz": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_hz": float(np.mean(errors)),
    }


def _decode_vectors_at_range(
    doppler_vector: torch.Tensor, range_bin: torch.Tensor, prf_hz: float
) -> torch.Tensor:
    if doppler_vector.ndim != 3 or doppler_vector.shape[-1] != 2:
        raise ValueError("doppler_vector must have shape [B,R,2]")
    if range_bin.ndim != 1 or range_bin.shape[0] != doppler_vector.shape[0]:
        raise ValueError("range_bin must contain one index per sample")
    if torch.any(range_bin < 0) or torch.any(range_bin >= doppler_vector.shape[1]):
        raise ValueError("range_bin index is outside the Doppler head")
    row = torch.arange(range_bin.shape[0], device=range_bin.device)
    vector = doppler_vector[row, range_bin.to(torch.long)]
    angle = torch.atan2(vector[:, 1], vector[:, 0])
    return torch.remainder(angle * (prf_hz / (2.0 * math.pi)) + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def decode_doppler_modes(
    outputs: dict[str, torch.Tensor],
    truth_range_bin: torch.Tensor,
    *,
    prf_hz: float,
) -> dict[str, torch.Tensor]:
    """Return ordinary predicted-range decoding and a separate oracle diagnostic.

    The oracle path only reads an already-computed Doppler head. It never feeds
    the true range back through the model or changes the ordinary decoder.
    """
    normal = decode_task_outputs(outputs, prf_hz)
    predicted_bin = outputs["range_logits"].argmax(dim=-1)
    truth_bin = truth_range_bin.to(device=predicted_bin.device, dtype=torch.long)
    oracle_doppler = _decode_vectors_at_range(outputs["doppler_vector"], truth_bin, prf_hz)
    return {
        "predicted_range_bin": predicted_bin,
        "predicted_range_cell": normal["range_cell"],
        "predicted_range_doppler_hz": normal["doppler_hz"],
        "oracle_range_doppler_hz": oracle_doppler,
    }
