"""D1-E001 classical estimators and evaluation primitives."""

from __future__ import annotations

import math
import json
import platform
import subprocess
import time
from typing import Any
from pathlib import Path

import numpy as np


def evaluation_chunk_seed(
    seed_base: int,
    snr_index: int,
    chunk_index: int,
    *,
    snr_stride: int,
    snr_count: int,
) -> int:
    """Derive deterministic, nonoverlapping seeds for streamed SNR chunks."""
    values = (seed_base, snr_index, chunk_index, snr_stride, snr_count)
    if any(int(value) != value for value in values):
        raise ValueError("evaluation seed components must be integers")
    if seed_base < 0 or not 0 <= snr_index < snr_count or chunk_index < 0:
        raise ValueError("invalid evaluation seed base, SNR index, or chunk index")
    if snr_stride <= 0 or snr_count <= 0 or snr_count >= snr_stride:
        raise ValueError("snr_count and snr_stride must be positive with snr_count < snr_stride")
    return int(seed_base + snr_index * snr_stride + chunk_index * snr_stride * snr_count)


def _as_iq_batch(iq: Any) -> np.ndarray:
    values = np.asarray(iq)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1] != 3 or values.shape[2] < 2 or values.shape[3] < 3:
        raise ValueError("iq must have shape [B,3,N,R] with N>=2 and R>=3")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    return values


def wrap_frequency(frequency_hz: np.ndarray, prf_hz: float) -> np.ndarray:
    if not math.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    values = np.asarray(frequency_hz, dtype=np.float64)
    return (values + prf_hz / 2.0) % prf_hz - prf_hz / 2.0


def _log_parabolic_offset(values: np.ndarray, left: np.ndarray, center: np.ndarray, right: np.ndarray) -> np.ndarray:
    tiny = np.finfo(np.float64).tiny
    lval = np.log(np.maximum(left, tiny))
    cval = np.log(np.maximum(center, tiny))
    rval = np.log(np.maximum(right, tiny))
    denominator = lval - 2.0 * cval + rval
    offset = np.divide(
        0.5 * (lval - rval),
        denominator,
        out=np.zeros_like(denominator),
        where=np.abs(denominator) > 1e-12,
    )
    return np.clip(offset, -0.5, 0.5)


def _estimate_range_from_energy(iq: np.ndarray) -> np.ndarray:
    range_energy = np.sum(np.abs(iq) ** 2, axis=(1, 2), dtype=np.float64)
    peak = np.argmax(range_energy, axis=-1)
    rows = np.arange(iq.shape[0])
    result = peak.astype(np.float64)
    interior = (peak > 0) & (peak < iq.shape[-1] - 1)
    if np.any(interior):
        row = rows[interior]
        index = peak[interior]
        offset = _log_parabolic_offset(
            range_energy,
            range_energy[row, index - 1],
            range_energy[row, index],
            range_energy[row, index + 1],
        )
        result[interior] += offset
    return result


def _estimate_spectral_frequency(iq: np.ndarray, prf_hz: float, nfft: int, *, refine: bool) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(iq, n=nfft, axis=2), axes=2)
    power = np.sum(np.abs(spectrum) ** 2, axis=(1, 3), dtype=np.float64)
    peak = np.argmax(power, axis=-1)
    offset = np.zeros(len(iq), dtype=np.float64)
    if refine:
        rows = np.arange(len(iq))
        offset = _log_parabolic_offset(
            power,
            power[rows, (peak - 1) % nfft],
            power[rows, peak],
            power[rows, (peak + 1) % nfft],
        )
    frequency = (peak.astype(np.float64) + offset - nfft // 2) * prf_hz / nfft
    return wrap_frequency(frequency, prf_hz)


def estimate_fft_baselines(iq: Any, prf_hz: float) -> dict[str, dict[str, np.ndarray]]:
    """Return pooled three-channel/range FFT estimates with a shared range estimate."""
    values = _as_iq_batch(iq)
    range_estimate = _estimate_range_from_energy(values)
    return {
        "native_fft_peak": {
            "range_cell": range_estimate.copy(),
            "doppler_hz": _estimate_spectral_frequency(values, prf_hz, values.shape[2], refine=False),
        },
        "native_fft_log_parabolic": {
            "range_cell": range_estimate.copy(),
            "doppler_hz": _estimate_spectral_frequency(values, prf_hz, values.shape[2], refine=True),
        },
        "fourfold_fft_parabolic": {
            "range_cell": range_estimate.copy(),
            "doppler_hz": _estimate_spectral_frequency(values, prf_hz, 4 * values.shape[2], refine=True),
        },
    }


def estimate_phase_difference_multichannel(iq: Any, prf_hz: float) -> np.ndarray:
    """Pool one-lag phase correlations over all channels and range cells."""
    values = _as_iq_batch(iq)
    correlation = np.sum(
        values[:, :, 1:, :] * np.conj(values[:, :, :-1, :]), axis=(1, 2, 3)
    )
    return wrap_frequency(np.angle(correlation) * prf_hz / (2.0 * np.pi), prf_hz)


def estimate_profile_ml_reference(
    iq: Any,
    *,
    prf_hz: float,
    range_interval: tuple[float, float],
    range_grid_step: float = 0.25,
    frequency_fft_factor: int = 16,
    chunk_size: int = 1,
) -> dict[str, np.ndarray]:
    """Profile unknown complex channel gains on a sinc-range/Doppler single-target model.

    This is a finite-grid numerical reference, not a claim of continuous global
    maximum likelihood. Range uses a configured grid followed by local quadratic
    refinement; Doppler uses an N*factor FFT followed by local refinement.
    """
    values = _as_iq_batch(iq)
    if not math.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    low, high = (float(item) for item in range_interval)
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("range_interval must be finite and increasing")
    if not math.isfinite(range_grid_step) or range_grid_step <= 0.0:
        raise ValueError("range_grid_step must be positive and finite")
    if frequency_fft_factor <= 0 or chunk_size <= 0:
        raise ValueError("frequency_fft_factor and chunk_size must be positive")

    range_candidates = np.arange(low, high, range_grid_step, dtype=np.float64)
    if range_candidates.size < 3:
        raise ValueError("profile ML range grid must contain at least three points")
    range_axis = np.arange(values.shape[-1], dtype=np.float64)
    templates = np.sinc(range_axis[None, :] - range_candidates[:, None])
    template_norm = np.sum(templates**2, axis=-1)
    nfft = int(frequency_fft_factor) * values.shape[2]
    if nfft % 2:
        raise ValueError("frequency FFT size must be even")

    range_out = np.empty(len(values), dtype=np.float64)
    doppler_out = np.empty(len(values), dtype=np.float64)
    for start in range(0, len(values), chunk_size):
        chunk = np.asarray(values[start : start + chunk_size], dtype=np.complex128)
        # The three channel gains are profiled independently at each candidate r,f.
        projection = np.einsum("bcpr,qr->bcpq", chunk, templates, optimize=True)
        projection = projection.transpose(0, 3, 1, 2).reshape(-1, values.shape[1], values.shape[2])
        spectrum = np.fft.fftshift(np.fft.fft(projection, n=nfft, axis=-1), axes=-1)
        spectrum = spectrum.reshape(len(chunk), len(range_candidates), values.shape[1], nfft)
        score = np.sum(np.abs(spectrum) ** 2, axis=2, dtype=np.float64)
        score /= template_norm[None, :, None]
        winners = np.argmax(score.reshape(len(chunk), -1), axis=-1)
        q_index, f_index = np.divmod(winners, nfft)
        rows = np.arange(len(chunk))

        selected_range = range_candidates[q_index].copy()
        interior_range = (q_index > 0) & (q_index < len(range_candidates) - 1)
        if np.any(interior_range):
            idx = np.flatnonzero(interior_range)
            selected_range[idx] += range_grid_step * _log_parabolic_offset(
                score,
                score[idx, q_index[idx] - 1, f_index[idx]],
                score[idx, q_index[idx], f_index[idx]],
                score[idx, q_index[idx] + 1, f_index[idx]],
            )

        selected_doppler = (f_index.astype(np.float64) - nfft // 2) * prf_hz / nfft
        selected_doppler += (prf_hz / nfft) * _log_parabolic_offset(
            score,
            score[rows, q_index, (f_index - 1) % nfft],
            score[rows, q_index, f_index],
            score[rows, q_index, (f_index + 1) % nfft],
        )
        range_out[start : start + len(chunk)] = selected_range
        doppler_out[start : start + len(chunk)] = wrap_frequency(selected_doppler, prf_hz)
    return {"range_cell": range_out, "doppler_hz": doppler_out}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial probability."""
    successes = int(successes)
    total = int(total)
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("require 0 <= successes <= total and total > 0")
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("z must be positive and finite")
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total**2)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == total else min(1.0, center + radius)
    return lower, upper


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(np.count_nonzero(labels))
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[start:end] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[sorted_labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_detection_metrics(
    scores: Any,
    labels: Any,
    snr_db: Any,
    *,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    """Report ROC AUC and strict score-greater threshold counts overall/by SNR."""
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels, dtype=np.bool_).reshape(-1)
    snr_values = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if score_values.size == 0 or label_values.shape != score_values.shape or snr_values.shape != score_values.shape:
        raise ValueError("scores, labels, and snr_db must be nonempty arrays with matching shapes")
    if np.any(~np.isfinite(score_values)) or np.any(~np.isfinite(snr_values)):
        raise ValueError("scores and snr_db must be finite")
    if not thresholds:
        raise ValueError("at least one calibrated detection threshold is required")

    result: dict[str, Any] = {"overall": {}, "by_snr_db": {}}
    groups = [("overall", np.ones(len(score_values), dtype=np.bool_))]
    groups.extend(
        (str(int(snr)) if float(snr).is_integer() else str(float(snr)), snr_values == snr)
        for snr in sorted(np.unique(snr_values))
    )
    for group_name, mask in groups:
        group_scores = score_values[mask]
        group_labels = label_values[mask]
        summary: dict[str, Any] = {
            "n_h0": int(np.count_nonzero(~group_labels)),
            "n_h1": int(np.count_nonzero(group_labels)),
            "auc": _auc(group_scores, group_labels),
            "thresholds": {},
        }
        for key, threshold in thresholds.items():
            if not math.isfinite(float(threshold)):
                raise ValueError(f"threshold {key} must be finite")
            detected = group_scores > float(threshold)
            tp = int(np.count_nonzero(detected & group_labels))
            fp = int(np.count_nonzero(detected & ~group_labels))
            tn = int(np.count_nonzero(~detected & ~group_labels))
            fn = int(np.count_nonzero(~detected & group_labels))
            pd = tp / summary["n_h1"] if summary["n_h1"] else float("nan")
            pfa = fp / summary["n_h0"] if summary["n_h0"] else float("nan")
            summary["thresholds"][key] = {
                "threshold": float(threshold),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "pd": pd,
                "pfa": pfa,
                "pd_wilson_95": list(wilson_interval(tp, summary["n_h1"])) if summary["n_h1"] else None,
                "pfa_wilson_95": list(wilson_interval(fp, summary["n_h0"])) if summary["n_h0"] else None,
            }
        if group_name == "overall":
            result["overall"] = summary
        else:
            result["by_snr_db"][group_name] = summary
    return result


def parameter_error_metrics(
    truth_range_cell: Any,
    predicted_range_cell: Any,
    truth_doppler_hz: Any,
    predicted_doppler_hz: Any,
    *,
    prf_hz: float,
    outlier_threshold_hz: float = 50.0,
) -> dict[str, Any]:
    """Return parameter-error samples and summary with Doppler circular wrapping."""
    truth_range = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    predicted_range = np.asarray(predicted_range_cell, dtype=np.float64).reshape(-1)
    truth_doppler = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    predicted_doppler = np.asarray(predicted_doppler_hz, dtype=np.float64).reshape(-1)
    if truth_range.size == 0 or any(
        array.shape != truth_range.shape
        for array in (predicted_range, truth_doppler, predicted_doppler)
    ):
        raise ValueError("truth and predicted range/Doppler arrays must be nonempty and match")
    if any(np.any(~np.isfinite(array)) for array in (truth_range, predicted_range, truth_doppler, predicted_doppler)):
        raise ValueError("parameter arrays must be finite")
    if not math.isfinite(outlier_threshold_hz) or outlier_threshold_hz <= 0.0:
        raise ValueError("outlier_threshold_hz must be positive and finite")
    range_error = predicted_range - truth_range
    doppler_error = wrap_frequency(predicted_doppler - truth_doppler, prf_hz)
    return {
        "n": int(len(range_error)),
        "range_mae_cells": float(np.mean(np.abs(range_error))),
        "range_rmse_cells": float(np.sqrt(np.mean(range_error**2))),
        "doppler_mae_hz": float(np.mean(np.abs(doppler_error))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error**2))),
        "doppler_bias_hz": float(np.mean(doppler_error)),
        "doppler_outlier_rate": float(np.mean(np.abs(doppler_error) > outlier_threshold_hz)),
        "range_errors_cells": range_error.tolist(),
        "doppler_errors_hz": doppler_error.tolist(),
    }


def parameter_bootstrap_intervals(
    range_errors_cells: Any,
    doppler_errors_hz: Any,
    snr_db: Any,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Stratified paired sample bootstrap intervals for parameter error metrics."""
    range_error = np.asarray(range_errors_cells, dtype=np.float64).reshape(-1)
    doppler_error = np.asarray(doppler_errors_hz, dtype=np.float64).reshape(-1)
    snr_values = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if range_error.size == 0 or any(array.shape != range_error.shape for array in (doppler_error, snr_values)):
        raise ValueError("error arrays and snr_db must be nonempty and match")
    if any(np.any(~np.isfinite(array)) for array in (range_error, doppler_error, snr_values)):
        raise ValueError("errors and snr_db must be finite")
    if replicates <= 0 or seed < 0:
        raise ValueError("replicates must be positive and seed nonnegative")

    groups = {
        str(int(snr)) if float(snr).is_integer() else str(float(snr)): snr_values == snr
        for snr in sorted(np.unique(snr_values))
    }

    def point_summary(r_error: np.ndarray, d_error: np.ndarray) -> dict[str, float]:
        return {
            "range_mae_cells": float(np.mean(np.abs(r_error))),
            "range_rmse_cells": float(np.sqrt(np.mean(r_error**2))),
            "doppler_mae_hz": float(np.mean(np.abs(d_error))),
            "doppler_rmse_hz": float(np.sqrt(np.mean(d_error**2))),
            "doppler_bias_hz": float(np.mean(d_error)),
            "doppler_outlier_rate": float(np.mean(np.abs(d_error) > 50.0)),
        }

    def summarize_bootstrap(
        masks: list[np.ndarray], r_error: np.ndarray, d_error: np.ndarray
    ) -> dict[str, Any]:
        point = point_summary(r_error, d_error)
        replicates_count = int(replicates)
        values = {
            key: np.empty(replicates_count, dtype=np.float64)
            for key in point
        }
        rng = np.random.default_rng(seed)
        cursor = 0
        while cursor < replicates_count:
            batch_count = min(128, replicates_count - cursor)
            count_total = sum(int(np.count_nonzero(mask)) for mask in masks)
            sums = {key: np.zeros(batch_count, dtype=np.float64) for key in ("range_sq", "range_abs", "doppler_sq", "doppler_abs", "doppler_signed", "outliers")}
            for mask in masks:
                r_group = r_error[mask]
                d_group = d_error[mask]
                indices = rng.integers(0, len(r_group), size=(batch_count, len(r_group)))
                sampled_range = r_group[indices]
                sampled_doppler = d_group[indices]
                sums["range_sq"] += np.sum(sampled_range**2, axis=1)
                sums["range_abs"] += np.sum(np.abs(sampled_range), axis=1)
                sums["doppler_sq"] += np.sum(sampled_doppler**2, axis=1)
                sums["doppler_abs"] += np.sum(np.abs(sampled_doppler), axis=1)
                sums["doppler_signed"] += np.sum(sampled_doppler, axis=1)
                sums["outliers"] += np.sum(np.abs(sampled_doppler) > 50.0, axis=1)
            values["range_mae_cells"][cursor : cursor + batch_count] = sums["range_abs"] / count_total
            values["range_rmse_cells"][cursor : cursor + batch_count] = np.sqrt(sums["range_sq"] / count_total)
            values["doppler_mae_hz"][cursor : cursor + batch_count] = sums["doppler_abs"] / count_total
            values["doppler_rmse_hz"][cursor : cursor + batch_count] = np.sqrt(sums["doppler_sq"] / count_total)
            values["doppler_bias_hz"][cursor : cursor + batch_count] = sums["doppler_signed"] / count_total
            values["doppler_outlier_rate"][cursor : cursor + batch_count] = sums["outliers"] / count_total
            cursor += batch_count
        result: dict[str, Any] = {"n": int(len(r_error)), **point}
        for key, distribution in values.items():
            result[f"{key}_95_ci"] = [float(value) for value in np.quantile(distribution, [0.025, 0.975])]
        return result

    overall_masks = [snr_values == snr for snr in sorted(np.unique(snr_values))]
    overall = summarize_bootstrap(overall_masks, range_error, doppler_error)
    by_snr = {}
    for index, (key, mask) in enumerate(groups.items()):
        by_snr[key] = summarize_bootstrap([mask], range_error, doppler_error)
    return {"overall": overall, "by_snr_db": by_snr}


def paired_rmse_delta_interval(
    student_errors: Any,
    baseline_errors: Any,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    """Paired sample bootstrap CI for student RMSE minus a fixed baseline RMSE."""
    student = np.asarray(student_errors, dtype=np.float64).reshape(-1)
    baseline = np.asarray(baseline_errors, dtype=np.float64).reshape(-1)
    if student.size == 0 or student.shape != baseline.shape:
        raise ValueError("paired error arrays must be nonempty and have matching shapes")
    if np.any(~np.isfinite(student)) or np.any(~np.isfinite(baseline)):
        raise ValueError("paired error arrays must be finite")
    if replicates <= 0 or seed < 0:
        raise ValueError("replicates must be positive and seed nonnegative")
    delta = float(np.sqrt(np.mean(student**2)) - np.sqrt(np.mean(baseline**2)))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    cursor = 0
    while cursor < replicates:
        count = min(128, replicates - cursor)
        indices = rng.integers(0, len(student), size=(count, len(student)))
        student_rmse = np.sqrt(np.mean(student[indices] ** 2, axis=1))
        baseline_rmse = np.sqrt(np.mean(baseline[indices] ** 2, axis=1))
        draws[cursor : cursor + count] = student_rmse - baseline_rmse
        cursor += count
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {"delta_rmse": delta, "lower_95": float(lower), "upper_95": float(upper)}


def _git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _model_checkpoint_paths(config: dict[str, Any], output: Path) -> list[Path]:
    paths = []
    for seed in config["splits"]["formal"]["training_seeds"]:
        for method in config["training"]["methods"]:
            paths.append(
                output
                / "formal"
                / str(seed["id"])
                / method
                / "best_inference.pt"
            )
    return paths


def _freeze_manifest(config: dict[str, Any], root: Path, output: Path) -> dict[str, Any]:
    code_paths = [
        Path("src/radardistill/evaluation/d1_evaluation.py"),
        Path("scripts/evaluate_d1_e001.py"),
    ]
    input_paths = [
        Path("experiments/D1-E001/config.yaml"),
        Path("experiments/D1-E001/protocol.md"),
        Path("experiments/D1-E001/config_used.yaml"),
    ]
    required = [root / path for path in code_paths + input_paths] + _model_checkpoint_paths(config, output)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"D1 blind-evaluation freeze inputs are missing: {missing}")
    relative = [path.relative_to(root).as_posix() for path in required]
    dirty = _git_output(root, "status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError("D1 evaluation code, config, or checkpoint files are not cleanly committed")
    untracked = [
        path
        for path in relative
        if not _git_output(root, "ls-files", "--error-unmatch", "--", path)
    ]
    if untracked:
        raise RuntimeError(f"D1 evaluation freeze files must be tracked by git: {untracked}")
    code_commit = _git_output(root, "log", "-1", "--format=%H", "--", *(p.as_posix() for p in code_paths))
    blob_oids = {
        path: _git_output(root, "hash-object", path)
        for path in relative
    }
    return {"code_commit": code_commit, "blob_oids": blob_oids}


def _assert_frozen_config(config: dict[str, Any], root: Path, output: Path) -> None:
    import yaml

    source = yaml.safe_load((root / "experiments/D1-E001/config.yaml").read_text(encoding="utf-8"))
    effective_path = output / "config_used.yaml"
    if not effective_path.is_file():
        raise FileNotFoundError(f"frozen effective config is missing: {effective_path}")
    effective = yaml.safe_load(effective_path.read_text(encoding="utf-8"))
    if config != source or config != effective:
        raise ValueError("evaluation config must match both committed config.yaml and config_used.yaml")


def _require_clean_tracked_paths(root: Path, paths: list[Path]) -> dict[str, str]:
    relative = [path.resolve().relative_to(root).as_posix() for path in paths]
    dirty = _git_output(root, "status", "--porcelain", "--", *relative)
    if dirty:
        raise RuntimeError(f"required D1 freeze artifacts must be committed and clean: {dirty}")
    result = {}
    for path in relative:
        _git_output(root, "ls-files", "--error-unmatch", "--", path)
        result[path] = _git_output(root, "hash-object", path)
    return result


def _verify_freeze_manifest(
    config: dict[str, Any], root: Path, output: Path, manifest: dict[str, Any]
) -> None:
    current = _freeze_manifest(config, root, output)
    if current != manifest:
        raise RuntimeError(
            "D1 frozen code/config/checkpoints differ from calibration; do not open blind data"
        )


def _analysis_replay_changed_paths(
    frozen: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    """Allow the evaluator correction and its replay-only CLI wiring."""
    frozen_blobs = frozen.get("blob_oids", {})
    current_blobs = current.get("blob_oids", {})
    if set(frozen_blobs) != set(current_blobs):
        raise RuntimeError("analysis replay manifest paths differ from the prediction freeze")
    changed = sorted(
        path for path in frozen_blobs if frozen_blobs[path] != current_blobs[path]
    )
    allowed = {
        "src/radardistill/evaluation/d1_evaluation.py",
        "scripts/evaluate_d1_e001.py",
    }
    unexpected = sorted(set(changed) - allowed)
    if unexpected:
        raise RuntimeError(
            "analysis replay found changed config, protocol, CLI, or model checkpoints: "
            f"{unexpected}"
        )
    if frozen.get("code_commit") != current.get("code_commit") and not changed:
        raise RuntimeError("analysis replay code commit changed without an evaluator blob change")
    return changed


def _require_completed_blind_predictions(
    partial: dict[str, Any], config: dict[str, Any], prediction_dir: Path
) -> list[str]:
    expected = sorted(
        f"{seed['id']}/{method}"
        for seed in config["splits"]["formal"]["training_seeds"]
        for method in config["training"]["methods"]
    )
    if (
        partial.get("phase") != "final_blind"
        or partial.get("status") != "RUNNING"
        or sorted(partial.get("completed_student_methods", [])) != expected
        or not partial.get("classical_baselines_completed")
    ):
        raise RuntimeError("analysis replay requires a failed run with every prediction completed")
    expected_paths = [
        prediction_dir / f"{method.replace('/', '__')}.npz" for method in expected
    ]
    expected_paths.append(prediction_dir / "classical_baselines.npz")
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"analysis replay prediction artifacts are missing: {missing}")
    return expected


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _evaluation_environment(device, batch_size: int, elapsed_seconds: float) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        driver = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device": properties.name,
        "cuda_driver": driver,
        "cuda_total_memory_bytes": int(properties.total_memory),
        "evaluation_device": str(device),
        "precision": "float32",
        "inference_batch_size": int(batch_size),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _load_evaluation_dependencies():
    from ..simulation.d0 import generate_d0_batch
    from ..teachers.apx15_doppler import APX15DopplerTeacher
    from ..training.d1_runner import calibrate_score_threshold, require_cuda

    return generate_d0_batch, APX15DopplerTeacher, calibrate_score_threshold, require_cuda


def _iter_generated_batches(
    generate_d0_batch,
    d0_config: dict[str, Any],
    *,
    total: int,
    snr_db: float,
    presence: float,
    seed_base: int,
    snr_index: int,
    snr_stride: int,
    snr_count: int,
    batch_size: int,
):
    offset = 0
    chunk_index = 0
    while offset < total:
        count = min(batch_size, total - offset)
        seed = evaluation_chunk_seed(
            seed_base,
            snr_index,
            chunk_index,
            snr_stride=snr_stride,
            snr_count=snr_count,
        )
        yield offset, generate_d0_batch(count, seed, snr_db, presence, d0_config)
        offset += count
        chunk_index += 1


def _load_student_checkpoint(checkpoint: Path, config: dict[str, Any], device):
    import torch

    from ..students.task_student import TaskStudent

    model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=False,
    )
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or "state_dict" not in artifact:
        raise ValueError(f"checkpoint {checkpoint} is not a D1 inference checkpoint envelope")
    checkpoint_student = artifact.get("student")
    if not isinstance(checkpoint_student, dict):
        raise ValueError(f"checkpoint {checkpoint} lacks its student configuration")
    if (
        int(checkpoint_student.get("width", -1)) != int(config["student"]["width"])
        or tuple(checkpoint_student.get("temporal_dilations", ()))
        != tuple(config["student"]["temporal_dilations"])
    ):
        raise ValueError(f"checkpoint {checkpoint} student configuration differs from frozen config")
    state = artifact["state_dict"]
    incompatibility = model.load_state_dict(state, strict=True)
    if incompatibility.missing_keys or incompatibility.unexpected_keys:
        raise RuntimeError(f"checkpoint {checkpoint} does not match the inference student")
    model.to(device)
    model.eval()
    return model


def _predict_student(model, iq: np.ndarray, device, prf_hz: float) -> dict[str, np.ndarray]:
    import torch

    from ..distillation.task_losses import decode_task_outputs

    values = np.asarray(iq, dtype=np.complex64)
    planes = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    inputs = torch.from_numpy(np.ascontiguousarray(planes)).to(device)
    with torch.no_grad():
        outputs = model(inputs, compute_spectrum=False)
        decoded = decode_task_outputs(outputs, prf_hz)
    return {
        "score": outputs["presence_logit"].detach().cpu().numpy().astype(np.float64),
        "range_cell": decoded["range_cell"].detach().cpu().numpy().astype(np.float64),
        "doppler_hz": decoded["doppler_hz"].detach().cpu().numpy().astype(np.float64),
    }


def _collect_student_calibration_scores(
    model,
    *,
    config: dict[str, Any],
    d0_config: dict[str, Any],
    generate_d0_batch,
    batch_size: int,
) -> np.ndarray:
    split = config["splits"]["detection_calibration"]
    snr_values = [float(value) for value in split["snr_db"]]
    per_snr = int(split["h0_examples_per_snr"])
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    for snr_index, snr in enumerate(snr_values):
        for _, batch in _iter_generated_batches(
            generate_d0_batch,
            d0_config,
            total=per_snr,
            snr_db=snr,
            presence=0.0,
            seed_base=int(split["seed_base"]),
            snr_index=snr_index,
            snr_stride=int(split["seed_stride_by_snr_index"]),
            snr_count=len(snr_values),
            batch_size=batch_size,
        ):
            chunks.append(
                _predict_student(model, batch.iq, device, float(config["radar"]["prf_hz"]))["score"]
            )
    return np.concatenate(chunks)


def _calibration_thresholds(scores: np.ndarray, config: dict[str, Any], calibrate_score_threshold) -> dict[str, Any]:
    pfa_values = [float(value) for value in config["evaluation"]["threshold_calibration"]["false_alarm_operating_points"]]
    return {
        f"{pfa:g}": {
            "threshold": calibrate_score_threshold(scores, pfa),
            "n_h0": int(len(scores)),
            "comparison": "score_strictly_greater_than_threshold",
        }
        for pfa in pfa_values
    }


def run_d1_calibration(
    config: dict[str, Any],
    *,
    root: str | Path,
    output: str | Path,
    device: str = "cuda:0",
    batch_size: int = 128,
) -> dict[str, Any]:
    """Calibrate all student and APX15 detection scores before blind evaluation."""
    import torch
    from ..training.d1_runner import _load_d0_config

    generate_d0_batch, teacher_type, calibrate_score_threshold, require_cuda = _load_evaluation_dependencies()
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    _assert_frozen_config(config, root_path, output_path)
    calibration_dir = output_path / "calibration"
    metrics_path = calibration_dir / "metrics.json"
    if metrics_path.exists() or (calibration_dir / "metrics_partial.json").exists():
        raise FileExistsError(f"calibration artifacts already exist in {calibration_dir}")
    cuda_device = require_cuda(device)
    started_at = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(cuda_device)
    manifest = _freeze_manifest(config, root_path, output_path)
    d0_config = _load_d0_config(root_path)
    thresholds_by_method: dict[str, Any] = {}
    partial: dict[str, Any] = {
        "experiment_id": "D1-E001",
        "phase": "detection_calibration",
        "status": "RUNNING",
        "freeze_manifest": manifest,
        "methods": thresholds_by_method,
    }
    calibration_dir.mkdir(parents=True, exist_ok=True)
    teacher = teacher_type(d0_config)
    try:
        split = config["splits"]["detection_calibration"]
        snr_values = [float(value) for value in split["snr_db"]]
        teacher_chunks = []
        for snr_index, snr in enumerate(snr_values):
            for _, batch in _iter_generated_batches(
                generate_d0_batch,
                d0_config,
                total=int(split["h0_examples_per_snr"]),
                snr_db=snr,
                presence=0.0,
                seed_base=int(split["seed_base"]),
                snr_index=snr_index,
                snr_stride=int(split["seed_stride_by_snr_index"]),
                snr_count=len(snr_values),
                batch_size=batch_size,
            ):
                teacher_chunks.append(teacher.predict(batch.iq).teacher_detection_score)
        teacher_scores = np.concatenate(teacher_chunks)
        thresholds_by_method["APX15_teacher"] = _calibration_thresholds(
            teacher_scores, config, calibrate_score_threshold
        )
        _write_json(calibration_dir / "metrics_partial.json", partial)

        for checkpoint in _model_checkpoint_paths(config, output_path):
            relative_method = checkpoint.relative_to(output_path / "formal").as_posix()
            method_key = relative_method.rsplit("/", 1)[0]
            model = _load_student_checkpoint(checkpoint, config, cuda_device)
            try:
                scores = _collect_student_calibration_scores(
                    model,
                    config=config,
                    d0_config=d0_config,
                    generate_d0_batch=generate_d0_batch,
                    batch_size=batch_size,
                )
                thresholds_by_method[method_key] = _calibration_thresholds(
                    scores, config, calibrate_score_threshold
                )
                partial["completed_methods"] = list(thresholds_by_method)
                _write_json(calibration_dir / "metrics_partial.json", partial)
            finally:
                del model
                torch.cuda.empty_cache()
        partial["status"] = "COMPLETED"
        partial["calibration_samples_per_method"] = int(
            config["splits"]["detection_calibration"]["h0_examples_total"]
        )
        partial["training_device"] = str(cuda_device)
        partial["precision"] = "float32"
        partial["evaluation_environment"] = _evaluation_environment(
            cuda_device, batch_size, time.perf_counter() - started_at
        )
        _write_json(metrics_path, partial)
        (calibration_dir / "metrics_partial.json").unlink(missing_ok=True)
        return partial
    finally:
        teacher.close()


def _snr_key(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(float(value))


def _save_prediction_arrays(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_prediction_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as contents:
        return {name: np.asarray(contents[name]) for name in contents.files}


def _collect_student_blind_predictions(
    model,
    *,
    config: dict[str, Any],
    d0_config: dict[str, Any],
    generate_d0_batch,
    batch_size: int,
) -> dict[str, np.ndarray]:
    split = config["splits"]["final_blind"]
    snr_values = [float(value) for value in config["data"]["evaluation_snr_db"]]
    stride = int(split["seed_stride_by_snr_index"])
    snr_count = len(snr_values)
    result: dict[str, np.ndarray] = {}
    device = next(model.parameters()).device
    prf_hz = float(config["radar"]["prf_hz"])
    for snr_index, snr in enumerate(snr_values):
        key = _snr_key(snr)
        h0_scores = []
        for _, batch in _iter_generated_batches(
            generate_d0_batch,
            d0_config,
            total=int(split["h0_examples_per_snr"]),
            snr_db=snr,
            presence=0.0,
            seed_base=int(split["seed_base_h0"]),
            snr_index=snr_index,
            snr_stride=stride,
            snr_count=snr_count,
            batch_size=batch_size,
        ):
            h0_scores.append(_predict_student(model, batch.iq, device, prf_hz)["score"])

        h1_scores = []
        pred_range = []
        pred_doppler = []
        truth_range = []
        truth_doppler = []
        target_amplitude = []
        for _, batch in _iter_generated_batches(
            generate_d0_batch,
            d0_config,
            total=int(split["h1_examples_per_snr"]),
            snr_db=snr,
            presence=1.0,
            seed_base=int(split["seed_base_h1"]),
            snr_index=snr_index,
            snr_stride=stride,
            snr_count=snr_count,
            batch_size=batch_size,
        ):
            prediction = _predict_student(model, batch.iq, device, prf_hz)
            h1_scores.append(prediction["score"])
            pred_range.append(prediction["range_cell"])
            pred_doppler.append(prediction["doppler_hz"])
            truth_range.append(np.asarray(batch.truth_range_cell, dtype=np.float64))
            truth_doppler.append(np.asarray(batch.truth_doppler_hz, dtype=np.float64))
            target_amplitude.append(np.asarray(batch.truth_target_amplitude, dtype=np.float64))
        result[f"h0_score__{key}"] = np.concatenate(h0_scores)
        result[f"h1_score__{key}"] = np.concatenate(h1_scores)
        result[f"pred_range_cell__{key}"] = np.concatenate(pred_range)
        result[f"pred_doppler_hz__{key}"] = np.concatenate(pred_doppler)
        result[f"truth_range_cell__{key}"] = np.concatenate(truth_range)
        result[f"truth_doppler_hz__{key}"] = np.concatenate(truth_doppler)
        result[f"truth_target_amplitude__{key}"] = np.concatenate(target_amplitude)
    return result


def _collect_classical_blind_predictions(
    *,
    config: dict[str, Any],
    d0_config: dict[str, Any],
    generate_d0_batch,
    teacher,
    batch_size: int,
) -> dict[str, np.ndarray]:
    split = config["splits"]["final_blind"]
    snr_values = [float(value) for value in config["data"]["evaluation_snr_db"]]
    stride = int(split["seed_stride_by_snr_index"])
    snr_count = len(snr_values)
    prf_hz = float(config["radar"]["prf_hz"])
    profile_count = int(config["splits"]["profile_ml_subset"]["h1_examples_per_snr"])
    methods: dict[str, list[np.ndarray]] = {}
    results: dict[str, np.ndarray] = {}
    for snr_index, snr in enumerate(snr_values):
        key = _snr_key(snr)
        h0_scores = []
        for _, batch in _iter_generated_batches(
            generate_d0_batch,
            d0_config,
            total=int(split["h0_examples_per_snr"]),
            snr_db=snr,
            presence=0.0,
            seed_base=int(split["seed_base_h0"]),
            snr_index=snr_index,
            snr_stride=stride,
            snr_count=snr_count,
            batch_size=batch_size,
        ):
            teacher_batch = teacher.predict(batch.iq)
            h0_scores.append(teacher_batch.teacher_detection_score)
        results[f"teacher_h0_score__{key}"] = np.concatenate(h0_scores)

        h1_teacher_scores = []
        h1_ranges = []
        h1_dopplers = []
        h1_truth_ranges = []
        h1_truth_dopplers = []
        h1_amplitudes = []
        for offset, batch in _iter_generated_batches(
            generate_d0_batch,
            d0_config,
            total=int(split["h1_examples_per_snr"]),
            snr_db=snr,
            presence=1.0,
            seed_base=int(split["seed_base_h1"]),
            snr_index=snr_index,
            snr_stride=stride,
            snr_count=snr_count,
            batch_size=batch_size,
        ):
            teacher_batch = teacher.predict(batch.iq)
            estimates = estimate_fft_baselines(batch.iq, prf_hz)
            phase_doppler = estimate_phase_difference_multichannel(batch.iq, prf_hz)
            h1_teacher_scores.append(teacher_batch.teacher_detection_score)
            h1_ranges.append(teacher_batch.teacher_range_cell)
            h1_dopplers.append(teacher_batch.teacher_doppler_hz)
            h1_truth_ranges.append(np.asarray(batch.truth_range_cell, dtype=np.float64))
            h1_truth_dopplers.append(np.asarray(batch.truth_doppler_hz, dtype=np.float64))
            h1_amplitudes.append(np.asarray(batch.truth_target_amplitude, dtype=np.float64))
            for method, prediction in estimates.items():
                methods.setdefault(f"{method}_range_cell__{key}", []).append(prediction["range_cell"])
                methods.setdefault(f"{method}_doppler_hz__{key}", []).append(prediction["doppler_hz"])
            methods.setdefault(f"phase_difference_range_cell__{key}", []).append(
                estimates["native_fft_peak"]["range_cell"]
            )
            methods.setdefault(f"phase_difference_doppler_hz__{key}", []).append(phase_doppler)
            if offset < profile_count:
                profile_batch = batch.iq[: min(len(batch.iq), profile_count - offset)]
                profile = estimate_profile_ml_reference(
                    profile_batch,
                    prf_hz=prf_hz,
                    range_interval=tuple(config["radar"]["target_range_cell_interval"]),
                    range_grid_step=float(config["baselines"]["profile_likelihood"]["range_grid_step_cells"]),
                    frequency_fft_factor=int(config["baselines"]["profile_likelihood"]["frequency_fft_factor"]),
                    chunk_size=4,
                )
                methods.setdefault(f"profile_ml_range_cell__{key}", []).append(profile["range_cell"])
                methods.setdefault(f"profile_ml_doppler_hz__{key}", []).append(profile["doppler_hz"])
                methods.setdefault(f"profile_ml_truth_range_cell__{key}", []).append(
                    np.asarray(batch.truth_range_cell[: len(profile_batch)], dtype=np.float64)
                )
                methods.setdefault(f"profile_ml_truth_doppler_hz__{key}", []).append(
                    np.asarray(batch.truth_doppler_hz[: len(profile_batch)], dtype=np.float64)
                )
        results[f"teacher_h1_score__{key}"] = np.concatenate(h1_teacher_scores)
        results[f"teacher_range_cell__{key}"] = np.concatenate(h1_ranges)
        results[f"teacher_doppler_hz__{key}"] = np.concatenate(h1_dopplers)
        results[f"truth_range_cell__{key}"] = np.concatenate(h1_truth_ranges)
        results[f"truth_doppler_hz__{key}"] = np.concatenate(h1_truth_dopplers)
        results[f"truth_target_amplitude__{key}"] = np.concatenate(h1_amplitudes)
    for name, chunks in methods.items():
        results[name] = np.concatenate(chunks)
    return results


def doppler_crlb_hz(
    truth_range_cell: Any,
    snr_db: Any,
    *,
    prf_hz: float,
    pulse_count: int,
    channel_count: int = 3,
) -> np.ndarray:
    """Ideal conditional frequency CRLB standard deviation for the frozen D0 model."""
    range_values = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    snr_values = np.asarray(snr_db, dtype=np.float64).reshape(-1)
    if range_values.size == 0 or range_values.shape != snr_values.shape:
        raise ValueError("truth range and SNR must be nonempty arrays with matching shapes")
    if np.any(~np.isfinite(range_values)) or np.any(~np.isfinite(snr_values)):
        raise ValueError("truth range and SNR must be finite")
    if not math.isfinite(prf_hz) or prf_hz <= 0 or pulse_count < 2 or channel_count <= 0:
        raise ValueError("PRF, pulse_count, and channel_count are invalid")
    range_energy = np.sum(
        np.sinc(np.arange(32, dtype=np.float64)[None, :] - range_values[:, None]) ** 2,
        axis=1,
    )
    pulse_index = np.arange(pulse_count, dtype=np.float64)
    centered_pulse_energy = float(np.sum((pulse_index - np.mean(pulse_index)) ** 2))
    snr_linear = np.power(10.0, snr_values / 10.0)
    denominator = (
        8.0
        * np.pi**2
        * snr_linear
        * channel_count
        * range_energy
        * centered_pulse_energy
    )
    return prf_hz / np.sqrt(denominator)


def _point_parameter_summary(range_error: np.ndarray, doppler_error: np.ndarray) -> dict[str, Any]:
    result = parameter_error_metrics(
        np.zeros_like(range_error),
        range_error,
        np.zeros_like(doppler_error),
        doppler_error,
        prf_hz=1300.0,
    )
    result.pop("range_errors_cells")
    result.pop("doppler_errors_hz")
    return result


def _parameter_groups(
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    target_amplitude: np.ndarray,
    range_interval: tuple[float, float],
    prf_hz: float,
) -> dict[str, np.ndarray]:
    low_range, high_range = range_interval
    range_edges = np.linspace(low_range, high_range, 5)
    groups: dict[str, np.ndarray] = {}
    for index in range(4):
        upper_ok = truth_range < range_edges[index + 1] if index < 3 else truth_range <= range_edges[index + 1]
        groups[f"range_q{index + 1}"] = (truth_range >= range_edges[index]) & upper_ok
    half_prf = prf_hz / 2.0
    neighborhood = 50.0
    groups["doppler_negative_nyquist_neighborhood"] = truth_doppler <= -half_prf + neighborhood
    groups["doppler_positive_nyquist_neighborhood"] = truth_doppler >= half_prf - neighborhood
    groups["doppler_negative_interior"] = (truth_doppler < 0.0) & ~groups["doppler_negative_nyquist_neighborhood"]
    groups["doppler_positive_interior"] = (truth_doppler >= 0.0) & ~groups["doppler_positive_nyquist_neighborhood"]
    quantiles = np.quantile(target_amplitude, [0.25, 0.5, 0.75])
    amp_edges = [-np.inf, *quantiles.tolist(), np.inf]
    for index in range(4):
        if index == 3:
            groups[f"amplitude_q{index + 1}"] = (target_amplitude >= amp_edges[index]) & (target_amplitude <= amp_edges[index + 1])
        else:
            groups[f"amplitude_q{index + 1}"] = (target_amplitude >= amp_edges[index]) & (target_amplitude < amp_edges[index + 1])
    return groups


def _summarize_estimation(
    by_snr: dict[str, dict[str, np.ndarray]],
    *,
    config: dict[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    snr_order = [_snr_key(float(value)) for value in config["data"]["evaluation_snr_db"]]
    available_snrs = [key for key in snr_order if len(by_snr[key]["truth_range_cell"]) > 0]
    if not available_snrs:
        empty = {
            "n": 0,
            "range_mae_cells": None,
            "range_rmse_cells": None,
            "doppler_mae_hz": None,
            "doppler_rmse_hz": None,
            "doppler_bias_hz": None,
            "doppler_outlier_rate": None,
            "confidence_intervals": {},
        }
        return {
            "overall": empty,
            "by_snr_db": {key: dict(empty) for key in snr_order},
            "groups": {},
            "crlb": {"rmse_reference_hz": None, "by_snr_db": {key: {"n": 0, "mean_hz": None, "median_hz": None} for key in snr_order}},
        }
    truth_range = np.concatenate([by_snr[key]["truth_range_cell"] for key in available_snrs])
    truth_doppler = np.concatenate([by_snr[key]["truth_doppler_hz"] for key in available_snrs])
    pred_range = np.concatenate([by_snr[key]["pred_range_cell"] for key in available_snrs])
    pred_doppler = np.concatenate([by_snr[key]["pred_doppler_hz"] for key in available_snrs])
    snr_values = np.concatenate(
        [np.full(len(by_snr[key]["truth_range_cell"]), float(key)) for key in available_snrs]
    )
    range_error = pred_range - truth_range
    doppler_error = wrap_frequency(
        pred_doppler - truth_doppler, float(config["radar"]["prf_hz"])
    )
    reps = int(config["evaluation"]["uncertainty"]["bootstrap_replicates"])
    bootstrap = parameter_bootstrap_intervals(
        range_error,
        doppler_error,
        snr_values,
        replicates=reps,
        seed=int(bootstrap_seed),
    )
    overall = _point_parameter_summary(range_error, doppler_error)
    overall["confidence_intervals"] = {
        key: value
        for key, value in bootstrap["overall"].items()
        if key.endswith("_95_ci")
    }
    per_snr: dict[str, Any] = {}
    cursor = 0
    for key in snr_order:
        count = len(by_snr[key]["truth_range_cell"])
        if count == 0:
            per_snr[key] = {
                "n": 0,
                "range_mae_cells": None,
                "range_rmse_cells": None,
                "doppler_mae_hz": None,
                "doppler_rmse_hz": None,
                "doppler_bias_hz": None,
                "doppler_outlier_rate": None,
                "confidence_intervals": {},
            }
            continue
        summary = _point_parameter_summary(
            range_error[cursor : cursor + count], doppler_error[cursor : cursor + count]
        )
        summary["confidence_intervals"] = {
            name: value
            for name, value in bootstrap["by_snr_db"][key].items()
            if name.endswith("_95_ci")
        }
        per_snr[key] = summary
        cursor += count

    amplitudes = np.concatenate(
        [by_snr[key]["truth_target_amplitude"] for key in available_snrs]
    )
    masks = _parameter_groups(
        truth_range,
        truth_doppler,
        amplitudes,
        tuple(float(x) for x in config["radar"]["target_range_cell_interval"]),
        float(config["radar"]["prf_hz"]),
    )
    grouped = {
        name: _point_parameter_summary(range_error[mask], doppler_error[mask])
        for name, mask in masks.items()
        if np.any(mask)
    }
    crlb = doppler_crlb_hz(
        truth_range,
        snr_values,
        prf_hz=float(config["radar"]["prf_hz"]),
        pulse_count=int(config["radar"]["pulse_count"]),
        channel_count=int(config["radar"]["channel_count"]),
    )
    crlb_summary = {
        "rmse_reference_hz": float(np.sqrt(np.mean(crlb**2))),
        "by_snr_db": {
            key: {
                "n": int(np.count_nonzero(snr_values == float(key))),
                "mean_hz": float(np.mean(crlb[snr_values == float(key)])),
                "median_hz": float(np.median(crlb[snr_values == float(key)])),
            }
            if np.any(snr_values == float(key))
            else {"n": 0, "mean_hz": None, "median_hz": None}
            for key in snr_order
        },
    }
    return {"overall": overall, "by_snr_db": per_snr, "groups": grouped, "crlb": crlb_summary}


def _conditional_detection_estimation(
    by_snr: dict[str, dict[str, np.ndarray]],
    *,
    thresholds: dict[str, Any],
    config: dict[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    snr_order = [_snr_key(float(value)) for value in config["data"]["evaluation_snr_db"]]
    score_threshold = float(thresholds["0.01"]["threshold"])
    conditional: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, dict[str, int]] = {}
    for snr in snr_order:
        values = by_snr[snr]
        detected = np.asarray(values["h1_score"], dtype=np.float64) > score_threshold
        counts[snr] = {
            "n_h1": int(len(detected)),
            "detected": int(np.count_nonzero(detected)),
            "missed": int(np.count_nonzero(~detected)),
        }
        conditional[snr] = {
            name: np.asarray(values[name])[detected]
            for name in (
                "pred_range_cell",
                "pred_doppler_hz",
                "truth_range_cell",
                "truth_doppler_hz",
                "truth_target_amplitude",
            )
        }
    summary = _summarize_estimation(
        conditional,
        config=config,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        **summary,
        "threshold_pfa_1e_2": score_threshold,
        "counts_by_snr_db": counts,
    }


def roc_curve_points(scores: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    label_values = np.asarray(labels, dtype=np.bool_).reshape(-1)
    if score_values.size == 0 or score_values.shape != label_values.shape:
        raise ValueError("scores and labels must be nonempty arrays with matching shapes")
    if np.any(~np.isfinite(score_values)) or not np.any(label_values) or np.all(label_values):
        raise ValueError("ROC requires finite scores and both classes")
    order = np.argsort(-score_values, kind="mergesort")
    sorted_scores = score_values[order]
    sorted_labels = label_values[order]
    cumulative_tp = np.cumsum(sorted_labels)
    cumulative_fp = np.cumsum(~sorted_labels)
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tpr = np.r_[0.0, cumulative_tp[distinct] / np.count_nonzero(label_values)]
    fpr = np.r_[0.0, cumulative_fp[distinct] / np.count_nonzero(~label_values)]
    return fpr, tpr


def _detection_method_summary(
    h0_scores_by_snr: dict[str, np.ndarray],
    h1_scores_by_snr: dict[str, np.ndarray],
    *,
    thresholds: dict[str, float],
    snr_order: list[str],
) -> dict[str, Any]:
    scores = []
    labels = []
    snr = []
    for key in snr_order:
        h0 = np.asarray(h0_scores_by_snr[key], dtype=np.float64)
        h1 = np.asarray(h1_scores_by_snr[key], dtype=np.float64)
        scores.extend((h0, h1))
        labels.extend((np.zeros(len(h0), dtype=np.bool_), np.ones(len(h1), dtype=np.bool_)))
        snr.extend((np.full(len(h0), float(key)), np.full(len(h1), float(key))))
    return binary_detection_metrics(
        np.concatenate(scores),
        np.concatenate(labels),
        np.concatenate(snr),
        thresholds=thresholds,
    )


def _joined_student_predictions(path: Path, snr_order: list[str]) -> dict[str, dict[str, np.ndarray]]:
    arrays = _load_prediction_arrays(path)
    return {
        key: {
            name: arrays[f"{name}__{key}"]
            for name in (
                "h0_score",
                "h1_score",
                "pred_range_cell",
                "pred_doppler_hz",
                "truth_range_cell",
                "truth_doppler_hz",
                "truth_target_amplitude",
            )
            if f"{name}__{key}" in arrays
        }
        for key in snr_order
    }


def _joined_classical_predictions(path: Path, snr_order: list[str]) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    arrays = _load_prediction_arrays(path)
    methods: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    names = (
        "APX15_teacher",
        "native_fft_peak",
        "native_fft_log_parabolic",
        "fourfold_fft_parabolic",
        "phase_difference",
        "profile_ml",
    )
    for name in names:
        methods[name] = {}
        for key in snr_order:
            fields = {
                "h0_score": arrays.get(f"teacher_h0_score__{key}"),
                "h1_score": arrays.get(f"teacher_h1_score__{key}"),
                "truth_range_cell": arrays.get(f"truth_range_cell__{key}"),
                "truth_doppler_hz": arrays.get(f"truth_doppler_hz__{key}"),
                "truth_target_amplitude": arrays.get(f"truth_target_amplitude__{key}"),
            }
            if name == "APX15_teacher":
                fields["pred_range_cell"] = arrays[f"teacher_range_cell__{key}"]
                fields["pred_doppler_hz"] = arrays[f"teacher_doppler_hz__{key}"]
            elif name == "profile_ml":
                fields["pred_range_cell"] = arrays[f"profile_ml_range_cell__{key}"]
                fields["pred_doppler_hz"] = arrays[f"profile_ml_doppler_hz__{key}"]
                fields["truth_range_cell"] = arrays[f"profile_ml_truth_range_cell__{key}"]
                fields["truth_doppler_hz"] = arrays[f"profile_ml_truth_doppler_hz__{key}"]
                fields["truth_target_amplitude"] = arrays[
                    f"truth_target_amplitude__{key}"
                ][: len(fields["truth_range_cell"])]
                fields["h0_score"] = None
                fields["h1_score"] = None
            else:
                fields["pred_range_cell"] = arrays[f"{name}_range_cell__{key}"]
                fields["pred_doppler_hz"] = arrays[f"{name}_doppler_hz__{key}"]
                fields["h0_score"] = None
                fields["h1_score"] = None
            methods[name][key] = {
                field: value for field, value in fields.items() if value is not None
            }
    return methods


def _build_blind_metrics(
    *,
    student_predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    classical_predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    snr_order = [_snr_key(float(value)) for value in config["data"]["evaluation_snr_db"]]
    bootstrap_seed = int(config["evaluation"]["uncertainty"]["bootstrap_seed"])
    detection: dict[str, Any] = {}
    estimation: dict[str, Any] = {}
    for method_key, by_snr in student_predictions.items():
        thresholds = {
            key: float(value["threshold"])
            for key, value in calibration["methods"][method_key].items()
        }
        detection[method_key] = _detection_method_summary(
            {snr: by_snr[snr]["h0_score"] for snr in snr_order},
            {snr: by_snr[snr]["h1_score"] for snr in snr_order},
            thresholds=thresholds,
            snr_order=snr_order,
        )
        estimation[method_key] = _summarize_estimation(
            by_snr,
            config=config,
            bootstrap_seed=bootstrap_seed,
        )
        estimation[method_key]["conditional_on_detection_pfa_1e_2"] = (
            _conditional_detection_estimation(
                by_snr,
                thresholds=calibration["methods"][method_key],
                config=config,
                bootstrap_seed=bootstrap_seed + 10,
            )
        )
    for method_key, by_snr in classical_predictions.items():
        if method_key == "APX15_teacher":
            thresholds = {
                key: float(value["threshold"])
                for key, value in calibration["methods"][method_key].items()
            }
            detection[method_key] = _detection_method_summary(
                {snr: by_snr[snr]["h0_score"] for snr in snr_order},
                {snr: by_snr[snr]["h1_score"] for snr in snr_order},
                thresholds=thresholds,
                snr_order=snr_order,
            )
        estimation[method_key] = _summarize_estimation(
            by_snr,
            config=config,
            bootstrap_seed=bootstrap_seed,
        )
        if method_key == "APX15_teacher":
            estimation[method_key]["conditional_on_detection_pfa_1e_2"] = (
                _conditional_detection_estimation(
                    by_snr,
                    thresholds=calibration["methods"][method_key],
                    config=config,
                    bootstrap_seed=bootstrap_seed + 10,
                )
            )
    crlb = estimation[next(iter(estimation))]["crlb"] if estimation else {}
    return {
        "detection": detection,
        "estimation": estimation,
        "crlb": crlb,
        "student_method_keys": list(student_predictions),
        "classical_method_keys": list(classical_predictions),
    }


def _joint_endpoint(
    by_snr_errors: dict[str, tuple[np.ndarray, np.ndarray]], snr_order: list[str]
) -> float:
    values = []
    for snr in snr_order:
        range_error, doppler_error = by_snr_errors[snr]
        values.append(0.5 * (np.sqrt(np.mean(range_error**2)) / 24.0 + np.sqrt(np.mean(doppler_error**2)) / 1300.0))
    return float(np.mean(values))


def _student_error_samples(
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    method_key: str,
    snr_key: str,
    *,
    prf_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = predictions[method_key][snr_key]
    return (
        np.asarray(values["pred_range_cell"], dtype=np.float64)
        - np.asarray(values["truth_range_cell"], dtype=np.float64),
        wrap_frequency(
            np.asarray(values["pred_doppler_hz"], dtype=np.float64)
            - np.asarray(values["truth_doppler_hz"], dtype=np.float64),
            prf_hz,
        ),
    )


def _hierarchical_joint_comparisons(
    predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    seed_ids: list[str],
    strategy_names: list[str],
    snr_order: list[str],
    *,
    prf_hz: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    strategy_keys = {
        strategy: {seed_id: f"{seed_id}/{strategy}" for seed_id in seed_ids}
        for strategy in strategy_names
    }
    per_seed_j: dict[str, dict[str, float]] = {strategy: {} for strategy in strategy_names}
    for strategy, seed_map in strategy_keys.items():
        for seed_id, method_key in seed_map.items():
            samples = {
                snr: _student_error_samples(predictions, method_key, snr, prf_hz=prf_hz)
                for snr in snr_order
            }
            per_seed_j[strategy][seed_id] = _joint_endpoint(samples, snr_order)

    rng = np.random.default_rng(seed)
    draws = {strategy: np.empty(replicates, dtype=np.float64) for strategy in strategy_names if strategy != "A_truth_only"}
    a_key_by_seed = strategy_keys["A_truth_only"]
    for replicate in range(replicates):
        chosen_seeds = rng.integers(0, len(seed_ids), size=len(seed_ids))
        delta_sum = {strategy: 0.0 for strategy in draws}
        for seed_index in chosen_seeds:
            seed_id = seed_ids[int(seed_index)]
            per_seed_endpoints = {strategy: 0.0 for strategy in strategy_names}
            for snr in snr_order:
                a_ranges, a_dopplers = _student_error_samples(
                    predictions, a_key_by_seed[seed_id], snr, prf_hz=prf_hz
                )
                sample_indices = rng.integers(0, len(a_ranges), size=len(a_ranges))
                a_rmse_range = float(np.sqrt(np.mean(a_ranges[sample_indices] ** 2)))
                a_rmse_doppler = float(np.sqrt(np.mean(a_dopplers[sample_indices] ** 2)))
                per_seed_endpoints["A_truth_only"] += 0.5 * (
                    a_rmse_range / 24.0 + a_rmse_doppler / 1300.0
                ) / len(snr_order)
                for strategy in draws:
                    method_key = strategy_keys[strategy][seed_id]
                    m_ranges, m_dopplers = _student_error_samples(
                        predictions, method_key, snr, prf_hz=prf_hz
                    )
                    per_seed_endpoints[strategy] += 0.5 * (
                        np.sqrt(np.mean(m_ranges[sample_indices] ** 2)) / 24.0
                        + np.sqrt(np.mean(m_dopplers[sample_indices] ** 2)) / 1300.0
                    ) / len(snr_order)
            for strategy in draws:
                delta_sum[strategy] += per_seed_endpoints[strategy] - per_seed_endpoints["A_truth_only"]
        for strategy in draws:
            draws[strategy][replicate] = delta_sum[strategy] / len(seed_ids)

    result = {}
    for strategy in draws:
        distribution = draws[strategy]
        lower, upper = np.quantile(distribution, [0.025, 0.975])
        seed_deltas = {
            seed_id: per_seed_j[strategy][seed_id] - per_seed_j["A_truth_only"][seed_id]
            for seed_id in seed_ids
        }
        result[strategy] = {
            "mean_delta_joint_endpoint": float(
                np.mean([per_seed_j[strategy][seed_id] for seed_id in seed_ids])
                - np.mean([per_seed_j["A_truth_only"][seed_id] for seed_id in seed_ids])
            ),
            "paired_hierarchical_bootstrap_95_ci": [float(lower), float(upper)],
            "per_seed_delta": seed_deltas,
            "seeds_improved": int(sum(value < 0.0 for value in seed_deltas.values())),
            "per_seed_joint_endpoint": {
                strategy: per_seed_j[strategy],
                "A_truth_only": per_seed_j["A_truth_only"],
            },
        }
    return result


def _range_doppler_metrics_for_arrays(
    truth_range: np.ndarray,
    pred_range: np.ndarray,
    truth_doppler: np.ndarray,
    pred_doppler: np.ndarray,
    prf_hz: float,
) -> dict[str, float]:
    result = parameter_error_metrics(
        truth_range,
        pred_range,
        truth_doppler,
        pred_doppler,
        prf_hz=prf_hz,
    )
    return {
        key: float(value)
        for key, value in result.items()
        if key in {"range_rmse_cells", "doppler_rmse_hz", "range_mae_cells", "doppler_mae_hz"}
    }


def _first_update_at_loss(history_path: Path, target_loss: float) -> int | None:
    import csv

    updates: list[int] = []
    with history_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                loss = float(row["validation_task_loss"])
                update = int(row["update"])
            except (TypeError, ValueError, KeyError):
                continue
            if math.isfinite(loss) and loss <= target_loss:
                updates.append(update)
    return min(updates) if updates else None


def _compute_stage_gates(
    *,
    metrics: dict[str, Any],
    student_predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    classical_predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    formal_metrics: dict[str, Any],
    config: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    detection = metrics["detection"]
    estimation = metrics["estimation"]
    seed_ids = [str(item["id"]) for item in config["splits"]["formal"]["training_seeds"]]
    snr_order = [_snr_key(float(value)) for value in config["data"]["evaluation_snr_db"]]
    reps = int(config["evaluation"]["uncertainty"]["bootstrap_replicates"])
    boot_seed = int(config["evaluation"]["uncertainty"]["bootstrap_seed"])
    prf_hz = float(config["radar"]["prf_hz"])

    g1_config = config["gates"]["G1_truth_supervised_task_learning"]
    g1_snr = _snr_key(float(g1_config["snr_db"]))
    g1_by_seed: dict[str, Any] = {}
    for seed_index, seed_id in enumerate(seed_ids):
        method_key = f"{seed_id}/A_truth_only"
        method_estimation = estimation[method_key]
        detection_snr = detection[method_key]["by_snr_db"][g1_snr]
        pfa = detection_snr["thresholds"]["0.01"]
        h1 = student_predictions[method_key][g1_snr]
        range_error = h1["pred_range_cell"] - h1["truth_range_cell"]
        doppler_error = wrap_frequency(
            h1["pred_doppler_hz"] - h1["truth_doppler_hz"], prf_hz
        )
        range_constant_error = 16.0 - h1["truth_range_cell"]
        doppler_constant_error = wrap_frequency(-h1["truth_doppler_hz"], prf_hz)
        range_delta = paired_rmse_delta_interval(
            range_error,
            range_constant_error,
            replicates=reps,
            seed=boot_seed + seed_index,
        )
        doppler_delta = paired_rmse_delta_interval(
            doppler_error,
            doppler_constant_error,
            replicates=reps,
            seed=boot_seed + 100 + seed_index,
        )
        point = method_estimation["by_snr_db"][g1_snr]
        passed = bool(
            detection_snr["auc"] >= float(g1_config["minimum_detection_auc"])
            and pfa["pd"] >= float(g1_config["minimum_pd_at_pfa_1e_2"])
            and point["range_rmse_cells"] <= float(g1_config["maximum_range_rmse_cells_all_h1"])
            and point["doppler_rmse_hz"] <= float(g1_config["maximum_doppler_rmse_hz_all_h1"])
            and range_delta["upper_95"] < 0.0
            and doppler_delta["upper_95"] < 0.0
        )
        g1_by_seed[seed_id] = {
            "auc": detection_snr["auc"],
            "pd_at_pfa_1e_2": pfa["pd"],
            "range_rmse_cells_all_h1": point["range_rmse_cells"],
            "doppler_rmse_hz_all_h1": point["doppler_rmse_hz"],
            "range_rmse_delta_vs_constant_95_ci": range_delta,
            "doppler_rmse_delta_vs_constant_95_ci": doppler_delta,
            "passed": passed,
        }
    g1_passed = len(g1_by_seed) == 3 and all(item["passed"] for item in g1_by_seed.values())

    high_snr = [_snr_key(float(value)) for value in config["evaluation"]["primary_distillation_endpoint"]["snr_db"]]
    strategy_names = list(config["training"]["methods"])
    paired_j = _hierarchical_joint_comparisons(
        student_predictions,
        seed_ids,
        strategy_names,
        high_snr,
        prf_hz=prf_hz,
        replicates=reps,
        seed=boot_seed + 1000,
    )
    g2_result: dict[str, Any] = {}
    for strategy in strategy_names:
        if strategy == "A_truth_only":
            continue
        method_auc_drops = []
        for seed_id in seed_ids:
            a_key = f"{seed_id}/A_truth_only"
            method_key = f"{seed_id}/{strategy}"
            for snr in high_snr:
                method_auc_drops.append(
                    detection[method_key]["by_snr_db"][snr]["auc"]
                    - detection[a_key]["by_snr_db"][snr]["auc"]
                )
        mean_auc_drop = float(np.mean(method_auc_drops))
        j_summary = paired_j[strategy]
        final_metric_passed = bool(
            j_summary["paired_hierarchical_bootstrap_95_ci"][1] < 0.0
            and j_summary["seeds_improved"] >= int(config["gates"]["G2_distillation_value"]["minimum_seeds_with_same_direction"])
            and mean_auc_drop >= -float(
                config["evaluation"]["primary_distillation_endpoint"]["detection_auc_noninferiority_margin"]
            )
        )
        convergence: dict[str, Any] = {}
        for seed_id in seed_ids:
            seed_summary = next(
                item for item in formal_metrics["seed_summaries"] if item["seed_id"] == seed_id
            )
            a_summary = next(
                item for item in seed_summary["strategy_summaries"] if item["method"] == "A_truth_only"
            )
            target_loss = float(a_summary["best_validation"]["task_loss"])
            a_best_update = int(a_summary["best_update"])
            history_path = output / "formal" / seed_id / strategy / "training_history.csv"
            first_update = _first_update_at_loss(history_path, target_loss)
            j_a = j_summary["per_seed_joint_endpoint"]["A_truth_only"][seed_id]
            j_method = j_summary["per_seed_joint_endpoint"][strategy][seed_id]
            acceptable_blind = bool(j_method <= 1.10 * j_a)
            reached_early = bool(first_update is not None and first_update <= 0.8 * a_best_update)
            convergence[seed_id] = {
                "A_best_validation_loss": target_loss,
                "A_best_update": a_best_update,
                "first_update_at_or_below_A_best_loss": first_update,
                "reached_20pct_earlier": reached_early,
                "final_blind_endpoint_within_10pct_of_A": acceptable_blind,
                "qualifies": bool(reached_early and acceptable_blind),
            }
        convergence_passed = sum(item["qualifies"] for item in convergence.values()) >= 2
        g2_result[strategy] = {
            **j_summary,
            "mean_high_snr_auc_drop_vs_A": mean_auc_drop,
            "final_metric_rule_passed": final_metric_passed,
            "convergence_by_seed": convergence,
            "convergence_rule_passed": convergence_passed,
            "passed": bool(final_metric_passed or convergence_passed),
        }

    adjacent_pairs = list(zip(high_snr[:-1], high_snr[1:]))
    g3_result: dict[str, Any] = {"strategies": {}}
    for strategy in strategy_names:
        strategy_snr_results: dict[str, Any] = {}
        for snr in high_snr:
            student_keys = [f"{seed_id}/{strategy}" for seed_id in seed_ids]
            student_pd = float(np.mean([
                detection[key]["by_snr_db"][snr]["thresholds"]["0.01"]["pd"]
                for key in student_keys
            ]))
            student_range_rmse = []
            student_doppler_rmse = []
            for method_key in student_keys:
                values = student_predictions[method_key][snr]
                n = int(config["splits"]["profile_ml_subset"]["h1_examples_per_snr"])
                sample_metrics = _range_doppler_metrics_for_arrays(
                    values["truth_range_cell"][:n],
                    values["pred_range_cell"][:n],
                    values["truth_doppler_hz"][:n],
                    values["pred_doppler_hz"][:n],
                    prf_hz,
                )
                student_range_rmse.append(sample_metrics["range_rmse_cells"])
                student_doppler_rmse.append(sample_metrics["doppler_rmse_hz"])
            teacher = classical_predictions["APX15_teacher"][snr]
            profile = classical_predictions["profile_ml"][snr]
            n = int(config["splits"]["profile_ml_subset"]["h1_examples_per_snr"])
            teacher_metrics = _range_doppler_metrics_for_arrays(
                teacher["truth_range_cell"][:n],
                teacher["pred_range_cell"][:n],
                teacher["truth_doppler_hz"][:n],
                teacher["pred_doppler_hz"][:n],
                prf_hz,
            )
            profile_metrics = _range_doppler_metrics_for_arrays(
                profile["truth_range_cell"],
                profile["pred_range_cell"],
                profile["truth_doppler_hz"],
                profile["pred_doppler_hz"],
                prf_hz,
            )
            teacher_pd = detection["APX15_teacher"]["by_snr_db"][snr]["thresholds"]["0.01"]["pd"]
            student_rmse_range = float(np.mean(student_range_rmse))
            student_rmse_doppler = float(np.mean(student_doppler_rmse))
            pd_gap = abs(student_pd - teacher_pd)
            range_limit = 2.0 * min(teacher_metrics["range_rmse_cells"], profile_metrics["range_rmse_cells"])
            doppler_limit = 2.0 * min(teacher_metrics["doppler_rmse_hz"], profile_metrics["doppler_rmse_hz"])
            strategy_snr_results[snr] = {
                "student_mean_pd_at_pfa_1e_2": student_pd,
                "teacher_pd_at_pfa_1e_2": teacher_pd,
                "absolute_pd_gap": pd_gap,
                "student_range_rmse_on_profile_subset": student_rmse_range,
                "teacher_range_rmse_on_profile_subset": teacher_metrics["range_rmse_cells"],
                "profile_ml_range_rmse": profile_metrics["range_rmse_cells"],
                "range_rmse_limit": range_limit,
                "student_doppler_rmse_on_profile_subset": student_rmse_doppler,
                "teacher_doppler_rmse_on_profile_subset": teacher_metrics["doppler_rmse_hz"],
                "profile_ml_doppler_rmse": profile_metrics["doppler_rmse_hz"],
                "doppler_rmse_limit": doppler_limit,
                "passes_at_snr": bool(
                    pd_gap <= 0.05
                    and student_rmse_range <= range_limit
                    and student_rmse_doppler <= doppler_limit
                ),
            }
        passing_pairs = [
            [left, right]
            for left, right in adjacent_pairs
            if strategy_snr_results[left]["passes_at_snr"]
            and strategy_snr_results[right]["passes_at_snr"]
        ]
        g3_result["strategies"][strategy] = {
            "by_snr_db": strategy_snr_results,
            "passing_adjacent_snr_pairs": passing_pairs,
            "passed": bool(passing_pairs),
        }
    any_g3 = any(item["passed"] for item in g3_result["strategies"].values())
    g3_result["passed"] = any_g3
    if not g1_passed:
        decision = "STOP"
    elif any(item["passed"] for item in g2_result.values()) or any_g3:
        decision = "GO"
    else:
        decision = "CONDITIONAL GO"
    return {
        "G1_truth_supervised_task_learning": {"by_seed": g1_by_seed, "passed": g1_passed},
        "G2_distillation_value": {"strategies": g2_result, "passed": any(item["passed"] for item in g2_result.values())},
        "G3_traditional_competitiveness": g3_result,
        "stage_decision": decision,
    }


def _write_result_csv(path: Path, metrics: dict[str, Any]) -> None:
    import csv

    fields = (
        "method",
        "snr_db",
        "n_h1",
        "range_mae_cells",
        "range_rmse_cells",
        "doppler_mae_hz",
        "doppler_rmse_hz",
        "doppler_bias_hz",
        "doppler_outlier_rate",
        "auc",
        "pd_pfa_1e_2",
        "pfa_1e_2",
        "tp_pfa_1e_2",
        "fp_pfa_1e_2",
        "tn_pfa_1e_2",
        "fn_pfa_1e_2",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, summary in metrics["estimation"].items():
            for snr, estimate in summary["by_snr_db"].items():
                detection = metrics["detection"].get(method, {}).get("by_snr_db", {}).get(snr)
                threshold = detection["thresholds"].get("0.01") if detection else None
                writer.writerow(
                    {
                        "method": method,
                        "snr_db": snr,
                        "n_h1": estimate["n"],
                        "range_mae_cells": estimate["range_mae_cells"],
                        "range_rmse_cells": estimate["range_rmse_cells"],
                        "doppler_mae_hz": estimate["doppler_mae_hz"],
                        "doppler_rmse_hz": estimate["doppler_rmse_hz"],
                        "doppler_bias_hz": estimate["doppler_bias_hz"],
                        "doppler_outlier_rate": estimate["doppler_outlier_rate"],
                        "auc": detection["auc"] if detection else "",
                        "pd_pfa_1e_2": threshold["pd"] if threshold else "",
                        "pfa_1e_2": threshold["pfa"] if threshold else "",
                        "tp_pfa_1e_2": threshold["tp"] if threshold else "",
                        "fp_pfa_1e_2": threshold["fp"] if threshold else "",
                        "tn_pfa_1e_2": threshold["tn"] if threshold else "",
                        "fn_pfa_1e_2": threshold["fn"] if threshold else "",
                    }
                )


def _plot_blind_results(
    output: Path,
    metrics: dict[str, Any],
    student_predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    classical_predictions: dict[str, dict[str, dict[str, np.ndarray]]],
    config: dict[str, Any],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr_values = [float(value) for value in config["data"]["evaluation_snr_db"]]
    snr_keys = [_snr_key(value) for value in snr_values]
    colors = {
        "A_truth_only": "#1f77b4",
        "B_teacher_pretrain_then_truth": "#ff7f0e",
        "C_joint_truth_and_teacher": "#2ca02c",
        "D_progressive_distillation": "#d62728",
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    strategy_names = list(config["training"]["methods"])
    for strategy in strategy_names:
        for metric_key, axis in (("range_rmse_cells", axes[0]), ("doppler_rmse_hz", axes[1])):
            per_seed = [
                [
                    metrics["estimation"][f"{seed_id}/{strategy}"]["by_snr_db"][snr][metric_key]
                    for snr in snr_keys
                ]
                for seed_id in (str(item["id"]) for item in config["splits"]["formal"]["training_seeds"])
            ]
            values = np.asarray(per_seed, dtype=np.float64)
            axis.errorbar(
                snr_values,
                np.mean(values, axis=0),
                yerr=np.std(values, axis=0, ddof=1),
                marker="o",
                capsize=2,
                color=colors[strategy],
                label=strategy.replace("_", " "),
            )
    baselines = [
        "APX15_teacher",
        "native_fft_peak",
        "native_fft_log_parabolic",
        "fourfold_fft_parabolic",
        "phase_difference",
        "profile_ml",
    ]
    baseline_colors = ["#000000", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#17becf"]
    for method, color in zip(baselines, baseline_colors):
        for metric_key, axis in (("range_rmse_cells", axes[0]), ("doppler_rmse_hz", axes[1])):
            values = [metrics["estimation"][method]["by_snr_db"][snr][metric_key] for snr in snr_keys]
            axis.plot(snr_values, values, marker="s", color=color, label=method.replace("_", " "))
    axes[0].set_ylabel("Range RMSE (cells)")
    axes[1].set_ylabel("Doppler RMSE (Hz)")
    for axis in axes:
        axis.set_xlabel("SNR (dB)")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    figure.suptitle("D1-E001 final blind estimation performance")
    figure.tight_layout()
    figure.savefig(output / "rmse_snr.png", dpi=160)
    plt.close(figure)

    roc_figure, roc_axis = plt.subplots(figsize=(8, 7))
    for method, summary in metrics["detection"].items():
        if method == "APX15_teacher":
            score_arrays = [
                classical_predictions[method][snr]["h0_score"]
                for snr in snr_keys
            ] + [
                classical_predictions[method][snr]["h1_score"]
                for snr in snr_keys
            ]
        else:
            by_snr = student_predictions[method]
            score_arrays = [by_snr[snr]["h0_score"] for snr in snr_keys] + [
                by_snr[snr]["h1_score"] for snr in snr_keys
            ]
        scores = np.concatenate(score_arrays)
        labels = np.concatenate(
            [np.zeros(sum(len(x) for x in score_arrays[: len(snr_keys)]), dtype=np.bool_),
             np.ones(sum(len(x) for x in score_arrays[len(snr_keys) :]), dtype=np.bool_)]
        )
        fpr, tpr = roc_curve_points(scores, labels)
        auc_value = summary["overall"]["auc"]
        roc_axis.plot(fpr, tpr, lw=1.0, label=f"{method} (AUC={auc_value:.3f})")
    roc_axis.plot([0, 1], [0, 1], "k--", lw=0.8)
    roc_axis.set_xlabel("False positive rate")
    roc_axis.set_ylabel("Probability of detection")
    roc_axis.set_title("D1-E001 final blind ROC")
    roc_axis.grid(True, alpha=0.25)
    roc_axis.legend(fontsize=6, ncol=2)
    roc_figure.tight_layout()
    roc_figure.savefig(output / "detection_roc.png", dpi=160)
    plt.close(roc_figure)


def _analysis_text(metrics: dict[str, Any]) -> str:
    gates = metrics["gates"]
    lines = [
        "# D1-E001 正式盲测分析",
        "",
        "盲测基于三种独立正式训练 seed、冻结的 D0 单目标复 AWGN 信号模型和新生成的独立 H0/H1 流。结果仅适用于该仿真条件；不代表杂波、多目标、通道失配或真实 GMTI 数据上的表现。",
        "",
        f"**阶段决策：{gates['stage_decision']}**。检测工作点为独立 H0 校准得到的 Pfa=1e-2/1e-3；没有声称达到 Pfa=1e-6。",
        "",
        "## 阶段门",
        "",
        f"- G1 真值监督：{'通过' if gates['G1_truth_supervised_task_learning']['passed'] else '未通过'}；要求三 seed 全部满足冻结的 20 dB 检测、全 H1 参数误差和常值预测配对比较。",
        f"- G2 蒸馏：{'通过' if gates['G2_distillation_value']['passed'] else '未通过'}；分别按配对层级 bootstrap 与收敛规则评估 B/C/D。",
        f"- G3 传统竞争触发：{'通过' if gates['G3_traditional_competitiveness']['passed'] else '未通过'}；学生在预设相邻高 SNR 上与 APX15 教师和 profile-ML 比较。",
        "",
        "## 方法解释",
        "",
        "A/B/C/D 使用同一三通道 TCN 推理结构、同一盲测 IQ 和冻结训练预算。教师只在 B/C/D 训练中提供完整软频谱响应；物理真值作为独立任务监督，教师硬距离/Doppler 输出不作为学生标签。FFT、聚合相位差、APX15 和 profile-ML 均单独列为传统/教师基线，学生结果不被解释为纯传统前端性能。",
        "",
        "逐方法、逐 SNR 指标、95% 区间、分组结果和门槛数值见 `metrics.json` 与 `result.csv`；RMSE-SNR 和 ROC 图分别见 `rmse_snr.png`、`detection_roc.png`。",
        "",
    ]
    replay = metrics.get("analysis_replay")
    if replay:
        lines[4:4] = [
            "",
            f"**复算记录：** 盲测预测沿用 `{replay['prediction_code_commit']}` 生成的已保存数组；分析代码为 `{replay['analysis_code_commit']}`。未重新生成盲测样本或调整阈值。",
            "",
        ]
    return "\n".join(lines)


def run_d1_blind_evaluation(
    config: dict[str, Any],
    *,
    root: str | Path,
    output: str | Path,
    device: str = "cuda:0",
    batch_size: int = 128,
    replay_existing_predictions: bool = False,
) -> dict[str, Any]:
    """Run blind evaluation, or recompute metrics from completed saved predictions."""
    import torch

    from ..training.d1_runner import _load_d0_config, require_cuda

    generate_d0_batch, teacher_type, _, _ = _load_evaluation_dependencies()
    if batch_size <= 0:
        raise ValueError("evaluation batch_size must be positive")
    root_path = Path(root).resolve()
    output_path = Path(output).resolve()
    _assert_frozen_config(config, root_path, output_path)
    blind_dir = output_path / "blind"
    metrics_path = blind_dir / "metrics.json"
    partial_path = blind_dir / "metrics_partial.json"
    prediction_dir = blind_dir / "predictions"
    if metrics_path.exists():
        raise FileExistsError(f"blind results already exist at {metrics_path}")
    calibration_path = output_path / "calibration" / "metrics.json"
    if not calibration_path.is_file():
        raise FileNotFoundError("D1 blind evaluation requires completed independent calibration")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("status") != "COMPLETED":
        raise RuntimeError("D1 calibration is incomplete; blind evaluation is not started")
    manifest = calibration.get("freeze_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("D1 calibration lacks the frozen code/config/checkpoint manifest")
    calibration_blob = _require_clean_tracked_paths(root_path, [calibration_path])
    replay_info = None
    existing_partial = None
    if replay_existing_predictions:
        if not partial_path.is_file():
            raise FileNotFoundError("analysis replay requires saved blind partial metadata")
        existing_partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if existing_partial.get("freeze_manifest") != manifest:
            raise RuntimeError("saved blind predictions do not match the calibration freeze")
        if existing_partial.get("calibration_blob_oids") != calibration_blob:
            raise RuntimeError("saved blind predictions do not match the committed calibration")
        expected_methods = _require_completed_blind_predictions(
            existing_partial, config, prediction_dir
        )
        current_manifest = _freeze_manifest(config, root_path, output_path)
        changed_paths = _analysis_replay_changed_paths(manifest, current_manifest)
        replay_info = {
            "predictions_reused": True,
            "blind_samples_regenerated": False,
            "student_methods_reused": expected_methods,
            "prediction_code_commit": manifest["code_commit"],
            "analysis_code_commit": current_manifest["code_commit"],
            "changed_frozen_paths": changed_paths,
            "reason": (
                "Corrected the G2 AUC non-inferiority margin lookup to read the "
                "pre-registered evaluation.primary_distillation_endpoint value. "
                "Models, predictions, thresholds, and metric definitions were reused."
            ),
        }
    else:
        _verify_freeze_manifest(config, root_path, output_path, manifest)
        if partial_path.exists():
            existing_partial = json.loads(partial_path.read_text(encoding="utf-8"))
            if existing_partial.get("freeze_manifest") != manifest:
                raise RuntimeError("existing blind partial artifacts use a different freeze manifest")

    cuda_device = torch.device("cpu") if replay_existing_predictions else require_cuda(device)
    started_at = time.perf_counter()
    if not replay_existing_predictions:
        torch.cuda.reset_peak_memory_stats(cuda_device)
    d0_config = _load_d0_config(root_path)
    snr_order = [_snr_key(float(value)) for value in config["data"]["evaluation_snr_db"]]

    if existing_partial is not None:
        partial = existing_partial
    else:
        if blind_dir.exists() and any(blind_dir.iterdir()):
            raise FileExistsError(
                f"blind directory contains unregistered artifacts; preserving {blind_dir}"
            )
        blind_dir.mkdir(parents=True, exist_ok=True)
        partial = {
            "experiment_id": "D1-E001",
            "phase": "final_blind",
            "status": "RUNNING",
            "freeze_manifest": manifest,
            "calibration_blob_oids": calibration_blob,
            "completed_student_methods": [],
            "classical_baselines_completed": False,
        }
        _write_json(partial_path, partial)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    completed = set(partial.get("completed_student_methods", []))
    for checkpoint in _model_checkpoint_paths(config, output_path):
        relative_method = checkpoint.relative_to(output_path / "formal").as_posix()
        method_key = relative_method.rsplit("/", 1)[0]
        file_key = method_key.replace("/", "__")
        prediction_path = prediction_dir / f"{file_key}.npz"
        if method_key in completed:
            if not prediction_path.is_file():
                raise FileNotFoundError(f"completed blind student predictions missing: {prediction_path}")
            continue
        if prediction_path.exists():
            # The same deterministic freeze may have been interrupted just before its completion marker.
            _load_prediction_arrays(prediction_path)
            completed.add(method_key)
            partial["completed_student_methods"] = sorted(completed)
            _write_json(partial_path, partial)
            continue
        model = _load_student_checkpoint(checkpoint, config, cuda_device)
        try:
            prediction = _collect_student_blind_predictions(
                model,
                config=config,
                d0_config=d0_config,
                generate_d0_batch=generate_d0_batch,
                batch_size=batch_size,
            )
            _save_prediction_arrays(prediction_path, prediction)
        finally:
            del model
            torch.cuda.empty_cache()
        completed.add(method_key)
        partial["completed_student_methods"] = sorted(completed)
        _write_json(partial_path, partial)

    classical_path = prediction_dir / "classical_baselines.npz"
    if partial.get("classical_baselines_completed"):
        if not classical_path.is_file():
            raise FileNotFoundError(f"classical blind predictions missing: {classical_path}")
    elif classical_path.exists():
        _load_prediction_arrays(classical_path)
        partial["classical_baselines_completed"] = True
        _write_json(partial_path, partial)
    else:
        teacher = teacher_type(d0_config)
        try:
            classical_prediction = _collect_classical_blind_predictions(
                config=config,
                d0_config=d0_config,
                generate_d0_batch=generate_d0_batch,
                teacher=teacher,
                batch_size=batch_size,
            )
            _save_prediction_arrays(classical_path, classical_prediction)
        finally:
            teacher.close()
        partial["classical_baselines_completed"] = True
        _write_json(partial_path, partial)

    student_predictions = {
        method_key: _joined_student_predictions(
            prediction_dir / f"{method_key.replace('/', '__')}.npz", snr_order
        )
        for method_key in sorted(completed)
    }
    expected_method_count = sum(
        len(config["training"]["methods"])
        for _ in config["splits"]["formal"]["training_seeds"]
    )
    if len(student_predictions) != expected_method_count:
        raise RuntimeError("not all frozen student checkpoints have blind predictions")
    classical_predictions = _joined_classical_predictions(classical_path, snr_order)
    metrics = _build_blind_metrics(
        student_predictions=student_predictions,
        classical_predictions=classical_predictions,
        calibration=calibration,
        config=config,
    )
    formal_metrics = json.loads(
        (output_path / "formal" / "metrics.json").read_text(encoding="utf-8")
    )
    metrics.update(
        {
            "experiment_id": "D1-E001",
            "phase": "final_blind",
            "status": "COMPLETED",
            "freeze_manifest": manifest,
            "calibration_thresholds": calibration["methods"],
            "sample_counts": {
                "calibration_h0_per_method": int(calibration["calibration_samples_per_method"]),
                "blind_h0_total": int(config["splits"]["final_blind"]["h0_examples_per_snr"])
                * len(snr_order),
                "blind_h1_total": int(config["splits"]["final_blind"]["h1_examples_per_snr"])
                * len(snr_order),
                "profile_ml_h1_per_snr": int(
                    config["splits"]["profile_ml_subset"]["h1_examples_per_snr"]
                ),
            },
            "training_resources_by_seed": {
                seed["seed_id"]: seed["training_environment"]
                for seed in formal_metrics["seed_summaries"]
            },
            "evaluation_environment": _evaluation_environment(
                cuda_device, batch_size, time.perf_counter() - started_at
            )
            if not replay_existing_predictions
            else {
                "student_prediction_device": device,
                "classical_prediction_device": "cpu",
                "analysis_device": "cpu",
                "analysis_replay": True,
                "elapsed_seconds": time.perf_counter() - started_at,
            },
        }
    )
    if replay_info is not None:
        metrics["analysis_replay"] = replay_info
    metrics["gates"] = _compute_stage_gates(
        metrics=metrics,
        student_predictions=student_predictions,
        classical_predictions=classical_predictions,
        formal_metrics=formal_metrics,
        config=config,
        output=output_path,
    )
    metrics["analysis_summary"] = _analysis_text(metrics)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(metrics_path, metrics)
    _write_json(output_path / "metrics.json", metrics)
    _write_result_csv(output_path / "result.csv", metrics)
    (output_path / "analysis.md").write_text(metrics["analysis_summary"], encoding="utf-8")
    _plot_blind_results(
        output_path,
        metrics,
        student_predictions,
        classical_predictions,
        config,
    )
    partial["status"] = "COMPLETED"
    partial["completed_student_methods"] = sorted(completed)
    if replay_info is not None:
        partial["analysis_replay"] = replay_info
    _write_json(partial_path, partial)
    return metrics
