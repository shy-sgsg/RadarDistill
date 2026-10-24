#!/usr/bin/env python3
"""Run development-only D1-R1 diagnostics on cached training/validation splits."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.distillation.task_losses import task_loss  # noqa: E402
from radardistill.evaluation.d1_evaluation import (  # noqa: E402
    _load_student_checkpoint,
    binary_detection_metrics,
)
from radardistill.evaluation.d1_r1_diagnostics import (  # noqa: E402
    circular_doppler_error_hz,
    diagnostic_truth_range_bins,
    decode_doppler_modes,
    fit_circular_ridge_probe,
    validate_diagnostic_cache_path,
)


_CACHE_FIELDS = (
    "iq",
    "truth_presence",
    "truth_range_cell",
    "truth_doppler_hz",
    "truth_initial_phase_rad",
    "truth_target_amplitude",
    "snr_db",
    "teacher_response",
    "teacher_detection_score",
    "teacher_doppler_hz",
)
_SNR_EDGES = np.linspace(-20.0, 20.0, 9)
_FIXED_SNR = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 20.0)


def _read_cache(root: Path, tag: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    cache_dir = validate_diagnostic_cache_path(
        root / "experiments" / "D1-E001" / "data" / "cache" / tag
    )
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing D1 cache manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("tag") != tag:
        raise ValueError(f"D1 cache tag mismatch at {cache_dir}: {manifest.get('tag')!r}")
    arrays = {}
    for name in _CACHE_FIELDS:
        path = cache_dir / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing D1 cache array: {path}")
        arrays[name] = np.load(path, mmap_mode="r")
        if len(arrays[name]) != int(manifest["total_examples"]):
            raise ValueError(f"D1 cache length disagrees with manifest for {path}")
    return manifest, arrays


def _iq_to_tensor(iq: np.ndarray, device: torch.device) -> torch.Tensor:
    values = np.asarray(iq, dtype=np.complex64)
    planes = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(planes)).to(device=device)


def _evaluate_model(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    *,
    prf_hz: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    result: dict[str, list[np.ndarray]] = {
        "score_logit": [],
        "predicted_range_bin": [],
        "predicted_range_cell": [],
        "predicted_offset": [],
        "oracle_offset": [],
        "predicted_doppler_hz": [],
        "oracle_doppler_hz": [],
    }
    count = len(arrays["truth_presence"])
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            indices = np.arange(start, min(start + batch_size, count), dtype=np.int64)
            iq = _iq_to_tensor(arrays["iq"][indices], device)
            outputs = model(iq, compute_spectrum=False)
            truth_bin = torch.as_tensor(
                diagnostic_truth_range_bins(
                    np.asarray(arrays["truth_range_cell"][indices]).copy(),
                    np.asarray(arrays["truth_presence"][indices]).copy(),
                    range_bin_count=outputs["range_logits"].shape[1],
                ),
                device=device,
                dtype=torch.long,
            )
            decoded = decode_doppler_modes(outputs, truth_bin, prf_hz=prf_hz)
            row = torch.arange(len(indices), device=device)
            predicted_bin = decoded["predicted_range_bin"]
            result["score_logit"].append(outputs["presence_logit"].cpu().numpy())
            result["predicted_range_bin"].append(predicted_bin.cpu().numpy())
            result["predicted_range_cell"].append(decoded["predicted_range_cell"].cpu().numpy())
            result["predicted_offset"].append(
                outputs["range_offset"][row, predicted_bin].cpu().numpy()
            )
            result["oracle_offset"].append(
                outputs["range_offset"][row, truth_bin].cpu().numpy()
            )
            result["predicted_doppler_hz"].append(
                decoded["predicted_range_doppler_hz"].cpu().numpy()
            )
            result["oracle_doppler_hz"].append(
                decoded["oracle_range_doppler_hz"].cpu().numpy()
            )
    return {name: np.concatenate(chunks).astype(np.float64, copy=False) for name, chunks in result.items()}


def _sample_summary(errors: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(errors, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan"), "bias": float("nan")}
    if np.any(~np.isfinite(values)):
        raise ValueError("diagnostic errors contain non-finite values")
    return {
        "n": int(values.size),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "mae": float(np.mean(np.abs(values))),
        "bias": float(np.mean(values)),
    }


def _group_summaries(
    errors: np.ndarray, labels: np.ndarray, group_names: list[str] | tuple[str, ...]
) -> dict[str, dict[str, float | int]]:
    result = {}
    for name in group_names:
        mask = labels == name
        result[str(name)] = _sample_summary(errors[mask])
    return result


def _diagnostic_strata(
    arrays: dict[str, np.ndarray], amplitude_edges: np.ndarray
) -> dict[str, np.ndarray]:
    h1 = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    frequency = np.asarray(arrays["truth_doppler_hz"], dtype=np.float64)
    phase = np.asarray(arrays["truth_initial_phase_rad"], dtype=np.float64)
    amplitude = np.asarray(arrays["truth_target_amplitude"], dtype=np.float64)
    range_cell = np.asarray(arrays["truth_range_cell"], dtype=np.float64)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    freq_edges = np.asarray([-650.0, -325.0, 0.0, 325.0, 650.0])
    if np.any(~np.isfinite(frequency[h1])) or np.any(~np.isfinite(phase[h1])):
        raise ValueError("H1 frequency and initial-phase truths must be finite")
    if np.any(~np.isfinite(amplitude[h1])) or np.any(~np.isfinite(range_cell[h1])):
        raise ValueError("H1 amplitude and range truths must be finite")
    frequency_band = np.full(len(h1), "masked_h0", dtype=object)
    frequency_band_names = np.asarray(["[-650,-325)", "[-325,0)", "[0,325)", "[325,650)"], dtype=object)
    freq_index = np.digitize(frequency[h1], freq_edges[1:-1], right=False)
    frequency_band[h1] = frequency_band_names[np.clip(freq_index, 0, 3)]
    nyquist_region = np.full(len(h1), "masked_h0", dtype=object)
    nyquist_region[h1] = np.where(np.abs(frequency[h1]) >= 585.0, "within_65_hz", "interior")
    phase_group = np.full(len(h1), "masked_h0", dtype=object)
    phase_index = np.floor(
        np.remainder(phase[h1] + math.pi, 2.0 * math.pi) / (math.pi / 4.0)
    ).astype(int)
    phase_group[h1] = np.asarray([f"phase_{i}" for i in range(8)], dtype=object)[
        np.clip(phase_index, 0, 7)
    ]
    range_group = np.full(len(h1), "masked_h0", dtype=object)
    range_index = np.floor(range_cell[h1]).astype(int)
    range_group[h1] = np.asarray(
        ["range_0_7", "range_8_15", "range_16_23", "range_24_31"], dtype=object
    )[np.clip(range_index // 8, 0, 3)]
    amplitude_group = np.full(len(h1), "masked_h0", dtype=object)
    amp_index = np.digitize(amplitude[h1], amplitude_edges[1:-1], right=False)
    amplitude_group[h1] = np.asarray([f"amp_q{i + 1}" for i in range(4)], dtype=object)[
        np.clip(amp_index, 0, 3)
    ]
    snr_name = np.asarray([str(int(value)) if float(value).is_integer() else str(value) for value in snr])
    return {
        "frequency_band_hz": frequency_band,
        "nyquist_region": nyquist_region,
        "initial_phase_octant": phase_group,
        "target_amplitude_quartile": amplitude_group,
        "range_block": range_group,
        "snr_db": snr_name,
        "h1": h1,
    }


def _model_metrics(
    arrays: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    *,
    prf_hz: float,
    amplitude_edges: np.ndarray,
) -> dict[str, Any]:
    presence = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    detection = binary_detection_metrics(
        predictions["score_logit"],
        presence,
        snr,
        thresholds={"uncalibrated_logit_zero_descriptive": 0.0},
    )
    h1 = presence
    truth_range = np.asarray(arrays["truth_range_cell"], dtype=np.float64)[h1]
    truth_bin = np.floor(truth_range).astype(np.int64)
    truth_offset = truth_range - truth_bin
    predicted_bin = predictions["predicted_range_bin"][h1].astype(np.int64)
    predicted_cell = predictions["predicted_range_cell"][h1]
    predicted_offset = predictions["predicted_offset"][h1]
    oracle_offset = predictions["oracle_offset"][h1]
    truth_doppler = np.asarray(arrays["truth_doppler_hz"], dtype=np.float64)[h1]
    normal_error = circular_doppler_error_hz(
        predictions["predicted_doppler_hz"][h1], truth_doppler, prf_hz=prf_hz
    )
    oracle_error = circular_doppler_error_hz(
        predictions["oracle_doppler_hz"][h1], truth_doppler, prf_hz=prf_hz
    )
    range_error = predicted_cell - truth_range
    oracle_offset_error = oracle_offset - truth_offset
    predicted_offset_error = predicted_offset - truth_offset
    class_correct = predicted_bin == truth_bin
    detected = predictions["score_logit"][h1] >= 0.0

    result: dict[str, Any] = {
        "detection": detection,
        "range": {
            "n_h1": int(np.count_nonzero(h1)),
            "range_bin_accuracy": float(np.mean(class_correct)),
            "range_bin_error_rate": float(np.mean(~class_correct)),
            "predicted_bin_continuous_error_cells": _sample_summary(range_error),
            "oracle_bin_offset_error_cells": _sample_summary(oracle_offset_error),
            "predicted_bin_offset_error_given_correct_class": _sample_summary(
                predicted_offset_error[class_correct]
            ),
        },
        "doppler": {
            "predicted_range_bin": _sample_summary(normal_error),
            "oracle_true_range_bin": _sample_summary(oracle_error),
            "predicted_range_bin_detected_logit_ge_0": _sample_summary(normal_error[detected]),
            "oracle_true_range_bin_detected_logit_ge_0": _sample_summary(oracle_error[detected]),
        },
    }
    strata = _diagnostic_strata(arrays, amplitude_edges)
    result["stratified"] = {}
    for name, labels in strata.items():
        if name == "h1":
            continue
        selected_labels = labels[h1]
        result["stratified"][name] = {
            "predicted_range_bin_doppler": _group_summaries(
                normal_error, selected_labels, sorted(set(map(str, selected_labels)))
            ),
            "oracle_true_range_bin_doppler": _group_summaries(
                oracle_error, selected_labels, sorted(set(map(str, selected_labels)))
            ),
        }
    result["detection_threshold_note"] = (
        "logit >= 0 is an uncalibrated descriptive subset only; it is not a Pd/Pfa claim"
    )
    return result


def _extract_features(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pre_chunks = []
    post_chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            input_tensor = _iq_to_tensor(arrays["iq"][batch_indices], device)
            truth_range = np.asarray(arrays["truth_range_cell"][batch_indices], dtype=np.float64)
            truth_bin = torch.as_tensor(
                np.floor(truth_range).astype(np.int64), device=device, dtype=torch.long
            )
            pre_pool = model.temporal_blocks(model.stem(input_tensor))
            post_pool = pre_pool.mean(dim=2)
            row = torch.arange(len(batch_indices), device=device)
            pre_by_range = pre_pool.permute(0, 3, 1, 2)[row, truth_bin]
            post_by_range = post_pool.permute(0, 2, 1)[row, truth_bin]
            pre_chunks.append(pre_by_range.flatten(start_dim=1).cpu().numpy())
            post_chunks.append(post_by_range.cpu().numpy())
    return (
        np.concatenate(pre_chunks).astype(np.float64, copy=False),
        np.concatenate(post_chunks).astype(np.float64, copy=False),
    )


def _stratified_h1_indices(arrays: dict[str, np.ndarray], *, seed: int) -> np.ndarray:
    presence = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    selected = []
    for index in range(len(_SNR_EDGES) - 1):
        lower, upper = _SNR_EDGES[index : index + 2]
        in_bin = (snr >= lower) & (snr < upper if index < len(_SNR_EDGES) - 2 else snr <= upper)
        candidates = np.flatnonzero(presence & in_bin)
        if candidates.size < 128:
            raise ValueError(
                f"training cache has only {candidates.size} H1 samples in SNR bin [{lower},{upper}]"
            )
        selected.append(rng.choice(candidates, size=128, replace=False))
    return np.concatenate(selected)


def _probe_summary(
    model: torch.nn.Module,
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    *,
    seed: int,
    prf_hz: float,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    train_indices = _stratified_h1_indices(train_arrays, seed=9010 + seed)
    val_indices = np.flatnonzero(np.asarray(validation_arrays["truth_presence"], dtype=np.bool_))
    train_pre, train_post = _extract_features(
        model, train_arrays, train_indices, device=device, batch_size=batch_size
    )
    val_pre, val_post = _extract_features(
        model, validation_arrays, val_indices, device=device, batch_size=batch_size
    )
    train_frequency = np.asarray(train_arrays["truth_doppler_hz"][train_indices], dtype=np.float64)
    val_frequency = np.asarray(validation_arrays["truth_doppler_hz"][val_indices], dtype=np.float64)
    val_snr = np.asarray(validation_arrays["snr_db"][val_indices], dtype=np.float64)
    output = {
        "train_samples": int(len(train_indices)),
        "validation_samples": int(len(val_indices)),
        "train_snr_bins": 8,
        "ridge_alpha": 1.0,
        "readout_range": "floor(truth_range_cell), diagnostic only",
        "pre_pool_feature_dim": int(train_pre.shape[1]),
        "post_pool_feature_dim": int(train_post.shape[1]),
        "pre_pool": fit_circular_ridge_probe(
            train_pre, train_frequency, val_pre, val_frequency, prf_hz=prf_hz, alpha=1.0
        ),
        "post_pool": fit_circular_ridge_probe(
            train_post, train_frequency, val_post, val_frequency, prf_hz=prf_hz, alpha=1.0
        ),
    }
    for key in ("pre_pool", "post_pool"):
        probe = output[key]
        errors = np.asarray(probe.pop("errors_hz"), dtype=np.float64)
        probe.pop("predicted_hz")
        probe["by_snr_db"] = {}
        for snr_value in sorted(np.unique(val_snr)):
            mask = val_snr == snr_value
            probe["by_snr_db"][str(int(snr_value))] = _sample_summary(errors[mask])
    return output


def _gradient_summary(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    *,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from radardistill.distillation.task_losses import task_loss

    shared = list(model.stem.parameters()) + list(model.temporal_blocks.parameters())
    result: dict[str, Any] = {}
    presence = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    rng = np.random.default_rng(4400 + seed)
    model.train()
    for index in range(len(_SNR_EDGES) - 1):
        lower, upper = _SNR_EDGES[index : index + 2]
        in_bin = (snr >= lower) & (snr < upper if index < len(_SNR_EDGES) - 2 else snr <= upper)
        group_indices = []
        for target_presence in (False, True):
            candidates = np.flatnonzero(in_bin & (presence == target_presence))
            if candidates.size < 16:
                raise ValueError(
                    f"training cache has only {candidates.size} samples for SNR [{lower},{upper}], "
                    f"H1={target_presence}"
                )
            group_indices.extend(rng.choice(candidates, size=16, replace=False))
        indices = np.asarray(group_indices, dtype=np.int64)
        iq = _iq_to_tensor(arrays["iq"][indices], device)
        truth = {
            "truth_presence": torch.as_tensor(
                np.asarray(arrays["truth_presence"][indices]).copy(), device=device, dtype=torch.float32
            ),
            "truth_range_cell": torch.as_tensor(
                np.asarray(arrays["truth_range_cell"][indices]).copy(), device=device, dtype=torch.float32
            ),
            "truth_doppler_hz": torch.as_tensor(
                np.asarray(arrays["truth_doppler_hz"][indices]).copy(), device=device, dtype=torch.float32
            ),
        }
        outputs = model(iq, compute_spectrum=False)
        losses = task_loss(outputs, truth, config, teacher_weight=0.0)
        vectors = {}
        norms = {}
        for name in ("presence", "range_cell", "range_offset", "doppler"):
            gradients = torch.autograd.grad(
                losses[name], shared, retain_graph=True, allow_unused=True
            )
            pieces = [gradient.reshape(-1) for gradient in gradients if gradient is not None]
            vector = torch.cat(pieces).detach()
            vectors[name] = vector
            norms[name] = float(torch.linalg.vector_norm(vector).cpu())
        cosines = {}
        names = tuple(vectors)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                left_vector = vectors[left]
                right_vector = vectors[right]
                denominator = torch.linalg.vector_norm(left_vector) * torch.linalg.vector_norm(right_vector)
                cosine = torch.dot(left_vector, right_vector) / denominator.clamp_min(1e-12)
                cosines[f"{left}__{right}"] = float(cosine.cpu())
        result[f"[{lower:g},{upper:g}{']' if index == len(_SNR_EDGES) - 2 else ')'}"] = {
            "n_h0": 16,
            "n_h1": 16,
            "gradient_norm": norms,
            "pairwise_cosine": cosines,
        }
        model.zero_grad(set_to_none=True)
        del outputs, losses, vectors
    model.eval()
    return {
        "shared_parameters": "stem + temporal_blocks",
        "samples_per_snr_bin": {"h0": 16, "h1": 16},
        "teacher_kl_gradient": {
            "available": False,
            "reason": "only inference checkpoints were saved; spectrum_head weights are absent",
        },
        "by_snr_bin_db": result,
    }


def _history_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing D1 training history: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    metrics = (
        "train_presence",
        "train_range_cell",
        "train_range_offset",
        "train_doppler",
        "train_teacher_kl",
        "train_total",
        "validation_task_loss",
        "validation_doppler_rmse_hz",
        "validation_range_rmse_cells",
    )
    result: dict[str, Any] = {"path": str(path.relative_to(ROOT)), "rows": len(rows), "metrics": {}}
    for name in metrics:
        values = []
        for row in rows:
            try:
                value = float(row.get(name, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        if values:
            result["metrics"][name] = {
                "first_finite": values[0],
                "last_finite": values[-1],
                "minimum": min(values),
                "maximum": max(values),
            }
    return result


def _teacher_metrics(
    arrays: dict[str, np.ndarray], *, prf_hz: float
) -> dict[str, Any]:
    presence = np.asarray(arrays["truth_presence"], dtype=np.bool_)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    scores = np.asarray(arrays["teacher_detection_score"], dtype=np.float64)
    detection = binary_detection_metrics(
        scores,
        presence,
        snr,
        thresholds={"uncalibrated_score_zero_descriptive": 0.0},
    )
    h1 = presence
    doppler_error = circular_doppler_error_hz(
        np.asarray(arrays["teacher_doppler_hz"], dtype=np.float64)[h1],
        np.asarray(arrays["truth_doppler_hz"], dtype=np.float64)[h1],
        prf_hz=prf_hz,
    )
    return {
        "detection": detection,
        "h1_doppler_by_snr_db": _group_summaries(
            doppler_error,
            np.asarray([str(int(x)) for x in snr[h1]], dtype=object),
            sorted(set(str(int(x)) for x in snr[h1])),
        ),
        "scope": "D1 common development pilot_validation only",
    }


def _git_sha(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    config_path = args.config.resolve()
    output = args.output.resolve()
    if args.device not in {"cuda", "cuda:0"}:
        raise ValueError("D1-R1 model diagnostics require --device cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required for D1-R1 model inference; no CPU fallback is used")
    if output.joinpath("metrics.json").exists() and not args.overwrite:
        raise FileExistsError(f"{output / 'metrics.json'} exists; preserve it or pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    prf_hz = float(config["radar"]["prf_hz"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    common_manifest, common_arrays = _read_cache(root, "pilot_validation")
    amplitude_manifest, amplitude_arrays = _read_cache(root, "seed_1_train")
    h1_amplitude = np.asarray(amplitude_arrays["truth_target_amplitude"])[
        np.asarray(amplitude_arrays["truth_presence"], dtype=np.bool_)
    ]
    amplitude_edges = np.quantile(h1_amplitude, [0.0, 0.25, 0.5, 0.75, 1.0])
    amplitude_edges[0] -= 1e-9
    amplitude_edges[-1] += 1e-9

    metric_doc: dict[str, Any] = {
        "experiment": "D1-R1",
        "protocol": str((output / "protocol.md").relative_to(root)) if output.is_relative_to(root) else str(output / "protocol.md"),
        "git_sha": _git_sha(root),
        "data_boundary": {
            "common_split": "pilot_validation",
            "common_manifest": common_manifest,
            "training_cache_for_amplitude_edges": "seed_1_train",
            "training_cache_manifest": amplitude_manifest,
            "final_blind_accessed": False,
            "thresholds_calibrated": False,
        },
        "config": str(config_path.relative_to(root)) if config_path.is_relative_to(root) else str(config_path),
        "device": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "name": torch.cuda.get_device_name(device),
            "memory_total_bytes": int(torch.cuda.get_device_properties(device).total_memory),
            "memory_allocated_before_bytes": int(torch.cuda.memory_allocated(device)),
        },
        "amplitude_quartile_edges_from_seed_1_train_h1": [float(x) for x in amplitude_edges],
        "models": {},
        "teacher_on_common_validation": _teacher_metrics(common_arrays, prf_hz=prf_hz),
    }

    for seed in (1, 2, 3):
        train_manifest, train_arrays = _read_cache(root, f"seed_{seed}_train")
        val_manifest, val_arrays = _read_cache(root, f"seed_{seed}_validation")
        metric_doc["models"][f"seed_{seed}"] = {
            "training_manifest": train_manifest,
            "validation_manifest": val_manifest,
            "strategies": {},
        }
        for strategy in ("A_truth_only", "D_progressive_distillation"):
            checkpoint = (
                root
                / "experiments"
                / "D1-E001"
                / "formal"
                / f"seed_{seed}"
                / strategy
                / "best_inference.pt"
            )
            model = _load_student_checkpoint(checkpoint, config, device)
            predictions = _evaluate_model(
                model,
                common_arrays,
                prf_hz=prf_hz,
                device=device,
                batch_size=args.batch_size,
            )
            metric_doc["models"][f"seed_{seed}"]["strategies"][strategy] = {
                "checkpoint": str(checkpoint.relative_to(root)),
                "common_pilot_validation": _model_metrics(
                    common_arrays, predictions, prf_hz=prf_hz, amplitude_edges=amplitude_edges
                ),
                "history": _history_summary(
                    root
                    / "experiments"
                    / "D1-E001"
                    / "formal"
                    / f"seed_{seed}"
                    / strategy
                    / "training_history.csv"
                ),
            }
            if strategy == "A_truth_only":
                metric_doc["models"][f"seed_{seed}"]["strategies"][strategy]["linear_probe"] = _probe_summary(
                    model,
                    train_arrays,
                    val_arrays,
                    seed=seed,
                    prf_hz=prf_hz,
                    device=device,
                    batch_size=args.batch_size,
                )
            metric_doc["models"][f"seed_{seed}"]["strategies"][strategy]["task_gradients"] = (
                _gradient_summary(model, train_arrays, seed=seed, config=config, device=device)
            )
            del model, predictions
            torch.cuda.empty_cache()

    metric_doc["device"]["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    metric_doc["runtime"] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "script": str(Path(__file__).relative_to(root)),
        "inference_batch_size": int(args.batch_size),
        "training_performed": False,
    }
    destination = output / "metrics.json"
    destination.write_text(
        json.dumps(metric_doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metric_doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=ROOT / "experiments/D1-E001/config.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/D1-R1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    result = _run(args)
    print(
        json.dumps(
            {
                "experiment": result["experiment"],
                "git_sha": result["git_sha"],
                "device": result["device"]["name"],
                "models": list(result["models"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
