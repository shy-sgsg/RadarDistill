"""Matched deterministic estimators used in D1-E002 validation and evaluation."""

from __future__ import annotations

import math
import csv
import gc
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _as_iq(iq: Any) -> np.ndarray:
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32):
        raise ValueError("iq must have shape [B,3,130,32]")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    return values


def _range_indices(range_bins: Any, count: int) -> np.ndarray:
    values = np.asarray(range_bins, dtype=np.int64).reshape(-1)
    if values.shape != (count,) or np.any((values < 0) | (values >= 32)):
        raise ValueError("range_bins must contain one valid predicted cell per sample")
    return values


def _wrap(frequency: np.ndarray, prf_hz: float) -> np.ndarray:
    if not math.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    return (frequency + prf_hz / 2.0) % prf_hz - prf_hz / 2.0


def energy_detection_score(iq: Any) -> np.ndarray:
    """Common noncoherent detector score for classical frequency estimators."""
    values = _as_iq(iq)
    energy_by_range = np.sum(np.abs(values) ** 2, axis=(1, 2), dtype=np.float64)
    return np.max(energy_by_range, axis=-1)


def evaluation_chunk_seed(
    seed_base: int,
    snr_index: int,
    chunk_index: int,
    *,
    snr_stride: int,
    snr_count: int,
) -> int:
    """Derive the preregistered nonoverlapping stream seed for one eval chunk."""
    if seed_base < 0 or snr_stride <= 0 or snr_count <= 0 or snr_count >= snr_stride:
        raise ValueError("invalid evaluation seed base, stride, or SNR count")
    if not 0 <= snr_index < snr_count or chunk_index < 0:
        raise ValueError("invalid evaluation SNR or chunk index")
    return int(seed_base + snr_index * snr_stride + chunk_index * snr_stride * snr_count)


def estimate_lag_one_at_range(
    iq: Any,
    predicted_range_bins: Any,
    prf_hz: float,
    *,
    epsilon: float = 1.0e-8,
) -> np.ndarray:
    """Equal-channel phase estimate from S2's normalized lag-one statistics."""
    values = _as_iq(iq)
    bins = _range_indices(predicted_range_bins, len(values))
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")
    estimates = np.empty(len(values), dtype=np.float64)
    for sample, range_bin in enumerate(bins):
        selected = values[sample, :, :, range_bin]
        left, right = selected[:, :-1], selected[:, 1:]
        numerator = np.sum(right * np.conj(left), axis=1)
        denominator = np.sqrt(
            np.sum(np.abs(left) ** 2, axis=1) * np.sum(np.abs(right) ** 2, axis=1)
        ) + epsilon
        coherence = numerator / denominator
        estimates[sample] = np.angle(np.sum(coherence)) * prf_hz / (2.0 * np.pi)
    return _wrap(estimates, prf_hz)


def estimate_fft_at_range(
    iq: Any,
    predicted_range_bins: Any,
    prf_hz: float,
    *,
    nfft: int = 520,
    estimator: str = "three_point_log_parabolic",
) -> np.ndarray:
    """Apply the fixed rectangular, zero-padded FFT and a matched local rule."""
    values = _as_iq(iq)
    bins = _range_indices(predicted_range_bins, len(values))
    if not math.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    if nfft < 130 or nfft % 2:
        raise ValueError("nfft must be even and at least 130")
    if estimator not in {"three_point_log_parabolic", "five_bin_excess_centroid"}:
        raise ValueError("unknown local FFT estimator")

    offsets = np.arange(-2, 3, dtype=np.float64)
    estimates = np.empty(len(values), dtype=np.float64)
    for sample, range_bin in enumerate(bins):
        spectrum = np.fft.fftshift(
            np.fft.fft(values[sample, :, :, range_bin], n=nfft, axis=1), axes=1
        )
        power = np.sum(np.abs(spectrum) ** 2, axis=0, dtype=np.float64)
        peak = int(np.argmax(power))
        if estimator == "three_point_log_parabolic":
            tiny = np.finfo(np.float64).tiny
            left = np.log(max(float(power[(peak - 1) % nfft]), tiny))
            center = np.log(max(float(power[peak]), tiny))
            right = np.log(max(float(power[(peak + 1) % nfft]), tiny))
            denominator = left - 2.0 * center + right
            delta = 0.0 if abs(denominator) <= 1e-12 else 0.5 * (left - right) / denominator
            delta = float(np.clip(delta, -0.5, 0.5))
        else:
            patch = power[(peak + np.arange(-2, 3)) % nfft]
            weights = np.maximum(patch - np.min(patch), 0.0)
            total = float(weights.sum())
            delta = 0.0 if total <= np.finfo(np.float64).tiny else float(np.dot(offsets, weights) / total)
            delta = float(np.clip(delta, -2.0, 2.0))
        estimates[sample] = (peak + delta - nfft // 2) * prf_hz / nfft
    return _wrap(estimates, prf_hz)


def estimate_pooled_fft(
    iq: Any,
    prf_hz: float,
    *,
    nfft: int,
    parabolic: bool = False,
) -> np.ndarray:
    """Estimate Doppler from the noncoherent three-channel/range pooled spectrum."""
    values = _as_iq(iq)
    if not math.isfinite(prf_hz) or prf_hz <= 0.0 or nfft < 130 or nfft % 2:
        raise ValueError("prf_hz must be positive and nfft even and >=130")
    spectrum = np.fft.fftshift(np.fft.fft(values, n=nfft, axis=2), axes=2)
    power = np.sum(np.abs(spectrum) ** 2, axis=(1, 3), dtype=np.float64)
    peak = np.argmax(power, axis=-1)
    offset = np.zeros(len(values), dtype=np.float64)
    if parabolic:
        rows = np.arange(len(values))
        tiny = np.finfo(np.float64).tiny
        left = np.log(np.maximum(power[rows, (peak - 1) % nfft], tiny))
        center = np.log(np.maximum(power[rows, peak], tiny))
        right = np.log(np.maximum(power[rows, (peak + 1) % nfft], tiny))
        denominator = left - 2.0 * center + right
        np.divide(
            0.5 * (left - right),
            denominator,
            out=offset,
            where=np.abs(denominator) > 1e-12,
        )
        np.clip(offset, -0.5, 0.5, out=offset)
    frequency = (peak.astype(np.float64) + offset - nfft // 2) * prf_hz / nfft
    return _wrap(frequency, prf_hz)


_E002_ARRAYS = (
    "iq",
    "truth_presence",
    "truth_range_cell",
    "truth_doppler_hz",
    "truth_initial_phase_rad",
    "truth_target_amplitude",
    "snr_db",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_clean_tracked(root: Path, relative_paths: list[str]) -> dict[str, str]:
    blobs: dict[str, str] = {}
    for relative in relative_paths:
        status = _git(root, "status", "--porcelain", "--", relative)
        if status:
            raise RuntimeError(f"frozen D1-E002 artifact must be committed and clean: {relative}")
        _git(root, "ls-files", "--error-unmatch", "--", relative)
        blobs[relative] = _git(root, "rev-parse", f"HEAD:{relative}")
    return blobs


def _verify_preregistered(root: Path, config_path: Path) -> str:
    protocol = root / "experiments" / "D1-E002" / "protocol.md"
    for path in (config_path, protocol):
        relative = path.resolve().relative_to(root).as_posix()
        committed = subprocess.run(
            ["git", "-C", str(root), "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if path.read_bytes() != committed:
            raise RuntimeError(f"preregistered D1-E002 file differs from HEAD: {relative}")
    return _git(root, "rev-parse", "HEAD")


def _d1_e002_code_paths() -> list[str]:
    return [
        "experiments/D1-E002/config.yaml",
        "experiments/D1-E002/protocol.md",
        "src/radardistill/training/d1_e002.py",
        "src/radardistill/students/d1_e002.py",
        "src/radardistill/evaluation/d1_e002.py",
        "scripts/run_d1_e002.py",
        "scripts/evaluate_d1_e002.py",
        "scripts/validate_d1_e002_protocol.py",
    ]


def _verify_code_unchanged(root: Path, code_commit: str) -> None:
    changed = _git(
        root,
        "diff",
        "--name-only",
        f"{code_commit}..HEAD",
        "--",
        *_d1_e002_code_paths(),
    ).splitlines()
    if changed:
        raise RuntimeError(f"frozen D1-E002 implementation changed: {changed}")


def freeze_d1_e002_models(root: str | Path, config_path: str | Path) -> Path:
    """Write a tracked model/checkpoint manifest after formal outputs are committed."""
    root_path = Path(root).resolve()
    config_file = Path(config_path).resolve()
    _verify_preregistered(root_path, config_file)
    config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    formal_dir = root_path / "experiments" / "D1-E002" / "formal"
    metrics_path = formal_dir / "metrics.json"
    if (formal_dir / "skipped.json").is_file():
        raise RuntimeError("D1-E002 pilot passed no candidates; model freeze and blind evaluation are prohibited")
    if not metrics_path.is_file():
        raise FileNotFoundError("formal metrics are required before checkpoint freeze")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "COMPLETED":
        raise RuntimeError("formal training must complete before checkpoint freeze")
    formal_run = json.loads((formal_dir / "run_manifest.json").read_text(encoding="utf-8"))
    code_commit = str(formal_run.get("code_commit", ""))
    if not code_commit:
        raise RuntimeError("formal run manifest lacks its implementation commit")
    _verify_code_unchanged(root_path, code_commit)
    candidates = list(metrics.get("pilot_selection", {}).get("formal_candidates", []))
    if not candidates:
        raise RuntimeError("formal metrics contain no selected candidate")
    if any(name not in {"S1", "S2", "S3"} for name in candidates):
        raise ValueError(f"unknown formal candidate in metrics: {candidates}")

    entries: list[dict[str, Any]] = []
    for index, split in enumerate(config["splits"]["formal"]):
        seed_id = str(split["id"])
        result = metrics.get("seeds", {}).get(seed_id)
        if not isinstance(result, dict):
            raise RuntimeError(f"formal metrics are missing seed {seed_id}")
        for model_name, checkpoint in (
            ("S0", config["models"]["S0"]["checkpoints"][index]),
            ("S0_D", config["models"]["S0_D"]["checkpoints"][index]),
        ):
            entries.append(
                {
                    "id": f"{model_name}/{seed_id}",
                    "kind": "historical_task_student",
                    "variant": model_name,
                    "seed_id": seed_id,
                    "checkpoint": str(checkpoint),
                }
            )
        for candidate in candidates:
            candidate_result = result.get("candidates", {}).get(candidate, {})
            training_result = candidate_result.get("training", {})
            if int(training_result.get("completed_updates", -1)) != int(
                config["models"][candidate]["optimizer_updates"]
            ):
                raise RuntimeError(f"formal candidate {candidate}/{seed_id} did not complete its frozen budget")
            checkpoint = f"experiments/D1-E002/formal/{seed_id}/{candidate}/best_inference.pt"
            entries.append(
                {
                    "id": f"{candidate}/{seed_id}",
                    "kind": "e002_candidate",
                    "variant": candidate,
                    "seed_id": seed_id,
                    "checkpoint": checkpoint,
                }
            )
    required = [
        "experiments/D1-E002/config.yaml",
        "experiments/D1-E002/protocol.md",
        "experiments/D1-E002/pilot/selection.json",
        "experiments/D1-E002/formal/config_used.yaml",
        "experiments/D1-E002/formal/metrics.json",
        "experiments/D1-E002/formal/run_manifest.json",
    ]
    for entry in entries:
        required.append(entry["checkpoint"])
        if entry["kind"] == "e002_candidate":
            prefix = f"experiments/D1-E002/formal/{entry['seed_id']}/{entry['variant']}"
            required.extend(
                f"{prefix}/{name}"
                for name in ("initial_inference.pt", "summary.json", "training_history.csv")
            )
    blobs = _require_clean_tracked(root_path, required)
    for entry in entries:
        entry["git_blob"] = blobs[entry["checkpoint"]]
    target = formal_dir / "model_manifest.json"
    if target.exists():
        raise FileExistsError(f"model manifest already exists; preserving it: {target}")
    manifest = {
        "experiment_id": "D1-E002",
        "status": "FROZEN",
        "code_commit": code_commit,
        "training_commit": _git(root_path, "rev-parse", "HEAD"),
        "candidate_order": candidates,
        "models": entries,
        "checkpoints_are_tracked_and_clean": True,
        "calibration_and_blind_data_generated": False,
    }
    _write_json(target, manifest)
    return target


def _evaluation_cache_specs(total: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    return {
        "iq": ((total, 3, 130, 32), np.complex64),
        "sample_id": ((total,), np.int64),
        "truth_presence": ((total,), np.bool_),
        "truth_range_cell": ((total,), np.float32),
        "truth_doppler_hz": ((total,), np.float32),
        "truth_initial_phase_rad": ((total,), np.float32),
        "truth_target_amplitude": ((total,), np.float32),
        "snr_db": ((total,), np.float32),
    }


def _write_generated_batch(maps: dict[str, np.ndarray], start: int, batch: Any) -> None:
    for name in _E002_ARRAYS:
        value = getattr(batch, name)
        maps[name][start : start + len(batch.iq)] = value


def _open_or_generate_evaluation_cache(
    *,
    root: Path,
    config: dict[str, Any],
    d0_config: dict[str, Any],
    kind: str,
) -> dict[str, np.ndarray]:
    from ..simulation.d0 import generate_d0_batch

    if kind not in {"calibration", "blind"}:
        raise ValueError("evaluation cache kind must be calibration or blind")
    split = config["splits"][kind]
    snrs = [float(value) for value in split.get("snr_db", config["data"]["snr_db"])]
    chunk_size = int(config["data"]["evaluation_generation_chunk_size"])
    stride = int(config["seeds"]["stream_stride"])
    if kind == "calibration":
        classes = [(0.0, int(split["h0_examples_per_snr"]))]
        seed_bases = {0.0: int(split["h0_seed_base"])}
    else:
        classes = [
            (0.0, int(split["h0_examples_per_snr"])),
            (1.0, int(split["h1_examples_per_snr"])),
        ]
        seed_bases = {0.0: int(split["h0_seed_base"]), 1.0: int(split["h1_seed_base"])}
    total = len(snrs) * sum(count for _, count in classes)
    cache_dir = root / "experiments" / "D1-E002" / "data" / f"{kind}_cache"
    expected = {
        "experiment_id": "D1-E002",
        "kind": kind,
        "stream_stride": stride,
        "chunk_size": chunk_size,
        "snr_db": snrs,
        "classes_per_snr": {str(int(presence)): count for presence, count in classes},
        "class_order": [int(presence) for presence, _ in classes],
        "sample_order": "snr_then_class_then_chunk",
    }
    manifest_path = cache_dir / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"existing {kind} cache does not match the frozen config")
        if any(not (cache_dir / f"{name}.npy").is_file() for name in (*_E002_ARRAYS, "sample_id")):
            raise FileNotFoundError(f"existing {kind} cache is incomplete")
        names = [*_E002_ARRAYS, "sample_id"]
        return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in names}
    if cache_dir.exists():
        raise FileExistsError(f"unmanifested {kind} cache exists; preserving it: {cache_dir}")
    building = cache_dir.with_name(cache_dir.name + ".building")
    if building.exists():
        raise FileExistsError(f"partial {kind} cache exists; preserving it: {building}")
    building.mkdir(parents=True, exist_ok=False)
    maps = {
        name: np.lib.format.open_memmap(building / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in _evaluation_cache_specs(total).items()
    }
    offset = 0
    snr_count = len(snrs)
    for snr_index, snr in enumerate(snrs):
        for presence, count in classes:
            seed_base = seed_bases[presence]
            for chunk_index, chunk_start in enumerate(range(0, count, chunk_size)):
                n = min(chunk_size, count - chunk_start)
                seed = evaluation_chunk_seed(
                    seed_base,
                    snr_index,
                    chunk_index,
                    snr_stride=stride,
                    snr_count=snr_count,
                )
                batch = generate_d0_batch(n, seed, snr, presence, d0_config)
                _write_generated_batch(maps, offset, batch)
                maps["sample_id"][offset : offset + n] = np.arange(offset, offset + n, dtype=np.int64)
                offset += n
    if offset != total:
        raise RuntimeError(f"generated {kind} cache has {offset} rows, expected {total}")
    for array in maps.values():
        array.flush()
    (building / "manifest.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(building, cache_dir)
    names = [*_E002_ARRAYS, "sample_id"]
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in names}


def _model_input(iq: np.ndarray) -> Any:
    import torch

    values = np.asarray(iq, dtype=np.complex64)
    planes = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(planes))


def _load_frozen_model(root: Path, entry: dict[str, Any], config: dict[str, Any], device: Any):
    import torch

    from ..students.d1_e002 import D1E002Student
    from ..students.task_student import TaskStudent

    path = root / entry["checkpoint"]
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("state_dict"), dict):
        raise ValueError(f"invalid inference checkpoint envelope: {entry['checkpoint']}")
    if entry["kind"] == "historical_task_student":
        d1 = yaml.safe_load((root / "experiments/D1-E001/config.yaml").read_text(encoding="utf-8"))
        student = d1["student"]
        model = TaskStudent(
            width=int(student["width"]),
            temporal_dilations=student["temporal_dilations"],
            include_spectrum_head=False,
        )
    elif entry["kind"] == "e002_candidate":
        d1 = yaml.safe_load((root / "experiments/D1-E001/config.yaml").read_text(encoding="utf-8"))
        student = d1["student"]
        model = D1E002Student(
            variant=entry["variant"],
            width=int(student["width"]),
            temporal_dilations=student["temporal_dilations"],
            prf_hz=float(config["radar"]["prf_hz"]),
            fft_nfft=int(config["models"]["S3"]["fft"]["nfft"]),
            coherence_epsilon=float(config["models"]["S2"]["statistic_epsilon"]),
            fft_epsilon=float(config["models"]["S3"]["fft"]["epsilon"]),
        )
    else:
        raise ValueError(f"unknown checkpoint kind {entry['kind']!r}")
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def _predict_model_batch(model: Any, iq: np.ndarray, device: Any, prf_hz: float) -> dict[str, np.ndarray]:
    import torch

    from ..distillation.task_losses import decode_task_outputs
    from ..students.task_student import TaskStudent

    inputs = _model_input(iq).to(device)
    with torch.no_grad():
        if isinstance(model, TaskStudent):
            outputs = model(inputs, compute_spectrum=False)
        else:
            outputs = model(inputs)
        decoded = decode_task_outputs(outputs, prf_hz)
    return {
        "score": outputs["presence_logit"].detach().cpu().numpy().astype(np.float64),
        "range_cell": decoded["range_cell"].detach().cpu().numpy().astype(np.float64),
        "range_bin": outputs["range_logits"].argmax(dim=-1).detach().cpu().numpy().astype(np.int64),
        "doppler_hz": decoded["doppler_hz"].detach().cpu().numpy().astype(np.float64),
    }


def _clean_model_manifest(root: Path, formal_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    path = formal_dir / "model_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("committed formal/model_manifest.json is required before calibration")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "FROZEN" or not manifest.get("models"):
        raise ValueError("D1-E002 model manifest is not frozen")
    _require_clean_tracked(root, _d1_e002_code_paths())
    relative_manifest = path.resolve().relative_to(root).as_posix()
    blobs = _require_clean_tracked(root, [relative_manifest])
    for entry in manifest["models"]:
        relative = str(entry["checkpoint"])
        _require_clean_tracked(root, [relative])
        if _git(root, "rev-parse", f"HEAD:{relative}") != entry.get("git_blob"):
            raise RuntimeError(f"frozen checkpoint blob changed: {relative}")
    code_commit = str(manifest["code_commit"])
    _verify_code_unchanged(root, code_commit)
    return manifest, _git(root, "rev-parse", "HEAD")


def _evaluation_environment(root: Path, device: Any, batch_size: int, elapsed: float) -> dict[str, Any]:
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
    code_paths = _d1_e002_code_paths()
    return {
        "git_sha": _git(root, "rev-parse", "HEAD"),
        "dirty_state": {
            "e002_implementation_paths_clean": not bool(
                _git(root, "status", "--porcelain", "--", *code_paths)
            ),
            "workspace_has_uncommitted_changes": bool(_git(root, "status", "--porcelain")),
        },
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "driver": driver,
        "gpu": properties.name,
        "gpu_total_memory_bytes": int(properties.total_memory),
        "cuda_device": str(device),
        "precision": "float32",
        "batch_size": int(batch_size),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds": float(elapsed),
    }


def _d0_config(root: Path) -> dict[str, Any]:
    return yaml.safe_load((root / "experiments/D0-E001/config.yaml").read_text(encoding="utf-8"))


def _require_cuda(device_name: str = "cuda:0"):
    import torch

    if device_name != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("D1-E002 calibration/blind evaluation requires CUDA cuda:0")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def _load_d1_e002_config(root: Path, config_path: Path) -> dict[str, Any]:
    from ..training.d1_e002 import validate_e002_config, validate_e002_seed_disjointness, prior_registered_seed_values

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"invalid D1-E002 configuration: {config_path}")
    validate_e002_config(config)
    validate_e002_seed_disjointness(config, existing_seed_values=prior_registered_seed_values(root))
    _verify_preregistered(root, config_path)
    return config


def _require_clean_calibration(root: Path, calibration_dir: Path) -> tuple[dict[str, Any], str]:
    metrics_path = calibration_dir / "metrics.json"
    manifest_path = calibration_dir / "run_manifest.json"
    scores_path = calibration_dir / "calibration_scores.npz"
    for path in (metrics_path, manifest_path, scores_path):
        if not path.is_file():
            raise FileNotFoundError(f"committed calibration artifact is required: {path}")
    relative = [path.resolve().relative_to(root).as_posix() for path in (metrics_path, manifest_path, scores_path)]
    _require_clean_tracked(root, relative)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metrics.get("status") != "COMPLETED" or manifest.get("status") != "COMPLETED":
        raise RuntimeError("D1-E002 calibration is incomplete")
    return metrics, _git(root, "rev-parse", "HEAD")


def _score_model(model: Any, iq: np.ndarray, device: Any, prf_hz: float) -> np.ndarray:
    return _predict_model_batch(model, iq, device, prf_hz)["score"]


def _run_model_calibration(
    model: Any,
    arrays: dict[str, np.ndarray],
    device: Any,
    prf_hz: float,
    batch_size: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    for start in range(0, len(arrays["iq"]), batch_size):
        end = min(start + batch_size, len(arrays["iq"]))
        iq = np.asarray(arrays["iq"][start:end]).copy()
        scores.append(_score_model(model, iq, device, prf_hz))
    return np.concatenate(scores)


def run_d1_e002_calibration(
    root: str | Path,
    config_path: str | Path,
    *,
    batch_size: int = 64,
) -> Path:
    """Generate/reuse new CPU H0 calibration data and freeze method thresholds."""
    import torch

    from ..simulation.d0 import generate_d0_batch
    from ..teachers.apx15_doppler import APX15DopplerTeacher
    from ..training.d1_runner import calibrate_score_threshold

    root_path = Path(root).resolve()
    config_file = Path(config_path).resolve()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = _load_d1_e002_config(root_path, config_file)
    device = _require_cuda(str(config["training"]["device"]))
    formal_dir = root_path / "experiments" / "D1-E002" / "formal"
    model_manifest, model_commit = _clean_model_manifest(root_path, formal_dir, config)
    calibration_dir = root_path / "experiments" / "D1-E002" / "calibration"
    if calibration_dir.exists():
        raise FileExistsError(f"calibration output exists; preserving it: {calibration_dir}")
    calibration_dir.mkdir(parents=True)
    started = time.perf_counter()
    run_manifest = {
        "experiment_id": "D1-E002",
        "phase": "calibration",
        "status": "RUNNING",
        "code_commit": model_manifest["code_commit"],
        "model_manifest_commit": model_commit,
        "blind_data_generated": False,
        "batch_size": batch_size,
    }
    _write_json(calibration_dir / "run_manifest.json", run_manifest)
    partial: dict[str, Any] = {
        "experiment_id": "D1-E002",
        "phase": "calibration",
        "status": "RUNNING",
        "model_manifest_commit": model_commit,
        "methods": {},
    }
    _write_json(calibration_dir / "metrics_partial.json", partial)
    scores_by_method: dict[str, np.ndarray] = {}
    d0 = _d0_config(root_path)
    try:
        calibration_arrays = _open_or_generate_evaluation_cache(
            root=root_path,
            config=config,
            d0_config=d0,
            kind="calibration",
        )
        pfa = float(config["acceptance"]["detection_guardrail"]["pfa"])
        prf = float(config["radar"]["prf_hz"])
        for entry in model_manifest["models"]:
            model = _load_frozen_model(root_path, entry, config, device)
            try:
                method_scores = _run_model_calibration(
                    model, calibration_arrays, device, prf, batch_size
                )
            finally:
                model.to("cpu")
                del model
                gc.collect()
                torch.cuda.empty_cache()
            threshold = calibrate_score_threshold(method_scores, pfa)
            scores_by_method[entry["id"]] = method_scores
            partial["methods"][entry["id"]] = {
                "score_source": "presence_logit",
                "threshold": threshold,
                "n_h0": int(len(method_scores)),
                "pfa_target": pfa,
                "calibration_pfa": float(np.mean(method_scores > threshold)),
                "comparison": "score_strictly_greater_than_threshold",
            }
            _write_json(calibration_dir / "metrics_partial.json", partial)
            print(f"[calibration] {entry['id']}: threshold={threshold:.6g}", flush=True)

        d0_teacher = APX15DopplerTeacher(d0)
        try:
            teacher_scores: list[np.ndarray] = []
            for start in range(0, len(calibration_arrays["iq"]), batch_size):
                end = min(start + batch_size, len(calibration_arrays["iq"]))
                result = d0_teacher.predict(np.asarray(calibration_arrays["iq"][start:end]).copy())
                teacher_scores.append(result.teacher_detection_score.astype(np.float64))
        finally:
            d0_teacher.close()
        teacher_values = np.concatenate(teacher_scores)
        threshold = calibrate_score_threshold(teacher_values, pfa)
        scores_by_method["APX15_teacher"] = teacher_values
        partial["methods"]["APX15_teacher"] = {
            "score_source": "D0_teacher_detection_score",
            "threshold": threshold,
            "n_h0": int(len(teacher_values)),
            "pfa_target": pfa,
            "calibration_pfa": float(np.mean(teacher_values > threshold)),
            "comparison": "score_strictly_greater_than_threshold",
        }
        energy_parts: list[np.ndarray] = []
        for start in range(0, len(calibration_arrays["iq"]), batch_size):
            end = min(start + batch_size, len(calibration_arrays["iq"]))
            energy_parts.append(energy_detection_score(calibration_arrays["iq"][start:end]))
        energy_scores = np.concatenate(energy_parts)
        energy_threshold = calibrate_score_threshold(energy_scores, pfa)
        scores_by_method["nonlearning_iq_energy_detector"] = energy_scores
        partial["methods"]["nonlearning_iq_energy_detector"] = {
            "score_source": "max_range_noncoherent_integrated_energy",
            "threshold": energy_threshold,
            "n_h0": int(len(energy_scores)),
            "pfa_target": pfa,
            "calibration_pfa": float(np.mean(energy_scores > energy_threshold)),
            "comparison": "score_strictly_greater_than_threshold",
        }
        partial["status"] = "COMPLETED"
        partial["calibration_samples_per_method"] = int(len(energy_scores))
        partial["threshold_rule"] = "ceil((N+1)*(1-Pfa))-1 order statistic; strict greater-than"
        partial["environment"] = _evaluation_environment(
            root_path, device, batch_size, time.perf_counter() - started
        )
        _write_json(calibration_dir / "environment.json", partial["environment"])
        score_file = calibration_dir / "calibration_scores.npz"
        np.savez_compressed(
            score_file,
            **{key.replace("/", "__"): value for key, value in scores_by_method.items()},
        )
        _write_json(calibration_dir / "metrics.json", partial)
        (calibration_dir / "metrics_partial.json").unlink(missing_ok=True)
        run_manifest["status"] = "COMPLETED"
        run_manifest["environment"] = partial["environment"]
        _write_json(calibration_dir / "run_manifest.json", run_manifest)
        return calibration_dir
    except BaseException as error:
        partial["status"] = "FAILED"
        partial["failure"] = f"{type(error).__name__}: {error}"
        _write_json(calibration_dir / "metrics_partial.json", partial)
        run_manifest["status"] = "FAILED"
        run_manifest["failure"] = partial["failure"]
        _write_json(calibration_dir / "run_manifest.json", run_manifest)
        raise


def _metadata_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        **{name: np.asarray(arrays[name]) for name in _E002_ARRAYS if name != "iq"},
        "sample_id": np.asarray(arrays["sample_id"]),
    }


def _write_prediction(path: Path, predictions: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **predictions)
    temporary.replace(path)


def _predict_frozen_model_cache(
    root: Path,
    entry: dict[str, Any],
    config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    device: Any,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    model = _load_frozen_model(root, entry, config, device)
    parts: dict[str, list[np.ndarray]] = {name: [] for name in ("score", "pred_range", "pred_doppler_hz")}
    matched_three: list[np.ndarray] = []
    matched_centroid: list[np.ndarray] = []
    matched_energy: list[np.ndarray] = []
    try:
        for start in range(0, len(arrays["iq"]), batch_size):
            end = min(start + batch_size, len(arrays["iq"]))
            iq = np.asarray(arrays["iq"][start:end]).copy()
            predicted = _predict_model_batch(model, iq, device, float(config["radar"]["prf_hz"]))
            parts["score"].append(predicted["score"])
            parts["pred_range"].append(predicted["range_cell"])
            parts["pred_doppler_hz"].append(predicted["doppler_hz"])
            if entry["variant"] == "S2":
                matched_energy.append(energy_detection_score(iq))
                matched_three.append(
                    estimate_lag_one_at_range(iq, predicted["range_bin"], float(config["radar"]["prf_hz"]))
                )
            elif entry["variant"] == "S3":
                matched_energy.append(energy_detection_score(iq))
                matched_three.append(
                    estimate_fft_at_range(
                        iq,
                        predicted["range_bin"],
                        float(config["radar"]["prf_hz"]),
                        nfft=520,
                        estimator="three_point_log_parabolic",
                    )
                )
                matched_centroid.append(
                    estimate_fft_at_range(
                        iq,
                        predicted["range_bin"],
                        float(config["radar"]["prf_hz"]),
                        nfft=520,
                        estimator="five_bin_excess_centroid",
                    )
                )
    finally:
        model.to("cpu")
        del model
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    metadata = _metadata_arrays(arrays)
    prediction = {
        **metadata,
        "score": np.concatenate(parts["score"]),
        "pred_range_cell": np.concatenate(parts["pred_range"]),
        "pred_doppler_hz": np.concatenate(parts["pred_doppler_hz"]),
    }
    matched = None
    if entry["variant"] == "S2":
        matched = {
            **metadata,
            "score": np.concatenate(matched_energy),
            "pred_range_cell": prediction["pred_range_cell"],
            "pred_doppler_hz": np.concatenate(matched_three),
        }
    elif entry["variant"] == "S3":
        matched = {
            "three_point": {
                **metadata,
                "score": np.concatenate(matched_energy),
                "pred_range_cell": prediction["pred_range_cell"],
                "pred_doppler_hz": np.concatenate(matched_three),
            },
            "five_bin_centroid": {
                **metadata,
                "score": np.concatenate(matched_energy),
                "pred_range_cell": prediction["pred_range_cell"],
                "pred_doppler_hz": np.concatenate(matched_centroid),
            },
        }
    return prediction, matched


def _predict_teacher_cache(
    root: Path,
    d0_config: dict[str, Any],
    arrays: dict[str, np.ndarray],
    batch_size: int,
) -> dict[str, np.ndarray]:
    from ..teachers.apx15_doppler import APX15DopplerTeacher

    teacher = APX15DopplerTeacher(d0_config)
    parts = {name: [] for name in ("score", "pred_range_cell", "pred_doppler_hz", "confidence")}
    try:
        for start in range(0, len(arrays["iq"]), batch_size):
            end = min(start + batch_size, len(arrays["iq"]))
            result = teacher.predict(np.asarray(arrays["iq"][start:end]).copy())
            parts["score"].append(result.teacher_detection_score)
            parts["pred_range_cell"].append(result.teacher_range_cell)
            parts["pred_doppler_hz"].append(result.teacher_doppler_hz)
            parts["confidence"].append(result.teacher_confidence)
    finally:
        teacher.close()
    return {
        **_metadata_arrays(arrays),
        **{name: np.concatenate(values) for name, values in parts.items()},
    }


def _predict_classical_cache(
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    batch_size: int,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    score_parts: list[np.ndarray] = []
    methods = (
        "native_fft_peak",
        "native_fft_log_parabolic",
        "fourfold_fft_peak",
        "fourfold_fft_parabolic",
        "lag_one_equal_channel",
    )
    range_parts = {name: [] for name in methods}
    doppler_parts = {name: [] for name in methods}
    prf = float(config["radar"]["prf_hz"])
    for start in range(0, len(arrays["iq"]), batch_size):
        end = min(start + batch_size, len(arrays["iq"]))
        iq = np.asarray(arrays["iq"][start:end]).copy()
        energy = energy_detection_score(iq)
        score_parts.append(energy)
        native_peak = estimate_pooled_fft(iq, prf, nfft=130, parabolic=False)
        native_parabolic = estimate_pooled_fft(iq, prf, nfft=130, parabolic=True)
        fourfold_peak = estimate_pooled_fft(iq, prf, nfft=520, parabolic=False)
        fourfold_parabolic = estimate_pooled_fft(iq, prf, nfft=520, parabolic=True)
        range_estimate = np.asarray(
            [np.argmax(np.sum(np.abs(sample) ** 2, axis=(0, 1))) for sample in iq], dtype=np.float64
        )
        range_bins = np.clip(np.rint(range_estimate), 0, 31).astype(np.int64)
        lag_estimate = estimate_lag_one_at_range(iq, range_bins, prf)
        estimates = {
            "native_fft_peak": native_peak,
            "native_fft_log_parabolic": native_parabolic,
            "fourfold_fft_peak": fourfold_peak,
            "fourfold_fft_parabolic": fourfold_parabolic,
            "lag_one_equal_channel": lag_estimate,
        }
        for name, estimate in estimates.items():
            range_parts[name].append(range_estimate)
            doppler_parts[name].append(estimate)
    metadata = _metadata_arrays(arrays)
    predicted = {
        name: {
            **metadata,
            "score": np.concatenate(score_parts),
            "pred_range_cell": np.concatenate(range_parts[name]),
            "pred_doppler_hz": np.concatenate(doppler_parts[name]),
            "score_source": np.asarray("nonlearning_iq_energy_detector"),
        }
        for name in methods
    }
    energy_prediction = {**metadata, "score": np.concatenate(score_parts)}
    return predicted, energy_prediction


def _profile_ml_prediction(
    root: Path,
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    from .d1_evaluation import estimate_profile_ml_reference

    d1 = yaml.safe_load((root / "experiments/D1-E001/config.yaml").read_text(encoding="utf-8"))
    count = int(config["baselines"]["profile_ml"]["h1_examples_per_snr"])
    snrs = [float(value) for value in config["data"]["snr_db"]]
    masks = [np.isclose(arrays["snr_db"], snr) & np.asarray(arrays["truth_presence"], dtype=bool) for snr in snrs]
    selected: list[int] = []
    for mask in masks:
        selected.extend(np.flatnonzero(mask)[:count].tolist())
    indices = np.asarray(selected, dtype=np.int64)
    profile = estimate_profile_ml_reference(
        np.asarray(arrays["iq"][indices]).copy(),
        prf_hz=float(config["radar"]["prf_hz"]),
        range_interval=tuple(d1["waveform"]["matched_filter_psf"]["target_range_cell_interval"]),
        range_grid_step=float(d1["baselines"]["profile_likelihood"]["range_grid_step_cells"]),
        frequency_fft_factor=int(d1["baselines"]["profile_likelihood"]["frequency_fft_factor"]),
        chunk_size=1,
    )
    return {
        **{name: np.asarray(arrays[name][indices]) for name in _E002_ARRAYS if name != "iq"},
        "sample_id": np.asarray(arrays["sample_id"][indices]),
        "score": np.full(len(indices), np.nan, dtype=np.float64),
        "pred_range_cell": profile["range_cell"],
        "pred_doppler_hz": profile["doppler_hz"],
    }


def _predict_physical_consistency(
    root: Path,
    entry: dict[str, Any],
    config: dict[str, Any],
    device: Any,
    batch_size: int,
) -> dict[str, Any]:
    from ..simulation.d0 import generate_d0_batch

    checks = config["splits"]["physical_consistency"]
    count = int(checks["examples"])
    snrs = list(checks["snr_db"])
    per_snr = count // len(snrs)
    snr_values = np.repeat(np.asarray(snrs, dtype=np.float64), per_snr)
    base = generate_d0_batch(
        count,
        int(checks["seed"]),
        snr_values,
        1.0,
        _d0_config(root),
    ).iq
    model = _load_frozen_model(root, entry, config, device)
    try:
        def predict_chunks(iq_values: np.ndarray) -> dict[str, np.ndarray]:
            pieces = {key: [] for key in ("score", "range_cell", "doppler_hz")}
            for start in range(0, len(iq_values), batch_size):
                output = _predict_model_batch(
                    model,
                    iq_values[start : start + batch_size],
                    device,
                    float(config["radar"]["prf_hz"]),
                )
                for key in pieces:
                    pieces[key].append(output[key])
            return {key: np.concatenate(value) for key, value in pieces.items()}

        baseline = predict_chunks(base)
        phase_rows = []
        for theta in checks["common_phase_rotation_radians"]:
            rotated = base * np.exp(1j * float(theta))
            output = predict_chunks(rotated)
            phase_rows.append(
                {
                    "rotation_rad": float(theta),
                    "mean_abs_wrapped_doppler_change_hz": float(
                        np.mean(np.abs(_wrap(output["doppler_hz"] - baseline["doppler_hz"], float(config["radar"]["prf_hz"]))))
                    ),
                    "p95_abs_wrapped_doppler_change_hz": float(
                        np.quantile(np.abs(_wrap(output["doppler_hz"] - baseline["doppler_hz"], float(config["radar"]["prf_hz"]))), 0.95)
                    ),
                    "mean_abs_presence_logit_change": float(np.mean(np.abs(output["score"] - baseline["score"]))),
                    "mean_abs_range_cell_change": float(np.mean(np.abs(output["range_cell"] - baseline["range_cell"]))),
                }
            )
        pulse = np.arange(base.shape[2], dtype=np.float64)[None, None, :, None]
        shift_rows = []
        for delta in checks["frequency_shift_equivariance_hz"]:
            for signed_delta in (-float(delta), float(delta)):
                shifted_iq = base * np.exp(
                    2j * np.pi * signed_delta * pulse / float(config["radar"]["prf_hz"])
                )
                output = predict_chunks(shifted_iq)
                residual = _wrap(
                    output["doppler_hz"] - baseline["doppler_hz"] - signed_delta,
                    float(config["radar"]["prf_hz"]),
                )
                shift_rows.append(
                    {
                        "frequency_shift_hz": signed_delta,
                        "mean_abs_equivariance_residual_hz": float(np.mean(np.abs(residual))),
                        "p95_abs_equivariance_residual_hz": float(np.quantile(np.abs(residual), 0.95)),
                        "mean_abs_presence_logit_change": float(np.mean(np.abs(output["score"] - baseline["score"]))),
                        "mean_abs_range_cell_change": float(np.mean(np.abs(output["range_cell"] - baseline["range_cell"]))),
                    }
                )
    finally:
        model.to("cpu")
        del model
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    return {"common_phase_rotation": phase_rows, "frequency_shift_equivariance": shift_rows}


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=bool).reshape(-1)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return math.nan
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
    return float(
        (ranks[sorted_labels].sum() - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total**2)) / denominator
    return [float(max(0.0, center - radius)), float(min(1.0, center + radius))]


def _crlb_doppler_sigma_hz(
    snr_db: np.ndarray,
    range_cell: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    prf = float(config["radar"]["prf_hz"])
    pulse_count = int(config["radar"]["pulse_count"])
    channel_count = int(config["radar"]["channel_count"])
    range_count = int(config["radar"]["range_bin_count"])
    n = np.arange(pulse_count, dtype=np.float64)
    temporal_energy = float(np.sum((n - n.mean()) ** 2))
    cells = np.arange(range_count, dtype=np.float64)
    range_energy = np.sum(
        np.sinc(np.asarray(range_cell, dtype=np.float64)[:, None] - cells[None, :]) ** 2,
        axis=1,
    )
    linear_snr = np.power(10.0, np.asarray(snr_db, dtype=np.float64) / 10.0)
    fisher = (
        2.0
        * linear_snr
        * channel_count
        * range_energy
        * (2.0 * np.pi / prf) ** 2
        * temporal_energy
    )
    return np.sqrt(1.0 / fisher)


def _estimation_summary(
    prediction: dict[str, np.ndarray],
    mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any] | None:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return None
    truth_range = np.asarray(prediction["truth_range_cell"], dtype=np.float64)[mask]
    truth_doppler = np.asarray(prediction["truth_doppler_hz"], dtype=np.float64)[mask]
    pred_range = np.asarray(prediction["pred_range_cell"], dtype=np.float64)[mask]
    pred_doppler = np.asarray(prediction["pred_doppler_hz"], dtype=np.float64)[mask]
    range_error = pred_range - truth_range
    doppler_error = _wrap(pred_doppler - truth_doppler, float(config["radar"]["prf_hz"]))
    threshold = float(config["acceptance"]["stratification"]["doppler_outlier_threshold_hz"])
    return {
        "n": int(mask.sum()),
        "range_mae_cells": float(np.mean(np.abs(range_error))),
        "range_rmse_cells": float(np.sqrt(np.mean(range_error**2))),
        "doppler_mae_hz": float(np.mean(np.abs(doppler_error))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error**2))),
        "doppler_bias_hz": float(np.mean(doppler_error)),
        "doppler_outlier_rate": float(np.mean(np.abs(doppler_error) > threshold)),
        "conditional_doppler_crlb_rmse_hz": float(
            np.mean(
                _crlb_doppler_sigma_hz(
                    np.asarray(prediction["snr_db"], dtype=np.float64)[mask],
                    truth_range,
                    config,
                )
            )
        ),
    }


def _group_masks(
    prediction: dict[str, np.ndarray],
    base_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    groups: dict[str, np.ndarray] = {}
    doppler = np.asarray(prediction["truth_doppler_hz"], dtype=np.float64)
    phase = np.asarray(prediction["truth_initial_phase_rad"], dtype=np.float64)
    amplitude = np.asarray(prediction["truth_target_amplitude"], dtype=np.float64)
    range_cell = np.asarray(prediction["truth_range_cell"], dtype=np.float64)
    doppler_edges = np.linspace(-650.0, 650.0, 5)
    range_edges = np.linspace(4.0, 28.0, 5)
    for index in range(4):
        left, right = doppler_edges[index : index + 2]
        local = (doppler >= left) & (doppler < right)
        groups[f"doppler_quartile_{index + 1}"] = base_mask & local
        rleft, rright = range_edges[index : index + 2]
        groups[f"range_quartile_{index + 1}"] = base_mask & (range_cell >= rleft) & (range_cell < rright)
    groups["doppler_negative"] = base_mask & (doppler < 0.0)
    groups["doppler_nonnegative"] = base_mask & (doppler >= 0.0)
    nyquist_width = float(config["acceptance"]["stratification"]["nyquist_neighborhood_half_width_hz"])
    groups["nyquist_neighborhood"] = base_mask & (np.abs(doppler) >= 650.0 - nyquist_width)
    finite_phase = np.where(np.isfinite(phase), phase, 0.0)
    phase_bin = np.floor((np.mod(finite_phase, 2.0 * np.pi) / (2.0 * np.pi)) * 8).astype(np.int64)
    for index in range(8):
        groups[f"initial_phase_octant_{index + 1}"] = base_mask & (phase_bin == index)
    if np.any(base_mask):
        quartiles = np.quantile(amplitude[base_mask], [0.25, 0.5, 0.75])
        amplitude_bin = np.digitize(amplitude, quartiles, right=False)
        for index in range(4):
            groups[f"amplitude_quartile_{index + 1}"] = base_mask & (amplitude_bin == index)
    return groups


def _summarize_prediction(
    name: str,
    prediction: dict[str, np.ndarray],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    presence = np.asarray(prediction["truth_presence"], dtype=bool)
    snr = np.asarray(prediction["snr_db"], dtype=np.float64)
    score = np.asarray(prediction["score"], dtype=np.float64)
    threshold_record = calibration["methods"].get(name)
    if threshold_record is None and "nonlearning_iq_energy_detector" in calibration["methods"]:
        threshold_record = calibration["methods"]["nonlearning_iq_energy_detector"]
    detection: dict[str, Any] | None = None
    if threshold_record is not None and np.isfinite(score).all():
        threshold = float(threshold_record["threshold"])
        detected = score > threshold
        by_snr: dict[str, Any] = {}
        for value in sorted(np.unique(snr)):
            group = np.isclose(snr, value)
            h0 = group & ~presence
            h1 = group & presence
            tp = int(np.count_nonzero(detected & h1))
            fp = int(np.count_nonzero(detected & h0))
            pfa = float(fp / max(int(h0.sum()), 1))
            pd = float(tp / max(int(h1.sum()), 1))
            by_snr[str(int(value))] = {
                "n_h0": int(h0.sum()),
                "n_h1": int(h1.sum()),
                "auc": _roc_auc(score[group], presence[group]),
                "pd": pd,
                "pd_wilson_95_ci": _wilson(tp, int(h1.sum())),
                "pfa": pfa,
                "pfa_wilson_95_ci": _wilson(fp, int(h0.sum())),
                "tp": tp,
                "fp": fp,
                "tn": int(h0.sum()) - fp,
                "fn": int(h1.sum()) - tp,
                "threshold": threshold,
            }
        detection = {"score_source": threshold_record["score_source"], "threshold": threshold, "by_snr_db": by_snr}
    h1 = presence & np.isfinite(np.asarray(prediction.get("pred_doppler_hz", np.full(len(presence), np.nan))))
    estimation: dict[str, Any] | None = None
    if np.any(h1):
        overall = _estimation_summary(prediction, h1, config)
        by_snr = {
            str(int(value)): _estimation_summary(prediction, h1 & np.isclose(snr, value), config)
            for value in sorted(np.unique(snr[h1]))
        }
        groups = {
            key: summary
            for key, mask in _group_masks(prediction, h1, config).items()
            if (summary := _estimation_summary(prediction, mask, config)) is not None
        }
        conditional = None
        if detection is not None:
            conditional = _estimation_summary(prediction, h1 & (score > float(threshold_record["threshold"])), config)
        estimation = {"overall": overall, "by_snr_db": by_snr, "by_group": groups, "detected_h1_conditionally": conditional}
    return {
        "method_kind": "profile_ml_estimation_only" if name == "profile_ml" else name.split("/")[0],
        "n_total": int(len(presence)),
        "n_h1": int(presence.sum()),
        "detection": detection,
        "estimation": estimation,
    }


def _metric_delta_bootstrap(
    candidates: list[dict[str, np.ndarray]],
    comparators: list[dict[str, np.ndarray]],
    *,
    metric: str,
    config: dict[str, Any],
    candidate_thresholds: list[float] | None = None,
    comparator_thresholds: list[float] | None = None,
) -> tuple[float, float]:
    reps = int(config["acceptance"]["uncertainty"]["bootstrap_replicates"])
    seed = int(config["acceptance"]["uncertainty"]["bootstrap_seed"])
    rng = np.random.default_rng(seed)
    if len(candidates) != 3 or len(comparators) != 3:
        raise ValueError("D1-E002 hierarchical bootstrap requires three paired formal seeds")
    values = np.empty(reps, dtype=np.float64)
    primary_snrs = [float(value) for value in config["acceptance"]["primary_endpoint"]["snr_db"]]
    low_snrs = [float(value) for value in config["acceptance"]["detection_guardrail"]["low_snr_db"]]
    prf = float(config["radar"]["prf_hz"])
    for rep in range(reps):
        chosen_seeds = rng.integers(0, 3, size=3)
        seed_deltas: list[float] = []
        for seed_index in chosen_seeds:
            cand = candidates[int(seed_index)]
            comp = comparators[int(seed_index)]
            if metric in {"doppler", "range"}:
                snrs = primary_snrs
                snr_values: list[float] = []
                for snr_value in snrs:
                    presence = np.asarray(cand["truth_presence"], dtype=bool)
                    mask = presence & np.isclose(cand["snr_db"], snr_value)
                    indices = np.flatnonzero(mask)
                    sampled = rng.choice(indices, size=len(indices), replace=True)
                    if metric == "doppler":
                        cand_error = _wrap(
                            np.asarray(cand["pred_doppler_hz"])[sampled]
                            - np.asarray(cand["truth_doppler_hz"])[sampled],
                            prf,
                        )
                        comp_error = _wrap(
                            np.asarray(comp["pred_doppler_hz"])[sampled]
                            - np.asarray(comp["truth_doppler_hz"])[sampled],
                            prf,
                        )
                    else:
                        cand_error = np.asarray(cand["pred_range_cell"])[sampled] - np.asarray(cand["truth_range_cell"])[sampled]
                        comp_error = np.asarray(comp["pred_range_cell"])[sampled] - np.asarray(comp["truth_range_cell"])[sampled]
                    snr_values.append(float(np.sqrt(np.mean(cand_error**2)) - np.sqrt(np.mean(comp_error**2))))
                seed_deltas.append(float(np.mean(snr_values)))
            elif metric == "detection":
                if candidate_thresholds is None or comparator_thresholds is None:
                    raise ValueError("detection bootstrap needs paired frozen thresholds")
                per_snr: list[float] = []
                for snr_value in low_snrs:
                    mask = np.asarray(cand["truth_presence"], dtype=bool) & np.isclose(cand["snr_db"], snr_value)
                    indices = np.flatnonzero(mask)
                    sampled = rng.choice(indices, size=len(indices), replace=True)
                    cand_pd = np.mean(np.asarray(cand["score"])[sampled] > candidate_thresholds[int(seed_index)])
                    comp_pd = np.mean(np.asarray(comp["score"])[sampled] > comparator_thresholds[int(seed_index)])
                    per_snr.append(float(cand_pd - comp_pd))
                seed_deltas.append(float(np.mean(per_snr)))
            else:
                raise ValueError(f"unknown bootstrap metric {metric}")
        values[rep] = float(np.mean(seed_deltas))
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _primary_rmse(summary: dict[str, Any], metric_name: str) -> float:
    values = summary["estimation"]["by_snr_db"]
    return float(np.mean([values[str(snr)][metric_name] for snr in (0, 5, 10, 20)]))


def _candidate_gate(
    variant: str,
    *,
    prediction_by_name: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict[str, Any]],
    calibration: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    seed_ids = [str(split["id"]) for split in config["splits"]["formal"]]
    candidate_names = [f"{variant}/{seed_id}" for seed_id in seed_ids]
    s0_names = [f"S0/{seed_id}" for seed_id in seed_ids]
    if variant == "S3":
        doppler_comparator_names = [f"S3_matched_three_point/{seed_id}" for seed_id in seed_ids]
        primary_label = "matched_nonlearning_three_point_fft"
    else:
        doppler_comparator_names = s0_names
        primary_label = "S0"
    candidate_metrics = [summaries[name] for name in candidate_names]
    comparator_metrics = [summaries[name] for name in doppler_comparator_names]
    s0_metrics = [summaries[name] for name in s0_names]
    for candidate_name, comparator_name in zip(candidate_names, doppler_comparator_names):
        if not np.array_equal(
            prediction_by_name[candidate_name]["sample_id"],
            prediction_by_name[comparator_name]["sample_id"],
        ):
            raise RuntimeError(f"paired blind sample IDs differ: {candidate_name} vs {comparator_name}")
    for candidate_name, comparator_name in zip(candidate_names, s0_names):
        if not np.array_equal(
            prediction_by_name[candidate_name]["sample_id"],
            prediction_by_name[comparator_name]["sample_id"],
        ):
            raise RuntimeError(f"paired blind sample IDs differ: {candidate_name} vs {comparator_name}")
    primary_key = "doppler_rmse_hz"
    range_key = "range_rmse_cells"
    candidate_rmse = float(np.mean([_primary_rmse(item, primary_key) for item in candidate_metrics]))
    comparator_rmse = float(np.mean([_primary_rmse(item, primary_key) for item in comparator_metrics]))
    relative_gain = (comparator_rmse - candidate_rmse) / comparator_rmse if comparator_rmse > 0 else math.nan
    per_seed_delta = [
        _primary_rmse(candidate, primary_key) - _primary_rmse(comparator, primary_key)
        for candidate, comparator in zip(candidate_metrics, comparator_metrics)
    ]
    primary_ci = _metric_delta_bootstrap(
        [prediction_by_name[name] for name in candidate_names],
        [prediction_by_name[name] for name in doppler_comparator_names],
        metric="doppler",
        config=config,
    )
    candidate_thresholds = [float(calibration["methods"][name]["threshold"]) for name in candidate_names]
    s0_thresholds = [float(calibration["methods"][name]["threshold"]) for name in s0_names]
    detection_ci = _metric_delta_bootstrap(
        [prediction_by_name[name] for name in candidate_names],
        [prediction_by_name[name] for name in s0_names],
        metric="detection",
        config=config,
        candidate_thresholds=candidate_thresholds,
        comparator_thresholds=s0_thresholds,
    )
    candidate_range = float(np.mean([_primary_rmse(item, range_key) for item in candidate_metrics]))
    s0_range = float(np.mean([_primary_rmse(item, range_key) for item in s0_metrics]))
    range_ci = _metric_delta_bootstrap(
        [prediction_by_name[name] for name in candidate_names],
        [prediction_by_name[name] for name in s0_names],
        metric="range",
        config=config,
    )
    low_snrs = [int(value) for value in config["acceptance"]["detection_guardrail"]["low_snr_db"]]
    pointwise_detection_delta: dict[str, float] = {}
    for snr in low_snrs:
        values = []
        for candidate_name, s0_name in zip(candidate_names, s0_names):
            c = summaries[candidate_name]["detection"]["by_snr_db"][str(snr)]["pd"]
            b = summaries[s0_name]["detection"]["by_snr_db"][str(snr)]["pd"]
            values.append(float(c - b))
        pointwise_detection_delta[str(snr)] = float(np.mean(values))
    gain_gate = relative_gain >= 0.10 and primary_ci[1] < 0.0 and sum(value < 0.0 for value in per_seed_delta) >= 2
    detection_gate = detection_ci[0] > -0.05 and min(pointwise_detection_delta.values()) >= -0.10
    range_gate = (candidate_range - s0_range) < 0.05 and range_ci[1] < 0.05
    return {
        "candidate": variant,
        "doppler_comparator": primary_label,
        "candidate_mean_primary_rmse_hz": candidate_rmse,
        "comparator_mean_primary_rmse_hz": comparator_rmse,
        "relative_gain": relative_gain,
        "per_seed_doppler_delta_hz": dict(zip(seed_ids, per_seed_delta)),
        "doppler_delta_paired_hierarchical_bootstrap_95_ci_hz": list(primary_ci),
        "low_snr_mean_pd_delta_paired_hierarchical_bootstrap_95_ci": list(detection_ci),
        "low_snr_pointwise_pd_delta": pointwise_detection_delta,
        "high_snr_range_rmse_delta_cells": float(candidate_range - s0_range),
        "high_snr_range_delta_paired_hierarchical_bootstrap_95_ci_cells": list(range_ci),
        "gates": {
            "doppler_gain": bool(gain_gate),
            "low_snr_detection_noninferiority": bool(detection_gate),
            "high_snr_range_noninferiority": bool(range_gate),
        },
        "passed": bool(gain_gate and detection_gate and range_gate),
    }


def _write_result_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    fields = (
        "method",
        "snr_db",
        "n_h0",
        "n_h1",
        "auc",
        "pd_at_pfa_1e_2",
        "pd_wilson_95_lower",
        "pd_wilson_95_upper",
        "pfa_at_pfa_1e_2",
        "range_mae_cells",
        "range_rmse_cells",
        "doppler_mae_hz",
        "doppler_rmse_hz",
        "doppler_bias_hz",
        "doppler_outlier_rate",
        "conditional_doppler_crlb_rmse_hz",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, summary in summaries.items():
            snrs = set()
            if summary.get("detection"):
                snrs.update(summary["detection"]["by_snr_db"])
            if summary.get("estimation"):
                snrs.update(summary["estimation"]["by_snr_db"])
            for snr in sorted(snrs, key=float):
                detection = summary.get("detection")
                detection_row = detection["by_snr_db"].get(snr) if detection else None
                estimation = summary.get("estimation")
                estimate_row = estimation["by_snr_db"].get(snr) if estimation else None
                interval = detection_row.get("pd_wilson_95_ci") if detection_row else None
                writer.writerow(
                    {
                        "method": name,
                        "snr_db": snr,
                        "n_h0": detection_row.get("n_h0") if detection_row else "",
                        "n_h1": detection_row.get("n_h1") if detection_row else (estimate_row or {}).get("n", ""),
                        "auc": detection_row.get("auc") if detection_row else "",
                        "pd_at_pfa_1e_2": detection_row.get("pd") if detection_row else "",
                        "pd_wilson_95_lower": interval[0] if interval else "",
                        "pd_wilson_95_upper": interval[1] if interval else "",
                        "pfa_at_pfa_1e_2": detection_row.get("pfa") if detection_row else "",
                        "range_mae_cells": (estimate_row or {}).get("range_mae_cells", ""),
                        "range_rmse_cells": (estimate_row or {}).get("range_rmse_cells", ""),
                        "doppler_mae_hz": (estimate_row or {}).get("doppler_mae_hz", ""),
                        "doppler_rmse_hz": (estimate_row or {}).get("doppler_rmse_hz", ""),
                        "doppler_bias_hz": (estimate_row or {}).get("doppler_bias_hz", ""),
                        "doppler_outlier_rate": (estimate_row or {}).get("doppler_outlier_rate", ""),
                        "conditional_doppler_crlb_rmse_hz": (estimate_row or {}).get("conditional_doppler_crlb_rmse_hz", ""),
                    }
                )


def _plot_blind_results(
    blind_dir: Path,
    predictions: dict[str, dict[str, np.ndarray]],
    summaries: dict[str, dict[str, Any]],
    snrs: list[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(12, 7))
    for name, summary in summaries.items():
        estimation = summary.get("estimation")
        if not estimation:
            continue
        metrics = estimation["by_snr_db"]
        y = [metrics[str(int(snr))]["doppler_rmse_hz"] for snr in snrs if str(int(snr)) in metrics]
        x = [snr for snr in snrs if str(int(snr)) in metrics]
        if y:
            axis.plot(x, y, marker="o", linewidth=1.1, label=name)
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Doppler RMSE (Hz)")
    axis.set_title("D1-E002 independent blind Doppler RMSE")
    axis.grid(True, which="both", alpha=0.25)
    axis.set_yscale("log")
    axis.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(blind_dir / "rmse_snr.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    preferred = [
        "S0/seed_1",
        "S0_D/seed_1",
        "APX15_teacher",
        "nonlearning_iq_energy_detector",
    ]
    preferred.extend(name for name in summaries if name.split("/")[0] in {"S1", "S2", "S3"} and name.endswith("seed_1"))
    for name in preferred:
        prediction = predictions.get(name)
        if prediction is None or not np.isfinite(prediction["score"]).all():
            continue
        labels = np.asarray(prediction["truth_presence"], dtype=bool)
        scores = np.asarray(prediction["score"], dtype=np.float64)
        order = np.argsort(scores, kind="mergesort")[::-1]
        sorted_labels = labels[order]
        tpr = np.cumsum(sorted_labels) / max(int(labels.sum()), 1)
        fpr = np.cumsum(~sorted_labels) / max(int((~labels).sum()), 1)
        axis.plot(fpr, tpr, label=name)
    axis.set_xlabel("False alarm probability")
    axis.set_ylabel("Probability of detection")
    axis.set_title("D1-E002 blind ROC")
    axis.set_xlim(0.0, 0.25)
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(blind_dir / "detection_roc.png", dpi=160)
    plt.close(fig)


def run_d1_e002_blind(
    root: str | Path,
    config_path: str | Path,
    *,
    batch_size: int = 64,
) -> Path:
    """Run the one-time, frozen-seed E002 blind evaluation after calibration commit."""
    import torch

    from ..simulation.d0 import generate_d0_batch

    root_path = Path(root).resolve()
    config_file = Path(config_path).resolve()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config = _load_d1_e002_config(root_path, config_file)
    device = _require_cuda(str(config["training"]["device"]))
    formal_dir = root_path / "experiments" / "D1-E002" / "formal"
    model_manifest, _ = _clean_model_manifest(root_path, formal_dir, config)
    calibration_dir = root_path / "experiments" / "D1-E002" / "calibration"
    calibration, calibration_commit = _require_clean_calibration(root_path, calibration_dir)
    if calibration.get("model_manifest_commit") is None:
        raise RuntimeError("calibration lacks its frozen model manifest reference")
    model_manifest_relative = (formal_dir / "model_manifest.json").resolve().relative_to(root_path).as_posix()
    frozen_manifest = subprocess.run(
        ["git", "-C", str(root_path), "show", f"{calibration['model_manifest_commit']}:{model_manifest_relative}"],
        check=True,
        capture_output=True,
    ).stdout
    if frozen_manifest != (root_path / model_manifest_relative).read_bytes():
        raise RuntimeError("model manifest changed after detection calibration")
    if calibration.get("status") != "COMPLETED":
        raise RuntimeError("completed calibration is required before blind generation")
    blind_dir = root_path / "experiments" / "D1-E002" / "blind"
    if blind_dir.exists():
        raise FileExistsError(f"blind output already exists; preserving it: {blind_dir}")
    blind_dir.mkdir(parents=True)
    shutil.copyfile(config_file, blind_dir / "config_used.yaml")
    formal_history = formal_dir / "training_history.csv"
    if formal_history.is_file():
        shutil.copyfile(formal_history, blind_dir / "training_history.csv")
    started = time.perf_counter()
    run_manifest = {
        "experiment_id": "D1-E002",
        "phase": "blind",
        "status": "RUNNING",
        "code_commit": model_manifest["code_commit"],
        "model_manifest_commit": calibration["model_manifest_commit"],
        "calibration_commit": calibration_commit,
        "blind_data_generated": False,
        "batch_size": batch_size,
        "blind_used_for_model_selection": False,
    }
    _write_json(blind_dir / "run_manifest.json", run_manifest)
    partial: dict[str, Any] = {
        "experiment_id": "D1-E002",
        "phase": "blind",
        "status": "RUNNING",
        "methods": {},
        "gates": {},
    }
    _write_json(blind_dir / "metrics_partial.json", partial)
    predictions: dict[str, dict[str, np.ndarray]] = {}
    prediction_dir = blind_dir / "predictions"
    prediction_dir.mkdir()
    try:
        blind_arrays = _open_or_generate_evaluation_cache(
            root=root_path,
            config=config,
            d0_config=_d0_config(root_path),
            kind="blind",
        )
        run_manifest["blind_data_generated"] = True
        run_manifest["blind_cache"] = "experiments/D1-E002/data/blind_cache"
        _write_json(blind_dir / "run_manifest.json", run_manifest)

        for entry in model_manifest["models"]:
            name = entry["id"]
            prediction, matched = _predict_frozen_model_cache(
                root_path, entry, config, blind_arrays, device, batch_size
            )
            predictions[name] = prediction
            _write_prediction(prediction_dir / f"{name.replace('/', '__')}.npz", prediction)
            if entry["variant"] == "S2" and isinstance(matched, dict):
                method_name = f"S2_matched_lag_one/{entry['seed_id']}"
                predictions[method_name] = matched
                _write_prediction(prediction_dir / f"{method_name.replace('/', '__')}.npz", matched)
            elif entry["variant"] == "S3" and isinstance(matched, dict):
                for key, label in (("three_point", "S3_matched_three_point"), ("five_bin_centroid", "S3_matched_centroid")):
                    method_name = f"{label}/{entry['seed_id']}"
                    predictions[method_name] = matched[key]
                    _write_prediction(prediction_dir / f"{method_name.replace('/', '__')}.npz", matched[key])
            print(f"[blind] completed {name}", flush=True)

        teacher = _predict_teacher_cache(root_path, _d0_config(root_path), blind_arrays, batch_size)
        predictions["APX15_teacher"] = teacher
        _write_prediction(prediction_dir / "APX15_teacher.npz", teacher)
        classical, energy = _predict_classical_cache(blind_arrays, config, batch_size)
        for name, prediction in classical.items():
            predictions[name] = prediction
            _write_prediction(prediction_dir / f"{name}.npz", prediction)
        predictions["nonlearning_iq_energy_detector"] = energy
        _write_prediction(prediction_dir / "nonlearning_iq_energy_detector.npz", energy)
        profile = _profile_ml_prediction(root_path, blind_arrays, config)
        predictions["profile_ml"] = profile
        _write_prediction(prediction_dir / "profile_ml.npz", profile)

        summaries = {
            name: _summarize_prediction(name, prediction, calibration, config)
            for name, prediction in predictions.items()
        }
        gates = {
            candidate: _candidate_gate(
                candidate,
                prediction_by_name=predictions,
                summaries=summaries,
                calibration=calibration,
                config=config,
            )
            for candidate in model_manifest["candidate_order"]
        }
        pure_pass = any(gates[name]["passed"] for name in ("S1", "S2") if name in gates)
        hybrid_pass = bool(gates.get("S3", {}).get("passed", False))
        decision = "GO" if pure_pass else "STOP / REDIRECT"
        if not pure_pass and hybrid_pass:
            decision += "（仅 FFT 辅助混合方法通过；纯 IQ 结构停止）"
        physical = {
            entry["id"]: _predict_physical_consistency(
                root_path, entry, config, device, batch_size
            )
            for entry in model_manifest["models"]
            if entry["kind"] == "e002_candidate"
        }
        partial.update(
            {
                "status": "COMPLETED",
                "methods": summaries,
                "gates": gates,
                "stage_decision": decision,
                "pure_iq_candidate_passed": pure_pass,
                "fft_hybrid_candidate_passed": hybrid_pass,
                "physical_consistency": physical,
                "sample_counts": {
                    "blind_h0": int(np.count_nonzero(~np.asarray(blind_arrays["truth_presence"]))),
                    "blind_h1": int(np.count_nonzero(np.asarray(blind_arrays["truth_presence"]))),
                    "profile_ml_h1_per_snr": int(config["baselines"]["profile_ml"]["h1_examples_per_snr"]),
                },
                "environment": _evaluation_environment(
                    root_path, device, batch_size, time.perf_counter() - started
                ),
                "calibration_thresholds": calibration["methods"],
                "blind_data_used_for_model_selection": False,
            }
        )
        _write_json(blind_dir / "metrics.json", partial)
        _write_json(blind_dir / "environment.json", partial["environment"])
        (blind_dir / "metrics_partial.json").unlink(missing_ok=True)
        _write_result_csv(blind_dir / "result.csv", summaries)
        _plot_blind_results(
            blind_dir,
            predictions,
            summaries,
            [float(value) for value in config["data"]["snr_db"]],
        )
        analysis = [
            "# D1-E002 独立盲测阶段结果",
            "",
            f"**阶段决策：{decision}。**",
            "",
            "本结果来自冻结的单目标、三通道复 AWGN 仿真；盲测仅用于最终评价，没有用于模型或阈值选择。频率误差按 PRF=1300 Hz 周期回绕。所有学生与 APX15 使用独立 H0 校准阈值；经典频率估计器共用预注册非相干能量检测分数，profile-ML 只报告固定 H1 子集参数误差。",
            "",
            "## 候选阶段门",
            "",
        ]
        for name, gate in gates.items():
            analysis.append(
                f"- **{name}**：{'通过' if gate['passed'] else '未通过'}；主 RMSE 相对 {gate['doppler_comparator']} 改善 {100*gate['relative_gain']:.2f}%；Doppler Δ 95% CI {gate['doppler_delta_paired_hierarchical_bootstrap_95_ci_hz']} Hz；检测与距离门分别为 {gate['gates']['low_snr_detection_noninferiority']} / {gate['gates']['high_snr_range_noninferiority']}。"
            )
        analysis.extend(
            [
                "",
                "## 结论限制",
                "",
                "D1-E001 的 G1/G2/G3 历史阶段结论没有改写。profile-ML 是 D1 已冻结的有限网格数值参考，不等同于连续全局 ML。盲测检测样本不支持 1e-6 虚警率声明；本阶段不验证杂波、多目标、通道误差、实时性或真实回波泛化。各方法完整逐 SNR 表格见 `result.csv`，模型与物理一致性细节见 `metrics.json`，RMSE 与 ROC 曲线见 PNG。",
            ]
        )
        (blind_dir / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")
        run_manifest["status"] = "COMPLETED"
        run_manifest["stage_decision"] = decision
        run_manifest["environment"] = partial["environment"]
        _write_json(blind_dir / "run_manifest.json", run_manifest)
        return blind_dir
    except BaseException as error:
        partial["status"] = "FAILED"
        partial["failure"] = f"{type(error).__name__}: {error}"
        _write_json(blind_dir / "metrics_partial.json", partial)
        run_manifest["status"] = "FAILED"
        run_manifest["failure"] = partial["failure"]
        _write_json(blind_dir / "run_manifest.json", run_manifest)
        raise
