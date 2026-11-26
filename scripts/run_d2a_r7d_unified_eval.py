#!/usr/bin/env python3
"""Re-evaluate frozen historical students and classical references on R7-D data."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from radardistill.d2a_baselines import run_b0, run_b1
from radardistill.evaluation.d1_evaluation import (
    estimate_fft_baselines,
    estimate_phase_difference_multichannel,
    estimate_profile_ml_reference,
)
from radardistill.evaluation.d2a_r7d import (
    binary_auc,
    bootstrap_error_intervals,
    build_historical_student,
    calibrate_score_threshold,
    decode_task_outputs,
    extract_checkpoint_state,
    make_no_clutter_pair,
    parameter_counts,
    summarize_detection_scores,
    summarize_parameter_errors,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch

R7D = ROOT / "experiments/D2-A-R7/R7-D"
CONFIG_PATH = R7D / "config.yaml"
INVENTORY_PATH = R7D / "checkpoint_inventory.csv"
SCENE_PLAN_PATH = R7D / "scene_plan.csv"
METRICS_PATH = R7D / "metrics.json"
OWNER_ID = "D2-A-R7-D"
CHUNK_SIZE = 64


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_configs() -> tuple[dict[str, Any], dict[str, Any]]:
    r7 = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    physics_path = ROOT / r7["physics"]["source_config"]
    physics = yaml.safe_load(physics_path.read_text(encoding="utf-8"))
    assert int(physics["radar"]["channel_count"]) == 3
    assert int(physics["radar"]["pulse_count"]) == 130
    assert int(physics["radar"]["range_bin_count"]) == 32
    assert float(physics["radar"]["prf_hz"]) == 1300.0
    return r7, physics


def _inventory() -> list[dict[str, str]]:
    with INVENTORY_PATH.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_inventory(rows: list[dict[str, str]]) -> None:
    seen: set[str] = set()
    for row in rows:
        path_text = row["checkpoint_path"]
        path = (ROOT / path_text).resolve()
        if ROOT.resolve() not in path.parents:
            raise ValueError(f"checkpoint path escapes repository: {path_text}")
        if path_text in seen:
            raise ValueError(f"duplicate checkpoint inventory path: {path_text}")
        seen.add(path_text)
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = _sha256(path)
        if digest != row["checkpoint_sha256"]:
            raise ValueError(f"checkpoint changed after R7-D freeze: {path_text}")


def preflight_models(r7: dict[str, Any], physics: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R7-D; CPU fallback is forbidden")
    device = torch.device(str(r7["runtime"]["device"]))
    if device.type != "cuda":
        raise ValueError("R7-D runtime device must be CUDA")
    torch.cuda.set_device(0 if device.index is None else int(device.index))
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    output: dict[str, Any] = {
        "status": "IN_PROGRESS",
        "protocol": "experiments/D2-A-R7/R7-D/protocol.md",
        "config": "experiments/D2-A-R7/R7-D/config.yaml",
        "checkpoint_inventory": "experiments/D2-A-R7/R7-D/checkpoint_inventory.csv",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": torch.cuda.get_device_name(device_index),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "free_vram_mib_start": round(free_bytes / (1024**2), 2),
        "total_vram_mib": round(total_bytes / (1024**2), 2),
        "checkpoint_count": len(rows),
        "models": [],
    }
    for index, row in enumerate(rows):
        item: dict[str, Any] = {
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "task_scope": row["task_scope"],
        }
        model = None
        try:
            payload = torch.load(ROOT / row["checkpoint_path"], map_location="cpu", weights_only=True)
            model, label = build_historical_student(row["checkpoint_path"], payload, physics)
            model.load_state_dict(extract_checkpoint_state(payload), strict=True)
            item.update({"model_label": label, **parameter_counts(model)})
            model = model.to(device).eval()
            with torch.inference_mode():
                outputs = _call_model(model, torch.zeros((1, 6, 130, 32), dtype=torch.float32, device=device))
            if row["task_scope"] != "parameter_only" and "presence_logit" not in outputs:
                raise ValueError("detection-capable checkpoint has no presence_logit")
            if row["task_scope"] != "detection_only":
                decoded = decode_task_outputs(outputs, 1300.0)
                if any(not np.isfinite(value).all() for value in decoded.values()):
                    raise FloatingPointError("student decoder produced non-finite values")
            item["status"] = "PASS"
            item["output_keys"] = sorted(outputs)
        except Exception as exc:
            item["status"] = "FAIL"
            item["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if model is not None:
                del model
            torch.cuda.empty_cache()
        output["models"].append(item)
        if (index + 1) % 8 == 0 or index + 1 == len(rows):
            print(f"  checkpoint adapters CUDA preflight {index + 1}/{len(rows)}")
    torch.cuda.synchronize(device)
    output["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 3)
    output["passed"] = sum(item["status"] == "PASS" for item in output["models"])
    output["failed"] = len(rows) - output["passed"]
    output["status"] = "PASS" if output["failed"] == 0 else "PARTIAL"
    output_path = ROOT / r7["outputs"]["preflight"]
    _json_write(output_path, output)
    return output


def _scene_seed(
    seed_start: int,
    split_index: int,
    snr_index: int,
    chunk_index: int,
    split_stride: int,
    snr_stride: int,
) -> int:
    return int(seed_start + split_index * split_stride + snr_index * snr_stride + chunk_index)


def _generator_config(physics: dict[str, Any], r7: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(physics)
    config["clutter"]["scatterer_count"] = 1  # generated component is discarded before caching
    config["target"]["range_cell_interval"] = list(r7["physics"]["target_range_cell_interval"])
    config["target"]["azimuth_interval_deg"] = list(r7["physics"]["target_azimuth_interval_deg"])
    config["target"]["reference_amplitude_db_interval"] = list(r7["physics"]["reference_amplitude_db_interval"])
    return config


def _write_scene_plan(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    fields = ("split", "snr_db", "scene_index", "scene_id", "generation_seed", "h0_h1_pairing")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(r7: dict[str, Any], physics: dict[str, Any], *, force: bool = False) -> None:
    cache_dir = ROOT / r7["outputs"]["temporary_cache_root"]
    owner_path = cache_dir / "owner.json"
    if cache_dir.exists():
        if force:
            raise ValueError("--force is not supported; remove only a verified R7-D-owned cache explicitly")
        if not owner_path.is_file():
            raise FileExistsError(f"refusing to overwrite cache directory without R7-D owner manifest {cache_dir}")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        if owner.get("owner_id") == OWNER_ID and owner.get("status") == "complete":
            print(f"R7-D cache already complete: {cache_dir.relative_to(ROOT)}")
            return
        allowed_files = {"l0_h1.npy", "l1_h0_calibration.npy", "l1_h0_verification.npy", "l1_h1.npy", "truth_arrays.npz"}
        owned_files = owner.get("files", [])
        if owner.get("owner_id") != OWNER_ID or owner.get("status") != "preparing" or not set(owned_files) <= allowed_files:
            raise FileExistsError(f"refusing to overwrite non-complete/unowned cache directory {cache_dir}")
        for name in owned_files:
            path = cache_dir / name
            if path.is_file():
                path.unlink()
        owner_path.unlink()
        cache_dir.rmdir()
        print("Removed only the incomplete R7-D-owned cache from a failed preparation.")

    snrs = [float(value) for value in r7["physics"]["snr_db"]]
    count = int(r7["splits"]["l1_h1_per_snr"])
    h0_count = int(r7["splits"]["l1_h0_calibration_per_snr"])
    verify_count = int(r7["splits"]["l1_h0_verification_per_snr"])
    l0_count = int(r7["splits"]["l0_h1_scenes"])
    channels, pulses, ranges = 3, 130, 32
    estimate_bytes = (l0_count + len(snrs) * (h0_count + verify_count + count)) * channels * pulses * ranges * np.dtype(np.complex64).itemsize
    free_bytes = shutil.disk_usage(ROOT).free
    if free_bytes < int(estimate_bytes * 1.20):
        raise RuntimeError(f"insufficient free disk for R7-D cache: need {estimate_bytes * 1.2:.0f}, have {free_bytes}")

    cache_dir.mkdir(parents=True)
    file_names = ["l0_h1.npy", "l1_h0_calibration.npy", "l1_h0_verification.npy", "l1_h1.npy", "truth_arrays.npz"]
    owner = {
        "owner_id": OWNER_ID,
        "status": "preparing",
        "files": file_names,
        "created_by": "scripts/run_d2a_r7d_unified_eval.py",
        "expected_iq_bytes": int(estimate_bytes),
    }
    _json_write(owner_path, owner)
    shape_l1 = (len(snrs), count, channels, pulses, ranges)
    l0_path = cache_dir / "l0_h1.npy"
    l0_iq = np.lib.format.open_memmap(l0_path, mode="w+", dtype=np.complex64, shape=(l0_count, channels, pulses, ranges))
    l1_arrays: dict[str, np.memmap] = {}
    for split_name, scene_count in (
        ("l1_h0_calibration", h0_count),
        ("l1_h0_verification", verify_count),
        ("l1_h1", count),
    ):
        l1_arrays[split_name] = np.lib.format.open_memmap(
            cache_dir / f"{split_name}.npy", mode="w+", dtype=np.complex64,
            shape=(len(snrs), scene_count, channels, pulses, ranges),
        )

    l0_range = np.empty(l0_count, dtype=np.float64)
    l0_doppler = np.empty(l0_count, dtype=np.float64)
    h1_range = np.empty((len(snrs), count), dtype=np.float64)
    h1_doppler = np.empty_like(h1_range)
    plan_rows: list[dict[str, Any]] = []
    generator_config = _generator_config(physics, r7)
    seed_start = int(r7["splits"]["seed_block_start"])
    split_stride = int(r7["splits"]["seed_stride_by_split"])
    snr_stride = int(r7["splits"]["seed_stride_by_snr_index"])
    velocity_low, velocity_high = (float(x) for x in r7["physics"]["target_radial_velocity_mps_uniform"])

    def generate_split(split_name: str, split_index: int, snr_db: float, snr_index: int, scene_count: int, target_enabled: bool) -> None:
        offset = 0
        iq_out = l0_iq if split_name == "l0_h1" else l1_arrays[split_name][snr_index]
        while offset < scene_count:
            size = min(CHUNK_SIZE, scene_count - offset)
            chunk_index = offset // CHUNK_SIZE
            seed = _scene_seed(seed_start, split_index, snr_index, chunk_index, split_stride, snr_stride)
            velocity_rng = np.random.default_rng(seed + 9_000_000)
            velocity = velocity_rng.uniform(velocity_low, velocity_high, size=size)
            batch = generate_d2a_paired_batch(
                size,
                seed,
                float(snr_db),
                0.0,
                generator_config,
                scenario_id="separated",
                target_radial_velocity_mps=velocity,
            )
            if split_name == "l0_h1":
                signal = np.asarray(batch.target_signal_iq, dtype=np.complex64)
                iq_out[offset : offset + size] = signal
                l0_range[offset : offset + size] = batch.truth_target_range_cell
                l0_doppler[offset : offset + size] = batch.truth_target_doppler_hz
            else:
                h0, h1 = make_no_clutter_pair(batch)
                expected = h1 if target_enabled else h0
                iq_out[offset : offset + size] = expected
                if target_enabled:
                    h1_range[snr_index, offset : offset + size] = batch.truth_target_range_cell
                    h1_doppler[snr_index, offset : offset + size] = batch.truth_target_doppler_hz
            ids = [str(value) for value in batch.base_scene_id]
            plan_rows.extend(
                {
                    "split": split_name,
                    "snr_db": "NO_NOISE" if split_name == "l0_h1" else float(snr_db),
                    "scene_index": offset + i,
                    "scene_id": scene_id,
                    "generation_seed": seed,
                    "h0_h1_pairing": "counterfactual generated; this split retains only its registered H0 or H1 side" if split_name.startswith("l1_") else "H1 target-only; H0 zero",
                }
                for i, scene_id in enumerate(ids)
            )
            offset += size
        iq_out.flush()

    print(f"Preparing R7-D IQ caches ({estimate_bytes / (1024**3):.2f} GiB expected)")
    generate_split("l0_h1", 3, 0.0, 0, l0_count, True)
    for snr_index, snr_db in enumerate(snrs):
        generate_split("l1_h0_calibration", 0, snr_db, snr_index, h0_count, False)
        generate_split("l1_h0_verification", 1, snr_db, snr_index, verify_count, False)
        generate_split("l1_h1", 2, snr_db, snr_index, count, True)
        if snr_index == 0:
            plan_rows.extend(
                {
                    "split": "l0_h0",
                    "snr_db": "NO_NOISE",
                    "scene_index": i,
                    "scene_id": f"r7d-l0-h0-{i:05d}",
                    "generation_seed": "deterministic-zero",
                    "h0_h1_pairing": "exact-zero IQ",
                }
                for i in range(l0_count)
            )
        if (snr_index + 1) % 2 == 0 or snr_index + 1 == len(snrs):
            print(f"  generated L1 conditions {snr_index + 1}/{len(snrs)}")

    np.savez_compressed(
        cache_dir / "truth_arrays.npz",
        l0_range=l0_range,
        l0_doppler=l0_doppler,
        l1_h1_range=h1_range,
        l1_h1_doppler=h1_doppler,
        snr_db=np.asarray(snrs, dtype=np.float64),
    )
    _write_scene_plan(plan_rows, SCENE_PLAN_PATH)
    owner["status"] = "complete"
    owner["files"] = file_names
    owner["scene_plan"] = str(SCENE_PLAN_PATH.relative_to(ROOT))
    owner["scene_count"] = len(plan_rows)
    owner["bytes_on_disk"] = int(sum((cache_dir / name).stat().st_size for name in file_names))
    _json_write(owner_path, owner)
    del l0_iq, l1_arrays
    print(f"R7-D cache ready: {owner['scene_count']} scene-plan rows, {owner['bytes_on_disk'] / (1024**3):.2f} GiB")


def _complex_to_ri(batch: np.ndarray, device: torch.device) -> torch.Tensor:
    values = np.asarray(batch, dtype=np.complex64)
    real_imag = np.empty((len(values), 6, 130, 32), dtype=np.float32)
    real_imag[:, :3] = values.real
    real_imag[:, 3:] = values.imag
    return torch.from_numpy(real_imag).to(device=device, non_blocking=False)


def _call_model(model: torch.nn.Module, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
    parameters = inspect.signature(model.forward).parameters
    if "compute_spectrum" in parameters:
        return model(iq_ri, compute_spectrum=False)
    return model(iq_ri)


def _predict_student(
    model: torch.nn.Module,
    iq: np.ndarray,
    *,
    device: torch.device,
    has_detection: bool,
    has_parameters: bool,
    batch_size: int,
) -> dict[str, np.ndarray | None]:
    def run(current_batch: int) -> dict[str, np.ndarray | None]:
        n = len(iq)
        result: dict[str, np.ndarray | None] = {
            "score": np.empty(n, dtype=np.float64) if has_detection else None,
            "range_cell": np.empty(n, dtype=np.float64) if has_parameters else None,
            "doppler_hz": np.empty(n, dtype=np.float64) if has_parameters else None,
        }
        for start in range(0, n, current_batch):
            stop = min(start + current_batch, n)
            input_tensor = _complex_to_ri(iq[start:stop], device)
            with torch.inference_mode():
                outputs = _call_model(model, input_tensor)
                if has_detection:
                    logits = outputs["presence_logit"].detach().float().reshape(-1)
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError("non-finite presence logits")
                    result["score"][start:stop] = torch.sigmoid(logits).cpu().numpy()
                if has_parameters:
                    decoded = decode_task_outputs(outputs, prf_hz=1300.0)
                    if not np.isfinite(decoded["range_cell"]).all() or not np.isfinite(decoded["doppler_hz"]).all():
                        raise FloatingPointError("non-finite parameter output")
                    result["range_cell"][start:stop] = decoded["range_cell"]
                    result["doppler_hz"][start:stop] = decoded["doppler_hz"]
        return result

    for current_batch in dict.fromkeys((batch_size, 8, 4)):
        try:
            return run(current_batch)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda" or current_batch == 4:
                raise
            torch.cuda.empty_cache()
    raise RuntimeError("unreachable batch-size retry state")


def _prediction_summary(
    *,
    calibration_scores: np.ndarray | None,
    verification_scores: np.ndarray | None,
    h1_scores: np.ndarray | None,
    range_prediction: np.ndarray | None,
    doppler_prediction: np.ndarray | None,
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    target_pfa: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    detection: dict[str, Any]
    detection_mask: np.ndarray | None = None
    if calibration_scores is None or verification_scores is None or h1_scores is None:
        auc = binary_auc(verification_scores, h1_scores) if verification_scores is not None and h1_scores is not None else None
        detection = {"status": "UNTHRESHOLDED", "auc": auc, "threshold": None,
                     "verification_pfa": None, "h1_pd": None,
                     "false_alarm_unit": "Pfa_CPI" if h1_scores is not None else "NOT_APPLICABLE",
                     "pfa_cell": "NOT_APPLICABLE_SINGLE_SCALAR_OUTPUT" if h1_scores is not None else "NOT_APPLICABLE"}
    else:
        threshold = calibrate_score_threshold(calibration_scores, target_pfa)
        detection = summarize_detection_scores(verification_scores, h1_scores, threshold)
        detection["status"] = "EVALUATED"
        detection["auc"] = binary_auc(verification_scores, h1_scores)
        detection["target_pfa_cpi"] = float(target_pfa)
        detection["false_alarm_unit"] = "Pfa_CPI"
        detection["pfa_cell"] = "NOT_APPLICABLE_SINGLE_SCALAR_OUTPUT"
        detection_mask = h1_scores > threshold

    parameters: dict[str, Any] | None = None
    if range_prediction is not None and doppler_prediction is not None:
        mask = detection_mask if detection_mask is not None else np.zeros(len(truth_range), dtype=np.bool_)
        parameters = summarize_parameter_errors(
            truth_range,
            range_prediction,
            truth_doppler,
            doppler_prediction,
            prf_hz=1300.0,
            detection_mask=mask,
            range_gate_cells=1.0,
            doppler_gate_hz=20.0,
        )
        for subset_name in ("all_h1", "detected_h1"):
            if subset_name == "detected_h1" and detection_mask is None:
                parameters[subset_name] = None
                continue
            subset = np.ones(len(truth_range), dtype=np.bool_) if subset_name == "all_h1" else detection_mask
            range_error = range_prediction[subset] - truth_range[subset]
            doppler_error = np.remainder(
                doppler_prediction[subset] - truth_doppler[subset] + 650.0, 1300.0
            ) - 650.0
            parameters[subset_name]["range_bootstrap_ci95"] = bootstrap_error_intervals(
                range_error, seed=bootstrap_seed
            ) if range_error.size else {"rmse_ci95": None, "mae_ci95": None}
            parameters[subset_name]["doppler_bootstrap_ci95"] = bootstrap_error_intervals(
                doppler_error, seed=bootstrap_seed + 1
            ) if doppler_error.size else {"rmse_ci95": None, "mae_ci95": None}
        if detection_mask is None:
            parameters["joint_success"] = None
            parameters["joint_success_count"] = None
            parameters["joint_success_ci95"] = None
    return {"detection": detection, "parameters": parameters}


def _load_cache(r7: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_dir = ROOT / r7["outputs"]["temporary_cache_root"]
    owner_path = cache_dir / "owner.json"
    if not owner_path.is_file():
        raise FileNotFoundError(f"R7-D owner manifest missing: {owner_path}")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner.get("owner_id") != OWNER_ID or owner.get("status") != "complete":
        raise ValueError("R7-D cache owner/status mismatch")
    arrays = {
        name: np.load(cache_dir / f"{name}.npy", mmap_mode="r")
        for name in ("l0_h1", "l1_h0_calibration", "l1_h0_verification", "l1_h1")
    }
    truth_npz = np.load(cache_dir / "truth_arrays.npz")
    truths = {name: truth_npz[name] for name in truth_npz.files}
    arrays.update({f"truth_{name}": value for name, value in truths.items()})
    return arrays, owner


def evaluate_students(r7: dict[str, Any], physics: dict[str, Any], rows: list[dict[str, str]], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R7-D historical student inference; CPU fallback is forbidden")
    device = torch.device(str(r7["runtime"]["device"]))
    if device.type != "cuda":
        raise ValueError("R7-D runtime device must be CUDA")
    device_index = 0 if device.index is None else int(device.index)
    torch.cuda.set_device(device_index)
    device_name = torch.cuda.get_device_name(device_index)
    metrics = {
        "experiment": r7["experiment"]["id"],
        "status": "IN_PROGRESS",
        "protocol": "experiments/D2-A-R7/R7-D/protocol.md",
        "config": "experiments/D2-A-R7/R7-D/config.yaml",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device": device_name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "checkpoint_count": len(rows),
        "models": [],
        "classical": [],
    }
    existing = []
    if METRICS_PATH.is_file():
        prior = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        if prior.get("experiment") == metrics["experiment"] and prior.get("source_commit") == metrics["source_commit"]:
            existing = prior.get("models", [])
    by_path = {entry.get("checkpoint_path"): entry for entry in existing if entry.get("status") in {"COMPLETE", "LOAD_FAILED", "FORWARD_FAILED"}}
    cache = ROOT / r7["outputs"]["temporary_cache_root"]
    truth = arrays
    target_pfa = float(r7["splits"]["target_pfa_cpi"])
    task_results: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device_index)
    for index, row in enumerate(rows):
        path_text = row["checkpoint_path"]
        if path_text in by_path:
            task_results.append(by_path[path_text])
            print(f"[{index + 1}/{len(rows)}] reuse completed result {path_text}")
            continue
        checkpoint_path = ROOT / path_text
        print(f"[{index + 1}/{len(rows)}] evaluating {path_text}")
        started = time.perf_counter()
        entry: dict[str, Any] = {
            "method_id": row["method_id"],
            "experiment_id": row["experiment_id"],
            "checkpoint_path": path_text,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_update": row["checkpoint_update"] or "NOT_RECORDED",
            "seed_id": row["seed_id"],
            "arm": row["arm"],
            "task_scope": row["task_scope"],
        }
        model = None
        model_loaded = False
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model, label = build_historical_student(path_text, payload, physics)
            state = extract_checkpoint_state(payload)
            model.load_state_dict(state, strict=True)
            model_loaded = True
            counts = parameter_counts(model)
            entry.update({"model_label": label, **counts})
            model = model.to(device).eval()
            probe = torch.zeros((1, 6, 130, 32), device=device, dtype=torch.float32)
            with torch.inference_mode():
                probe_output = _call_model(model, probe)
            if "presence_logit" not in probe_output:
                raise ValueError("forward output has no presence_logit")
            if row["task_scope"] != "detection_only":
                decode_task_outputs(probe_output, 1300.0)
            has_detection = row["task_scope"] != "parameter_only"
            has_parameters = row["task_scope"] != "detection_only"
            torch.cuda.synchronize(device)
            zero_iq = np.zeros((1, 3, 130, 32), dtype=np.complex64)
            zero_pred = _predict_student(
                model, zero_iq, device=device, has_detection=has_detection,
                has_parameters=has_parameters, batch_size=int(r7["runtime"]["inference_batch_size"]),
            )
            l0_h0_scores = np.repeat(zero_pred["score"][0], int(r7["splits"]["l0_h0_scenes"])) if has_detection else None
            l0_pred = _predict_student(
                model, arrays["l0_h1"], device=device, has_detection=has_detection,
                has_parameters=has_parameters, batch_size=int(r7["runtime"]["inference_batch_size"]),
            )
            l0 = _prediction_summary(
                calibration_scores=None, verification_scores=l0_h0_scores,
                h1_scores=l0_pred["score"], range_prediction=l0_pred["range_cell"],
                doppler_prediction=l0_pred["doppler_hz"], truth_range=arrays["truth_l0_range"],
                truth_doppler=arrays["truth_l0_doppler"], target_pfa=target_pfa,
                bootstrap_seed=int(r7["splits"]["bootstrap_seed"]) + index * 100,
            )
            l0["detection"]["status"] = "UNTHRESHOLDED_ZERO_H0_DIAGNOSTIC" if has_detection else "NOT_APPLICABLE"
            per_snr: list[dict[str, Any]] = []
            for snr_index, snr_db in enumerate(r7["physics"]["snr_db"]):
                cal = _predict_student(
                    model, arrays["l1_h0_calibration"][snr_index], device=device,
                    has_detection=has_detection, has_parameters=False,
                    batch_size=int(r7["runtime"]["inference_batch_size"]),
                )
                verify = _predict_student(
                    model, arrays["l1_h0_verification"][snr_index], device=device,
                    has_detection=has_detection, has_parameters=False,
                    batch_size=int(r7["runtime"]["inference_batch_size"]),
                )
                h1_pred = _predict_student(
                    model, arrays["l1_h1"][snr_index], device=device,
                    has_detection=has_detection, has_parameters=has_parameters,
                    batch_size=int(r7["runtime"]["inference_batch_size"]),
                )
                score_summary = _prediction_summary(
                    calibration_scores=cal["score"], verification_scores=verify["score"],
                    h1_scores=h1_pred["score"], range_prediction=h1_pred["range_cell"],
                    doppler_prediction=h1_pred["doppler_hz"],
                    truth_range=arrays["truth_l1_h1_range"][snr_index],
                    truth_doppler=arrays["truth_l1_h1_doppler"][snr_index],
                    target_pfa=target_pfa,
                    bootstrap_seed=int(r7["splits"]["bootstrap_seed"]) + index * 100 + snr_index * 2,
                )
                score_summary.update({
                    "dataset": "L1_AWGN_NO_CLUTTER",
                    "snr_db": float(snr_db),
                    "h0_calibration_n": len(arrays["l1_h0_calibration"][snr_index]),
                    "h0_verification_n": len(arrays["l1_h0_verification"][snr_index]),
                    "h1_n": len(arrays["l1_h1"][snr_index]),
                    "threshold_scope": "per-method, per-SNR calibration; not a global unknown-SNR threshold",
                })
                per_snr.append(score_summary)
            torch.cuda.synchronize(device)
            entry.update({
                "status": "COMPLETE",
                "datasets": {"L0_TARGET_ONLY": l0, "L1_AWGN_NO_CLUTTER": per_snr},
                "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 3),
                "wall_seconds": round(time.perf_counter() - started, 3),
            })
            del model
            torch.cuda.empty_cache()
        except (RuntimeError, ValueError, KeyError, TypeError, FloatingPointError) as exc:
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                entry["status"] = "FORWARD_FAILED"
            else:
                entry["status"] = "FORWARD_FAILED" if model_loaded else "LOAD_FAILED"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            entry["wall_seconds"] = round(time.perf_counter() - started, 3)
            if model is not None:
                del model
            torch.cuda.empty_cache()
        task_results.append(entry)
        metrics["models"] = task_results
        metrics["models_completed"] = sum(item.get("status") == "COMPLETE" for item in task_results)
        metrics["models_failed"] = sum(item.get("status") in {"LOAD_FAILED", "FORWARD_FAILED"} for item in task_results)
        metrics["gpu_peak_allocated_mib"] = round(torch.cuda.max_memory_allocated(device_index) / (1024**2), 3)
        _json_write(METRICS_PATH, metrics)
        print(f"  {entry['status']} {entry.get('model_label', path_text)} ({entry['wall_seconds']} s)")
    metrics["models"] = task_results
    return metrics


def _classical_batch(iq: np.ndarray, physics: dict[str, Any]) -> dict[str, Any]:
    b0 = run_b0(iq, 1300.0)
    b1 = run_b1(iq, 1300.0)
    fft = estimate_fft_baselines(iq, 1300.0)
    phase = estimate_phase_difference_multichannel(iq, 1300.0)
    return {"b0": b0, "b1": b1, "fft": fft, "phase": phase}


def _classical_values(iq: np.ndarray, physics: dict[str, Any], *, with_parameters: bool) -> dict[str, dict[str, np.ndarray]]:
    accum: dict[str, dict[str, list[np.ndarray]]] = {
        name: {key: [] for key in ("score", "range_cell", "doppler_hz")}
        for name in ("B0_single_channel_FFT_peak", "B1_three_channel_noncoherent_FFT_peak",
                     "B1_detection_plus_native_FFT_parameters", "B1_detection_plus_log_parabolic_parameters",
                     "B1_detection_plus_4x_zero_pad_parameters", "lag_one_phase_difference_estimator")
    }
    for start in range(0, len(iq), CHUNK_SIZE):
        chunk = np.asarray(iq[start : start + CHUNK_SIZE])
        result = _classical_batch(chunk, physics)
        b0, b1 = result["b0"], result["b1"]
        accum["B0_single_channel_FFT_peak"]["score"].append(np.asarray(b0.score))
        accum["B1_three_channel_noncoherent_FFT_peak"]["score"].append(np.asarray(b1.score))
        if not with_parameters:
            continue
        accum["B0_single_channel_FFT_peak"]["range_cell"].append(np.asarray(b0.range_cell))
        accum["B0_single_channel_FFT_peak"]["doppler_hz"].append(np.asarray(b0.doppler_hz))
        accum["B1_three_channel_noncoherent_FFT_peak"]["range_cell"].append(np.asarray(b1.range_cell))
        accum["B1_three_channel_noncoherent_FFT_peak"]["doppler_hz"].append(np.asarray(b1.doppler_hz))
        for name, value in result["fft"].items():
            target = {
                "native_fft_peak": "B1_detection_plus_native_FFT_parameters",
                "native_fft_log_parabolic": "B1_detection_plus_log_parabolic_parameters",
                "fourfold_fft_parabolic": "B1_detection_plus_4x_zero_pad_parameters",
            }[name]
            accum[target]["range_cell"].append(np.asarray(value["range_cell"]))
            accum[target]["doppler_hz"].append(np.asarray(value["doppler_hz"]))
        accum["lag_one_phase_difference_estimator"]["range_cell"].append(np.asarray(result["fft"]["native_fft_peak"]["range_cell"]))
        accum["lag_one_phase_difference_estimator"]["doppler_hz"].append(np.asarray(result["phase"]))
    merged: dict[str, dict[str, np.ndarray]] = {}
    for name, fields in accum.items():
        merged[name] = {}
        for key, values in fields.items():
            if values:
                merged[name][key] = np.concatenate(values).astype(np.float64)
    return merged


def evaluate_classical(r7: dict[str, Any], physics: dict[str, Any], arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seed = int(r7["splits"]["bootstrap_seed"]) + 90_000
    # L0 contains deterministic zero H0 and target-only H1; it has no calibrated operating point.
    l0_h1 = _classical_values(arrays["l0_h1"], physics, with_parameters=True)
    zero_scores = _classical_values(np.zeros((int(r7["splits"]["l0_h0_scenes"]), 3, 130, 32), dtype=np.complex64), physics, with_parameters=False)
    l0_truth_r = arrays["truth_l0_range"]
    l0_truth_d = arrays["truth_l0_doppler"]
    snrs = [float(value) for value in r7["physics"]["snr_db"]]
    for method_index, name in enumerate(l0_h1):
        detection_available = name != "lag_one_phase_difference_estimator"
        detector_name = "B0_single_channel_FFT_peak" if name == "B0_single_channel_FFT_peak" else "B1_three_channel_noncoherent_FFT_peak"
        score0 = zero_scores[detector_name].get("score") if detection_available else None
        score1 = l0_h1[detector_name].get("score") if detection_available else None
        param = _prediction_summary(
            calibration_scores=None,
            verification_scores=score0,
            h1_scores=score1,
            range_prediction=l0_h1[name].get("range_cell"),
            doppler_prediction=l0_h1[name].get("doppler_hz"),
            truth_range=l0_truth_r,
            truth_doppler=l0_truth_d,
            target_pfa=float(r7["splits"]["target_pfa_cpi"]),
            bootstrap_seed=seed + method_index,
        )
        param["detection"]["status"] = "UNTHRESHOLDED_ZERO_H0_DIAGNOSTIC" if detection_available else "NOT_APPLICABLE"
        results.append({"method_id": name, "task_scope": "hybrid_detector_estimator" if detection_available else "parameter_only",
                        "datasets": {"L0_TARGET_ONLY": param}})

    for snr_index, snr_db in enumerate(snrs):
        h0cal = _classical_values(arrays["l1_h0_calibration"][snr_index], physics, with_parameters=False)
        h0ver = _classical_values(arrays["l1_h0_verification"][snr_index], physics, with_parameters=False)
        h1 = _classical_values(arrays["l1_h1"][snr_index], physics, with_parameters=True)
        truth_r = arrays["truth_l1_h1_range"][snr_index]
        truth_d = arrays["truth_l1_h1_doppler"][snr_index]
        profiles = None
        profile_count = int(r7["splits"]["profile_ml_h1_per_snr"])
        profile = estimate_profile_ml_reference(
            arrays["l1_h1"][snr_index, :profile_count],
            prf_hz=1300.0,
            range_interval=tuple(float(x) for x in r7["splits"]["profile_ml_range_cell_interval"]),
            range_grid_step=float(r7["splits"]["profile_ml_range_grid_step_cell"]),
            frequency_fft_factor=int(r7["splits"]["profile_ml_frequency_fft_factor"]),
            chunk_size=16,
        )
        profiles = {"range_cell": profile["range_cell"], "doppler_hz": profile["doppler_hz"]}
        methods: dict[str, tuple[str | None, dict[str, np.ndarray] | None]] = {
            "B0_single_channel_FFT_peak": ("B0_single_channel_FFT_peak", h1["B0_single_channel_FFT_peak"]),
            "B1_three_channel_noncoherent_FFT_peak": ("B1_three_channel_noncoherent_FFT_peak", h1["B1_three_channel_noncoherent_FFT_peak"]),
            "B1_detection_plus_native_FFT_parameters": ("B1_three_channel_noncoherent_FFT_peak", h1["B1_detection_plus_native_FFT_parameters"]),
            "B1_detection_plus_log_parabolic_parameters": ("B1_three_channel_noncoherent_FFT_peak", h1["B1_detection_plus_log_parabolic_parameters"]),
            "B1_detection_plus_4x_zero_pad_parameters": ("B1_three_channel_noncoherent_FFT_peak", h1["B1_detection_plus_4x_zero_pad_parameters"]),
            "lag_one_phase_difference_estimator": (None, h1["lag_one_phase_difference_estimator"]),
            "profile_ML_finite_grid_256_H1": (None, profiles),
        }
        for method_index, (name, (score_method, estimates)) in enumerate(methods.items()):
            if score_method is None:
                cal_scores = val_scores = h1_scores = None
            else:
                cal_scores = h0cal[score_method]["score"]
                val_scores = h0ver[score_method]["score"]
                h1_scores = h1[score_method]["score"]
            fields = estimates or {}
            param = _prediction_summary(
                calibration_scores=cal_scores,
                verification_scores=val_scores,
                h1_scores=h1_scores,
                range_prediction=fields.get("range_cell"),
                doppler_prediction=fields.get("doppler_hz"),
                truth_range=truth_r[:len(fields.get("range_cell", []))] if fields.get("range_cell") is not None else truth_r,
                truth_doppler=truth_d[:len(fields.get("doppler_hz", []))] if fields.get("doppler_hz") is not None else truth_d,
                target_pfa=float(r7["splits"]["target_pfa_cpi"]),
                bootstrap_seed=seed + snr_index * 100 + method_index,
            )
            param["detection"]["status"] = "EVALUATED" if score_method else "NOT_APPLICABLE"
            param.update({"dataset": "L1_AWGN_NO_CLUTTER", "snr_db": snr_db,
                          "h0_calibration_n": int(r7["splits"]["l1_h0_calibration_per_snr"]),
                          "h0_verification_n": int(r7["splits"]["l1_h0_verification_per_snr"]),
                          "h1_n": len(fields.get("range_cell", [])) if name.startswith("profile_ML") else int(r7["splits"]["l1_h1_per_snr"]),
                          "parameter_subset": "first_fixed_indices" if name.startswith("profile_ML") else "all_H1"})
            existing = next((entry for entry in results if entry["method_id"] == name), None)
            if existing is None:
                existing = {"method_id": name, "task_scope": "hybrid_detector_estimator" if score_method else "parameter_only", "datasets": {}}
                results.append(existing)
            existing["datasets"].setdefault("L1_AWGN_NO_CLUTTER", []).append(param)
        print(f"  classical L1 condition {snr_index + 1}/{len(snrs)} complete")
    return results


def cleanup_cache(r7: dict[str, Any]) -> dict[str, Any]:
    cache_dir = ROOT / r7["outputs"]["temporary_cache_root"]
    owner_path = cache_dir / "owner.json"
    if not owner_path.is_file():
        return {"status": "NO_CACHE"}
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if owner.get("owner_id") != OWNER_ID or owner.get("status") != "complete":
        raise ValueError("refusing to clean a cache without a complete D2-A-R7-D owner manifest")
    removed: list[str] = []
    for name in owner.get("files", []):
        path = (cache_dir / name).resolve()
        if path.parent != cache_dir.resolve() or path.name not in {"l0_h1.npy", "l1_h0_calibration.npy", "l1_h0_verification.npy", "l1_h1.npy", "truth_arrays.npz"}:
            raise ValueError(f"unexpected R7-D cache entry: {name}")
        if path.exists():
            size = path.stat().st_size
            path.unlink()
            removed.append(f"{name}:{size}")
    owner_path.unlink()
    cache_dir.rmdir()
    return {"status": "REMOVED_OWNER_SCOPED_CACHE", "files": removed,
            "scene_plan_retained": str(SCENE_PLAN_PATH.relative_to(ROOT))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="generate R7-D temporary IQ caches")
    parser.add_argument("--evaluate-only", action="store_true", help="evaluate existing completed caches")
    parser.add_argument("--preflight-only", action="store_true", help="strict-load every frozen checkpoint and run a CUDA shape/finite probe")
    parser.add_argument("--cleanup-cache", action="store_true", help="remove only the complete owner-listed R7-D cache")
    args = parser.parse_args()
    if sum((args.prepare_only, args.evaluate_only, args.preflight_only, args.cleanup_cache)) > 1:
        parser.error("select only one of --prepare-only, --evaluate-only, --preflight-only, --cleanup-cache")
    r7, physics = _load_configs()
    if args.cleanup_cache:
        print(json.dumps(cleanup_cache(r7), indent=2))
        return 0
    rows = _inventory()
    _verify_inventory(rows)
    if args.preflight_only:
        result = preflight_models(r7, physics, rows)
        print(json.dumps({key: result[key] for key in ("status", "passed", "failed", "device", "gpu_peak_allocated_mib")}, indent=2))
        return 0 if result["failed"] == 0 else 1
    if args.prepare_only:
        prepare_data(r7, physics)
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("R7-D is frozen to AISAR CUDA; CPU fallback is forbidden")
    device = torch.device(str(r7["runtime"]["device"]))
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    print(f"CUDA: {torch.cuda.get_device_name(device)}; free={free_bytes / 1024**3:.2f} GiB; total={total_bytes / 1024**3:.2f} GiB")
    print(f"Frozen checkpoints verified: {len(rows)}")
    if not args.evaluate_only:
        prepare_data(r7, physics)
    arrays, owner = _load_cache(r7)
    metrics = evaluate_students(r7, physics, rows, arrays)
    metrics["classical"] = evaluate_classical(r7, physics, arrays)
    metrics["cache_bytes"] = owner["bytes_on_disk"]
    metrics["status"] = "COMPLETE"
    metrics["models_completed"] = sum(item.get("status") == "COMPLETE" for item in metrics["models"])
    metrics["models_failed"] = sum(item.get("status") in {"LOAD_FAILED", "FORWARD_FAILED"} for item in metrics["models"])
    metrics["training_performed"] = False
    metrics["distillation_performed"] = False
    _json_write(METRICS_PATH, metrics)
    print(f"R7-D evaluation complete: {metrics['models_completed']}/{len(rows)} students loaded and evaluated")
    print(f"Metrics: {METRICS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
