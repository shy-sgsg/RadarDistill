#!/usr/bin/env python3
"""Run the preregistered D2-A-R6-D joint architecture pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import yaml
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STAGE = ROOT / "experiments/D2-A-R6/R6-D"
CACHE = STAGE / "cache" / "training"
RUNS = STAGE / "runs" / "seed_505"
PREDICTIONS = STAGE / "predictions"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"

from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_r6a import build_split_plan, calibrate_threshold, summarize_parameter_estimator, wrapped_error
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_joint import (
    R6D_ARMS,
    build_r6d_student,
    count_r6d_parameters,
)
from radardistill.training.d2a_r6b_hard_negative import (
    assemble_independent_h0_h1_batch,
    build_h0_sample_plan,
    build_repeated_h1_plan,
)
from radardistill.training.d2a_r6d_joint import (
    audit_r6d_scene_plans,
    build_r6d_paired_plan,
    r6d_learning_rate,
    r6d_loss,
    r6d_scene_ids,
)
from radardistill.distillation.task_losses import decode_task_outputs


def _read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    r6d = _read_yaml("experiments/D2-A-R6/R6-D/config.yaml")
    r6a = _read_yaml("experiments/D2-A-R6/R6-A/config.yaml")
    physics = _read_yaml(r6d["radar"]["source_config"])
    return r6d, r6a, physics, {"radar": r6a["radar"]}


def _all_split_plans(
    config: Mapping[str, Any], r6a: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    conditions = r6a["radar"]["conditions"]
    data = config["data"]
    plans = {
        "train": build_r6d_paired_plan(data, conditions, "train"),
        "validation": build_r6d_paired_plan(data, conditions, "validation"),
    }
    eval_names = ("h0_calibration", "h0_verification", "h1_development")
    eval_data = {
        key: data[key]
        for key in ("condition_count", "condition_seed_stride", "chunk_size", "chunk_seed_stride")
        if key in data
    }
    eval_data.update({
        "condition_count": int(config["radar"]["condition_count"]),
        "splits": {name: data["splits"][name] for name in eval_names},
    })
    eval_config = {"data": eval_data, "radar": {"conditions": conditions}}
    for name in eval_names:
        plans[name] = build_split_plan(eval_config, name)
    return plans


def _prior_scene_ids() -> set[str]:
    roots = (
        ROOT / "experiments/D2-A-R5",
        ROOT / "experiments/D2-A-R6/R6-A",
        ROOT / "experiments/D2-A-R6/R6-B",
        ROOT / "experiments/D2-A-R6/R6-C",
    )
    paths: set[Path] = set()
    for base in roots:
        if base.exists():
            paths.update(base.rglob("scene_manifest*.csv"))
            paths.update(base.rglob("selected_scene_ids.csv"))
    result: set[str] = set()
    for path in sorted(paths):
        if STAGE in path.parents:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                for key in ("scene_id", "base_scene_id", "base_scene_ids"):
                    value = row.get(key)
                    if value:
                        result.update(str(value).split(";"))
    return {value for value in result if value}


def _json_scene_counts(plans: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    return {name: sum(int(row["count"]) for row in plan) for name, plan in plans.items()}


def preflight() -> dict[str, Any]:
    """G0: bind pushed implementation, protocol, scene IDs, disk, and CUDA."""
    config, r6a, _, _ = _configs()
    head = _git("rev-parse", "HEAD")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-D G0 requires the committed, pushed implementation on main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R6-D G0 requires a clean tracked worktree")
    if config["experiment"]["protocol_version"] != 1.2 or config["experiment"]["status"] != "preregistered":
        raise RuntimeError("R6-D protocol/config is not frozen as registered")
    if config["experiment"]["final_blind_authorized"] is not False:
        raise RuntimeError("R6-D final blind is not authorized")
    plans = _all_split_plans(config, r6a)
    expected = {name: int(value) for name, value in _json_scene_counts(plans).items()}
    g1_seed = int(config["data"]["g1_test_seed"])
    g1_plan = [{"seed": g1_seed, "count": 16}]
    plans["g1_test"] = g1_plan
    expected["g1_test"] = 16
    prior = _prior_scene_ids()
    audit = audit_r6d_scene_plans(plans, expected_counts=expected, prior_scene_ids=prior)
    if CACHE.exists():
        raise RuntimeError("R6-D cache exists before G0; inspect it manually, do not overwrite it")
    free, disk_total = shutil.disk_usage(STAGE).free, shutil.disk_usage(STAGE).total
    required = int(config["data"]["minimum_free_disk_bytes_before_generation"])
    if free < required:
        raise RuntimeError(f"R6-D needs {required} free bytes before cache generation; found {free}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-D requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    if gpu != DEVICE_NAME:
        raise RuntimeError(f"R6-D requires {DEVICE_NAME}, found {gpu}")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    counts: dict[str, Any] = {}
    for arm in R6D_ARMS:
        model = build_r6d_student(
            arm, width=int(config["models"]["width"]),
            seed=int(config["training"]["training_seed"]),
            prf_hz=float(config["radar"]["prf_hz"]),
        )
        counts[arm] = count_r6d_parameters(model)
        del model
    protocol_path = STAGE / "protocol.md"
    config_path = STAGE / "config.yaml"
    result = {
        "status": "G0_PASS",
        "git_sha": head,
        "protocol_sha256": _sha256(protocol_path),
        "config_sha256": _sha256(config_path),
        "protocol_version": float(config["experiment"]["protocol_version"]),
        "split_scene_counts": audit["scene_counts"],
        "prior_scene_id_count": len(prior),
        "prior_scene_id_overlap": 0,
        "g1_test_seed": g1_seed,
        "g1_test_scene_ids": 16,
        "disk_free_before_generation_bytes": int(free),
        "disk_total_bytes": int(disk_total),
        "minimum_free_bytes": required,
        "estimated_train_validation_cache_bytes": int(config["data"]["estimated_train_validation_cache_bytes"]),
        "gpu": gpu,
        "device": "cuda:0",
        "free_vram_before_g1_bytes": int(free_vram),
        "total_vram_bytes": int(total_vram),
        "model_parameter_counts": counts,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "formal_data_generated": False,
        "final_blind_generated": False,
    }
    _write_json(STAGE / "preflight.json", result)
    return result


def _require_preflight() -> tuple[dict[str, Any], str]:
    path = STAGE / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run R6-D preflight before G1 or formal scene generation")
    record = json.loads(path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    config, _, _, _ = _configs()
    if (
        record.get("status") != "G0_PASS"
        or record.get("git_sha") != head
        or record.get("protocol_sha256") != _sha256(STAGE / "protocol.md")
        or record.get("config_sha256") != _sha256(STAGE / "config.yaml")
    ):
        raise RuntimeError("R6-D preflight does not match the current code/protocol/config")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-D execution requires its frozen implementation pushed to main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked source changed after R6-D G0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != DEVICE_NAME:
        raise RuntimeError("R6-D requires the registered RTX 3050 Laptop CUDA device")
    return record, head


def _paired_iq_and_truth(generated: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    count = len(generated.base_scene_id)
    interleaved = np.empty((count, 2, 3, 130, 32), dtype=np.complex64)
    interleaved[:, 0] = generated.iq_h0
    interleaved[:, 1] = generated.iq_h1
    flat = interleaved.reshape(2 * count, 3, 130, 32)
    iq_ri = np.concatenate((flat.real, flat.imag), axis=1).astype(np.float32, copy=False)
    presence = np.tile(np.asarray([0.0, 1.0], dtype=np.float32), count)
    range_truth = np.zeros(2 * count, dtype=np.float32)
    doppler_truth = np.zeros(2 * count, dtype=np.float32)
    range_truth[1::2] = np.asarray(generated.truth_target_range_cell, dtype=np.float32)
    doppler_truth[1::2] = np.asarray(generated.truth_target_doppler_hz, dtype=np.float32)
    return np.ascontiguousarray(iq_ri), {
        "truth_presence": presence,
        "truth_range_cell": range_truth,
        "truth_doppler_hz": doppler_truth,
    }


def _loss_on_device(
    model: torch.nn.Module,
    iq: torch.Tensor,
    truth: Mapping[str, torch.Tensor],
    *,
    arm: str,
    microbatch: int,
    prf_hz: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    totals: dict[str, torch.Tensor] = {}
    for start in range(0, len(iq), microbatch):
        stop = min(start + microbatch, len(iq))
        inputs = iq[start:stop].to(device)
        labels = {key: value[start:stop].to(device) for key, value in truth.items()}
        outputs = model(inputs)
        terms = r6d_loss(outputs, labels, arm=arm, prf_hz=prf_hz)
        scale = (stop - start) / len(iq)
        for key, value in terms.items():
            totals[key] = totals.get(key, value.detach() * 0.0) + value * scale
    return totals


def _g1_probe_microbatch(
    microbatch: int, *, seed: int, prf_hz: float, width: int, device: torch.device
) -> dict[str, Any] | None:
    reports: dict[str, Any] = {}
    for index, arm in enumerate(R6D_ARMS):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = build_r6d_student(arm, width=width, seed=seed, prf_hz=prf_hz).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        x = torch.randn(microbatch, 6, 130, 32, device=device)
        truth = {
            "truth_presence": torch.tensor(([0.0, 1.0] * ((microbatch + 1) // 2))[:microbatch], device=device),
            "truth_range_cell": torch.full((microbatch,), 11.25, device=device),
            "truth_doppler_hz": torch.full((microbatch,), 649.9, device=device),
        }
        try:
            outputs = model(x)
            terms = r6d_loss(outputs, truth, arm=arm, prf_hz=prf_hz)
            optimizer.zero_grad(set_to_none=True)
            terms["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite G1 gradient norm")
            optimizer.step()
            reports[arm] = {
                "parameters": count_r6d_parameters(model),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        except torch.cuda.OutOfMemoryError:
            del model, optimizer
            torch.cuda.empty_cache()
            return None
        del model, optimizer, x, truth, outputs, terms
        torch.cuda.empty_cache()
    return reports


def run_g1() -> dict[str, Any]:
    """Run CUDA architecture/resource and 16-pair physical micro-overfit gates."""
    preflight_record, head = _require_preflight()
    previous = STAGE / "g1.json"
    if previous.exists():
        record = json.loads(previous.read_text(encoding="utf-8"))
        if record.get("git_sha") == head and record.get("status") == "G1_PASS":
            return record
    config, r6a, physics, _ = _configs()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    selected_microbatch = None
    probe_reports = None
    for microbatch in (16, 8, 4):
        probe_reports = _g1_probe_microbatch(
            microbatch,
            seed=int(config["training"]["training_seed"]),
            prf_hz=float(config["radar"]["prf_hz"]),
            width=int(config["models"]["width"]),
            device=device,
        )
        if probe_reports is not None:
            selected_microbatch = microbatch
            break
    if selected_microbatch is None or probe_reports is None:
        raise RuntimeError("all R6-D arms OOM at microbatch4; G1 failed without changing effective batch")

    conditions = r6a["radar"]["conditions"]
    g1_condition = next(
        condition for condition in conditions
        if condition["course"] == "C2" and condition["scenario_id"] == "partial_overlap"
    )
    generated = generate_d2a_paired_batch(
        16,
        int(config["data"]["g1_test_seed"]),
        float(g1_condition["snr_db"]),
        float(g1_condition["cnr_db"]),
        dict(physics),
        scenario_id=str(g1_condition["scenario_id"]),
    )
    x_np, truth_np = _paired_iq_and_truth(generated)
    # Duplicate the 16 physical pairs to exercise effective batch64 accumulation;
    # the G1 learning gate still contains only 16 unique H0/H1 scene pairs.
    x_np = np.concatenate((x_np, x_np), axis=0)
    truth_np = {key: np.concatenate((value, value), axis=0) for key, value in truth_np.items()}
    x = torch.from_numpy(x_np)
    truth = {key: torch.from_numpy(value) for key, value in truth_np.items()}
    reports: dict[str, Any] = {}
    min_reduction = float(config["training"]["g1_min_relative_loss_reduction"])
    updates = int(config["training"]["g1_micro_overfit_updates"])
    if len(generated.base_scene_id) != int(config["training"]["g1_micro_overfit_pairs"]):
        raise RuntimeError("R6-D G1 physical sample count mismatch")
    for arm in R6D_ARMS:
        torch.cuda.empty_cache()
        model = build_r6d_student(
            arm,
            width=int(config["models"]["width"]),
            seed=int(config["training"]["training_seed"]),
            prf_hz=float(config["radar"]["prf_hz"]),
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        initial_terms = _loss_on_device(
            model, x, truth, arm=arm, microbatch=selected_microbatch,
            prf_hz=float(config["radar"]["prf_hz"]), device=device,
        )
        history: list[float] = []
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        for update in range(1, updates + 1):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(x), selected_microbatch):
                stop = start + selected_microbatch
                xb = x[start:stop].to(device)
                tb = {key: value[start:stop].to(device) for key, value in truth.items()}
                outputs = model(xb)
                terms = r6d_loss(outputs, tb, arm=arm, prf_hz=float(config["radar"]["prf_hz"]))
                scaled = terms["total"] * (selected_microbatch / len(x))
                if not torch.isfinite(scaled):
                    raise FloatingPointError(f"nonfinite R6-D G1 loss for {arm}")
                scaled.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"nonfinite R6-D G1 gradient for {arm}")
            optimizer.step()
            with torch.inference_mode():
                current = _loss_on_device(
                    model, x, truth, arm=arm, microbatch=selected_microbatch,
                    prf_hz=float(config["radar"]["prf_hz"]), device=device,
                )
            history.append(float(current["total"].item()))
        initial_loss = float(initial_terms["total"].item())
        final_loss = float(np.mean(history[-20:]))
        reduction = (initial_loss - final_loss) / max(abs(initial_loss), 1e-12)
        passes = bool(np.isfinite(history).all() and reduction >= min_reduction)
        reports[arm] = {
            "initial_loss": initial_loss,
            "last20_mean_loss": final_loss,
            "relative_loss_reduction": reduction,
            "minimum_required_relative_reduction": min_reduction,
            "updates": updates,
            "unique_paired_scenes": 16,
            "effective_batch_size": 64,
            "microbatch": selected_microbatch,
            "passed": passes,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "seconds": time.perf_counter() - start_time,
        }
        del model, optimizer
        torch.cuda.empty_cache()
    status = "G1_PASS" if all(value["passed"] for value in reports.values()) else "G1_FAIL"
    record = {
        "status": status,
        "git_sha": head,
        "protocol_sha256": preflight_record["protocol_sha256"],
        "config_sha256": preflight_record["config_sha256"],
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "fp32": True,
        "effective_batch_size": 64,
        "formal_microbatch": selected_microbatch,
        "resource_probe": probe_reports,
        "physical_test_seed": int(config["data"]["g1_test_seed"]),
        "physical_test_scene_ids": [str(value) for value in generated.base_scene_id],
        "condition": dict(g1_condition),
        "arms": reports,
        "formal_data_generated": False,
    }
    _write_json(STAGE / "g1.json", record)
    return record


def _paired_cache_paths(split: str) -> dict[str, Path]:
    directory = CACHE / split
    names = ("iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id")
    return {name: directory / f"{name}.npy" for name in names}


def _cache_owner(split: str, head: str, plan: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    config, _, _, _ = _configs()
    plan_hash = hashlib.sha256(
        json.dumps(list(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "owner": "D2-A-R6-D-paired-cache",
        "git_sha": head,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "split": split,
        "scene_count": sum(int(row["count"]) for row in plan),
        "plan_sha256": plan_hash,
        "source_config": config["radar"]["source_config"],
    }


def _remaining_owned_cache_bytes(config: Mapping[str, Any], r6a: Mapping[str, Any], head: str) -> int:
    plans = _all_split_plans(config, r6a)
    bytes_per_row = 2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize + 2 * np.dtype(np.float32).itemsize + np.dtype(np.uint8).itemsize + 48 * np.dtype("<U1").itemsize
    remaining = 0
    for split in ("train", "validation"):
        plan = plans[split]
        count = sum(int(row["count"]) for row in plan)
        written = 0
        directory = CACHE / split
        if directory.exists():
            owner = _cache_owner(split, head, plan)
            owner_path, progress_path = directory / "owner.json", directory / "progress.json"
            if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
                raise RuntimeError(f"refusing to estimate free space for an unowned cache: {directory}")
            if progress_path.is_file():
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
                if progress.get("owner") != owner:
                    raise RuntimeError(f"R6-D {split} progress owner mismatch")
                written = int(progress.get("rows_written", -1))
                if not 0 <= written <= count:
                    raise RuntimeError(f"R6-D {split} progress row count is invalid")
        remaining += (count - written) * bytes_per_row
    return int(remaining + 2 * 1024 * 1024)


def generate_paired_cache(split: str) -> dict[str, Any]:
    preflight_record, head = _require_preflight()
    g1_path = STAGE / "g1.json"
    if not g1_path.is_file() or json.loads(g1_path.read_text(encoding="utf-8")).get("status") != "G1_PASS":
        raise RuntimeError("R6-D paired data generation requires G1_PASS")
    if split not in {"train", "validation"}:
        raise ValueError("only train/validation paired caches are materialized")
    config, r6a, physics, _ = _configs()
    free_now = int(shutil.disk_usage(STAGE).free)
    reserve = 4 * 1024 ** 3
    remaining_bytes = _remaining_owned_cache_bytes(config, r6a, head)
    if free_now < remaining_bytes + reserve:
        raise RuntimeError(
            f"R6-D cache generation needs {remaining_bytes + reserve} free bytes including the 4 GiB reserve; found {free_now}"
        )
    plan = _all_split_plans(config, r6a)[split]
    count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    owner = _cache_owner(split, head, plan)
    existed = directory.exists()
    if existed:
        owner_path = directory / "owner.json"
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
            raise RuntimeError(f"refusing to reuse unowned or differently-owned R6-D cache: {directory}")
    else:
        directory.mkdir(parents=True)
        _write_json(directory / "owner.json", owner)
    paths = _paired_cache_paths(split)
    progress_path = directory / "progress.json"
    manifest_path = STAGE / f"scene_manifest_{split}.csv"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("R6-D cache progress owner mismatch")
        completed = int(progress["completed_chunks"])
        rows_written = int(progress["rows_written"])
        expected_rows = sum(int(row["count"]) for row in plan[:completed])
        if not 0 <= completed <= len(plan) or rows_written != expected_rows:
            raise RuntimeError("R6-D cache cursor differs from the frozen plan")
        if manifest_path.exists():
            with manifest_path.open("r+b") as handle:
                handle.truncate(int(progress["manifest_offset"]))
        elif rows_written:
            raise RuntimeError("R6-D cache manifest missing for a nonempty cache")
    else:
        unexpected = [entry.name for entry in directory.iterdir() if entry.name != "owner.json"]
        if existed and (unexpected or manifest_path.exists()):
            raise RuntimeError("R6-D partial cache lacks a valid progress record")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite unowned R6-D manifest: {manifest_path}")
        completed = rows_written = 0
        progress = {"owner": owner, "completed_chunks": 0, "rows_written": 0, "manifest_offset": 0}
        _write_json(progress_path, progress)
    shapes = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "iq_h1": ((count, 3, 130, 32), np.complex64),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
        "condition_index": ((count,), np.uint8),
        "scene_id": ((count,), "<U48"),
    }
    for name, (shape, dtype) in shapes.items():
        if not paths[name].is_file():
            np.lib.format.open_memmap(paths[name], mode="w+", dtype=dtype, shape=shape).flush()
        else:
            existing_array = np.load(paths[name], mmap_mode="r")
            if existing_array.shape != shape or existing_array.dtype != np.dtype(dtype):
                raise RuntimeError(f"R6-D {split} cache shape/dtype mismatch for {name}")
    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    offset = rows_written
    with manifest_path.open("a", newline="", encoding="utf-8") as manifest:
        writer = csv.writer(manifest, lineterminator="\n")
        if manifest.tell() == 0:
            writer.writerow(("scene_id", "split", "seed", "condition_index", "course", "scenario_id",
                             "snr_db", "cnr_db", "truth_range_cell", "truth_doppler_hz"))
            manifest.flush()
            progress["manifest_offset"] = manifest.tell()
            _write_json(progress_path, progress)
        for chunk_index, spec in enumerate(plan[completed:], start=completed):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            stop = offset + int(spec["count"])
            expected_shape = (int(spec["count"]), 3, 130, 32)
            if generated.iq_h0.shape != expected_shape or generated.iq_h1.shape != expected_shape:
                raise RuntimeError("R6-D simulator returned an unexpected paired IQ shape")
            arrays["iq_h0"][offset:stop] = generated.iq_h0
            arrays["iq_h1"][offset:stop] = generated.iq_h1
            arrays["truth_range_cell"][offset:stop] = generated.truth_target_range_cell.astype(np.float32)
            arrays["truth_doppler_hz"][offset:stop] = generated.truth_target_doppler_hz.astype(np.float32)
            arrays["condition_index"][offset:stop] = int(spec["condition_index"])
            arrays["scene_id"][offset:stop] = generated.base_scene_id.astype("<U48")
            for index, scene_id in enumerate(generated.base_scene_id):
                writer.writerow((
                    str(scene_id), split, int(spec["seed"]), int(spec["condition_index"]), spec["course"],
                    condition["scenario_id"], repr(float(generated.snr_db[index])),
                    repr(float(generated.requested_cnr_db[index])),
                    repr(float(generated.truth_target_range_cell[index])),
                    repr(float(generated.truth_target_doppler_hz[index])),
                ))
            manifest.flush()
            for value in arrays.values():
                value.flush()
            offset = stop
            progress = {
                "owner": owner,
                "completed_chunks": chunk_index + 1,
                "rows_written": offset,
                "manifest_offset": manifest.tell(),
            }
            _write_json(progress_path, progress)
            if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
                print(json.dumps({
                    "stage": "generate", "split": split, "chunks_done": chunk_index + 1,
                    "chunks_total": len(plan), "scenes_done": offset,
                }, ensure_ascii=False), flush=True)
    if offset != count:
        raise RuntimeError(f"R6-D cache wrote {offset} scenes; expected {count}")
    del arrays
    return {
        "split": split,
        "complete": True,
        "scenes": count,
        "chunks": len(plan),
        "cache_bytes": sum(path.stat().st_size for path in paths.values()),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "preflight_sha": preflight_record["git_sha"],
    }


def _load_pair_cache(split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("unknown R6-D paired split")
    head = _git("rev-parse", "HEAD")
    config, r6a, _, _ = _configs()
    plan = _all_split_plans(config, r6a)[split]
    expected_count = sum(int(row["count"]) for row in plan)
    owner = _cache_owner(split, head, plan)
    directory = CACHE / split
    owner_path, progress_path = directory / "owner.json", directory / "progress.json"
    if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
        raise RuntimeError(f"R6-D {split} cache owner mismatch")
    if not progress_path.is_file():
        raise RuntimeError(f"R6-D {split} cache has no progress record")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("owner") != owner or int(progress.get("completed_chunks", -1)) != len(plan):
        raise RuntimeError(f"R6-D {split} cache is incomplete")
    if int(progress.get("rows_written", -1)) != expected_count:
        raise RuntimeError(f"R6-D {split} cache row count mismatch")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in _paired_cache_paths(split).items()}
    if any(len(value) != expected_count for value in arrays.values()):
        raise RuntimeError(f"R6-D {split} cache array length mismatch")
    return arrays


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "update", "learning_rate", "interval_train_total", "interval_train_presence",
        "interval_train_range_cell", "interval_train_range_offset", "interval_train_doppler",
        "interval_train_candidate", "validation_total", "validation_presence",
        "validation_range_cell", "validation_range_offset", "validation_doppler",
        "validation_candidate", "validation_auc", "validation_range_rmse_cells",
        "validation_doppler_rmse_hz", "seconds_since_previous_checkpoint",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _task_loss_weights(config: Mapping[str, Any]) -> dict[str, float]:
    source = config["training"]["task_losses"]
    return {
        "presence": float(source["presence_bce_weight"]),
        "range_cell": float(source["range_cell_cross_entropy_weight"]),
        "range_offset": float(source["range_offset_smooth_l1_weight"]),
        "doppler": float(source["doppler_circular_cosine_weight"]),
        "candidate": float(source["candidate_cross_entropy_weight"]),
    }


def _arm_objective(arm: str, terms: Mapping[str, torch.Tensor]) -> float:
    return float(terms["total"].item())


def _validation_summary(
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    *,
    arm: str,
    microbatch: int,
    prf_hz: float,
    device: torch.device,
) -> dict[str, float]:
    count = len(arrays["iq_h0"])
    keys = ("total", "presence", "range_cell", "range_offset", "doppler", "candidate")
    weighted = {key: 0.0 for key in keys}
    h0_scores: list[np.ndarray] = []
    h1_scores: list[np.ndarray] = []
    h1_range: list[np.ndarray] = []
    h1_doppler: list[np.ndarray] = []
    truth_range: list[np.ndarray] = []
    truth_doppler: list[np.ndarray] = []
    pair_chunk = max(1, microbatch // 2)
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, pair_chunk):
            indices = np.arange(start, min(start + pair_chunk, count), dtype=np.int64)
            batch = assemble_independent_h0_h1_batch(arrays, indices, indices)
            x = torch.from_numpy(batch["iq_ri"])
            truth = {
                key: torch.from_numpy(batch[key])
                for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
            }
            pair_count = len(indices)
            batch_terms = {key: 0.0 for key in keys}
            scores: list[np.ndarray] = []
            ranges: list[np.ndarray] = []
            dopplers: list[np.ndarray] = []
            for sub_start in range(0, len(x), microbatch):
                sub_stop = min(sub_start + microbatch, len(x))
                sub_x = x[sub_start:sub_stop].to(device)
                sub_truth = {key: value[sub_start:sub_stop].to(device) for key, value in truth.items()}
                output = model(sub_x)
                terms = r6d_loss(output, sub_truth, arm=arm, prf_hz=prf_hz)
                sub_weight = (sub_stop - sub_start) / len(x)
                for key in keys:
                    batch_terms[key] += float(terms[key].item()) * sub_weight
                if arm != "j1_parameter":
                    decoded = decode_task_outputs(output, prf_hz)
                    scores.append(decoded["presence_probability"].cpu().numpy())
                    if arm != "j1_detection":
                        ranges.append(decoded["range_cell"].cpu().numpy())
                        dopplers.append(decoded["doppler_hz"].cpu().numpy())
            for key in keys:
                weighted[key] += batch_terms[key] * pair_count / count
            if arm != "j1_parameter":
                score = np.concatenate(scores)
                h0_scores.append(score[::2])
                h1_scores.append(score[1::2])
            if arm not in {"j1_parameter", "j1_detection"}:
                pred_range = np.concatenate(ranges)
                pred_doppler = np.concatenate(dopplers)
                h1_range.append(pred_range[1::2])
                h1_doppler.append(pred_doppler[1::2])
                truth_range.append(batch["truth_range_cell"][1::2])
                truth_doppler.append(batch["truth_doppler_hz"][1::2])
            del x, truth, batch
    if arm == "j1_parameter":
        auc = None
    else:
        p0, p1 = np.concatenate(h0_scores), np.concatenate(h1_scores)
        auc = float(empirical_auc(p0, p1))
    if arm in {"j1_parameter", "j1_detection"}:
        range_rmse = None
        doppler_rmse = None
    else:
        pr, pd = np.concatenate(h1_range), np.concatenate(h1_doppler)
        tr = np.concatenate(truth_range).astype(np.float64)
        td = np.concatenate(truth_doppler).astype(np.float64)
        range_rmse = float(np.sqrt(np.mean((pr - tr) ** 2)))
        doppler_rmse = float(np.sqrt(np.mean(wrapped_error(pd, td, prf_hz) ** 2)))
    return {
        "total": weighted["total"],
        "presence": weighted["presence"],
        "range_cell": weighted["range_cell"],
        "range_offset": weighted["range_offset"],
        "doppler": weighted["doppler"],
        "candidate": weighted["candidate"],
        "auc": auc,
        "range_rmse_cells": range_rmse,
        "doppler_rmse_hz": doppler_rmse,
    }


def _training_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    *,
    arm: str,
    microbatch: int,
    effective_batch: int,
    prf_hz: float,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    if effective_batch % microbatch or microbatch % 2 or len(batch["iq_ri"]) != effective_batch:
        raise ValueError("R6-D microbatch must divide effective batch and preserve H0/H1 balance")
    x = torch.from_numpy(batch["iq_ri"])
    truth = {
        key: torch.from_numpy(batch[key])
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("total", "presence", "range_cell", "range_offset", "doppler", "candidate")}
    for start in range(0, effective_batch, microbatch):
        stop = start + microbatch
        xb = x[start:stop].to(device)
        tb = {key: value[start:stop].to(device) for key, value in truth.items()}
        outputs = model(xb)
        terms = r6d_loss(outputs, tb, arm=arm, prf_hz=prf_hz)
        scaled = terms["total"] * microbatch / effective_batch
        if not torch.isfinite(scaled):
            raise FloatingPointError(f"nonfinite R6-D training loss for {arm}")
        scaled.backward()
        for key in totals:
            totals[key] += float(terms[key].detach().item()) * microbatch / effective_batch
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"nonfinite R6-D gradient norm for {arm}")
    optimizer.step()
    return totals


def _training_owner(arm: str, config: Mapping[str, Any], head: str) -> dict[str, Any]:
    model = build_r6d_student(
        arm,
        width=int(config["models"]["width"]),
        seed=int(config["training"]["training_seed"]),
        prf_hz=float(config["radar"]["prf_hz"]),
    )
    return {
        "owner": "D2-A-R6-D-training-run",
        "git_sha": head,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "arm": arm,
        "training_seed": int(config["training"]["training_seed"]),
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "updates": int(config["training"]["fixed_updates"]),
        "microbatch": int(json.loads((STAGE / "g1.json").read_text())["formal_microbatch"]),
        "parameters": count_r6d_parameters(model),
        "precision": "FP32",
        "data": "train_51200_paired_scenes",
    }


def _training_plan(train: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    conditions = np.asarray(train["condition_index"], dtype=np.int64)
    updates = int(config["training"]["fixed_updates"])
    effective_batch = int(config["training"]["effective_batch_size_cpi"])
    seed = int(config["training"]["training_seed"])
    h1_plan = build_repeated_h1_plan(
        conditions,
        updates=updates,
        h1_per_update=effective_batch // 2,
        seed=seed,
    )
    h0_plan, audit = build_h0_sample_plan(
        h1_plan,
        conditions,
        conditions,
        {},
        mode="uniform",
        seed=int(config["training"]["sampler_seed"]),
        hard_fraction=0.0,
    )
    if h0_plan.shape != (updates, effective_batch // 2) or h1_plan.shape != h0_plan.shape:
        raise RuntimeError("R6-D sampler plans do not match the frozen update/batch budget")
    hashes = {
        "h0_plan_sha256": hashlib.sha256(h0_plan.tobytes()).hexdigest(),
        "h1_plan_sha256": hashlib.sha256(h1_plan.tobytes()).hexdigest(),
    }
    return h0_plan, h1_plan, {**audit, **hashes}


def train_arm(arm: str) -> dict[str, Any]:
    if arm not in R6D_ARMS:
        raise ValueError(f"unknown arm {arm}")
    _, head = _require_preflight()
    config, _, _, _ = _configs()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-D training requires G1_PASS")
    microbatch = int(g1["formal_microbatch"])
    effective_batch = int(config["training"]["effective_batch_size_cpi"])
    if effective_batch != 64 or microbatch not in (16, 8, 4):
        raise RuntimeError("R6-D batch/microbatch differs from frozen protocol")
    train = _load_pair_cache("train")
    validation = _load_pair_cache("validation")
    h0_plan, h1_plan, sampler_audit = _training_plan(train, config)
    owner = _training_owner(arm, config, head)
    out = RUNS / arm
    out.mkdir(parents=True, exist_ok=True)
    owner_path = out / "owner.json"
    if owner_path.exists() and json.loads(owner_path.read_text(encoding="utf-8")) != owner:
        raise RuntimeError(f"refusing to reuse a differently-owned training run: {out}")
    if not owner_path.exists():
        if any(out.iterdir()):
            raise RuntimeError(f"refusing to reuse a nonempty unowned run directory: {out}")
        _write_json(owner_path, owner)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(int(config["training"]["training_seed"]))
    np.random.seed(int(config["training"]["training_seed"]))
    model = build_r6d_student(
        arm,
        width=int(config["models"]["width"]),
        seed=int(config["training"]["training_seed"]),
        prf_hz=float(config["radar"]["prf_hz"]),
    ).to(device)
    counts = count_r6d_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate_initial"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    target_updates = int(config["training"]["fixed_updates"])
    last_path = out / "last_checkpoint.pt"
    best_path = out / "best_checkpoint.pt"
    history_path = out / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_objective = float("inf")
    best_update = 0
    start_update = 0
    total_seconds_before = 0.0
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("owner") != owner:
            raise RuntimeError("R6-D resume checkpoint owner mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state["update"])
        history = list(state.get("history", []))
        history = [row for row in history if int(row["update"]) <= start_update]
        best_objective = float(state.get("best_objective", float("inf")))
        best_update = int(state.get("best_update", 0))
        total_seconds_before = float(state.get("training_seconds_completed", 0.0))
        if best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            if best_state.get("owner") != owner:
                raise RuntimeError("R6-D best checkpoint owner mismatch")
            if float(best_state["validation_objective"]) < best_objective:
                best_objective = float(best_state["validation_objective"])
                best_update = int(best_state["update"])
        _write_history(history_path, history)
    elif any(out.iterdir()):
        raise RuntimeError("R6-D training directory has artifacts but no owner-bound resume checkpoint")

    validation_interval = int(config["training"]["validation_interval_updates"])
    checkpoint_interval = int(config["training"]["resume_checkpoint_interval_updates"])
    prf_hz = float(config["radar"]["prf_hz"])
    clip = float(config["training"]["gradient_clip_norm"])
    interval_sums = {key: 0.0 for key in ("total", "presence", "range_cell", "range_offset", "doppler", "candidate")}
    interval_count = 0
    segment_start = time.perf_counter()
    run_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for update in range(start_update + 1, target_updates + 1):
        lr = r6d_learning_rate(
            update,
            total_updates=target_updates,
            initial=float(config["training"]["learning_rate_initial"]),
            final=float(config["training"]["learning_rate_final"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = assemble_independent_h0_h1_batch(
            train, h0_plan[update - 1], h1_plan[update - 1]
        )
        terms = _training_update(
            model, optimizer, sampled, arm=arm, microbatch=microbatch,
            effective_batch=effective_batch, prf_hz=prf_hz, device=device, gradient_clip=clip,
        )
        interval_count += 1
        for key in interval_sums:
            interval_sums[key] += terms[key]
        if update % validation_interval == 0:
            validation_metrics = _validation_summary(
                model, validation, arm=arm, microbatch=microbatch, prf_hz=prf_hz, device=device
            )
            objective = float(validation_metrics["total"])
            if objective < best_objective:
                best_objective = objective
                best_update = update
                _atomic_torch_save({
                    "owner": owner,
                    "update": update,
                    "validation_objective": objective,
                    "model_state_dict": _cpu_tree(model.state_dict()),
                    "validation_metrics": validation_metrics,
                }, best_path)
            row = {
                "update": update,
                "learning_rate": lr,
                **{f"interval_train_{key}": interval_sums[key] / max(interval_count, 1) for key in interval_sums},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "seconds_since_previous_checkpoint": time.perf_counter() - segment_start,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            history.append(row)
            _write_history(history_path, history)
            print(json.dumps({"stage": "train", "arm": arm, "update": update,
                              "train_total": row["interval_train_total"], "validation_total": objective,
                              "validation_auc": row["validation_auc"], "best_update": best_update,
                              "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"]},
                             ensure_ascii=False), flush=True)
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_count = 0
            segment_start = time.perf_counter()
        if update % checkpoint_interval == 0:
            _atomic_torch_save({
                "owner": owner,
                "update": update,
                "model_state_dict": _cpu_tree(model.state_dict()),
                "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                "history": history,
                "best_objective": best_objective,
                "best_update": best_update,
                "training_seconds_completed": total_seconds_before + time.perf_counter() - run_start,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
                "sampler_audit": sampler_audit,
            }, last_path)
    train_seconds = total_seconds_before + time.perf_counter() - run_start
    if not best_path.is_file():
        raise RuntimeError(f"R6-D {arm} completed without a validation-selected checkpoint")
    record = {
        "status": "complete",
        "arm": arm,
        "owner": owner,
        "updates": target_updates,
        "best_update": best_update,
        "best_validation_objective": best_objective,
        "training_seconds": train_seconds,
        "effective_batch_size": effective_batch,
        "microbatch": microbatch,
        "unique_train_paired_scenes": len(train["scene_id"]),
        "scene_visits_per_presence": target_updates * (effective_batch // 2) / len(train["scene_id"]),
        "total_cpi_exposures_per_presence": target_updates * (effective_batch // 2),
        "model_parameter_counts": counts,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "sampler_audit": sampler_audit,
        "history": str(history_path.relative_to(ROOT)),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
    }
    _write_json(out / "training_record.json", record)
    _atomic_torch_save({
        "owner": owner,
        "update": target_updates,
        "model_state_dict": _cpu_tree(model.state_dict()),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "history": history,
        "best_objective": best_objective,
        "best_update": best_update,
        "training_seconds_completed": train_seconds,
        "sampler_audit": sampler_audit,
    }, last_path)
    del model, optimizer, train, validation
    torch.cuda.empty_cache()
    return record


def _eval_plans(config: Mapping[str, Any], r6a: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    conditions = r6a["radar"]["conditions"]
    source = config["data"]
    data = {
        "condition_count": int(config["radar"]["condition_count"]),
        "condition_seed_stride": int(source["condition_seed_stride"]),
        "chunk_size": int(source["chunk_size"]),
        "chunk_seed_stride": int(source["chunk_seed_stride"]),
        "splits": {name: source["splits"][name] for name in (
            "h0_calibration", "h0_verification", "h1_development"
        )},
    }
    evaluation_config = {"data": data, "radar": {"conditions": conditions}}
    return {
        name: build_split_plan(evaluation_config, name)
        for name in ("h0_calibration", "h0_verification", "h1_development")
    }


def _predict_eval_split(
    model: torch.nn.Module,
    split: str,
    plan: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    *,
    device: torch.device,
    microbatch: int,
    prf_hz: float,
    estimate_parameters: bool = True,
    estimate_detection: bool = True,
) -> dict[str, np.ndarray]:
    names = ["scene_id", "condition_index"]
    if estimate_detection:
        names.append("score")
    if split == "h1_development" and estimate_parameters:
        names += ["predicted_range_cell", "predicted_doppler_hz", "truth_range_cell", "truth_doppler_hz"]
    values: dict[str, list[np.ndarray]] = {name: [] for name in names}
    model.eval()
    with torch.inference_mode():
        for chunk_index, spec in enumerate(plan, start=1):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            iq = generated.iq_h0 if split != "h1_development" else generated.iq_h1
            iq_ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
            scores: list[np.ndarray] = []
            ranges: list[np.ndarray] = []
            dopplers: list[np.ndarray] = []
            for start in range(0, len(iq_ri), microbatch):
                x = torch.from_numpy(np.ascontiguousarray(iq_ri[start : start + microbatch])).to(device)
                outputs = model(x)
                decoded = decode_task_outputs(outputs, prf_hz) if (estimate_detection or estimate_parameters) else None
                if estimate_detection:
                    scores.append(decoded["presence_probability"].cpu().numpy())
                if split == "h1_development" and estimate_parameters:
                    ranges.append(decoded["range_cell"].cpu().numpy())
                    dopplers.append(decoded["doppler_hz"].cpu().numpy())
            count = len(generated.base_scene_id)
            values["scene_id"].append(np.asarray(generated.base_scene_id).astype("<U48"))
            values["condition_index"].append(np.full(count, int(spec["condition_index"]), dtype=np.uint8))
            if estimate_detection:
                values["score"].append(np.concatenate(scores).astype(np.float32))
            if split == "h1_development" and estimate_parameters:
                values["predicted_range_cell"].append(np.concatenate(ranges).astype(np.float32))
                values["predicted_doppler_hz"].append(np.concatenate(dopplers).astype(np.float32))
                values["truth_range_cell"].append(np.asarray(generated.truth_target_range_cell, dtype=np.float32))
                values["truth_doppler_hz"].append(np.asarray(generated.truth_target_doppler_hz, dtype=np.float32))
            if chunk_index % 1000 == 0 or chunk_index == len(plan):
                print(json.dumps({"stage": "evaluate_generate", "split": split,
                                  "chunks_done": chunk_index, "chunks_total": len(plan)},
                                 ensure_ascii=False), flush=True)
    return {key: np.concatenate(parts) for key, parts in values.items()}


def _exact_upper(false_alarms: int, total: int, alpha: float = 0.05) -> float:
    if total <= 0 or not 0 <= false_alarms <= total or not 0 < alpha < 1:
        raise ValueError("invalid exact-binomial interval inputs")
    return 1.0 if false_alarms == total else float(beta.ppf(1.0 - alpha, false_alarms + 1, total - false_alarms))


def _exact_two_sided(successes: int, total: int, confidence: float = 0.95) -> list[float]:
    if total <= 0 or not 0 <= successes <= total or not 0 < confidence < 1:
        raise ValueError("invalid exact-binomial interval inputs")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2.0, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1.0 - alpha / 2.0, successes + 1, total - successes))
    return [lower, upper]


def _h0_metrics(
    calibration: Mapping[str, np.ndarray],
    verification: Mapping[str, np.ndarray],
    *,
    threshold: float,
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    cal_scores = np.asarray(calibration["score"], dtype=np.float64)
    ver_scores = np.asarray(verification["score"], dtype=np.float64)
    cal_idx = np.asarray(calibration["condition_index"], dtype=np.int64)
    ver_idx = np.asarray(verification["condition_index"], dtype=np.int64)
    alarms = ver_scores >= threshold
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "calibration": {
            "n": int(len(cal_scores)),
            "pfa_gt": float(np.mean(cal_scores > threshold)),
            "pfa_ge": float(np.mean(cal_scores >= threshold)),
            "tie_count": int(np.count_nonzero(cal_scores == threshold)),
        },
        "verification": {
            "n": int(len(ver_scores)),
            "false_alarms": int(alarms.sum()),
            "pfa": float(alarms.mean()),
            "pfa_ci95_exact_upper": _exact_upper(int(alarms.sum()), len(alarms)),
            "tie_count": int(np.count_nonzero(ver_scores == threshold)),
        },
        "verification_by_condition": {},
        "c3_pooled": {},
        "c3_cell_familywise_alpha": 0.05,
    }
    c3_mask = np.asarray([str(c["course"]) == "C3" for c in conditions], dtype=bool)[ver_idx]
    c3_alarms = alarms[c3_mask]
    result["c3_pooled"] = {
        "n": int(c3_mask.sum()),
        "false_alarms": int(c3_alarms.sum()),
        "pfa": float(c3_alarms.mean()),
        "pfa_ci95_exact_upper": _exact_upper(int(c3_alarms.sum()), len(c3_alarms)),
    }
    for index, condition in enumerate(conditions):
        selected_cal = cal_idx == index
        selected_ver = ver_idx == index
        selected_alarms = alarms[selected_ver]
        entry = {
            "condition": dict(condition),
            "calibration_n": int(selected_cal.sum()),
            "calibration_pfa_ge": float(np.mean(cal_scores[selected_cal] >= threshold)),
            "verification_n": int(selected_ver.sum()),
            "false_alarms": int(selected_alarms.sum()),
            "pfa": float(selected_alarms.mean()),
            "pfa_ci95_exact_upper": _exact_upper(int(selected_alarms.sum()), int(selected_ver.sum())),
        }
        if str(condition["course"]) == "C3":
            entry["pfa_familywise_exact_upper"] = _exact_upper(
                int(selected_alarms.sum()), int(selected_ver.sum()), alpha=0.05 / 6.0
            )
        result["verification_by_condition"][str(index)] = entry
    result["c3_six_cells_familywise_pass"] = all(
        entry.get("pfa_familywise_exact_upper", 1.0) <= 0.10
        for entry in result["verification_by_condition"].values()
        if entry["condition"]["course"] == "C3"
    )
    return result


def _h1_metrics(
    predictions: Mapping[str, np.ndarray],
    *,
    threshold: float | None,
    conditions: Sequence[Mapping[str, Any]],
    estimate_parameters: bool = True,
) -> dict[str, Any]:
    course = np.asarray([str(item["course"]) for item in conditions])
    condition_idx = np.asarray(predictions["condition_index"], dtype=np.int64)
    group_masks = {
        "all": np.ones(len(condition_idx), dtype=bool),
        "C2_C3": np.isin(course[condition_idx], ("C2", "C3")),
        "C3": course[condition_idx] == "C3",
    }
    score = np.asarray(predictions["score"], dtype=np.float64) if threshold is not None else None
    count = len(condition_idx)
    result: dict[str, Any] = {"n": int(count), "by_group": {}}
    if threshold is not None:
        detected = score >= threshold
        result["pd"] = float(detected.mean())
        result["pd_ci95_exact"] = _exact_two_sided(int(detected.sum()), len(detected))
        result["pd_by_condition"] = {
            str(index): {
                "n": int(np.count_nonzero(condition_idx == index)),
                "pd": float(detected[condition_idx == index].mean()),
                "pd_ci95_exact": _exact_two_sided(
                    int(detected[condition_idx == index].sum()),
                    int(np.count_nonzero(condition_idx == index)),
                ),
            }
            for index in range(len(conditions))
        }
    else:
        detected = np.ones(count, dtype=bool)
        result["detector_metrics"] = "undefined_for_parameter_only_arm"
    if not estimate_parameters:
        result["parameter_metrics"] = "undefined_for_detection_only_arm"
        return result

    for group, mask in group_masks.items():
        summary = summarize_parameter_estimator(
            np.asarray(predictions["predicted_range_cell"])[mask],
            np.asarray(predictions["truth_range_cell"])[mask],
            np.asarray(predictions["predicted_doppler_hz"])[mask],
            np.asarray(predictions["truth_doppler_hz"])[mask],
            detected[mask],
            prf_hz=1300.0,
            range_gate_cells=1.0,
            doppler_gate_hz=20.0,
        )
        if threshold is not None:
            successes = int(summary["joint_success_count"])
            summary["joint_success_ci95_exact"] = _exact_two_sided(successes, int(mask.sum()))
            summary["pd"] = float(detected[mask].mean())
            summary["pd_ci95_exact"] = _exact_two_sided(int(detected[mask].sum()), int(mask.sum()))
        if threshold is None:
            summary = {
                "n": int(mask.sum()),
                "all_h1": summary["all_h1"],
                "detection_metrics": "undefined_for_parameter_only_arm",
            }
        result["by_group"][group] = summary
    result["by_condition"] = {}
    for index, condition in enumerate(conditions):
        mask = condition_idx == index
        summary = summarize_parameter_estimator(
            np.asarray(predictions["predicted_range_cell"])[mask],
            np.asarray(predictions["truth_range_cell"])[mask],
            np.asarray(predictions["predicted_doppler_hz"])[mask],
            np.asarray(predictions["truth_doppler_hz"])[mask],
            detected[mask],
            prf_hz=1300.0,
            range_gate_cells=1.0,
            doppler_gate_hz=20.0,
        )
        if threshold is not None:
            successes = int(summary["joint_success_count"])
            summary["joint_success_ci95_exact"] = _exact_two_sided(successes, int(mask.sum()))
            summary["pd"] = float(detected[mask].mean())
            summary["pd_ci95_exact"] = _exact_two_sided(int(detected[mask].sum()), int(mask.sum()))
        if threshold is None:
            summary = {
                "n": int(mask.sum()),
                "all_h1": summary["all_h1"],
                "detection_metrics": "undefined_for_parameter_only_arm",
            }
        result["by_condition"][str(index)] = {"condition": dict(condition), **summary}
    return result


def _save_prediction_archive(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _paired_quality_bootstrap(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
    *,
    threshold_first: float,
    threshold_second: float,
    conditions: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    ids_first = np.asarray(first["scene_id"]).astype(str)
    ids_second = np.asarray(second["scene_id"]).astype(str)
    if not np.array_equal(ids_first, ids_second):
        raise ValueError("R6-D paired H1 arms do not share identical scene IDs and order")
    condition_index = np.asarray(first["condition_index"], dtype=np.int64)
    if not np.array_equal(condition_index, np.asarray(second["condition_index"], dtype=np.int64)):
        raise ValueError("R6-D paired H1 arms do not share condition assignments")
    courses = np.asarray([str(value["course"]) for value in conditions])
    mask = np.isin(courses[condition_index], ("C2", "C3"))

    def task_outputs(prediction: Mapping[str, np.ndarray], threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        detected = np.asarray(prediction["score"]) >= threshold
        range_ok = np.abs(np.asarray(prediction["predicted_range_cell"]) -
                          np.asarray(prediction["truth_range_cell"])) <= 1.0
        doppler_err = wrapped_error(prediction["predicted_doppler_hz"], prediction["truth_doppler_hz"], 1300.0)
        joint_success = detected & range_ok & (np.abs(doppler_err) <= 20.0)
        return detected, joint_success, np.asarray(doppler_err, dtype=np.float64)

    detected_first, joint_first, doppler_error_first = task_outputs(first, threshold_first)
    detected_second, joint_second, doppler_error_second = task_outputs(second, threshold_second)
    range_error_first = np.asarray(first["predicted_range_cell"], dtype=np.float64) - np.asarray(first["truth_range_cell"], dtype=np.float64)
    range_error_second = np.asarray(second["predicted_range_cell"], dtype=np.float64) - np.asarray(second["truth_range_cell"], dtype=np.float64)
    joint_delta = joint_second[mask].astype(np.float64) - joint_first[mask].astype(np.float64)
    if not joint_delta.size:
        raise ValueError("R6-D primary C2+C3 paired comparison has no scenes")
    rng = np.random.default_rng(seed)
    joint_samples = np.empty(replicates, dtype=np.float64)
    pd_samples = np.empty(replicates, dtype=np.float64)
    range_ratio_samples = np.empty(replicates, dtype=np.float64)
    doppler_ratio_samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        all_indices = rng.integers(0, len(ids_first), size=len(ids_first))
        joint_indices = rng.integers(0, len(joint_delta), size=len(joint_delta))
        joint_samples[replicate] = np.mean(joint_delta[joint_indices])
        pd_samples[replicate] = np.mean(
            detected_second[all_indices].astype(np.float64) - detected_first[all_indices].astype(np.float64)
        )
        range_ratio_samples[replicate] = math.sqrt(
            np.mean(range_error_second[all_indices] ** 2) / max(np.mean(range_error_first[all_indices] ** 2), 1e-24)
        )
        doppler_ratio_samples[replicate] = math.sqrt(
            np.mean(doppler_error_second[all_indices] ** 2) / max(np.mean(doppler_error_first[all_indices] ** 2), 1e-24)
        )
    joint_ci = [float(x) for x in np.quantile(joint_samples, [0.025, 0.975])]
    pd_difference = float(np.mean(detected_second.astype(np.float64) - detected_first.astype(np.float64)))
    pd_ci = [float(x) for x in np.quantile(pd_samples, [0.025, 0.975])]
    range_ratio = float(math.sqrt(np.mean(range_error_second ** 2) / max(np.mean(range_error_first ** 2), 1e-24)))
    range_ci = [float(x) for x in np.quantile(range_ratio_samples, [0.025, 0.975])]
    doppler_ratio = float(math.sqrt(np.mean(doppler_error_second ** 2) / max(np.mean(doppler_error_first ** 2), 1e-24)))
    doppler_ci = [float(x) for x in np.quantile(doppler_ratio_samples, [0.025, 0.975])]
    return {
        "contrast": "j2_joint_minus_j1_joint",
        "group": "identifiable_C2_C3",
        "n_paired_scenes": int(len(joint_delta)),
        "difference": float(np.mean(joint_delta)),
        "paired_bootstrap_ci95": joint_ci,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "meaningful_gain_threshold": 0.02,
        "screening_pass": bool(np.mean(joint_delta) >= 0.02 and joint_ci[0] > 0.0),
        "overall_pd_difference_j2_minus_j1": {
            "n_paired_scenes": int(len(ids_first)),
            "difference": pd_difference,
            "paired_bootstrap_ci95": pd_ci,
            "protection_lower_minimum": -0.02,
            "protection_pass": bool(pd_ci[0] >= -0.02),
        },
        "all_h1_range_rmse_ratio_j2_over_j1": {
            "ratio": range_ratio,
            "paired_bootstrap_ci95": range_ci,
            "protection_upper_maximum": 1.05,
            "protection_pass": bool(range_ci[1] <= 1.05),
        },
        "all_h1_doppler_rmse_ratio_j2_over_j1": {
            "ratio": doppler_ratio,
            "paired_bootstrap_ci95": doppler_ci,
            "protection_upper_maximum": 1.05,
            "protection_pass": bool(doppler_ci[1] <= 1.05),
        },
    }


def run_evaluations() -> dict[str, Any]:
    """Freeze checkpoints first, then stream independent calibration/verification/development scenes."""
    preflight_record, head = _require_preflight()
    config, r6a, physics, _ = _configs()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-D evaluation requires G1_PASS")
    completed = {}
    for arm in R6D_ARMS:
        record_path = RUNS / arm / "training_record.json"
        if not record_path.is_file():
            raise RuntimeError(f"R6-D arm {arm} has not completed its fixed update budget")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or int(record.get("updates", -1)) != int(config["training"]["fixed_updates"]):
            raise RuntimeError(f"R6-D arm {arm} did not finish the frozen budget")
        completed[arm] = record

    conditions = r6a["radar"]["conditions"]
    plans = _eval_plans(config, r6a)
    device = torch.device("cuda:0")
    microbatch = int(g1["formal_microbatch"])
    thresholds: dict[str, float] = {}
    h1_by_arm: dict[str, dict[str, np.ndarray]] = {}
    arm_metrics: dict[str, Any] = {}
    for arm in R6D_ARMS:
        if arm == "j1_parameter":
            cal = None
            ver = None
            threshold = None
            h0_summary = None
            h0_seconds = None
        else:
            cal_plan = plans["h0_calibration"]
            ver_plan = plans["h0_verification"]
            model = build_r6d_student(
                arm, width=int(config["models"]["width"]),
                seed=int(config["training"]["training_seed"]),
                prf_hz=float(config["radar"]["prf_hz"]),
            ).to(device)
            checkpoint = torch.load(RUNS / arm / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            expected_owner = _training_owner(arm, config, head)
            if checkpoint.get("owner") != expected_owner:
                raise RuntimeError(f"R6-D {arm} best checkpoint owner mismatch")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            h0_timer = time.perf_counter()
            cal = _predict_eval_split(
                model, "h0_calibration", cal_plan, physics, device=device, microbatch=microbatch,
                prf_hz=float(config["radar"]["prf_hz"]),
            )
            threshold = float(calibrate_threshold(cal["score"], float(config["evaluation"]["target_pfa_cpi"]))["threshold"])
            ver = _predict_eval_split(
                model, "h0_verification", ver_plan, physics, device=device, microbatch=microbatch,
                prf_hz=float(config["radar"]["prf_hz"]),
            )
            h0_summary = _h0_metrics(cal, ver, threshold=threshold, conditions=conditions)
            h0_seconds = time.perf_counter() - h0_timer
            thresholds[arm] = threshold
            del model, checkpoint
            torch.cuda.empty_cache()
        model = build_r6d_student(
            arm, width=int(config["models"]["width"]),
            seed=int(config["training"]["training_seed"]),
            prf_hz=float(config["radar"]["prf_hz"]),
        ).to(device)
        checkpoint = torch.load(RUNS / arm / "best_checkpoint.pt", map_location="cpu", weights_only=False)
        if checkpoint.get("owner") != _training_owner(arm, config, head):
            raise RuntimeError(f"R6-D {arm} best checkpoint owner mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        timer = time.perf_counter()
        h1 = _predict_eval_split(
            model, "h1_development", plans["h1_development"], physics, device=device,
            microbatch=microbatch, prf_hz=float(config["radar"]["prf_hz"]),
            estimate_parameters=arm != "j1_detection",
            estimate_detection=arm != "j1_parameter",
        )
        h1_seconds = time.perf_counter() - timer
        h1_by_arm[arm] = h1
        h1_summary = _h1_metrics(
            h1,
            threshold=threshold,
            conditions=conditions,
            estimate_parameters=arm != "j1_detection",
        )
        if ver is not None:
            h1_summary["roc_auc_vs_h0_verification"] = float(empirical_auc(ver["score"], h1["score"]))
            h1_summary["roc_auc_by_condition"] = {
                str(index): float(empirical_auc(
                    ver["score"][ver["condition_index"] == index],
                    h1["score"][h1["condition_index"] == index],
                ))
                for index in range(len(conditions))
            }
        output_dir = PREDICTIONS / arm
        _save_prediction_archive(output_dir / "h1_development.npz", **h1)
        if cal is not None and ver is not None:
            _save_prediction_archive(output_dir / "h0_calibration.npz", **cal)
            _save_prediction_archive(output_dir / "h0_verification.npz", **ver)
        arm_metrics[arm] = {
            "training": completed[arm],
            "h0": h0_summary,
            "h1_development": h1_summary,
            "threshold": threshold,
            "h0_calibration_and_verification_seconds": h0_seconds,
            "h1_development_seconds": h1_seconds,
            "evaluation_seconds_total": None if h0_seconds is None else h0_seconds + h1_seconds,
            "checkpoint_update": int(checkpoint["update"]),
        }
        print(json.dumps({"stage": "evaluation_complete", "arm": arm,
                          "threshold": threshold, "h1_joint": h1_summary.get("by_group", {}).get("C2_C3", {}).get("joint_success"),
                          "h0_seconds": h0_seconds, "h1_seconds": h1_seconds}, ensure_ascii=False), flush=True)
        del model, checkpoint
        torch.cuda.empty_cache()

    paired = _paired_quality_bootstrap(
        h1_by_arm["j1_joint"],
        h1_by_arm["j2_joint"],
        threshold_first=thresholds["j1_joint"],
        threshold_second=thresholds["j2_joint"],
        conditions=conditions,
        replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
        seed=int(config["evaluation"]["paired_bootstrap_seed"]),
    )
    result = {
        "status": "development_evaluation_complete",
        "git_sha": head,
        "protocol_sha256": preflight_record["protocol_sha256"],
        "config_sha256": preflight_record["config_sha256"],
        "protocol_version": float(config["experiment"]["protocol_version"]),
        "development_only": True,
        "final_blind_generated": False,
        "primary_contrast": paired,
        "arms": arm_metrics,
        "independent_scene_counts": {
            name: sum(int(row["count"]) for row in plan)
            for name, plan in plans.items()
        },
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "microbatch": microbatch,
    }
    _write_json(STAGE / "metrics.json", result)
    return result


def _cache_cleanup() -> dict[str, Any]:
    _, head = _require_preflight()
    config, r6a, _, _ = _configs()
    required_reports = (STAGE / "metrics.json", STAGE / "analysis.md", ROOT / "docs/stage_reviews/D2-A-R6D_review.md")
    missing = [str(path.relative_to(ROOT)) for path in required_reports if not path.is_file()]
    if missing:
        raise RuntimeError(f"refusing R6-D cleanup before reports exist: {missing}")
    removed = []
    for split in ("train", "validation"):
        plan = _all_split_plans(config, r6a)[split]
        expected_owner = _cache_owner(split, head, plan)
        directory = CACHE / split
        if not directory.exists():
            continue
        owner_path = directory / "owner.json"
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != expected_owner:
            raise RuntimeError(f"refusing to remove unowned R6-D cache: {directory}")
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        if progress.get("owner") != expected_owner or int(progress.get("rows_written", -1)) != sum(
            int(row["count"]) for row in plan
        ):
            raise RuntimeError(f"refusing to remove incomplete R6-D cache: {directory}")
        byte_count = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
        removed.append({"split": split, "path": str(directory.relative_to(ROOT)), "bytes": int(byte_count)})
    usage = shutil.disk_usage(ROOT)
    result = {
        "status": "complete",
        "owner": "D2-A-R6-D-paired-cache",
        "git_sha": head,
        "removed": removed,
        "removed_bytes": int(sum(item["bytes"] for item in removed)),
        "workspace_used_bytes_after_cleanup": int(usage.total - usage.free),
        "workspace_limit_bytes": int(float(config["resources"]["project_final_limit_gib"]) * (1024 ** 3)),
    }
    _write_json(STAGE / "cleanup.json", result)
    return result


def _valid_saved_preflight() -> bool:
    path = STAGE / "preflight.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return (
            record.get("git_sha") == _git("rev-parse", "HEAD")
            and record.get("protocol_sha256") == _sha256(STAGE / "protocol.md")
            and record.get("config_sha256") == _sha256(STAGE / "config.yaml")
        )
    except (OSError, json.JSONDecodeError, subprocess.CalledProcessError):
        return False


def run_all() -> dict[str, Any]:
    if not _valid_saved_preflight():
        print(json.dumps(preflight(), ensure_ascii=False), flush=True)
    _require_preflight()
    g1_path = STAGE / "g1.json"
    g1 = json.loads(g1_path.read_text(encoding="utf-8")) if g1_path.exists() else None
    if g1 is None or g1.get("status") != "G1_PASS":
        g1 = run_g1()
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-D G1 failed; formal data/training was not started")
    for split in ("train", "validation"):
        directory = CACHE / split
        progress_path = directory / "progress.json"
        if not progress_path.is_file():
            print(json.dumps(generate_paired_cache(split), ensure_ascii=False), flush=True)
        else:
            arrays = _load_pair_cache(split)
            print(json.dumps({"stage": "cache_ready", "split": split, "rows": len(arrays["scene_id"])}), flush=True)
    for arm in R6D_ARMS:
        record_path = RUNS / arm / "training_record.json"
        if record_path.is_file():
            saved = json.loads(record_path.read_text(encoding="utf-8"))
            if saved.get("status") == "complete" and int(saved.get("updates", 0)) == 25_600:
                print(json.dumps({"stage": "training_already_complete", "arm": arm}, ensure_ascii=False), flush=True)
                continue
        print(json.dumps(train_arm(arm), ensure_ascii=False), flush=True)
    return run_evaluations()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "g1", "generate", "train", "train-all", "evaluate", "all", "cleanup"))
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--arm", choices=R6D_ARMS)
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        result = preflight()
    elif arguments.command == "g1":
        result = run_g1()
    elif arguments.command == "generate":
        if arguments.split is None:
            parser.error("generate requires --split")
        result = generate_paired_cache(arguments.split)
    elif arguments.command == "train":
        if arguments.arm is None:
            parser.error("train requires --arm")
        result = train_arm(arguments.arm)
    elif arguments.command == "train-all":
        result = {arm: train_arm(arm) for arm in R6D_ARMS}
    elif arguments.command == "evaluate":
        result = run_evaluations()
    elif arguments.command == "cleanup":
        result = _cache_cleanup()
    else:
        result = run_all()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
