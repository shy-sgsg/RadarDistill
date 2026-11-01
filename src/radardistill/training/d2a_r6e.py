"""Data, G0/G1 gates, and matched CUDA training for D2-A-R6-E."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import yaml

from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.evaluation.d2a_r6a import build_split_plan
from radardistill.students.r6_architectures import (
    R6E_ARMS,
    build_r6e_student,
    count_r6e_parameters,
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


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-E"
CACHE = STAGE / "cache/training"
RUNS = STAGE / "runs/seed_606"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"
R6E_CODE = Path(__file__).resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read_yaml(STAGE / "config.yaml")
    r6a = _read_yaml(ROOT / config["radar"]["condition_source"])
    physics = _read_yaml(ROOT / config["radar"]["source_config"])
    return config, r6a, physics


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _all_split_plans(
    config: Mapping[str, Any], r6a: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    conditions = r6a["radar"]["conditions"]
    data = config["data"]
    plans = {
        "train": build_r6d_paired_plan(data, conditions, "train"),
        "validation": build_r6d_paired_plan(data, conditions, "validation"),
    }
    evaluation_data = {
        key: data[key]
        for key in ("condition_seed_stride", "chunk_size", "chunk_seed_stride")
    }
    evaluation_data["condition_count"] = int(config["radar"]["condition_count"])
    eval_names = ("h0_calibration", "h0_verification", "h1_development")
    evaluation_data["splits"] = {name: data["splits"][name] for name in eval_names}
    evaluation_config = {"data": evaluation_data, "radar": {"conditions": conditions}}
    plans.update({name: build_split_plan(evaluation_config, name) for name in eval_names})
    plans["g1_test"] = [{"seed": int(data["g1_test_seed"]), "count": 16}]
    return plans


def _prior_scene_ids() -> set[str]:
    result: set[str] = set()
    for path in sorted((ROOT / "experiments").rglob("*.csv")):
        if "manifest" not in path.name and path.name != "selected_scene_ids.csv":
            continue
        if path == STAGE or STAGE in path.parents:
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            key = next((name for name in ("scene_id", "base_scene_id", "base_scene_ids") if name in fields), None)
            if key is None:
                continue
            for row in reader:
                value = row.get(key, "")
                if value:
                    result.update(item for item in str(value).split(";") if item)
    return result


def _scene_counts(plans: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    return {name: sum(int(row["count"]) for row in rows) for name, rows in plans.items()}


def _source_sha256() -> dict[str, str]:
    return {
        "trainer": _sha256(R6E_CODE),
        "architectures": _sha256(ROOT / "src/radardistill/students/r6_architectures.py"),
        "simulator": _sha256(ROOT / "src/radardistill/simulation/d2a.py"),
        "paired_plan": _sha256(ROOT / "src/radardistill/training/d2a_r6d_joint.py"),
        "sampler": _sha256(ROOT / "src/radardistill/training/d2a_r6b_hard_negative.py"),
        "loss": _sha256(ROOT / "src/radardistill/distillation/task_losses.py"),
        "evaluation": _sha256(ROOT / "src/radardistill/evaluation/d2a_r6e.py"),
        "entrypoint": _sha256(ROOT / "scripts/run_d2a_r6e_architecture.py"),
    }


def _cache_owner(split: str, plan: Sequence[Mapping[str, Any]], git_sha: str) -> dict[str, Any]:
    config, _, _ = _configs()
    plan_hash = hashlib.sha256(
        json.dumps(list(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "owner": "D2-A-R6-E-paired-cache",
        "git_sha": git_sha,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_sha256(),
        "split": split,
        "scene_count": sum(int(row["count"]) for row in plan),
        "plan_sha256": plan_hash,
        "source_config": config["radar"]["source_config"],
    }


def _owner_matches(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
        return False
    expected_identity = {key: value for key, value in expected.items() if key != "git_sha"}
    actual_identity = {key: value for key, value in actual.items() if key != "git_sha"}
    return expected_identity == actual_identity


def _cache_paths(split: str) -> dict[str, Path]:
    directory = CACHE / split
    names = ("iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id")
    return {name: directory / f"{name}.npy" for name in names}


def _bytes_per_paired_scene() -> int:
    return int(
        2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
        + 2 * np.dtype(np.float32).itemsize
        + np.dtype(np.uint8).itemsize
        + 48 * np.dtype("<U1").itemsize
    )


def _remaining_owned_cache_bytes(
    config: Mapping[str, Any], r6a: Mapping[str, Any], git_sha: str
) -> int:
    plans = _all_split_plans(config, r6a)
    remaining = 0
    for split in ("train", "validation"):
        count = _scene_counts({split: plans[split]})[split]
        directory = CACHE / split
        written = 0
        if directory.exists():
            owner_path, progress_path = directory / "owner.json", directory / "progress.json"
            if not owner_path.is_file() or not progress_path.is_file():
                raise RuntimeError(f"refusing to reuse unowned R6-E cache: {directory}")
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if not _owner_matches(_cache_owner(split, plans[split], git_sha), owner):
                raise RuntimeError(f"R6-E cache owner differs from registered plan: {directory}")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("owner") != owner:
                raise RuntimeError(f"R6-E cache progress owner mismatch: {directory}")
            written = int(progress.get("rows_written", -1))
            if not 0 <= written <= count:
                raise RuntimeError(f"R6-E cache progress is invalid: {directory}")
        remaining += (count - written) * _bytes_per_paired_scene()
    return int(remaining + 2 * 1024 * 1024)


def _require_cuda() -> tuple[torch.device, str, int, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-E requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    device_name = torch.cuda.get_device_name(0)
    if device_name != DEVICE_NAME:
        raise RuntimeError(f"R6-E requires {DEVICE_NAME}, found {device_name}")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    config, _, _ = _configs()
    minimum = int(config["resources"].get("minimum_free_vram_bytes", 0))
    if free_vram < minimum:
        raise RuntimeError(f"R6-E needs {minimum} free GPU bytes, found {free_vram}")
    return torch.device("cuda:0"), device_name, int(free_vram), int(total_vram)


def r6e_preflight() -> dict[str, Any]:
    """G0: verify frozen pushed code/config, fresh scene IDs, disk, and the actual GPU."""
    config, r6a, _ = _configs()
    head = _git("rev-parse", "HEAD")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-E G0 requires the frozen implementation pushed to main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R6-E G0 requires a clean tracked worktree")
    if config["experiment"]["protocol_version"] != 1.0 or config["experiment"]["status"] != "preregistered":
        raise RuntimeError("R6-E protocol/config is not frozen as registered")
    if config["experiment"]["final_blind_authorized"] is not False:
        raise RuntimeError("R6-E final blind is not authorized")

    plans = _all_split_plans(config, r6a)
    expected = _scene_counts(plans)
    prior = _prior_scene_ids()
    scene_audit = audit_r6d_scene_plans(
        plans, expected_counts=expected, prior_scene_ids=prior
    )
    total_train_val = expected["train"] + expected["validation"]
    estimated_bytes = total_train_val * _bytes_per_paired_scene()
    disk = shutil.disk_usage(STAGE)
    remaining_cache = _remaining_owned_cache_bytes(config, r6a, head)
    reserve = 4 * 1024**3
    minimum_free = max(
        remaining_cache + reserve,
        int(config["data"]["minimum_free_disk_bytes_before_generation"]),
    )
    if disk.free < minimum_free:
        raise RuntimeError(f"R6-E needs {minimum_free} free bytes before cache generation; found {disk.free}")

    device, gpu, free_vram, total_vram = _require_cuda()
    parameter_counts = {}
    for arm in R6E_ARMS:
        model = build_r6e_student(arm, seed=int(config["data"]["training_seed"]))
        parameter_counts[arm] = count_r6e_parameters(model)
        del model
    baseline = int(parameter_counts["j1_joint"]["parameters"])
    tolerance = float(config["models"]["parameter_match_relative_tolerance"])
    parameter_bounds = [math.ceil(baseline * (1 - tolerance)), math.floor(baseline * (1 + tolerance))]
    for arm in ("ce_tcn", "rax_transformer"):
        count = int(parameter_counts[arm]["parameters"])
        if not parameter_bounds[0] <= count <= parameter_bounds[1]:
            raise RuntimeError(f"R6-E {arm} parameter count is outside the frozen match interval")

    driver = "unavailable"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    record = {
        "status": "G0_PASS",
        "git_sha": head,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_sha256(),
        "protocol_version": float(config["experiment"]["protocol_version"]),
        "split_scene_counts": scene_audit["scene_counts"],
        "historical_scene_id_count": int(len(prior)),
        "historical_scene_id_overlap": int(scene_audit["prior_overlap_count"]),
        "cross_split_scene_id_overlap": int(scene_audit["overlap_count"]),
        "parameter_count_bounds": parameter_bounds,
        "model_parameter_counts": parameter_counts,
        "estimated_train_validation_cache_bytes": estimated_bytes,
        "remaining_owned_cache_bytes": remaining_cache,
        "disk_free_before_generation_bytes": int(disk.free),
        "disk_total_bytes": int(disk.total),
        "minimum_free_bytes": minimum_free,
        "gpu": gpu,
        "device": str(device),
        "driver_version": driver,
        "free_vram_before_g1_bytes": free_vram,
        "total_vram_bytes": total_vram,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "formal_data_generated": False,
        "final_blind_generated": False,
    }
    _write_json(STAGE / "preflight.json", record)
    return record


def _require_preflight() -> tuple[dict[str, Any], str]:
    path = STAGE / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run R6-E preflight before G1 or formal data generation")
    record = json.loads(path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    if (
        record.get("status") != "G0_PASS"
        or record.get("git_sha") != head
        or record.get("protocol_sha256") != _sha256(STAGE / "protocol.md")
        or record.get("config_sha256") != _sha256(STAGE / "config.yaml")
        or record.get("source_sha256") != _source_sha256()
    ):
        raise RuntimeError("R6-E preflight does not match the current code/protocol/config")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-E requires its frozen implementation pushed to main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked source changed after R6-E G0")
    _require_cuda()
    return record, head


def _paired_iq_and_truth(generated: Any) -> dict[str, np.ndarray]:
    count = len(generated.base_scene_id)
    iq_h0 = np.asarray(generated.iq_h0, dtype=np.complex64)
    iq_h1 = np.asarray(generated.iq_h1, dtype=np.complex64)
    if iq_h0.shape != (count, 3, 130, 32) or iq_h1.shape != iq_h0.shape:
        raise ValueError("R6-E paired generator returned an unexpected IQ shape")
    return {
        "iq_h0": iq_h0,
        "iq_h1": iq_h1,
        "truth_range_cell": np.asarray(generated.truth_target_range_cell, dtype=np.float32),
        "truth_doppler_hz": np.asarray(generated.truth_target_doppler_hz, dtype=np.float32),
        "scene_id": np.asarray(generated.base_scene_id).astype("<U48"),
    }


def _loss_on_device(
    model: torch.nn.Module,
    batch: Mapping[str, np.ndarray],
    *,
    microbatch: int,
    device: torch.device,
    gradients: bool,
) -> dict[str, torch.Tensor]:
    inputs = torch.from_numpy(batch["iq_ri"])
    labels = {
        key: torch.from_numpy(np.asarray(batch[key], dtype=np.float32))
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    totals: dict[str, torch.Tensor] = {}
    context = torch.enable_grad() if gradients else torch.inference_mode()
    with context:
        for start in range(0, len(inputs), microbatch):
            stop = min(start + microbatch, len(inputs))
            x = inputs[start:stop].to(device)
            truth = {key: value[start:stop].to(device) for key, value in labels.items()}
            outputs = model(x)
            terms = r6d_loss(outputs, truth, arm="j1_joint", prf_hz=1300.0)
            scale = (stop - start) / len(inputs)
            for key, value in terms.items():
                totals[key] = totals.get(key, value.detach() * 0.0) + value * scale
    return totals


def _training_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Mapping[str, np.ndarray],
    *,
    microbatch: int,
    effective_batch: int,
    device: torch.device,
    gradient_clip: float,
) -> dict[str, float]:
    if len(batch["iq_ri"]) != effective_batch or effective_batch % microbatch:
        raise ValueError("microbatch must divide the registered effective batch")
    inputs = torch.from_numpy(batch["iq_ri"])
    labels = {
        key: torch.from_numpy(np.asarray(batch[key], dtype=np.float32))
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("total", "presence", "range_cell", "range_offset", "doppler", "candidate")}
    for start in range(0, effective_batch, microbatch):
        stop = start + microbatch
        x = inputs[start:stop].to(device)
        truth = {key: value[start:stop].to(device) for key, value in labels.items()}
        terms = r6d_loss(model(x), truth, arm="j1_joint", prf_hz=1300.0)
        weight = microbatch / effective_batch
        loss = terms["total"] * weight
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite R6-E training loss")
        loss.backward()
        for key in totals:
            totals[key] += float(terms[key].detach().item()) * weight
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError("nonfinite R6-E gradient norm")
    optimizer.step()
    return totals


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
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _paired_cache_paths(split: str) -> dict[str, Path]:
    return _cache_paths(split)


def _load_pair_cache(split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("R6-E paired cache split must be train or validation")
    config, r6a, _ = _configs()
    plan = _all_split_plans(config, r6a)[split]
    expected_count = sum(int(row["count"]) for row in plan)
    git_sha = _git("rev-parse", "HEAD")
    expected_owner = _cache_owner(split, plan, git_sha)
    directory = CACHE / split
    owner_path, progress_path = directory / "owner.json", directory / "progress.json"
    if not owner_path.is_file() or not progress_path.is_file():
        raise RuntimeError(f"R6-E {split} cache owner/progress record is missing")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if not _owner_matches(expected_owner, owner):
        raise RuntimeError(f"R6-E {split} cache does not match the registered data identity")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("owner") != owner or progress.get("rows_written") != expected_count:
        raise RuntimeError(f"R6-E {split} cache is incomplete")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in _paired_cache_paths(split).items()}
    expected_shapes = {
        "iq_h0": (expected_count, 3, 130, 32),
        "iq_h1": (expected_count, 3, 130, 32),
        "truth_range_cell": (expected_count,),
        "truth_doppler_hz": (expected_count,),
        "condition_index": (expected_count,),
        "scene_id": (expected_count,),
    }
    expected_dtypes = {
        "iq_h0": np.dtype(np.complex64),
        "iq_h1": np.dtype(np.complex64),
        "truth_range_cell": np.dtype(np.float32),
        "truth_doppler_hz": np.dtype(np.float32),
        "condition_index": np.dtype(np.uint8),
        "scene_id": np.dtype("<U48"),
    }
    if any(arrays[name].shape != expected_shapes[name] or arrays[name].dtype != expected_dtypes[name]
           for name in arrays):
        raise RuntimeError(f"R6-E {split} cache shape/dtype differs from the registered plan")
    if not _cache_manifest_matches(split, plan, arrays):
        raise RuntimeError(f"R6-E {split} cache scene IDs differ from the frozen plan/manifest")
    return arrays


def _cache_manifest_matches(split: str, plan: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> bool:
    manifest = STAGE / f"scene_manifest_{split}.csv"
    if not manifest.is_file():
        return False
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_count = sum(int(item["count"]) for item in plan)
    if len(rows) != expected_count or len(arrays["scene_id"]) != expected_count:
        return False
    manifest_ids = np.asarray([row["scene_id"] for row in rows], dtype=str)
    cached_ids = np.asarray(arrays["scene_id"]).astype(str)
    plan_ids = np.asarray([
        f"d2a-scene-{int(item['seed'])}-{index}"
        for item in plan
        for index in range(int(item["count"]))
    ], dtype=str)
    return np.array_equal(manifest_ids, cached_ids) and np.array_equal(cached_ids, plan_ids)


def generate_paired_cache(split: str) -> dict[str, Any]:
    preflight, git_sha = _require_preflight()
    g1_path = STAGE / "g1.json"
    if not g1_path.is_file() or json.loads(g1_path.read_text(encoding="utf-8")).get("status") != "G1_PASS":
        raise RuntimeError("R6-E paired data generation requires G1_PASS")
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation are persisted")
    config, r6a, physics = _configs()
    plan = _all_split_plans(config, r6a)[split]
    directory = CACHE / split
    expected_owner = _cache_owner(split, plan, git_sha)
    count = sum(int(row["count"]) for row in plan)
    free = int(shutil.disk_usage(STAGE).free)
    remaining = _remaining_owned_cache_bytes(config, r6a, git_sha)
    reserve = 4 * 1024**3
    if free < remaining + reserve:
        raise RuntimeError(f"R6-E cache generation needs {remaining + reserve} free bytes; found {free}")

    if directory.exists():
        owner_path = directory / "owner.json"
        if not owner_path.is_file() or not _owner_matches(
            expected_owner, json.loads(owner_path.read_text(encoding="utf-8"))
        ):
            raise RuntimeError(f"refusing to reuse an unowned R6-E cache: {directory}")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
    else:
        directory.mkdir(parents=True)
        owner = expected_owner
        _write_json(directory / "owner.json", owner)

    paths = _paired_cache_paths(split)
    progress_path = directory / "progress.json"
    manifest_path = STAGE / f"scene_manifest_{split}.csv"
    shapes = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "iq_h1": ((count, 3, 130, 32), np.complex64),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
        "condition_index": ((count,), np.uint8),
        "scene_id": ((count,), "<U48"),
    }
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("R6-E progress owner mismatch")
        completed = int(progress["completed_chunks"])
        offset = int(progress["rows_written"])
        expected_offset = sum(int(row["count"]) for row in plan[:completed])
        if not 0 <= completed <= len(plan) or offset != expected_offset:
            raise RuntimeError("R6-E cache cursor differs from the frozen plan")
        if manifest_path.exists():
            with manifest_path.open("r+b") as handle:
                handle.truncate(int(progress["manifest_offset"]))
        elif offset:
            raise RuntimeError("R6-E scene manifest is missing for a nonempty cache")
    else:
        if directory.exists() and any(path.name != "owner.json" for path in directory.iterdir()):
            raise RuntimeError("R6-E partial cache has no valid progress record")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite unowned R6-E manifest: {manifest_path}")
        completed = offset = 0
        progress = {"owner": owner, "completed_chunks": 0, "rows_written": 0, "manifest_offset": 0}
        _write_json(progress_path, progress)

    for name, (shape, dtype) in shapes.items():
        if not paths[name].is_file():
            np.lib.format.open_memmap(paths[name], mode="w+", dtype=dtype, shape=shape).flush()
        else:
            current = np.load(paths[name], mmap_mode="r")
            if current.shape != shape or current.dtype != np.dtype(dtype):
                raise RuntimeError(f"R6-E cache shape/dtype mismatch for {name}")
    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    start_offset = offset
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if handle.tell() == 0:
            writer.writerow(("scene_id", "split", "seed", "condition_index", "course", "scenario_id",
                             "snr_db", "cnr_db", "truth_range_cell", "truth_doppler_hz"))
            handle.flush()
            progress["manifest_offset"] = handle.tell()
            _write_json(progress_path, progress)
        offset = start_offset
        for chunk_index, spec in enumerate(plan[completed:], start=completed):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            stop = offset + int(spec["count"])
            arrays["iq_h0"][offset:stop] = generated.iq_h0
            arrays["iq_h1"][offset:stop] = generated.iq_h1
            arrays["truth_range_cell"][offset:stop] = np.asarray(generated.truth_target_range_cell, dtype=np.float32)
            arrays["truth_doppler_hz"][offset:stop] = np.asarray(generated.truth_target_doppler_hz, dtype=np.float32)
            arrays["condition_index"][offset:stop] = int(spec["condition_index"])
            arrays["scene_id"][offset:stop] = np.asarray(generated.base_scene_id).astype("<U48")
            for index, scene_id in enumerate(generated.base_scene_id):
                writer.writerow((
                    str(scene_id), split, int(spec["seed"]), int(spec["condition_index"]), spec["course"],
                    condition["scenario_id"], repr(float(generated.snr_db[index])),
                    repr(float(generated.requested_cnr_db[index])),
                    repr(float(generated.truth_target_range_cell[index])),
                    repr(float(generated.truth_target_doppler_hz[index])),
                ))
            handle.flush()
            for array in arrays.values():
                array.flush()
            offset = stop
            progress = {
                "owner": owner,
                "completed_chunks": chunk_index + 1,
                "rows_written": offset,
                "manifest_offset": handle.tell(),
            }
            _write_json(progress_path, progress)
            if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
                print(json.dumps({"stage": "generate", "split": split,
                                  "chunks_done": chunk_index + 1, "chunks_total": len(plan),
                                  "scenes_done": offset}, ensure_ascii=False), flush=True)
    del arrays
    arrays = _load_pair_cache(split)
    if not _cache_manifest_matches(split, plan, arrays):
        raise RuntimeError(f"R6-E {split} manifest does not match its cached scene IDs")
    return {
        "split": split,
        "complete": True,
        "scenes": count,
        "chunks": len(plan),
        "cache_bytes": sum(path.stat().st_size for path in paths.values()),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "preflight_git_sha": preflight["git_sha"],
        "cache_generator_git_sha": owner["git_sha"],
    }


def _g1_batch(
    data: Mapping[str, np.ndarray], h0_indices: np.ndarray, h1_indices: np.ndarray
) -> dict[str, np.ndarray]:
    return assemble_independent_h0_h1_batch(data, h0_indices, h1_indices)


def _probe_common_microbatch(device: torch.device, seed: int) -> tuple[int, dict[str, Any]]:
    reports: dict[str, Any] = {}
    for microbatch in (16, 8, 4):
        feasible = True
        pass_reports: dict[str, Any] = {}
        for arm in R6E_ARMS:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = build_r6e_student(arm, seed=seed).to(device)
            try:
                x = torch.randn(microbatch, 6, 130, 32, device=device)
                output = model(x)
                loss = sum(value.square().mean() for value in output.values())
                loss.backward()
                if not torch.isfinite(loss) or any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError(f"R6-E {arm} resource probe produced invalid gradients")
                pass_reports[arm] = {
                    "status": "pass",
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            except torch.cuda.OutOfMemoryError:
                feasible = False
                pass_reports[arm] = {"status": "oom"}
            finally:
                del model
                torch.cuda.empty_cache()
        reports[str(microbatch)] = pass_reports
        if feasible:
            return microbatch, reports
    raise RuntimeError("R6-E cannot run all models at microbatch 4 while preserving effective batch 64")


def _evaluate_batch_loss(
    model: torch.nn.Module,
    batch: Mapping[str, np.ndarray],
    *,
    microbatch: int,
    device: torch.device,
) -> dict[str, float]:
    terms = _loss_on_device(model, batch, microbatch=microbatch, device=device, gradients=False)
    return {key: float(value.item()) for key, value in terms.items()}


def _overfit_g1_arm(
    arm: str,
    data: Mapping[str, np.ndarray],
    pair_plan: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    microbatch: int,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model = build_r6e_student(arm, seed=int(config["data"]["training_seed"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate_initial"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    first_batch = _g1_batch(data, *pair_plan[0])
    initial = _evaluate_batch_loss(model, first_batch, microbatch=microbatch, device=device)["total"]
    recent: list[float] = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for h0_indices, h1_indices in pair_plan:
        batch = _g1_batch(data, h0_indices, h1_indices)
        _training_update(
            model, optimizer, batch, microbatch=microbatch,
            effective_batch=int(config["training"]["effective_batch_size_cpi"]),
            device=device, gradient_clip=float(config["training"]["gradient_clip_norm"]),
        )
        recent.append(_evaluate_batch_loss(model, batch, microbatch=microbatch, device=device)["total"])
    last_mean = float(np.mean(recent[-20:]))
    reduction = (initial - last_mean) / max(abs(initial), 1e-12)
    report = {
        "initial_loss": float(initial),
        "last20_mean_loss": last_mean,
        "relative_loss_reduction": float(reduction),
        "minimum_relative_reduction": float(config["training"]["g1_min_relative_loss_reduction"]),
        "updates": len(pair_plan),
        "unique_paired_scenes": 16,
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "microbatch": microbatch,
        "passed": bool(np.isfinite(recent).all() and reduction >= float(config["training"]["g1_min_relative_loss_reduction"])),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "seconds": time.perf_counter() - start,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def r6e_run_g1() -> dict[str, Any]:
    preflight, _ = _require_preflight()
    config, r6a, physics = _configs()
    device, gpu, _, _ = _require_cuda()
    seed = int(config["data"]["g1_test_seed"])
    condition = r6a["radar"]["conditions"][0]
    generated = generate_d2a_paired_batch(
        16, seed, float(condition["snr_db"]), float(condition["cnr_db"]),
        dict(physics), scenario_id=str(condition["scenario_id"]),
    )
    data = _paired_iq_and_truth(generated)
    if len(data["scene_id"]) != 16:
        raise RuntimeError("R6-E G1 simulator returned an unexpected scene count")
    microbatch, resource_probe = _probe_common_microbatch(device, int(config["data"]["training_seed"]))
    rng = np.random.default_rng(int(config["data"]["sampler_seed"]))
    pair_plan = [
        (rng.integers(0, 16, size=32, dtype=np.int64), rng.integers(0, 16, size=32, dtype=np.int64))
        for _ in range(int(config["training"]["g1_micro_overfit_updates"]))
    ]
    arm_reports = {}
    for arm in R6E_ARMS:
        arm_reports[arm] = _overfit_g1_arm(
            arm, data, pair_plan, microbatch=microbatch, device=device, config=config
        )
    status = "G1_PASS" if all(value["passed"] for value in arm_reports.values()) else "G1_FAIL"
    record = {
        "status": status,
        "git_sha": preflight["git_sha"],
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "gpu": gpu,
        "device": str(device),
        "precision": "FP32",
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "formal_microbatch": microbatch,
        "resource_probe": resource_probe,
        "physical_test_seed": seed,
        "physical_test_scene_ids": [str(value) for value in generated.base_scene_id],
        "condition": dict(condition),
        "arms": arm_reports,
        "formal_train_validation_data_generated": any(
            (CACHE / split / "progress.json").is_file() for split in ("train", "validation")
        ),
    }
    _write_json(STAGE / "g1.json", record)
    return record


def _validation_summary(
    model: torch.nn.Module,
    validation: Mapping[str, np.ndarray],
    *,
    microbatch: int,
    device: torch.device,
) -> dict[str, float]:
    count = len(validation["scene_id"])
    totals: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, 32):
            indices = np.arange(start, min(start + 32, count), dtype=np.int64)
            batch = assemble_independent_h0_h1_batch(validation, indices, indices)
            current = _evaluate_batch_loss(model, batch, microbatch=microbatch, device=device)
            scale = len(indices) / count
            for key, value in current.items():
                totals[key] = totals.get(key, 0.0) + value * scale
    return totals


def _training_owner(
    arm: str, config: Mapping[str, Any], git_sha: str, sampler_audit: Mapping[str, Any]
) -> dict[str, Any]:
    model = build_r6e_student(arm, seed=int(config["data"]["training_seed"]))
    return {
        "owner": "D2-A-R6-E-training-run",
        "git_sha": git_sha,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_sha256(),
        "arm": arm,
        "training_seed": int(config["data"]["training_seed"]),
        "sampler_seed": int(config["data"]["sampler_seed"]),
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "updates": int(config["training"]["fixed_updates_per_model"]),
        "microbatch": int(json.loads((STAGE / "g1.json").read_text())["formal_microbatch"]),
        "parameters": count_r6e_parameters(model),
        "sampler_audit": dict(sampler_audit),
        "precision": "FP32",
        "data": "train_51200_paired_scenes",
    }


def _has_training_payload(directory: Path) -> bool:
    return any(path.name != "owner.json" for path in directory.iterdir())


def _ensure_training_owner(directory: Path, owner: Mapping[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    owner_path = directory / "owner.json"
    if owner_path.is_file():
        existing = json.loads(owner_path.read_text(encoding="utf-8"))
        if existing == owner:
            return
        if _has_training_payload(directory):
            raise RuntimeError(f"refusing to rebind a nonempty R6-E training directory: {directory}")
    elif _has_training_payload(directory):
        raise RuntimeError(f"refusing to reuse unowned R6-E training directory: {directory}")
    _write_json(owner_path, dict(owner))


def r6e_train_arm(arm: str) -> dict[str, Any]:
    if arm not in R6E_ARMS:
        raise ValueError(f"unknown R6-E arm: {arm}")
    preflight, git_sha = _require_preflight()
    config, _, _ = _configs()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-E training requires G1_PASS for all candidates")
    effective_batch = int(config["training"]["effective_batch_size_cpi"])
    microbatch = int(g1["formal_microbatch"])
    if effective_batch != 64 or microbatch not in (16, 8, 4):
        raise RuntimeError("R6-E effective batch or microbatch differs from the frozen protocol")
    train = _load_pair_cache("train")
    validation = _load_pair_cache("validation")
    h1_plan = build_repeated_h1_plan(
        np.asarray(train["condition_index"], dtype=np.int64),
        updates=int(config["training"]["fixed_updates_per_model"]),
        h1_per_update=effective_batch // 2,
        seed=int(config["data"]["training_seed"]),
    )
    h0_plan, h0_audit = build_h0_sample_plan(
        h1_plan,
        np.asarray(train["condition_index"], dtype=np.int64),
        np.asarray(train["condition_index"], dtype=np.int64),
        {}, mode="uniform", seed=int(config["data"]["sampler_seed"]), hard_fraction=0.0,
    )
    sampler_audit = {
        **h0_audit,
        "h0_plan_sha256": hashlib.sha256(h0_plan.tobytes()).hexdigest(),
        "h1_plan_sha256": hashlib.sha256(h1_plan.tobytes()).hexdigest(),
    }
    owner = _training_owner(arm, config, git_sha, sampler_audit)
    out = RUNS / arm
    _ensure_training_owner(out, owner)
    device, _, _, _ = _require_cuda()
    torch.manual_seed(int(config["data"]["training_seed"]))
    torch.cuda.manual_seed_all(int(config["data"]["training_seed"]))
    model = build_r6e_student(arm, seed=int(config["data"]["training_seed"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate_initial"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    target_updates = int(config["training"]["fixed_updates_per_model"])
    validation_interval = int(config["training"]["validation_interval_updates"])
    checkpoint_interval = int(config["training"]["resume_checkpoint_interval_updates"])
    last_path, best_path = out / "last_checkpoint.pt", out / "best_checkpoint.pt"
    history_path = out / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_objective, best_update, start_update = float("inf"), 0, 0
    elapsed_before = 0.0
    if last_path.is_file():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        if state.get("owner") != owner:
            raise RuntimeError("R6-E resume checkpoint owner mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state["update"])
        history = [row for row in state.get("history", []) if int(row["update"]) <= start_update]
        best_objective = float(state["best_objective"])
        best_update = int(state["best_update"])
        elapsed_before = float(state.get("training_seconds_completed", 0.0))
        if "numpy_rng_state" in state:
            np.random.set_state(state["numpy_rng_state"])
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"])
        if "cuda_rng_state" in state:
            torch.cuda.set_rng_state(state["cuda_rng_state"], device=device)
        if best_path.is_file():
            best_state = torch.load(best_path, map_location="cpu", weights_only=False)
            if best_state.get("owner") != owner:
                raise RuntimeError("R6-E best checkpoint owner mismatch")
            saved_best_objective = float(best_state["validation_objective"])
            if saved_best_objective < best_objective:
                best_objective = saved_best_objective
                best_update = int(best_state["update"])
            elif not math.isclose(saved_best_objective, best_objective, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("R6-E best checkpoint objective conflicts with its resume checkpoint")
        _write_history(history_path, history)
    elif _has_training_payload(out):
        raise RuntimeError("R6-E run contains artifacts without an owner-bound resume checkpoint")
    else:
        _atomic_torch_save({
            "owner": owner,
            "update": 0,
            "model_state_dict": _cpu_tree(model.state_dict()),
            "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            "history": history,
            "best_objective": best_objective,
            "best_update": best_update,
            "training_seconds_completed": 0.0,
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
            "sampler_audit": sampler_audit,
        }, last_path)

    interval_sums = {key: 0.0 for key in ("total", "presence", "range_cell", "range_offset", "doppler", "candidate")}
    interval_count = 0
    run_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for update in range(start_update + 1, target_updates + 1):
        lr = r6d_learning_rate(
            update, total_updates=target_updates,
            initial=float(config["training"]["learning_rate_initial"]),
            final=float(config["training"]["learning_rate_final"]),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        sampled = assemble_independent_h0_h1_batch(
            train, h0_plan[update - 1], h1_plan[update - 1]
        )
        terms = _training_update(
            model, optimizer, sampled, microbatch=microbatch, effective_batch=effective_batch,
            device=device, gradient_clip=float(config["training"]["gradient_clip_norm"]),
        )
        interval_count += 1
        for key in interval_sums:
            interval_sums[key] += terms[key]

        if update % validation_interval == 0:
            validation_metrics = _validation_summary(model, validation, microbatch=microbatch, device=device)
            objective = float(validation_metrics["total"])
            if objective < best_objective:
                best_objective, best_update = objective, update
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
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            history.append(row)
            _write_history(history_path, history)
            print(json.dumps({"stage": "train", "arm": arm, "update": update,
                              "train_total": row["interval_train_total"],
                              "validation_total": objective, "best_update": best_update,
                              "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"]},
                             ensure_ascii=False), flush=True)
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_count = 0
            model.train()

        if update % checkpoint_interval == 0:
            elapsed = elapsed_before + time.perf_counter() - run_start
            _atomic_torch_save({
                "owner": owner,
                "update": update,
                "model_state_dict": _cpu_tree(model.state_dict()),
                "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                "history": history,
                "best_objective": best_objective,
                "best_update": best_update,
                "training_seconds_completed": elapsed,
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(),
                "sampler_audit": sampler_audit,
            }, last_path)

    train_seconds = elapsed_before + time.perf_counter() - run_start
    if not best_path.is_file():
        raise RuntimeError(f"R6-E {arm} completed without a validation-selected checkpoint")
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
        "model_parameter_counts": count_r6e_parameters(model),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "sampler_audit": sampler_audit,
        "history": str(history_path.relative_to(ROOT)),
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "last_checkpoint": str(last_path.relative_to(ROOT)),
        "preflight_git_sha": preflight["git_sha"],
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


def cleanup_owned_training_cache() -> dict[str, Any]:
    """Delete only the train/validation cache directories with valid R6-E owners."""
    preflight, git_sha = _require_preflight()
    config, r6a, _ = _configs()
    plans = _all_split_plans(config, r6a)
    removed: list[dict[str, Any]] = []
    for split in ("train", "validation"):
        directory = CACHE / split
        if not directory.exists():
            continue
        owner_path, progress_path = directory / "owner.json", directory / "progress.json"
        if not owner_path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"refusing to clean unowned R6-E cache: {directory}")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        expected = _cache_owner(split, plans[split], git_sha)
        if not _owner_matches(expected, owner):
            raise RuntimeError(f"refusing to clean cache with a different R6-E owner: {directory}")
        removed_bytes = sum(child.stat().st_size for child in directory.iterdir() if child.is_file())
        for child in directory.iterdir():
            if child.is_file():
                child.unlink()
            else:
                raise RuntimeError(f"unexpected directory inside R6-E cache: {child}")
        directory.rmdir()
        removed.append({"split": split, "bytes": removed_bytes})
    try:
        CACHE.rmdir()
    except OSError:
        pass
    return {
        "status": "cleaned",
        "preflight_git_sha": preflight["git_sha"],
        "removed": removed,
        "scene_manifests_preserved": True,
        "user_untracked_files_touched": False,
    }
