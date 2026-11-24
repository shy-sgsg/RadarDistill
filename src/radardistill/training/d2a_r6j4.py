"""Matched truth-only training and scene planning for D2-A-R6-J4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from radardistill.evaluation.d2a_r6a import build_split_plan
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_j4 import R6J4Student, r6j4_loss
from radardistill.students.r6_joint import R6J1Student
from radardistill.training.d2a_r6d_joint import build_r6d_paired_plan, r6d_loss


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-J4"
CACHE = STAGE / "cache/training"
RUNS = STAGE / "runs/seed_909"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"
ARMS = ("j1_joint", "j4_joint")
PAIR_FIELDS = (
    "iq_h0",
    "iq_h1",
    "truth_range_cell",
    "truth_doppler_hz",
    "condition_index",
    "scene_id",
)


def read_r6j4_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def r6j4_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_r6j4_yaml(STAGE / "config.yaml")
    conditions = read_r6j4_yaml(ROOT / config["radar"]["condition_source"])
    physics = read_r6j4_yaml(ROOT / config["radar"]["source_config"])
    if len(conditions["radar"]["conditions"]) != int(config["radar"]["condition_count"]):
        raise ValueError("R6-J4 condition count differs from R6-A")
    if config["radar"]["conditions"] != conditions["radar"]["conditions"]:
        raise ValueError("R6-J4 conditions differ from the frozen R6-A condition grid")
    return config, conditions, physics


def build_r6j4_scene_plans(
    config: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Build all registered train, validation, evaluation and G1 scene chunks."""
    data = config["data"]
    if len(conditions) != int(config["radar"]["condition_count"]):
        raise ValueError("R6-J4 condition count mismatch")
    paired_data = {
        "condition_count": int(config["radar"]["condition_count"]),
        "condition_offset_stride": int(data["condition_offset_stride"]),
        "chunk_size": int(data["chunk_size"]),
        "chunk_seed_stride": int(data["chunk_seed_stride"]),
        "splits": {
            name: data["splits"][name]
            for name in ("train", "validation")
        },
    }
    plans = {
        name: build_r6d_paired_plan(paired_data, list(conditions), name)
        for name in ("train", "validation")
    }
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        eval_config = {
            "data": {
                "condition_count": int(config["radar"]["condition_count"]),
                "condition_seed_stride": int(data["condition_seed_stride"]),
                "chunk_size": int(data["chunk_size"]),
                "chunk_seed_stride": int(data["chunk_seed_stride"]),
                "splits": {name: data["splits"][name]},
            },
            "radar": {"conditions": list(conditions)},
        }
        plans[name] = build_split_plan(eval_config, name)
        for row in plans[name]:
            row["presence"] = data["splits"][name]["presence"]
    g1_condition_index = 0
    plans["g1"] = [{
        "split": "g1",
        "presence": "paired",
        "condition_index": g1_condition_index,
        "course": str(conditions[g1_condition_index]["course"]),
        "condition": dict(conditions[g1_condition_index]),
        "count": int(data["g1_test_count"]),
        "seed": int(data["g1_test_seed"]),
    }]
    return plans


def r6j4_scene_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for row in rows:
        seed = int(row["seed"])
        count = int(row["count"])
        if seed < 0 or count <= 0:
            raise ValueError("scene seed and count must be positive")
        ids.update(f"d2a-scene-{seed}-{index}" for index in range(count))
    expected = sum(int(row["count"]) for row in rows)
    if len(ids) != expected:
        raise ValueError("duplicate scene IDs within a registered split")
    return ids


def audit_r6j4_scene_plans(
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_counts: Mapping[str, int],
    prior_scene_ids: set[str],
) -> dict[str, Any]:
    if set(plans) != set(expected_counts):
        raise ValueError("R6-J4 plans and expected split names differ")
    seen: set[str] = set()
    ids_by_split: dict[str, set[str]] = {}
    for name, rows in plans.items():
        ids = r6j4_scene_ids(rows)
        if len(ids) != int(expected_counts[name]):
            raise RuntimeError(f"R6-J4 {name} count differs from its registered plan")
        overlap = seen.intersection(ids)
        if overlap:
            raise RuntimeError(f"R6-J4 split scene IDs overlap at {name}")
        prior_overlap = ids.intersection(prior_scene_ids)
        if prior_overlap:
            raise RuntimeError(f"R6-J4 {name} reuses {len(prior_overlap)} prior IDs")
        ids_by_split[str(name)] = ids
        seen.update(ids)
    return {
        "scene_counts": {name: len(values) for name, values in ids_by_split.items()},
        "split_overlap_count": 0,
        "prior_overlap_count": 0,
        "scene_ids": ids_by_split,
    }


def validate_r6j4_cuda_contract(
    *,
    available: bool,
    device_name: str | None,
    device_index: int | None,
    free_bytes: int,
    required_name: str = DEVICE_NAME,
    required_index: int = 0,
    minimum_free_bytes: int = 1_610_612_736,
) -> dict[str, Any]:
    if not available:
        raise RuntimeError("R6-J4 requires CUDA; CPU fallback is forbidden")
    if device_name != required_name or device_index != required_index:
        raise RuntimeError("R6-J4 CUDA device does not match the registered device")
    if int(free_bytes) < int(minimum_free_bytes):
        raise RuntimeError("R6-J4 free VRAM is below the registered minimum")
    return {
        "device": f"cuda:{required_index}",
        "gpu_name": str(device_name),
        "free_vram_bytes": int(free_bytes),
        "minimum_free_vram_bytes": int(minimum_free_bytes),
        "cpu_fallback": False,
    }


def r6j4_learning_rate(
    update: int,
    *,
    total_updates: int = 25_600,
    initial: float = 1e-3,
    final: float = 1e-5,
) -> float:
    if isinstance(update, bool) or not isinstance(update, int) or not 1 <= update <= total_updates:
        raise ValueError("update must be within the registered budget")
    if total_updates < 2 or not math.isfinite(initial) or not math.isfinite(final) or initial <= final or final <= 0:
        raise ValueError("invalid cosine learning-rate schedule")
    fraction = (update - 1) / (total_updates - 1)
    return float(final + 0.5 * (initial - final) * (1.0 + math.cos(math.pi * fraction)))


def r6j4_presence_labels(batch_size: int) -> torch.Tensor:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0 or batch_size % 2:
        raise ValueError("effective batch size must be positive and even")
    half = batch_size // 2
    return torch.cat((torch.zeros(half), torch.ones(half)))


def checkpoint_owner_matches(payload: Mapping[str, Any], expected_owner: Mapping[str, Any]) -> bool:
    return isinstance(payload.get("owner"), Mapping) and dict(payload["owner"]) == dict(expected_owner)


def build_r6j4_pair_order(scene_count: int, *, updates: int, pairs_per_update: int, seed: int) -> np.ndarray:
    """Visit a fixed paired-scene pool uniformly and repeat whole deterministic epochs."""
    if min(scene_count, updates, pairs_per_update) <= 0 or seed < 0:
        raise ValueError("invalid R6-J4 pair-order arguments")
    total = updates * pairs_per_update
    if total % scene_count:
        raise ValueError("registered train scenes must divide the fixed pair exposure")
    rng = np.random.default_rng(seed)
    epochs = total // scene_count
    order = np.concatenate([rng.permutation(scene_count) for _ in range(epochs)])
    result = order.reshape(updates, pairs_per_update).astype(np.int32, copy=False)
    if result.shape != (updates, pairs_per_update):
        raise RuntimeError("R6-J4 sampler plan has an incorrect shape")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _git_sha() -> str:
    return __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _write_scene_manifest(path: Path, split: str, plan: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("scene_id", "split", "seed", "condition_index", "course", "scenario_id", "snr_db", "cnr_db"))
        for spec in plan:
            condition = spec["condition"]
            for index in range(int(spec["count"])):
                writer.writerow((
                    f"d2a-scene-{int(spec['seed'])}-{index}", split, int(spec["seed"]),
                    int(spec["condition_index"]), condition["course"], condition["scenario_id"],
                    condition["snr_db"], condition["cnr_db"],
                ))
    temporary.replace(path)


def _cache_paths(split: str) -> dict[str, Path]:
    directory = CACHE / split
    return {
        "iq_h0": directory / "iq_h0.npy",
        "iq_h1": directory / "iq_h1.npy",
        "truth_range_cell": directory / "truth_range_cell.npy",
        "truth_doppler_hz": directory / "truth_doppler_hz.npy",
        "condition_index": directory / "condition_index.npy",
        "scene_id": directory / "scene_id.npy",
    }


def _expected_cache_shapes(count: int) -> dict[str, tuple[tuple[int, ...], Any]]:
    return {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "iq_h1": ((count, 3, 130, 32), np.complex64),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
        "condition_index": ((count,), np.uint8),
        "scene_id": ((count,), "<U48"),
    }


def _cache_owner(split: str, plan: Sequence[Mapping[str, Any]], git_sha: str, config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "owner": "D2-A-R6-J4-paired-cache",
        "split": split,
        "git_sha": git_sha,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "scene_count": sum(int(row["count"]) for row in plan),
        "seed_plan_sha256": hashlib.sha256(json.dumps(list(plan), sort_keys=True).encode()).hexdigest(),
        "dtype": "complex64 IQ / float32 truth",
        "stage": config["experiment"]["id"],
    }


def _load_pair_cache(split: str) -> dict[str, np.ndarray]:
    config, conditions, _ = r6j4_configs()
    plan = build_r6j4_scene_plans(config, conditions["radar"]["conditions"])[split]
    count = sum(int(row["count"]) for row in plan)
    paths = _cache_paths(split)
    owner = _cache_owner(split, plan, _git_sha(), config)
    owner_path = CACHE / split / "owner.json"
    progress_path = CACHE / split / "progress.json"
    if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
        raise RuntimeError(f"R6-J4 {split} cache owner is missing or mismatched")
    if not progress_path.is_file():
        raise RuntimeError(f"R6-J4 {split} cache progress is missing")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("owner") != owner or int(progress.get("completed_chunks", -1)) != len(plan):
        raise RuntimeError(f"R6-J4 {split} cache is incomplete or owned by another run")
    if int(progress.get("rows_written", -1)) != count:
        raise RuntimeError(f"R6-J4 {split} cache row count differs from the frozen plan")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    for name, (shape, dtype) in _expected_cache_shapes(count).items():
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
            raise RuntimeError(f"R6-J4 {split} cache shape/dtype mismatch for {name}")
    expected_ids = np.asarray([
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in plan
        for index in range(int(row["count"]))
    ], dtype="<U48")
    expected_conditions = np.asarray([
        int(row["condition_index"]) for row in plan for _ in range(int(row["count"]))
    ], dtype=np.int64)
    if not np.array_equal(arrays["condition_index"].astype(np.int64), expected_conditions):
        raise RuntimeError(f"R6-J4 {split} cache order differs from its registered condition plan")
    if not np.array_equal(arrays["scene_id"], expected_ids):
        raise RuntimeError(f"R6-J4 {split} cache scene IDs differ from its registered plan")
    if any(not np.isfinite(arrays[name]).all() for name in PAIR_FIELDS if name != "scene_id"):
        raise RuntimeError(f"R6-J4 {split} cache contains nonfinite values")
    return arrays


def generate_r6j4_pair_cache(split: str) -> dict[str, Any]:
    """Generate or resume only owner-bound Train/Validation IQ caches."""
    if split not in {"train", "validation"}:
        raise ValueError("only R6-J4 train and validation caches may be persisted")
    config, condition_config, physics = r6j4_configs()
    preflight_path, g1_path = STAGE / "preflight.json", STAGE / "g1.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text())["status"] != "G0_PASS":
        raise RuntimeError("R6-J4 cache generation requires G0_PASS")
    if not g1_path.is_file() or json.loads(g1_path.read_text())["status"] != "G1_PASS":
        raise RuntimeError("R6-J4 cache generation requires G1_PASS")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    git_sha = _git_sha()
    if git_sha != preflight["git_sha"]:
        raise RuntimeError("R6-J4 source HEAD changed after G0")
    plans = build_r6j4_scene_plans(config, condition_config["radar"]["conditions"])
    plan = plans[split]
    count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    directory.mkdir(parents=True, exist_ok=True)
    owner = _cache_owner(split, plan, git_sha, config)
    owner_path = directory / "owner.json"
    root_owner_path = STAGE / "owner_manifest.json"
    root_owner = {
        "owner": "D2-A-R6-J4-cache-owner",
        "git_sha": git_sha,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "cache_root": str(CACHE.relative_to(ROOT)),
        "owned_splits": {},
        "cleanup_permitted_paths": [],
    }
    if root_owner_path.is_file():
        current_root_owner = json.loads(root_owner_path.read_text(encoding="utf-8"))
        for key in ("owner", "git_sha", "protocol_sha256", "config_sha256", "cache_root"):
            if current_root_owner.get(key) != root_owner[key]:
                raise RuntimeError("R6-J4 root cache owner manifest mismatch")
        root_owner = current_root_owner
    elif CACHE.exists() and any(path for path in CACHE.iterdir() if path.name != split):
        raise RuntimeError("R6-J4 refuses to bind an existing cache root without its owner manifest")
    if owner_path.is_file():
        if json.loads(owner_path.read_text(encoding="utf-8")) != owner:
            raise RuntimeError(f"R6-J4 {split} cache owner mismatch")
    elif any(directory.iterdir()):
        raise RuntimeError(f"R6-J4 refuses to reuse unowned cache directory {directory}")
    else:
        _atomic_json(owner_path, owner)

    shapes = _expected_cache_shapes(count)
    paths = _cache_paths(split)
    progress_path = directory / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError(f"R6-J4 {split} cache progress owner mismatch")
        completed = int(progress["completed_chunks"])
        offset = int(progress["rows_written"])
        if not 0 <= completed <= len(plan) or offset != sum(int(row["count"]) for row in plan[:completed]):
            raise RuntimeError(f"R6-J4 {split} cache cursor mismatch")
    else:
        completed = offset = 0
        progress = {"owner": owner, "completed_chunks": 0, "rows_written": 0}
        _atomic_json(progress_path, progress)
    for name, (shape, dtype) in shapes.items():
        if not paths[name].exists():
            np.lib.format.open_memmap(paths[name], mode="w+", dtype=dtype, shape=shape).flush()
        else:
            current = np.load(paths[name], mmap_mode="r")
            if current.shape != shape or current.dtype != np.dtype(dtype):
                raise RuntimeError(f"R6-J4 {split} cache array {name} mismatches plan")
    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    manifest = STAGE / f"scene_manifest_{split}.csv"
    manifest_owner = STAGE / f"scene_manifest_{split}.owner.json"
    if manifest_owner.exists() and json.loads(manifest_owner.read_text(encoding="utf-8")) != owner:
        raise RuntimeError(f"R6-J4 {split} cache manifest owner mismatch")
    if completed == 0 and manifest.exists() and not manifest_owner.exists():
        raise RuntimeError(f"R6-J4 refuses to overwrite unowned manifest {manifest}")
    if not manifest_owner.exists():
        _atomic_json(manifest_owner, owner)
    if manifest.exists() and completed:
        with manifest.open("r+b") as handle:
            handle.truncate(int(progress["manifest_offset"]))
    elif completed == 0:
        mode = "w"
    else:
        mode = "a"
    if completed > 0:
        mode = "a"
    with manifest.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if handle.tell() == 0:
            writer.writerow(("scene_id", "split", "seed", "condition_index", "course", "scenario_id", "snr_db", "cnr_db"))
            handle.flush()
        for chunk_index, spec in enumerate(plan[completed:], start=completed):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            stop = offset + int(spec["count"])
            arrays["iq_h0"][offset:stop] = np.asarray(generated.iq_h0, dtype=np.complex64)
            arrays["iq_h1"][offset:stop] = np.asarray(generated.iq_h1, dtype=np.complex64)
            arrays["truth_range_cell"][offset:stop] = np.asarray(generated.truth_target_range_cell, dtype=np.float32)
            arrays["truth_doppler_hz"][offset:stop] = np.asarray(generated.truth_target_doppler_hz, dtype=np.float32)
            arrays["condition_index"][offset:stop] = int(spec["condition_index"])
            arrays["scene_id"][offset:stop] = np.asarray(generated.base_scene_id).astype("<U48")
            for index, scene_id in enumerate(generated.base_scene_id):
                writer.writerow((str(scene_id), split, int(spec["seed"]), int(spec["condition_index"]),
                                 condition["course"], condition["scenario_id"], condition["snr_db"], condition["cnr_db"]))
            handle.flush()
            for array in arrays.values():
                array.flush()
            offset = stop
            progress = {"owner": owner, "completed_chunks": chunk_index + 1, "rows_written": offset,
                        "manifest_offset": handle.tell()}
            _atomic_json(progress_path, progress)
    del arrays
    root_owner.setdefault("owned_splits", {})[split] = {
        "owner": owner,
        "files": [str(path.relative_to(ROOT)) for path in paths.values()],
        "manifest": str(manifest.relative_to(ROOT)),
        "manifest_owner": str(manifest_owner.relative_to(ROOT)),
    }
    root_owner["cleanup_permitted_paths"] = sorted({
        value
        for item in root_owner["owned_splits"].values()
        for value in [*item["files"], str((CACHE / split / "owner.json").relative_to(ROOT)),
                      str((CACHE / split / "progress.json").relative_to(ROOT))]
    })
    _atomic_json(root_owner_path, root_owner)
    loaded = _load_pair_cache(split)
    return {"split": split, "scene_count": len(loaded["scene_id"]), "cache_bytes": sum(path.stat().st_size for path in paths.values()), "complete": True}


def _paired_batch(data: Mapping[str, np.ndarray], indices: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    indices = np.asarray(indices, dtype=np.int64)
    h0 = np.asarray(data["iq_h0"][indices])
    h1 = np.asarray(data["iq_h1"][indices])
    def ri(value: np.ndarray) -> np.ndarray:
        return np.concatenate((value.real, value.imag), axis=1).astype(np.float32, copy=False)
    inputs = np.stack((ri(h0), ri(h1)), axis=1).reshape(-1, 6, 130, 32)
    range_truth = np.asarray(data["truth_range_cell"][indices], dtype=np.float32)
    doppler_truth = np.asarray(data["truth_doppler_hz"][indices], dtype=np.float32)
    truth = {
        "truth_presence": np.tile(np.asarray([0.0, 1.0], dtype=np.float32), len(indices)),
        "truth_range_cell": np.stack((np.zeros_like(range_truth), range_truth), axis=1).reshape(-1),
        "truth_doppler_hz": np.stack((np.zeros_like(doppler_truth), doppler_truth), axis=1).reshape(-1),
    }
    return inputs, truth


def _model(arm: str, config: Mapping[str, Any], physics: Mapping[str, Any], seed: int) -> torch.nn.Module:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if arm == "j1_joint":
            return R6J1Student(width=int(config["models"][arm]["width"]))
        if arm == "j4_joint":
            from radardistill.d2a_baselines import dpca_time_shift_s
            return R6J4Student(
                dpca_delay_s=dpca_time_shift_s(dict(physics)),
                width=int(config["models"][arm]["width"]),
                prf_hz=float(config["radar"]["prf_hz"]),
                nfft=int(config["models"]["j4_frontend"]["doppler_fft_bins"]),
            )
        raise ValueError(f"unknown R6-J4 arm: {arm}")


def _loss(arm: str, output: Mapping[str, torch.Tensor], truth: Mapping[str, torch.Tensor], config: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    if arm == "j1_joint":
        return r6d_loss(output, truth, arm="j1_joint", prf_hz=float(config["radar"]["prf_hz"]))
    return r6j4_loss(output, truth, prf_hz=float(config["radar"]["prf_hz"]))


def _loss_batch(model: torch.nn.Module, iq: torch.Tensor, truth: Mapping[str, torch.Tensor], arm: str, config: Mapping[str, Any], microbatch: int) -> dict[str, float]:
    totals: dict[str, float] = {}
    count = len(iq)
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, microbatch):
            stop = min(count, start + microbatch)
            output = model(iq[start:stop])
            sliced = {key: value[start:stop] for key, value in truth.items()}
            values = _loss(arm, output, sliced, config)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value) * (stop - start) / count
    return totals


def _accumulate_validation_losses(
    totals: dict[str, float], values: Mapping[str, float], weight: float
) -> None:
    """Accumulate one validation batch without assuming both arms share loss names."""
    if totals and set(totals) != set(values):
        raise ValueError("R6-J4 validation loss keys changed between batches")
    if not totals:
        totals.update({key: 0.0 for key in values})
    for key, value in values.items():
        totals[key] += float(value) * weight


def _training_update(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iq: torch.Tensor,
                     truth: Mapping[str, torch.Tensor], *, arm: str, config: Mapping[str, Any],
                     microbatch: int, device: torch.device) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    totals: dict[str, float] = {}
    count = len(iq)
    for start in range(0, count, microbatch):
        stop = min(count, start + microbatch)
        inputs = iq[start:stop].to(device, non_blocking=True)
        batch_truth = {key: value[start:stop].to(device, non_blocking=True) for key, value in truth.items()}
        outputs = model(inputs)
        terms = _loss(arm, outputs, batch_truth, config)
        scaled = terms["total"] * ((stop - start) / count)
        if not torch.isfinite(scaled):
            raise FloatingPointError(f"R6-J4 {arm} produced a nonfinite loss")
        scaled.backward()
        for key, value in terms.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach()) * ((stop - start) / count)
    torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
    optimizer.step()
    return totals


def _owner(arm: str, config: Mapping[str, Any], source_sha: Mapping[str, str], sampler_sha: str) -> dict[str, Any]:
    model = _model(arm, config, r6j4_configs()[2], int(config["training"]["training_seed_by_arm"][arm]))
    return {
        "owner": "D2-A-R6-J4-training-run",
        "arm": arm,
        "git_sha": _git_sha(),
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": dict(source_sha),
        "training_seed": int(config["training"]["training_seed_by_arm"][arm]),
        "sampler_sha256": sampler_sha,
        "updates": int(config["training"]["fixed_updates"]),
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "neural_network_precision": "FP32",
        "fixed_dpca_shift_accumulation": "complex128",
        "fixed_doppler_fft_precision": "complex64",
    }


def r6j4_train_all() -> dict[str, Any]:
    """Train the matched J1-J and J4 arms sequentially with resumable checkpoints."""
    config, _, physics = r6j4_configs()
    preflight = json.loads((STAGE / "preflight.json").read_text(encoding="utf-8"))
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "G0_PASS" or g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-J4 training requires passing G0 and G1")
    if _git_sha() != preflight.get("git_sha"):
        raise RuntimeError("R6-J4 code changed after G0")
    data = {split: _load_pair_cache(split) for split in ("train", "validation")}
    training = config["training"]
    update_count = int(training["fixed_updates"])
    effective_batch = int(training["effective_batch_size_cpi"])
    pairs_per_update = int(training["per_update_examples"]["H1"])
    pair_plan = build_r6j4_pair_order(
        len(data["train"]["scene_id"]), updates=update_count,
        pairs_per_update=pairs_per_update, seed=int(training["sampler_seed"]),
    )
    sampler_file = STAGE / "pair_order_seed_910.npy"
    if sampler_file.exists():
        saved = np.load(sampler_file, mmap_mode="r")
        if saved.shape != pair_plan.shape or not np.array_equal(saved, pair_plan):
            raise RuntimeError("R6-J4 saved pair order differs from frozen sampler")
    else:
        np.save(sampler_file, pair_plan)
    sampler_sha = _sha256(sampler_file)
    source_sha = preflight["source_sha256"]
    training_summary_path = STAGE / "training_summary.json"
    summary: dict[str, Any] = {"status": "in_progress", "arms": {}, "registered_order": list(ARMS)}
    if training_summary_path.exists():
        summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
        if summary.get("registered_order") != list(ARMS):
            raise RuntimeError("R6-J4 training summary arm order mismatch")
    _atomic_json(training_summary_path, summary)
    device = torch.device("cuda:0")
    for arm in ARMS:
        if arm in summary["arms"] and summary["arms"][arm].get("status") == "complete":
            continue
        train = data["train"]
        validation = data["validation"]
        owner = _owner(arm, config, source_sha, sampler_sha)
        out = RUNS / arm
        out.mkdir(parents=True, exist_ok=True)
        owner_path = out / "owner.json"
        if owner_path.exists():
            if json.loads(owner_path.read_text(encoding="utf-8")) != owner:
                raise RuntimeError(f"R6-J4 {arm} run owner mismatch")
        elif any(out.iterdir()):
            raise RuntimeError(f"R6-J4 refuses to reuse unowned run directory: {out}")
        else:
            _atomic_json(owner_path, owner)
        model = _model(arm, config, physics, int(training["training_seed_by_arm"][arm])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(training["learning_rate_initial"]),
            weight_decay=float(training["weight_decay"]),
        )
        last_path, best_path = out / "last_checkpoint.pt", out / "best_checkpoint.pt"
        history_path = out / "training_history.json"
        history: list[dict[str, Any]] = []
        start_update, best_objective, best_update, elapsed_before = 0, float("inf"), 0, 0.0
        microbatch = int(g1["formal_microbatch"])
        if last_path.exists():
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            if not checkpoint_owner_matches(state, owner):
                raise RuntimeError(f"R6-J4 {arm} last checkpoint owner mismatch")
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            start_update = int(state["update"])
            best_objective = float(state["best_objective"])
            best_update = int(state["best_update"])
            history = list(state["history"])
            elapsed_before = float(state.get("training_seconds", 0.0))
            microbatch = int(state["microbatch"])
        elif history_path.exists() or best_path.exists():
            raise RuntimeError(f"R6-J4 {arm} has partial training artifacts without a resumable checkpoint")
        else:
            _atomic_torch_save({
                "owner": owner, "update": 0, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "history": history,
                "best_objective": best_objective, "best_update": best_update,
                "training_seconds": 0.0, "microbatch": microbatch,
            }, last_path)
        run_start = time.perf_counter()
        interval: dict[str, float] = {}
        interval_n = 0
        torch.cuda.reset_peak_memory_stats(device)
        for update in range(start_update + 1, update_count + 1):
            lr = r6j4_learning_rate(
                update, total_updates=update_count,
                initial=float(training["learning_rate_initial"]),
                final=float(training["learning_rate_final"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            selected = pair_plan[update - 1]
            batch_np, truth_np = _paired_batch(train, selected)
            batch = torch.from_numpy(batch_np)
            truth = {key: torch.from_numpy(value) for key, value in truth_np.items()}
            while True:
                try:
                    terms = _training_update(model, optimizer, batch, truth, arm=arm, config=config,
                                             microbatch=microbatch, device=device)
                    break
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    candidates = [int(value) for value in training["microbatch_candidates_after_oom"]]
                    smaller = [value for value in candidates if value < microbatch and effective_batch % value == 0]
                    if not smaller:
                        raise RuntimeError(f"R6-J4 {arm} OOM at minimum microbatch; stopping without CPU fallback")
                    microbatch = max(smaller)
            interval_n += 1
            for key, value in terms.items():
                interval[key] = interval.get(key, 0.0) + float(value)
            validate_every = int(training["validation_interval_updates"])
            checkpoint_every = int(training["checkpoint_interval_updates"])
            if update % validate_every == 0:
                # Validation follows the same 32 H0 / 32 H1 contract but streams scene pairs.
                val_losses: dict[str, float] = {}
                for start in range(0, len(validation["scene_id"]), pairs_per_update):
                    indices = np.arange(start, min(len(validation["scene_id"]), start + pairs_per_update))
                    val_np, val_truth_np = _paired_batch(validation, indices)
                    val_tensor = torch.from_numpy(val_np).to(device)
                    val_truth = {key: torch.from_numpy(value).to(device) for key, value in val_truth_np.items()}
                    vals = _loss_batch(model, val_tensor, val_truth, arm, config, microbatch)
                    _accumulate_validation_losses(
                        val_losses, vals, len(indices) / len(validation["scene_id"])
                    )
                objective = float(val_losses["total"])
                if objective < best_objective:
                    best_objective, best_update = objective, update
                    _atomic_torch_save({"owner": owner, "update": update, "validation_objective": objective,
                                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                                        "validation_metrics": val_losses}, best_path)
                row = {
                    "update": update, "learning_rate": lr,
                    "train_interval": {key: value / max(interval_n, 1) for key, value in interval.items()},
                    "validation": val_losses, "best_update": best_update,
                    "microbatch": microbatch,
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
                history.append(row)
                _atomic_json(history_path, {"owner": owner, "rows": history})
                print(json.dumps({"stage": "train", "arm": arm, "update": update,
                                  "train_total": row["train_interval"]["total"], "validation_total": objective,
                                  "best_update": best_update, "microbatch": microbatch,
                                  "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"]}, flush=True))
                interval, interval_n = {}, 0
                model.train()
            if update % checkpoint_every == 0:
                elapsed = elapsed_before + time.perf_counter() - run_start
                _atomic_torch_save({
                    "owner": owner, "update": update,
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state_dict": optimizer.state_dict(), "history": history,
                    "best_objective": best_objective, "best_update": best_update,
                    "training_seconds": elapsed, "microbatch": microbatch,
                }, last_path)
        elapsed = elapsed_before + time.perf_counter() - run_start
        if not best_path.is_file():
            raise RuntimeError(f"R6-J4 {arm} finished without a selected validation checkpoint")
        record = {
            "status": "complete", "arm": arm, "owner": owner, "updates": update_count,
            "best_update": best_update, "best_validation_objective": best_objective,
            "training_seconds": elapsed, "effective_batch_size": effective_batch,
            "microbatch": microbatch, "scene_visits_per_presence": update_count * pairs_per_update / len(train["scene_id"]),
            "cpi_exposures_per_presence": update_count * pairs_per_update,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "history_path": str(history_path.relative_to(ROOT)),
            "best_checkpoint_path": str(best_path.relative_to(ROOT)),
        }
        _atomic_json(out / "training_record.json", record)
        _atomic_torch_save({
            "owner": owner, "update": update_count,
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(), "history": history,
            "best_objective": best_objective, "best_update": best_update,
            "training_seconds": elapsed, "microbatch": microbatch,
        }, last_path)
        del model, optimizer
        torch.cuda.empty_cache()
        summary["arms"][arm] = record
        _atomic_json(training_summary_path, summary)
    summary["status"] = "complete"
    _atomic_json(training_summary_path, summary)
    return summary
