#!/usr/bin/env python3
"""Run the frozen D2-A-R6-C capacity, training-budget and data-scale study."""

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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STAGE = ROOT / "experiments/D2-A-R6/R6-C"
CACHE = STAGE / "cache" / "training"
RUNS = STAGE / "runs" / "seed_404"
PREDICTIONS = STAGE / "predictions"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"
G1_TEST_SEED = 2_100_000_000

from radardistill.distillation.task_losses import decode_task_outputs, task_loss
from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_r6a import build_split_plan, summarize_parameter_estimator, wrapped_error
from radardistill.students.r6_scaled_cnn import (
    R6C_MODEL_SPECS,
    build_r6c_student,
    count_r6c_parameters,
    validate_r6c_student,
)
from radardistill.training.d2a_e002 import scale_d2a_e002_microbatch_loss
from radardistill.training.d2a_r6b_hard_negative import assemble_independent_h0_h1_batch
from radardistill.training.d2a_r6c_scaling import (
    build_r6c_h0_plan,
    build_r6c_h1_plan,
    build_r6c_paired_split_plan,
    r6c_extension_gate,
    r6c_extension_pair_decision,
    r6c_learning_rate,
    r6c_select_extension_candidate,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _disk_usage_bytes(path: Path) -> tuple[int, int]:
    usage = shutil.disk_usage(path)
    return int(usage.free), int(usage.total)


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    r6c = _read_yaml("experiments/D2-A-R6/R6-C/config.yaml")
    r6a = _read_yaml("experiments/D2-A-R6/R6-A/config.yaml")
    physics = _read_yaml(r6c["radar"]["source_config"])
    e003 = _read_yaml("experiments/D2-A-E003/config.yaml")
    return r6c, r6a, physics, e003


def _task_config(e003: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(e003["training"])
    training["loss"] = training["losses"]
    return {"radar": e003["radar"], "training": training}


def _technical_batch_task_loss(
    model: torch.nn.Module,
    x_cpu: torch.Tensor,
    truth_cpu: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    microbatch: int,
    loss_config: Mapping[str, Any],
) -> float:
    """Evaluate the complete balanced G1 batch in bounded device chunks."""
    if microbatch <= 0 or microbatch % 2 or len(x_cpu) % microbatch:
        raise ValueError("technical evaluation microbatch must be even and divide the balanced batch")
    total = 0.0
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x_cpu), microbatch):
            stop = start + microbatch
            x = x_cpu[start:stop].to(device)
            truth = {key: value[start:stop].to(device) for key, value in truth_cpu.items()}
            outputs = model(x, compute_spectrum=False)
            loss = task_loss(outputs, truth, dict(loss_config), teacher_weight=0.0)["task_total"]
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite R6-C technical evaluation loss")
            total += float(loss.item()) * (stop - start) / len(x_cpu)
    return total


def _eval_config(r6c: Mapping[str, Any], r6a: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(r6c["data"])
    data["condition_count"] = len(r6a["radar"]["conditions"])
    data["splits"] = {}
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        value = dict(r6c["data"]["splits"][name])
        value["scene_count"] = int(value["scenes"])
        data["splits"][name] = value
    return {"data": data, "radar": dict(r6a["radar"])}


def _all_split_plans(
    r6c: Mapping[str, Any], r6a: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    conditions = r6a["radar"]["conditions"]
    plans = {
        name: build_r6c_paired_split_plan(r6c["data"], conditions, name)
        for name in ("train_core", "train_extended", "validation")
    }
    eval_config = _eval_config(r6c, r6a)
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        plans[name] = build_split_plan(eval_config, name)
    return plans


def _scene_ids(plan: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in plan
        for index in range(int(row["count"]))
    }


def _prior_scene_ids() -> set[str]:
    paths: set[Path] = set()
    for base in (ROOT / "experiments/D2-A-R5", ROOT / "experiments/D2-A-R6/R6-A", ROOT / "experiments/D2-A-R6/R6-B"):
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
                        result.add(value)
    return result


def _check_nested_scene_ids(plans: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    extended = _scene_ids(plans["train_extended"])
    core = _scene_ids(plans["train_core"])
    validation = _scene_ids(plans["validation"])
    if not core <= extended or len(core) != 51_200 or len(extended) != 102_400:
        raise RuntimeError("R6-C Train core is not the registered nested subset of Train extended")
    if extended & validation:
        raise RuntimeError("R6-C train and validation scene IDs overlap")
    output = {"train_extended": extended, "validation": validation}
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        ids = _scene_ids(plans[name])
        if any(ids & values for values in output.values()):
            raise RuntimeError(f"R6-C scene IDs overlap the {name} split")
        output[name] = ids
    for first_index, first in enumerate(output):
        for second in list(output)[first_index + 1 :]:
            if output[first] & output[second]:
                raise RuntimeError(f"R6-C splits {first} and {second} overlap")
    return {name: ids for name, ids in output.items()}


def preflight() -> dict[str, Any]:
    """G0: validate frozen source, disjoint scene plan, disk, and required CUDA GPU."""
    r6c, r6a, _, _ = _configs()
    head = _git("rev-parse", "HEAD")
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R6-C execution requires main")
    if _git("rev-parse", "origin/main") != head:
        raise RuntimeError("G0 requires the frozen R6-C implementation commit pushed to origin/main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("G0 requires a clean tracked worktree")
    if r6c["experiment"]["protocol_version"] != 1.2 or r6c["experiment"]["status"] != "preregistered":
        raise RuntimeError("R6-C protocol/config identity or status changed")
    if r6c["experiment"]["final_blind_authorized"] is not False:
        raise RuntimeError("R6-C final blind is not authorized")
    plans = _all_split_plans(r6c, r6a)
    split_ids = _check_nested_scene_ids(plans)
    new_ids = set().union(*split_ids.values())
    prior_ids = _prior_scene_ids()
    g1_test_ids = {f"d2a-scene-{G1_TEST_SEED}-{index}" for index in range(16)}
    if g1_test_ids & new_ids:
        raise RuntimeError("R6-C technical G1 scenes overlap a registered scientific split")
    prior_overlap = (new_ids | g1_test_ids) & prior_ids
    if prior_overlap:
        raise RuntimeError(f"R6-C scene plan reuses {len(prior_overlap)} previously registered scene IDs")
    free, total_disk = _disk_usage_bytes(STAGE)
    required_disk = int(r6c["data"]["minimum_free_disk_bytes_before_data_generation"])
    if free < required_disk:
        raise RuntimeError(
            f"R6-C requires {required_disk} free bytes before cache generation; only {free} are available"
        )
    if CACHE.exists():
        raise RuntimeError("R6-C cache exists before G0; inspect it manually, do not overwrite it")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-C requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != DEVICE_NAME:
        raise RuntimeError(f"R6-C requires {DEVICE_NAME}, found {gpu_name}")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    model_counts = {}
    for name, spec in R6C_MODEL_SPECS.items():
        model = build_r6c_student(name, seed=int(r6c["models"]["training_seed"]))
        counts = validate_r6c_student(model, name)
        if counts["trainable_parameters"] != int(spec["parameters"]):
            raise RuntimeError(f"R6-C {name} parameter count mismatch")
        model_counts[name] = counts
        del model
    record = {
        "status": "G0_PASS",
        "git_sha": head,
        "protocol_version": 1.2,
        "protocol_sha256": hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest(),
        "split_scene_counts": {name: len(ids) for name, ids in split_ids.items()},
        "planned_seed_counts": {name: len({int(row["seed"]) for row in rows}) for name, rows in plans.items()},
        "train_core_is_exact_subset_of_train_extended": True,
        "prior_scene_id_count": len(prior_ids),
        "prior_scene_id_overlap": 0,
        "g1_test_seed": G1_TEST_SEED,
        "g1_test_scene_ids_reserved": len(g1_test_ids),
        "disk_free_before_generation_bytes": int(free),
        "disk_total_bytes": int(total_disk),
        "minimum_required_free_bytes": required_disk,
        "gpu": gpu_name,
        "device": "cuda:0",
        "free_vram_before_g1_bytes": int(free_vram),
        "total_vram_bytes": int(total_vram),
        "model_parameter_counts": model_counts,
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
        raise RuntimeError("run R6-C G0 preflight before any experiment data generation")
    record = json.loads(path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    if (
        record.get("status") != "G0_PASS"
        or record.get("git_sha") != head
        or record.get("protocol_sha256") != hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest()
        or record.get("config_sha256") != hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest()
    ):
        raise RuntimeError("R6-C G0 record does not match current pushed code/protocol/config")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-C preflight must bind the pushed main commit")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked source changed after R6-C G0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != DEVICE_NAME:
        raise RuntimeError("R6-C requires the registered RTX 3050 Laptop CUDA device")
    return record, head


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = (
        "update", "learning_rate", "interval_train_task_loss", "interval_presence_loss",
        "interval_range_cell_loss", "interval_range_offset_loss", "interval_doppler_loss",
        "validation_task_loss", "validation_presence_loss", "validation_range_cell_loss",
        "validation_range_offset_loss", "validation_doppler_loss", "validation_auc",
        "validation_range_rmse_cells", "validation_doppler_rmse_hz",
        "validation_range_cell_mismatch_rate", "seconds_since_last_checkpoint",
        "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _paired_cache_paths(split: str) -> dict[str, Path]:
    directory = CACHE / split
    names = ("iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id")
    return {name: directory / f"{name}.npy" for name in names}


def _cache_owner(split: str, head: str, plan: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    r6c, _, _, _ = _configs()
    plan_hash = hashlib.sha256(
        json.dumps(list(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "owner": "D2-A-R6-C-paired-cache",
        "git_sha": head,
        "protocol_version": 1.2,
        "split": split,
        "scene_count": sum(int(row["count"]) for row in plan),
        "plan_sha256": plan_hash,
        "config_sha256": hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest(),
        "source_config": r6c["radar"]["source_config"],
    }


def _owner_cache_dir(path: Path, owner: Mapping[str, Any]) -> None:
    if path.exists():
        owner_path = path / "owner.json"
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != dict(owner):
            raise RuntimeError(f"refusing to reuse an unowned or differently-owned R6-C cache: {path}")
        return
    path.mkdir(parents=True)
    _write_json(path / "owner.json", dict(owner))


def generate_paired_cache(split: str) -> dict[str, Any]:
    """Stream the registered extended-train or validation pairs to owned memmaps."""
    preflight_record, head = _require_preflight()
    g1_path = STAGE / "g1.json"
    if not g1_path.is_file() or json.loads(g1_path.read_text(encoding="utf-8")).get("status") != "G1_PASS":
        raise RuntimeError("formal R6-C data generation requires G1_PASS")
    if split not in {"train_extended", "validation"}:
        raise ValueError("only train_extended and validation caches are materialized")
    r6c, r6a, physics, _ = _configs()
    plan = _all_split_plans(r6c, r6a)[split]
    count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    owner = _cache_owner(split, head, plan)
    existed = directory.exists()
    _owner_cache_dir(directory, owner)
    paths = _paired_cache_paths(split)
    progress_path = directory / "progress.json"
    manifest_path = STAGE / f"scene_manifest_{split}.csv"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("R6-C cache progress owner mismatch")
        completed = int(progress["completed_chunks"])
        rows_written = int(progress["rows_written"])
        expected_rows = sum(int(row["count"]) for row in plan[:completed])
        if not 0 <= completed <= len(plan) or rows_written != expected_rows:
            raise RuntimeError("R6-C cache progress cursor differs from registered plan")
        if manifest_path.exists():
            with manifest_path.open("r+b") as handle:
                handle.truncate(int(progress["manifest_offset"]))
        elif rows_written:
            raise RuntimeError("R6-C cache manifest is missing for a nonempty resumed cache")
    else:
        unexpected = [entry.name for entry in directory.iterdir() if entry.name != "owner.json"]
        if existed and (unexpected or manifest_path.exists()):
            raise RuntimeError("R6-C partial cache has no valid progress cursor")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite unowned scene manifest: {manifest_path}")
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
            existing = np.load(paths[name], mmap_mode="r")
            if existing.shape != shape or existing.dtype != np.dtype(dtype):
                raise RuntimeError(f"R6-C cache shape/dtype mismatch for {name}")
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
        from radardistill.simulation.d2a import generate_d2a_paired_batch

        for chunk_index, spec in enumerate(plan[completed:], start=completed):
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            expected_shape = (int(spec["count"]), 3, 130, 32)
            if generated.iq_h0.shape != expected_shape or generated.iq_h1.shape != expected_shape:
                raise RuntimeError("R6-C simulator returned unexpected paired IQ shape")
            stop = offset + int(spec["count"])
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
        raise RuntimeError(f"R6-C {split} cache wrote {offset} scenes, expected {count}")
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
    if split not in {"train_extended", "validation"}:
        raise ValueError("unknown R6-C paired cache split")
    _, head = _require_preflight()
    r6c, r6a, _, _ = _configs()
    plan = _all_split_plans(r6c, r6a)[split]
    expected_count = sum(int(row["count"]) for row in plan)
    owner = _cache_owner(split, head, plan)
    directory = CACHE / split
    owner_path = directory / "owner.json"
    progress_path = directory / "progress.json"
    if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
        raise RuntimeError(f"R6-C {split} cache owner mismatch")
    if not progress_path.is_file():
        raise RuntimeError(f"R6-C {split} cache lacks progress record")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("owner") != owner or int(progress.get("completed_chunks", -1)) != len(plan):
        raise RuntimeError(f"R6-C {split} cache is incomplete")
    if int(progress.get("rows_written", -1)) != expected_count:
        raise RuntimeError(f"R6-C {split} cache scene count mismatch")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in _paired_cache_paths(split).items()}
    expected_shapes = {
        "iq_h0": (expected_count, 3, 130, 32),
        "iq_h1": (expected_count, 3, 130, 32),
        "truth_range_cell": (expected_count,),
        "truth_doppler_hz": (expected_count,),
        "condition_index": (expected_count,),
        "scene_id": (expected_count,),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise RuntimeError(f"R6-C {split} cache has unexpected array shapes")
    return arrays


class _IndexedArray:
    """Small-index view of a memmap; only requested batches are copied."""

    def __init__(self, array: np.ndarray, indices: np.ndarray) -> None:
        self.array = array
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: Any) -> np.ndarray:
        return self.array[self.indices[index]]


def train_core_indices(train_extended: Mapping[str, np.ndarray]) -> np.ndarray:
    """Map the registered core scene IDs into the materialized extended memmap."""
    r6c, r6a, _, _ = _configs()
    plans = _all_split_plans(r6c, r6a)
    core_ids = _scene_ids(plans["train_core"])
    extended_ids = np.asarray(train_extended["scene_id"]).astype(str)
    id_to_index = {scene_id: index for index, scene_id in enumerate(extended_ids)}
    if len(id_to_index) != len(extended_ids):
        raise RuntimeError("R6-C extended cache contains duplicate scene IDs")
    missing = core_ids.difference(id_to_index)
    if missing:
        raise RuntimeError(f"R6-C Train core has {len(missing)} scene IDs missing from extended cache")
    return np.fromiter((id_to_index[scene_id] for scene_id in sorted(core_ids)), dtype=np.int64)


def _validation_summary(
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    loss_config: Mapping[str, Any],
    *,
    device: torch.device,
    pair_chunk: int = 16,
) -> dict[str, float]:
    count = len(arrays["iq_h0"])
    weighted = {key: 0.0 for key in ("task_total", "presence", "range_cell", "range_offset", "doppler")}
    h0_scores: list[np.ndarray] = []
    h1_scores: list[np.ndarray] = []
    pred_range: list[np.ndarray] = []
    pred_doppler: list[np.ndarray] = []
    true_range: list[np.ndarray] = []
    true_doppler: list[np.ndarray] = []
    mismatch_count = 0
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, pair_chunk):
            indices = np.arange(start, min(start + pair_chunk, count), dtype=np.int64)
            batch = assemble_independent_h0_h1_batch(arrays, indices, indices)
            x = torch.from_numpy(batch["iq_ri"]).to(device)
            truth = {
                key: torch.from_numpy(batch[key]).to(device)
                for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
            }
            outputs = model(x, compute_spectrum=False)
            terms = task_loss(outputs, truth, dict(loss_config), teacher_weight=0.0)
            h1_count = len(indices)
            for key in weighted:
                weighted[key] += float(terms[key].item()) * h1_count / count
            decoded = decode_task_outputs(outputs, 1300.0)
            h0_scores.append(decoded["presence_probability"][::2].cpu().numpy())
            h1_scores.append(decoded["presence_probability"][1::2].cpu().numpy())
            predicted_cells = outputs["range_logits"][1::2].argmax(dim=-1)
            true_cells = torch.floor(truth["truth_range_cell"][1::2]).long().clamp(
                0, outputs["range_logits"].shape[1] - 1
            )
            mismatch_count += int(torch.count_nonzero(predicted_cells != true_cells).item())
            pred_range.append(decoded["range_cell"][1::2].cpu().numpy())
            pred_doppler.append(decoded["doppler_hz"][1::2].cpu().numpy())
            true_range.append(batch["truth_range_cell"][1::2])
            true_doppler.append(batch["truth_doppler_hz"][1::2])
    p0 = np.concatenate(h0_scores)
    p1 = np.concatenate(h1_scores)
    r0 = np.concatenate(pred_range)
    d0 = np.concatenate(pred_doppler)
    r1 = np.concatenate(true_range).astype(np.float64)
    d1 = np.concatenate(true_doppler).astype(np.float64)
    return {
        "task_loss": float(weighted["task_total"]),
        "presence_loss": float(weighted["presence"]),
        "range_cell_loss": float(weighted["range_cell"]),
        "range_offset_loss": float(weighted["range_offset"]),
        "doppler_loss": float(weighted["doppler"]),
        "auc": float(empirical_auc(p0, p1)),
        "range_rmse_cells": float(np.sqrt(np.mean((r0 - r1) ** 2))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(wrapped_error(d0, d1, 1300.0) ** 2))),
        "range_cell_mismatch_rate": float(mismatch_count / count),
    }


def _to_tensors(batch: Mapping[str, np.ndarray]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x = torch.from_numpy(batch["iq_ri"])
    truth = {
        key: torch.from_numpy(batch[key])
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    return x, truth


def _train_update(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    truth: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    microbatch: int,
    effective_batch: int,
    loss_config: Mapping[str, Any],
) -> dict[str, float]:
    if effective_batch % microbatch or microbatch % 2:
        raise ValueError("microbatch must divide effective batch and preserve balanced H0/H1 pairs")
    optimizer.zero_grad(set_to_none=True)
    totals = {key: 0.0 for key in ("task_total", "presence", "range_cell", "range_offset", "doppler")}
    for start in range(0, effective_batch, microbatch):
        stop = start + microbatch
        xm = x[start:stop].to(device)
        tm = {key: value[start:stop].to(device) for key, value in truth.items()}
        outputs = model(xm, compute_spectrum=False)
        terms = task_loss(outputs, tm, dict(loss_config), teacher_weight=0.0)
        scaled = scale_d2a_e002_microbatch_loss(
            terms,
            {"training": {"losses": dict(loss_config["training"]["losses"])}},
            microbatch_size=microbatch,
            effective_batch_size=effective_batch,
            microbatch_h1=microbatch // 2,
            effective_h1=effective_batch // 2,
            teacher_weight=0.0,
        )
        if not torch.isfinite(scaled):
            raise FloatingPointError("nonfinite R6-C task loss")
        scaled.backward()
        totals["task_total"] += float(scaled.detach().item())
        totals["presence"] += float(terms["presence"].detach().item()) * microbatch / effective_batch
        for key in ("range_cell", "range_offset", "doppler"):
            totals[key] += float(terms[key].detach().item()) * (microbatch // 2) / (effective_batch // 2)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError("nonfinite R6-C gradient norm")
    optimizer.step()
    return totals


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _model_owner(model_name: str, data_name: str, seed: int, head: str) -> dict[str, Any]:
    r6c, _, _, _ = _configs()
    return {
        "owner": "D2-A-R6-C-training-run",
        "git_sha": head,
        "protocol_sha256": hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest(),
        "model": model_name,
        "width": int(r6c["models"]["candidates"][model_name]["width"]),
        "data": data_name,
        "training_seed": int(seed),
        "effective_batch_size": int(r6c["training"]["effective_batch_size_cpi"]),
        "schedule": "r6c_cosine_6400_6400_12800_25600_51200",
        "precision": "FP32",
    }


def _run_training(
    model_name: str,
    data_name: str,
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    *,
    target_updates: int,
    seed: int = 404,
    microbatch: int,
) -> dict[str, Any]:
    preflight_record, head = _require_preflight()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS" or int(g1.get("formal_microbatch", -1)) != microbatch:
        raise RuntimeError("R6-C training requires passing G1 and its registered common microbatch")
    if target_updates not in {12_800, 25_600, 51_200}:
        raise ValueError("R6-C target updates must be a registered checkpoint")
    owner = _model_owner(model_name, data_name, seed, head)
    out = RUNS / data_name / model_name
    out.mkdir(parents=True, exist_ok=True)
    owner_path = out / "owner.json"
    if owner_path.exists() and json.loads(owner_path.read_text(encoding="utf-8")) != owner:
        raise RuntimeError(f"refusing to reuse a differently-owned training directory: {out}")
    if not owner_path.exists():
        if any(out.iterdir()):
            raise RuntimeError(f"refusing to reuse a nonempty unowned training directory: {out}")
        _write_json(owner_path, owner)

    r6c, _, _, e003 = _configs()
    effective_batch = int(r6c["training"]["effective_batch_size_cpi"])
    if len(train["iq_h0"]) != len(train["iq_h1"]):
        raise ValueError("R6-C paired train cache H0/H1 counts differ")
    condition_array = train["condition_index"]
    if isinstance(condition_array, _IndexedArray):
        train_conditions = np.asarray(condition_array.array[condition_array.indices], dtype=np.int64)
    else:
        train_conditions = np.asarray(condition_array, dtype=np.int64)
    full_updates = 51_200
    h1_plan = build_r6c_h1_plan(
        train_conditions,
        updates=full_updates,
        h1_per_update=effective_batch // 2,
        seed=seed,
    )
    h0_plan, sampling_audit = build_r6c_h0_plan(
        h1_plan,
        train_conditions,
        train_conditions,
        seed=seed + 1,
    )
    if h1_plan.shape != (full_updates, effective_batch // 2) or h0_plan.shape != h1_plan.shape:
        raise RuntimeError("R6-C H0/H1 training plans differ from effective batch/update requirements")
    loss_config = _task_config(e003)
    model = build_r6c_student(model_name, seed=seed).to("cuda:0")
    counts = validate_r6c_student(model, model_name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    state_path = out / "last_checkpoint.pt"
    history_path = out / "training_history.csv"
    history: list[dict[str, Any]] = []
    best_validation: dict[str, float] = {"task_loss": float("inf")}
    best_update = 0
    start_update = 0
    seconds_before = 0.0
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("owner") != owner:
            raise RuntimeError("R6-C resume checkpoint owner mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state["update"])
        history = list(state["history"])
        best_validation = dict(state["best_validation"])
        best_update = int(state["best_update"])
        seconds_before = float(state.get("training_seconds_completed", 0.0))
        if start_update >= target_updates:
            segment_path = out / f"stage_record_{target_updates}.json"
            if segment_path.is_file():
                return json.loads(segment_path.read_text(encoding="utf-8"))
    if start_update == 0:
        initial = _validation_summary(model, validation, loss_config, device=torch.device("cuda:0"))
        best_validation = initial
        history.append({"update": 0, **{f"validation_{key}" if key != "task_loss" else "validation_task_loss": value
                                       for key, value in initial.items()},
                        "interval_train_task_loss": initial["task_loss"], "learning_rate": 1e-3})
        _atomic_torch_save({
            "owner": owner,
            "update": 0,
            "model_state_dict": _cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_validation": best_validation,
            "best_update": 0,
            "history": history,
            "training_seconds_completed": 0.0,
            "sampling_audit": sampling_audit,
        }, state_path)
        _atomic_torch_save({
            "owner": owner,
            "model": model_name,
            "data": data_name,
            "update": 0,
            "model_state_dict": _cpu_state_dict(model),
            "validation": initial,
            "parameters": counts,
        }, out / "best_checkpoint.pt")
    torch.cuda.reset_peak_memory_stats(0)
    update_timer = time.perf_counter()
    interval_timer = time.perf_counter()
    interval_sums = {key: 0.0 for key in ("task_total", "presence", "range_cell", "range_offset", "doppler")}
    interval_updates = 0
    model.train()
    for update_index in range(start_update, target_updates):
        update = update_index + 1
        lr = r6c_learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = assemble_independent_h0_h1_batch(train, h0_plan[update_index], h1_plan[update_index])
        x, truth = _to_tensors(batch)
        values = _train_update(
            model, optimizer, x, truth, device=torch.device("cuda:0"),
            microbatch=microbatch, effective_batch=effective_batch, loss_config=loss_config,
        )
        for key, value in values.items():
            interval_sums[key] += value
        interval_updates += 1
        if update % int(r6c["training"]["validation_interval_updates"]) == 0:
            model.eval()
            metrics = _validation_summary(model, validation, loss_config, device=torch.device("cuda:0"))
            row = {
                "update": update,
                "learning_rate": lr,
                "interval_train_task_loss": interval_sums["task_total"] / interval_updates,
                "interval_presence_loss": interval_sums["presence"] / interval_updates,
                "interval_range_cell_loss": interval_sums["range_cell"] / interval_updates,
                "interval_range_offset_loss": interval_sums["range_offset"] / interval_updates,
                "interval_doppler_loss": interval_sums["doppler"] / interval_updates,
                **{f"validation_{key}": value for key, value in metrics.items()},
                "seconds_since_last_checkpoint": time.perf_counter() - interval_timer,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            }
            history.append(row)
            if metrics["task_loss"] < best_validation["task_loss"]:
                best_validation = dict(metrics)
                best_update = update
                _atomic_torch_save({
                    "owner": owner,
                    "model": model_name,
                    "data": data_name,
                    "update": update,
                    "model_state_dict": _cpu_state_dict(model),
                    "validation": metrics,
                    "parameters": counts,
                }, out / "best_checkpoint.pt")
            if update % int(r6c["training"]["resume_checkpoint_interval_updates"]) == 0:
                seconds_completed = seconds_before + time.perf_counter() - update_timer
                _atomic_torch_save({
                    "owner": owner,
                    "update": update,
                    "model_state_dict": _cpu_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_validation": best_validation,
                    "best_update": best_update,
                    "history": history,
                    "training_seconds_completed": seconds_completed,
                    "sampling_audit": sampling_audit,
                }, state_path)
            _write_history(history_path, history)
            print(json.dumps({
                "stage": "train", "model": model_name, "data": data_name,
                "update": update, "target_updates": target_updates,
                "validation_task_loss": metrics["task_loss"],
                "interval_seconds": row["seconds_since_last_checkpoint"],
                "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"],
            }, ensure_ascii=False), flush=True)
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_updates = 0
            interval_timer = time.perf_counter()
            model.train()

    torch.cuda.synchronize(0)
    seconds_completed = seconds_before + time.perf_counter() - update_timer
    if not (out / "best_checkpoint.pt").is_file():
        _atomic_torch_save({
            "owner": owner, "model": model_name, "data": data_name,
            "update": target_updates, "model_state_dict": _cpu_state_dict(model),
            "validation": best_validation, "parameters": counts,
        }, out / "best_checkpoint.pt")
    stage_record = {
        "status": "TRAINED",
        "owner": owner,
        "model": model_name,
        "data": data_name,
        "effective_batch": effective_batch,
        "microbatch": microbatch,
        "updates_completed": target_updates,
        "best_update": best_update,
        "selected_validation": best_validation,
        "training_seconds_total": seconds_completed,
        "seconds_per_update": seconds_completed / max(target_updates, 1),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "parameters": counts,
        "history_rows": len(history),
        "history_interval_updates": 400,
        "per_update_history_persisted": False,
        "training_examples_exposed": target_updates * effective_batch,
        "unique_training_pairs": len(train["iq_h0"]),
        "fixed_dataset_passes_h1": target_updates * (effective_batch // 2) / len(train["iq_h1"]),
        "sampling_audit": sampling_audit,
        "final_blind": False,
        "preflight_sha": preflight_record["git_sha"],
    }
    _write_json(out / f"stage_record_{target_updates}.json", stage_record)
    if target_updates in {12_800, 25_600, 51_200}:
        stage_best = out / f"best_checkpoint_{target_updates}.pt"
        if not stage_best.exists():
            # The global best checkpoint is guaranteed to be no later than this stage end.
            shutil.copy2(out / "best_checkpoint.pt", stage_best)
    seconds_completed = float(stage_record["training_seconds_total"])
    _atomic_torch_save({
        "owner": owner,
        "update": target_updates,
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_validation": best_validation,
        "best_update": best_update,
        "history": history,
        "training_seconds_completed": seconds_completed,
        "sampling_audit": sampling_audit,
    }, state_path)
    _write_history(history_path, history)
    del optimizer, model
    torch.cuda.empty_cache()
    return stage_record


def _training_arrays(split: str) -> dict[str, np.ndarray]:
    return _load_pair_cache(split)


def _core_view(extended: Mapping[str, np.ndarray]) -> dict[str, Any]:
    indices = train_core_indices(extended)
    return {name: _IndexedArray(value, indices) for name, value in extended.items()}


def _continuation_history(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _history_best(history: Sequence[Mapping[str, Any]], through_update: int) -> dict[str, float]:
    rows = [row for row in history if int(row["update"]) <= through_update]
    if not rows:
        raise RuntimeError("R6-C training history is empty")
    best = min(rows, key=lambda row: float(row["validation_task_loss"]))
    return {"task_loss": float(best["validation_task_loss"]), "update": int(best["update"])}


def _copy_best_at_stage(out: Path, target_updates: int) -> None:
    source = out / "best_checkpoint.pt"
    if not source.is_file():
        raise FileNotFoundError(source)
    shutil.copy2(source, out / f"best_checkpoint_{target_updates}.pt")


def run_g1() -> dict[str, Any]:
    """Run CUDA-only shape/gradient/16-pair overfit and shared microbatch gate."""
    _, head = _require_preflight()
    output = STAGE / "g1.json"
    if output.is_file():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("git_sha") == head and prior.get("status") == "G1_PASS":
            return prior
    r6c, r6a, physics, e003 = _configs()
    from radardistill.simulation.d2a import generate_d2a_paired_batch

    condition = r6a["radar"]["conditions"][0]
    generated = generate_d2a_paired_batch(
        16, G1_TEST_SEED, float(condition["snr_db"]), float(condition["cnr_db"]),
        dict(physics), scenario_id=str(condition["scenario_id"]),
    )
    arrays = {
        "iq_h0": generated.iq_h0,
        "iq_h1": generated.iq_h1,
        "truth_range_cell": generated.truth_target_range_cell,
        "truth_doppler_hz": generated.truth_target_doppler_hz,
    }
    batch = assemble_independent_h0_h1_batch(arrays, np.arange(16), np.arange(16))
    x_cpu, truth_cpu = _to_tensors(batch)
    loss_config = _task_config(e003)
    device = torch.device("cuda:0")
    microbatch_candidates = tuple(int(value) for value in r6c["training"]["microbatch_candidates"])
    if any(32 % value or value % 2 for value in microbatch_candidates):
        raise RuntimeError("R6-C G1 microbatch candidates must divide its balanced technical batch")

    # Probe the largest preregistered model once; the selected microbatch is then common to all arms.
    formal_microbatch = None
    memory_attempts: list[dict[str, Any]] = []
    for candidate_microbatch in microbatch_candidates:
        model = build_r6c_student("XLarge", seed=int(r6c["models"]["training_seed"])).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        torch.cuda.reset_peak_memory_stats(0)
        try:
            values = _train_update(
                model, optimizer, x_cpu, truth_cpu, device=device, microbatch=candidate_microbatch,
                effective_batch=32, loss_config=loss_config,
            )
            torch.cuda.synchronize(0)
            formal_microbatch = candidate_microbatch
            memory_attempts.append({
                "microbatch": candidate_microbatch,
                "status": "PASS",
                "probe_loss": values["task_total"],
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
            })
            del optimizer, model
            torch.cuda.empty_cache()
            break
        except torch.cuda.OutOfMemoryError as error:
            memory_attempts.append({"microbatch": candidate_microbatch, "status": "OOM", "error": str(error)[:600]})
            del optimizer, model
            torch.cuda.empty_cache()
    if formal_microbatch is None:
        raise RuntimeError("R6-C XLarge technical batch OOMed at the minimum microbatch")

    arms: dict[str, Any] = {}
    failed_microbatches: list[dict[str, Any]] = []
    for name in R6C_MODEL_SPECS:
        model = build_r6c_student(name, seed=int(r6c["models"]["training_seed"])).to(device)
        initial_loss = _technical_batch_task_loss(
            model, x_cpu, truth_cpu, device=device,
            microbatch=formal_microbatch, loss_config=loss_config,
        )
        del model
        torch.cuda.empty_cache()

        result = None
        attempts: list[dict[str, Any]] = []
        for candidate_microbatch in (formal_microbatch, *[v for v in microbatch_candidates if v < formal_microbatch]):
            model = build_r6c_student(name, seed=int(r6c["models"]["training_seed"])).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            torch.cuda.reset_peak_memory_stats(0)
            current_loss = initial_loss
            started = time.perf_counter()
            try:
                for step in range(200):
                    lr = 1e-3
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    _train_update(
                        model, optimizer, x_cpu, truth_cpu, device=device,
                        microbatch=candidate_microbatch, effective_batch=32, loss_config=loss_config,
                    )
                current_loss = _technical_batch_task_loss(
                    model, x_cpu, truth_cpu, device=device,
                    microbatch=candidate_microbatch, loss_config=loss_config,
                )
                torch.cuda.synchronize(0)
                result = {
                    "status": "PASS" if math.isfinite(current_loss) and current_loss < initial_loss else "FAIL",
                    "initial_task_loss": initial_loss,
                    "final_task_loss": current_loss,
                    "task_loss_ratio": current_loss / initial_loss,
                    "updates": 200,
                    "technical_batch_cpi": 32,
                    "microbatch": candidate_microbatch,
                    "training_seconds": time.perf_counter() - started,
                    "parameter_counts": validate_r6c_student(model, name),
                    "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
                    "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
                }
                del optimizer, model
                torch.cuda.empty_cache()
                break
            except torch.cuda.OutOfMemoryError as error:
                attempts.append({"microbatch": candidate_microbatch, "status": "OOM", "error": str(error)[:600]})
                del optimizer, model
                torch.cuda.empty_cache()
        if result is None:
            raise RuntimeError(f"R6-C G1 {name} OOMed at all available microbatch sizes")
        result["oom_attempts"] = attempts
        arms[name] = result

    free_vram, total_vram = torch.cuda.mem_get_info(0)
    passed = all(value["status"] == "PASS" for value in arms.values())
    record = {
        "status": "G1_PASS" if passed else "G1_FAIL",
        "git_sha": head,
        "technical_only": True,
        "weights_discarded": True,
        "test_seed": G1_TEST_SEED,
        "technical_pairs": 16,
        "technical_batch_cpi": 32,
        "updates_per_model": 200,
        "models_sequential": True,
        "formal_effective_batch_cpi": int(r6c["training"]["effective_batch_size_cpi"]),
        "formal_microbatch": int(formal_microbatch),
        "formal_microbatch_probe": memory_attempts,
        "gpu": torch.cuda.get_device_name(0),
        "device": "cuda:0",
        "free_vram_after_bytes": int(free_vram),
        "total_vram_bytes": int(total_vram),
        "arms": arms,
        "formal_data_generated": False,
        "final_blind_generated": False,
    }
    _write_json(output, record)
    if not passed:
        raise RuntimeError("R6-C G1 technical overfit/finite check did not pass for all widths")
    return record


def audit_cache() -> dict[str, Any]:
    """Verify generated cache shapes, scene IDs, and exact core nesting."""
    _require_preflight()
    r6c, r6a, _, _ = _configs()
    plans = _all_split_plans(r6c, r6a)
    train = _load_pair_cache("train_extended")
    validation = _load_pair_cache("validation")
    train_ids = np.asarray(train["scene_id"]).astype(str)
    validation_ids = np.asarray(validation["scene_id"]).astype(str)
    if len(set(train_ids)) != 102_400 or len(set(validation_ids)) != 5_000:
        raise RuntimeError("R6-C cache scene IDs are not unique")
    core_index = train_core_indices(train)
    if len(core_index) != 51_200 or len(set(train_ids[core_index])) != 51_200:
        raise RuntimeError("R6-C Train core cache index is incomplete or duplicated")
    if set(train_ids) & set(validation_ids):
        raise RuntimeError("R6-C materialized train and validation caches overlap")
    expected_condition = np.asarray([row["condition_index"] for row in plans["train_extended"] for _ in range(int(row["count"]))], dtype=np.uint8)
    if not np.array_equal(np.asarray(train["condition_index"]), expected_condition):
        raise RuntimeError("R6-C training cache condition ordering differs from the registered plan")
    expected_train_ids = np.asarray([
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in plans["train_extended"]
        for index in range(int(row["count"]))
    ])
    expected_validation_ids = np.asarray([
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in plans["validation"]
        for index in range(int(row["count"]))
    ])
    if not np.array_equal(train_ids, expected_train_ids) or not np.array_equal(validation_ids, expected_validation_ids):
        raise RuntimeError("R6-C cached scene IDs or ordering differ from registered seed plans")
    record = {
        "status": "CACHE_PASS",
        "git_sha": _git("rev-parse", "HEAD"),
        "train_extended_pairs": len(train_ids),
        "train_core_pairs": len(core_index),
        "validation_pairs": len(validation_ids),
        "core_is_exact_scene_id_subset": True,
        "train_validation_disjoint": True,
        "scene_ids_unique": True,
        "training_cache_bytes": sum(path.stat().st_size for path in _paired_cache_paths("train_extended").values()),
        "validation_cache_bytes": sum(path.stat().st_size for path in _paired_cache_paths("validation").values()),
    }
    _write_json(STAGE / "cache_audit.json", record)
    return record


def run_scale_training() -> dict[str, Any]:
    """Train three model scales, then continue the matched pair only through gates."""
    _require_preflight()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-C scale training requires passing G1")
    cache_audit_path = STAGE / "cache_audit.json"
    if not cache_audit_path.is_file() or json.loads(cache_audit_path.read_text(encoding="utf-8")).get("status") != "CACHE_PASS":
        raise RuntimeError("R6-C scale training requires a passing paired-cache audit")
    extended = _training_arrays("train_extended")
    train = _core_view(extended)
    validation = _training_arrays("validation")
    microbatch = int(g1["formal_microbatch"])
    records: dict[str, Any] = {}
    for name in R6C_MODEL_SPECS:
        records[name] = _run_training(
            name, "train_core", train, validation,
            target_updates=12_800, seed=404, microbatch=microbatch,
        )
    best_large = r6c_select_extension_candidate({
        name: {"task_loss": float(records[name]["selected_validation"]["task_loss"])}
        for name in ("Large", "XLarge")
    })
    pair = ("Medium", best_large)
    histories = {
        name: _continuation_history(RUNS / "train_core" / name / "training_history.csv")
        for name in pair
    }
    first_gates = {
        name: r6c_extension_gate(history, start_update=11_200, end_update=12_800)
        for name, history in histories.items()
    }
    decisions = [{
        "from_update": 12_800,
        "extension_pair": list(pair),
        "gates": first_gates,
        **r6c_extension_pair_decision(first_gates["Medium"], first_gates[best_large]),
    }]
    if decisions[-1]["continue_both"]:
        for name in pair:
            records[name] = _run_training(
                name, "train_core", train, validation,
                target_updates=25_600, seed=404, microbatch=microbatch,
            )
        histories = {
            name: _continuation_history(RUNS / "train_core" / name / "training_history.csv")
            for name in pair
        }
        second_gates = {
            name: r6c_extension_gate(history, start_update=24_000, end_update=25_600)
            for name, history in histories.items()
        }
        decisions.append({
            "from_update": 25_600,
            "extension_pair": list(pair),
            "gates": second_gates,
            **r6c_extension_pair_decision(second_gates["Medium"], second_gates[best_large]),
        })
        if decisions[-1]["continue_both"]:
            for name in pair:
                records[name] = _run_training(
                    name, "train_core", train, validation,
                    target_updates=51_200, seed=404, microbatch=microbatch,
                )
    _write_json(STAGE / "extension_decisions.json", {
        "git_sha": _git("rev-parse", "HEAD"),
        "selection_basis": "minimum common validation truth task loss only",
        "selected_large_candidate": best_large,
        "extension_pair": list(pair),
        "decisions": decisions,
        "h0_calibration_read": False,
        "h0_verification_read": False,
        "h1_development_read": False,
    })
    _write_json(STAGE / "scale_training.json", {
        "git_sha": _git("rev-parse", "HEAD"),
        "models": records,
        "extension_pair": list(pair),
        "selected_large_candidate": best_large,
        "decisions": decisions,
        "g1_formal_microbatch": microbatch,
        "effective_batch": 64,
    })
    return records


def run_data_scale_training() -> dict[str, Any]:
    """Train only the 102,400-pair Medium arm; core arm is the matched scale pilot."""
    _require_preflight()
    scale_path = STAGE / "scale_training.json"
    if not scale_path.is_file():
        raise RuntimeError("R6-C data-scale comparison follows the completed model-scale pilot")
    core_record_path = RUNS / "train_core" / "Medium" / "stage_record_12800.json"
    if not core_record_path.is_file():
        raise RuntimeError("R6-C data-scale contrast requires the frozen Medium core 12,800-update record")
    core_record = json.loads(core_record_path.read_text(encoding="utf-8"))
    if int(core_record["updates_completed"]) != 12_800:
        raise RuntimeError("R6-C data-scale contrast requires the Medium core 12,800-update checkpoint")
    extended = _training_arrays("train_extended")
    validation = _training_arrays("validation")
    microbatch = int(json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))["formal_microbatch"])
    record = _run_training(
        "Medium", "train_extended", extended, validation,
        target_updates=12_800, seed=404, microbatch=microbatch,
    )
    data_scale = {
        "git_sha": _git("rev-parse", "HEAD"),
        "core_arm": core_record,
        "extended_arm": record,
        "unique_pairs": {"core": 51_200, "extended": 102_400},
        "updates": {"core": 12_800, "extended": 12_800},
        "effective_batch": {"core": 64, "extended": 64},
        "total_examples_exposed": {"core": 819_200, "extended": 819_200},
        "fixed_dataset_passes_h1": {
            "core": 8.0,
            "extended": 4.0,
        },
        "same_initialization_seed": 404,
        "final_blind": False,
    }
    _write_json(STAGE / "data_scale_training.json", data_scale)
    return data_scale


def _prediction_owner(tag: str, split: str, head: str, count: int) -> dict[str, Any]:
    return {
        "owner": "D2-A-R6-C-student-predictions",
        "git_sha": head,
        "tag": tag,
        "split": split,
        "scene_count": int(count),
        "protocol_sha256": hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest(),
    }


def _prediction_work_arrays(
    directory: Path, count: int, owner: Mapping[str, Any]
) -> dict[str, np.memmap]:
    owner_path = directory / "owner.json"
    if directory.exists():
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != dict(owner):
            raise RuntimeError(f"refusing to reuse an unowned prediction work directory: {directory}")
    else:
        directory.mkdir(parents=True)
        _write_json(owner_path, dict(owner))
    shapes = {
        "scene_id": ((count,), "<U48"),
        "seed": ((count,), np.int64),
        "condition_index": ((count,), np.uint8),
        "score": ((count,), np.float32),
        "predicted_range_cell": ((count,), np.float32),
        "predicted_doppler_hz": ((count,), np.float32),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
    }
    arrays: dict[str, np.memmap] = {}
    for name, (shape, dtype) in shapes.items():
        path = directory / f"{name}.npy"
        if not path.is_file():
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()
        arrays[name] = np.load(path, mmap_mode="r+")
        if arrays[name].shape != shape or arrays[name].dtype != np.dtype(dtype):
            raise RuntimeError(f"R6-C prediction work array {name} has an unexpected shape/dtype")
    return arrays


def _append_eval_manifest(
    split: str, plan: Sequence[Mapping[str, Any]], scene_ids: np.ndarray
) -> None:
    path = STAGE / f"scene_manifest_{split}.csv"
    owner_path = STAGE / f"scene_manifest_{split}.owner.json"
    expected_owner = {
        "owner": "D2-A-R6-C-evaluation-manifest",
        "split": split,
        "protocol_sha256": hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest(),
    }
    expected_ids = [
        f"d2a-scene-{int(spec['seed'])}-{index}"
        for spec in plan
        for index in range(int(spec["count"]))
    ]
    if len(expected_ids) != len(scene_ids) or not np.array_equal(np.asarray(expected_ids), scene_ids.astype(str)):
        raise RuntimeError(f"R6-C {split} prediction scene IDs differ from registered order")
    if path.exists() or owner_path.exists():
        if not path.is_file() or not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != expected_owner:
            raise RuntimeError(f"refusing to overwrite an unowned R6-C evaluation manifest: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            existing_ids = [row["scene_id"] for row in csv.DictReader(handle)]
        if existing_ids != expected_ids:
            raise RuntimeError(f"existing R6-C {split} scene manifest differs from registered plan")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("scene_id", "seed", "condition_index", "course", "scenario_id", "presence"))
        offset = 0
        for spec in plan:
            for index in range(int(spec["count"])):
                writer.writerow((
                    expected_ids[offset], int(spec["seed"]), int(spec["condition_index"]),
                    str(spec["condition"]["course"]), str(spec["condition"]["scenario_id"]),
                    str(spec["presence"]),
                ))
                offset += 1
    _write_json(owner_path, expected_owner)


def evaluate_student_split(
    tag: str,
    model_name: str,
    checkpoint_path: Path,
    split: str,
    *,
    run_data: str,
) -> dict[str, Any]:
    """Generate one independent evaluation split and score it on CUDA only."""
    preflight, head = _require_preflight()
    r6c, r6a, physics, _ = _configs()
    if split not in {"h0_calibration", "h0_verification", "h1_development"}:
        raise ValueError("unknown R6-C evaluation split")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"R6-C checkpoint missing: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model") != model_name or checkpoint.get("data") != run_data:
        raise RuntimeError("R6-C evaluation checkpoint owner mismatch")
    if checkpoint.get("owner", {}).get("git_sha") != head:
        raise RuntimeError("R6-C evaluation checkpoint was trained under a different source commit")

    plan = _all_split_plans(r6c, r6a)[split]
    count = sum(int(row["count"]) for row in plan)
    final_path = PREDICTIONS / tag / f"{split}.npz"
    work = PREDICTIONS / ".work" / f"{tag}_{split}"
    owner = _prediction_owner(tag, split, head, count)
    if final_path.is_file():
        with np.load(final_path, allow_pickle=False) as existing:
            stored_owner = json.loads(str(existing["owner_json"].item()))
            if stored_owner != owner or len(existing["scene_id"]) != count:
                raise RuntimeError("R6-C completed prediction archive identity mismatch")
        return {"tag": tag, "split": split, "scenes": count, "archive": str(final_path.relative_to(ROOT)), "reused": True}

    progress_path = work / "progress.json"
    arrays = _prediction_work_arrays(work, count, owner)
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("R6-C evaluation progress owner mismatch")
        chunk_start = int(progress["chunks_done"])
        offset = int(progress["rows_done"])
        if not 0 <= chunk_start <= len(plan) or offset != sum(int(row["count"]) for row in plan[:chunk_start]):
            raise RuntimeError("R6-C evaluation progress cursor differs from the registered split plan")
    else:
        chunk_start = offset = 0
        progress = {"owner": owner, "chunks_done": 0, "rows_done": 0}
        _write_json(progress_path, progress)

    model = build_r6c_student(model_name).to("cuda:0").eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    from radardistill.simulation.d2a import generate_d2a_paired_batch

    for chunk_index, spec in enumerate(plan[chunk_start:], start=chunk_start):
        condition = spec["condition"]
        generated = generate_d2a_paired_batch(
            int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
            float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
        )
        presence = str(spec["presence"])
        iq = generated.iq_h0 if presence == "H0" else generated.iq_h1
        iq_ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
        x = torch.from_numpy(np.ascontiguousarray(iq_ri)).to("cuda:0")
        with torch.inference_mode():
            outputs = model(x, compute_spectrum=False)
            decoded = decode_task_outputs(outputs, float(r6c["radar"]["prf_hz"]))
        stop = offset + len(iq)
        arrays["scene_id"][offset:stop] = generated.base_scene_id.astype("<U48")
        arrays["seed"][offset:stop] = int(spec["seed"])
        arrays["condition_index"][offset:stop] = int(spec["condition_index"])
        arrays["score"][offset:stop] = decoded["presence_probability"].cpu().numpy().astype(np.float32)
        arrays["predicted_range_cell"][offset:stop] = decoded["range_cell"].cpu().numpy().astype(np.float32)
        arrays["predicted_doppler_hz"][offset:stop] = decoded["doppler_hz"].cpu().numpy().astype(np.float32)
        if presence == "H1":
            arrays["truth_range_cell"][offset:stop] = generated.truth_target_range_cell.astype(np.float32)
            arrays["truth_doppler_hz"][offset:stop] = generated.truth_target_doppler_hz.astype(np.float32)
        else:
            arrays["truth_range_cell"][offset:stop] = np.nan
            arrays["truth_doppler_hz"][offset:stop] = np.nan
        for value in arrays.values():
            value.flush()
        offset = stop
        progress = {"owner": owner, "chunks_done": chunk_index + 1, "rows_done": offset}
        _write_json(progress_path, progress)
        if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
            print(json.dumps({
                "stage": "evaluate", "tag": tag, "split": split,
                "chunks_done": chunk_index + 1, "chunks_total": len(plan), "scenes_done": offset,
            }, ensure_ascii=False), flush=True)
    if offset != count:
        raise RuntimeError(f"R6-C evaluation generated {offset} rows; expected {count}")
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    packed = {name: np.asarray(value).copy() for name, value in arrays.items()}
    packed["owner_json"] = np.asarray(json.dumps(owner, sort_keys=True))
    packed["split"] = np.asarray(split)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = final_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    temporary.replace(final_path)
    _append_eval_manifest(split, plan, packed["scene_id"])
    environment = {
        "git_sha": head,
        "tag": tag,
        "data_arm": run_data,
        "model": model_name,
        "split": split,
        "gpu": torch.cuda.get_device_name(0),
        "device": "cuda:0",
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "elapsed_seconds": elapsed,
        "scenes": count,
        "evaluation_iq_persisted": False,
        "final_blind": False,
        "preflight_sha": preflight["git_sha"],
    }
    _write_json(PREDICTIONS / tag / f"environment_{split}.json", environment)
    del model, arrays, packed
    torch.cuda.empty_cache()
    owner_path = work / "owner.json"
    if owner_path.is_file() and json.loads(owner_path.read_text(encoding="utf-8")) == owner:
        shutil.rmtree(work)
    return {"tag": tag, "split": split, "scenes": count, "archive": str(final_path.relative_to(ROOT)), "elapsed_seconds": elapsed}


def _load_prediction(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _exact_upper(events: int, trials: int, alpha: float = 0.05) -> float:
    if trials <= 0 or not 0 <= events <= trials:
        raise ValueError("invalid exact-binomial inputs")
    if events == trials:
        return 1.0
    from scipy.stats import beta

    return float(beta.ppf(1.0 - alpha, events + 1, trials - events))


def _bootstrap_interval(
    differences: np.ndarray, *, replicates: int, seed: int, confidence: float
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired bootstrap input must be finite nonempty aligned differences")
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        boot[index] = np.mean(values[rng.integers(0, len(values), size=len(values))])
    tail = (1.0 - confidence) / 2.0
    return {
        "difference": float(np.mean(values)),
        "bootstrap_ci": [float(value) for value in np.quantile(boot, [tail, 1.0 - tail])],
        "confidence": float(confidence),
        "replicates": int(replicates),
        "seed": int(seed),
        "unit": "base_scene_id",
    }


def _paired_boolean_difference(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)


def _summarize_prediction_tag(tag: str, target_pfas: Sequence[float] = (0.05, 0.01)) -> dict[str, Any]:
    r6c, r6a, _, _ = _configs()
    conditions = r6a["radar"]["conditions"]
    cal = _load_prediction(PREDICTIONS / tag / "h0_calibration.npz")
    ver = _load_prediction(PREDICTIONS / tag / "h0_verification.npz")
    h1 = _load_prediction(PREDICTIONS / tag / "h1_development.npz")
    expected_ids = np.asarray(h1["scene_id"]).astype(str)
    if len(set(expected_ids)) != len(expected_ids):
        raise RuntimeError(f"R6-C {tag} H1 development scene IDs are duplicated")
    result: dict[str, Any] = {
        "tag": tag,
        "sample_counts": {
            "h0_calibration": len(cal["score"]),
            "h0_verification": len(ver["score"]),
            "h1_development": len(h1["score"]),
        },
        "roc_auc_global": float(empirical_auc(ver["score"], h1["score"])),
        "roc_auc_by_condition": {},
        "workpoints": {},
    }
    for index in range(len(conditions)):
        mask0 = ver["condition_index"] == index
        mask1 = h1["condition_index"] == index
        result["roc_auc_by_condition"][str(index)] = float(empirical_auc(ver["score"][mask0], h1["score"][mask1]))

    for target_index, target_pfa in enumerate(target_pfas):
        threshold = float(np.quantile(cal["score"], 1.0 - target_pfa, method="higher"))
        false_alarms = ver["score"] >= threshold
        detected = h1["score"] >= threshold
        condition_pfa: dict[str, Any] = {}
        condition_h1: dict[str, Any] = {}
        for index, condition in enumerate(conditions):
            mask0 = ver["condition_index"] == index
            n0 = int(mask0.sum())
            k0 = int(np.count_nonzero(false_alarms[mask0]))
            mask1 = h1["condition_index"] == index
            summary = summarize_parameter_estimator(
                h1["predicted_range_cell"][mask1], h1["truth_range_cell"][mask1],
                h1["predicted_doppler_hz"][mask1], h1["truth_doppler_hz"][mask1],
                detected[mask1], prf_hz=float(r6c["radar"]["prf_hz"]),
            )
            condition_pfa[str(index)] = {
                "course": condition["course"],
                "n": n0,
                "false_alarms": k0,
                "pfa": k0 / n0,
                "pfa_exact95_upper": _exact_upper(k0, n0),
            }
            if condition["course"] == "C3":
                condition_pfa[str(index)]["pfa_bonferroni_familywise_upper"] = _exact_upper(
                    k0, n0, alpha=0.05 / 6.0
                )
            condition_h1[str(index)] = {
                "n": int(mask1.sum()),
                "pd": float(np.mean(detected[mask1])),
                "joint_success": summary["joint_success"],
                "range_rmse_cells": summary["all_h1"]["range_rmse_cells"],
                "doppler_rmse_hz": summary["all_h1"]["doppler_rmse_hz"],
                "detected_correct_candidate": summary["detected_correct_candidate"],
            }
        pooled = summarize_parameter_estimator(
            h1["predicted_range_cell"], h1["truth_range_cell"],
            h1["predicted_doppler_hz"], h1["truth_doppler_hz"], detected,
            prf_hz=float(r6c["radar"]["prf_hz"]),
        )
        identifiable = np.asarray([condition["course"] in {"C2", "C3"} for condition in conditions], dtype=bool)
        h1_identifiable = np.isin(h1["condition_index"], np.flatnonzero(identifiable))
        c3_cell_indices = [index for index, condition in enumerate(conditions) if condition["course"] == "C3"]
        c3_h0 = np.isin(ver["condition_index"], c3_cell_indices)
        c3_events = int(false_alarms[c3_h0].sum())
        c3_n = int(c3_h0.sum())
        c3_cell_upper_ok = all(
            condition_pfa[str(index)]["pfa_bonferroni_familywise_upper"] <= 0.10
            for index in c3_cell_indices
        )
        result["workpoints"][f"{target_pfa:g}"] = {
            "target_pfa": float(target_pfa),
            "calibration": {
                "threshold": threshold,
                "n": len(cal["score"]),
                "pfa_ge": float(np.mean(cal["score"] >= threshold)),
                "tie_count": int(np.count_nonzero(cal["score"] == threshold)),
            },
            "verification": {
                "n": len(ver["score"]),
                "false_alarms": int(false_alarms.sum()),
                "pfa": float(np.mean(false_alarms)),
                "pfa_exact95_upper": _exact_upper(int(false_alarms.sum()), len(false_alarms)),
                "by_condition": condition_pfa,
                "c3_pooled": {
                    "n": c3_n,
                    "false_alarms": c3_events,
                    "pfa": c3_events / c3_n,
                    "pfa_exact95_upper": _exact_upper(c3_events, c3_n),
                    "all_six_cells_upper_le_0_10": c3_cell_upper_ok,
                },
            },
            "h1": {
                "n": len(h1["score"]),
                "pd": float(np.mean(detected)),
                "auc_global": result["roc_auc_global"],
                "parameters": pooled,
                "identifiable_c2_c3": {
                    "n": int(h1_identifiable.sum()),
                    "joint_success": float(np.mean(
                        (np.abs(h1["predicted_range_cell"][h1_identifiable] - h1["truth_range_cell"][h1_identifiable]) <= 1.0)
                        & (np.abs(wrapped_error(
                            h1["predicted_doppler_hz"][h1_identifiable],
                            h1["truth_doppler_hz"][h1_identifiable],
                            float(r6c["radar"]["prf_hz"]),
                        )) <= 20.0)
                        & detected[h1_identifiable]
                    )),
                    "by_condition": {str(index): condition_h1[str(index)] for index in range(len(conditions)) if conditions[index]["course"] in {"C2", "C3"}},
                },
                "by_condition": condition_h1,
            },
        }
    result["formal_seed_advancement_gate_at_0_05"] = {
        "c3_pooled_pfa_upper_le_0_10": result["workpoints"]["0.05"]["verification"]["c3_pooled"]["pfa_exact95_upper"] <= 0.10,
        "all_six_c3_cell_pfa_upper_le_0_10": result["workpoints"]["0.05"]["verification"]["c3_pooled"]["all_six_cells_upper_le_0_10"],
    }
    return result


def run_evaluations() -> dict[str, Any]:
    """Evaluate scale and data-size checkpoints on shared deterministic fresh scenes."""
    _require_preflight()
    scale_path = STAGE / "scale_training.json"
    data_path = STAGE / "data_scale_training.json"
    if not scale_path.is_file() or not data_path.is_file():
        raise RuntimeError("R6-C development evaluation requires completed scale and data-scale pilots")
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    data_scale = json.loads(data_path.read_text(encoding="utf-8"))
    evaluation_arms = _evaluation_arm_plan(scale, data_scale)
    split_records: dict[str, Any] = {}
    for tag, (model_name, data_name, checkpoint) in evaluation_arms.items():
        split_records[tag] = {}
        for split in ("h0_calibration", "h0_verification", "h1_development"):
            split_records[tag][split] = evaluate_student_split(
                tag, model_name, checkpoint, split, run_data=data_name
            )
    summaries = {tag: _summarize_prediction_tag(tag) for tag in evaluation_arms}
    replicates = int(_configs()[0]["evaluation"]["paired_bootstrap_replicates"])
    bootstrap_seed = int(_configs()[0]["data"]["pair_assignment_seed"]) + 1000

    def paired_comparison(reference: str, candidate: str, confidence: float, seed_offset: int) -> dict[str, Any]:
        ref = _load_prediction(PREDICTIONS / reference / "h1_development.npz")
        cand = _load_prediction(PREDICTIONS / candidate / "h1_development.npz")
        if not np.array_equal(ref["scene_id"], cand["scene_id"]):
            raise RuntimeError(f"R6-C paired comparison scene IDs differ: {reference} vs {candidate}")
        r6c, r6a, _, _ = _configs()
        cond_course = np.asarray([r6a["radar"]["conditions"][int(i)]["course"] for i in ref["condition_index"]])
        identifiable = np.isin(cond_course, ("C2", "C3"))
        summaries = {}
        for target_index, target in enumerate((0.05, 0.01)):
            ref_threshold = summaries_threshold(reference, target)
            cand_threshold = summaries_threshold(candidate, target)
            ref_detected = ref["score"] >= ref_threshold
            cand_detected = cand["score"] >= cand_threshold
            ref_joint = ref_detected & (np.abs(ref["predicted_range_cell"] - ref["truth_range_cell"]) <= 1.0) & (
                np.abs(wrapped_error(ref["predicted_doppler_hz"], ref["truth_doppler_hz"], float(r6c["radar"]["prf_hz"]))) <= 20.0
            )
            cand_joint = cand_detected & (np.abs(cand["predicted_range_cell"] - cand["truth_range_cell"]) <= 1.0) & (
                np.abs(wrapped_error(cand["predicted_doppler_hz"], cand["truth_doppler_hz"], float(r6c["radar"]["prf_hz"]))) <= 20.0
            )
            mask = identifiable
            summaries[f"{target:g}"] = {
                "identifiable_c2_c3_joint_success": _bootstrap_interval(
                    _paired_boolean_difference(cand_joint[mask], ref_joint[mask]),
                    replicates=replicates,
                    seed=bootstrap_seed + seed_offset + target_index,
                    confidence=confidence,
                ),
                "global_pd": _bootstrap_interval(
                    _paired_boolean_difference(cand_detected, ref_detected),
                    replicates=replicates,
                    seed=bootstrap_seed + 100 + seed_offset + target_index,
                    confidence=0.95,
                ),
                "c2_c3_range_rmse_ratio": _paired_metric_ratio_summary(
                    cand["predicted_range_cell"][mask] - cand["truth_range_cell"][mask],
                    ref["predicted_range_cell"][mask] - ref["truth_range_cell"][mask],
                    replicates=replicates,
                    seed=bootstrap_seed + 200 + seed_offset + target_index,
                    confidence=0.95,
                ),
                "c2_c3_doppler_rmse_ratio": _paired_metric_ratio_summary(
                    wrapped_error(cand["predicted_doppler_hz"][mask], cand["truth_doppler_hz"][mask], float(r6c["radar"]["prf_hz"])),
                    wrapped_error(ref["predicted_doppler_hz"][mask], ref["truth_doppler_hz"][mask], float(r6c["radar"]["prf_hz"])),
                    replicates=replicates,
                    seed=bootstrap_seed + 300 + seed_offset + target_index,
                    confidence=0.95,
                ),
            }
        return {"reference": reference, "candidate": candidate, "workpoints": summaries}

    def summaries_threshold(tag: str, target: float) -> float:
        cal = _load_prediction(PREDICTIONS / tag / "h0_calibration.npz")
        return float(np.quantile(cal["score"], 1.0 - target, method="higher"))

    model_comparisons = {
        name: paired_comparison("scale_Medium", f"scale_{name}", 0.975, index * 1000)
        for index, name in enumerate(("Large", "XLarge"), start=1)
    }
    data_comparison = paired_comparison("scale_Medium", "data_extended_Medium", 0.95, 4000)
    extension_pair = list(scale["extension_pair"])
    training_budget_comparisons = {
        name: paired_comparison(f"scale_{name}", f"scale_extended_{name}", 0.95, 5000 + index * 1000)
        for index, name in enumerate(extension_pair)
        if f"scale_extended_{name}" in summaries
    }
    gate_config = _configs()[0]["evaluation"]
    for name, comparison in model_comparisons.items():
        primary = comparison["workpoints"]["0.05"]
        candidate = summaries[f"scale_{name}"]["workpoints"]["0.05"]
        joint_ci = primary["identifiable_c2_c3_joint_success"]["bootstrap_ci"]
        pd_ci = primary["global_pd"]["bootstrap_ci"]
        range_ratio_ci = primary["c2_c3_range_rmse_ratio"]["bootstrap_ci"]
        doppler_ratio_ci = primary["c2_c3_doppler_rmse_ratio"]["bootstrap_ci"]
        comparison["formal_seed_advancement_gate"] = {
            "joint_point_gain_at_least_0_02": primary["identifiable_c2_c3_joint_success"]["difference"]
            >= float(gate_config["meaningful_joint_success_gain"]),
            "bonferroni_joint_ci_lower_above_zero": joint_ci[0] > 0.0,
            "overall_pd_lower_ge_minus_0_02": pd_ci[0] >= -float(gate_config["max_overall_pd_degradation"]),
            "range_rmse_ratio_upper_le_1_05": range_ratio_ci[1] <= float(gate_config["max_range_or_doppler_rmse_ratio_upper"]),
            "doppler_rmse_ratio_upper_le_1_05": doppler_ratio_ci[1] <= float(gate_config["max_range_or_doppler_rmse_ratio_upper"]),
            "c3_pooled_pfa_upper_le_0_10": candidate["verification"]["c3_pooled"]["pfa_exact95_upper"]
            <= float(gate_config["c3_pooled_pfa_upper_max_for_formal_seed_advancement"]),
            "all_six_c3_cell_pfa_familywise_upper_le_0_10": candidate["verification"]["c3_pooled"]["all_six_cells_upper_le_0_10"],
        }
        comparison["formal_seed_advancement_gate"]["pass"] = all(
            comparison["formal_seed_advancement_gate"].values()
        )
    result = {
        "git_sha": _git("rev-parse", "HEAD"),
        "protocol_version": 1.2,
        "models": summaries,
        "model_scale_comparisons": model_comparisons,
        "training_budget_comparisons": training_budget_comparisons,
        "data_scale_comparison": data_comparison,
        "prediction_runs": split_records,
        "bootstrap": {"replicates": replicates, "seed_base": bootstrap_seed, "unit": "base_scene_id"},
        "final_blind": False,
    }
    _write_json(STAGE / "metrics.json", result)
    return result


def _evaluation_arm_plan(
    scale_training: Mapping[str, Any], data_scale_training: Mapping[str, Any]
) -> dict[str, tuple[str, str, Path]]:
    """Use the common 12,800-update point for scale and data comparisons.

    The optional longer-budget arms are evaluated separately at the same final
    update for the matched pair admitted by the validation-only extension gate.
    """
    runs = RUNS
    models = scale_training["models"]
    arms: dict[str, tuple[str, str, Path]] = {
        f"scale_{name}": (
            name,
            "train_core",
            runs / "train_core" / name / "best_checkpoint_12800.pt",
        )
        for name in R6C_MODEL_SPECS
    }
    extension_pair = list(scale_training["extension_pair"])
    if len(extension_pair) != 2 or extension_pair[0] != "Medium" or extension_pair[1] not in {"Large", "XLarge"}:
        raise RuntimeError("R6-C extension must contain Medium and exactly one preselected large candidate")
    pair_updates = [int(models[name]["updates_completed"]) for name in extension_pair]
    if pair_updates[0] != pair_updates[1]:
        raise RuntimeError("R6-C matched extension pair has different completed update budgets")
    extension_updates = pair_updates[0]
    if extension_updates not in {12_800, 25_600, 51_200}:
        raise RuntimeError("R6-C extension endpoint is outside the registered matched budgets")
    for name in R6C_MODEL_SPECS:
        if name not in extension_pair and int(models[name]["updates_completed"]) != 12_800:
            raise RuntimeError("R6-C unselected model was trained beyond the registered 12,800-update screen")
    if extension_updates > 12_800:
        for name in extension_pair:
            arms[f"scale_extended_{name}"] = (
                name,
                "train_core",
                runs / "train_core" / name / f"best_checkpoint_{extension_updates}.pt",
            )

    core_record = data_scale_training.get("core_arm", {})
    extended_record = data_scale_training.get("extended_arm", {})
    if int(core_record.get("updates_completed", -1)) != 12_800:
        raise RuntimeError("R6-C data-scale core must reuse the matched Medium 12,800-update arm")
    if int(extended_record.get("updates_completed", -1)) != 12_800:
        raise RuntimeError("R6-C data-scale extended arm must use the registered 12,800 updates")
    arms["data_extended_Medium"] = (
        "Medium",
        "train_extended",
        runs / "train_extended" / "Medium" / "best_checkpoint_12800.pt",
    )
    return arms


def _paired_metric_ratio_summary(
    candidate_error: np.ndarray,
    reference_error: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    candidate = np.asarray(candidate_error, dtype=np.float64)
    reference = np.asarray(reference_error, dtype=np.float64)
    if candidate.shape != reference.shape or candidate.ndim != 1 or not len(candidate):
        raise ValueError("paired RMSE errors must be nonempty aligned vectors")
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(candidate), size=len(candidate))
        candidate_rmse = float(np.sqrt(np.mean(candidate[selected] ** 2)))
        reference_rmse = float(np.sqrt(np.mean(reference[selected] ** 2)))
        estimates[index] = candidate_rmse / max(reference_rmse, 1e-12)
    tail = (1.0 - confidence) / 2.0
    return {
        "ratio": float(np.sqrt(np.mean(candidate ** 2)) / max(float(np.sqrt(np.mean(reference ** 2))), 1e-12)),
        "bootstrap_ci": [float(value) for value in np.quantile(estimates, [tail, 1.0 - tail])],
        "confidence": float(confidence),
        "replicates": replicates,
        "seed": seed,
        "unit": "base_scene_id",
    }


def analyze_stage() -> dict[str, Any]:
    """Write compact R6-C tables and figures from frozen predictions and histories."""
    metrics_path = STAGE / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("R6-C analysis requires completed prediction metrics")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    r6c, r6a, _, _ = _configs()
    figure_dir = STAGE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    colors = {"Medium": "#2864a5", "Large": "#c04435", "XLarge": "#2b8a5a"}
    for name in R6C_MODEL_SPECS:
        history_path = RUNS / "train_core" / name / "training_history.csv"
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        updates = np.asarray([int(row["update"]) for row in history], dtype=np.int64)
        losses = np.asarray([float(row["validation_task_loss"]) for row in history], dtype=np.float64)
        axes[0].plot(updates, losses, label=name, color=colors[name], linewidth=1.8)
    axes[0].set_xlabel("Optimizer updates")
    axes[0].set_ylabel("Validation task loss")
    axes[0].set_title("Matched width screen and shared extension")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    tags = ["scale_Medium", "scale_Large", "scale_XLarge", "data_extended_Medium"]
    labels = ["Medium 12.8k", "Large 12.8k", "XLarge 12.8k", "Extended data"]
    bar_colors = ["#2864a5", "#c04435", "#2b8a5a", "#e29d26"]
    for name in ("Medium", "Large", "XLarge"):
        extended_tag = f"scale_extended_{name}"
        if extended_tag in metrics["models"]:
            tags.append(extended_tag)
            labels.append(f"{name} extended")
            bar_colors.append(colors[name])
    for tag, label, color in zip(tags, labels, bar_colors, strict=True):
        row = metrics["models"][tag]["workpoints"]["0.05"]["h1"]["identifiable_c2_c3"]
        axes[1].bar(label, row["joint_success"], color=color)
    axes[1].set_ylabel("Joint success")
    axes[1].set_title("C2+C3 at global Pfa 0.05")
    axes[1].tick_params(axis="x", labelrotation=25)
    axes[1].grid(axis="y", alpha=0.25)

    scale_tags = ("scale_Medium", "scale_Large", "scale_XLarge")
    for tag in scale_tags:
        item = metrics["models"][tag]["workpoints"]["0.05"]["h1"]["by_condition"]
        doppler_by_snr: dict[float, list[float]] = {}
        for index, condition in enumerate(r6a["radar"]["conditions"]):
            doppler_by_snr.setdefault(float(condition["snr_db"]), []).append(
                float(item[str(index)]["doppler_rmse_hz"])
            )
        xs = sorted(doppler_by_snr)
        ys = [float(np.mean(doppler_by_snr[value])) for value in xs]
        axes[2].plot(xs, ys, marker="o", label=tag.removeprefix("scale_"), color=colors[tag.removeprefix("scale_")])
    axes[2].set_xlabel("SNR (dB)")
    axes[2].set_ylabel("Mean condition Doppler RMSE (Hz)")
    axes[2].set_title("Equal-weight condition average")
    axes[2].grid(alpha=0.25)
    axes[2].legend()
    fig.tight_layout()
    figure_path = figure_dir / "r6c_training_joint_doppler.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    lines = [
        "# D2-A-R6-C 分析",
        "",
        f"- 提交：`{metrics['git_sha']}`；协议v{metrics['protocol_version']}。",
        "- 结果来自单个screening训练seed404，所有评价均为development；不是final blind，也不估计训练seed方差。",
        "- 阈值仅从各模型相同的独立H0 calibration流分别确定；H0 verification未参与模型/checkpoint选择。",
        "- 模型规模的继续训练只使用Medium与validation筛出的一个大模型，且两臂共同过门才同步延长。",
        "",
        "## 模型与H0保护",
        "",
        "| Arm | Active parameters | Updates | C2+C3 joint @ global Pfa .05 | C3 pooled Pfa upper | All six C3 cell familywise upper ≤.10 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in R6C_MODEL_SPECS:
        tag = f"scale_{name}"
        metrics_row = metrics["models"][tag]["workpoints"]["0.05"]
        record_path = RUNS / "train_core" / name / "stage_record_12800.json"
        if not record_path.is_file():
            raise RuntimeError(f"R6-C analysis is missing the common-budget record: {record_path}")
        pfa = metrics_row["verification"]["c3_pooled"]
        lines.append(
            f"| {name} | {R6C_MODEL_SPECS[name]['parameters']:,} | 12,800 | "
            f"{metrics_row['h1']['identifiable_c2_c3']['joint_success']:.4f} | "
            f"{pfa['pfa_exact95_upper']:.4f} | {pfa['all_six_cells_upper_le_0_10']} |"
        )
    lines.extend(["", "## Matched模型比较", ""])
    for name, comparison in metrics["model_scale_comparisons"].items():
        primary = comparison["workpoints"]["0.05"]["identifiable_c2_c3_joint_success"]
        gate = comparison["formal_seed_advancement_gate"]
        lines.append(
            f"- {name}−Medium：C2+C3 joint差{primary['difference']:+.4f}，97.5%配对CI "
            f"`{primary['bootstrap_ci']}`；正式seed晋级门：**{'PASS' if gate['pass'] else 'FAIL'}**。"
        )
    lines.extend(["", "## 配对延长训练", ""])
    if metrics.get("training_budget_comparisons"):
        scale_training = json.loads((STAGE / "scale_training.json").read_text(encoding="utf-8"))
        for name, comparison in metrics["training_budget_comparisons"].items():
            primary = comparison["workpoints"]["0.05"]["identifiable_c2_c3_joint_success"]
            final_updates = int(scale_training["models"][name]["updates_completed"])
            lines.append(
                f"- {name}：同一seed、Train core与effective batch64从12,800共同延长到{final_updates:,}步；"
                f"C2+C3 joint差{primary['difference']:+.4f}，95%配对CI `{primary['bootstrap_ci']}`。"
            )
    else:
        lines.append("- Medium和validation选出的较大模型未共同通过延长门；不进行延长训练比较。")
    data_cmp = metrics["data_scale_comparison"]["workpoints"]["0.05"]["identifiable_c2_c3_joint_success"]
    lines.extend([
        "",
        "## 训练数据规模对照",
        "",
        f"Medium 102,400−51,200唯一训练对：C2+C3 joint差{data_cmp['difference']:+.4f}，95%配对CI `{data_cmp['bootstrap_ci']}`。两臂均12,800 updates、effective batch64、总CPI exposure相同；核心区别是4遍与8遍固定训练集遍历。数据规模对照的core臂直接复用模型规模实验中的Medium 12,800步checkpoint。",
        "",
        "## 解释边界",
        "",
        "- 单seed开发对照只能筛选候选，不能证明跨初始化稳定性或盲测泛化。",
        "- C3 pooled与各cell虚警保护必须独立满足；joint或RMSE改善不抵消条件虚警失败。",
        "- 若验证曲线在51,200前仍持续改善，只能称为预算内未观察平台，不推断已经收敛。",
        "- 等权SNR曲线对条件指标作描述性平均，不替代按场景配对的主指标。",
        "",
        f"图：`{figure_path.relative_to(ROOT)}`。",
        "",
    ])
    analysis_path = STAGE / "analysis.md"
    analysis_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {
        "status": "ANALYSIS_WRITTEN",
        "git_sha": metrics["git_sha"],
        "analysis": str(analysis_path.relative_to(ROOT)),
        "figures": [str(figure_path.relative_to(ROOT))],
        "protocol_version": r6c["experiment"]["protocol_version"],
    }
    _write_json(STAGE / "analysis_record.json", summary)
    return summary


def cleanup_owned_cache() -> dict[str, Any]:
    """Remove only the completed R6-C owned memmaps after compact results exist."""
    _require_preflight()
    required = (STAGE / "cache_audit.json", STAGE / "metrics.json", STAGE / "analysis.md")
    if any(not path.is_file() for path in required):
        raise RuntimeError("R6-C cache cleanup requires passing cache audit, metrics, and analysis")
    owner_dirs = (CACHE / "train_extended", CACHE / "validation")
    if not CACHE.is_dir() or any(not (path / "owner.json").is_file() for path in owner_dirs):
        raise RuntimeError("R6-C cache owner records are missing; refusing cleanup")
    for split, path in zip(("train_extended", "validation"), owner_dirs, strict=True):
        r6c, r6a, _, _ = _configs()
        plan = _all_split_plans(r6c, r6a)[split]
        expected = _cache_owner(split, _git("rev-parse", "HEAD"), plan)
        if json.loads((path / "owner.json").read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"R6-C {split} cache owner identity differs; refusing cleanup")
    unexpected = [path for path in CACHE.iterdir() if path.name not in {"train_extended", "validation"}]
    if unexpected:
        raise RuntimeError(f"R6-C cache contains unowned entries: {[item.name for item in unexpected]}")
    removed_bytes = sum(path.stat().st_size for path in CACHE.rglob("*") if path.is_file())
    shutil.rmtree(CACHE)
    project_bytes = int(subprocess.check_output(["du", "-sb", str(ROOT)], text=True).split()[0])
    limit = 20 * 1024**3
    record = {
        "status": "CLEANUP_PASS" if project_bytes < limit else "CLEANUP_PROJECT_LIMIT_NOT_MET",
        "removed_owned_cache_bytes": int(removed_bytes),
        "project_bytes_after_cleanup": project_bytes,
        "project_limit_bytes": limit,
        "only_owner_manifest_authorized_cache_removed": True,
        "user_untracked_files_deleted": False,
        "git_sha": _git("rev-parse", "HEAD"),
    }
    _write_json(STAGE / "cache_cleanup.json", record)
    if record["status"] != "CLEANUP_PASS":
        raise RuntimeError("R6-C cache was removed, but the complete project still exceeds 20 GiB")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "preflight", "g1", "generate-train", "generate-validation", "audit-cache",
        "train-scale", "train-data-scale", "evaluate", "analyze", "cleanup-cache",
    ))
    args = parser.parse_args()
    commands = {
        "preflight": preflight,
        "g1": run_g1,
        "generate-train": lambda: generate_paired_cache("train_extended"),
        "generate-validation": lambda: generate_paired_cache("validation"),
        "audit-cache": audit_cache,
        "train-scale": run_scale_training,
        "train-data-scale": run_data_scale_training,
        "evaluate": run_evaluations,
        "analyze": analyze_stage,
        "cleanup-cache": cleanup_owned_cache,
    }
    print(json.dumps(commands[args.command](), indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
