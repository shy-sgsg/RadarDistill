#!/usr/bin/env python3
"""Execute the frozen D2-A-R6-B low-Pfa / hard-negative branch R experiment."""

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
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
STAGE = ROOT / "experiments/D2-A-R6/R6-B"
CACHE = STAGE / "cache" / "training"
RUNS = STAGE / "runs" / "seed_303"
PREDICTIONS = STAGE / "predictions"
METHODS = ("B0", "B1", "B2", "B3", "R3_T1", "R3_T2", "Medium_Frozen",
           "Medium_TruthOnly", "Medium_HardNegative")
FREQUENCIES = ("fft_peak", "fft_interpolation", "fft_zero_pad_4x",
               "lag_one_phase_difference", "local_profile_ml")
R6B_BOOTSTRAP_SEED = 2026102914
R6B_OPTIMIZER = {
    "source": "R5-A capacity Medium two-segment schedule / E003 truth-task optimizer settings",
    "optimizer": "AdamW",
    "initial_updates": 6400,
    "final_updates": 12800,
    "learning_rate_initial": 0.001,
    "learning_rate_at_update_6400": 0.0001,
    "extension_learning_rate_final": 0.00001,
    "schedule": "two_segment_cosine_6400_plus_6400",
    "weight_decay": 0.0001,
    "gradient_clip_norm": 1.0,
    "validation_interval_updates": 400,
    "microbatch_initial": 32,
    "oom_microbatch_fallbacks": [16, 8, 4],
}

from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_r6a import (
    audit_seed_isolation,
    build_split_plan,
    calibrate_threshold,
    candidate_frequency_estimates,
    classical_detector_predictions,
    load_frozen_medium,
    predict_medium,
    summarize_parameter_estimator,
    wrapped_error,
)
from radardistill.training.d2a_e002 import scale_d2a_e002_microbatch_loss
from radardistill.training.d2a_r6b_hard_negative import (
    assemble_independent_h0_h1_batch,
    build_h0_sample_plan,
    build_hard_negative_pools,
    build_r6b_paired_split_plan,
    build_repeated_h1_plan,
    clopper_pearson_upper,
    required_zero_event_sample_size,
)


def _read_yaml(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    r6b = _read_yaml("experiments/D2-A-R6/R6-B/config.yaml")
    r6a = _read_yaml("experiments/D2-A-R6/R6-A/config.yaml")
    physics = _read_yaml(r6b["radar"]["source_config"])
    r3 = _read_yaml("experiments/D2-A-R3/config.yaml")
    return r6b, r6a, physics, r3


def _r6b_eval_config(config: Mapping[str, Any], r6a: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(config["data"])
    data["splits"] = {}
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        split = dict(config["data"]["splits"][name])
        split["scene_count"] = int(split["scenes"])
        data["splits"][name] = split
    return {"data": data, "radar": dict(r6a["radar"])}


def _all_split_plans(config: Mapping[str, Any], r6a: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    data = config["data"]
    conditions = r6a["radar"]["conditions"]
    result = {
        name: build_r6b_paired_split_plan(data, conditions, name)
        for name in ("train", "validation")
    }
    eval_config = _r6b_eval_config(config, r6a)
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        result[name] = build_split_plan(eval_config, name)
    return result


def _planned_ids(plans: Mapping[str, list[dict[str, Any]]]) -> set[str]:
    output: set[str] = set()
    for rows in plans.values():
        for row in rows:
            for index in range(int(row["count"])):
                scene_id = f"d2a-scene-{int(row['seed'])}-{index}"
                if scene_id in output:
                    raise ValueError(f"duplicate registered R6-B scene ID: {scene_id}")
                output.add(scene_id)
    return output


def _prior_scene_ids(paths: list[Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"required prior scene manifest missing: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            key = next((name for name in ("base_scene_id", "scene_id") if name in columns), None)
            if key is None:
                raise ValueError(f"prior manifest has no scene ID column: {path}")
            ids.update(row[key] for row in reader)
    return ids


def preflight() -> dict[str, Any]:
    """G0: verify frozen branch, source, IDs, disk, and CUDA before generating R6-B data."""
    config, r6a, _, _ = _configs()
    head = _git("rev-parse", "HEAD")
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R6-B requires execution on main")
    remote = _git("ls-remote", "origin", "refs/heads/main").split()
    if not remote or remote[0] != head or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("G0 requires the committed R6-B implementation pushed to origin/main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("G0 requires a clean tracked worktree")
    if config["experiment"]["status"] != "preregistered" or config["experiment"]["final_blind_authorized"]:
        raise RuntimeError("R6-B must remain preregistered and final blind disabled")
    selection = json.loads((STAGE / "branch_selection.json").read_text(encoding="utf-8"))
    if selection.get("selected_branch") != "R" or selection.get("final_blind_read_or_generated") is not False:
        raise RuntimeError("R6-B may run only the recorded training-only branch R")
    selection_sha = hashlib.sha256((STAGE / "branch_selection.json").read_bytes()).hexdigest()
    protocol_sha = hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest()
    config_sha = hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest()
    if int(config["training"]["effective_batch_size_cpi"]) != 64 or int(config["training"]["updates"]) != 12800:
        raise RuntimeError("frozen R6-B effective batch/update budget changed")
    if config["training"]["precision"] != "FP32" or config["resources"]["cpu_fallback"] != "forbidden":
        raise RuntimeError("R6-B training must remain FP32 CUDA-only")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-B G0 requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    gpu = torch.cuda.get_device_name(0)
    if gpu != "NVIDIA GeForce RTX 3050 Laptop GPU":
        raise RuntimeError(f"R6-B expects the RTX 3050 Laptop GPU, found {gpu}")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    if free_vram < 2 * 1024**3:
        raise RuntimeError("R6-B requires at least 2 GiB free CUDA memory for the frozen run")
    out = STAGE / "preflight.json"
    if out.is_file():
        existing = json.loads(out.read_text(encoding="utf-8"))
        if (
            existing.get("status") != "G0_PASS"
            or existing.get("git_sha") != head
            or existing.get("branch_selection_sha256") != selection_sha
            or existing.get("protocol_sha256") != protocol_sha
            or existing.get("config_sha256") != config_sha
            or existing.get("selected_branch") != "R"
        ):
            raise RuntimeError("existing R6-B G0 record does not match the pushed frozen source")
        # Keep the original capacity snapshot when resuming after cache generation.
        return existing
    disk_free = shutil.disk_usage(ROOT).free
    required = int(config["training"]["paired_train_validation_iq_estimated_bytes_if_branch_R"]) + 2 * 1024**3
    if disk_free < required:
        raise RuntimeError(f"R6-B needs {required} free bytes before creating its owned IQ cache")
    plans = _all_split_plans(config, r6a)
    seed_values = [int(row["seed"]) for rows in plans.values() for row in rows]
    if len(seed_values) != len(set(seed_values)):
        raise RuntimeError("R6-B train/validation/calibration/verification/development chunk seeds overlap")
    r6a_seeds = audit_seed_isolation(
        r6a,
        str(ROOT / "experiments/D2-A-R5/R5-C-teaching/seed_manifest.csv"),
        config,
    )
    known_paths = [
        ROOT / "experiments/D2-A-R6/R6-A/scene_manifest.csv",
        ROOT / "experiments/D2-A-R5/R5-C-teaching/scene_manifest.csv",
        ROOT / "experiments/D2-A-R5/R5-C/selected_scene_ids.csv",
        ROOT / "experiments/D2-A-E003/scene_manifest.csv",
        ROOT / "experiments/D2-A-R5/R5-A/scene_manifest.csv",
    ]
    planned = _planned_ids(plans)
    prior_ids = _prior_scene_ids(known_paths)
    overlap = planned & prior_ids
    if overlap:
        raise RuntimeError(f"R6-B scene ID overlap with prior stages: {sorted(overlap)[:3]}")
    record = {
        "status": "G0_PASS",
        "git_sha": head,
        "selected_branch": "R",
        "branch_selection_sha": selection["selection_git_sha"],
        "branch_selection_sha256": selection_sha,
        "protocol_sha256": protocol_sha,
        "config_sha256": config_sha,
        "training_constants": {
            **R6B_OPTIMIZER,
            "effective_batch_size_cpi": 64,
            "optimizer_updates": 12800,
            "h0_per_update": 32,
            "h1_per_update": 32,
            "pair_plan_seed": int(config["data"]["splits"]["train"]["assignment_seed"]),
            "paired_bootstrap_seed": R6B_BOOTSTRAP_SEED,
        },
        "gpu": gpu,
        "cuda_device": "cuda:0",
        "free_vram_bytes": int(free_vram),
        "total_vram_bytes": int(total_vram),
        "disk_free_bytes": int(disk_free),
        "training_cache_estimated_bytes": int(config["training"]["paired_train_validation_iq_estimated_bytes_if_branch_R"]),
        "planned_scene_count": len(planned),
        "prior_scene_count": len(prior_ids),
        "scene_id_overlap": 0,
        "r6a_seed_audit": r6a_seeds,
        "split_scene_counts": {name: sum(int(row["count"]) for row in rows) for name, rows in plans.items()},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "final_blind_generated": False,
    }
    out = STAGE / "preflight.json"
    _write_json(out, record)
    return record


def _require_frozen_preflight() -> tuple[dict[str, Any], str]:
    path = STAGE / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run R6-B G0 preflight before data generation")
    record = json.loads(path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    if record.get("git_sha") != head:
        raise RuntimeError("R6-B source changed after G0; commit and rerun preflight before continuing")
    config, _, _, _ = _configs()
    if (
        record.get("status") != "G0_PASS"
        or record.get("selected_branch") != "R"
        or record.get("protocol_sha256") != hashlib.sha256((STAGE / "protocol.md").read_bytes()).hexdigest()
        or record.get("config_sha256") != hashlib.sha256((STAGE / "config.yaml").read_bytes()).hexdigest()
        or record.get("branch_selection_sha256")
        != hashlib.sha256((STAGE / "branch_selection.json").read_bytes()).hexdigest()
        or config["experiment"]["final_blind_authorized"] is not False
    ):
        raise RuntimeError("R6-B G0 record, branch, protocol, or config identity is invalid")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-B requires the G0 commit on pushed main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked source changed after G0")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 3050 Laptop GPU":
        raise RuntimeError("R6-B requires the registered RTX 3050 CUDA device")
    return record, head


def _owner_cache_dir(path: Path, owner: dict[str, Any]) -> None:
    if path.exists():
        owner_path = path / "owner.json"
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
            raise RuntimeError(f"refusing to reuse an unowned or differently-owned cache: {path}")
        return
    path.mkdir(parents=True)
    _write_json(path / "owner.json", owner)


def _paired_cache_paths(split: str) -> dict[str, Path]:
    directory = CACHE / split
    return {name: directory / f"{name}.npy" for name in
            ("iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id")}


def generate_paired_cache(split: str) -> dict[str, Any]:
    """Generate only the branch-R paired train/validation IQ cache, with resumable chunks."""
    preflight, head = _require_frozen_preflight()
    if split not in {"train", "validation"}:
        raise ValueError("only paired train/validation cache generation is permitted")
    config, r6a, physics, _ = _configs()
    plans = _all_split_plans(config, r6a)
    plan = plans[split]
    count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    directory_existed = directory.exists()
    owner = {"owner": "D2-A-R6-B-training-cache", "git_sha": head, "split": split, "scene_count": count}
    _owner_cache_dir(directory, owner)
    paths = _paired_cache_paths(split)
    progress_path = directory / "progress.json"
    manifest_path = STAGE / f"scene_manifest_{split}.csv"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner:
            raise RuntimeError("paired cache progress belongs to another run")
        completed = int(progress["completed_chunks"])
        rows_written = int(progress["rows_written"])
        expected_rows = sum(int(row["count"]) for row in plan[:completed])
        if not 0 <= completed <= len(plan) or rows_written != expected_rows:
            raise RuntimeError("paired cache progress cursor differs from the registered chunk plan")
        if manifest_path.exists():
            with manifest_path.open("r+b") as handle:
                handle.truncate(int(progress["manifest_offset"]))
        elif rows_written != 0 or int(progress["manifest_offset"]) != 0:
            raise RuntimeError("paired cache progress exists without its completed scene manifest")
    else:
        unexpected = [path.name for path in directory.iterdir() if path.name != "owner.json"]
        if directory_existed and (unexpected or manifest_path.exists()):
            raise RuntimeError("partial paired cache has no valid progress record")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite an unowned scene manifest: {manifest_path}")
        completed = rows_written = 0
        progress = {"owner": owner, "completed_chunks": 0, "rows_written": 0, "manifest_offset": 0}
        # Write the resumable cursor before allocating the large memmaps.
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
    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="", encoding="utf-8") as manifest:
        if manifest.tell() == 0:
            csv.writer(manifest, lineterminator="\n").writerow(
                ("scene_id", "split", "seed", "condition_index", "course", "scenario_id", "snr_db", "cnr_db", "truth_range_cell", "truth_doppler_hz")
            )
        if int(progress["manifest_offset"]) == 0:
            manifest.flush()
            progress["manifest_offset"] = manifest.tell()
            _write_json(progress_path, progress)
        offset = rows_written
        for chunk_index, spec in enumerate(plan[completed:], start=completed):
            generated = __import__("radardistill.simulation.d2a", fromlist=["generate_d2a_paired_batch"]).generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(spec["condition"]["snr_db"]),
                float(spec["condition"]["cnr_db"]), dict(physics), scenario_id=str(spec["condition"]["scenario_id"]),
            )
            stop = offset + int(spec["count"])
            if generated.iq_h0.shape != (int(spec["count"]), 3, 130, 32):
                raise RuntimeError("simulator returned an unexpected R6-B training IQ shape")
            arrays["iq_h0"][offset:stop] = generated.iq_h0
            arrays["iq_h1"][offset:stop] = generated.iq_h1
            arrays["truth_range_cell"][offset:stop] = generated.truth_target_range_cell.astype(np.float32)
            arrays["truth_doppler_hz"][offset:stop] = generated.truth_target_doppler_hz.astype(np.float32)
            arrays["condition_index"][offset:stop] = int(spec["condition_index"])
            arrays["scene_id"][offset:stop] = generated.base_scene_id.astype("<U48")
            for row_index, scene_id in enumerate(generated.base_scene_id):
                manifest.write(",".join((
                    str(scene_id), split, str(int(spec["seed"])), str(int(spec["condition_index"])),
                    str(spec["course"]), str(spec["condition"]["scenario_id"]),
                    repr(float(generated.snr_db[row_index])), repr(float(generated.requested_cnr_db[row_index])),
                    repr(float(generated.truth_target_range_cell[row_index])), repr(float(generated.truth_target_doppler_hz[row_index])),
                )) + "\n")
            manifest.flush()
            for value in arrays.values():
                value.flush()
            offset = stop
            progress = {
                "owner": owner, "completed_chunks": chunk_index + 1,
                "rows_written": offset, "manifest_offset": manifest.tell(),
            }
            _write_json(progress_path, progress)
            if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
                print(json.dumps({"stage": "generate", "split": split,
                                  "chunks_done": chunk_index + 1, "chunks_total": len(plan),
                                  "scenes_done": offset}, ensure_ascii=False), flush=True)
    complete = len(plan) == int(progress["completed_chunks"]) and count == int(progress["rows_written"])
    if not complete:
        return {"split": split, "complete": False, **progress}
    return {"split": split, "complete": True, "scenes": count, "chunks": len(plan),
            "cache_bytes": sum(path.stat().st_size for path in paths.values()),
            "manifest": str(manifest_path.relative_to(ROOT))}


def _load_pair_cache(split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("R6-B paired cache split must be train or validation")
    config, r6a, _, _ = _configs()
    _, head = _require_frozen_preflight()
    paths = _paired_cache_paths(split)
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError(f"R6-B {split} cache is incomplete")
    expected_count = 51200 if split == "train" else 5000
    expected_owner = {"owner": "D2-A-R6-B-training-cache", "git_sha": head,
                      "split": split, "scene_count": expected_count}
    owner_path = CACHE / split / "owner.json"
    if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != expected_owner:
        raise RuntimeError(f"R6-B {split} cache ownership identity is invalid")
    progress_path = CACHE / split / "progress.json"
    if not progress_path.is_file():
        raise RuntimeError(f"R6-B {split} cache has no completion cursor")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    plan = _all_split_plans(config, r6a)[split]
    if (progress.get("owner") != expected_owner
            or int(progress.get("completed_chunks", -1)) != len(plan)
            or int(progress.get("rows_written", -1)) != expected_count):
        raise RuntimeError(f"R6-B {split} cache is incomplete or belongs to another split plan")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
    expected_shapes = {
        "iq_h0": (expected_count, 3, 130, 32), "iq_h1": (expected_count, 3, 130, 32),
        "truth_range_cell": (expected_count,), "truth_doppler_hz": (expected_count,),
        "condition_index": (expected_count,), "scene_id": (expected_count,),
    }
    if any(arrays[name].shape != shape for name, shape in expected_shapes.items()):
        raise RuntimeError(f"R6-B {split} cache array shapes differ from the frozen protocol")
    return arrays


def compute_h0_hardness() -> dict[str, Any]:
    """Score only the R6-B H0 training split with the frozen R5-A Medium teacher."""
    _, head = _require_frozen_preflight()
    config, _, _, _ = _configs()
    train = _load_pair_cache("train")
    output = CACHE / "train_h0_hardness.npy"
    metadata_path = CACHE / "train_h0_hardness.json"
    retained_selection_path = STAGE / "hard_negative_selection.npz"
    retained_metadata_path = STAGE / "hard_negative_selection.json"
    model_path = ROOT / config["training"]["hardness_teacher_checkpoint"]
    training_record_path = model_path.parent / "training_record.json"
    if not model_path.is_file() or not training_record_path.is_file():
        raise FileNotFoundError("frozen R5-A Medium checkpoint or record is missing")
    checkpoint_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if metadata_path.is_file():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior.get("git_sha") != head or prior.get("checkpoint_sha256") != checkpoint_sha:
            raise RuntimeError("H0 hardness score cache belongs to another R6-B code/checkpoint")
        values = np.load(output, mmap_mode="r")
        if values.shape != train["condition_index"].shape:
            raise RuntimeError("H0 hardness score count differs from the train split")
        _retain_hardness_artifacts(train, values, config, head, metadata_path,
                                   retained_selection_path, retained_metadata_path)
        return prior
    if output.exists():
        raise RuntimeError("hardness scores exist without a matching provenance record")
    model = load_frozen_medium(str(model_path), str(training_record_path))
    scores = np.lib.format.open_memmap(output, mode="w+", dtype=np.float32, shape=(len(train["iq_h0"]),))
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    batch_size = 32
    for start in range(0, len(scores), batch_size):
        stop = min(start + batch_size, len(scores))
        predicted = predict_medium(model, np.asarray(train["iq_h0"][start:stop]))
        scores[start:stop] = np.asarray(predicted["score"], dtype=np.float32)
        if stop % 4096 == 0 or stop == len(scores):
            scores.flush()
            print(json.dumps({"stage": "hardness", "h0_train_scored": stop,
                              "total": len(scores)}, ensure_ascii=False), flush=True)
    scores.flush()
    torch.cuda.synchronize(0)
    record = {
        "owner": "D2-A-R6-B-training-H0-hardness",
        "git_sha": head, "split": "train", "presence": "H0",
        "scene_count": int(len(scores)), "batch_size": batch_size,
        "method": "frozen_R5-A_Medium_presence_score",
        "checkpoint": str(model_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "training_record_source_git_sha": json.loads(training_record_path.read_text(encoding="utf-8"))["source_git_sha"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "validation_calibration_verification_h1_development_used": False,
    }
    del model
    torch.cuda.empty_cache()
    _write_json(metadata_path, record)
    _retain_hardness_artifacts(train, scores, config, head, metadata_path,
                               retained_selection_path, retained_metadata_path)
    return record


def _retain_hardness_artifacts(
    train: Mapping[str, np.ndarray], scores: np.ndarray, config: Mapping[str, Any],
    head: str, source_record_path: Path, selection_path: Path, metadata_path: Path,
) -> None:
    if not source_record_path.is_file():
        raise FileNotFoundError("R6-B hardness provenance record is missing")
    source_record = json.loads(source_record_path.read_text(encoding="utf-8"))
    if (source_record.get("owner") != "D2-A-R6-B-training-H0-hardness"
            or source_record.get("git_sha") != head
            or source_record.get("validation_calibration_verification_h1_development_used") is not False):
        raise RuntimeError("R6-B hardness provenance does not match the training-only source")
    condition_indices = np.asarray(train["condition_index"], dtype=np.uint8)
    hard_pools = build_hard_negative_pools(
        np.asarray(scores), condition_indices.astype(np.int64),
        top_fraction=float(config["training"]["h0_hard_pool_top_score_fraction"]),
    )
    retained = {
        "scores": np.asarray(scores, dtype=np.float32).copy(),
        "condition_index": condition_indices,
        **{f"hard_pool_{condition}": values for condition, values in hard_pools.items()},
    }
    metadata = {
        "owner": "D2-A-R6-B-training-H0-hardness", "git_sha": head,
        "source_hardness_record": str(source_record_path.relative_to(ROOT)),
        "scene_order_manifest": str((STAGE / "scene_manifest_train.csv").relative_to(ROOT)),
        "teacher_checkpoint": source_record["checkpoint"],
        "teacher_checkpoint_sha256": source_record["checkpoint_sha256"],
        "training_record_source_git_sha": source_record["training_record_source_git_sha"],
        "hardness_record_sha256": hashlib.sha256(source_record_path.read_bytes()).hexdigest(),
        "train_scene_count": int(len(scores)),
        "top_score_fraction_within_condition": float(config["training"]["h0_hard_pool_top_score_fraction"]),
        "hard_pool_count_by_condition": {str(key): int(len(value)) for key, value in hard_pools.items()},
        "validation_calibration_verification_h1_development_used": False,
    }
    if selection_path.is_file():
        with np.load(selection_path, allow_pickle=False) as previous:
            if not np.array_equal(previous["scores"], retained["scores"]):
                raise RuntimeError("retained R6-B H0 scores differ from the train-only score cache")
            if not np.array_equal(previous["condition_index"], condition_indices):
                raise RuntimeError("retained R6-B condition order differs from the train manifest")
            for condition, indices in hard_pools.items():
                if not np.array_equal(previous[f"hard_pool_{condition}"], indices):
                    raise RuntimeError("retained R6-B hard pool differs from deterministic top quartile")
    else:
        with selection_path.open("xb") as handle:
            np.savez_compressed(handle, **retained)
    if metadata_path.is_file():
        if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
            raise RuntimeError("retained R6-B hardness metadata differs from the training-only design")
    else:
        _write_json(metadata_path, metadata)


def _initial_medium_state(seed: int = 303) -> dict[str, torch.Tensor]:
    from radardistill.students.task_student import TaskStudent

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TaskStudent(width=96, include_spectrum_head=True)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    del model
    return state


def _task_model(state: Mapping[str, torch.Tensor], *, device: torch.device) -> torch.nn.Module:
    from radardistill.students.task_student import TaskStudent

    model = TaskStudent(width=96, include_spectrum_head=True)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _task_config(e003: Mapping[str, Any]) -> dict[str, Any]:
    training = dict(e003["training"])
    training["loss"] = training["losses"]
    return {"radar": e003["radar"], "training": training}


def _r6b_learning_rate(update: int) -> float:
    from radardistill.training.d2a_r5 import R5CosineSchedule

    schedule = R5CosineSchedule(
        initial_updates=6400, final_updates=12800,
        initial_lr=0.001, initial_final_lr=0.0001, extension_final_lr=0.00001,
    )
    return schedule.lr_for_update(update)


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "update", "learning_rate", "interval_train_task_loss", "interval_presence_loss",
        "interval_range_cell_loss", "interval_range_offset_loss", "interval_doppler_loss",
        "validation_task_loss", "validation_auc", "validation_doppler_rmse_hz",
        "seconds_since_last_checkpoint", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validation_summary(
    model: torch.nn.Module,
    arrays: Mapping[str, np.ndarray],
    loss_config: Mapping[str, Any],
    *,
    device: torch.device,
    pair_chunk: int = 64,
) -> dict[str, float]:
    from radardistill.distillation.task_losses import decode_task_outputs, task_loss

    count = len(arrays["iq_h0"])
    weighted_loss = 0.0
    h0_scores: list[np.ndarray] = []
    h1_scores: list[np.ndarray] = []
    predicted_range: list[np.ndarray] = []
    predicted_doppler: list[np.ndarray] = []
    truth_range: list[np.ndarray] = []
    truth_doppler: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, count, pair_chunk):
            ids = np.arange(start, min(start + pair_chunk, count), dtype=np.int64)
            batch = assemble_independent_h0_h1_batch(arrays, ids, ids)
            x = torch.from_numpy(batch["iq_ri"]).to(device)
            truth = {
                "truth_presence": torch.from_numpy(batch["truth_presence"]).to(device),
                "truth_range_cell": torch.from_numpy(batch["truth_range_cell"]).to(device),
                "truth_doppler_hz": torch.from_numpy(batch["truth_doppler_hz"]).to(device),
            }
            outputs = model(x, compute_spectrum=False)
            terms = task_loss(outputs, truth, dict(loss_config), teacher_weight=0.0)
            weighted_loss += float(terms["task_total"].item()) * len(ids) / count
            decoded = decode_task_outputs(outputs, 1300.0)
            score = decoded["presence_probability"].detach().cpu().numpy()
            h0_scores.append(score[::2])
            h1_scores.append(score[1::2])
            predicted_range.append(decoded["range_cell"][1::2].detach().cpu().numpy())
            predicted_doppler.append(decoded["doppler_hz"][1::2].detach().cpu().numpy())
            truth_range.append(np.asarray(arrays["truth_range_cell"][ids], dtype=np.float64))
            truth_doppler.append(np.asarray(arrays["truth_doppler_hz"][ids], dtype=np.float64))
    from radardistill.evaluation.d2a_e001 import empirical_auc

    return {
        "task_loss": float(weighted_loss),
        "auc": float(empirical_auc(np.concatenate(h0_scores), np.concatenate(h1_scores))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(wrapped_error(
            np.concatenate(predicted_doppler), np.concatenate(truth_doppler), 1300.0
        ) ** 2))),
        "range_rmse_cells": float(np.sqrt(np.mean((
            np.concatenate(predicted_range) - np.concatenate(truth_range)
        ) ** 2))),
    }


def _train_arm_attempt(
    *,
    arm: str,
    train: Mapping[str, np.ndarray],
    validation: Mapping[str, np.ndarray],
    h1_plan: np.ndarray,
    h0_plan: np.ndarray,
    initial_state: Mapping[str, torch.Tensor],
    e003: Mapping[str, Any],
    device: torch.device,
    microbatch: int,
    output_dir: Path,
    updates: int = 12800,
    effective_batch: int = 64,
    validation_interval: int = 400,
    save_artifacts: bool = True,
) -> dict[str, Any]:
    from radardistill.distillation.task_losses import task_loss

    if h1_plan.shape != h0_plan.shape or h1_plan.shape[0] != updates:
        raise ValueError("matched H0/H1 plans must match the requested update budget")
    if effective_batch % microbatch or microbatch % 2:
        raise ValueError("microbatch must preserve the effective H0/H1 balanced batch")
    if effective_batch // 2 != h1_plan.shape[1]:
        raise ValueError("plan H1 count does not match the effective CPI batch")
    owner = {"owner": "D2-A-R6-B-training-attempt", "arm": arm,
             "microbatch": microbatch, "updates": updates, "formal": save_artifacts}
    if save_artifacts:
        if output_dir.exists():
            owner_path = output_dir / "owner.json"
            if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
                raise RuntimeError("refusing to reuse an unowned R6-B training attempt directory")
        else:
            output_dir.mkdir(parents=True)
            _write_json(output_dir / "owner.json", owner)
    model = _task_model(initial_state, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=R6B_OPTIMIZER["learning_rate_initial"],
        weight_decay=R6B_OPTIMIZER["weight_decay"],
    )
    loss_config = _task_config(e003)
    history: list[dict[str, Any]] = []
    best_validation = {"task_loss": float("inf")}
    best_update = 0
    start_update = 0
    elapsed_before = 0.0
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if save_artifacts and last_checkpoint.is_file():
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        if checkpoint.get("owner") != owner:
            raise ValueError("R6-B checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_update = int(checkpoint["update"])
        best_update = int(checkpoint["best_update"])
        best_validation = dict(checkpoint["best_validation"])
        history = list(checkpoint["history"])
        elapsed_before = float(checkpoint.get("training_seconds_completed", 0.0))
        if start_update == updates:
            completed_record = output_dir / "training_record.json"
            if completed_record.is_file():
                return json.loads(completed_record.read_text(encoding="utf-8"))
    initial_validation = _validation_summary(model, validation, loss_config, device=device)
    if start_update == 0:
        best_validation = initial_validation
        history.append({
            "update": 0, "validation_task_loss": initial_validation["task_loss"],
            "validation_auc": initial_validation["auc"],
            "validation_doppler_rmse_hz": initial_validation["doppler_rmse_hz"],
            "seconds_since_last_checkpoint": 0.0,
        })
    interval_sums = {key: 0.0 for key in ("presence", "range_cell", "range_offset", "doppler", "task_total")}
    interval_updates = 0
    interval_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    update_started = time.perf_counter()
    model.train()
    for update_index in range(start_update, updates):
        update = update_index + 1
        learning_rate = _r6b_learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = assemble_independent_h0_h1_batch(
            train, h0_plan[update_index], h1_plan[update_index]
        )
        x = torch.from_numpy(batch["iq_ri"])
        truth_all = {
            key: torch.from_numpy(batch[key])
            for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
        }
        optimizer.zero_grad(set_to_none=True)
        update_terms = {key: 0.0 for key in interval_sums}
        for micro_start in range(0, effective_batch, microbatch):
            micro_stop = micro_start + microbatch
            xm = x[micro_start:micro_stop].to(device)
            truth = {key: value[micro_start:micro_stop].to(device) for key, value in truth_all.items()}
            outputs = model(xm, compute_spectrum=False)
            terms = task_loss(outputs, truth, loss_config, teacher_weight=0.0)
            scaled = scale_d2a_e002_microbatch_loss(
                terms, loss_config, microbatch_size=microbatch,
                effective_batch_size=effective_batch, microbatch_h1=microbatch // 2,
                effective_h1=effective_batch // 2, teacher_weight=0.0,
            )
            if not torch.isfinite(scaled):
                raise FloatingPointError(f"non-finite R6-B {arm} loss at update {update}")
            scaled.backward()
            update_terms["presence"] += float(terms["presence"].detach().item()) * microbatch / effective_batch
            for key in ("range_cell", "range_offset", "doppler"):
                update_terms[key] += float(terms[key].detach().item()) * (microbatch // 2) / (effective_batch // 2)
            update_terms["task_total"] += float(scaled.detach().item())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), R6B_OPTIMIZER["gradient_clip_norm"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite R6-B {arm} gradient at update {update}")
        optimizer.step()
        for key, value in update_terms.items():
            interval_sums[key] += value
        interval_updates += 1
        if update % validation_interval == 0 or update == updates:
            model.eval()
            validation_metrics = _validation_summary(model, validation, loss_config, device=device)
            interval_seconds = time.perf_counter() - interval_started
            row = {
                "update": update, "learning_rate": learning_rate,
                "interval_train_task_loss": interval_sums["task_total"] / interval_updates,
                "interval_presence_loss": interval_sums["presence"] / interval_updates,
                "interval_range_cell_loss": interval_sums["range_cell"] / interval_updates,
                "interval_range_offset_loss": interval_sums["range_offset"] / interval_updates,
                "interval_doppler_loss": interval_sums["doppler"] / interval_updates,
                "validation_task_loss": validation_metrics["task_loss"],
                "validation_auc": validation_metrics["auc"],
                "validation_doppler_rmse_hz": validation_metrics["doppler_rmse_hz"],
                "seconds_since_last_checkpoint": interval_seconds,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            history.append(row)
            if validation_metrics["task_loss"] < best_validation["task_loss"]:
                best_validation = validation_metrics
                best_update = update
                if save_artifacts:
                    _atomic_torch_save({
                        "owner": owner, "arm": arm, "update": update,
                        "model_state_dict": _cpu_state_dict(model),
                        "validation": validation_metrics,
                    }, output_dir / "best_checkpoint.pt")
            if save_artifacts:
                elapsed_completed = elapsed_before + time.perf_counter() - update_started
                _atomic_torch_save({
                    "owner": owner, "update": update, "model_state_dict": _cpu_state_dict(model),
                    "optimizer_state_dict": optimizer.state_dict(), "best_update": best_update,
                    "best_validation": best_validation, "history": history,
                    "training_seconds_completed": elapsed_completed,
                }, last_checkpoint)
                _write_history(output_dir / "training_history.csv", history)
            print(json.dumps({"stage": "train", "arm": arm, "update": update,
                              "updates": updates, "validation_task_loss": validation_metrics["task_loss"],
                              "interval_seconds": interval_seconds}, ensure_ascii=False), flush=True)
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_updates = 0
            interval_started = time.perf_counter()
            model.train()
    torch.cuda.synchronize(device)
    elapsed = elapsed_before + time.perf_counter() - update_started
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    if save_artifacts:
        best_path = output_dir / "best_checkpoint.pt"
        if not best_path.is_file():
            _atomic_torch_save({
                "owner": owner, "arm": arm, "update": updates,
                "model_state_dict": _cpu_state_dict(model), "validation": best_validation,
            }, best_path)
    record = {
        "arm": arm, "microbatch_size": int(microbatch), "effective_batch_size": int(effective_batch),
        "updates_completed": int(updates), "selected_update": int(best_update),
        "selected_validation": best_validation, "training_seconds": float(elapsed),
        "seconds_per_update": float(elapsed / max(updates, 1)),
        "cuda_peak_allocated_bytes": peak_allocated, "cuda_peak_reserved_bytes": peak_reserved,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history_rows": len(history), "per_update_history_persisted": False,
    }
    if save_artifacts:
        record["output_dir"] = str(output_dir.relative_to(ROOT))
        _write_json(output_dir / "training_record.json", record)
    del optimizer, model
    torch.cuda.empty_cache()
    return record


def _train_with_oom_fallback(
    *, arm: str, train: Mapping[str, np.ndarray], validation: Mapping[str, np.ndarray],
    h1_plan: np.ndarray, h0_plan: np.ndarray, initial_state: Mapping[str, torch.Tensor],
    e003: Mapping[str, Any], device: torch.device,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    for microbatch in (32, 16, 8, 4):
        output_dir = RUNS / arm / f"microbatch_{microbatch}"
        try:
            result = _train_arm_attempt(
                arm=arm, train=train, validation=validation, h1_plan=h1_plan, h0_plan=h0_plan,
                initial_state=initial_state, e003=e003, device=device,
                microbatch=microbatch, output_dir=output_dir,
            )
            result["oom_attempts"] = failures
            return result
        except torch.cuda.OutOfMemoryError as error:
            failures.append({"microbatch_size": microbatch, "error": str(error)[:1000]})
            _write_json(RUNS / arm / "oom_attempts.json", failures)
            torch.cuda.empty_cache()
            failed_dir = RUNS / arm / f"microbatch_{microbatch}"
            owner_path = failed_dir / "owner.json"
            expected_owner = {"owner": "D2-A-R6-B-training-attempt", "arm": arm,
                              "microbatch": microbatch, "updates": 12800, "formal": True}
            if owner_path.is_file() and json.loads(owner_path.read_text(encoding="utf-8")) == expected_owner:
                shutil.rmtree(failed_dir)
            if microbatch == 4:
                raise RuntimeError("R6-B microbatch 4 OOMed; no CPU/AMP/effective-batch fallback") from error
    raise RuntimeError(f"R6-B {arm} did not complete")


def run_g1() -> dict[str, Any]:
    """Technical 16-pair/200-update GPU check; its weights are discarded."""
    _, head = _require_frozen_preflight()
    prior_path = STAGE / "g1.json"
    if prior_path.is_file():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if prior.get("git_sha") == head and prior.get("status") == "G1_PASS":
            return prior
    config, _, _, _ = _configs()
    train = _load_pair_cache("train")
    validation = _load_pair_cache("validation")
    hardness = np.load(CACHE / "train_h0_hardness.npy", mmap_mode="r")
    conditions = np.asarray(train["condition_index"], dtype=np.int64)
    hard = build_hard_negative_pools(np.asarray(hardness), conditions, top_fraction=0.25)
    rng = np.random.default_rng(int(config["training"]["h0_sampler_seed"]))
    h1_plan = rng.integers(0, len(conditions), size=(200, 16), dtype=np.int64)
    h0_results = {}
    g1_validation = {key: value[:256] for key, value in validation.items()}
    for arm, mode in (("TruthOnly", "uniform"), ("HardNegative", "hard_negative")):
        h0_plan, audit = build_h0_sample_plan(
            h1_plan, conditions, conditions, hard, mode=mode,
            seed=int(config["training"]["h0_sampler_seed"]),
            hard_fraction=float(config["training"]["h0_intervention_hard_fraction"]),
        )
        oom_attempts = []
        result = None
        for microbatch in (32, 16, 8):
            try:
                result = _train_arm_attempt(
                    arm=f"G1_{arm}", train=train, validation=g1_validation,
                    h1_plan=h1_plan, h0_plan=h0_plan, initial_state=_initial_medium_state(303),
                    e003=_read_yaml("experiments/D2-A-E003/config.yaml"), device=torch.device("cuda:0"),
                    microbatch=microbatch, output_dir=RUNS / "g1" / arm, updates=200,
                    effective_batch=32, validation_interval=200, save_artifacts=False,
                )
                break
            except torch.cuda.OutOfMemoryError as error:
                oom_attempts.append({"microbatch_size": microbatch, "error": str(error)[:500]})
                torch.cuda.empty_cache()
        if result is None:
            raise RuntimeError(f"R6-B technical G1 OOMed for {arm} at microbatch 8")
        result["oom_attempts"] = oom_attempts
        result["sampler_audit"] = audit
        h0_results[arm] = result
    free, total = torch.cuda.mem_get_info(0)
    record = {
        "status": "G1_PASS" if all(math.isfinite(item["selected_validation"]["task_loss"]) for item in h0_results.values()) else "G1_FAIL",
        "git_sha": head, "technical_only": True, "weights_discarded": True,
        "paired_scenes_per_update": 16, "technical_batch_size_cpi": 32,
        "updates_per_sampler_path": 200, "formal_effective_batch_changed": False,
        "formal_batch_cpi": 64,
        "formal_batch_changed": False,
        "technical_validation_source": "first_256_scenes_of_independent_R6-B_validation_split",
        "gpu": torch.cuda.get_device_name(0), "free_vram_after_bytes": int(free),
        "total_vram_bytes": int(total), "arms": h0_results,
        "final_blind_generated": False,
    }
    _write_json(STAGE / "g1.json", record)
    if record["status"] != "G1_PASS":
        raise RuntimeError("R6-B technical G1 did not pass")
    return record


def run_formal_training() -> dict[str, Any]:
    preflight, head = _require_frozen_preflight()
    g1_path = STAGE / "g1.json"
    if not g1_path.is_file() or json.loads(g1_path.read_text(encoding="utf-8")).get("status") != "G1_PASS":
        raise RuntimeError("R6-B formal training requires a passing G1 technical check")
    config, _, _, _ = _configs()
    if json.loads((STAGE / "branch_selection.json").read_text(encoding="utf-8"))["selected_branch"] != "R":
        raise RuntimeError("formal training is only permitted for the frozen R branch")
    train = _load_pair_cache("train")
    validation = _load_pair_cache("validation")
    if len(train["iq_h0"]) != 51200 or len(validation["iq_h0"]) != 5000:
        raise RuntimeError("R6-B train/validation cache counts differ from protocol")
    hardness_metadata = json.loads((CACHE / "train_h0_hardness.json").read_text(encoding="utf-8"))
    if hardness_metadata.get("validation_calibration_verification_h1_development_used") is not False:
        raise RuntimeError("hard-negative mining provenance indicates forbidden split leakage")
    if np.any(~np.isfinite(np.asarray(np.load(CACHE / "train_h0_hardness.npy", mmap_mode="r")))):
        raise RuntimeError("training H0 hardness scores are non-finite")
    conditions = np.asarray(train["condition_index"], dtype=np.int64)
    h1_plan = build_repeated_h1_plan(
        conditions, updates=12800, h1_per_update=32,
        seed=int(config["data"]["splits"]["train"]["assignment_seed"]),
    )
    hard_pools = build_hard_negative_pools(
        np.asarray(np.load(CACHE / "train_h0_hardness.npy", mmap_mode="r")),
        conditions, top_fraction=float(config["training"]["h0_hard_pool_top_score_fraction"]),
    )
    sampler_seed = int(config["training"]["h0_sampler_seed"])
    baseline_h0, baseline_audit = build_h0_sample_plan(
        h1_plan, conditions, conditions, hard_pools, mode="uniform", seed=sampler_seed, hard_fraction=0.0,
    )
    intervention_h0, intervention_audit = build_h0_sample_plan(
        h1_plan, conditions, conditions, hard_pools, mode="hard_negative", seed=sampler_seed,
        hard_fraction=float(config["training"]["h0_intervention_hard_fraction"]),
    )
    if np.array_equal(baseline_h0, intervention_h0):
        raise RuntimeError("truth-only and hard-negative H0 plans unexpectedly match")
    plan_path = STAGE / "training_sampling_plan.npz"
    if plan_path.exists():
        with np.load(plan_path) as previous:
            if (not np.array_equal(previous["h1_plan"], h1_plan)
                    or not np.array_equal(previous["h0_truth_only"], baseline_h0)
                    or not np.array_equal(previous["h0_hard_negative"], intervention_h0)):
                raise RuntimeError("saved R6-B sampling plan differs from deterministic frozen plan")
    else:
        np.savez_compressed(
            plan_path, h1_plan=h1_plan,
            h0_truth_only=baseline_h0, h0_hard_negative=intervention_h0,
        )
    audit_record = {
        "git_sha": head, "h1_order_identical_across_arms": True,
        "h1_scene_count_per_epoch": 51200, "epochs": 8,
        "truth_only_h0": baseline_audit, "hard_negative_h0": intervention_audit,
        "hard_pool_top_fraction": float(config["training"]["h0_hard_pool_top_score_fraction"]),
        "hard_fraction": float(config["training"]["h0_intervention_hard_fraction"]),
        "H0_and_H1_share_base_scene_id_required": False,
    }
    audit_path = STAGE / "training_sampling_audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text(encoding="utf-8")) != audit_record:
        raise RuntimeError("refusing to overwrite a different R6-B sampling audit")
    if not audit_path.exists():
        _write_json(audit_path, audit_record)
    initial_state = _initial_medium_state(303)
    model = _task_model(initial_state, device=torch.device("cpu"))
    if sum(p.numel() for p in model.parameters()) != 82887:
        raise RuntimeError("R6-B Medium training architecture parameter count changed")
    if not all(torch.equal(initial_state[key], value.cpu()) for key, value in model.state_dict().items()):
        raise RuntimeError("R6-B shared seed-303 model initialization is inconsistent")
    del model
    torch.cuda.empty_cache()
    e003 = _read_yaml("experiments/D2-A-E003/config.yaml")
    print("[R6-B] CUDA truth-only matched baseline", flush=True)
    baseline = _train_with_oom_fallback(
        arm="TruthOnly", train=train, validation=validation, h1_plan=h1_plan,
        h0_plan=baseline_h0, initial_state=initial_state, e003=e003, device=torch.device("cuda:0"),
    )
    torch.cuda.empty_cache()
    print("[R6-B] CUDA hard-negative intervention", flush=True)
    intervention = _train_with_oom_fallback(
        arm="HardNegative", train=train, validation=validation, h1_plan=h1_plan,
        h0_plan=intervention_h0, initial_state=initial_state, e003=e003, device=torch.device("cuda:0"),
    )
    free, total = torch.cuda.mem_get_info(0)
    record = {
        "git_sha": head,
        "branch": "R",
        "training_seed": 303,
        "initialization_matched": True,
        "h1_scene_order_and_condition_quotas_matched": True,
        "optimizer_and_update_budget_matched": True,
        "only_training_h0_sampling_changed": True,
        "effective_batch_size_cpi": 64,
        "updates_per_arm": 12800,
        "h1_exposures_per_arm": 409600,
        "full_train_set_traversals": 8,
        "precision": "FP32",
        "device": "cuda:0",
        "gpu": torch.cuda.get_device_name(0),
        "free_vram_after_bytes": int(free), "total_vram_bytes": int(total),
        "optimizer": R6B_OPTIMIZER,
        "truth_only": baseline,
        "hard_negative": intervention,
        "final_blind_generated": False,
    }
    _write_json(RUNS / "training.json", record)
    return record


def _load_evaluation_states(training_record: Mapping[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    from radardistill.students.task_student import TaskStudent

    config, _, _, _ = _configs()
    frozen = load_frozen_medium(
        str(ROOT / config["methods"]["checkpoint_source"]),
        str(ROOT / config["methods"]["checkpoint_record"]),
    )
    states: dict[str, dict[str, torch.Tensor]] = {
        "Medium_Frozen": {key: value.detach().cpu().clone()
                           for key, value in frozen.state_dict().items()}
    }
    del frozen
    torch.cuda.empty_cache()
    for label, key in (("Medium_TruthOnly", "truth_only"), ("Medium_HardNegative", "hard_negative")):
        run_record = training_record[key]
        best_path = ROOT / run_record["output_dir"] / "best_checkpoint.pt"
        payload = torch.load(best_path, map_location="cpu", weights_only=False)
        state = {name: value for name, value in payload["model_state_dict"].items()
                 if not name.startswith("spectrum_head.")}
        expected = TaskStudent(width=96, include_spectrum_head=False).state_dict()
        if set(state) != set(expected):
            raise ValueError(f"R6-B {label} inference checkpoint keys differ")
        states[label] = {name: tensor.detach().cpu().clone() for name, tensor in state.items()}
    if any(sum(t.numel() for t in state.values()) != 70277 for state in states.values()):
        raise ValueError("R6-B inference models must each have 70,277 parameters")
    return states


def _predict_one_student(
    model: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
    iq: np.ndarray,
) -> dict[str, np.ndarray]:
    from radardistill.distillation.task_losses import decode_task_outputs

    model.load_state_dict(state, strict=True)
    model.eval()
    x = np.asarray(iq)
    if x.ndim != 4 or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("R6-B model input must be a finite complex IQ batch")
    real = np.concatenate((x.real, x.imag), axis=1).astype(np.float32, copy=False)
    with torch.inference_mode():
        output = model(torch.from_numpy(np.ascontiguousarray(real)).to("cuda:0"), compute_spectrum=False)
        decoded = decode_task_outputs(output, 1300.0)
    score = decoded["presence_probability"].cpu().numpy().astype(np.float32)
    range_cell = decoded["range_cell"].cpu().numpy().astype(np.float32)
    doppler = decoded["doppler_hz"].cpu().numpy().astype(np.float32)
    return {"score": score, "range_cell": range_cell,
            "range_bin": np.floor(range_cell).astype(np.int8), "doppler_hz": doppler}


def _prediction_work_paths(split: str) -> tuple[Path, Path]:
    work = PREDICTIONS / ".work" / split
    return work, PREDICTIONS / f"{split}.npz"


def _prediction_arrays(work: Path, count: int, owner: dict[str, Any]) -> dict[str, np.memmap]:
    owner_path = work / "owner.json"
    if work.exists():
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != owner:
            raise RuntimeError(f"refusing to reuse unowned R6-B prediction work: {work}")
    else:
        work.mkdir(parents=True)
        _write_json(owner_path, owner)
    shapes = {
        "scene_id": ((count,), "<U48"),
        "seed": ((count,), np.int64),
        "condition_index": ((count,), np.uint8),
        "score": ((count, len(METHODS)), np.float32),
        "range_cell": ((count, len(METHODS)), np.float32),
        "range_bin": ((count, len(METHODS)), np.int8),
        "doppler_hz": ((count, len(METHODS)), np.float32),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
        "observable_scale": ((count,), np.float32),
        "noise_component_power": ((count,), np.float64),
        "clutter_component_power": ((count,), np.float64),
        "frequency_estimates": ((count, len(METHODS), len(FREQUENCIES)), np.float32),
    }
    output: dict[str, np.memmap] = {}
    for name, (shape, dtype) in shapes.items():
        path = work / f"{name}.npy"
        if not path.exists():
            np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape).flush()
        output[name] = np.load(path, mmap_mode="r+")
    return output


def _append_eval_manifest(path: Path, ids: np.ndarray, seeds: np.ndarray, conditions: np.ndarray,
                          r6a_conditions: list[Mapping[str, Any]], split: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite R6-B scene manifest: {path}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        fields = ("scene_id", "split", "seed", "condition_index", "course", "scenario_id", "snr_db", "cnr_db")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, scene_id in enumerate(ids):
            condition = r6a_conditions[int(conditions[index])]
            writer.writerow({
                "scene_id": str(scene_id), "split": split, "seed": int(seeds[index]),
                "condition_index": int(conditions[index]), "course": condition["course"],
                "scenario_id": condition["scenario_id"], "snr_db": condition["snr_db"],
                "cnr_db": condition["cnr_db"],
            })


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    result = float(spearmanr(x, y).statistic)
    return result if math.isfinite(result) else None


def run_evaluation_split(split: str) -> dict[str, Any]:
    """Generate and score one registered evaluation split; evaluation IQ stays streamed."""
    preflight, head = _require_frozen_preflight()
    training_path = RUNS / "training.json"
    if not training_path.is_file():
        raise RuntimeError("R6-B matched formal models must finish before evaluation")
    training_record = json.loads(training_path.read_text(encoding="utf-8"))
    config, r6a, physics, r3 = _configs()
    if split not in ("h0_calibration", "h0_verification", "h1_development"):
        raise ValueError("unknown R6-B evaluation split")
    spec = config["data"]["splits"][split]
    count = int(spec["scenes"])
    work, final_path = _prediction_work_paths(split)
    final_owner = {"owner": "D2-A-R6-B-predictions", "git_sha": head,
                   "split": split, "scene_count": count, "methods": list(METHODS)}
    if final_path.is_file():
        with np.load(final_path, allow_pickle=False) as existing:
            if str(existing["git_sha"].item()) != head or int(existing["scene_id"].shape[0]) != count:
                raise RuntimeError("completed R6-B prediction archive identity mismatch")
            ids = existing["scene_id"]
            seeds = existing["seed"]
            conditions = existing["condition_index"]
        manifest_path = STAGE / f"scene_manifest_{split}.csv"
        if not manifest_path.exists():
            _append_eval_manifest(manifest_path, ids, seeds, conditions, r6a["radar"]["conditions"], split)
        work, _ = _prediction_work_paths(split)
        owner_path = work / "owner.json"
        expected_owner = {"owner": "D2-A-R6-B-predictions", "git_sha": head,
                          "split": split, "scene_count": count, "methods": list(METHODS)}
        if owner_path.is_file() and json.loads(owner_path.read_text(encoding="utf-8")) == expected_owner:
            shutil.rmtree(work)
        return {"split": split, "scenes": count, "archive": str(final_path.relative_to(ROOT)), "reused": True}
    eval_config = _r6b_eval_config(config, r6a)
    plan = build_split_plan(eval_config, split)
    state_path = work / "progress.json"
    arrays = _prediction_arrays(work, count, final_owner)
    if state_path.exists():
        progress = json.loads(state_path.read_text(encoding="utf-8"))
        if progress.get("owner") != final_owner:
            raise RuntimeError("R6-B evaluation progress belongs to another run")
        chunk_start = int(progress["chunks_done"])
        offset = int(progress["rows_done"])
        if (not 0 <= chunk_start <= len(plan)
                or offset != sum(int(row["count"]) for row in plan[:chunk_start])):
            raise RuntimeError("R6-B evaluation progress cursor differs from the registered chunk plan")
    else:
        chunk_start = offset = 0
        progress = {"owner": final_owner, "chunks_done": 0, "rows_done": 0}
        _write_json(state_path, progress)
    states = _load_evaluation_states(training_record)
    from radardistill.students.task_student import TaskStudent

    cuda_model = TaskStudent(width=96, include_spectrum_head=False).to("cuda:0").eval()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    from radardistill.simulation.d2a import generate_d2a_paired_batch

    for chunk_index, chunk_spec in enumerate(plan[chunk_start:], start=chunk_start):
        generated = generate_d2a_paired_batch(
            int(chunk_spec["count"]), int(chunk_spec["seed"]),
            float(chunk_spec["condition"]["snr_db"]), float(chunk_spec["condition"]["cnr_db"]),
            dict(physics), scenario_id=str(chunk_spec["condition"]["scenario_id"]),
        )
        presence = str(chunk_spec["presence"])
        iq = generated.iq_h0 if presence == "H0" else generated.iq_h1
        truth = None if presence == "H0" else {
            "range_cell": generated.truth_target_range_cell,
            "doppler_hz": generated.truth_target_doppler_hz,
        }
        # Labels remain outside every observed detector and model call.
        classical = classical_detector_predictions(iq, physics, r3, use_cuda_r3=True)
        observed: dict[str, dict[str, np.ndarray]] = {}
        for name, output in classical.items():
            observed[name] = {
                "score": np.asarray(output.score, dtype=np.float32),
                "range_cell": np.asarray(output.range_cell, dtype=np.float32),
                "range_bin": np.asarray(output.range_cell, dtype=np.int8),
                "doppler_hz": np.asarray(output.doppler_hz, dtype=np.float32),
            }
        for label in ("Medium_Frozen", "Medium_TruthOnly", "Medium_HardNegative"):
            observed[label] = _predict_one_student(cuda_model, states[label], iq)
        row_stop = offset + len(iq)
        arrays["scene_id"][offset:row_stop] = np.asarray(generated.base_scene_id, dtype="<U48")
        arrays["seed"][offset:row_stop] = int(chunk_spec["seed"])
        arrays["condition_index"][offset:row_stop] = int(chunk_spec["condition_index"])
        for method_index, method in enumerate(METHODS):
            prediction = observed[method]
            for field in ("score", "range_cell", "range_bin", "doppler_hz"):
                arrays[field][offset:row_stop, method_index] = prediction[field]
            if truth is not None:
                frequencies = candidate_frequency_estimates(iq, prediction["range_bin"], physics)
                for frequency_index, frequency_name in enumerate(FREQUENCIES):
                    arrays["frequency_estimates"][offset:row_stop, method_index, frequency_index] = frequencies[frequency_name]
        arrays["observable_scale"][offset:row_stop] = np.sqrt(
            np.median(np.abs(iq) ** 2, axis=(1, 2, 3))
        ).astype(np.float32)
        arrays["noise_component_power"][offset:row_stop] = np.mean(
            np.abs(generated.noise_iq) ** 2, axis=(1, 2, 3)
        )
        arrays["clutter_component_power"][offset:row_stop] = np.mean(
            np.abs(generated.clutter_iq) ** 2, axis=(1, 2, 3)
        )
        if truth is None:
            arrays["truth_range_cell"][offset:row_stop] = np.nan
            arrays["truth_doppler_hz"][offset:row_stop] = np.nan
        else:
            arrays["truth_range_cell"][offset:row_stop] = truth["range_cell"]
            arrays["truth_doppler_hz"][offset:row_stop] = truth["doppler_hz"]
        for value in arrays.values():
            value.flush()
        offset = row_stop
        progress = {"owner": final_owner, "chunks_done": chunk_index + 1, "rows_done": offset}
        _write_json(state_path, progress)
        if (chunk_index + 1) % 100 == 0 or chunk_index + 1 == len(plan):
            print(json.dumps({"stage": "evaluate", "split": split, "chunks_done": chunk_index + 1,
                              "chunks_total": len(plan), "scenes_done": offset}, ensure_ascii=False), flush=True)
    if offset != count:
        raise RuntimeError(f"R6-B {split} generated {offset} rows; expected {count}")
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    packed = {name: np.asarray(value).copy() for name, value in arrays.items()}
    packed["methods"] = np.asarray(METHODS, dtype="<U32")
    packed["frequencies"] = np.asarray(FREQUENCIES, dtype="<U32")
    packed["split"] = np.asarray(split)
    packed["git_sha"] = np.asarray(head)
    temporary = final_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **packed)
    temporary.replace(final_path)
    _append_eval_manifest(STAGE / f"scene_manifest_{split}.csv", packed["scene_id"],
                          packed["seed"], packed["condition_index"], r6a["radar"]["conditions"], split)
    environment = {
        "git_sha": head, "split": split, "gpu": torch.cuda.get_device_name(0),
        "device": "cuda:0", "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "elapsed_seconds": float(elapsed), "scenes": count, "chunk_size": int(config["data"]["chunk_size"]),
        "evaluation_iq_persisted": False, "teacher_maps_persisted": False,
        "simultaneous_cuda_student_models": 1,
    }
    _write_json(STAGE / f"environment_{split}.json", environment)
    del cuda_model
    torch.cuda.empty_cache()
    if not final_path.is_file():
        raise RuntimeError("R6-B prediction archive was not written")
    del arrays, packed
    owner_path = work / "owner.json"
    if owner_path.is_file() and json.loads(owner_path.read_text(encoding="utf-8")) == final_owner:
        shutil.rmtree(work)
    if _git("rev-parse", "HEAD") != preflight["git_sha"]:
        raise RuntimeError("source changed during R6-B evaluation")
    return {"split": split, "scenes": count, "archive": str(final_path.relative_to(ROOT)),
            "elapsed_seconds": elapsed, "cuda_peak_allocated_bytes": environment["cuda_peak_allocated_bytes"]}


def _archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"R6-B prediction archive is missing: {path}")
    with np.load(path, allow_pickle=False) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _estimate_metrics(
    predictions: Mapping[str, np.ndarray],
    mask_h1: np.ndarray,
    detected: np.ndarray,
    method_index: int,
    doppler_values: np.ndarray,
) -> dict[str, Any]:
    return summarize_parameter_estimator(
        predictions["range_cell"][mask_h1, method_index],
        predictions["truth_range_cell"][mask_h1],
        doppler_values,
        predictions["truth_doppler_hz"][mask_h1],
        detected,
        prf_hz=1300.0,
        range_gate_cells=1.0,
        doppler_gate_hz=20.0,
    )


def _bootstrap_paired_mean_difference(
    candidate: np.ndarray, reference: np.ndarray, *, replicates: int, seed: int
) -> dict[str, Any]:
    x = np.asarray(candidate, dtype=np.float64)
    y = np.asarray(reference, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not x.size or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("paired bootstrap vectors must be aligned finite nonempty arrays")
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.integers(0, len(x), size=len(x))
        differences[index] = np.mean(x[sampled] - y[sampled])
    return {"difference": float(np.mean(x - y)),
            "bootstrap_ci95": [float(value) for value in np.quantile(differences, [.025, .975])],
            "replicates": int(replicates), "seed": int(seed)}


def analyze_predictions() -> dict[str, Any]:
    """Apply frozen H0 calibration, independent exact Pfa gates and paired task analysis."""
    preflight, head = _require_frozen_preflight()
    config, r6a, _, _ = _configs()
    paths = {name: PREDICTIONS / f"{name}.npz" for name in
             ("h0_calibration", "h0_verification", "h1_development")}
    data = {name: _archive(path) for name, path in paths.items()}
    methods = tuple(str(value) for value in data["h0_calibration"]["methods"])
    if any(tuple(str(value) for value in item["methods"]) != methods for item in data.values()):
        raise ValueError("R6-B prediction methods differ across splits")
    if methods != METHODS:
        raise ValueError("R6-B prediction method order differs from the frozen method list")
    ids = {name: item["scene_id"] for name, item in data.items()}
    if any(len(np.unique(values)) != len(values) for values in ids.values()):
        raise ValueError("R6-B prediction split contains duplicate scene IDs")
    if any(set(ids[a]) & set(ids[b]) for a, b in (("h0_calibration", "h0_verification"),
                                                   ("h0_calibration", "h1_development"),
                                                   ("h0_verification", "h1_development"))):
        raise ValueError("R6-B evaluation split scene IDs overlap")
    cal, ver, h1 = data["h0_calibration"], data["h0_verification"], data["h1_development"]
    if set(np.unique(cal["condition_index"])) != set(range(12)) or set(np.unique(ver["condition_index"])) != set(range(12)):
        raise ValueError("R6-B H0 evaluation misses registered conditions")
    if set(np.unique(h1["condition_index"])) != set(range(12)):
        raise ValueError("R6-B H1 development misses registered conditions")
    expected_counts = ((cal, 10000), (ver, 10000), (h1, 512))
    if any(any(np.count_nonzero(item["condition_index"] == condition) != expected
               for condition in range(12)) for item, expected in expected_counts):
        raise ValueError("R6-B evaluation condition counts differ from the frozen design")
    targets = [float(value) for value in config["data"]["target_pfa_cpi"]]
    methods_result: dict[str, Any] = {}
    thresholds_by_method: dict[str, dict[str, float]] = {}
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    for method_index, method in enumerate(methods):
        cal_scores = np.asarray(cal["score"][:, method_index], dtype=np.float64)
        ver_scores = np.asarray(ver["score"][:, method_index], dtype=np.float64)
        h1_scores = np.asarray(h1["score"][:, method_index], dtype=np.float64)
        if any(not np.isfinite(values).all() for values in (cal_scores, ver_scores, h1_scores)):
            raise ValueError(f"non-finite R6-B scores for {method}")
        method_record: dict[str, Any] = {
            "roc_auc_global": float(empirical_auc(ver_scores, h1_scores)),
            "roc_auc_by_condition": {}, "h0_calibration_score_nuisance_spearman": {}, "workpoints": {},
        }
        nuisance_fields = {
            "observable_scale": cal["observable_scale"],
            "noise_component_power_offline": cal["noise_component_power"],
            "clutter_component_power_offline": cal["clutter_component_power"],
        }
        for nuisance_name, nuisance_values in nuisance_fields.items():
            rho = _spearman(cal_scores, nuisance_values)
            method_record["h0_calibration_score_nuisance_spearman"][nuisance_name] = rho
        method_record["h0_calibration_score_nuisance_spearman"]["by_condition_observable_scale"] = {
            str(condition): _spearman(
                cal_scores[cal["condition_index"] == condition],
                cal["observable_scale"][cal["condition_index"] == condition],
            )
            for condition in range(12)
        }
        for condition in range(12):
            mask0 = ver["condition_index"] == condition
            mask1 = h1["condition_index"] == condition
            method_record["roc_auc_by_condition"][str(condition)] = float(
                empirical_auc(ver_scores[mask0], h1_scores[mask1])
            )
        thresholds_by_method[method] = {}
        for target_index, target in enumerate(targets):
            calibration = calibrate_threshold(cal_scores, target)
            threshold = float(calibration["threshold"])
            thresholds_by_method[method][f"{target:g}"] = threshold
            false_alarm = ver_scores >= threshold
            detection = h1_scores >= threshold
            calibration_by_condition = {}
            for condition in range(12):
                mask_cal = cal["condition_index"] == condition
                calibration_by_condition[str(condition)] = {
                    "n": int(mask_cal.sum()),
                    "pfa_gt": float(np.mean(cal_scores[mask_cal] > threshold)),
                    "pfa_ge": float(np.mean(cal_scores[mask_cal] >= threshold)),
                    "tie_count": int(np.count_nonzero(cal_scores[mask_cal] == threshold)),
                }
            verification_record = {
                "n": int(len(ver_scores)), "false_alarms": int(false_alarm.sum()),
                "pfa": float(np.mean(false_alarm)),
                "pfa_ci95_exact_upper": clopper_pearson_upper(int(false_alarm.sum()), len(false_alarm)),
                "pfa_gt": float(np.mean(ver_scores > threshold)),
                "tie_count": int(np.count_nonzero(ver_scores == threshold)),
            }
            verification_by_condition: dict[str, Any] = {}
            h1_by_condition: dict[str, Any] = {}
            for condition in range(12):
                mask0 = ver["condition_index"] == condition
                mask1 = h1["condition_index"] == condition
                k = int(np.count_nonzero(false_alarm[mask0]))
                n = int(mask0.sum())
                alpha = 0.05 / 6 if condition >= 6 else 0.05
                verification_by_condition[str(condition)] = {
                    "n": n, "false_alarms": k, "pfa": k / n,
                    "pfa_ci95_exact_upper": clopper_pearson_upper(k, n, alpha=alpha),
                    "ci_method": "one_sided_exact_CP_Bonferroni_C3" if condition >= 6 else "one_sided_exact_CP_95",
                }
                h1_by_condition[str(condition)] = {
                    "n": int(mask1.sum()), "pd": float(np.mean(detection[mask1])),
                    "threshold_ties": int(np.count_nonzero(h1_scores[mask1] == threshold)),
                }
            c3_mask = ver["condition_index"] >= 6
            hard_mask0 = ver["condition_index"] == 10
            c3_h1_mask = h1["condition_index"] >= 6
            hard_h1_mask = h1["condition_index"] == 10
            direct_metrics = _estimate_metrics(
                h1, np.ones(len(h1_scores), dtype=bool), detection, method_index,
                h1["doppler_hz"][:, method_index].astype(np.float64),
            )
            direct_c3 = _estimate_metrics(
                h1, c3_h1_mask, detection[c3_h1_mask], method_index,
                h1["doppler_hz"][c3_h1_mask, method_index].astype(np.float64),
            )
            direct_hard = _estimate_metrics(
                h1, hard_h1_mask, detection[hard_h1_mask], method_index,
                h1["doppler_hz"][hard_h1_mask, method_index].astype(np.float64),
            )
            direct_by_condition = {
                str(condition): _estimate_metrics(
                    h1, h1["condition_index"] == condition,
                    detection[h1["condition_index"] == condition], method_index,
                    h1["doppler_hz"][h1["condition_index"] == condition, method_index].astype(np.float64),
                )
                for condition in range(12)
            }
            frequency_records: dict[str, Any] = {}
            for frequency_index, frequency_name in enumerate(FREQUENCIES):
                estimate = h1["frequency_estimates"][:, method_index, frequency_index].astype(np.float64)
                frequency_records[frequency_name] = {
                    "all_h1": _estimate_metrics(
                        h1, np.ones(len(h1_scores), dtype=bool), detection, method_index, estimate
                    ),
                    "C3": _estimate_metrics(
                        h1, c3_h1_mask, detection[c3_h1_mask], method_index, estimate[c3_h1_mask]
                    ),
                    "hard_cell": _estimate_metrics(
                        h1, hard_h1_mask, detection[hard_h1_mask], method_index, estimate[hard_h1_mask]
                    ),
                    "by_condition": {
                        str(condition): _estimate_metrics(
                            h1, h1["condition_index"] == condition,
                            detection[h1["condition_index"] == condition], method_index,
                            estimate[h1["condition_index"] == condition],
                        )
                        for condition in range(12)
                    },
                }
            method_record["workpoints"][f"{target:g}"] = {
                "calibration": calibration,
                "calibration_by_condition": calibration_by_condition,
                "verification": verification_record,
                "verification_by_condition": verification_by_condition,
                "c3_pooled_pfa": {
                    "n": int(c3_mask.sum()), "false_alarms": int(false_alarm[c3_mask].sum()),
                    "pfa": float(np.mean(false_alarm[c3_mask])),
                    "pfa_ci95_exact_upper": clopper_pearson_upper(
                        int(false_alarm[c3_mask].sum()), int(c3_mask.sum())
                    ),
                },
                "hard_cell_pfa": {
                    "n": int(hard_mask0.sum()), "false_alarms": int(false_alarm[hard_mask0].sum()),
                    "pfa": float(np.mean(false_alarm[hard_mask0])),
                    "pfa_ci95_exact_upper": clopper_pearson_upper(
                        int(false_alarm[hard_mask0].sum()), int(hard_mask0.sum())
                    ),
                },
                "h1": {"n": int(len(detection)), "pd": float(np.mean(detection)),
                       "c3_pd": float(np.mean(detection[h1["condition_index"] >= 6])),
                       "hard_cell_pd": float(np.mean(detection[h1["condition_index"] == 10])),
                       "pd_by_condition": h1_by_condition},
                "parameters_direct": {"all_h1": direct_metrics, "C3": direct_c3,
                                      "hard_cell": direct_hard, "by_condition": direct_by_condition},
                "parameters_frequency_estimators": frequency_records,
            }
        methods_result[method] = method_record
    paired: dict[str, Any] = {}
    base_index = methods.index("Medium_TruthOnly")
    intervention_index = methods.index("Medium_HardNegative")
    for target_index, target in enumerate(targets):
        key = f"{target:g}"
        reference_threshold = thresholds_by_method["Medium_TruthOnly"][key]
        candidate_threshold = thresholds_by_method["Medium_HardNegative"][key]
        ref_detection = h1["score"][:, base_index] >= reference_threshold
        int_detection = h1["score"][:, intervention_index] >= candidate_threshold
        reference_joint = np.zeros(len(h1["scene_id"]), dtype=bool)
        candidate_joint = np.zeros_like(reference_joint)
        for index, method_index in enumerate((base_index, intervention_index)):
            estimate = h1["doppler_hz"][:, method_index]
            range_error = h1["range_cell"][:, method_index] - h1["truth_range_cell"]
            doppler_error = wrapped_error(estimate, h1["truth_doppler_hz"], 1300.0)
            joint = (np.abs(range_error) <= 1.0) & (np.abs(doppler_error) <= 20.0)
            if index == 0:
                reference_joint = ref_detection & joint
            else:
                candidate_joint = int_detection & joint
        paired[key] = {
            "global_pd_difference": _bootstrap_paired_mean_difference(
                int_detection.astype(float), ref_detection.astype(float),
                replicates=bootstrap_replicates, seed=R6B_BOOTSTRAP_SEED + target_index,
            ),
            "global_joint_success_difference": _bootstrap_paired_mean_difference(
                candidate_joint.astype(float), reference_joint.astype(float),
                replicates=bootstrap_replicates, seed=R6B_BOOTSTRAP_SEED + 100 + target_index,
            ),
        }
        for label, mask in (("C3", h1["condition_index"] >= 6),
                            ("hard_cell", h1["condition_index"] == 10)):
            paired[key][label] = {
                "pd_difference": _bootstrap_paired_mean_difference(
                    int_detection[mask].astype(float), ref_detection[mask].astype(float),
                    replicates=bootstrap_replicates, seed=R6B_BOOTSTRAP_SEED + 200 + target_index * 2 + int(label == "hard_cell"),
                ),
                "joint_success_difference": _bootstrap_paired_mean_difference(
                    candidate_joint[mask].astype(float), reference_joint[mask].astype(float),
                    replicates=bootstrap_replicates, seed=R6B_BOOTSTRAP_SEED + 300 + target_index * 2 + int(label == "hard_cell"),
                ),
            }
    result = {
        "experiment": "D2-A-R6-B", "git_sha": head,
        "protocol_sha256": preflight["protocol_sha256"], "config_sha256": preflight["config_sha256"],
        "branch": "R", "methods": methods_result, "paired_hard_negative_vs_truth_only": paired,
        "sample_counts": {name: int(len(item["scene_id"])) for name, item in data.items()},
        "target_pfa_cpi": targets, "threshold_source": "H0_calibration_only",
        "verification_is_independent": True,
        "C3_cell_ci": "one-sided exact Clopper-Pearson with Bonferroni alpha=0.05/6",
        "final_blind_generated": False,
        "zero_event_sample_sizes_one_sided_95": {
            str(p): required_zero_event_sample_size(float(p)) for p in (0.05, 0.01, 0.001, 0.0001, 1e-6)
        },
    }
    metrics_path = STAGE / "metrics.json"
    _write_json(metrics_path, result)
    _write_analysis_report(result)
    _write_figures(result, data, r6a["radar"]["conditions"])
    _write_stage_review(result)
    _write_chatgpt_review_request(result)
    return result


def _write_figures(metrics: Mapping[str, Any], data: Mapping[str, Mapping[str, np.ndarray]],
                   conditions: list[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = STAGE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    target = "0.05"
    method_records = metrics["methods"]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for method in METHODS:
        values = [method_records[method]["workpoints"][target]["verification_by_condition"][str(i)]["pfa"]
                  for i in range(12)]
        axis.plot(np.arange(12), values, marker="o", linewidth=1.5, label=method)
    axis.axhline(0.05, color="black", linestyle="--", linewidth=1, label="target Pfa=0.05")
    axis.set(xticks=np.arange(12), xlabel="Registered condition index", ylabel="Verification Pfa per CPI",
             title="R6-B independent H0 verification by condition")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "pfa_by_condition.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for method in METHODS:
        values = [method_records[method]["workpoints"][target]["parameters_direct"]["by_condition"][str(i)]["joint_success"]
                  for i in range(12)]
        axis.plot(np.arange(12), values, marker="o", linewidth=1.5, label=method)
    axis.set(xticks=np.arange(12), xlabel="Registered condition index", ylabel="Joint detection and parameter success",
             title="R6-B H1 development joint success at global calibration Pfa=0.05")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(figure_dir / "joint_success_by_condition.png", dpi=160)
    plt.close(fig)

    h1 = data["h1_development"]
    snr_by_condition = np.asarray([float(row["snr_db"]) for row in conditions])
    snr_values = np.asarray([snr_by_condition[int(index)] for index in h1["condition_index"]])
    rmse_lines = [
        ("Medium_Frozen direct", "Medium_Frozen", None),
        ("Medium_TruthOnly direct", "Medium_TruthOnly", None),
        ("Medium_HardNegative direct", "Medium_HardNegative", None),
        ("B3 direct bin", "B3", None),
        ("B3 local profile ML", "B3", FREQUENCIES.index("local_profile_ml")),
    ]
    fig, axis = plt.subplots(figsize=(8, 5.2))
    for label, method, frequency_index in rmse_lines:
        method_index = METHODS.index(method)
        rmse = []
        for snr in sorted(np.unique(snr_values)):
            mask = snr_values == snr
            estimate = (h1["doppler_hz"][mask, method_index] if frequency_index is None
                        else h1["frequency_estimates"][mask, method_index, frequency_index])
            error = wrapped_error(estimate, h1["truth_doppler_hz"][mask], 1300.0)
            rmse.append(float(np.sqrt(np.mean(error ** 2))))
        axis.plot(sorted(np.unique(snr_values)), rmse, marker="o", linewidth=1.6, label=label)
    axis.set(xlabel="SNR (dB)", ylabel="All-H1 Doppler RMSE (Hz)",
             title="R6-B Doppler error by SNR (H1 development)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "doppler_rmse_by_snr.png", dpi=160)
    plt.close(fig)


def _write_analysis_report(metrics: Mapping[str, Any]) -> None:
    methods = metrics["methods"]
    training_path = RUNS / "training.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    preflight = json.loads((STAGE / "preflight.json").read_text(encoding="utf-8"))
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    branch = json.loads((STAGE / "branch_selection.json").read_text(encoding="utf-8"))
    lines = [
        "# D2-A-R6-B analysis",
        "",
        f"- Result generation commit: `{metrics['git_sha']}`",
        f"- Preflight: `{preflight['status']}`; technical G1: `{g1['status']}`; selected branch: `{branch['selected_branch']}`.",
        f"- Independent scenes: calibration {metrics['sample_counts']['h0_calibration']:,}, verification {metrics['sample_counts']['h0_verification']:,}, H1 development {metrics['sample_counts']['h1_development']:,}.",
        "- H0 thresholds come from calibration only; all reported Pfa values use independent verification and exact one-sided Clopper–Pearson upper limits.",
        "- Training used effective batch 64, 12,800 updates per arm, seed 303, FP32 CUDA, and the shared two-segment R5-A Medium cosine schedule. History is recorded at validation/checkpoint intervals only.",
        "- The hard-negative pool and its top-quartile ordering use only the R6-B training H0 split. No evaluation IQ or teacher response maps were retained.",
        "",
        "## Training record",
        "",
        "| Arm | Selected update | Microbatch | Seconds | Peak allocated GiB | Validation task loss | Validation AUC | Doppler RMSE (Hz) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("truth_only", "Truth-only"), ("hard_negative", "Hard-negative")):
        item = training[key]
        lines.append(
            f"| {label} | {item['selected_update']} | {item['microbatch_size']} | {item['training_seconds']:.1f} | "
            f"{item['cuda_peak_allocated_bytes'] / 1024**3:.3f} | {item['selected_validation']['task_loss']:.6f} | "
            f"{item['selected_validation']['auc']:.5f} | {item['selected_validation']['doppler_rmse_hz']:.3f} |"
        )
    lines += [
        "",
        "## Detection and task results",
        "",
        "| Method | Pfa upper @ .05 | C3 Pfa upper @ .05 | Hard-cell Pfa upper @ .05 | Pd | C3 Pd | Hard-cell Pd | Joint success | C3 joint | Hard-cell joint |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        work = methods[method]["workpoints"]["0.05"]
        lines.append(
            f"| {method} | {work['verification']['pfa_ci95_exact_upper']:.5f} | "
            f"{work['c3_pooled_pfa']['pfa_ci95_exact_upper']:.5f} | "
            f"{work['hard_cell_pfa']['pfa_ci95_exact_upper']:.5f} | {work['h1']['pd']:.5f} | "
            f"{work['h1']['c3_pd']:.5f} | {work['h1']['hard_cell_pd']:.5f} | "
            f"{work['parameters_direct']['all_h1']['joint_success']:.5f} | "
            f"{work['parameters_direct']['C3']['joint_success']:.5f} | "
            f"{work['parameters_direct']['hard_cell']['joint_success']:.5f} |"
        )
    paired = metrics["paired_hard_negative_vs_truth_only"]["0.05"]
    lines += [
        "",
        "## Matched intervention contrast at global target Pfa 0.05",
        "",
        f"- Global Pd difference (hard-negative minus truth-only): {paired['global_pd_difference']['difference']:.5f}, paired 95% bootstrap CI {paired['global_pd_difference']['bootstrap_ci95']}. ",
        f"- Global joint-success difference: {paired['global_joint_success_difference']['difference']:.5f}, paired 95% bootstrap CI {paired['global_joint_success_difference']['bootstrap_ci95']}.",
        f"- C3 joint-success difference: {paired['C3']['joint_success_difference']['difference']:.5f}, paired 95% bootstrap CI {paired['C3']['joint_success_difference']['bootstrap_ci95']}.",
        f"- Hard-cell joint-success difference: {paired['hard_cell']['joint_success_difference']['difference']:.5f}, paired 95% bootstrap CI {paired['hard_cell']['joint_success_difference']['bootstrap_ci95']}.",
        "- For a conservative no-loss statement, the paired 95% interval must have a nonnegative lower bound; intervals crossing zero are inconclusive, not proof of equivalence.",
        "",
        "## Low-Pfa qualification",
        "",
    ]
    for target in ("0.05", "0.01", "0.001"):
        qualified = []
        for method in ("Medium_TruthOnly", "Medium_HardNegative"):
            item = methods[method]["workpoints"][target]
            upper = item["verification"]["pfa_ci95_exact_upper"]
            qualified.append(f"{method}: upper={upper:.6g} ({'PASS' if upper <= float(target) else 'not qualified'})")
        lines.append(f"- target {target}: " + "; ".join(qualified) + ".")
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "- This is one matched training seed. It estimates this pilot's effect; it does not estimate training-seed uncertainty or authorize final blind evaluation.",
        "- R6-B compares train-only H0 resampling against a matched truth-only retrain. It does not compare a new architecture, normalization, teacher loss, or larger effective batch.",
        "- The frozen original Medium is a contextual baseline. The causal comparison for the R intervention is the paired TruthOnly versus HardNegative retraining.",
        "- A low-Pfa point is qualified only when its independent verification exact upper bound meets the target. Pfa point estimates, ROC curves, and calibration results do not replace this gate.",
        "- Review the condition-level nuisance correlations, RMSE curves, hard-cell protection and the protocol-order deviation documented in the task alignment before a stage decision.",
        "",
    ]
    (STAGE / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _write_stage_review(metrics: Mapping[str, Any]) -> None:
    branch = json.loads((STAGE / "branch_selection.json").read_text(encoding="utf-8"))
    work = metrics["methods"]["Medium_HardNegative"]["workpoints"]["0.05"]
    paired = metrics["paired_hard_negative_vs_truth_only"]["0.05"]
    training = json.loads((RUNS / "training.json").read_text(encoding="utf-8"))
    decision = "REFINE"
    text = f"""# D2-A-R6-B 阶段审查报告

**提交基准：** `{metrics['git_sha']}`
**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3（本阶段未修改）
**选支：** R（{'; '.join(branch['selection_reasons'])}）
**阶段判断：** **{decision}**

## 科研问题与主线

本阶段服务总纲中的GMTI任务级检测和参数估计目标，重点检验困难C3条件ROC较差时，仅通过增加训练期H0困难样本覆盖能否改善学生的跨条件检测和联合任务质量。R6-B不是独立CFAR项目，也没有进行知识蒸馏或final blind。

## 工程与实验执行

- G0/G1：见 `preflight.json` 与 `g1.json`；本阶段要求RTX 3050 `cuda:0`，FP32，无CPU回退。
- 训练：两臂均Medium width96、seed303初始化、有效batch64、12,800 updates。H1计划、条件配额、初始化、优化器、验证集与checkpoint规则匹配；仅H0采样变化。模型顺序驻留GPU。
- hard-negative：冻结R5-A Medium只对R6-B训练H0打分，按条件取训练分数最高四分位；干预臂每条件全程H0槽位50%困难池、50%均匀池。验证、校准、verification和H1 development未用于挖掘。
- 数据：H0 calibration/verification各120,000；H1 development 6,144。scene IDs与历史阶段不重用。评价IQ不落盘；训练缓存已按stage owner清理。

## 关键证据

- HardNegative在目标Pfa 0.05下，H0 verification全局Pfa上界 `{work['verification']['pfa_ci95_exact_upper']:.5f}`，C3 pooled上界 `{work['c3_pooled_pfa']['pfa_ci95_exact_upper']:.5f}`，hard-cell上界 `{work['hard_cell_pfa']['pfa_ci95_exact_upper']:.5f}`。
- H1 development Pd `{work['h1']['pd']:.5f}`，C3 Pd `{work['h1']['c3_pd']:.5f}`，hard-cell Pd `{work['h1']['hard_cell_pd']:.5f}`。
- paired HardNegative−TruthOnly joint-success差：全局 `{paired['global_joint_success_difference']['difference']:.5f}`，95% CI `{paired['global_joint_success_difference']['bootstrap_ci95']}`；C3 `{paired['C3']['joint_success_difference']['difference']:.5f}`，CI `{paired['C3']['joint_success_difference']['bootstrap_ci95']}`；hard cell `{paired['hard_cell']['joint_success_difference']['difference']:.5f}`，CI `{paired['hard_cell']['joint_success_difference']['bootstrap_ci95']}`。
- Optimizer updates/数据遍历/训练时长、条件指标和低Pfa资格见 `training.json`、`metrics.json` 与 `analysis.md`。只有一个训练seed，不能给出跨seed稳定性结论。

## 研究判断

- R6-A支持总体C3有区分信息，但condition 10条件ROC较差，因此预注册R分支有依据；R6-B只运行R分支。
- hard-negative机制的价值由匹配重训差值判断；不得把它解释为更大batch、更强模型或蒸馏的证据。
- 0.05/0.01/0.001工作点分别按verification单侧exact上界判断资格；H1保护和困难格结果须同时查看。R6-B是development证据，不产生最终系统或部署结论。
- 当前仍应由独立审查评价hard-negative是否值得进入多seed，以及检测困难是否仍主要由condition 10支配。研究总纲v1.3保持不变。

## 决定与下一步

**阶段裁决：{decision}。** 本阶段只支持对训练期H0覆盖机制作一次有界判断。若HardNegative未展示稳定、成对且不损害虚警/联合任务的收益，停止该分支，不做更多hardness策略搜索；若有清晰收益，先提交ChatGPT审查，再讨论独立三seed确认协议。不得在本阶段生成blind或增加蒸馏损失。

"""
    (STAGE / "stage_review.md").write_text(text, encoding="utf-8")


def _write_chatgpt_review_request(metrics: Mapping[str, Any]) -> None:
    training = json.loads((RUNS / "training.json").read_text(encoding="utf-8"))
    config = _read_yaml("experiments/D2-A-R6/R6-B/config.yaml")
    review = f"""# ChatGPT 阶段审查请求：D2-A-R6-B

请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。

请按 `docs/governance/CHATGPT_REVIEW_PROTOCOL.md` 的四层审查流程执行。先核对提交 SHA、总纲版本和材料路径；不要假设了解仓库或本次对话历史。

**提交 SHA：** `{metrics['git_sha']}`（结果在该提交对应代码下生成；本报告和指标随后归档提交）
**工作区是否干净：** 结果归档完成后确认
**阶段 / 实验：** D2-A-R6-B branch R：训练期 H0 hard-negative resampling
**总纲路径及版本：** `docs/RESEARCH_CHARTER.md` v1.3，路径保持不变
**总体研究目标：** 通过物理真值和合格传统算法知识，训练任务级GMTI学生，以检测、连续参数估计和完整处理链评价其价值。
**对应科学问题：** 总纲Q1/Q4/Q6/Q9/Q10；本阶段具体检验训练期困难H0覆盖能否缓解C3困难条件的条件ROC/虚警问题。
**预注册门：** `experiments/D2-A-R6/R6-B/protocol.md` 与 `config.yaml`；选支R依据 `branch_selection.json`。
**希望审查的决策：** hard-negative R分支是否值得开展至少三seed确认；低Pfa与任务保护是否允许候选晋级。

## 模型、训练与数据

- Medium TaskStudent width=96，训练参数82,887，推理参数70,277。
- 两臂初始化seed303，12,800 optimizer updates，effective batch64（每步32 H0+32 H1），FP32 CUDA RTX 3050 Laptop。
- 学习率沿用R5-A Medium两段cosine：6400步0.001→0.0001，后6400步0.0001→0.00001；AdamW、weight decay 0.0001、gradient clip 1.0。
- H1样本顺序和条件配额相同；H0仅在HardNegative臂以每条件全程50% top-quartile困难样本、50%均匀样本抽取。训练日志按400步验证/检查点记录，不逐更新落盘。
- 样本量：H0 calibration 120,000；H0 verification 120,000；H1 development 6,144；训练51,200配对场景，验证5,000配对场景。所有新seed/scene ID与历史阶段互斥。
- 只有一个正式训练seed；无教师蒸馏、无final blind。

## 本阶段材料

- protocol/config：`experiments/D2-A-R6/R6-B/{protocol.md,config.yaml}`
- 实现：`scripts/run_d2a_r6b_low_pfa.py`、`src/radardistill/training/d2a_r6b_hard_negative.py`、`tests/test_d2a_r6b_hard_negative.py`
- 选支/G0/G1/训练/资源：`branch_selection.json`、`preflight.json`、`g1.json`、`runs/seed_303/training.json`、各臂训练历史与检查点
- 结果：`metrics.json`、`analysis.md`、`figures/`、`scene_manifest_*.csv`、`environment_*.json`
- 阶段报告：`stage_review.md`
- 结果内容SHA：`{metrics['git_sha']}`（指标声明的执行代码提交）
- 关键估计：HardNegative 对比 TruthOnly 的 paired bootstrap、C3/hard-cell条件指标与低Pfa exact upper bound均见 `metrics.json`。

请判断本阶段是否推动总纲目标，hard-negative问题是否值得继续，是否存在更有价值的替代方向，H-R6-DET-001及相关解释如何更新。请区分单seed开发证据、阶段门资格和可推广结论。最终只选 CONTINUE / REFINE / REDIRECT / STOP，并给唯一下一项优先任务和停止条件。
"""
    (STAGE / "review_request.md").write_text(review, encoding="utf-8")


def cleanup_training_cache() -> dict[str, Any]:
    """Remove only cache files owned by this R6-B run after reports exist."""
    preflight, head = _require_frozen_preflight()
    if not (STAGE / "metrics.json").is_file() or not (STAGE / "analysis.md").is_file():
        raise RuntimeError("R6-B cache cleanup requires completed evaluation and analysis")
    removed = 0
    removed_paths: list[str] = []
    for split in ("train", "validation"):
        directory = CACHE / split
        if not directory.exists():
            continue
        owner_path = directory / "owner.json"
        expected = {"owner": "D2-A-R6-B-training-cache", "git_sha": head,
                    "split": split, "scene_count": 51200 if split == "train" else 5000}
        if not owner_path.is_file() or json.loads(owner_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(f"refusing to remove unowned R6-B cache directory: {directory}")
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        shutil.rmtree(directory)
        removed += size
        removed_paths.append(str(directory.relative_to(ROOT)))
    hardness_meta = CACHE / "train_h0_hardness.json"
    hardness_scores = CACHE / "train_h0_hardness.npy"
    if hardness_meta.exists() or hardness_scores.exists():
        if not hardness_meta.is_file() or not hardness_scores.is_file():
            raise RuntimeError("incomplete R6-B train H0 hardness cache; refusing implicit deletion")
        record = json.loads(hardness_meta.read_text(encoding="utf-8"))
        if record.get("owner") != "D2-A-R6-B-training-H0-hardness" or record.get("git_sha") != head:
            raise RuntimeError("refusing to remove an unowned R6-B hardness cache")
        removed += hardness_meta.stat().st_size + hardness_scores.stat().st_size
        hardness_meta.unlink()
        hardness_scores.unlink()
        removed_paths.extend((str(hardness_meta.relative_to(ROOT)), str(hardness_scores.relative_to(ROOT))))
    if CACHE.exists() and not any(CACHE.iterdir()):
        CACHE.rmdir()
    total_bytes = int(subprocess.check_output(["du", "-sb", str(ROOT)], text=True).split()[0])
    record = {
        "git_sha": head, "removed_stage_owned_bytes": int(removed),
        "removed_paths": removed_paths, "project_bytes_after_cleanup": total_bytes,
        "project_limit_bytes": 20 * 1024**3,
        "project_below_20_gib": total_bytes < 20 * 1024**3,
        "retained_hard_negative_selection_bytes": (STAGE / "hard_negative_selection.npz").stat().st_size,
        "final_blind_generated": False,
    }
    _write_json(STAGE / "cleanup.json", record)
    if not record["project_below_20_gib"]:
        raise RuntimeError("project exceeds the final 20 GiB target; no non-stage files were removed")
    return record


def run_all() -> dict[str, Any]:
    results: dict[str, Any] = {"preflight": preflight()}
    results["train_cache"] = generate_paired_cache("train")
    results["validation_cache"] = generate_paired_cache("validation")
    results["hardness"] = compute_h0_hardness()
    results["g1"] = run_g1()
    results["training"] = run_formal_training()
    for split in ("h0_calibration", "h0_verification", "h1_development"):
        results[split] = run_evaluation_split(split)
    results["metrics"] = analyze_predictions()
    results["cleanup"] = cleanup_training_cache()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=(
        "preflight", "generate", "hardness", "g1", "train", "eval-calibration",
        "eval-verification", "eval-development", "analyze", "cleanup", "all",
    ))
    args = parser.parse_args(argv)
    actions = {
        "preflight": preflight,
        "hardness": compute_h0_hardness,
        "g1": run_g1,
        "train": run_formal_training,
        "eval-calibration": lambda: run_evaluation_split("h0_calibration"),
        "eval-verification": lambda: run_evaluation_split("h0_verification"),
        "eval-development": lambda: run_evaluation_split("h1_development"),
        "analyze": analyze_predictions,
        "cleanup": cleanup_training_cache,
        "all": run_all,
    }
    if args.phase == "generate":
        result = {
            "train": generate_paired_cache("train"),
            "validation": generate_paired_cache("validation"),
        }
    else:
        result = actions[args.phase]()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
