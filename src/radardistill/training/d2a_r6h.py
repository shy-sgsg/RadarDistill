"""Matched truth-only training utilities for the D2-A-R6-H architecture screen."""

from __future__ import annotations

import math
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from radardistill.evaluation.d2a_r6a import build_split_plan
from radardistill.evaluation.d2a_r6e import (
    h0_metrics,
    h1_metrics,
    paired_quality_bootstrap,
    physical_diagnostics,
    predict_eval_split,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_ssm import R6H_ARMS, build_r6h_student
from radardistill.training.d2a_r6b_hard_negative import (
    assemble_independent_h0_h1_batch,
    build_h0_sample_plan,
    build_repeated_h1_plan,
)
from radardistill.training.d2a_r6d_joint import (
    audit_r6d_scene_plans,
    build_r6d_paired_plan,
    r6d_learning_rate,
)
from radardistill.training.d2a_r6e import (
    _atomic_torch_save,
    _cpu_tree,
    _evaluate_batch_loss,
    _training_update,
    _validation_summary,
    _write_history,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-H"
ARTIFACTS = STAGE / "runs/seed_808"
CACHE = STAGE / "cache/training"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"


def ensure_r6h_owned_directory(directory: Path, owner: Mapping[str, Any]) -> None:
    """Create or validate an owner record without rebinding existing payloads."""

    directory.mkdir(parents=True, exist_ok=True)
    owner_path = directory / "owner.json"
    if owner_path.is_file():
        current = json.loads(owner_path.read_text(encoding="utf-8"))
        if current == dict(owner):
            return
        if any(path.name != "owner.json" for path in directory.iterdir()):
            raise RuntimeError(f"refusing to rebind a nonempty R6-H directory: {directory}")
    elif any(directory.iterdir()):
        raise RuntimeError(f"refusing to reuse unowned R6-H directory: {directory}")
    _write_json(owner_path, dict(owner))


def read_r6h_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def r6h_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_r6h_yaml(STAGE / "config.yaml")
    conditions = read_r6h_yaml(ROOT / config["radar"]["condition_source"])
    physics = read_r6h_yaml(ROOT / config["radar"]["physics_source"])
    return config, conditions, physics


def validate_r6h_config(
    config: Mapping[str, Any], *, require_execution: bool = True
) -> dict[str, Any]:
    """Validate fixed experiment invariants without querying CUDA or creating data."""

    try:
        experiment = config["experiment"]
        models = config["models"]
        data = config["data"]
        training = config["training"]
        evaluation = config["evaluation"]
        resources = config["resources"]
    except (KeyError, TypeError) as exc:
        raise ValueError("R6-H config is missing a required section") from exc

    if experiment.get("id") != "D2-A-R6-H":
        raise ValueError("R6-H config has an unexpected experiment id")
    if tuple(models.get("arms", ())) != R6H_ARMS:
        raise ValueError("R6-H must use the three registered arms: ce_tcn, complex_ssm, cnn_ssm")
    if models.get("baseline") != "ce_tcn":
        raise ValueError("R6-H baseline must be ce_tcn")
    tolerance = float(models.get("parameter_match_relative_tolerance", -1.0))
    expected = models.get("expected_parameters", {})
    reference = int(expected.get("ce_tcn", -1))
    if not 0.0 < tolerance < 1.0 or reference <= 0:
        raise ValueError("R6-H parameter matching configuration is invalid")
    lower, upper = reference * (1.0 - tolerance), reference * (1.0 + tolerance)
    for arm in R6H_ARMS:
        count = int(expected.get(arm, -1))
        if count <= 0 or not lower <= count <= upper:
            raise ValueError(f"R6-H {arm} parameter count falls outside the registered match range")

    if training.get("device") != "cuda:0" or resources.get("required_device") != "cuda:0":
        raise ValueError("R6-H requires explicit cuda:0")
    if training.get("required_gpu") != DEVICE_NAME or resources.get("required_gpu_name") != DEVICE_NAME:
        raise ValueError(f"R6-H requires {DEVICE_NAME}")
    if training.get("cpu_fallback") != "forbidden":
        raise ValueError("R6-H CPU fallback must be forbidden")
    if training.get("precision") != "FP32" or training.get("amp") is not False:
        raise ValueError("R6-H requires FP32 with AMP disabled")

    effective_batch = int(training.get("effective_batch_size_cpi", -1))
    per_update = training.get("per_update_examples", {})
    h0_count, h1_count = int(per_update.get("H0", -1)), int(per_update.get("H1", -1))
    if effective_batch != 64 or h0_count != h1_count or h0_count + h1_count != effective_batch:
        raise ValueError("R6-H effective batch must contain 32 H0 and 32 H1 examples")
    if int(training.get("fixed_updates_per_model", -1)) != 25_600:
        raise ValueError("R6-H fixed training budget must be 25,600 updates per model")
    if int(training.get("g1_micro_overfit_updates", -1)) != 200:
        raise ValueError("R6-H G1 must use the registered 200-update overfit check")
    if float(training.get("g1_min_relative_loss_reduction", -1.0)) != 0.10:
        raise ValueError("R6-H G1 minimum task-loss reduction must be 10%")
    if int(training.get("validation_interval_updates", -1)) <= 0:
        raise ValueError("R6-H validation interval must be positive")
    if int(training.get("resume_checkpoint_interval_updates", -1)) <= 0:
        raise ValueError("R6-H checkpoint interval must be positive")
    if training.get("history_written_every_update") is not False:
        raise ValueError("R6-H history must not be written at every optimizer update")
    if tuple(training.get("train_models_sequentially", ())) != R6H_ARMS:
        raise ValueError("R6-H arms must train sequentially in registered order")
    if training.get("supervision") != "physical_truth_only_E1":
        raise ValueError("R6-H architecture screen must use E1 physical truth only")

    if bool(experiment.get("final_blind_authorized", True)) or bool(
        evaluation.get("final_blind_authorized", True)
    ):
        raise ValueError("R6-H final blind is not authorized")
    if float(evaluation.get("target_pfa_cpi", -1.0)) != 0.05:
        raise ValueError("R6-H architecture screen target Pfa must remain 0.05")

    status = str(experiment.get("status", ""))
    execution_ready = (
        status == "preregistered"
        and experiment.get("execution_authorized") is True
        and bool(experiment.get("final_blind_authorized") is False)
    )
    prerequisites = config.get("prerequisites", {})
    if prerequisites:
        completion_sha = str(prerequisites.get("r6g_completion_sha", ""))
        execution_ready = execution_ready and (
            prerequisites.get("r6g_training_complete") is True
            and prerequisites.get("r6g_gpu_released") is True
            and len(completion_sha) == 40
        )
    else:
        execution_ready = False

    summary = {
        "execution_ready": bool(execution_ready),
        "protocol_status": status,
        "registered_arms": list(R6H_ARMS),
        "effective_batch_size_cpi": effective_batch,
        "updates_per_model": int(training["fixed_updates_per_model"]),
        "final_blind_authorized": False,
    }
    if require_execution and not execution_ready:
        raise RuntimeError("R6-H protocol is not frozen and authorized or R6-G prerequisite is incomplete")
    return summary


def build_r6h_split_plans(
    config: Mapping[str, Any], condition_config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Build all registered splits using the common D2-A scene planners."""

    data = config["data"]
    radar = config["radar"]
    conditions = condition_config["radar"]["conditions"]
    plans: dict[str, list[dict[str, Any]]] = {
        "train": build_r6d_paired_plan(data, conditions, "train"),
        "validation": build_r6d_paired_plan(data, conditions, "validation"),
    }
    eval_data = {
        key: data[key]
        for key in ("condition_seed_stride", "chunk_size", "chunk_seed_stride")
    }
    eval_data["condition_count"] = int(radar["condition_count"])
    eval_data["splits"] = {
        name: data["splits"][name]
        for name in ("h0_calibration", "h0_verification", "h1_development")
    }
    eval_config = {"data": eval_data, "radar": {"conditions": conditions}}
    plans.update({
        name: build_split_plan(eval_config, name)
        for name in ("h0_calibration", "h0_verification", "h1_development")
    })
    plans["g1_test"] = [{"seed": int(data["g1_test_seed"]), "count": 16}]
    return plans


def r6h_scene_plan_audit(
    plans: Mapping[str, list[Mapping[str, Any]]], *, prior_scene_ids: set[str]
) -> dict[str, Any]:
    counts = {
        name: sum(int(row["count"]) for row in rows)
        for name, rows in plans.items()
    }
    return audit_r6d_scene_plans(
        {name: list(rows) for name, rows in plans.items()},
        expected_counts=counts,
        prior_scene_ids=prior_scene_ids,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _source_sha256() -> dict[str, str]:
    paths = {
        "trainer": Path(__file__).resolve(),
        "model": ROOT / "src/radardistill/students/r6_ssm.py",
        "simulator": ROOT / "src/radardistill/simulation/d2a.py",
        "paired_plan": ROOT / "src/radardistill/training/d2a_r6d_joint.py",
        "sampler": ROOT / "src/radardistill/training/d2a_r6b_hard_negative.py",
        "loss": ROOT / "src/radardistill/distillation/task_losses.py",
        "evaluation": ROOT / "src/radardistill/evaluation/d2a_r6e.py",
        "entrypoint": ROOT / "scripts/run_d2a_r6h.py",
    }
    return {name: _sha256(path) for name, path in paths.items()}


def _prior_scene_ids() -> set[str]:
    result: set[str] = set()
    experiment_root = ROOT / "experiments"
    for path in experiment_root.rglob("*.csv"):
        if "manifest" not in path.name and path.name != "selected_scene_ids.csv":
            continue
        if STAGE == path.parent or STAGE in path.parents:
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


def _require_cuda() -> tuple[torch.device, str, int, int]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-H requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if name != DEVICE_NAME:
        raise RuntimeError(f"R6-H requires {DEVICE_NAME}, found {name}")
    free_vram, total_vram = torch.cuda.mem_get_info(0)
    if free_vram <= 0:
        raise RuntimeError("R6-H cuda:0 reports no free device memory")
    return torch.device("cuda:0"), name, int(free_vram), int(total_vram)


def r6h_preflight() -> dict[str, Any]:
    """G0: verify the frozen pushed experiment, disjoint scenes, storage, and CUDA."""

    config, condition_config, _ = r6h_configs()
    summary = validate_r6h_config(config, require_execution=True)
    experiment = config["experiment"]
    if experiment.get("protocol_version") != 1.0 or experiment.get("status") != "preregistered":
        raise RuntimeError("R6-H protocol/config is not frozen as preregistered")
    head = _git("rev-parse", "HEAD")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-H G0 requires the frozen implementation pushed to main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("R6-H G0 requires a clean tracked worktree")

    plans = build_r6h_split_plans(config, condition_config)
    counts = {name: sum(int(row["count"]) for row in rows) for name, rows in plans.items()}
    prior_ids = _prior_scene_ids()
    audit = r6h_scene_plan_audit(plans, prior_scene_ids=prior_ids)
    expected = {
        name: int(config["data"]["splits"][name]["paired_scenes"])
        for name in ("train", "validation")
    }
    for name in ("h0_calibration", "h0_verification", "h1_development"):
        expected[name] = int(config["data"]["splits"][name]["scene_count"])
    expected["g1_test"] = 16
    if counts != expected:
        raise RuntimeError(f"R6-H scene plan counts differ from config: {counts}")

    params: dict[str, int] = {}
    for arm in R6H_ARMS:
        model = build_r6h_student(arm, seed=int(config["data"]["training_seed"]))
        params[arm] = sum(int(parameter.numel()) for parameter in model.parameters())
        del model
    reference = params[config["models"]["baseline"]]
    tolerance = float(config["models"]["parameter_match_relative_tolerance"])
    bounds = [math.ceil(reference * (1 - tolerance)), math.floor(reference * (1 + tolerance))]
    if any(not bounds[0] <= params[arm] <= bounds[1] for arm in R6H_ARMS):
        raise RuntimeError(f"R6-H model parameter count is outside registered bounds {bounds}: {params}")

    disk = shutil.disk_usage(STAGE)
    remaining = _remaining_owned_cache_bytes(plans, git_sha)
    minimum = max(
        remaining + 4 * 1024**3,
        int(config["data"]["minimum_free_disk_bytes_before_generation"]),
    )
    if disk.free < minimum:
        raise RuntimeError(f"R6-H needs {minimum} free bytes before cache generation; found {disk.free}")
    device, gpu, free_vram, total_vram = _require_cuda()
    driver = "unavailable"
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    record = {
        "status": "G0_PASS",
        "git_sha": head,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_sha256(),
        "prerequisite_r6g_completion_sha": config["prerequisites"]["r6g_completion_sha"],
        "scene_counts": audit["scene_counts"],
        "historical_scene_id_count": len(prior_ids),
        "prior_scene_id_overlap": audit["prior_overlap_count"],
        "cross_split_scene_id_overlap": audit["overlap_count"],
        "model_parameters": params,
        "parameter_bounds": bounds,
        "disk_free_before_generation_bytes": int(disk.free),
        "minimum_free_bytes": minimum,
        "disk_total_bytes": int(disk.total),
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
        **summary,
    }
    _write_json(STAGE / "preflight.json", record)
    return record


def _require_preflight(*, require_cuda: bool = True) -> tuple[dict[str, Any], str]:
    path = STAGE / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run R6-H G0 preflight before G1 or formal data generation")
    record = json.loads(path.read_text(encoding="utf-8"))
    head = _git("rev-parse", "HEAD")
    if (
        record.get("status") != "G0_PASS"
        or record.get("git_sha") != head
        or record.get("protocol_sha256") != _sha256(STAGE / "protocol.md")
        or record.get("config_sha256") != _sha256(STAGE / "config.yaml")
        or record.get("source_sha256") != _source_sha256()
    ):
        raise RuntimeError("R6-H preflight does not match current code/protocol/config")
    if _git("branch", "--show-current") != "main" or _git("rev-parse", "origin/main") != head:
        raise RuntimeError("R6-H requires its frozen implementation pushed to main")
    if _git("status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("tracked source changed after R6-H G0")
    if require_cuda:
        _require_cuda()
    return record, head


def _cache_paths(split: str) -> dict[str, Path]:
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation caches are persisted")
    directory = CACHE / split
    names = ("iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id")
    return {name: directory / f"{name}.npy" for name in names}


def _cache_owner(split: str, plan: Sequence[Mapping[str, Any]], git_sha: str) -> dict[str, Any]:
    plan_hash = hashlib.sha256(
        json.dumps(list(plan), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {
        "owner": "D2-A-R6-H-paired-cache",
        "git_sha": git_sha,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_sha256(),
        "split": split,
        "scene_count": sum(int(row["count"]) for row in plan),
        "plan_sha256": plan_hash,
    }


def _remaining_owned_cache_bytes(
    plans: Mapping[str, Sequence[Mapping[str, Any]]], git_sha: str
) -> int:
    per_scene = int(
        2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
        + 2 * np.dtype(np.float32).itemsize
        + np.dtype(np.uint8).itemsize
        + 48 * np.dtype("<U1").itemsize
    )
    remaining = 0
    for split in ("train", "validation"):
        plan = plans[split]
        count = sum(int(spec["count"]) for spec in plan)
        directory = CACHE / split
        written = 0
        if directory.exists():
            owner_path, progress_path = directory / "owner.json", directory / "progress.json"
            if not owner_path.is_file() or not progress_path.is_file():
                raise RuntimeError(f"refusing to reuse unowned R6-H cache: {directory}")
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            if owner != _cache_owner(split, plan, git_sha):
                raise RuntimeError(f"R6-H cache owner differs from registered plan: {directory}")
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if progress.get("owner") != owner:
                raise RuntimeError(f"R6-H cache progress owner mismatch: {directory}")
            written = int(progress.get("rows_written", -1))
            if not 0 <= written <= count:
                raise RuntimeError(f"R6-H cache progress is invalid: {directory}")
        remaining += (count - written) * per_scene
    return int(remaining + 2 * 1024 * 1024)


def _cache_manifest_matches(split: str, plan: Sequence[Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> bool:
    manifest = STAGE / f"scene_manifest_{split}.csv"
    if not manifest.is_file():
        return False
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = np.asarray([
        f"d2a-scene-{int(spec['seed'])}-{index}"
        for spec in plan
        for index in range(int(spec["count"]))
    ], dtype=str)
    ids = np.asarray(arrays["scene_id"]).astype(str)
    return (
        len(rows) == len(expected)
        and np.array_equal(np.asarray([row["scene_id"] for row in rows], dtype=str), expected)
        and np.array_equal(ids, expected)
    )


def _load_pair_cache(split: str) -> dict[str, np.ndarray]:
    config, condition_config, _ = r6h_configs()
    plan = build_r6h_split_plans(config, condition_config)[split]
    expected_count = sum(int(row["count"]) for row in plan)
    directory = CACHE / split
    owner_path, progress_path = directory / "owner.json", directory / "progress.json"
    if not owner_path.is_file() or not progress_path.is_file():
        raise RuntimeError(f"R6-H {split} cache owner/progress record is missing")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    expected_owner = _cache_owner(split, plan, _git("rev-parse", "HEAD"))
    if owner != expected_owner:
        raise RuntimeError(f"R6-H {split} cache owner differs from the registered plan")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("owner") != owner or progress.get("rows_written") != expected_count:
        raise RuntimeError(f"R6-H {split} cache is incomplete")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in _cache_paths(split).items()}
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
    if any(arrays[name].shape != expected_shapes[name] or arrays[name].dtype != expected_dtypes[name] for name in arrays):
        raise RuntimeError(f"R6-H {split} cache shape/dtype differs from registered plan")
    if not _cache_manifest_matches(split, plan, arrays):
        raise RuntimeError(f"R6-H {split} manifest differs from cached scene IDs")
    return arrays


def generate_r6h_paired_cache(split: str) -> dict[str, Any]:
    """Generate or resume one owner-scoped paired H0/H1 CPU cache."""

    preflight, git_sha = _require_preflight(require_cuda=False)
    g1_path = STAGE / "g1.json"
    if not g1_path.is_file() or json.loads(g1_path.read_text(encoding="utf-8")).get("status") != "G1_PASS":
        raise RuntimeError("R6-H formal data generation requires G1_PASS")
    config, condition_config, physics = r6h_configs()
    plans = build_r6h_split_plans(config, condition_config)
    if split not in {"train", "validation"}:
        raise ValueError("only train and validation caches are persisted")
    plan = plans[split]
    count = sum(int(spec["count"]) for spec in plan)
    directory = CACHE / split
    owner = _cache_owner(split, plan, git_sha)
    paths = _cache_paths(split)
    free = int(shutil.disk_usage(STAGE).free)
    remaining = _remaining_owned_cache_bytes(plans, git_sha)
    reserve = 4 * 1024**3
    if free < remaining + reserve:
        raise RuntimeError(f"R6-H cache generation needs {remaining + reserve} free bytes; found {free}")
    ensure_r6h_owned_directory(directory, owner)
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
            raise RuntimeError("R6-H cache progress owner mismatch")
        completed = int(progress["completed_chunks"])
        offset = int(progress["rows_written"])
        expected_offset = sum(int(spec["count"]) for spec in plan[:completed])
        if not 0 <= completed <= len(plan) or offset != expected_offset:
            raise RuntimeError("R6-H cache cursor differs from its frozen plan")
        if manifest_path.is_file():
            with manifest_path.open("r+b") as handle:
                handle.truncate(int(progress["manifest_offset"]))
        elif offset:
            raise RuntimeError("R6-H manifest is missing for a nonempty cache")
    else:
        if any(path.name != "owner.json" for path in directory.iterdir()):
            raise RuntimeError("R6-H partial cache has no valid progress record")
        if manifest_path.exists():
            raise FileExistsError(f"refusing to overwrite unowned R6-H manifest: {manifest_path}")
        completed = offset = 0
        progress = {"owner": owner, "completed_chunks": 0, "rows_written": 0, "manifest_offset": 0}
        _write_json(progress_path, progress)
    for name, (shape, dtype) in shapes.items():
        if not paths[name].is_file():
            np.lib.format.open_memmap(paths[name], mode="w+", dtype=dtype, shape=shape).flush()
        else:
            current = np.load(paths[name], mmap_mode="r")
            if current.shape != shape or current.dtype != np.dtype(dtype):
                raise RuntimeError(f"R6-H cache shape/dtype mismatch for {name}")
    arrays = {name: np.load(path, mmap_mode="r+") for name, path in paths.items()}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        if handle.tell() == 0:
            writer.writerow(("scene_id", "split", "seed", "condition_index", "course", "scenario_id",
                             "snr_db", "cnr_db", "truth_range_cell", "truth_doppler_hz"))
            handle.flush()
            progress["manifest_offset"] = handle.tell()
            _write_json(progress_path, progress)
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
    del arrays
    arrays = _load_pair_cache(split)
    return {
        "split": split,
        "complete": True,
        "scenes": count,
        "chunks": len(plan),
        "cache_bytes": sum(path.stat().st_size for path in paths.values()),
        "manifest": str(manifest_path.relative_to(ROOT)),
        "preflight_git_sha": preflight["git_sha"],
        "cache_generator_git_sha": git_sha,
    }


def _paired_iq_and_truth(generated: Any) -> dict[str, np.ndarray]:
    count = len(generated.base_scene_id)
    iq_h0 = np.asarray(generated.iq_h0, dtype=np.complex64)
    iq_h1 = np.asarray(generated.iq_h1, dtype=np.complex64)
    if iq_h0.shape != (count, 3, 130, 32) or iq_h1.shape != iq_h0.shape:
        raise ValueError("R6-H paired simulator returned an unexpected IQ shape")
    return {
        "iq_h0": iq_h0,
        "iq_h1": iq_h1,
        "truth_range_cell": np.asarray(generated.truth_target_range_cell, dtype=np.float32),
        "truth_doppler_hz": np.asarray(generated.truth_target_doppler_hz, dtype=np.float32),
        "condition_index": np.zeros(count, dtype=np.uint8),
        "scene_id": np.asarray(generated.base_scene_id).astype("<U48"),
    }


def _probe_microbatch(device: torch.device, seed: int, config: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    reports: dict[str, Any] = {}
    initial = int(config["training"]["microbatch_initial"])
    fallbacks = [int(value) for value in config["training"]["microbatch_oom_fallbacks"]]
    for microbatch in (initial, *fallbacks):
        feasible = True
        arm_reports: dict[str, Any] = {}
        for arm in R6H_ARMS:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = build_r6h_student(arm, seed=seed).to(device)
            try:
                x = torch.randn(microbatch, 6, 130, 32, device=device)
                output = model(x)
                expected = {"presence_logit", "range_logits", "range_offset", "doppler_vector"}
                if set(output) != expected or any(not torch.isfinite(value).all() for value in output.values()):
                    raise FloatingPointError(f"R6-H {arm} probe returned invalid outputs")
                loss = sum(value.square().mean() for value in output.values())
                loss.backward()
                if not torch.isfinite(loss) or any(
                    parameter.grad is None or not torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError(f"R6-H {arm} probe returned invalid gradients")
                arm_reports[arm] = {
                    "status": "pass",
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            except torch.cuda.OutOfMemoryError:
                feasible = False
                arm_reports[arm] = {"status": "oom"}
            finally:
                del model
                torch.cuda.empty_cache()
        reports[str(microbatch)] = arm_reports
        if feasible:
            return microbatch, reports
    raise RuntimeError("R6-H cannot fit any registered microbatch while preserving effective batch 64")


def r6h_run_g1() -> dict[str, Any]:
    """G1: check output/gradient contracts and equal-budget 16-pair overfit."""

    preflight, _ = _require_preflight(require_cuda=False)
    config, condition_config, physics = r6h_configs()
    device, gpu, _, _ = _require_cuda()
    seed = int(config["data"]["g1_test_seed"])
    condition = condition_config["radar"]["conditions"][0]
    generated = generate_d2a_paired_batch(
        16, seed, float(condition["snr_db"]), float(condition["cnr_db"]), dict(physics),
        scenario_id=str(condition["scenario_id"]),
    )
    data = _paired_iq_and_truth(generated)
    microbatch, resource_probe = _probe_microbatch(device, int(config["data"]["training_seed"]), config)
    rng = np.random.default_rng(int(config["data"]["sampler_seed"]))
    update_count = int(config["training"]["g1_micro_overfit_updates"])
    pair_plan = [
        (rng.integers(0, 16, size=32, dtype=np.int64), rng.integers(0, 16, size=32, dtype=np.int64))
        for _ in range(update_count)
    ]
    arm_reports: dict[str, Any] = {}
    for arm in R6H_ARMS:
        torch.manual_seed(int(config["data"]["training_seed"]))
        torch.cuda.manual_seed_all(int(config["data"]["training_seed"]))
        model = build_r6h_student(arm, seed=int(config["data"]["training_seed"])).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=float(config["training"]["learning_rate_initial"]),
            weight_decay=float(config["training"]["weight_decay"]),
        )
        initial_batch = assemble_independent_h0_h1_batch(data, *pair_plan[0])
        initial_loss = _evaluate_batch_loss(model, initial_batch, microbatch=microbatch, device=device)["total"]
        model.train()
        updates: list[float] = []
        for h0_indices, h1_indices in pair_plan:
            batch = assemble_independent_h0_h1_batch(data, h0_indices, h1_indices)
            terms = _training_update(
                model, optimizer, batch, microbatch=microbatch,
                effective_batch=int(config["training"]["effective_batch_size_cpi"]),
                device=device, gradient_clip=float(config["training"]["gradient_clip_norm"]),
            )
            updates.append(float(terms["total"]))
        last_mean = float(np.mean(updates[-20:]))
        reduction = (initial_loss - last_mean) / max(abs(initial_loss), 1e-12)
        arm_reports[arm] = {
            "initial_loss": float(initial_loss),
            "last20_mean_loss": last_mean,
            "relative_loss_reduction": float(reduction),
            "minimum_relative_reduction": float(config["training"]["g1_min_relative_loss_reduction"]),
            "updates": update_count,
            "unique_paired_scenes": 16,
            "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
            "microbatch": microbatch,
            "passed": bool(np.isfinite(updates).all() and reduction >= float(config["training"]["g1_min_relative_loss_reduction"])),
        }
        del model, optimizer
        torch.cuda.empty_cache()
    status = "G1_PASS" if all(row["passed"] for row in arm_reports.values()) else "G1_FAIL"
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
        "formal_train_validation_data_generated": any((CACHE / split / "progress.json").is_file() for split in ("train", "validation")),
    }
    _write_json(STAGE / "g1.json", record)
    return record


def _training_owner(arm: str, config: Mapping[str, Any], git_sha: str, sampler_audit: Mapping[str, Any]) -> dict[str, Any]:
    model = build_r6h_student(arm, seed=int(config["data"]["training_seed"]))
    return {
        "owner": "D2-A-R6-H-training-run",
        "git_sha": git_sha,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": _source_sha256(),
        "arm": arm,
        "training_seed": int(config["data"]["training_seed"]),
        "sampler_seed": int(config["data"]["sampler_seed"]),
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
        "updates": int(config["training"]["fixed_updates_per_model"]),
        "parameters": sum(int(parameter.numel()) for parameter in model.parameters()),
        "sampler_audit": dict(sampler_audit),
        "precision": "FP32",
        "data": "train_51200_paired_scenes",
    }


def r6h_train_arm(arm: str) -> dict[str, Any]:
    if arm not in R6H_ARMS:
        raise ValueError(f"unknown R6-H arm: {arm}")
    preflight, git_sha = _require_preflight(require_cuda=False)
    config, _, _ = r6h_configs()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-H training requires G1_PASS for all candidates")
    effective_batch = int(config["training"]["effective_batch_size_cpi"])
    microbatch = int(g1["formal_microbatch"])
    if effective_batch != 64 or microbatch not in (4, 2, 1):
        raise RuntimeError("R6-H effective batch or common microbatch differs from the frozen protocol")
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
    out = ARTIFACTS / arm
    ensure_r6h_owned_directory(out, owner)
    device, _, _, _ = _require_cuda()
    torch.manual_seed(int(config["data"]["training_seed"]))
    torch.cuda.manual_seed_all(int(config["data"]["training_seed"]))
    model = build_r6h_student(arm, seed=int(config["data"]["training_seed"])).to(device)
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
            raise RuntimeError("R6-H resume checkpoint owner mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state["update"])
        microbatch = int(state.get("microbatch", microbatch))
        history = [row for row in state.get("history", []) if int(row["update"]) <= start_update]
        best_objective, best_update = float(state["best_objective"]), int(state["best_update"])
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
                raise RuntimeError("R6-H best checkpoint owner mismatch")
            saved_best = float(best_state["validation_objective"])
            if saved_best < best_objective:
                best_objective = saved_best
                best_update = int(best_state["update"])
            elif not math.isclose(saved_best, best_objective, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError("R6-H best checkpoint objective conflicts with resume checkpoint")
        _write_history(history_path, history)
    elif any(path.name != "owner.json" for path in out.iterdir()):
        raise RuntimeError("R6-H run contains artifacts without an owner-bound checkpoint")
    else:
        _atomic_torch_save({
            "owner": owner, "update": 0,
            "model_state_dict": _cpu_tree(model.state_dict()),
            "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            "history": history, "best_objective": best_objective, "best_update": best_update,
            "training_seconds_completed": 0.0, "microbatch": microbatch,
            "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(), "sampler_audit": sampler_audit,
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
        sampled = assemble_independent_h0_h1_batch(train, h0_plan[update - 1], h1_plan[update - 1])
        while True:
            try:
                terms = _training_update(
                    model, optimizer, sampled, microbatch=microbatch, effective_batch=effective_batch,
                    device=device, gradient_clip=float(config["training"]["gradient_clip_norm"]),
                )
                break
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                fallbacks = [int(value) for value in config["training"]["microbatch_oom_fallbacks"]]
                smaller = [value for value in fallbacks if value < microbatch and effective_batch % value == 0]
                if not smaller:
                    raise RuntimeError("R6-H OOM at minimum microbatch; effective batch remains unchanged")
                microbatch = max(smaller)
        interval_count += 1
        for key in interval_sums:
            interval_sums[key] += terms[key]
        if update % validation_interval == 0:
            validation_metrics = _validation_summary(model, validation, microbatch=microbatch, device=device)
            objective = float(validation_metrics["total"])
            if objective < best_objective:
                best_objective, best_update = objective, update
                _atomic_torch_save({
                    "owner": owner, "update": update, "validation_objective": objective,
                    "model_state_dict": _cpu_tree(model.state_dict()), "validation_metrics": validation_metrics,
                }, best_path)
            row = {
                "update": update, "learning_rate": lr,
                **{f"interval_train_{key}": interval_sums[key] / max(interval_count, 1) for key in interval_sums},
                **{f"validation_{key}": value for key, value in validation_metrics.items()},
                "microbatch": microbatch,
                "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            history.append(row)
            _write_history(history_path, history)
            print(json.dumps({
                "stage": "train", "arm": arm, "update": update,
                "train_total": row["interval_train_total"], "validation_total": objective,
                "best_update": best_update, "microbatch": microbatch,
                "cuda_peak_allocated_bytes": row["cuda_peak_allocated_bytes"],
            }, ensure_ascii=False), flush=True)
            interval_sums = {key: 0.0 for key in interval_sums}
            interval_count = 0
            model.train()
        if update % checkpoint_interval == 0:
            elapsed = elapsed_before + time.perf_counter() - run_start
            _atomic_torch_save({
                "owner": owner, "update": update,
                "model_state_dict": _cpu_tree(model.state_dict()),
                "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
                "history": history, "best_objective": best_objective, "best_update": best_update,
                "training_seconds_completed": elapsed, "microbatch": microbatch,
                "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state(device).cpu(), "sampler_audit": sampler_audit,
            }, last_path)
    train_seconds = elapsed_before + time.perf_counter() - run_start
    if not best_path.is_file():
        raise RuntimeError(f"R6-H {arm} completed without a validation-selected checkpoint")
    record = {
        "status": "complete", "arm": arm, "owner": owner,
        "updates": target_updates, "best_update": best_update,
        "best_validation_objective": best_objective, "training_seconds": train_seconds,
        "effective_batch_size": effective_batch, "minimum_microbatch": microbatch,
        "unique_train_paired_scenes": len(train["scene_id"]),
        "scene_visits_per_presence": target_updates * (effective_batch // 2) / len(train["scene_id"]),
        "total_cpi_exposures_per_presence": target_updates * (effective_batch // 2),
        "model_parameter_count": sum(int(parameter.numel()) for parameter in model.parameters()),
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
        "owner": owner, "update": target_updates,
        "model_state_dict": _cpu_tree(model.state_dict()),
        "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
        "history": history, "best_objective": best_objective, "best_update": best_update,
        "training_seconds_completed": train_seconds, "microbatch": microbatch,
        "sampler_audit": sampler_audit,
    }, last_path)
    del model, optimizer, train, validation
    torch.cuda.empty_cache()
    return record


def r6h_train_all() -> dict[str, Any]:
    _require_preflight(require_cuda=False)
    path = ARTIFACTS / "training_summary.json"
    summary: dict[str, Any] = {
        "status": "in_progress",
        "registered_order": list(R6H_ARMS),
        "arms": {},
    }
    _write_json(path, summary)
    for arm in R6H_ARMS:
        summary["arms"][arm] = r6h_train_arm(arm)
        _write_json(path, summary)
    summary["status"] = "complete"
    _write_json(path, summary)
    return summary


def validate_r6h_training_complete(summary: Mapping[str, Any]) -> None:
    arms = summary.get("arms")
    if (
        summary.get("status") != "complete"
        or not isinstance(arms, Mapping)
        or set(arms) != set(R6H_ARMS)
        or any(not isinstance(arms[arm], Mapping) or arms[arm].get("status") != "complete" for arm in R6H_ARMS)
    ):
        raise RuntimeError("R6-H cache cleanup requires all registered arms must finish")


def _load_best_model(arm: str, *, device: torch.device, seed: int) -> torch.nn.Module:
    checkpoint_path = ARTIFACTS / arm / "best_checkpoint.pt"
    training_path = ARTIFACTS / arm / "training_record.json"
    if not checkpoint_path.is_file() or not training_path.is_file():
        raise RuntimeError(f"R6-H {arm} training result is incomplete")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("status") != "complete" or training.get("arm") != arm:
        raise RuntimeError(f"R6-H {arm} training record is invalid")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("owner") != training.get("owner"):
        raise RuntimeError(f"R6-H {arm} best checkpoint owner mismatch")
    model = build_r6h_student(arm, seed=seed).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def r6h_evaluate() -> dict[str, Any]:
    """Evaluate selected checkpoints once on the registered calibration/development splits."""

    preflight, _ = _require_preflight(require_cuda=False)
    config, condition_config, physics = r6h_configs()
    data = config["data"]
    conditions = condition_config["radar"]["conditions"]
    plans = build_r6h_split_plans(config, condition_config)
    device, gpu, free_vram, total_vram = _require_cuda()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    microbatch = int(g1["formal_microbatch"])
    metrics: dict[str, Any] = {
        "status": "evaluated",
        "git_sha": preflight["git_sha"],
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "gpu": gpu,
        "device": str(device),
        "free_vram_before_evaluation_bytes": free_vram,
        "total_vram_bytes": total_vram,
        "precision": "FP32",
        "microbatch": microbatch,
        "final_blind_generated": False,
        "arms": {},
        "contrasts": {},
    }
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    thresholds: dict[str, float] = {}
    for arm in R6H_ARMS:
        model = _load_best_model(arm, device=device, seed=int(data["training_seed"]))
        current = {
            split: predict_eval_split(
                model, split, plans[split], dict(physics), device=device, microbatch=microbatch,
                prf_hz=float(config["radar"]["prf_hz"]),
            )
            for split in ("h0_calibration", "h0_verification", "h1_development")
        }
        threshold = float(np.quantile(
            current["h0_calibration"]["score"],
            float(config["evaluation"]["calibration_quantile"]),
            method=str(config["evaluation"]["calibration_quantile_method"]),
        ))
        h0 = h0_metrics(
            current["h0_calibration"], current["h0_verification"],
            threshold=threshold, conditions=conditions,
        )
        h1 = h1_metrics(
            current["h1_development"], threshold=threshold, conditions=conditions,
            h0_verification=current["h0_verification"], prf_hz=float(config["radar"]["prf_hz"]),
        )
        physical = physical_diagnostics(
            model, plans["h1_development"], dict(physics), current["h1_development"],
            threshold=threshold, device=device, microbatch=microbatch,
            prf_hz=float(config["radar"]["prf_hz"]),
            phase_count=int(config["evaluation"]["common_phase_rotation_count"]),
            frequency_shifts_hz=config["evaluation"]["frequency_shift_intervention_hz"],
        )
        predictions[arm] = current
        thresholds[arm] = threshold
        metrics["arms"][arm] = {
            "threshold": threshold,
            "h0": h0,
            "h1": h1,
            "physical_diagnostics": physical,
            "training_record": str((ARTIFACTS / arm / "training_record.json").relative_to(ROOT)),
        }
        _write_json(STAGE / "metrics.partial.json", metrics)
        del model
        torch.cuda.empty_cache()
    baseline = str(config["models"]["baseline"])
    evaluation = config["evaluation"]
    for index, candidate in enumerate(R6H_ARMS):
        if candidate == baseline:
            continue
        contrast = paired_quality_bootstrap(
            predictions[baseline]["h1_development"], predictions[candidate]["h1_development"],
            threshold_baseline=thresholds[baseline], threshold_candidate=thresholds[candidate],
            conditions=conditions,
            replicates=int(evaluation["paired_bootstrap_replicates"]),
            seed=int(evaluation["paired_bootstrap_seed"]) + index,
            confidence=float(evaluation["candidate_contrast_ci_level"]),
        )
        metrics["contrasts"][f"{candidate}_minus_{baseline}"] = contrast
    partial = STAGE / "metrics.partial.json"
    if partial.exists():
        partial.unlink()
    _write_json(STAGE / "metrics.json", metrics)
    return metrics


def cleanup_owned_r6h_training_cache() -> dict[str, Any]:
    """Remove only complete cache directories with matching R6-H owner records."""

    preflight, git_sha = _require_preflight(require_cuda=False)
    training_summary_path = ARTIFACTS / "training_summary.json"
    if not training_summary_path.is_file():
        raise RuntimeError("R6-H cache cleanup requires all registered arms must finish")
    validate_r6h_training_complete(json.loads(training_summary_path.read_text(encoding="utf-8")))
    config, condition_config, _ = r6h_configs()
    plans = build_r6h_split_plans(config, condition_config)
    removed: list[dict[str, Any]] = []
    expected_files = {"owner.json", "progress.json", *(f"{name}.npy" for name in (
        "iq_h0", "iq_h1", "truth_range_cell", "truth_doppler_hz", "condition_index", "scene_id",
    ))}

    def require_no_symlink_in_cache_path(path: Path) -> None:
        root = Path(ROOT)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"refusing to clean a cache outside the project root: {path}") from exc
        current = path
        while True:
            if current.is_symlink():
                raise RuntimeError(f"refusing to clean through a symlinked R6-H cache path: {current}")
            if current == root:
                break
            current = current.parent

    require_no_symlink_in_cache_path(CACHE)
    validated_directories: list[tuple[Path, list[Path], int]] = []
    for split in ("train", "validation"):
        directory = CACHE / split
        require_no_symlink_in_cache_path(directory)
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise RuntimeError(f"refusing to clean a non-directory R6-H cache: {directory}")
        owner_path, progress_path = directory / "owner.json", directory / "progress.json"
        if owner_path.is_symlink() or progress_path.is_symlink():
            raise RuntimeError(f"refusing to clean an R6-H cache with symlinked owner metadata: {directory}")
        if not owner_path.is_file() or not progress_path.is_file():
            raise RuntimeError(f"refusing to clean unowned R6-H cache: {directory}")
        owner = json.loads(owner_path.read_text(encoding="utf-8"))
        if owner != _cache_owner(split, plans[split], git_sha):
            raise RuntimeError(f"refusing to clean cache with a different R6-H owner: {directory}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("owner") != owner or int(progress.get("rows_written", -1)) != int(owner["scene_count"]):
            raise RuntimeError(f"refusing to clean incomplete R6-H cache: {directory}")
        children = list(directory.iterdir())
        if (
            {child.name for child in children} != expected_files
            or any(child.is_symlink() or not child.is_file() for child in children)
        ):
            raise RuntimeError(f"R6-H cache contents differ from the complete registered file set: {directory}")
        removed_bytes = sum(child.stat().st_size for child in children)
        validated_directories.append((directory, children, removed_bytes))

    for directory, children, removed_bytes in validated_directories:
        require_no_symlink_in_cache_path(directory)
        for child in children:
            child.unlink()
        directory.rmdir()
        removed.append({"split": directory.name, "bytes": removed_bytes})
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
