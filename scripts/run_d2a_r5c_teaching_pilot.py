#!/usr/bin/env python3
"""Run the reviewed D2-A-R5-C teaching-content pilot in explicit stages."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
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
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_d2a_r5 as r5_runner
from radardistill.students.task_student import TaskStudent, export_inference_model
from radardistill.evaluation.d2a_r5c_teaching_pilot import (
    iter_generated_split_chunks,
    paired_teaching_contrasts,
    predict_k3_teacher_localization,
    predict_task_student_chunk,
    summarize_pilot_arm,
)
from radardistill.training.d2a_r5c_teaching_pilot import (
    FROZEN_INPUTS,
    _require_training_authorization,
    _require_training_cuda,
    _require_training_preflight,
    assert_split_seed_disjoint,
    build_initial_state,
    build_all_seed_specs,
    _validate_pair_plan,
    run_overfit_sanity,
    train_arms_sequentially,
    validate_r5c_teaching_config,
)
from radardistill.training.d2a_e002_data import materialize_paired_split
from radardistill.training.d2a_r5 import build_r5_repeated_pair_plan

CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-C-teaching/config.yaml"
OUTPUT_ROOT = ROOT / "experiments/D2-A-R5/R5-C-teaching"
R5_CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-A/config.yaml"
PHYSICS_CONFIG_PATH = ROOT / "experiments/D2-A-E001/config.yaml"
E003_CONFIG_PATH = ROOT / "experiments/D2-A-E003/config.yaml"
STAGE_ROOT = ROOT / "experiments/D2-A-R5/R5-C-teaching"
DATA_GENERATION_PATH = STAGE_ROOT / "data_generation.json"
SCENE_MANIFEST_PATH = STAGE_ROOT / "scene_manifest.csv"
SEED_MANIFEST_PATH = STAGE_ROOT / "seed_manifest.csv"
STAGE_REVIEW_PATH = ROOT / "docs/stage_reviews/D2-A-R5C-teaching_pilot_review.md"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _require_committed_main() -> str:
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R5-C teaching requires the user-authorized main branch")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "origin/main"):
        raise RuntimeError("R5-C teaching requires reviewed code/config pushed to origin/main")
    for relative in FROZEN_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        check = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
        if check.returncode != 0:
            raise RuntimeError(f"R5-C teaching frozen input has local changes: {relative}")
    return head


def _require_committed_config(config: dict[str, Any]) -> None:
    if config != _read_yaml(CONFIG_PATH):
        raise RuntimeError("R5-C teaching config must match the committed config on main")


def _existing_seed_collisions(specs: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    from run_d2a_r5c_teacher_audit import _existing_scene_seeds

    lower = min(int(spec["seed_block_start"]) for spec in specs)
    upper = max(int(spec["seed_block_end_exclusive"]) for spec in specs)
    return _existing_scene_seeds(lower, upper)


def _project_size_bytes() -> int:
    result = subprocess.run(
        ["du", "-sb", str(ROOT)], check=True, text=True, capture_output=True
    )
    return int(result.stdout.split()[0])


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _old_r5a_scene_ids() -> set[str]:
    """Read only the scene-ID columns from the historical R5-A train/validation caches."""
    r5_config = _read_yaml(R5_CONFIG_PATH)
    cache_root = ROOT / r5_config["data"]["cache_root"]
    arrays = {
        split: r5_runner._load_split(cache_root, split)
        for split in ("train", "validation")
    }
    expected = {
        "train": int(r5_config["data"]["paired_training_scenes"]),
        "validation": int(r5_config["data"]["paired_validation_scenes"]),
    }
    result: set[str] = set()
    for split, values in arrays.items():
        ids = np.asarray(values["base_scene_id"]).reshape(-1).astype(str)
        if ids.size != expected[split] or np.unique(ids).size != ids.size:
            raise RuntimeError(f"historical R5-A {split} IDs are missing or invalid")
        result.update(ids.tolist())
    return result


def _cache_file_records(cache_root: Path) -> list[dict[str, Any]]:
    records = []
    for split in ("train", "validation"):
        directory = cache_root / split
        if not directory.is_dir():
            raise FileNotFoundError(f"R5-C teaching cache split is missing: {directory}")
        for path in sorted(directory.iterdir()):
            if path.is_file():
                records.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "size_bytes": int(path.stat().st_size),
                    }
                )
    return records


def _load_stage_cache(config: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError(f"unsupported R5-C teaching cache split: {split}")
    configured_path = config["data"]["training_source" if split == "train" else "validation_source"]
    directory = ROOT / configured_path
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"R5-C teaching {split} cache has no generation manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_count = int(config["data"]["training_pairs" if split == "train" else "validation_pairs"])
    if manifest.get("split") != split or manifest.get("total_paired_scenes") != expected_count:
        raise ValueError(f"R5-C teaching {split} cache identity/count differs from its config")
    names = (
        "iq_h0", "iq_h1", "base_scene_id", "course_level", "scenario_id",
        "truth_range_cell", "truth_doppler_hz", "truth_radial_velocity_mps",
        "truth_azimuth_deg", "truth_phase_rad", "truth_amplitude", "snr_db",
        "requested_cnr_db", "noise_power", "measured_snr_db", "measured_cnr_db",
        "measured_scnr_db",
    )
    missing = [name for name in names if not (directory / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"R5-C teaching {split} cache is incomplete: {missing}")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in names}
    if any(len(value) != expected_count for value in arrays.values()):
        raise ValueError(f"R5-C teaching {split} cache array length differs from its manifest")
    return arrays


def _require_data_generation(config: dict[str, Any], head: str, *, require_iq: bool) -> dict[str, Any]:
    if not DATA_GENERATION_PATH.is_file():
        raise RuntimeError("R5-C teaching fresh train/validation data must be generated after preflight")
    record = json.loads(DATA_GENERATION_PATH.read_text(encoding="utf-8"))
    preflight_path = OUTPUT_ROOT / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("R5-C teaching generated data has no preflight provenance")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    source_head = preflight.get("git_sha")
    if (
        record.get("git_sha") != source_head
        or preflight.get("config") != config
        or (require_iq and head != source_head)
        or record.get("config") != config
    ):
        raise RuntimeError("R5-C teaching generated data does not match this pushed code/config")
    for split, key in (("train", "training_source"), ("validation", "validation_source")):
        directory = ROOT / config["data"][key]
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"R5-C teaching {split} cache manifest is missing")
        actual_manifest = json.loads(manifest.read_text(encoding="utf-8"))
        if record.get("cache_manifests", {}).get(split) != actual_manifest:
            raise ValueError(f"R5-C teaching {split} cache manifest changed after data generation")
        if require_iq:
            _load_stage_cache(config, split)
    if require_iq:
        for item in record.get("cache_files", []):
            path = (ROOT / item["path"]).resolve()
            try:
                path.relative_to(STAGE_ROOT.resolve())
            except ValueError as error:
                raise RuntimeError("R5-C teaching data manifest points outside its stage directory") from error
            if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
                raise RuntimeError(f"R5-C teaching generated cache file is missing/changed: {path}")
    pair_plan_path = ROOT / config["data"]["pair_plan_source"]
    if not pair_plan_path.is_file():
        raise FileNotFoundError("R5-C teaching fixed pair plan is missing")
    if not SCENE_MANIFEST_PATH.is_file() or not SEED_MANIFEST_PATH.is_file():
        raise FileNotFoundError("R5-C teaching scene/seed audit manifests are missing")
    return record


def _training_validation_scene_ids(config: dict[str, Any]) -> set[str]:
    if not SCENE_MANIFEST_PATH.is_file():
        raise FileNotFoundError("R5-C teaching train/validation scene manifest is missing")
    seen: set[str] = set()
    counts = {"train": 0, "validation": 0}
    with SCENE_MANIFEST_PATH.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"split", "base_scene_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("R5-C teaching scene manifest lacks split or base_scene_id")
        for row in reader:
            split = row["split"]
            scene_id = row["base_scene_id"]
            if split not in counts or not scene_id or scene_id in seen:
                raise ValueError("R5-C teaching train/validation scene manifest has invalid/duplicate IDs")
            seen.add(scene_id)
            counts[split] += 1
    expected = {
        "train": int(config["data"]["training_pairs"]),
        "validation": int(config["data"]["validation_pairs"]),
    }
    if counts != expected:
        raise ValueError(f"R5-C teaching scene manifest counts differ: {counts}")
    return seen


def _assert_fresh_scene_ids(
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    historical_ids: set[str],
    *,
    train_count: int,
    validation_count: int,
) -> None:
    train_values = np.asarray(train_ids).reshape(-1).astype(str)
    validation_values = np.asarray(validation_ids).reshape(-1).astype(str)
    if train_values.size != train_count or validation_values.size != validation_count:
        raise RuntimeError("R5-C teaching train/validation scene count differs from the frozen protocol")
    if np.unique(train_values).size != train_values.size or np.unique(validation_values).size != validation_values.size:
        raise RuntimeError("R5-C teaching generated duplicate base_scene_id values")
    if set(train_values).intersection(validation_values):
        raise RuntimeError("R5-C teaching train and validation scene IDs overlap")
    if set(train_values).intersection(historical_ids) or set(validation_values).intersection(historical_ids):
        raise RuntimeError("R5-C teaching generated scenes overlap frozen R5-A train/validation IDs")


def _model_parameter_counts() -> tuple[int, int]:
    training_model = TaskStudent(width=96, spectrum_frequency_bins=520)
    training_count = sum(parameter.numel() for parameter in training_model.parameters())
    inference_model = export_inference_model(training_model)
    inference_count = sum(parameter.numel() for parameter in inference_model.parameters())
    del inference_model, training_model
    return training_count, inference_count


def run_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Check review/freeze, all seed blocks, CUDA and disk; generate nothing."""
    _require_training_authorization(config)
    head = _require_committed_main()
    _require_committed_config(config)
    if (OUTPUT_ROOT / "preflight.json").is_file():
        previous = json.loads((OUTPUT_ROOT / "preflight.json").read_text(encoding="utf-8"))
        if previous.get("git_sha") != head or previous.get("config") != config:
            raise RuntimeError("existing R5-C teaching preflight belongs to another frozen run")
        return previous
    specs = build_all_seed_specs(config)
    collisions = _existing_seed_collisions(specs)
    assert_split_seed_disjoint(specs, [seed for _, seed, _ in collisions])

    device = _require_training_cuda("cuda:0")
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    minimum_free_vram = int(config["resources"]["minimum_free_vram_bytes"])
    if free_vram < minimum_free_vram:
        raise RuntimeError(f"R5-C teaching needs {minimum_free_vram} free VRAM bytes; found {free_vram}")
    free_disk = shutil.disk_usage(ROOT).free
    minimum_free_disk = int(config["resources"]["minimum_free_disk_bytes"])
    train_count = int(config["data"]["training_pairs"])
    validation_count = int(config["data"]["validation_pairs"])
    iq_bytes_per_pair = 2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
    estimated_iq_bytes = (train_count + validation_count) * iq_bytes_per_pair
    required_free_disk = max(minimum_free_disk, estimated_iq_bytes + 3 * 1024**3)
    if free_disk < required_free_disk:
        raise RuntimeError(
            f"R5-C teaching needs {required_free_disk / 1024**3:.2f} GiB free for new paired caches "
            f"and working reserve; found {free_disk / 1024**3:.2f} GiB"
        )
    project_size = _project_size_bytes()
    old_ids = _old_r5a_scene_ids()

    for key in ("training_source", "validation_source", "pair_plan_source"):
        path = (ROOT / config["data"][key]).resolve()
        try:
            path.relative_to(STAGE_ROOT.resolve())
        except ValueError as error:
            raise RuntimeError(f"R5-C teaching {key} must remain inside the stage directory") from error

    initial_state = build_initial_state(int(config["training"]["initialization_seed"]))
    training_parameters, inference_parameters = _model_parameter_counts()
    if training_parameters != int(config["model"]["training_parameter_count_expected"]):
        raise RuntimeError(
            f"R5-C teaching training parameter count differs: {training_parameters}"
        )
    if inference_parameters != int(config["model"]["inference_parameter_count_expected"]):
        raise RuntimeError(
            f"R5-C teaching inference parameter count differs: {inference_parameters}"
        )
    environment = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_sha": head,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_free_bytes_before": int(free_vram),
        "cuda_total_bytes": int(total_vram),
        "disk_free_bytes": int(free_disk),
        "estimated_new_paired_iq_bytes": int(estimated_iq_bytes),
        "required_free_disk_bytes": int(required_free_disk),
        "project_size_bytes_before": int(project_size),
        "training_pair_plan_shape_expected": [12800, 32],
        "training_pair_plan_seed_base": int(config["data"]["pair_plan_seed_base"]),
        "training_parameter_count": training_parameters,
        "inference_parameter_count": inference_parameters,
        "initial_state_tensor_count": len(initial_state),
        "new_train_validation_evaluation_chunks": len(specs),
        "new_train_validation_evaluation_scenes": sum(int(spec["scene_count"]) for spec in specs),
        "historical_r5a_scene_ids_checked": len(old_ids),
        "training_data_reused": False,
        "validation_data_reused": False,
        "config": config,
        "seed_collisions": 0,
        "final_blind_generated": False,
        "passed": True,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_ROOT / "preflight.json", environment)
    return environment


def _write_csv_once(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            existing = list(reader)
            if reader.fieldnames != fieldnames or len(existing) != len(rows):
                raise FileExistsError(f"existing R5-C teaching audit manifest differs; preserving: {path}")
            for existing_row, row in zip(existing, rows, strict=True):
                if any(str(existing_row[key]) != str(row[key]) for key in fieldnames):
                    raise FileExistsError(
                        f"existing R5-C teaching audit manifest differs; preserving: {path}"
                    )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _save_pair_plan_once(path: Path, plan: np.ndarray, scene_ids: np.ndarray, seed_base: int) -> None:
    if path.exists():
        with np.load(path, allow_pickle=False) as archive:
            if (
                int(archive["seed_base"]) != int(seed_base)
                or not np.array_equal(archive["plan"], plan)
                or not np.array_equal(archive["scene_ids"].astype(str), scene_ids.astype(str))
            ):
                raise FileExistsError(f"existing R5-C teaching pair plan differs; preserving: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            plan=np.asarray(plan, dtype=np.int64),
            scene_ids=np.asarray(scene_ids).astype("U96"),
            seed_base=np.asarray(seed_base, dtype=np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def run_prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    """Generate new R5-C train/validation pairs and audit metadata after G0."""
    preflight = _require_preflight_complete(config)
    head = preflight["git_sha"]
    if DATA_GENERATION_PATH.exists():
        record = _require_data_generation(config, head, require_iq=True)
        return record

    train_path = (ROOT / config["data"]["training_source"]).resolve()
    validation_path = (ROOT / config["data"]["validation_source"]).resolve()
    cache_root = train_path.parent
    try:
        validation_path.relative_to(cache_root)
        cache_root.relative_to(STAGE_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("R5-C teaching train/validation caches must remain under the stage cache root") from error
    if train_path.name != "train" or validation_path.name != "validation":
        raise RuntimeError("R5-C teaching cache leaf names must be train and validation")

    physics_config = _read_yaml(PHYSICS_CONFIG_PATH)
    e003 = _read_yaml(E003_CONFIG_PATH)
    course_levels = e003["scenarios"]
    started = time.perf_counter()
    train_arrays = materialize_paired_split(
        cache_root,
        "train",
        physics_config,
        course_levels,
        config["data"]["training_cell_counts"],
        config["data"]["training_seed_bases_by_course"],
        assignment_seed=int(config["data"]["training_assignment_seed"]),
        chunk_size=int(config["data"]["chunk_size"]),
        chunk_seed_stride=int(config["data"]["chunk_seed_stride"]),
        include_k2=False,
    )
    validation_arrays = materialize_paired_split(
        cache_root,
        "validation",
        physics_config,
        course_levels,
        config["data"]["validation_cell_counts"],
        config["data"]["validation_seed_bases_by_course"],
        assignment_seed=int(config["data"]["validation_assignment_seed"]),
        chunk_size=int(config["data"]["chunk_size"]),
        chunk_seed_stride=int(config["data"]["chunk_seed_stride"]),
        include_k2=False,
    )
    train_ids = np.asarray(train_arrays["base_scene_id"]).reshape(-1).astype(str)
    validation_ids = np.asarray(validation_arrays["base_scene_id"]).reshape(-1).astype(str)
    old_ids = _old_r5a_scene_ids()
    _assert_fresh_scene_ids(
        train_ids,
        validation_ids,
        old_ids,
        train_count=int(config["data"]["training_pairs"]),
        validation_count=int(config["data"]["validation_pairs"]),
    )

    plan = build_r5_repeated_pair_plan(
        train_ids,
        updates=int(config["training"]["updates"]),
        pairs_per_update=int(config["training"]["paired_scenes_per_update"]),
        seed_base=int(config["data"]["pair_plan_seed_base"]),
    )
    plan = _validate_pair_plan(plan, train_ids.size)
    pair_plan_path = ROOT / config["data"]["pair_plan_source"]
    _save_pair_plan_once(
        pair_plan_path,
        plan,
        train_ids,
        int(config["data"]["pair_plan_seed_base"]),
    )

    scene_rows: list[dict[str, Any]] = []
    for split_name, arrays in (("train", train_arrays), ("validation", validation_arrays)):
        count = len(arrays["base_scene_id"])
        for index in range(count):
            scene_rows.append(
                {
                    "split": split_name,
                    "row_index": index,
                    "base_scene_id": str(arrays["base_scene_id"][index]),
                    "course_level": str(arrays["course_level"][index]),
                    "scenario_id": str(arrays["scenario_id"][index]),
                    "snr_db": float(arrays["snr_db"][index]),
                    "requested_cnr_db": float(arrays["requested_cnr_db"][index]),
                    "truth_range_cell": float(arrays["truth_range_cell"][index]),
                    "truth_doppler_hz": float(arrays["truth_doppler_hz"][index]),
                    "truth_amplitude": float(arrays["truth_amplitude"][index]),
                }
            )
    scene_fields = list(scene_rows[0])
    _write_csv_once(SCENE_MANIFEST_PATH, scene_fields, scene_rows)

    seed_specs = build_all_seed_specs(config)
    seed_fields = sorted({key for spec in seed_specs for key in spec})
    seed_rows = [
        {
            key: json.dumps(spec[key], ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list, tuple))
            else value
            for key, value in ((key, spec.get(key, "")) for key in seed_fields)
        }
        for spec in seed_specs
    ]
    _write_csv_once(SEED_MANIFEST_PATH, seed_fields, seed_rows)
    cache_records = _cache_file_records(cache_root)
    cache_manifests = {
        split: json.loads((cache_root / split / "manifest.json").read_text(encoding="utf-8"))
        for split in ("train", "validation")
    }
    result = {
        "experiment": "D2-A-R5-C-teaching",
        "git_sha": head,
        "config": config,
        "source_e003_config": str(E003_CONFIG_PATH.relative_to(ROOT)),
        "source_physics_config": str(PHYSICS_CONFIG_PATH.relative_to(ROOT)),
        "train_scene_count": int(train_ids.size),
        "validation_scene_count": int(validation_ids.size),
        "train_validation_scene_ids_disjoint": True,
        "historical_r5a_train_validation_ids_checked": len(old_ids),
        "historical_r5a_overlap_count": 0,
        "train_scene_ids_unique": True,
        "validation_scene_ids_unique": True,
        "training_pair_plan_seed_base": int(config["data"]["pair_plan_seed_base"]),
        "training_pair_plan_shape": list(plan.shape),
        "training_pair_plan_full_passes": 8,
        "training_pair_plan_reuses_exact_scene_ids": True,
        "scene_manifest_rows": len(scene_rows),
        "seed_manifest_rows": len(seed_rows),
        "cache_files": cache_records,
        "cache_bytes": int(sum(item["size_bytes"] for item in cache_records)),
        "cache_manifests": cache_manifests,
        "generation_seconds": float(time.perf_counter() - started),
        "generation_backend": "CPU simulation memmaps; no IQ cache on GPU",
        "final_blind_generated": False,
    }
    _write_json(DATA_GENERATION_PATH, result)
    return result


def run_cleanup_data(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only the R5-C train/validation IQ arrays after analysis is complete."""
    head = _require_committed_main()
    _require_committed_config(config)
    if not (OUTPUT_ROOT / "training.json").is_file():
        raise RuntimeError("R5-C teaching data cleanup requires completed T0–T3 training")
    if not (OUTPUT_ROOT / "metrics.json").is_file() or not (OUTPUT_ROOT / "analysis.md").is_file():
        raise RuntimeError("R5-C teaching data cleanup requires completed evaluation and analysis")
    if not STAGE_REVIEW_PATH.is_file():
        raise RuntimeError("R5-C teaching data cleanup requires a committed stage review record")
    record = _require_data_generation(config, head, require_iq=False)
    expected_paths = {
        str((ROOT / config["data"]["training_source"] / name).relative_to(ROOT))
        for name in ("iq_h0.npy", "iq_h1.npy")
    } | {
        str((ROOT / config["data"]["validation_source"] / name).relative_to(ROOT))
        for name in ("iq_h0.npy", "iq_h1.npy")
    }
    recorded = {item["path"]: item for item in record["cache_files"]}
    if expected_paths - set(recorded):
        raise RuntimeError("R5-C teaching cache manifest does not authorize all four IQ files")
    for key in ("training_source", "validation_source"):
        directory = ROOT / config["data"][key]
        try:
            directory.resolve().relative_to(STAGE_ROOT.resolve())
        except ValueError as error:
            raise RuntimeError("R5-C teaching cleanup cache resolves outside its stage directory") from error
        if directory.resolve() != directory.absolute():
            raise RuntimeError("R5-C teaching cleanup refuses a symlinked cache path")
    deleted = []
    for relative in sorted(expected_paths):
        path = ROOT / relative
        if path.is_symlink() or path.resolve() != path.absolute():
            raise RuntimeError(f"refusing to remove a symlinked R5-C teaching cache path: {path}")
        if path.is_file():
            size = int(path.stat().st_size)
            if size != int(recorded[relative]["size_bytes"]):
                raise RuntimeError(
                    f"R5-C teaching cache file changed after generation; preserving: {path}"
                )
            path.unlink()
            deleted.append({"path": relative, "size_bytes": size})
    result = {
        "git_sha": head,
        "deleted_stage_owned_iq_files": deleted,
        "preserved_cache_manifests_and_truth_metadata": True,
        "preserved_untracked_user_files": True,
        "project_size_bytes_after_iq_cleanup": _project_size_bytes(),
        "project_size_final_limit_bytes": int(config["resources"]["project_size_final_limit_gib"] * 1024**3),
    }
    _write_json(OUTPUT_ROOT / "cleanup.json", result)
    result["project_size_bytes_after_iq_cleanup"] = _project_size_bytes()
    _write_json(OUTPUT_ROOT / "cleanup.json", result)
    if result["project_size_bytes_after_iq_cleanup"] > result["project_size_final_limit_bytes"]:
        raise RuntimeError(
            "R5-C stage IQ cleanup completed, but the project still exceeds the frozen final size limit; "
            f"current bytes={result['project_size_bytes_after_iq_cleanup']}"
        )
    return result


def _require_preflight_complete(
    config: dict[str, Any], *, require_g1: bool = False
) -> dict[str, Any]:
    head = _require_committed_main()
    _require_committed_config(config)
    path = OUTPUT_ROOT / "preflight.json"
    if not path.is_file():
        raise RuntimeError("R5-C teaching preflight must pass before overfit or training")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("git_sha") != head
        or record.get("passed") is not True
        or record.get("config") != config
    ):
        raise RuntimeError("R5-C teaching preflight is missing, failed, or from another commit")
    if require_g1:
        _require_training_preflight(config, require_g1=True)
    return record


def _load_training_inputs(config: dict[str, Any]):
    _require_training_authorization(config)
    head = _require_committed_main()
    _require_data_generation(config, head, require_iq=True)
    train_arrays = _load_stage_cache(config, "train")
    validation_arrays = _load_stage_cache(config, "validation")
    train_ids = np.asarray(train_arrays["base_scene_id"]).reshape(-1).astype(str)
    validation_ids = np.asarray(validation_arrays["base_scene_id"]).reshape(-1).astype(str)
    if set(train_ids).intersection(validation_ids):
        raise RuntimeError("R5-C teaching train and validation scene IDs overlap")
    with np.load(ROOT / config["data"]["pair_plan_source"], allow_pickle=False) as archive:
        pair_plan = archive["plan"].copy()
        pair_scene_ids = archive["scene_ids"].astype(str)
        seed_base = int(archive["seed_base"])
    if seed_base != int(config["data"]["pair_plan_seed_base"]):
        raise RuntimeError("R5-C teaching training pair-plan seed does not match config")
    if not np.array_equal(pair_scene_ids, train_ids):
        raise RuntimeError("R5-C teaching training pair-plan IDs differ from the fresh R5-C cache")
    pair_plan = _validate_pair_plan(pair_plan, train_ids.size)
    expected_plan = build_r5_repeated_pair_plan(
        train_ids,
        updates=int(config["training"]["updates"]),
        pairs_per_update=int(config["training"]["paired_scenes_per_update"]),
        seed_base=seed_base,
    )
    if not np.array_equal(pair_plan, expected_plan):
        raise RuntimeError("R5-C teaching pair plan differs from its frozen seed schedule")
    return train_arrays, validation_arrays, pair_plan, _read_yaml(PHYSICS_CONFIG_PATH)


def run_overfit(config: dict[str, Any]) -> dict[str, Any]:
    _require_preflight_complete(config)
    train_arrays, _, _, physics_config = _load_training_inputs(config)
    initial_state = build_initial_state(int(config["training"]["initialization_seed"]))
    initial_microbatch = int(config["training"]["initial_microbatch_size"])
    microbatches = [
        initial_microbatch,
        *[
            int(value)
            for value in config["training"]["oom_microbatch_fallbacks"]
            if int(value) < initial_microbatch
        ],
    ]
    attempts: list[int] = []
    for microbatch in microbatches:
        attempts.append(microbatch)
        try:
            result = run_overfit_sanity(
                initial_state=initial_state,
                train_arrays=train_arrays,
                config=config,
                device="cuda:0",
                physics_config=physics_config,
                microbatch=microbatch,
            )
            result["microbatch_attempts"] = attempts
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if microbatch == microbatches[-1]:
                raise
    result["git_sha"] = _require_committed_main()
    result["config"] = config
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "g1_overfit_sanity.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def run_training(config: dict[str, Any]) -> list[dict[str, Any]]:
    _require_preflight_complete(config, require_g1=True)
    sanity_path = OUTPUT_ROOT / "g1_overfit_sanity.json"
    if not sanity_path.is_file() or json.loads(sanity_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("R5-C teaching G1 overfit gate must pass all arms before training")
    train_arrays, validation_arrays, pair_plan, physics_config = _load_training_inputs(config)
    initial_state = build_initial_state(int(config["training"]["initialization_seed"]))
    results = train_arms_sequentially(
        initial_state=initial_state,
        train_arrays=train_arrays,
        validation_arrays=validation_arrays,
        pair_plan=pair_plan,
        config=config,
        device="cuda:0",
        microbatch=int(config["training"]["initial_microbatch_size"]),
        arm_root=OUTPUT_ROOT / "runs" / f"seed_{config['training']['initialization_seed']}",
        physics_config=physics_config,
    )
    (OUTPUT_ROOT / "training.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results


def _load_evaluation_models(config: dict[str, Any], head: str) -> dict[str, torch.nn.Module]:
    training_path = OUTPUT_ROOT / "training.json"
    if not training_path.is_file():
        raise RuntimeError("R5-C teaching evaluation requires completed four-arm training.json")
    records = json.loads(training_path.read_text(encoding="utf-8"))
    if (
        not isinstance(records, list)
        or len(records) != len(config["training"]["arms"])
        or {item.get("arm") for item in records} != set(config["training"]["arms"])
    ):
        raise RuntimeError("R5-C teaching training record does not contain exactly four arms")

    models: dict[str, torch.nn.Module] = {}
    for record in records:
        arm = str(record["arm"])
        if int(record.get("updates_completed", -1)) != int(config["training"]["updates"]):
            raise RuntimeError(f"R5-C teaching arm {arm} did not complete the frozen update budget")
        checkpoint_path = Path(record["best_checkpoint_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"R5-C teaching arm {arm} checkpoint is missing: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("experiment") != "D2-A-R5"
            or payload.get("subphase") != "R5-C-teaching"
            or payload.get("arm") != arm
            or payload.get("source_git_sha") != head
            or payload.get("config") != config
            or int(payload.get("global_update", -1)) != int(record.get("best_update", -2))
        ):
            raise RuntimeError(f"R5-C teaching arm {arm} checkpoint provenance is inconsistent")
        training_model = TaskStudent(
            width=int(config["model"]["width"]),
            spectrum_frequency_bins=int(config["model"]["spectrum_frequency_bins"]),
        )
        training_model.load_state_dict(payload["model_state_dict"], strict=True)
        inference_model = export_inference_model(training_model).eval().cpu()
        parameter_count = sum(parameter.numel() for parameter in inference_model.parameters())
        if parameter_count != int(config["model"]["inference_parameter_count_expected"]):
            raise RuntimeError(f"R5-C teaching arm {arm} inference parameter count changed")
        models[arm] = inference_model
        del training_model, payload
    return models


def _append_arrays(target: dict[str, list[np.ndarray]], values: dict[str, Any]) -> None:
    for name, value in values.items():
        target.setdefault(name, []).append(np.asarray(value))


def _join_arrays(source: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    if not source:
        raise RuntimeError("R5-C teaching evaluation produced an empty split")
    result = {name: np.concatenate(parts, axis=0) for name, parts in source.items()}
    lengths = {values.shape[0] for values in result.values()}
    if len(lengths) != 1:
        raise RuntimeError("R5-C teaching streamed split arrays have inconsistent row counts")
    return result


def _write_prediction_csv(path: Path, rows: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows)
    count = len(rows[fieldnames[0]])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    name: (
                        rows[name][index].item()
                        if isinstance(rows[name][index], np.generic)
                        else rows[name][index]
                    )
                    for name in fieldnames
                }
            )
    temporary.replace(path)


def _save_arm_predictions(arm: str, payload: dict[str, Any]) -> None:
    directory = OUTPUT_ROOT / "predictions" / arm
    if directory.exists():
        raise FileExistsError(
            f"R5-C teaching prediction directory already exists; preserve it and inspect before rerun: {directory}"
        )
    _write_prediction_csv(directory / "h0_calibration.csv", payload["calibration"])
    _write_prediction_csv(directory / "h0_verification.csv", payload["verification"])
    _write_prediction_csv(directory / "c3_h0_verification.csv", payload["c3_h0_verification"])
    _write_prediction_csv(directory / "h1_development.csv", payload["development"])


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Stream independent H0/H1 splits and persist per-scene predictions only."""
    _require_training_authorization(config)
    preflight = _require_preflight_complete(config, require_g1=True)
    head = _require_committed_main()
    _require_data_generation(config, head, require_iq=False)
    training_validation_ids = _training_validation_scene_ids(config)
    if (OUTPUT_ROOT / "metrics.json").exists() or (OUTPUT_ROOT / "split_manifest.json").exists():
        raise FileExistsError(
            "R5-C teaching evaluation records already exist; preserve and inspect before rerun"
        )
    for arm in config["training"]["arms"]:
        if (OUTPUT_ROOT / "predictions" / arm).exists():
            raise FileExistsError(
                "R5-C teaching prediction output already exists; preserve and inspect: "
                f"{OUTPUT_ROOT / 'predictions' / arm}"
            )
    models = _load_evaluation_models(config, head)
    device = _require_training_cuda("cuda:0")
    physics_config = _read_yaml(PHYSICS_CONFIG_PATH)
    split_names = (
        "h0_calibration",
        "h0_verification",
        "c3_h0_verification",
        "h1_development",
    )
    per_arm: dict[str, dict[str, dict[str, Any]]] = {
        arm: {
            split: {}
            for split in ("calibration", "verification", "c3_h0_verification", "development")
        }
        for arm in config["training"]["arms"]
    }
    split_scene_ids: dict[str, set[str]] = {split: set() for split in split_names}
    split_manifest: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    cuda_peak_allocated = 0
    cuda_peak_reserved = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for split_name in split_names:
        for chunk in iter_generated_split_chunks(config, physics_config, split_name):
            batch = chunk["batch"]
            metadata = chunk["metadata"]
            split_manifest[split_name].append(chunk["spec"])
            ids = metadata["base_scene_ids"].astype(str)
            if split_scene_ids[split_name].intersection(ids.tolist()):
                raise RuntimeError(f"R5-C teaching split {split_name} generated duplicate scene IDs")
            split_scene_ids[split_name].update(ids.tolist())
            if split_name == "h1_development":
                teacher_quality = predict_k3_teacher_localization(
                    batch.iq_h1,
                    batch.truth_target_range_cell,
                    batch.truth_target_doppler_hz,
                    physics_config,
                    device,
                    prf_hz=float(config["radar"]["prf_hz"]),
                )
                torch.cuda.synchronize(device)
            else:
                teacher_quality = None
            for arm in config["training"]["arms"]:
                model = models[arm].to(device)
                try:
                    if split_name == "h1_development":
                        h1 = predict_task_student_chunk(
                            model,
                            batch.iq_h1,
                            device,
                            truth_range_cell=batch.truth_target_range_cell,
                            prf_hz=float(config["radar"]["prf_hz"]),
                        )
                    else:
                        h0 = predict_task_student_chunk(
                            model,
                            batch.iq_h0,
                            device,
                            prf_hz=float(config["radar"]["prf_hz"]),
                        )
                    torch.cuda.synchronize(device)
                finally:
                    models[arm] = model.cpu()
                cuda_peak_allocated = max(cuda_peak_allocated, int(torch.cuda.max_memory_allocated(device)))
                cuda_peak_reserved = max(cuda_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

                if split_name == "h0_calibration":
                    _append_arrays(
                        per_arm[arm]["calibration"],
                        {"base_scene_ids": ids, "h0_score": h0["score"]},
                    )
                elif split_name == "h0_verification":
                    _append_arrays(
                        per_arm[arm]["verification"],
                        {**metadata, "h0_score": h0["score"]},
                    )
                elif split_name == "c3_h0_verification":
                    _append_arrays(
                        per_arm[arm]["c3_h0_verification"],
                        {**metadata, "h0_score": h0["score"]},
                    )
                else:
                    _append_arrays(
                        per_arm[arm]["development"],
                        {
                            **metadata,
                            "h1_score": h1["score"],
                            "truth_range_cell": batch.truth_target_range_cell,
                            "predicted_range_cell": h1["predicted_range_cell"],
                            "predicted_range_bin": h1["predicted_range_bin"],
                            "truth_doppler_hz": batch.truth_target_doppler_hz,
                            "predicted_doppler_hz": h1["predicted_doppler_hz"],
                            "true_range_gated_doppler_hz": h1[
                                "true_range_gated_doppler_hz"
                            ],
                            **teacher_quality,
                        },
                    )
                del model
                if split_name == "h1_development":
                    del h1
                else:
                    del h0

    expected_counts = {
        name: int(config["data"]["splits"][name]["scene_count"]) for name in split_names
    }
    if any(len(ids) != expected_counts[name] for name, ids in split_scene_ids.items()):
        raise RuntimeError("R5-C teaching generated split count differs from the preregistered config")
    if any(
        split_scene_ids[left].intersection(split_scene_ids[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    ):
        raise RuntimeError("R5-C teaching generated split IDs overlap")
    evaluation_ids = set().union(*split_scene_ids.values())
    if training_validation_ids.intersection(evaluation_ids):
        raise RuntimeError("R5-C teaching train/validation IDs overlap independent evaluation IDs")

    summarized: dict[str, dict[str, Any]] = {}
    paired_payloads: dict[str, dict[str, Any]] = {}
    for arm in config["training"]["arms"]:
        for split in per_arm[arm]:
            per_arm[arm][split] = _join_arrays(per_arm[arm][split])
        evaluation_payload = {
            "development": per_arm[arm]["development"],
        }
        verification_payload = {
            "overall": per_arm[arm]["verification"],
            "c3": per_arm[arm]["c3_h0_verification"],
        }
        summarized[arm] = summarize_pilot_arm(
            evaluation_payload,
            per_arm[arm]["calibration"],
            verification_payload,
            config,
        )
        paired_payload = {
            "calibration": per_arm[arm]["calibration"],
            "development": per_arm[arm]["development"],
            "c3_h0_verification": per_arm[arm]["c3_h0_verification"],
        }
        paired_payloads[arm] = paired_payload
        _save_arm_predictions(arm, per_arm[arm])

    contrasts = paired_teaching_contrasts(paired_payloads, config)
    elapsed = time.perf_counter() - started
    metrics = {
        "experiment": "D2-A-R5-C-teaching",
        "protocol_version": config["experiment"]["protocol_version"],
        "git_sha": head,
        "config": config,
        "preflight_git_sha": preflight["git_sha"],
        "per_arm": summarized,
        "paired_contrasts": contrasts,
        "evaluation_seconds": float(elapsed),
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_peak_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_reserved_bytes": cuda_peak_reserved,
        "split_scene_counts": {name: len(ids) for name, ids in split_scene_ids.items()},
        "full_iq_saved": False,
        "teacher_maps_saved": False,
        "final_blind": False,
    }
    manifest_path = OUTPUT_ROOT / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics_path = OUTPUT_ROOT / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_analysis(config: dict[str, Any]) -> dict[str, Any]:
    """Write a concise report from the frozen metrics without regenerating data."""
    _require_training_authorization(config)
    head = _require_committed_main()
    metrics_path = OUTPUT_ROOT / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("R5-C teaching analysis requires metrics.json from the evaluation phase")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("git_sha") != head or metrics.get("config") != config:
        raise RuntimeError("R5-C teaching metrics do not match the pushed code/config")
    rows = []
    c3_cell_rows = []
    teacher_strata_rows = []
    for arm in config["training"]["arms"]:
        result = metrics["per_arm"][arm]
        dev = result["development"]["overall"]
        identifiable = result["development"]["identifiable_C2_C3"]
        overall_h0 = result["h0_verification"]["overall"]["overall"]
        c3_h0 = result["h0_verification"]["c3"]["overall"]
        rows.append(
            f"| {arm} | {overall_h0['pfa_ge']:.4f} | "
            f"{overall_h0['pfa_ci95_exact_upper']:.4f} | {c3_h0['pfa_ge']:.4f} | "
            f"{c3_h0['pfa_ci95_exact_upper']:.4f} | {dev['pd']:.4f} | {dev['auc']:.4f} | "
            f"{identifiable['joint_success']:.4f} | "
            f"{identifiable['range_rmse_all_h1_cells']:.3f} | "
            f"{identifiable['doppler_rmse_all_h1_hz']:.3f} |"
        )
        h0_c3_cells = result["h0_verification"]["c3"]["c3_cells"]
        for condition, cell in result["development"]["c3_cells"].items():
            h0_cell = h0_c3_cells.get(condition)
            h0_pfa = "n/a" if h0_cell is None else f"{h0_cell['pfa_ge']:.4f}"
            h0_upper = "n/a" if h0_cell is None else f"{h0_cell['pfa_ci95_exact_upper']:.4f}"
            c3_cell_rows.append(
                f"| {arm} | {condition} | {cell['pd']:.4f} | {cell['joint_success']:.4f} | "
                f"{cell['range_rmse_all_h1_cells']:.3f} | {cell['doppler_rmse_all_h1_hz']:.3f} | "
                f"{h0_pfa} | {h0_upper} |"
            )
        for subgroup, strata in result["development"]["k3_teacher_quality_and_student_strata"].items():
            group_n = strata["n_base_scenes"]
            teacher_rate = strata["k3_localization_rate"]
            correct = strata["student_metrics_by_teacher_correctness"]["teacher_correct"]
            incorrect = strata["student_metrics_by_teacher_correctness"]["teacher_incorrect"]
            def _stratum_value(row: dict[str, Any], key: str) -> str:
                return "n/a" if row["n_base_scenes"] == 0 else f"{row[key]:.4f}"
            teacher_strata_rows.append(
                f"| {arm} | {subgroup} | {group_n} | "
                f"{'n/a' if teacher_rate is None else f'{teacher_rate:.4f}'} | "
                f"{correct['n_base_scenes']} / {_stratum_value(correct, 'joint_success')} / "
                f"{_stratum_value(correct, 'pd')} / {_stratum_value(correct, 'doppler_rmse_all_h1_hz')} | "
                f"{incorrect['n_base_scenes']} / {_stratum_value(incorrect, 'joint_success')} / "
                f"{_stratum_value(incorrect, 'pd')} / {_stratum_value(incorrect, 'doppler_rmse_all_h1_hz')} |"
            )
    contrasts = metrics["paired_contrasts"]
    decision_rows = []
    for name in ("T1_minus_T0", "T2_minus_T0", "T3_minus_T0", "T2_minus_T3"):
        item = (
            contrasts["secondary"][name]
            if name in {"T3_minus_T0", "T2_minus_T3"}
            else contrasts["primary"][name]
        )
        gate = (
            "descriptive"
            if name in {"T3_minus_T0", "T2_minus_T3"}
            else str(item["passed"])
        )
        decision_rows.append(
            f"| {name} | {item['point_difference']:+.4f} | "
            f"[{item['nominal_95_ci'][0]:+.4f}, {item['nominal_95_ci'][1]:+.4f}] | "
            f"{gate} |"
        )
    passing = [
        arm
        for arm in ("T1", "T2")
        if contrasts["primary"][f"{arm}_minus_T0"]["passed"]
        and contrasts["task_protection"][arm]["passed"]
    ]
    recommendation = (
        f"仅 {', '.join(passing)} 同时通过本 pilot 的主收益与保护筛选；这仍需独立多种子确认。"
        if passing
        else "T1/T2 均未同时通过主收益和任务保护筛选；停止当前教师内容分支，不启动时机、可靠性或机制扩展。"
    )
    protection_rows = []
    for arm in ("T1", "T2"):
        gate = contrasts["task_protection"][arm]
        primary = contrasts["primary"][f"{arm}_minus_T0"]
        range_ratio = gate["C2_C3_range_rmse_ratio"]
        doppler_ratio = gate["C2_C3_doppler_rmse_ratio"]
        c3_delta = gate["C3_H0_pfa_difference"]
        protection_rows.append(
            f"| {arm} | {primary['point_difference']:+.4f} | "
            f"{primary['simultaneous_one_sided_97_5_lower_bound']:+.4f} | "
            f"{gate['overall_pd']['point_difference']:+.4f} | "
            f"{gate['overall_joint']['point_difference']:+.4f} | "
            f"{range_ratio['point_ratio']:.3f} [{range_ratio['nominal_95_ci'][0]:.3f}, {range_ratio['nominal_95_ci'][1]:.3f}] | "
            f"{doppler_ratio['point_ratio']:.3f} [{doppler_ratio['nominal_95_ci'][0]:.3f}, {doppler_ratio['nominal_95_ci'][1]:.3f}] | "
            f"{gate['C3_H0_pfa_absolute']['point_pfa']:.4f} | "
            f"{c3_delta['nominal_95_upper']:+.4f} | {gate['passed']} |"
        )
    analysis = "\n".join(
        [
            "# D2-A-R5-C Teaching Content Pilot 分析",
            "",
            f"Git SHA：`{head}`；总纲：`docs/RESEARCH_CHARTER.md` v1.3。",
            "",
            "本结果是单训练 seed 的新开发场景筛选，不是跨 seed 确认、最终盲测或检测器资格认证。",
            "AUC 采用独立 H0 verification 分数按冻结条件分层，并按 H1 development 条件样本数加权；C3 使用专用 H0 verification split。",
            "",
            "## 每臂指标",
            "",
            "| 臂 | overall H0 Pfa | overall exact 95% upper | C3 H0 Pfa | C3 exact 95% upper | Pd | AUC | C2+C3 joint | C2+C3 range RMSE | C2+C3 Doppler RMSE (Hz) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## C3 分组",
            "",
            "| 臂 | 条件 | Pd | Joint | Range RMSE | Doppler RMSE (Hz) | 独立 C3 H0 Pfa | exact 95% upper |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *c3_cell_rows,
            "",
            "## 配对对比",
            "",
            "| 对比 | C2+C3 joint 差 | nominal 95% CI | 主门 |",
            "|---|---:|---:|---|",
            *decision_rows,
            "",
            "## K3 教师正确/错误分层（离线评价）",
            "",
            "教师正确性定义为 K3 峰值同时满足距离误差 ≤1 cell、周期 Doppler 误差 ≤20 Hz。该标签只用于本表离线分层，不进入训练、推理或模型选择。",
            "",
            "| 臂 | 子组 | N | K3 定位正确率 | 教师正确：N / joint / Pd / Doppler RMSE Hz | 教师错误：N / joint / Pd / Doppler RMSE Hz |",
            "|---|---|---:|---:|---|---|",
            *teacher_strata_rows,
            "",
            "## T1/T2 任务保护",
            "",
            "| 臂 | C2+C3 joint 差 | simultaneous 97.5% lower | overall Pd 差 | overall joint 差 | range RMSE ratio [95% CI] | Doppler RMSE ratio [95% CI] | C3 H0 Pfa | C3 Pfa 差 95% upper | 全部保护 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *protection_rows,
            "",
            "## 阶段筛选",
            "",
            recommendation,
            "",
            "T3 是 truth-assisted 对照，不是教师输出；T2−T3 是描述性对照，不设晋级门。T1/T2 的辅助响应由 IQ-only K3 计算；所有臂保留物理真值任务监督。场景 bootstrap 仅描述场景抽样，不代表训练 seed 方差。",
            "",
        ]
    )
    path = OUTPUT_ROOT / "analysis.md"
    if path.exists():
        raise FileExistsError(f"R5-C teaching analysis already exists; preserve and inspect: {path}")
    path.write_text(analysis, encoding="utf-8")
    return {"git_sha": head, "analysis_path": str(path), "passing_teacher_arms": passing}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=("preflight", "prepare-data", "overfit", "train", "evaluate", "analyze", "cleanup-data"),
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = _read_yaml(args.config)
    validate_r5c_teaching_config(config)

    if args.phase == "preflight":
        result = run_preflight(config)
    elif args.phase == "prepare-data":
        result = run_prepare_data(config)
    elif args.phase == "overfit":
        result = run_overfit(config)
    elif args.phase == "train":
        result = run_training(config)
    elif args.phase == "evaluate":
        result = run_evaluation(config)
    elif args.phase == "analyze":
        result = run_analysis(config)
    else:
        result = run_cleanup_data(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
