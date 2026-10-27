#!/usr/bin/env python3
"""Staged CUDA-only execution for the D2-A-R5 truth-supervised pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_d2a_e003 as e003_runner
from radardistill.distillation.task_losses import decode_task_outputs, task_loss
from radardistill.evaluation.d2a_e002 import calibrate_student_threshold
from radardistill.students.physics_guided_student import PhysicsGuidedTaskStudent
from radardistill.students.task_student import TaskStudent
from radardistill.training.d2a_e002 import (
    _cpu_state_dict,
    _paired_validation,
    make_paired_examples,
    scale_d2a_e002_microbatch_loss,
)
from radardistill.training.d2a_e003 import e003_seed_manifest, validate_e003_config
from radardistill.training.d2a_e003_data import (
    materialize_e003_h0_split,
    materialize_e003_paired_split,
    validate_e003_split_disjointness,
)
from radardistill.training.d2a_e002_data import materialize_identifiability_boundary
from radardistill.training.d2a_r5 import (
    R5CosineSchedule,
    build_r5_repeated_pair_plan,
    should_extend_r5_training,
    validate_r5a_config,
)


R5_CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-A/config.yaml"
E003_CONFIG_PATH = ROOT / "experiments/D2-A-E003/config.yaml"
PHYSICS_CONFIG_PATH = ROOT / "experiments/D2-A-E001/config.yaml"
ARTIFACTS = ROOT / "experiments/D2-A-R5/R5-A"
FROZEN_INPUTS = (
    "AGENTS.md",
    "docs/RESEARCH_CHARTER.md",
    "docs/CURRENT_STATUS.md",
    "docs/RESEARCH_ROADMAP.md",
    "docs/research/ASSUMPTION_REGISTER.md",
    "experiments/D2-A-R5/R5-A/protocol.md",
    "experiments/D2-A-R5/R5-A/config.yaml",
    "experiments/D2-A-E003/protocol.md",
    "experiments/D2-A-E003/config.yaml",
    "experiments/D2-A-E001/config.yaml",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/students/task_student.py",
    "src/radardistill/students/physical_coherence.py",
    "src/radardistill/students/physics_guided_student.py",
    "src/radardistill/distillation/task_losses.py",
    "src/radardistill/evaluation/d2a_e001.py",
    "src/radardistill/evaluation/d2a_e002.py",
    "src/radardistill/evaluation/physical_consistency.py",
    "src/radardistill/training/d2a_e002.py",
    "src/radardistill/training/d2a_e002_data.py",
    "src/radardistill/training/d2a_e003.py",
    "src/radardistill/training/d2a_e003_data.py",
    "src/radardistill/training/d2a_r5.py",
    "scripts/run_d2a_e003.py",
    "scripts/run_d2a_r5.py",
    "tests/test_d2a_r5_training.py",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    r5 = _read_yaml(R5_CONFIG_PATH)
    e003 = _read_yaml(E003_CONFIG_PATH)
    physics = _read_yaml(PHYSICS_CONFIG_PATH)
    validate_r5a_config(r5)
    validate_e003_config(e003)
    return r5, e003, physics


def _require_frozen_main() -> str:
    branch = _git("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"D2-A-R5 requires main, found {branch!r}")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *FROZEN_INPUTS], cwd=ROOT).returncode != 0:
        raise RuntimeError("R5-A protocol, source or frozen input has tracked uncommitted changes")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError("R5-A requires the committed code/protocol to be pushed to origin/main")
    return head


def _nvidia_smi() -> dict[str, Any]:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    full = subprocess.run(["nvidia-smi"], text=True, capture_output=True) if query.returncode == 0 else None
    return {
        "returncode": int(query.returncode),
        "query": query.stdout.strip(),
        "stderr": query.stderr.strip(),
        "full": full.stdout.strip() if full is not None else None,
    }


def _environment_record(commit: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(0)
    return {
        "git_sha": commit,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_memory_free_bytes": int(free),
        "cuda_memory_total_bytes": int(total),
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "nvidia_smi": _nvidia_smi(),
    }


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R5-A requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    expected = "NVIDIA GeForce RTX 3050 Laptop GPU"
    actual = torch.cuda.get_device_name(0)
    if actual != expected:
        raise RuntimeError(f"R5-A requires {expected}, found {actual}")
    return torch.device("cuda:0")


def _estimated_cache_bytes(e003: Mapping[str, Any]) -> int:
    data = e003["data"]
    paired_count = sum(
        int(data[name]["paired_base_scenes"])
        for name in ("train", "validation", "development", "exact_overlap_boundary")
    )
    paired_bytes = paired_count * 2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
    h0_count = sum(
        int(data[name]["scenes"])
        for name in ("h0_calibration", "h0_verification", "c3_h0_verification")
    )
    h0_bytes = h0_count * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
    return int((paired_bytes + h0_bytes) * 1.15 + 2 * 1024**3)


def run_preflight() -> dict[str, Any]:
    _, e003, _ = _load_configs()
    commit = _require_frozen_main()
    device = _require_cuda()
    estimated = _estimated_cache_bytes(e003)
    free = shutil.disk_usage(ROOT).free
    reserve = int(5 * 1024**3)
    required = estimated + reserve
    if free < required:
        raise RuntimeError(
            f"insufficient free disk for R5 cache: need {required / 1024**3:.2f} GiB, "
            f"have {free / 1024**3:.2f} GiB"
        )
    if ARTIFACTS.joinpath("preflight.json").is_file():
        prior = json.loads(ARTIFACTS.joinpath("preflight.json").read_text(encoding="utf-8"))
        if prior.get("git_sha") != commit:
            raise RuntimeError("R5-A preflight belongs to a different source commit; preserve and review it")
    record = {
        "git_sha": commit,
        "passed": True,
        "training_device": str(device),
        "environment": _environment_record(commit),
        "estimated_cache_bytes": estimated,
        "required_free_disk_bytes_including_reserve": required,
        "disk_free_bytes_before_generation": free,
        "cuda_required": True,
        "cpu_fallback": "forbidden",
        "final_blind_generated": False,
    }
    _write_json(ARTIFACTS / "preflight.json", record)
    return record


def _require_preflight() -> dict[str, Any]:
    path = ARTIFACTS / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run R5-A preflight before generating data or training")
    record = json.loads(path.read_text(encoding="utf-8"))
    commit = _require_frozen_main()
    if record.get("git_sha") != commit or record.get("passed") is not True:
        raise RuntimeError("R5-A preflight is missing, stale or failed")
    _require_cuda()
    return record


def _require_posttraining_provenance(*, require_cuda: bool) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Allow a pushed evaluation-code fix while preserving the training source SHA."""
    evaluation_sha = _require_frozen_main()
    preflight_path = ARTIFACTS / "preflight.json"
    training_path = ARTIFACTS / "training.json"
    if not preflight_path.is_file() or not training_path.is_file():
        raise RuntimeError("R5-A training preflight and completed training record are required")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if (
        preflight.get("passed") is not True
        or preflight.get("git_sha") != training.get("git_sha")
        or training.get("protocol_version") != 1.1
        or training.get("final_blind_generated") is not False
        or training.get("distillation_run") is not False
        or any(int(training.get("arms", {}).get(arm, {}).get("updates_completed", -1)) < 6400 for arm in ("S0", "S2"))
    ):
        raise RuntimeError("R5-A post-training provenance is incomplete or inconsistent")
    if require_cuda:
        _require_cuda()
    return preflight, training, evaluation_sha


def _cache_file_manifest(cache_root: Path) -> list[dict[str, Any]]:
    return [
        {"path": path.relative_to(cache_root).as_posix(), "size_bytes": int(path.stat().st_size)}
        for path in sorted(cache_root.rglob("*"))
        if path.is_file()
    ]


def _write_scene_manifest(splits: Mapping[str, Mapping[str, np.ndarray]]) -> int:
    path = ARTIFACTS / "scene_manifest.csv"
    columns = ("split", "base_scene_id", "course_level", "scenario_id", "snr_db", "requested_cnr_db")
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for split, arrays in splits.items():
            count = len(arrays["base_scene_id"])
            for index in range(count):
                writer.writerow({
                    "split": split,
                    "base_scene_id": str(arrays["base_scene_id"][index]),
                    "course_level": str(arrays.get("course_level", np.full(count, ""))[index]),
                    "scenario_id": str(arrays.get("scenario_id", np.full(count, ""))[index]),
                    "snr_db": float(arrays.get("snr_db", np.full(count, np.nan))[index]),
                    "requested_cnr_db": float(arrays.get("requested_cnr_db", np.full(count, np.nan))[index]),
                })
                rows += 1
    return rows


def run_generate_data() -> dict[str, Any]:
    preflight = _require_preflight()
    r5, e003, physics = _load_configs()
    cache_root = ROOT / r5["data"]["cache_root"]
    cache_root.mkdir(parents=True, exist_ok=True)
    top_manifest_path = ARTIFACTS / "cache_manifest.json"
    if top_manifest_path.is_file():
        manifest = json.loads(top_manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            file_path = cache_root / item["path"]
            if not file_path.is_file() or file_path.stat().st_size != int(item["size_bytes"]):
                raise FileNotFoundError(f"R5-A cache differs from its manifest; preserving: {file_path}")
        if len(_cache_file_manifest(cache_root)) != len(manifest["files"]):
            raise RuntimeError("R5-A cache contains files outside its completed manifest; preserving all files")
        return json.loads((ARTIFACTS / "data_generation.json").read_text(encoding="utf-8"))

    started = time.perf_counter()
    splits: dict[str, dict[str, np.ndarray]] = {}
    print("[R5-A] materializing the frozen E003 training cohort", flush=True)
    splits["train"] = materialize_e003_paired_split(cache_root, "train", e003, physics)
    print("[R5-A] materializing the frozen E003 validation cohort", flush=True)
    splits["validation"] = materialize_e003_paired_split(cache_root, "validation", e003, physics)
    for name in ("h0_calibration", "h0_verification", "c3_h0_verification"):
        section = e003["data"][name]
        print(f"[R5-A] materializing {name}: {int(section['scenes']):,} H0 scenes", flush=True)
        splits[name] = materialize_e003_h0_split(
            cache_root,
            name,
            count=int(section["scenes"]),
            course_levels=e003["scenarios"],
            seed_base=int(section["seed_base"]),
            assignment_seed=int(section.get("mixture_assignment_seed", section.get("assignment_seed"))),
            physics_config=physics,
            chunk_size=int(section["generation_chunk_size"]),
            seed_stride=int(e003["data"]["scene_seed_stride"]),
            dedicated_c3=name == "c3_h0_verification",
        )
    print("[R5-A] materializing the frozen E003 development cohort", flush=True)
    splits["development"] = materialize_e003_paired_split(cache_root, "development", e003, physics)
    boundary = e003["data"]["exact_overlap_boundary"]
    print("[R5-A] materializing the exact-overlap descriptive boundary", flush=True)
    splits["exact_overlap_boundary"] = materialize_identifiability_boundary(
        cache_root,
        "exact_overlap_boundary",
        int(boundary["paired_base_scenes"]),
        physics,
        snr_db=float(boundary["snr_db"]),
        cnr_db=float(boundary["cnr_db"]),
        seed_base=int(boundary["seed_base"]),
        chunk_size=int(boundary["generation_chunk_size"]),
        chunk_seed_stride=int(e003["data"]["scene_seed_stride"]),
    )
    disjoint_counts = validate_e003_split_disjointness(splits)
    scene_rows = _write_scene_manifest(splits)
    files = _cache_file_manifest(cache_root)
    cache_bytes = int(sum(item["size_bytes"] for item in files))
    cache_manifest = {
        "experiment": "D2-A-R5/R5-A",
        "source_data_protocol": "D2-A-E003 v1.1",
        "cache_root": str(cache_root.relative_to(ROOT)),
        "files": files,
        "total_bytes": cache_bytes,
        "cleanup_eligible_files_only": True,
    }
    _write_json(top_manifest_path, cache_manifest)
    generation = {
        "git_sha": preflight["git_sha"],
        "source_e003_seed_manifest": e003_seed_manifest(e003),
        "split_counts": disjoint_counts,
        "all_base_scene_ids_disjoint": True,
        "scene_manifest_rows": scene_rows,
        "cache_file_count": len(files),
        "cache_bytes": cache_bytes,
        "generation_seconds": time.perf_counter() - started,
        "cache_root": str(cache_root.relative_to(ROOT)),
        "source_dataset_reused_by_ids": True,
        "final_blind_generated": False,
    }
    _write_json(ARTIFACTS / "data_generation.json", generation)
    return generation


def _load_split(cache_root: Path, name: str) -> dict[str, np.ndarray]:
    path = cache_root / name
    if (path / "e003_manifest.json").is_file():
        return e003_runner._load_cached_split(cache_root, name)
    if name == "exact_overlap_boundary" and (path / "manifest.json").is_file():
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("split") != name or manifest.get("scenario") != "exact_overlap":
            raise ValueError(f"R5-A exact-overlap cache identity differs from its manifest: {path}")
        names = ("iq_h0", "iq_h1", "base_scene_id", "truth_range_cell", "truth_doppler_hz")
        missing = [key for key in names if not (path / f"{key}.npy").is_file()]
        if missing:
            raise FileNotFoundError(f"R5-A exact-overlap cache is incomplete: {missing}")
        arrays = {key: np.load(path / f"{key}.npy", mmap_mode="r") for key in names}
        if any(len(value) != int(manifest.get("scenes", -1)) for value in arrays.values()):
            raise ValueError(f"R5-A exact-overlap cache scene count differs from its manifest: {path}")
        return arrays
    raise FileNotFoundError(f"R5-A cache split is missing or incomplete: {path}")


def _initial_student_state(seed: int, *, width: int = 32) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    model = TaskStudent(width=width, include_spectrum_head=True)
    state = _cpu_state_dict(model)
    del model
    return state


def _build_model(
    arm: str,
    initial_state: Mapping[str, torch.Tensor],
    *,
    width: int = 32,
    seed: int = 101,
) -> torch.nn.Module:
    if arm not in ("S0", "S2"):
        raise ValueError(f"unknown R5-A arm: {arm}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        if arm == "S0":
            model: torch.nn.Module = TaskStudent(width=width, include_spectrum_head=True)
            model.load_state_dict(initial_state, strict=True)
        else:
            model = PhysicsGuidedTaskStudent(width=width, include_spectrum_head=True)
            model.student.load_state_dict(initial_state, strict=True)
    return model


def _training_loss_config(e003_config: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in e003_config.items()}
    result["training"] = dict(e003_config["training"])
    result["training"]["loss"] = dict(e003_config["training"]["losses"])
    return result


def run_overfit_sanity() -> dict[str, Any]:
    preflight = _require_preflight()
    r5, e003, _ = _load_configs()
    cache_root = ROOT / r5["data"]["cache_root"]
    train = _load_split(cache_root, "train")
    sanity_config = r5["gates"]["G1_overfit"]
    pair_ids = np.arange(int(sanity_config["paired_scenes"]), dtype=np.int64)
    batch = make_paired_examples(train, pair_ids, include_teacher=False)
    x = torch.from_numpy(batch["iq_ri"])
    truth_cpu = {
        key: torch.from_numpy(batch[key])
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    loss_config = _training_loss_config(e003)
    device = _require_cuda()
    seed = int(r5["training"]["seed"])
    initial_state = _initial_student_state(seed)
    model_results: dict[str, Any] = {}
    for arm in ("S0", "S2"):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = _build_model(arm, initial_state, seed=seed).to(device)
        x_gpu = x.to(device)
        truth = {key: value.to(device) for key, value in truth_cpu.items()}
        model.eval()
        initial_output = model(x_gpu, compute_spectrum=False)
        initial_terms = task_loss(initial_output, truth, loss_config, teacher_weight=0.0)
        task_modules = {
            "range_cell": model.student.range_head if arm == "S2" else model.range_head,
            "doppler": model.student.doppler_head if arm == "S2" else model.doppler_head,
        }
        gradient_norms: dict[str, float] = {}
        for task_name, module in task_modules.items():
            task_gradients = torch.autograd.grad(
                initial_terms[task_name], tuple(module.parameters()), retain_graph=True, allow_unused=True
            )
            norms = [gradient.detach().double().square().sum() for gradient in task_gradients if gradient is not None]
            gradient_norms[task_name] = float(torch.sqrt(torch.stack(norms).sum()).item()) if norms else 0.0
        gradients_passed = all(np.isfinite(value) and value > 0.0 for value in gradient_norms.values())

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(e003["training"]["learning_rate_initial"]),
            weight_decay=float(e003["training"]["weight_decay"]),
        )
        initial_loss = float(initial_terms["task_total"].detach().item())
        updates = int(sanity_config["updates"])
        model.train()
        last_gradient_norm = float("nan")
        for _ in range(updates):
            optimizer.zero_grad(set_to_none=True)
            output = model(x_gpu, compute_spectrum=False)
            loss = task_loss(output, truth, loss_config, teacher_weight=0.0)["task_total"]
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite {arm} G1 overfit loss")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(e003["training"]["gradient_clip_norm"]))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite {arm} G1 overfit gradient")
            last_gradient_norm = float(grad_norm.detach().item())
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            final_output = model(x_gpu, compute_spectrum=False)
            final_terms = task_loss(final_output, truth, loss_config, teacher_weight=0.0)
            presence_correct = (final_output["presence_logit"] >= 0.0) == truth["truth_presence"].bool()
            h1_mask = truth["truth_presence"].bool()
            range_accuracy = (
                final_output["range_logits"][h1_mask].argmax(dim=-1)
                == torch.floor(truth["truth_range_cell"][h1_mask]).long()
            ).float().mean()
        final_loss = float(final_terms["task_total"].item())
        ratio = final_loss / max(initial_loss, np.finfo(np.float64).tiny)
        presence_accuracy = float(presence_correct.float().mean().item())
        range_cell_accuracy = float(range_accuracy.item())
        passed = bool(
            gradients_passed
            and np.isfinite(last_gradient_norm)
            and ratio <= 1.0 - float(sanity_config["required_train_truth_task_loss_relative_drop"])
            and presence_accuracy >= 1.0
            and range_cell_accuracy >= 0.95
        )
        model_results[arm] = {
            "updates": updates,
            "paired_scenes": int(len(pair_ids)),
            "initial_truth_task_loss": initial_loss,
            "final_truth_task_loss": final_loss,
            "loss_ratio_final_to_initial": ratio,
            "presence_accuracy_on_overfit_batch": presence_accuracy,
            "h1_range_cell_accuracy_on_overfit_batch": range_cell_accuracy,
            "task_gradient_norms": gradient_norms,
            "last_gradient_norm": last_gradient_norm,
            "passed": passed,
        }
        del optimizer, model, x_gpu, truth
        torch.cuda.empty_cache()

    result = {
        "git_sha": preflight["git_sha"],
        "purpose": "engineering_only; excluded from scientific metrics",
        "final_blind_generated": False,
        "arms": model_results,
        "passed": all(value["passed"] for value in model_results.values()),
    }
    _write_json(ARTIFACTS / "g1_overfit_sanity.json", result)
    return result


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all(),
    }


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda_all"])


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_history(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in ("update", "epoch"):
            if row.get(key):
                row[key] = int(row[key])
        for key, value in list(row.items()):
            if key not in ("arm", "status", "update", "epoch") and value not in (None, ""):
                try:
                    row[key] = float(value)
                except (TypeError, ValueError):
                    pass
    return rows


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = (
        "update", "epoch", "arm", "microbatch", "learning_rate", "train_task_loss",
        "train_presence_loss", "train_range_cell_loss", "train_range_offset_loss", "train_doppler_loss",
        "validation_task_loss", "validation_presence_loss", "validation_range_cell_loss",
        "validation_range_offset_loss", "validation_doppler_loss", "validation_auc", "validation_range_rmse_cells",
        "validation_doppler_rmse_hz", "validation_true_cell_doppler_rmse_hz",
        "validation_range_cell_mismatch_rate",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _make_train_row(
    update: int,
    arm: str,
    microbatch: int,
    learning_rate: float,
    train_terms: Mapping[str, float],
    validation: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "update": int(update),
        "epoch": int(math.ceil(update / 1600)) if update else 0,
        "arm": arm,
        "microbatch": int(microbatch),
        "learning_rate": float(learning_rate),
        "train_task_loss": float(train_terms.get("task_total", float("nan"))),
        "train_presence_loss": float(train_terms.get("presence", float("nan"))),
        "train_range_cell_loss": float(train_terms.get("range_cell", float("nan"))),
        "train_range_offset_loss": float(train_terms.get("range_offset", float("nan"))),
        "train_doppler_loss": float(train_terms.get("doppler", float("nan"))),
    }
    if validation is not None:
        row.update({
            "validation_task_loss": float(validation["task_loss"]),
            "validation_presence_loss": float(validation["presence_loss"]),
            "validation_range_cell_loss": float(validation["range_cell_loss"]),
            "validation_range_offset_loss": float(validation["range_offset_loss"]),
            "validation_doppler_loss": float(validation["doppler_loss"]),
            "validation_auc": float(validation["auc"]),
            "validation_range_rmse_cells": float(validation["range_rmse_all_h1_cells"]),
            "validation_doppler_rmse_hz": float(validation["doppler_rmse_all_h1_hz"]),
            "validation_true_cell_doppler_rmse_hz": float(validation["true_cell_doppler_rmse_hz"]),
            "validation_range_cell_mismatch_rate": float(validation["predicted_range_cell_mismatch_rate"]),
        })
    return row


def _save_resume_state(
    path: Path,
    *,
    arm: str,
    width: int,
    global_update: int,
    microbatch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    schedule: R5CosineSchedule,
    best_validation_loss: float,
    best_update: int,
    validation: Mapping[str, Any],
    data_plan_seed: int,
) -> None:
    _atomic_torch_save({
        "experiment": "D2-A-R5",
        "subphase": "R5-A",
        "protocol_version": 1.1,
        "source_git_sha": _git("rev-parse", "HEAD"),
        "r5_config": _read_yaml(R5_CONFIG_PATH),
        "arm": arm,
        "width": width,
        "global_update": int(global_update),
        "epoch": int(math.ceil(global_update / 1600)) if global_update else 0,
        "next_pair_plan_row": int(global_update),
        "microbatch_size": int(microbatch),
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": schedule.state_dict(),
        "rng_state": _capture_rng_state(),
        "best_validation_loss": float(best_validation_loss),
        "best_update": int(best_update),
        "latest_validation": dict(validation),
        "pair_order_seed_base": int(data_plan_seed),
    }, path)


def _load_resume_state(
    path: Path,
    *,
    arm: str,
    width: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = (payload.get("experiment"), payload.get("subphase"), payload.get("arm"), payload.get("width"))
    if identity != ("D2-A-R5", "R5-A", arm, width):
        raise ValueError(f"resume checkpoint identity mismatch: {path}")
    if (
        float(payload.get("protocol_version", -1.0)) != 1.1
        or payload.get("source_git_sha") != _git("rev-parse", "HEAD")
        or payload.get("r5_config") != _read_yaml(R5_CONFIG_PATH)
    ):
        raise ValueError(f"resume checkpoint source/config does not match the frozen run: {path}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    schedule = R5CosineSchedule.from_state_dict(payload["scheduler_state_dict"])
    _restore_rng_state(payload["rng_state"])
    return {**payload, "schedule": schedule}


def _train_attempt(
    *,
    arm: str,
    width: int,
    initial_state: Mapping[str, torch.Tensor],
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    plan: np.ndarray,
    e003_config: Mapping[str, Any],
    r5_config: Mapping[str, Any],
    device: torch.device,
    microbatch: int,
    target_updates: int,
    arm_root: Path,
) -> dict[str, Any]:
    training = r5_config["training"]
    effective_batch = int(training["effective_batch_size"])
    effective_h1 = int(training["paired_scenes_per_update"])
    if effective_batch % microbatch or microbatch % 2:
        raise ValueError("R5 microbatch must be an even divisor of effective batch 64")
    if plan.shape != (12800, effective_h1):
        raise ValueError("R5 pair plan must predefine all eight possible complete epochs")
    if not 0 < target_updates <= plan.shape[0]:
        raise ValueError("target updates must lie within the frozen R5 pair plan")
    loss_config = _training_loss_config(e003_config)
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = _build_model(arm, initial_state, width=width, seed=seed).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate_initial"]),
        weight_decay=float(e003_config["training"]["weight_decay"]),
    )
    schedule = R5CosineSchedule(
        initial_updates=int(training["max_updates_initial"]),
        final_updates=int(training["continuation_max_updates"]),
        initial_lr=float(training["learning_rate_initial"]),
        initial_final_lr=float(training["learning_rate_final"]),
        extension_final_lr=float(training["extension_learning_rate_final"]),
    )
    resume_path = arm_root / "latest_training_state.pt"
    start_update = 0
    best_loss = float("inf")
    best_update = 0
    if resume_path.is_file():
        state = _load_resume_state(resume_path, arm=arm, width=width, model=model, optimizer=optimizer)
        schedule = state["schedule"]
        start_update = int(state["global_update"])
        best_loss = float(state["best_validation_loss"])
        best_update = int(state["best_update"])
        if int(state["pair_order_seed_base"]) != int(training["pair_order_seed_base"]):
            raise ValueError("resume checkpoint pair-plan seed differs from frozen R5 config")
    history_path = arm_root / "training_history.csv"
    history = [row for row in _load_history(history_path) if int(row["update"]) <= start_update]
    if start_update == 0:
        model.eval()
        initial_validation = _paired_validation(
            model, validation_arrays, loss_config, device, microbatch
        )
        best_loss = float(initial_validation["task_loss"])
        best_update = 0
        best_path = arm_root / "best_checkpoint.pt"
        _atomic_torch_save({
            "experiment": "D2-A-R5", "subphase": "R5-A", "protocol_version": 1.1,
            "source_git_sha": _git("rev-parse", "HEAD"), "r5_config": _read_yaml(R5_CONFIG_PATH),
            "arm": arm, "width": width,
            "global_update": 0, "model_state_dict": _cpu_state_dict(model),
            "validation": initial_validation,
        }, best_path)
        history = [
            _make_train_row(0, arm, microbatch, float(training["learning_rate_initial"]), {}, initial_validation)
        ]
        _save_resume_state(
            resume_path,
            arm=arm,
            width=width,
            global_update=0,
            microbatch=microbatch,
            model=model,
            optimizer=optimizer,
            schedule=schedule,
            best_validation_loss=best_loss,
            best_update=best_update,
            validation=initial_validation,
            data_plan_seed=int(training["pair_order_seed_base"]),
        )
        model.train()
    _write_history(history_path, history)
    accumulations = effective_batch // microbatch
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    latest_validation: dict[str, Any] = {}

    for update_index in range(start_update, target_updates):
        update = update_index + 1
        learning_rate = schedule.lr_for_update(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        pair_ids = plan[update_index]
        batch = make_paired_examples(train_arrays, pair_ids, include_teacher=False)
        optimizer.zero_grad(set_to_none=True)
        train_terms = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "task_total")}
        for micro_start in range(0, effective_batch, microbatch):
            micro_stop = micro_start + microbatch
            x = torch.from_numpy(batch["iq_ri"][micro_start:micro_stop]).to(device)
            truth = {
                key: torch.from_numpy(batch[key][micro_start:micro_stop]).to(device)
                for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
            }
            outputs = model(x, compute_spectrum=False)
            losses = task_loss(outputs, truth, loss_config, teacher_weight=0.0)
            micro_h1 = int(truth["truth_presence"].sum().item())
            scaled = scale_d2a_e002_microbatch_loss(
                losses,
                loss_config,
                microbatch_size=micro_stop - micro_start,
                effective_batch_size=effective_batch,
                microbatch_h1=micro_h1,
                effective_h1=effective_h1,
                teacher_weight=0.0,
            )
            if not torch.isfinite(scaled):
                raise FloatingPointError(f"non-finite R5-A {arm} loss at update {update}")
            scaled.backward()
            sample_fraction = (micro_stop - micro_start) / effective_batch
            h1_fraction = micro_h1 / effective_h1
            train_terms["presence"] += float(losses["presence"].detach().item()) * sample_fraction
            for name in ("range_cell", "range_offset", "doppler"):
                train_terms[name] += float(losses[name].detach().item()) * h1_fraction
            train_terms["task_total"] += float(scaled.detach().item())
            del outputs, losses, scaled, truth, x
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(e003_config["training"]["gradient_clip_norm"]))
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite R5-A {arm} gradient at update {update}")
        optimizer.step()
        row = _make_train_row(update, arm, microbatch, learning_rate, train_terms)

        if update % int(training["validation_interval_updates"]) == 0:
            model.eval()
            validation = _paired_validation(model, validation_arrays, loss_config, device, microbatch)
            if not all(np.isfinite(float(value)) for value in validation.values()):
                raise FloatingPointError(f"non-finite R5-A {arm} validation metrics at update {update}")
            latest_validation = validation
            row.update(_make_train_row(update, arm, microbatch, learning_rate, train_terms, validation))
            if float(validation["task_loss"]) < best_loss:
                best_loss = float(validation["task_loss"])
                best_update = update
                _atomic_torch_save({
                    "experiment": "D2-A-R5", "subphase": "R5-A", "protocol_version": 1.1,
                    "source_git_sha": _git("rev-parse", "HEAD"), "r5_config": _read_yaml(R5_CONFIG_PATH),
                    "arm": arm, "width": width,
                    "global_update": update, "model_state_dict": _cpu_state_dict(model),
                    "validation": dict(validation),
                }, arm_root / "best_checkpoint.pt")
            history.append(row)
            _write_history(history_path, history)
            _save_resume_state(
                resume_path,
                arm=arm,
                width=width,
                global_update=update,
                microbatch=microbatch,
                model=model,
                optimizer=optimizer,
                schedule=schedule,
                best_validation_loss=best_loss,
                best_update=best_update,
                validation=validation,
                data_plan_seed=int(training["pair_order_seed_base"]),
            )
            model.train()
            print(
                f"[R5-A] {arm} update {update}/{target_updates}; "
                f"val_task_loss={validation['task_loss']:.6f}; microbatch={microbatch}",
                flush=True,
            )
        else:
            history.append(row)
            if update % 20 == 0:
                _write_history(history_path, history)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    final_state = torch.load(resume_path, map_location="cpu", weights_only=False)
    result = {
        "protocol_version": 1.1,
        "source_git_sha": _git("rev-parse", "HEAD"),
        "arm": arm,
        "width": width,
        "updates_completed": int(final_state["global_update"]),
        "target_updates": int(target_updates),
        "selected_update": int(final_state["best_update"]),
        "selected_validation": torch.load(
            arm_root / "best_checkpoint.pt", map_location="cpu", weights_only=False
        )["validation"],
        "microbatch_size": int(final_state["microbatch_size"]),
        "effective_batch_size": effective_batch,
        "gradient_accumulation_steps": accumulations,
        "training_seconds_this_invocation": float(elapsed),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history_path": str(history_path.relative_to(ROOT)),
        "best_checkpoint_path": str((arm_root / "best_checkpoint.pt").relative_to(ROOT)),
        "resume_checkpoint_path": str(resume_path.relative_to(ROOT)),
    }
    _write_json(arm_root / "training_record.json", result)
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def _train_arm(
    *,
    arm: str,
    initial_state: Mapping[str, torch.Tensor],
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    plan: np.ndarray,
    r5_config: Mapping[str, Any],
    e003_config: Mapping[str, Any],
    device: torch.device,
    target_updates: int,
) -> dict[str, Any]:
    arm_root = ARTIFACTS / "runs" / "seed_101" / arm
    record_path = arm_root / "training_record.json"
    previous_record = None
    if record_path.is_file():
        previous_record = json.loads(record_path.read_text(encoding="utf-8"))
        if int(previous_record.get("updates_completed", -1)) >= target_updates:
            return previous_record
    training = r5_config["training"]
    fallback_batches = (int(training["initial_microbatch_size"]), *map(int, training["oom_microbatch_fallbacks"]))
    attempts: list[dict[str, Any]] = []
    invocation_started = time.perf_counter()
    for microbatch in fallback_batches:
        try:
            result = _train_attempt(
                arm=arm,
                width=32,
                initial_state=initial_state,
                train_arrays=train_arrays,
                validation_arrays=validation_arrays,
                plan=plan,
                e003_config=e003_config,
                r5_config=r5_config,
                device=device,
                microbatch=microbatch,
                target_updates=target_updates,
                arm_root=arm_root,
            )
            result["prior_oom_attempts"] = attempts
            result = _accumulate_training_timing(
                result,
                previous_record=previous_record,
                invocation_seconds=time.perf_counter() - invocation_started,
            )
            _write_json(record_path, result)
            return result
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({"microbatch_size": microbatch, "error": str(exc)[:1000]})
            with (arm_root / "oom_attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attempts[-1], ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()
            print(f"[R5-A] {arm} OOM at microbatch={microbatch}; retrying smaller with latest full state", flush=True)
    _write_json(arm_root / "failed.json", {"arm": arm, "attempts": attempts, "cpu_fallback": False})
    raise RuntimeError(f"R5-A {arm} OOMed at microbatch 4; stopped without CPU/AMP/effective-batch changes")


def _accumulate_training_timing(
    result: Mapping[str, Any],
    *,
    previous_record: Mapping[str, Any] | None,
    invocation_seconds: float,
) -> dict[str, Any]:
    """Retain wall time across the initial and continuation training segments."""
    current = dict(result)
    previous = dict(previous_record or {})
    previous_seconds = float(
        previous.get(
            "training_seconds_cumulative",
            previous.get("training_seconds_this_invocation", 0.0),
        )
    )
    start_update = int(previous.get("updates_completed", 0))
    end_update = int(current["updates_completed"])
    elapsed = float(invocation_seconds)
    current["training_seconds_this_invocation"] = elapsed
    current["training_seconds_cumulative"] = previous_seconds + elapsed
    current["training_segments"] = [
        *previous.get("training_segments", []),
        {
            "start_update": start_update,
            "end_update": end_update,
            "wall_seconds": elapsed,
        },
    ]
    return current


def _read_arm_history(arm: str) -> list[dict[str, Any]]:
    path = ARTIFACTS / "runs" / "seed_101" / arm / "training_history.csv"
    return _load_history(path)


def _load_plan(train_arrays: Mapping[str, np.ndarray], r5_config: Mapping[str, Any]) -> np.ndarray:
    path = ARTIFACTS / "training_pair_plan.npz"
    seed_base = int(r5_config["training"]["pair_order_seed_base"])
    if path.is_file():
        payload = np.load(path)
        if int(payload["seed_base"]) != seed_base or not np.array_equal(payload["scene_ids"].astype(str), np.asarray(train_arrays["base_scene_id"]).astype(str)):
            raise ValueError("saved R5 pair plan does not match the frozen scene IDs/seed")
        plan = payload["plan"]
    else:
        plan = build_r5_repeated_pair_plan(
            np.asarray(train_arrays["base_scene_id"]).astype(str),
            updates=12800,
            pairs_per_update=int(r5_config["training"]["paired_scenes_per_update"]),
            seed_base=seed_base,
        )
        temporary = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(temporary, plan=plan, scene_ids=np.asarray(train_arrays["base_scene_id"]).astype(str), seed_base=seed_base)
        os.replace(temporary, path)
    return np.asarray(plan, dtype=np.int64)


def run_training() -> dict[str, Any]:
    preflight = _require_preflight()
    r5_config, e003_config, _ = _load_configs()
    generation_path = ARTIFACTS / "data_generation.json"
    if not generation_path.is_file() or not json.loads(generation_path.read_text(encoding="utf-8")).get("all_base_scene_ids_disjoint"):
        raise RuntimeError("R5-A data generation/split audit must pass before training")
    overfit_path = ARTIFACTS / "g1_overfit_sanity.json"
    if not overfit_path.is_file() or not json.loads(overfit_path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("R5-A G1 small-subset overfit gate did not pass")
    cache_root = ROOT / r5_config["data"]["cache_root"]
    train = _load_split(cache_root, "train")
    validation = _load_split(cache_root, "validation")
    if len(train["base_scene_id"]) != 51_200 or len(validation["base_scene_id"]) != 5_000:
        raise RuntimeError("R5-A train/validation counts differ from the frozen E003 source")
    plan = _load_plan(train, r5_config)
    seed = int(r5_config["training"]["seed"])
    initial_state = _initial_student_state(seed)
    initial_path = ARTIFACTS / "initial_student_state.pt"
    if initial_path.is_file():
        saved_initial = torch.load(initial_path, map_location="cpu", weights_only=False)
        if (
            saved_initial.get("seed") != seed
            or saved_initial.get("source_git_sha") != preflight["git_sha"]
            or saved_initial.get("protocol_version") != 1.1
            or saved_initial.get("r5_config") != r5_config
        ):
            raise ValueError("saved R5-A initial state belongs to a different seed, source or config")
        initial_state = saved_initial["state_dict"]
    else:
        _atomic_torch_save({
            "seed": seed,
            "source_git_sha": preflight["git_sha"],
            "protocol_version": 1.1,
            "r5_config": r5_config,
            "state_dict": initial_state,
        }, initial_path)

    torch.manual_seed(seed)
    s2_init = _build_model("S2", initial_state, seed=seed)
    shared_equal = all(
        torch.equal(s2_init.student.state_dict()[key].cpu(), value.cpu())
        for key, value in initial_state.items()
    )
    parameter_counts = {
        "S0": int(sum(value.numel() for value in initial_state.values())),
        "S2": int(s2_init.total_parameter_count),
        "S2_projection": int(s2_init.projection_parameter_count),
    }
    del s2_init
    if not shared_equal:
        raise RuntimeError("R5-A S0/S2 shared TaskStudent initialization is not bit-identical")
    torch.cuda.empty_cache()

    records: dict[str, Any] = {}
    for arm in ("S0", "S2"):
        print(f"[R5-A] starting sequential CUDA training: {arm}", flush=True)
        records[arm] = _train_arm(
            arm=arm,
            initial_state=initial_state,
            train_arrays=train,
            validation_arrays=validation,
            plan=plan,
            r5_config=r5_config,
            e003_config=e003_config,
            device=_require_cuda(),
            target_updates=6400,
        )
        torch.cuda.empty_cache()

    extension = {
        arm: should_extend_r5_training(_read_arm_history(arm))
        for arm in ("S0", "S2")
    }
    extend_both = any(extension.values())
    decision_path = ARTIFACTS / "extension_decision.json"
    prior_decision = None
    if decision_path.is_file():
        prior_decision = json.loads(decision_path.read_text(encoding="utf-8"))
        if prior_decision.get("extend_both_arms") != extend_both or prior_decision.get("arm_criteria") != extension:
            raise RuntimeError("saved extension decision conflicts with the frozen validation history")
    else:
        _write_json(decision_path, {
            "git_sha": preflight["git_sha"],
            "arm_criteria": extension,
            "extend_both_arms": extend_both,
            "criterion": "relative validation task-loss drop >=1% from updates 4800 to 6400 and >=3/4 intervals without >0.5% increase",
            "training_seed_uncertainty_estimated": False,
        })
    if extend_both:
        for arm in ("S0", "S2"):
            print(f"[R5-A] preregistered trend passed; continuing {arm} to 12,800 with optimizer state", flush=True)
            records[arm] = _train_arm(
                arm=arm,
                initial_state=initial_state,
                train_arrays=train,
                validation_arrays=validation,
                plan=plan,
                r5_config=r5_config,
                e003_config=e003_config,
                device=_require_cuda(),
                target_updates=12800,
            )
            torch.cuda.empty_cache()

    result = {
        "experiment": "D2-A-R5/R5-A",
        "protocol_version": 1.1,
        "git_sha": preflight["git_sha"],
        "training_seed": seed,
        "supervision": "physical_truth_only_E1",
        "teacher_losses_used": False,
        "same_train_and_validation_scene_ids_as_E003": True,
        "same_pair_plan_for_S0_S2": True,
        "same_shared_taskstudent_initialization": shared_equal,
        "pair_order_seed_base": int(r5_config["training"]["pair_order_seed_base"]),
        "initial_pair_plan_rows": 6400,
        "continuation_decision": json.loads(decision_path.read_text(encoding="utf-8")),
        "final_blind_generated": False,
        "distillation_run": False,
        "parameter_counts": parameter_counts,
        "arms": records,
    }
    _write_json(ARTIFACTS / "training.json", result)
    return result


def _map_r5a_protocol_gates(legacy_comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Translate inherited E003 gate labels into the frozen R5-A protocol terms."""
    required = {
        "G3_task_learning",
        "G4_task_protection",
        "G4_primary_C2_C3_joint_gain",
        "S2_route_candidate_passed",
    }
    missing = required - set(legacy_comparison)
    if missing:
        raise KeyError(f"E003 comparison is missing expected fields: {sorted(missing)}")

    mapped = {
        key: value
        for key, value in legacy_comparison.items()
        if key not in required
    }
    mapped["G1_basic_learning"] = legacy_comparison["G3_task_learning"]
    mapped["S2_task_protection_screen"] = legacy_comparison["G4_task_protection"]
    mapped["S2_primary_C2_C3_joint_gain_screen"] = legacy_comparison[
        "G4_primary_C2_C3_joint_gain"
    ]
    mapped["G3_architecture_value"] = {
        "candidate": "S2",
        "task_protection_passed": bool(legacy_comparison["G4_task_protection"]["passed"]),
        "primary_gain_passed": bool(
            legacy_comparison["G4_primary_C2_C3_joint_gain"]["passed"]
        ),
        "passed": bool(legacy_comparison["S2_route_candidate_passed"]),
    }
    return mapped


def run_evaluate() -> dict[str, Any]:
    preflight, training, evaluation_sha = _require_posttraining_provenance(require_cuda=True)
    r5_config, e003_config, physics = _load_configs()
    training_path = ARTIFACTS / "training.json"
    if not training_path.is_file():
        raise RuntimeError("R5-A training must finish before evaluation")
    if (
        training.get("protocol_version") != 1.1
        or training.get("git_sha") != preflight["git_sha"]
        or training.get("final_blind_generated") is not False
        or training.get("distillation_run") is not False
    ):
        raise RuntimeError("R5-A evaluation requires truth-only/no-blind training record")
    cache_root = ROOT / r5_config["data"]["cache_root"]
    arrays = {
        name: _load_split(cache_root, name)
        for name in (
            "h0_calibration", "h0_verification", "c3_h0_verification",
            "development", "exact_overlap_boundary",
        )
    }
    train_arrays = _load_split(cache_root, "train")
    selected_h0, h0_rows = e003_runner._selected_course_indices(arrays["h0_verification"], e003_config)
    selected_h1, h1_rows = e003_runner._selected_course_indices(arrays["development"], e003_config)
    selected_h0_flat = np.concatenate([selected_h0[level] for level in ("C0", "C1", "C2", "C3")])
    selected_h1_flat = np.concatenate([selected_h1[level] for level in ("C0", "C1", "C2", "C3")])

    existing_ids_path = ROOT / r5_config["data"]["physical_diagnostic_scene_ids"]
    selected_by_split: dict[str, set[str]] = {"h0_verification": set(), "development": set()}
    if existing_ids_path.is_file():
        with existing_ids_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("split") in selected_by_split:
                    selected_by_split[row["split"]].add(row["base_scene_id"])
        actual_h0 = set(np.asarray(arrays["h0_verification"]["base_scene_id"])[selected_h0_flat].astype(str))
        actual_h1 = set(np.asarray(arrays["development"]["base_scene_id"])[selected_h1_flat].astype(str))
        if selected_by_split["h0_verification"] != actual_h0 or selected_by_split["development"] != actual_h1:
            raise RuntimeError("R5 physical diagnostic scene selection differs from frozen E003 IDs")
    else:
        raise FileNotFoundError("frozen E003 selected_scene_ids.csv is required for R5-A physical diagnostics")

    device = _require_cuda()
    arm_results: dict[str, dict[str, Any]] = {}
    physical_results: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []
    roc_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    predictions_for_archive: dict[str, Any] = {}
    seed = int(r5_config["training"]["seed"])
    initial_state = torch.load(ARTIFACTS / "initial_student_state.pt", map_location="cpu", weights_only=False)["state_dict"]

    for arm in ("S0", "S2"):
        print(f"[R5-A] evaluating selected {arm} checkpoint with frozen E003 evaluation", flush=True)
        checkpoint_path = ARTIFACTS / "runs" / "seed_101" / arm / "best_checkpoint.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("experiment") != "D2-A-R5"
            or checkpoint.get("arm") != arm
            or checkpoint.get("protocol_version") != 1.1
            or checkpoint.get("source_git_sha") != preflight["git_sha"]
            or checkpoint.get("r5_config") != r5_config
        ):
            raise ValueError(f"R5-A checkpoint identity mismatch: {checkpoint_path}")
        model = _build_model(arm, initial_state, seed=seed)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model = model.to(device).eval()
        arm_data, physical, arm_roc, arm_rows, arm_compute = e003_runner._run_arm_evaluation(
            arm, model, device, e003_config, arrays, selected_h0_flat, selected_h1_flat, physics
        )
        arm_results[arm] = arm_data
        physical_results[arm] = physical
        result_rows.extend(arm_rows)
        roc_rows.extend(arm_roc)
        compute_rows.extend(arm_compute)
        predictions_for_archive[arm] = arm_data["predictions"]
        del model
        torch.cuda.empty_cache()

    comparison = _map_r5a_protocol_gates(
        e003_runner._compare_arms(arm_results, arrays, train_arrays, e003_config)
    )
    metrics = {
        "experiment": "D2-A-R5/R5-A",
        "git_sha": preflight["git_sha"],
        "evaluation_git_sha": evaluation_sha,
        "protocol_version": 1.1,
        "supervision": "physical_truth_only_E1",
        "training": training,
        "arms": {arm: arm_results[arm]["metrics"] for arm in ("S0", "S2")},
        "comparison_and_gates": comparison,
        "physical_consistency": {arm: physical_results[arm] for arm in ("S0", "S2")},
        "compute": compute_rows,
        "scene_prediction_table": "experiments/D2-A-R5/R5-A/scene_predictions.csv",
        "final_blind_generated": False,
        "distillation_run": False,
        "bootstrap_unit": "paired_base_scene_id; single seed uncertainty not estimated",
    }
    _write_json(ARTIFACTS / "metrics.json", metrics)
    e003_runner._write_csv(ARTIFACTS / "result.csv", result_rows)
    e003_runner._write_csv(ARTIFACTS / "roc.csv", roc_rows)
    _write_scene_predictions(ARTIFACTS / "scene_predictions.csv", arrays, predictions_for_archive)
    _save_prediction_archive(ARTIFACTS / "predictions.npz", arm_results)
    return {
        "git_sha": preflight["git_sha"],
        "evaluation_git_sha": evaluation_sha,
        "G1_basic_learning": comparison["G1_basic_learning"],
        "S2_task_protection_screen": comparison["S2_task_protection_screen"],
        "S2_primary_C2_C3_joint_gain_screen": comparison[
            "S2_primary_C2_C3_joint_gain_screen"
        ],
        "G3_architecture_value": comparison["G3_architecture_value"],
        "S2_route_candidate_passed": comparison["S2_route_candidate_passed"],
        "final_blind_generated": False,
        "distillation_run": False,
    }


def _write_scene_predictions(
    path: Path,
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    predictions: Mapping[str, Mapping[str, Any]],
) -> None:
    columns = (
        "arm", "split", "base_scene_id", "course_level", "scenario_id", "snr_db",
        "truth_presence", "truth_range_cell", "truth_doppler_hz", "score", "predicted_range_cell", "predicted_doppler_hz",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for arm, split_predictions in predictions.items():
            for split, values in split_predictions.items():
                dataset = arrays[split]
                hypotheses = ("h0", "h1") if "iq_h1" in dataset else ("h0",)
                count = len(dataset["base_scene_id"])
                for index in range(count):
                    for hypothesis in hypotheses:
                        prediction = values[hypothesis]
                        truth_presence = hypothesis == "h1"
                        writer.writerow({
                            "arm": arm,
                            "split": split,
                            "base_scene_id": str(dataset["base_scene_id"][index]),
                            "course_level": str(dataset.get("course_level", np.full(count, ""))[index]),
                            "scenario_id": str(dataset.get("scenario_id", np.full(count, ""))[index]),
                            "snr_db": float(dataset.get("snr_db", np.full(count, np.nan))[index]),
                            "truth_presence": int(truth_presence),
                            "truth_range_cell": float(dataset["truth_range_cell"][index]) if truth_presence else 0.0,
                            "truth_doppler_hz": float(dataset["truth_doppler_hz"][index]) if truth_presence else 0.0,
                            "score": float(prediction["score"][index]),
                            "predicted_range_cell": float(prediction["range_cell"][index]) if truth_presence else "",
                            "predicted_doppler_hz": float(prediction["doppler_hz"][index]) if truth_presence else "",
                        })


def _save_prediction_archive(path: Path, arm_results: Mapping[str, Mapping[str, Any]]) -> None:
    arrays_to_save: dict[str, np.ndarray] = {}
    for arm, arm_data in arm_results.items():
        for split, split_data in arm_data["predictions"].items():
            for hypothesis in ("h0", "h1"):
                if hypothesis not in split_data:
                    continue
                for field, values in split_data[hypothesis].items():
                    arrays_to_save[f"{arm}_{split}_{hypothesis}_{field}"] = np.asarray(values)
    np.savez_compressed(path, **arrays_to_save)


def run_cleanup() -> dict[str, Any]:
    """Delete only this stage's manifest-listed reconstructible cache files."""
    preflight, training, cleanup_sha = _require_posttraining_provenance(require_cuda=False)
    metrics_path = ARTIFACTS / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("R5-A evaluation must finish before deleting reconstructible data")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("git_sha") != training.get("git_sha") or metrics.get("final_blind_generated") is not False:
        raise RuntimeError("R5-A metrics provenance is inconsistent; preserve the cache")
    cache_root = ROOT / _read_yaml(R5_CONFIG_PATH)["data"]["cache_root"]
    manifest_path = ARTIFACTS / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("R5-A cache manifest is missing; no cleanup is permitted")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    listed = {item["path"]: int(item["size_bytes"]) for item in manifest["files"]}
    actual = {
        path.relative_to(cache_root).as_posix(): int(path.stat().st_size)
        for path in cache_root.rglob("*")
        if path.is_file()
    }
    if actual != listed:
        raise RuntimeError("R5-A cache differs from its manifest; preserve all files and review manually")
    removed_bytes = 0
    removed_files: list[str] = []
    for relative, size in listed.items():
        target = cache_root / relative
        if target.is_file() and not target.is_symlink():
            target.unlink()
            removed_bytes += size
            removed_files.append(relative)
    for directory in sorted((path for path in cache_root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass
    try:
        cache_root.rmdir()
    except OSError:
        pass
    audit = {
        "training_git_sha": preflight["git_sha"],
        "cleanup_code_git_sha": cleanup_sha,
        "cache_root": str(cache_root.relative_to(ROOT)),
        "removed_file_count": len(removed_files),
        "removed_bytes": removed_bytes,
        "removed_files": removed_files,
        "only_manifest_listed_regular_files_removed": True,
        "final_blind_generated": False,
    }
    _write_json(ARTIFACTS / "cleanup.json", audit)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "data", "overfit", "train", "evaluate", "cleanup"))
    args = parser.parse_args(argv)
    actions = {
        "preflight": run_preflight,
        "data": run_generate_data,
        "overfit": run_overfit_sanity,
        "train": run_training,
        "evaluate": run_evaluate,
        "cleanup": run_cleanup,
    }
    result = actions[args.phase]()
    print(json.dumps(_json_safe(result), indent=2, ensure_ascii=False), flush=True)
    if isinstance(result, dict) and result.get("passed") is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
