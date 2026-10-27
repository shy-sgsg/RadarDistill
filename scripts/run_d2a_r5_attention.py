#!/usr/bin/env python3
"""CUDA-only execution for the preregistered D2-A-R5-B TFAS pilot."""

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
import run_d2a_r5 as r5_runner
from radardistill.distillation.task_losses import task_loss
from radardistill.students.temporal_frequency_attention import TemporalFrequencyAttentionStudent
from radardistill.training.d2a_e002 import (
    _cpu_state_dict,
    _paired_validation,
    make_paired_examples,
    scale_d2a_e002_microbatch_loss,
)


ARTIFACTS = ROOT / "experiments/D2-A-R5/R5-B"
RUN_ROOT = ARTIFACTS / "runs/seed_101/TFAS"
CONFIG_PATH = ARTIFACTS / "config.yaml"
PROTOCOL_PATH = ARTIFACTS / "protocol.md"
SURVEY_PATH = ARTIFACTS / "architecture_survey.md"
R5_CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-A/config.yaml"
E003_CONFIG_PATH = ROOT / "experiments/D2-A-E003/config.yaml"
PHYSICS_CONFIG_PATH = ROOT / "experiments/D2-A-E001/config.yaml"
CAPACITY_ROOT = ROOT / "experiments/D2-A-R5/R5-A/capacity"
TRAINING_PAIR_PLAN_PATH = ROOT / "experiments/D2-A-R5/R5-A/training_pair_plan.npz"
FROZEN_INPUTS = (
    "AGENTS.md",
    "docs/RESEARCH_CHARTER.md",
    "docs/CURRENT_STATUS.md",
    "docs/RESEARCH_ROADMAP.md",
    "docs/research/ASSUMPTION_REGISTER.md",
    "experiments/D2-A-R5/R5-B/protocol.md",
    "experiments/D2-A-R5/R5-B/config.yaml",
    "experiments/D2-A-R5/R5-B/architecture_survey.md",
    "experiments/D2-A-R5/R5-A/protocol.md",
    "experiments/D2-A-R5/R5-A/config.yaml",
    "experiments/D2-A-E003/protocol.md",
    "experiments/D2-A-E003/config.yaml",
    "experiments/D2-A-E001/config.yaml",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/students/task_student.py",
    "src/radardistill/students/temporal_frequency_attention.py",
    "src/radardistill/distillation/task_losses.py",
    "src/radardistill/evaluation/d2a_e001.py",
    "src/radardistill/evaluation/d2a_e002.py",
    "src/radardistill/evaluation/physical_consistency.py",
    "src/radardistill/training/d2a_e002.py",
    "src/radardistill/training/d2a_e003.py",
    "src/radardistill/training/d2a_e003_data.py",
    "src/radardistill/training/d2a_r5.py",
    "scripts/run_d2a_e003.py",
    "scripts/run_d2a_r5.py",
    "scripts/run_d2a_r5_attention.py",
    "tests/test_temporal_frequency_attention_student.py",
    "tests/test_d2a_r5_attention_runner.py",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
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
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True)
    return result.stdout.strip()


def _attention_learning_rate(update: int) -> float:
    """Return the preregistered two-segment cosine LR for updates 1..12,800."""
    if not 1 <= int(update) <= 12_800:
        raise ValueError("attention update must be in the frozen range 1..12800")
    if update <= 6400:
        start, end, position = 1e-3, 1e-4, update - 1
    else:
        start, end, position = 1e-4, 1e-5, update - 6401
    progress = position / 6399.0
    return float(end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress)))


def _microbatch_candidates(training: Mapping[str, Any]) -> tuple[int, ...]:
    effective = int(training["effective_batch_size"])
    values = (int(training["initial_microbatch_size"]), *map(int, training["oom_microbatch_fallbacks"]))
    unique: list[int] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    if any(value <= 0 or value > effective or value % 2 or effective % value for value in unique):
        raise ValueError("attention microbatch candidates must be even divisors of the effective batch")
    if not unique:
        raise ValueError("at least one attention microbatch candidate is required")
    return tuple(unique)


def _validate_attention_pair_plan(
    plan: np.ndarray,
    *,
    scene_count: int,
    pairs_per_update: int,
    expected_updates: int,
) -> None:
    values = np.asarray(plan)
    if values.shape != (expected_updates, pairs_per_update):
        raise ValueError("attention pair plan shape differs from the frozen update budget")
    if scene_count <= 0 or pairs_per_update <= 0 or scene_count % pairs_per_update:
        raise ValueError("attention pair plan requires complete pass-sized batches")
    if values.size and (int(values.min()) < 0 or int(values.max()) >= scene_count):
        raise ValueError("attention pair plan contains an index outside the train split")
    if np.any(np.apply_along_axis(lambda row: np.unique(row).size != row.size, 1, values)):
        raise ValueError("an attention pair-plan update repeats a scene")
    updates_per_pass = scene_count // pairs_per_update
    if expected_updates % updates_per_pass:
        raise ValueError("attention pair plan must comprise complete pass-sized epochs")
    for epoch_start in range(0, expected_updates, updates_per_pass):
        epoch = values[epoch_start:epoch_start + updates_per_pass].reshape(-1)
        if not np.array_equal(np.sort(epoch), np.arange(scene_count, dtype=epoch.dtype)):
            raise ValueError("each attention pair-plan epoch must be one complete pass")


def _load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read_yaml(CONFIG_PATH)
    r5 = _read_yaml(R5_CONFIG_PATH)
    e003 = _read_yaml(E003_CONFIG_PATH)
    physics = _read_yaml(PHYSICS_CONFIG_PATH)
    _validate_config(config, r5, e003)
    return config, r5, e003, physics


def _validate_config(config: Mapping[str, Any], r5: Mapping[str, Any], e003: Mapping[str, Any]) -> None:
    experiment = config.get("experiment", {})
    model = config.get("model", {})
    data = config.get("data", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    expected_model = {
        "name": "TemporalFrequencyAttentionStudent",
        "input_layout": ["batch", 6, 130, 32],
        "embedding_width": 56,
        "attention_heads": 4,
        "temporal_encoder_layers": 2,
        "range_encoder_layers": 1,
        "feedforward_width": 112,
        "expected_parameters": 79_638,
    }
    if experiment.get("subphase") != "R5-B" or experiment.get("protocol_version") != 1.0:
        raise ValueError("R5-B config identity or protocol version changed")
    if experiment.get("final_blind_generation_authorized") is not False or experiment.get("distillation_authorized") is not False:
        raise ValueError("R5-B does not authorize final blind generation or distillation")
    for key, value in expected_model.items():
        if model.get(key) != value:
            raise ValueError(f"R5-B model setting is not preregistered: {key}")
    if (
        training.get("seed") != 101
        or training.get("target_updates") != 12_800
        or training.get("paired_scenes_per_update") != 32
        or training.get("effective_batch_size") != 64
        or training.get("precision") != "FP32"
        or training.get("device") != "cuda:0"
        or training.get("cpu_fallback") != "forbidden"
        or training.get("amp") is not False
        or training.get("supervision") != "physical_truth_only_E1"
        or training.get("teacher_hard_labels") is not False
        or training.get("teacher_soft_responses") is not False
        or training.get("weight_decay") != 1e-4
        or training.get("gradient_clip_norm") != 1.0
        or training.get("validation_interval_updates") != 400
        or training.get("learning_rate_schedule") != [
            {"updates": [1, 6400], "type": "cosine", "start": 1e-3, "end": 1e-4},
            {"updates": [6401, 12800], "type": "cosine", "start": 1e-4, "end": 1e-5},
        ]
    ):
        raise ValueError("R5-B training budget, device or supervision differs from preregistration")
    if (
        data.get("pair_plan") != "experiments/D2-A-R5/R5-A/training_pair_plan.npz"
        or data.get("reuse_frozen_e003_r5a_splits")
        != ["train", "validation", "h0_calibration", "h0_verification", "c3_h0_verification", "development", "exact_overlap_boundary"]
    ):
        raise ValueError("R5-B must reuse the fixed R5-A plan and E003 splits")
    if r5["training"]["effective_batch_size"] != training["effective_batch_size"]:
        raise ValueError("R5-B effective batch must match the R5-A CNN capacity comparison")
    if e003["detection"]["target_pfa_cpi"] != evaluation["target_pfa_cpi"]:
        raise ValueError("R5-B threshold calibration target must inherit E003")
    if _microbatch_candidates(training) != (16, 8, 4):
        raise ValueError("R5-B CUDA microbatch/OOM fallback is frozen to 16, 8, 4")


def _require_frozen_main() -> str:
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R5-B execution requires the main branch")
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", *FROZEN_INPUTS], cwd=ROOT).returncode != 0:
        raise RuntimeError("R5-B code, protocol, config or frozen inputs have tracked uncommitted changes")
    head, remote = _git("rev-parse", "HEAD"), _git("rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError("R5-B implementation/protocol must be committed and pushed before execution")
    return head


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R5-B requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    expected = "NVIDIA GeForce RTX 3050 Laptop GPU"
    actual = torch.cuda.get_device_name(0)
    if actual != expected:
        raise RuntimeError(f"R5-B requires {expected}, found {actual}")
    return torch.device("cuda:0")


def _environment_record(commit: str) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info(0)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return {
        "git_sha": commit,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_memory_free_bytes": int(free),
        "cuda_memory_total_bytes": int(total),
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "nvidia_smi_query": smi.stdout.strip(),
        "nvidia_smi_returncode": int(smi.returncode),
    }


def _verify_r5a_artifacts() -> dict[str, Any]:
    preflight = json.loads((CAPACITY_ROOT / "preflight.json").read_text(encoding="utf-8"))
    training = json.loads((CAPACITY_ROOT / "training.json").read_text(encoding="utf-8"))
    metrics = json.loads((CAPACITY_ROOT / "metrics.json").read_text(encoding="utf-8"))
    if (
        preflight.get("passed") is not True
        or training.get("capacity_git_sha") != preflight.get("capacity_git_sha")
        or metrics.get("capacity_git_sha") != training.get("capacity_git_sha")
        or training.get("training_seed") != 101
        or training.get("effective_batch_size") != 64
        or training.get("target_updates") != 12_800
        or training.get("same_pair_plan_as_R5A") is not True
        or training.get("teacher_losses_used") is not False
        or training.get("final_blind_generated") is not False
        or training.get("distillation_run") is not False
        or any(training.get("arms", {}).get(name, {}).get("updates_completed") != 12_800 for name in ("Small", "Medium"))
    ):
        raise RuntimeError("R5-A capacity training/evaluation provenance is incomplete or inconsistent")
    return {"preflight": preflight, "training": training, "metrics": metrics}


def _verify_cache(cache_root: Path, config: Mapping[str, Any], r5_config: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = ROOT / "experiments/D2-A-R5/R5-A/cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    generation = json.loads((ROOT / "experiments/D2-A-R5/R5-A/data_generation.json").read_text(encoding="utf-8"))
    if generation.get("all_base_scene_ids_disjoint") is not True or generation.get("final_blind_generated") is not False:
        raise RuntimeError("R5-A source split audit is missing, failed, or claims a final blind")
    if Path(manifest.get("cache_root", "")).as_posix() != r5_config["data"]["cache_root"]:
        raise ValueError("R5-A cache manifest root differs from the frozen R5 cache configuration")
    for item in manifest["files"]:
        path = cache_root / item["path"]
        if not path.is_file() or path.stat().st_size != int(item["size_bytes"]):
            raise RuntimeError(f"frozen R5-A cache differs from its manifest: {path}")
    if len(list(path for path in cache_root.rglob("*") if path.is_file())) != len(manifest["files"]):
        raise RuntimeError("frozen R5-A cache has unlisted files; preserve and review before proceeding")
    expected_counts = {
        "train": int(config["data"]["candidate_training_pairs"]),
        "validation": int(config["data"]["candidate_validation_pairs"]),
    }
    splits: dict[str, dict[str, np.ndarray]] = {}
    for name in config["data"]["reuse_frozen_e003_r5a_splits"]:
        arrays = r5_runner._load_split(cache_root, name)
        splits[name] = arrays
        if name in expected_counts and len(arrays["base_scene_id"]) != expected_counts[name]:
            raise RuntimeError(f"R5-B frozen {name} scene count changed")
        ids = np.asarray(arrays["base_scene_id"]).astype(str)
        if len(np.unique(ids)) != len(ids):
            raise RuntimeError(f"R5-B frozen {name} split contains duplicate scene IDs")
    train_ids = np.asarray(splits["train"]["base_scene_id"]).astype(str)
    plan_payload = np.load(TRAINING_PAIR_PLAN_PATH, allow_pickle=False)
    if int(plan_payload["seed_base"]) != int(r5_config["training"]["pair_order_seed_base"]):
        raise RuntimeError("R5-B pair-plan seed differs from the frozen R5-A seed")
    if not np.array_equal(plan_payload["scene_ids"].astype(str), train_ids):
        raise RuntimeError("R5-B training pair plan scene IDs differ from the frozen train cache")
    plan = np.asarray(plan_payload["plan"], dtype=np.int64)
    _validate_attention_pair_plan(
        plan,
        scene_count=len(train_ids),
        pairs_per_update=int(config["training"]["paired_scenes_per_update"]),
        expected_updates=int(config["training"]["target_updates"]),
    )
    expected_ids = np.load(TRAINING_PAIR_PLAN_PATH, allow_pickle=False)["scene_ids"].astype(str)
    cache_id_consistency = {
        name: bool(np.array_equal(np.asarray(splits[name]["base_scene_id"]).astype(str), expected_ids))
        for name in ("train",)
    }
    if not all(cache_id_consistency.values()):
        raise RuntimeError("R5-B data/pair-plan scene provenance failed")
    disjoint_counts = r5_runner.validate_e003_split_disjointness(splits)
    if disjoint_counts != generation.get("split_counts"):
        raise RuntimeError("R5-B frozen cache split IDs/counts differ from the R5-A data audit")
    return {
        "cache_manifest_file_count": len(manifest["files"]),
        "cache_manifest_total_bytes": int(manifest["total_bytes"]),
        "split_scene_counts": {name: len(value["base_scene_id"]) for name, value in splits.items()},
        "train_scene_ids_match_pair_plan": True,
        "e003_split_ids_disjoint": True,
        "pair_plan_shape": list(plan.shape),
        "pair_plan_complete_passes": 8,
        "splits": splits,
        "plan": plan,
    }


def run_preflight() -> dict[str, Any]:
    config, r5_config, e003_config, _ = _load_configs()
    commit = _require_frozen_main()
    device = _require_cuda()
    r5a = _verify_r5a_artifacts()
    cache_root = ROOT / r5_config["data"]["cache_root"]
    cache_record = _verify_cache(cache_root, config, r5_config)
    model = TemporalFrequencyAttentionStudent()
    parameters = int(sum(parameter.numel() for parameter in model.parameters()))
    reference = int(r5_config["capacity"]["parameter_count_expected"]["Medium"])
    relative_difference = abs(parameters - reference) / reference
    lower = 1.0 - float(config["model"]["parameter_match_relative_tolerance"])
    upper = 1.0 + float(config["model"]["parameter_match_relative_tolerance"])
    if parameters != int(config["model"]["expected_parameters"]) or not lower <= parameters / reference <= upper:
        raise RuntimeError("TFAS parameter count falls outside its preregistered capacity window")
    existing = ARTIFACTS / "preflight.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("git_sha") != commit:
            raise RuntimeError("R5-B preflight belongs to another source commit; preserve and inspect it")
        return previous
    record = {
        "experiment": "D2-A-R5-B",
        "git_sha": commit,
        "protocol_version": 1.0,
        "passed": True,
        "device": str(device),
        "precision": "FP32",
        "amp": False,
        "cpu_fallback": False,
        "parameter_count": parameters,
        "parameter_count_reference_medium": reference,
        "parameter_count_relative_difference": relative_difference,
        "r5a_training_sha": r5a["training"]["capacity_git_sha"],
        "r5a_small_medium_updates": {key: r5a["training"]["arms"][key]["updates_completed"] for key in ("Small", "Medium")},
        "data_audit": {key: value for key, value in cache_record.items() if key not in ("splits", "plan")},
        "environment": _environment_record(commit),
        "final_blind_generated": False,
        "distillation_run": False,
    }
    _write_json(existing, record)
    print(f"[R5-B] G0 passed; parameters={parameters:,}; device={device}", flush=True)
    return record


def _require_preflight() -> dict[str, Any]:
    path = ARTIFACTS / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run the pushed R5-B preflight before G1 or training")
    record = json.loads(path.read_text(encoding="utf-8"))
    commit = _require_frozen_main()
    if record.get("git_sha") != commit or record.get("passed") is not True:
        raise RuntimeError("R5-B preflight is stale or failed")
    _require_cuda()
    return record


def _head_gradient_norms(model: torch.nn.Module) -> dict[str, float]:
    results: dict[str, float] = {}
    for name in ("presence_head", "range_head", "range_offset_head", "doppler_head"):
        gradients = [parameter.grad for parameter in getattr(model, name).parameters()]
        if not gradients or any(value is None or not torch.isfinite(value).all() for value in gradients):
            raise FloatingPointError(f"TFAS {name} has a missing or non-finite gradient")
        norm = float(torch.sqrt(sum(torch.sum(value.detach() ** 2) for value in gradients)).item())
        if not math.isfinite(norm) or norm <= 0.0:
            raise FloatingPointError(f"TFAS {name} has a zero or non-finite gradient norm")
        results[name] = norm
    return results


def _overfit_attempt(
    *,
    train_arrays: Mapping[str, np.ndarray],
    loss_config: Mapping[str, Any],
    device: torch.device,
    microbatch: int,
) -> dict[str, Any]:
    torch.manual_seed(101)
    torch.cuda.manual_seed_all(101)
    model = TemporalFrequencyAttentionStudent().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = make_paired_examples(train_arrays, np.arange(16, dtype=np.int64), include_teacher=False)
    cpu_x = torch.from_numpy(batch["iq_ri"])
    cpu_truth = {
        key: torch.from_numpy(batch[key])
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    effective = len(cpu_x)
    effective_h1 = 16
    if microbatch <= 0 or effective % microbatch or microbatch % 2:
        raise ValueError("G1 microbatch must split H0/H1 pairs and divide its 32-example effective batch")
    model.eval()
    with torch.no_grad():
        output = model(cpu_x.to(device), compute_spectrum=False)
        initial_terms = task_loss(output, {key: value.to(device) for key, value in cpu_truth.items()}, loss_config, teacher_weight=0.0)
        initial_loss = float(initial_terms["task_total"].item())
    model.train()
    optimizer.zero_grad(set_to_none=True)
    probe_x = cpu_x[:microbatch].to(device)
    probe_truth = {key: value[:microbatch].to(device) for key, value in cpu_truth.items()}
    probe_losses = task_loss(model(probe_x, compute_spectrum=False), probe_truth, loss_config, teacher_weight=0.0)
    probe_losses["task_total"].backward()
    gradient_norms = _head_gradient_norms(model)
    optimizer.zero_grad(set_to_none=True)
    del probe_x, probe_truth, probe_losses

    for update in range(200):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, effective, microbatch):
            stop = start + microbatch
            x = cpu_x[start:stop].to(device)
            truth = {key: value[start:stop].to(device) for key, value in cpu_truth.items()}
            terms = task_loss(model(x, compute_spectrum=False), truth, loss_config, teacher_weight=0.0)
            micro_h1 = int(truth["truth_presence"].sum().item())
            scaled = scale_d2a_e002_microbatch_loss(
                terms,
                dict(loss_config),
                microbatch_size=stop - start,
                effective_batch_size=effective,
                microbatch_h1=micro_h1,
                effective_h1=effective_h1,
                teacher_weight=0.0,
            )
            if not torch.isfinite(scaled):
                raise FloatingPointError("non-finite TFAS G1 overfit loss")
            scaled.backward()
            del x, truth, terms, scaled
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("non-finite TFAS G1 gradient norm")
        optimizer.step()

    model.eval()
    with torch.inference_mode():
        final_outputs = model(cpu_x.to(device), compute_spectrum=False)
        final_terms = task_loss(final_outputs, {key: value.to(device) for key, value in cpu_truth.items()}, loss_config, teacher_weight=0.0)
        final_loss = float(final_terms["task_total"].item())
        presence_accuracy = float(((final_outputs["presence_logit"] >= 0.0) == cpu_truth["truth_presence"].to(device)).float().mean().item())
        truth_cells = torch.floor(cpu_truth["truth_range_cell"][1::2].to(device)).long().clamp(0, 31)
        range_accuracy = float((final_outputs["range_logits"][1::2].argmax(dim=-1) == truth_cells).float().mean().item())
    result = {
        "microbatch_size": int(microbatch),
        "paired_scenes": 16,
        "updates": 200,
        "initial_task_loss": initial_loss,
        "final_task_loss": final_loss,
        "relative_loss_reduction": float(1.0 - final_loss / max(initial_loss, 1e-12)),
        "presence_accuracy": presence_accuracy,
        "h1_range_cell_accuracy": range_accuracy,
        "task_head_gradient_norms": gradient_norms,
        "finite_outputs_and_losses": bool(math.isfinite(initial_loss) and math.isfinite(final_loss)),
        "passed": bool(
            final_loss <= 0.5 * initial_loss
            and presence_accuracy >= 1.0
            and range_accuracy >= 0.95
            and all(math.isfinite(value) and value > 0.0 for value in gradient_norms.values())
        ),
    }
    del model, optimizer, cpu_x, cpu_truth
    torch.cuda.empty_cache()
    return result


def run_g1() -> dict[str, Any]:
    preflight = _require_preflight()
    config, _, e003, _ = _load_configs()
    cache_root = ROOT / config["data"]["cache_root"]
    train = r5_runner._load_split(cache_root, "train")
    loss_config = r5_runner._training_loss_config(e003)
    device = _require_cuda()
    training = config["training"]
    candidates = (int(training["initial_microbatch_size"]), *map(int, training["oom_microbatch_fallbacks"]))
    attempts: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    for microbatch in candidates:
        try:
            print(f"[R5-B] G1 real-data overfit check at microbatch={microbatch}", flush=True)
            result = _overfit_attempt(
                train_arrays=train,
                loss_config=loss_config,
                device=device,
                microbatch=microbatch,
            )
            result["prior_oom_attempts"] = attempts
            break
        except torch.cuda.OutOfMemoryError as exc:
            attempts.append({"microbatch_size": microbatch, "error": str(exc)[:1000]})
            torch.cuda.empty_cache()
            print(f"[R5-B] G1 OOM at microbatch={microbatch}; retrying smaller", flush=True)
    if result is None:
        result = {"passed": False, "reason": "CUDA OOM at microbatch 4", "prior_oom_attempts": attempts}
    record = {
        "experiment": "D2-A-R5-B",
        "git_sha": preflight["git_sha"],
        "gate": "G1",
        "result": result,
        "cuda_device": torch.cuda.get_device_name(0),
        "cpu_fallback": False,
    }
    _write_json(ARTIFACTS / "g1_overfit_sanity.json", record)
    print(f"[R5-B] G1 {'passed' if result.get('passed') else 'failed'}: "
          f"loss reduction={result.get('relative_loss_reduction')}, "
          f"presence={result.get('presence_accuracy')}, range={result.get('h1_range_cell_accuracy')}", flush=True)
    if not result.get("passed"):
        raise RuntimeError("R5-B G1 failed; preregistered 12,800-update pilot is stopped")
    return record


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "update", "epoch", "microbatch", "learning_rate", "train_task_loss",
        "train_presence_loss", "train_range_cell_loss", "train_range_offset_loss", "train_doppler_loss",
        "validation_task_loss", "validation_presence_loss", "validation_range_cell_loss",
        "validation_range_offset_loss", "validation_doppler_loss", "validation_auc",
        "validation_range_rmse_cells", "validation_doppler_rmse_hz",
        "validation_true_cell_doppler_rmse_hz", "validation_range_cell_mismatch_rate",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _history_row(
    update: int,
    microbatch: int,
    learning_rate: float,
    train_terms: Mapping[str, float],
    validation: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "update": int(update),
        "epoch": int(math.ceil(update / 1600)) if update else 0,
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
            "validation_task_loss": validation["task_loss"],
            "validation_presence_loss": validation["presence_loss"],
            "validation_range_cell_loss": validation["range_cell_loss"],
            "validation_range_offset_loss": validation["range_offset_loss"],
            "validation_doppler_loss": validation["doppler_loss"],
            "validation_auc": validation["auc"],
            "validation_range_rmse_cells": validation["range_rmse_all_h1_cells"],
            "validation_doppler_rmse_hz": validation["doppler_rmse_all_h1_hz"],
            "validation_true_cell_doppler_rmse_hz": validation["true_cell_doppler_rmse_hz"],
            "validation_range_cell_mismatch_rate": validation["predicted_range_cell_mismatch_rate"],
        })
    return row


def _save_training_state(
    path: Path,
    *,
    commit: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    update: int,
    microbatch: int,
    best_loss: float,
    best_update: int,
) -> None:
    _atomic_torch_save(path, {
        "experiment": "D2-A-R5-B",
        "protocol_version": 1.0,
        "git_sha": commit,
        "update": int(update),
        "microbatch_size": int(microbatch),
        "best_validation_loss": float(best_loss),
        "best_update": int(best_update),
        "model_state_dict": _cpu_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
    })


def _train_attempt(
    *,
    commit: str,
    train: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    plan: np.ndarray,
    loss_config: Mapping[str, Any],
    device: torch.device,
    microbatch: int,
) -> dict[str, Any]:
    config = _read_yaml(CONFIG_PATH)
    training = config["training"]
    effective = int(training["effective_batch_size"])
    effective_h1 = int(training["paired_scenes_per_update"])
    if effective % microbatch or microbatch % 2:
        raise ValueError("TFAS microbatch must be an even divisor of effective batch 64")
    _validate_attention_pair_plan(
        plan,
        scene_count=len(train["base_scene_id"]),
        pairs_per_update=int(training["paired_scenes_per_update"]),
        expected_updates=int(training["target_updates"]),
    )
    torch.manual_seed(int(training["seed"]))
    torch.cuda.manual_seed_all(int(training["seed"]))
    model = TemporalFrequencyAttentionStudent().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=float(training["weight_decay"])
    )
    latest_path = RUN_ROOT / "latest_training_state.pt"
    best_path = RUN_ROOT / "best_checkpoint.pt"
    history_path = RUN_ROOT / "training_history.csv"
    start_update = 0
    best_loss = float("inf")
    best_update = 0
    history: list[dict[str, Any]] = []
    if latest_path.is_file():
        state = torch.load(latest_path, map_location="cpu", weights_only=False)
        if state.get("git_sha") != commit or state.get("experiment") != "D2-A-R5-B" or state.get("protocol_version") != 1.0:
            raise RuntimeError("TFAS resume state belongs to different source/protocol; preserve and inspect")
        # A smaller microbatch is a permitted OOM recovery because the optimizer
        # update and effective batch stay fixed and dropout is frozen to zero.
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_update = int(state["update"])
        best_loss = float(state["best_validation_loss"])
        best_update = int(state["best_update"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        random.setstate(state["python_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        history = [row for row in r5_runner._load_history(history_path) if int(row["update"]) <= start_update]
    else:
        model.eval()
        initial_validation = _paired_validation(model, validation_arrays, dict(loss_config), device, microbatch)
        best_loss = float(initial_validation["task_loss"])
        _atomic_torch_save(best_path, {
            "experiment": "D2-A-R5-B", "protocol_version": 1.0, "git_sha": commit,
            "model_name": "TemporalFrequencyAttentionStudent", "seed": int(training["seed"]),
            "update": 0, "model_state_dict": _cpu_state_dict(model), "validation": initial_validation,
        })
        history = [_history_row(0, microbatch, _attention_learning_rate(1), {}, initial_validation)]
        _save_training_state(
            latest_path, commit=commit, model=model, optimizer=optimizer, update=0,
            microbatch=microbatch, best_loss=best_loss, best_update=0,
        )
        model.train()
    _write_history(history_path, history)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    validation_interval = int(training["validation_interval_updates"])
    target_updates = int(training["target_updates"])
    accumulation_steps = effective // microbatch

    for update_index in range(start_update, target_updates):
        update = update_index + 1
        learning_rate = _attention_learning_rate(update)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        batch = make_paired_examples(train, plan[update_index], include_teacher=False)
        optimizer.zero_grad(set_to_none=True)
        train_terms = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "task_total")}
        for micro_start in range(0, effective, microbatch):
            micro_stop = micro_start + microbatch
            x = torch.from_numpy(batch["iq_ri"][micro_start:micro_stop]).to(device)
            truth = {
                key: torch.from_numpy(batch[key][micro_start:micro_stop]).to(device)
                for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
            }
            outputs = model(x, compute_spectrum=False)
            losses = task_loss(outputs, truth, dict(loss_config), teacher_weight=0.0)
            micro_h1 = int(truth["truth_presence"].sum().item())
            scaled = scale_d2a_e002_microbatch_loss(
                losses,
                dict(loss_config),
                microbatch_size=micro_stop - micro_start,
                effective_batch_size=effective,
                microbatch_h1=micro_h1,
                effective_h1=effective_h1,
                teacher_weight=0.0,
            )
            if not torch.isfinite(scaled):
                raise FloatingPointError(f"non-finite TFAS loss at update {update}")
            scaled.backward()
            sample_fraction = (micro_stop - micro_start) / effective
            h1_fraction = micro_h1 / effective_h1
            train_terms["presence"] += float(losses["presence"].detach().item()) * sample_fraction
            for name in ("range_cell", "range_offset", "doppler"):
                train_terms[name] += float(losses[name].detach().item()) * h1_fraction
            train_terms["task_total"] += float(scaled.detach().item())
            del outputs, losses, scaled, truth, x
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite TFAS gradient norm at update {update}")
        optimizer.step()
        row = _history_row(update, microbatch, learning_rate, train_terms)

        if update % validation_interval == 0:
            model.eval()
            validation = _paired_validation(model, validation_arrays, dict(loss_config), device, microbatch)
            if not all(math.isfinite(float(value)) for value in validation.values()):
                raise FloatingPointError(f"non-finite TFAS validation metric at update {update}")
            row.update(_history_row(update, microbatch, learning_rate, train_terms, validation))
            if float(validation["task_loss"]) < best_loss:
                best_loss = float(validation["task_loss"])
                best_update = update
                _atomic_torch_save(best_path, {
                    "experiment": "D2-A-R5-B", "protocol_version": 1.0, "git_sha": commit,
                    "model_name": "TemporalFrequencyAttentionStudent", "seed": int(training["seed"]),
                    "update": update, "model_state_dict": _cpu_state_dict(model), "validation": dict(validation),
                })
            history.append(row)
            _write_history(history_path, history)
            _save_training_state(
                latest_path, commit=commit, model=model, optimizer=optimizer, update=update,
                microbatch=microbatch, best_loss=best_loss, best_update=best_update,
            )
            model.train()
            print(
                f"[R5-B] TFAS update {update}/{target_updates}; val_task_loss={validation['task_loss']:.6f}; "
                f"best={best_loss:.6f}@{best_update}; microbatch={microbatch}",
                flush=True,
            )
        else:
            history.append(row)
            if update % 20 == 0:
                _write_history(history_path, history)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    result = {
        "git_sha": commit,
        "protocol_version": 1.0,
        "model": "TemporalFrequencyAttentionStudent",
        "seed": int(training["seed"]),
        "updates_completed": target_updates,
        "target_updates": target_updates,
        "selected_update": best_update,
        "selected_validation": torch.load(best_path, map_location="cpu", weights_only=False)["validation"],
        "microbatch_size": microbatch,
        "effective_batch_size": effective,
        "gradient_accumulation_steps": accumulation_steps,
        "training_seconds": float(elapsed),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "history_path": str(history_path.relative_to(ROOT)),
        "best_checkpoint_path": str(best_path.relative_to(ROOT)),
        "resume_checkpoint_path": str(latest_path.relative_to(ROOT)),
        "prior_oom_attempts": [],
        "teacher_losses_used": False,
        "final_blind_generated": False,
        "distillation_run": False,
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return result


def run_training() -> dict[str, Any]:
    preflight = _require_preflight()
    g1_path = ARTIFACTS / "g1_overfit_sanity.json"
    if not g1_path.is_file() or json.loads(g1_path.read_text(encoding="utf-8")).get("result", {}).get("passed") is not True:
        raise RuntimeError("R5-B G1 did not pass; formal pilot training is not authorized")
    config, r5_config, e003_config, _ = _load_configs()
    cache_root = ROOT / r5_config["data"]["cache_root"]
    train = r5_runner._load_split(cache_root, "train")
    validation = r5_runner._load_split(cache_root, "validation")
    audit = _verify_cache(cache_root, config, r5_config)
    plan = audit["plan"]
    loss_config = r5_runner._training_loss_config(e003_config)
    device = _require_cuda()
    record_path = ARTIFACTS / "training.json"
    if record_path.is_file():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if previous.get("git_sha") != preflight["git_sha"]:
            raise RuntimeError("existing R5-B training record belongs to another source SHA")
        if previous.get("updates_completed") == 12_800:
            return previous
    training = config["training"]
    failures: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    started = time.perf_counter()
    for microbatch in _microbatch_candidates(training):
        try:
            result = _train_attempt(
                commit=preflight["git_sha"],
                train=train,
                validation_arrays=validation,
                plan=plan,
                loss_config=loss_config,
                device=device,
                microbatch=microbatch,
            )
            result["prior_oom_attempts"] = failures
            break
        except torch.cuda.OutOfMemoryError as exc:
            failures.append({"microbatch_size": microbatch, "error": str(exc)[:1000]})
            with (RUN_ROOT / "oom_attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failures[-1], ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()
            print(f"[R5-B] TFAS OOM at microbatch={microbatch}; retrying from latest complete state", flush=True)
    if result is None:
        _write_json(RUN_ROOT / "failed.json", {"oom_attempts": failures, "cpu_fallback": False})
        raise RuntimeError("R5-B TFAS OOMed at microbatch 4; stopped without CPU/AMP/effective-batch changes")
    record = {
        "experiment": "D2-A-R5-B",
        "git_sha": preflight["git_sha"],
        "protocol_version": 1.0,
        "training_seed": int(training["seed"]),
        "supervision": training["supervision"],
        "teacher_losses_used": False,
        "same_frozen_train_validation_ids_as_E003": True,
        "same_pair_plan_as_R5A": True,
        "target_updates": int(training["target_updates"]),
        "updates_completed": int(result["updates_completed"]),
        "effective_batch_size": int(training["effective_batch_size"]),
        "device": str(device),
        "precision": "FP32",
        "amp": False,
        "training_seconds_cumulative": float(result["training_seconds"]),
        "invocation_wall_seconds": float(time.perf_counter() - started),
        "cuda_peak_allocated_bytes": int(result["cuda_peak_allocated_bytes"]),
        "cuda_peak_reserved_bytes": int(result["cuda_peak_reserved_bytes"]),
        "microbatch_size": int(result["microbatch_size"]),
        "gradient_accumulation_steps": int(result["gradient_accumulation_steps"]),
        "parameter_count": int(result["parameter_count"]),
        "selected_update": int(result["selected_update"]),
        "selected_validation": result["selected_validation"],
        "arms": {"TFAS": result},
        "prior_oom_attempts": failures,
        "final_blind_generated": False,
        "distillation_run": False,
    }
    _write_json(record_path, record)
    print(f"[R5-B] TFAS training finished {result['updates_completed']}/12800 updates; "
          f"selected update={result['selected_update']}; wall={record['training_seconds_cumulative']:.1f}s", flush=True)
    return record


def _load_capacity_reference(label: str, arrays: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
    metrics = json.loads((CAPACITY_ROOT / "metrics.json").read_text(encoding="utf-8"))
    archive = np.load(CAPACITY_ROOT / "predictions.npz", allow_pickle=False)
    arm_metrics = metrics["arms"][label]
    predictions: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for split in ("development", "c3_h0_verification"):
        predictions[split] = {}
        hypotheses = ("h0", "h1") if split == "development" else ("h0",)
        for hypothesis in hypotheses:
            predictions[split][hypothesis] = {
                field: archive[f"{label}_{split}_{hypothesis}_{field}"]
                for field in ("score", "range_cell", "doppler_hz")
            }
    return {
        "threshold": float(arm_metrics["threshold_calibration"]["threshold"]),
        "predictions": predictions,
        "metrics": arm_metrics,
    }


def _candidate_comparison(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    train_arrays: Mapping[str, np.ndarray],
    e003_config: Mapping[str, Any],
    *,
    baseline_label: str,
) -> dict[str, Any]:
    raw = e003_runner._compare_arms(
        {"S0": baseline, "S2": candidate}, dict(arrays), dict(train_arrays), dict(e003_config)
    )
    components = dict(raw["G4_task_protection"]["components"])
    # The frozen Medium CNN already fails its C3 absolute-Pfa guard. That legacy
    # failure remains reported as the comparator status, while TFAS is tested
    # against the preregistered candidate-specific absolute limit.
    components.pop("s0_c3_h0_pfa_point", None)
    candidate_c3_pfa = float(candidate["metrics"]["c3_h0_verification"]["overall"]["pfa_ge"])
    absolute_pfa_pass = candidate_c3_pfa <= float(e003_config["gates"]["G4_task_protection"]["per_model_C3_pfa_point_max"])
    components["candidate_c3_h0_pfa_absolute"] = absolute_pfa_pass
    protection_passed = all(bool(value) for value in components.values())
    primary = raw["G4_primary_C2_C3_joint_gain"]
    basic = raw["G3_task_learning"]["S2"]
    return {
        "baseline": baseline_label,
        "candidate": "TFAS",
        "basic_learning": basic,
        "primary_C2_C3_joint_gain": primary,
        "task_protection": {
            "components": components,
            "passed": protection_passed,
            "baseline_c3_h0_pfa_ge": float(baseline["metrics"]["c3_h0_verification"]["overall"]["pfa_ge"]),
            "candidate_c3_h0_pfa_ge": candidate_c3_pfa,
            "baseline_absolute_pfa_guard_passed": float(baseline["metrics"]["c3_h0_verification"]["overall"]["pfa_ge"]) <= float(e003_config["gates"]["G4_task_protection"]["per_model_C3_pfa_point_max"]),
            "candidate_absolute_pfa_guard_passed": absolute_pfa_pass,
            "legacy_comparator_absolute_pfa_failure_does_not_erase_candidate_specific_check": True,
        },
        "candidate_retained": bool(basic["passed"] and primary["passed"] and protection_passed),
        "raw_inherited_comparison": raw,
    }


def _plot_rmse_by_snr(
    path: Path,
    metrics_by_model: Mapping[str, Mapping[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    snr_levels = sorted({
        float(label.split("/SNR", 1)[1].split("/", 1)[0])
        for metrics in metrics_by_model.values()
        for label in metrics["development"]["by_scenario_snr"]
    })
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for label, metrics in metrics_by_model.items():
        rows = metrics["development"]["by_scenario_snr"]
        groups: dict[float, list[Mapping[str, Any]]] = {snr: [] for snr in snr_levels}
        for key, values in rows.items():
            snr = float(key.split("/SNR", 1)[1].split("/", 1)[0])
            groups[snr].append(values)
        pooled_range_rmse = {
            snr: float(np.sqrt(sum(int(row["n_paired_scenes"]) * float(row["range_rmse_all_h1_cells"]) ** 2 for row in groups[snr])
                                  / sum(int(row["n_paired_scenes"]) for row in groups[snr])))
            for snr in snr_levels
        }
        pooled_doppler_rmse = {
            snr: float(np.sqrt(sum(int(row["n_paired_scenes"]) * float(row["doppler_rmse_all_h1_hz"]) ** 2 for row in groups[snr])
                                  / sum(int(row["n_paired_scenes"]) for row in groups[snr])))
            for snr in snr_levels
        }
        axes[0].plot(
            snr_levels,
            [pooled_range_rmse[snr] for snr in snr_levels],
            marker="o", label=label,
        )
        axes[1].plot(
            snr_levels,
            [pooled_doppler_rmse[snr] for snr in snr_levels],
            marker="o", label=label,
        )
    axes[0].set(xlabel="SNR (dB)", ylabel="Range RMSE (cells)", title="Development range RMSE")
    axes[1].set(xlabel="SNR (dB)", ylabel="Circular Doppler RMSE (Hz)", title="Development Doppler RMSE")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_evaluate() -> dict[str, Any]:
    preflight = _require_preflight()
    training_path = ARTIFACTS / "training.json"
    if not training_path.is_file():
        raise RuntimeError("R5-B CUDA training must finish before evaluation")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    commit = _require_frozen_main()
    if (
        training.get("git_sha") != preflight.get("git_sha")
        or training.get("git_sha") != commit
        or training.get("updates_completed") != 12_800
        or training.get("teacher_losses_used") is not False
        or training.get("final_blind_generated") is not False
        or training.get("distillation_run") is not False
    ):
        raise RuntimeError("R5-B training provenance or no-KD/no-blind declaration is invalid")
    config, r5_config, e003, physics = _load_configs()
    cache_root = ROOT / r5_config["data"]["cache_root"]
    arrays = {
        name: r5_runner._load_split(cache_root, name)
        for name in ("h0_calibration", "h0_verification", "c3_h0_verification", "development", "exact_overlap_boundary")
    }
    train_arrays = r5_runner._load_split(cache_root, "train")
    selected_h0_by_course, h0_rows = e003_runner._selected_course_indices(arrays["h0_verification"], e003)
    selected_h1_by_course, h1_rows = e003_runner._selected_course_indices(arrays["development"], e003)
    selected_h0 = np.concatenate([selected_h0_by_course[level] for level in ("C0", "C1", "C2", "C3")])
    selected_h1 = np.concatenate([selected_h1_by_course[level] for level in ("C0", "C1", "C2", "C3")])
    e003_runner._write_csv(
        ARTIFACTS / "selected_scene_ids.csv",
        [{"split": "h0_verification", **row} for row in h0_rows]
        + [{"split": "development", **row} for row in h1_rows],
    )
    checkpoint = torch.load(RUN_ROOT / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    if (
        checkpoint.get("experiment") != "D2-A-R5-B"
        or checkpoint.get("git_sha") != commit
        or checkpoint.get("model_name") != "TemporalFrequencyAttentionStudent"
        or int(checkpoint.get("seed", -1)) != 101
    ):
        raise RuntimeError("TFAS best checkpoint identity/source does not match current R5-B training")
    model = TemporalFrequencyAttentionStudent().to(_require_cuda()).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print("[R5-B] evaluating TFAS with frozen calibration, verification and development IDs", flush=True)
    arm_data, physical, roc_rows, result_rows, compute_rows = e003_runner._run_arm_evaluation(
        "TFAS", model, torch.device("cuda:0"), e003, arrays, selected_h0, selected_h1, physics
    )
    del model
    torch.cuda.empty_cache()
    references = {name: _load_capacity_reference(name, arrays) for name in ("Small", "Medium")}
    comparisons = {
        name: _candidate_comparison(arm_data, reference, arrays, train_arrays, e003, baseline_label=name)
        for name, reference in references.items()
    }
    retained = comparisons["Medium"]["candidate_retained"]
    metrics = {
        "experiment": "D2-A-R5-B",
        "git_sha": commit,
        "protocol_version": 1.0,
        "training_seed": 101,
        "selected_update": int(checkpoint["update"]),
        "selected_validation": checkpoint["validation"],
        "parameter_count": int(sum(value.numel() for value in checkpoint["model_state_dict"].values())),
        "model_metrics": arm_data["metrics"],
        "comparisons": comparisons,
        "primary_comparison": "TFAS_minus_Medium",
        "primary_candidate_retained": bool(retained),
        "physical_consistency": physical,
        "compute": compute_rows,
        "development_is_final_blind": False,
        "final_blind_generated": False,
        "teacher_losses_used": False,
        "distillation_run": False,
        "training_seed_uncertainty_estimated": False,
        "interpretation_scope": "one-seed pilot on reused E003 development and frozen static-clutter simulation only",
    }
    _write_json(ARTIFACTS / "metrics.json", metrics)
    _write_json(ARTIFACTS / "physical_diagnostics.json", physical)
    e003_runner._write_csv(ARTIFACTS / "result.csv", result_rows)
    e003_runner._write_csv(ARTIFACTS / "roc.csv", roc_rows)
    r5_runner._write_scene_predictions(ARTIFACTS / "scene_predictions.csv", arrays, {"TFAS": arm_data["predictions"]})
    r5_runner._save_prediction_archive(ARTIFACTS / "predictions.npz", {"TFAS": arm_data})
    _plot_rmse_by_snr(
        ARTIFACTS / "figures/rmse_by_snr.png",
        {"Small": references["Small"]["metrics"], "Medium": references["Medium"]["metrics"], "TFAS": arm_data["metrics"]},
    )
    print(
        f"[R5-B] G2 basic learning TFAS={comparisons['Medium']['basic_learning']['passed']}; "
        f"G3 candidate retained={retained}; C3 H0 Pfa={arm_data['metrics']['c3_h0_verification']['overall']['pfa_ge']:.4f}; "
        f"C2/C3 joint gain={comparisons['Medium']['primary_C2_C3_joint_gain']['point_difference_s2_minus_s0']:+.4f}",
        flush=True,
    )
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "g1", "train", "evaluate", "all"))
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        run_preflight()
    elif args.stage == "g1":
        run_g1()
    elif args.stage == "train":
        run_training()
    elif args.stage == "evaluate":
        run_evaluate()
    else:
        run_preflight()
        run_g1()
        run_training()
        run_evaluate()


if __name__ == "__main__":
    main()
