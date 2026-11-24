"""Bounded observable-scale score correction for D2-A-R6-I.

The module evaluates frozen R6-H students.  It contains no optimizer or
training path; scene generation is streamed and all student inference must use
the registered CUDA device.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from radardistill.d2a_baselines import run_b3
from radardistill.distillation.task_losses import decode_task_outputs
from radardistill.evaluation.d2a_r6a import build_split_plan, wrapped_error
from radardistill.evaluation.d2a_r6e import h0_metrics, h1_metrics
from radardistill.evaluation.d2a_r6f import _validate_model_outputs, build_r6f_split_plans
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_ssm import build_r6h_student
from radardistill.training.d2a_r6h import build_r6h_split_plans


ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-I"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"
ARMS = ("b3", "ce_tcn", "cnn_ssm_raw", "cnn_ssm_corrected")
STUDENT_ARMS = ("ce_tcn", "cnn_ssm")
SCORE_KEYS = {
    "b3": "score_b3",
    "ce_tcn": "score_ce_tcn",
    "cnn_ssm_raw": "score_cnn_ssm_raw",
    "cnn_ssm_corrected": "score_cnn_ssm_corrected",
}


def _require_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def robust_iq_power(iq: np.ndarray) -> np.ndarray:
    """Return the median observed |IQ|^2 per CPI over channel, pulse, and range."""
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[0] == 0 or not np.iscomplexobj(values):
        raise ValueError("iq must be a nonempty complex [B,C,N,R] array")
    if not np.isfinite(values.real).all() or not np.isfinite(values.imag).all():
        raise ValueError("iq must contain only finite values")
    power = np.median(np.abs(values) ** 2, axis=(1, 2, 3))
    if not np.isfinite(power).all() or np.any(power <= 0.0):
        raise ValueError("each CPI must have finite, positive robust IQ power")
    return np.asarray(power, dtype=np.float64)


def fit_tail_scale_slope(
    logits: np.ndarray,
    log_power: np.ndarray,
    *,
    bin_count: int = 20,
    quantile: float = 0.95,
    minimum_bin_count: int = 50,
) -> dict[str, Any]:
    """Fit one OLS slope to equal-count H0 right-tail summaries."""
    score = _require_finite_vector(logits, "logits")
    scale = _require_finite_vector(log_power, "log_power")
    if score.shape != scale.shape:
        raise ValueError("logits and log_power must have equal length")
    if not isinstance(bin_count, int) or bin_count < 2:
        raise ValueError("bin_count must be an integer >= 2")
    if not isinstance(minimum_bin_count, int) or minimum_bin_count <= 0:
        raise ValueError("minimum_bin_count must be positive")
    if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    order = np.argsort(scale, kind="stable")
    groups = np.array_split(order, bin_count)
    if any(len(group) < minimum_bin_count for group in groups):
        raise ValueError("one or more equal-count bins have fewer than the minimum samples")
    tail_x = np.asarray([np.median(scale[group]) for group in groups], dtype=np.float64)
    tail_y = np.asarray([
        np.quantile(score[group], quantile, method="higher") for group in groups
    ], dtype=np.float64)
    if not np.isfinite(tail_x).all() or not np.isfinite(tail_y).all():
        raise ValueError("tail summaries must be finite")
    if float(np.ptp(tail_x)) <= 1e-12:
        raise ValueError("log_power must vary across the equal-count bins")
    design = np.column_stack((tail_x, np.ones_like(tail_x)))
    beta, intercept = np.linalg.lstsq(design, tail_y, rcond=None)[0]
    center = float(np.median(scale))
    if not math.isfinite(float(beta)) or not math.isfinite(center):
        raise ValueError("fitted scale parameters must be finite")
    return {
        "beta": float(beta),
        "center": center,
        "intercept": float(intercept),
        "bin_count": int(bin_count),
        "minimum_bin_count": int(minimum_bin_count),
        "tail_quantile": float(quantile),
        "tail_quantile_method": "higher",
        "bin_counts": [int(len(group)) for group in groups],
        "tail_log_power_medians": tail_x.tolist(),
        "tail_logit_quantiles": tail_y.tolist(),
    }


def correct_presence_logit(
    logits: np.ndarray,
    log_power: np.ndarray,
    *,
    beta: float,
    center: float,
) -> np.ndarray:
    """Apply the single preregistered score correction, without labels/metadata."""
    score = np.asarray(logits, dtype=np.float64)
    scale = np.asarray(log_power, dtype=np.float64)
    if score.ndim != 1 or scale.ndim != 1 or score.shape != scale.shape:
        raise ValueError("logits and log_power must be equal-length vectors")
    if not np.isfinite(score).all() or not np.isfinite(scale).all():
        raise ValueError("logits and log_power must be finite")
    if not math.isfinite(beta) or not math.isfinite(center):
        raise ValueError("beta and center must be finite")
    return score - float(beta) * (scale - float(center))


def read_r6i_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected YAML mapping at {path}")
    return data


def r6i_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = read_r6i_yaml(STAGE / "config.yaml")
    conditions = read_r6i_yaml(ROOT / config["radar"]["condition_source"])
    physics = read_r6i_yaml(ROOT / config["radar"]["physics_source"])
    return config, conditions, physics


def build_r6i_split_plans(
    config: Mapping[str, Any], condition_config: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic split plans and the small, separate G1 smoke set."""
    radar = config["radar"]
    data = config["data"]
    conditions = condition_config["radar"]["conditions"]
    if len(conditions) != int(radar["condition_count"]):
        raise ValueError("registered R6-I condition count does not match R6-A")
    eval_data = {
        key: data[key]
        for key in ("condition_seed_stride", "chunk_size", "chunk_seed_stride", "condition_count")
    }
    eval_data["splits"] = dict(data["splits"])
    eval_config = {"data": eval_data, "radar": {"conditions": conditions}}
    plans = {
        name: build_split_plan(eval_config, name)
        for name in data["splits"]
    }
    g1_seed = int(data["g1_test_seed"])
    plans["g1_test"] = [
        {
            "split": "g1_test",
            "presence": presence,
            "condition_index": 0,
            "condition": conditions[0],
            "count": 8,
            "seed": g1_seed + index * int(data["chunk_seed_stride"]),
        }
        for index, presence in enumerate(("H0", "H1"))
    ]
    return plans


def _expand_plan_scene_ids(plans: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for split, rows in plans.items():
        ids: set[str] = set()
        for row in rows:
            seed = int(row["seed"])
            count = int(row["count"])
            if seed < 0 or count <= 0:
                raise ValueError("scene plan seeds and counts must be positive")
            ids.update(f"d2a-scene-{seed}-{index}" for index in range(count))
        expected = sum(int(row["count"]) for row in rows)
        if len(ids) != expected:
            raise ValueError(f"duplicate scene IDs within split {split}")
        result[str(split)] = ids
    return result


def audit_r6i_split_plans(
    plans: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Verify every registered sample and G1 scene ID is internally disjoint."""
    expanded = _expand_plan_scene_ids(plans)
    all_ids: set[str] = set()
    overlap_count = 0
    counts: dict[str, int] = {}
    condition_counts: dict[str, dict[str, int]] = {}
    for split, rows in plans.items():
        ids = expanded[str(split)]
        overlap_count += len(all_ids.intersection(ids))
        all_ids.update(ids)
        counts[str(split)] = len(ids)
        by_condition: dict[str, int] = {}
        for row in rows:
            key = str(int(row["condition_index"]))
            by_condition[key] = by_condition.get(key, 0) + int(row["count"])
        condition_counts[str(split)] = by_condition
    if overlap_count:
        raise ValueError("registered R6-I split/G1 scene IDs overlap")
    return {
        "scene_counts_by_split": counts,
        "condition_counts": condition_counts,
        "planned_scene_ids": int(sum(counts.values())),
        "overlap_count": int(overlap_count),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _require_cuda(config: Mapping[str, Any]) -> dict[str, Any]:
    resources = config["resources"]
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("R6-I requires CUDA; CPU fallback is forbidden")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if name != str(resources["required_gpu_name"]):
        raise RuntimeError(f"R6-I requires {resources['required_gpu_name']}, found {name}")
    free, total = torch.cuda.mem_get_info(0)
    minimum = int(resources["minimum_free_vram_bytes"])
    if free < minimum:
        raise RuntimeError(f"R6-I requires at least {minimum} free VRAM bytes, found {free}")
    return {
        "available": True,
        "device": "cuda:0",
        "name": name,
        "free_vram_bytes": int(free),
        "total_vram_bytes": int(total),
        "torch_cuda_version": torch.version.cuda,
    }


def _require_aisar_environment(config: Mapping[str, Any]) -> dict[str, str]:
    required = str(config["resources"]["required_environment"])
    active = os.environ.get("CONDA_DEFAULT_ENV", "")
    executable = str(Path(sys.executable).resolve())
    if active != required and f"/envs/{required}/" not in executable:
        raise RuntimeError(f"R6-I requires the {required} Conda environment, found {active or executable}")
    return {
        "required_environment": required,
        "conda_default_env": active,
        "python_executable": executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }


def _manifest_scene_ids() -> tuple[set[str], list[str]]:
    ids: set[str] = set()
    sources: list[str] = []
    experiments = ROOT / "experiments"
    for path in experiments.rglob("*.csv"):
        if path == STAGE or STAGE in path.parents:
            continue
        if "manifest" not in path.name.lower() and path.name != "selected_scene_ids.csv":
            continue
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                key = next((name for name in ("scene_id", "base_scene_id", "base_scene_ids") if name in fields), None)
                if key is None:
                    continue
                for row in reader:
                    value = row.get(key, "")
                    if value:
                        ids.update(item for item in str(value).split(";") if item)
            sources.append(str(path.relative_to(ROOT)))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise RuntimeError(f"failed to scan prior manifest {path}") from exc
    return ids, sources


def _registered_prior_scene_ids(
    config: Mapping[str, Any], condition_config: Mapping[str, Any]
) -> tuple[set[str], dict[str, int]]:
    conditions = condition_config["radar"]["conditions"]
    combined: set[str] = set()
    counts: dict[str, int] = {}
    for stage in config["data"]["prior_registered_plan_stages"]:
        stage_name = str(stage)
        path = ROOT / f"experiments/D2-A-R6/{stage_name}/config.yaml"
        prior_config = read_r6i_yaml(path)
        if stage_name == "R6-F":
            plans = build_r6f_split_plans(prior_config, conditions)
            plans["g1_test"] = [{
                "seed": int(prior_config["data"]["g1_test_seed"]),
                "count": int(prior_config["data"]["g1_test_count"]),
            }]
        else:
            plans = build_r6h_split_plans(prior_config, condition_config)
        expanded = _expand_plan_scene_ids(plans)
        stage_ids = set().union(*expanded.values()) if expanded else set()
        if len(stage_ids) != sum(len(value) for value in expanded.values()):
            raise RuntimeError(f"registered {stage_name} plan has cross-split scene overlap")
        counts[stage_name] = len(stage_ids)
        combined.update(stage_ids)
    return combined, counts


def _checkpoint_metadata(config: Mapping[str, Any], model_key: str) -> dict[str, Any]:
    model_spec = config["models"][model_key]
    checkpoint = ROOT / model_spec["checkpoint"]
    training_path = ROOT / model_spec["training_record"]
    owner_path = checkpoint.parent / "owner.json"
    if not checkpoint.is_file() or not training_path.is_file() or not owner_path.is_file():
        raise RuntimeError(f"R6-I fixed checkpoint files are incomplete for {model_key}")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    owner_file = json.loads(owner_path.read_text(encoding="utf-8"))
    if training.get("status") != "complete" or training.get("arm") != model_key:
        raise RuntimeError(f"R6-I training record is invalid for {model_key}")
    if int(training.get("updates", -1)) != 25600 or int(training.get("best_update", -1)) <= 0:
        raise RuntimeError(f"R6-I checkpoint does not match the frozen R6-H budget for {model_key}")
    if owner_file != training.get("owner"):
        raise RuntimeError(f"R6-I checkpoint owner record mismatch for {model_key}")
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_state.get("owner") != owner_file:
        raise RuntimeError(f"R6-I checkpoint payload owner mismatch for {model_key}")
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": _sha256(checkpoint),
        "checkpoint_bytes": int(checkpoint.stat().st_size),
        "training_record": str(training_path.relative_to(ROOT)),
        "training_record_sha256": _sha256(training_path),
        "owner": owner_file,
        "training_seed": int(training.get("owner", {}).get("training_seed", -1)),
        "updates": int(training.get("updates", -1)),
        "best_update": int(training.get("best_update", -1)),
        "parameter_count": int(training.get("model_parameter_count", -1)),
    }


def _project_source_paths() -> dict[str, Path]:
    return {
        "evaluation": Path(__file__).resolve(),
        "entrypoint": ROOT / "scripts/run_d2a_r6i.py",
        "tests": ROOT / "tests/test_d2a_r6i.py",
        "simulator": ROOT / "src/radardistill/simulation/d2a.py",
        "model": ROOT / "src/radardistill/students/r6_ssm.py",
        "decoder": ROOT / "src/radardistill/distillation/task_losses.py",
        "r6a_planner_metrics": ROOT / "src/radardistill/evaluation/d2a_r6a.py",
        "r6f_contract_planner": ROOT / "src/radardistill/evaluation/d2a_r6f.py",
        "r6h_registered_plans": ROOT / "src/radardistill/training/d2a_r6h.py",
        "r6d_paired_planner": ROOT / "src/radardistill/training/d2a_r6d_joint.py",
        "baseline": ROOT / "src/radardistill/d2a_baselines.py",
        "metric_helpers": ROOT / "src/radardistill/evaluation/d2a_r6e.py",
    }


def _tracked_worktree_clean() -> bool:
    return not _git("status", "--porcelain", "--untracked-files=no")


def _require_frozen_commit() -> dict[str, str]:
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if branch != "main" or head != remote:
        raise RuntimeError("R6-I G0 requires the frozen implementation at pushed main HEAD")
    if not _tracked_worktree_clean():
        raise RuntimeError("R6-I G0 requires a clean tracked worktree")
    required = (
        "experiments/D2-A-R6/R6-I/protocol.md",
        "experiments/D2-A-R6/R6-I/config.yaml",
        "src/radardistill/evaluation/d2a_r6i.py",
        "scripts/run_d2a_r6i.py",
        "tests/test_d2a_r6i.py",
        "docs/task_alignments/D2-A-R6-I_alignment.md",
    )
    for path in required:
        subprocess.check_call(["git", "ls-files", "--error-unmatch", path], cwd=ROOT,
                              stdout=subprocess.DEVNULL)
    return {"branch": branch, "git_sha": head, "origin_main_sha": remote}


def _write_scene_manifests(plans: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    for split, rows in plans.items():
        path = STAGE / f"scene_manifest_{split}.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(("scene_id", "split", "presence", "seed", "condition_index",
                             "course", "scenario_id", "snr_db", "cnr_db"))
            for row in rows:
                condition = row["condition"]
                for index in range(int(row["count"])):
                    writer.writerow((
                        f"d2a-scene-{int(row['seed'])}-{index}", split, row["presence"],
                        int(row["seed"]), int(row["condition_index"]), condition["course"],
                        condition["scenario_id"], condition["snr_db"], condition["cnr_db"],
                    ))
        os.replace(temporary, path)


def r6i_preflight() -> dict[str, Any]:
    """G0: freeze source/checkpoint identity and audit all registered scene IDs."""
    config, condition_config, _ = r6i_configs()
    if config["experiment"].get("id") != "D2-A-R6-I" or config["experiment"].get("status") != "preregistered":
        raise RuntimeError("R6-I protocol/config is not in its preregistered state")
    if config["models"].get("training_performed") is not False:
        raise RuntimeError("R6-I must not train or update student models")
    expected_conditions = int(config["radar"]["condition_count"])
    conditions = condition_config["radar"]["conditions"]
    expected_radar = {
        "channel_count": 3,
        "prf_hz": 1300.0,
        "pulse_count": 130,
        "range_bin_count": 32,
    }
    for name, expected in expected_radar.items():
        actual = config["radar"].get(name)
        if float(actual) != float(expected):
            raise RuntimeError(f"R6-I radar invariant {name} must equal {expected}, found {actual}")
    if len(conditions) != expected_conditions or expected_conditions != 12:
        raise RuntimeError("R6-I must preserve the frozen R6-A twelve-condition design")
    commit = _require_frozen_commit()
    plans = build_r6i_split_plans(config, condition_config)
    plan_audit = audit_r6i_split_plans(plans)
    expected_counts = {
        name: int(split["scene_count"]) for name, split in config["data"]["splits"].items()
    }
    expected_counts["g1_test"] = int(config["data"]["g1_test_count"])
    if plan_audit["scene_counts_by_split"] != expected_counts:
        raise RuntimeError("R6-I split plan counts differ from the registered config")
    for name, split in config["data"]["splits"].items():
        expected = {str(index): int(split["count_per_condition"]) for index in range(expected_conditions)}
        if plan_audit["condition_counts"][name] != expected:
            raise RuntimeError(f"R6-I split {name} does not have the registered balanced allocation")

    manifest_ids, manifest_paths = _manifest_scene_ids()
    registered_ids, registered_counts = _registered_prior_scene_ids(config, condition_config)
    planned_ids = set().union(*_expand_plan_scene_ids(plans).values())
    prior_ids = manifest_ids | registered_ids
    overlap = planned_ids & prior_ids
    if overlap:
        raise RuntimeError(f"R6-I scene plan overlaps {len(overlap)} prior registered/manifest IDs")

    disk_free = int(shutil.disk_usage(ROOT).free)
    minimum_disk = int(config["resources"]["minimum_free_disk_bytes"])
    if disk_free < minimum_disk:
        raise RuntimeError(f"R6-I requires at least {minimum_disk} free disk bytes, found {disk_free}")
    environment = _require_aisar_environment(config)
    gpu = _require_cuda(config)
    checkpoints = {
        name: _checkpoint_metadata(config, name)
        for name in STUDENT_ARMS
    }
    for name, details in checkpoints.items():
        if details["training_seed"] != 808 or details["updates"] != 25600:
            raise RuntimeError(f"R6-I {name} does not match the frozen R6-H seed/update record")

    protocol = ROOT / "experiments/D2-A-R6/R6-I/protocol.md"
    config_path = ROOT / "experiments/D2-A-R6/R6-I/config.yaml"
    source_paths = _project_source_paths()
    missing_source = [name for name, path in source_paths.items() if not path.is_file()]
    if missing_source:
        raise RuntimeError(f"R6-I source files are missing: {missing_source}")
    _write_scene_manifests(plans)
    scene_manifest_hashes = {
        split: _sha256(STAGE / f"scene_manifest_{split}.csv") for split in plans
    }
    result = {
        "status": "PASS",
        "phase": "G0",
        "utc_unix_time": time.time(),
        "commit": commit,
        "protocol_sha256": _sha256(protocol),
        "config_sha256": _sha256(config_path),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "checkpoints": checkpoints,
        "environment": environment,
        "gpu": gpu,
        "free_disk_bytes": disk_free,
        "minimum_free_disk_bytes": minimum_disk,
        "condition_count": expected_conditions,
        "plan_audit": plan_audit,
        "scene_manifest_sha256": scene_manifest_hashes,
        "historical_manifest_scene_ids": len(manifest_ids),
        "historical_manifest_files": manifest_paths,
        "registered_prior_scene_ids": len(registered_ids),
        "registered_prior_plan_scene_counts": registered_counts,
        "scene_overlap_count": 0,
        "final_blind_generated": False,
    }
    _write_json(STAGE / "preflight.json", result)
    return result


def _load_frozen_student(config: Mapping[str, Any], arm: str) -> torch.nn.Module:
    model_config = config["models"][arm]
    checkpoint = ROOT / model_config["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_r6h_student(arm, seed=808)
    model.load_state_dict(state["model_state_dict"], strict=True)
    training = json.loads((ROOT / model_config["training_record"]).read_text(encoding="utf-8"))
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != int(training.get("model_parameter_count", -1)):
        raise RuntimeError(f"R6-I {arm} model parameter count differs from its frozen training record")
    model.eval()
    return model


def _student_predict(
    model: torch.nn.Module,
    iq: np.ndarray,
    *,
    prf_hz: float,
    device: torch.device,
    microbatch: int,
    oom_fallbacks: Sequence[int],
) -> dict[str, np.ndarray]:
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("student IQ must have complex shape [B,3,130,32]")
    if not np.isfinite(values).all():
        raise ValueError("student IQ must be finite")
    inputs = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    candidates = [int(microbatch), *(int(value) for value in oom_fallbacks)]
    if any(value <= 0 for value in candidates):
        raise ValueError("microbatch values must be positive")
    last_oom: BaseException | None = None
    for current_microbatch in dict.fromkeys(candidates):
        logits: list[np.ndarray] = []
        ranges: list[np.ndarray] = []
        dopplers: list[np.ndarray] = []
        model.to(device).eval()
        tensor: torch.Tensor | None = None
        outputs: Mapping[str, torch.Tensor] | None = None
        decoded: Mapping[str, torch.Tensor] | None = None
        try:
            with torch.inference_mode():
                for start in range(0, len(inputs), current_microbatch):
                    tensor = torch.from_numpy(np.ascontiguousarray(
                        inputs[start:start + current_microbatch]
                    )).to(device)
                    outputs = model(tensor)
                    _validate_model_outputs(outputs, len(tensor))
                    decoded = decode_task_outputs(outputs, prf_hz)
                    logits.append(outputs["presence_logit"].detach().cpu().numpy().astype(np.float32))
                    ranges.append(decoded["range_cell"].detach().cpu().numpy().astype(np.float32))
                    dopplers.append(decoded["doppler_hz"].detach().cpu().numpy().astype(np.float32))
            return {
                "score": np.concatenate(logits),
                "range_cell": np.concatenate(ranges),
                "doppler_hz": np.concatenate(dopplers),
                "microbatch_used": np.asarray([current_microbatch], dtype=np.int16),
            }
        except torch.cuda.OutOfMemoryError as exc:
            last_oom = exc
            tensor = None
            outputs = None
            decoded = None
            torch.cuda.empty_cache()
        finally:
            model.to(torch.device("cpu"))
    if last_oom is not None:
        raise last_oom
    raise RuntimeError("student inference did not run")


def r6i_g1() -> dict[str, Any]:
    """G1: validate the frozen model outputs and scale arithmetic on CUDA."""
    config, _, physics = r6i_configs()
    preflight_path = STAGE / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("R6-I G1 requires a passing G0 record")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    commit = _git("rev-parse", "HEAD")
    if preflight.get("status") != "PASS" or preflight["commit"]["git_sha"] != commit:
        raise RuntimeError("R6-I G0 record does not match current frozen HEAD")
    environment = _require_aisar_environment(config)
    device_info = _require_cuda(config)
    device = torch.device("cuda:0")
    plans = build_r6i_split_plans(config, read_r6i_yaml(ROOT / config["radar"]["condition_source"]))
    scale_plan = plans["g1_test"]
    prf_hz = float(config["radar"]["prf_hz"])
    conditions: list[dict[str, Any]] = []
    for spec in scale_plan:
        generated = generate_d2a_paired_batch(
            int(spec["count"]), int(spec["seed"]),
            float(spec["condition"]["snr_db"]), float(spec["condition"]["cnr_db"]),
            physics, scenario_id=str(spec["condition"]["scenario_id"]),
        )
        iq = generated.iq_h0 if spec["presence"] == "H0" else generated.iq_h1
        expected_ids = [f"d2a-scene-{int(spec['seed'])}-{index}" for index in range(int(spec["count"]))]
        actual_ids = np.asarray(generated.base_scene_id).astype(str).tolist()
        if actual_ids != expected_ids:
            raise RuntimeError("R6-I G1 generated scene IDs do not match the registered plan")
        power = robust_iq_power(iq)
        for scale in (0.5, 2.0):
            scaled_power = robust_iq_power(scale * iq)
            relative = np.abs(scaled_power / power - scale**2) / scale**2
            if float(relative.max()) > 1e-5:
                raise RuntimeError("R6-I G1 robust power fails the common-amplitude scaling tolerance")
        classical = run_b3(iq, prf_hz, physics)
        if not np.isfinite(np.asarray(classical.score)).all():
            raise FloatingPointError("R6-I G1 B3 score is nonfinite")
        for arm in STUDENT_ARMS:
            metadata = _checkpoint_metadata(config, arm)
            if metadata["checkpoint_sha256"] != preflight["checkpoints"][arm]["checkpoint_sha256"]:
                raise RuntimeError(f"R6-I {arm} checkpoint changed after G0")
            model = _load_frozen_student(config, arm)
            prediction = _student_predict(
                model, iq, prf_hz=prf_hz, device=device,
                microbatch=int(config["resources"]["inference_microbatch"]),
                oom_fallbacks=config["resources"]["oom_microbatch_fallbacks"],
            )
            if any(not np.isfinite(prediction[key]).all() for key in ("score", "range_cell", "doppler_hz")):
                raise FloatingPointError(f"R6-I G1 {arm} output is nonfinite")
            conditions.append({
                "presence": spec["presence"],
                "arm": arm,
                "count": int(len(iq)),
                "microbatch_used": int(prediction["microbatch_used"][0]),
                "finite_outputs": True,
            })

    correction_probe = correct_presence_logit(
        np.asarray([0.2, -0.8]), np.asarray([-0.5, 1.0]), beta=0.3, center=0.1
    )
    expected_probe = np.asarray([0.2, -0.8]) - 0.3 * (np.asarray([-0.5, 1.0]) - 0.1)
    if not np.allclose(correction_probe, expected_probe, rtol=0.0, atol=1e-12):
        raise RuntimeError("R6-I G1 correction equation mismatch")
    result = {
        "status": "PASS",
        "phase": "G1",
        "git_sha": commit,
        "environment": environment,
        "gpu": device_info,
        "precision": "FP32",
        "models_resident_on_gpu": 1,
        "student_output_checks": conditions,
        "common_scale_relative_error_max": 1e-5,
        "correction_equation_check": "PASS",
        "final_blind_generated": False,
    }
    _write_json(STAGE / "g1.json", result)
    return result


def _load_prediction_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: payload[name] for name in payload.files}


def _save_prediction_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def _expected_ids_for_rows(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values = [
        f"d2a-scene-{int(row['seed'])}-{index}"
        for row in rows
        for index in range(int(row["count"]))
    ]
    return np.asarray(values, dtype="<U48")


def _generate_selected_iq(
    spec: Mapping[str, Any], physics: Mapping[str, Any]
) -> tuple[Any, np.ndarray, np.ndarray]:
    condition = spec["condition"]
    generated = generate_d2a_paired_batch(
        int(spec["count"]), int(spec["seed"]),
        float(condition["snr_db"]), float(condition["cnr_db"]),
        dict(physics), scenario_id=str(condition["scenario_id"]),
    )
    expected = [
        f"d2a-scene-{int(spec['seed'])}-{index}"
        for index in range(int(spec["count"]))
    ]
    actual = np.asarray(generated.base_scene_id).astype(str).tolist()
    if actual != expected:
        raise RuntimeError("generated scene IDs differ from the registered R6-I plan")
    return generated, generated.iq_h0, generated.iq_h1


def _b3_predict(iq: np.ndarray, physics: Mapping[str, Any], prf_hz: float) -> dict[str, np.ndarray]:
    output = run_b3(iq, prf_hz, dict(physics))
    return {
        "score": np.asarray(output.score, dtype=np.float64).reshape(-1),
        "range_cell": np.asarray(output.range_cell, dtype=np.float32).reshape(-1),
        "doppler_hz": np.asarray(output.doppler_hz, dtype=np.float32).reshape(-1),
    }


def _check_runtime_freeze(preflight: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    head = _git("rev-parse", "HEAD")
    if preflight.get("status") != "PASS" or preflight.get("commit", {}).get("git_sha") != head:
        raise RuntimeError("R6-I evaluation requires a passing G0 record for current HEAD")
    if not _tracked_worktree_clean():
        raise RuntimeError("tracked source changed after R6-I freeze")
    protocol = ROOT / "experiments/D2-A-R6/R6-I/protocol.md"
    config_path = ROOT / "experiments/D2-A-R6/R6-I/config.yaml"
    if _sha256(protocol) != preflight["protocol_sha256"] or _sha256(config_path) != preflight["config_sha256"]:
        raise RuntimeError("R6-I protocol/config changed after G0")
    for name, path in _project_source_paths().items():
        if _sha256(path) != preflight["source_sha256"][name]:
            raise RuntimeError(f"R6-I frozen source changed after G0: {name}")
    for split, expected_hash in preflight["scene_manifest_sha256"].items():
        manifest_path = STAGE / f"scene_manifest_{split}.csv"
        if not manifest_path.is_file() or _sha256(manifest_path) != expected_hash:
            raise RuntimeError(f"R6-I frozen scene manifest changed after G0: {split}")
    for arm in STUDENT_ARMS:
        current = _checkpoint_metadata(config, arm)
        if current["checkpoint_sha256"] != preflight["checkpoints"][arm]["checkpoint_sha256"]:
            raise RuntimeError(f"R6-I checkpoint changed after G0: {arm}")


def _threshold(scores: np.ndarray, target_pfa: float) -> float:
    values = _require_finite_vector(scores, "calibration_scores")
    if not math.isfinite(target_pfa) or not 0.0 < target_pfa < 1.0:
        raise ValueError("target_pfa must be between zero and one")
    return float(np.quantile(values, 1.0 - target_pfa, method="higher"))


def _record_chunks(
    path: Path, arrays: Sequence[Mapping[str, np.ndarray]], cursor: int, git_sha: str
) -> None:
    merged: dict[str, np.ndarray] = {}
    if arrays:
        keys = arrays[0].keys()
        merged = {key: np.concatenate([chunk[key] for chunk in arrays]) for key in keys}
    merged["_cursor"] = np.asarray([cursor], dtype=np.int64)
    merged["_git_sha"] = np.asarray([git_sha])
    _save_prediction_npz(path, merged)


def _process_eval_chunk(
    split: str,
    spec: Mapping[str, Any],
    physics: Mapping[str, Any],
    models: Mapping[str, torch.nn.Module],
    *,
    config: Mapping[str, Any],
    device: torch.device,
    beta: float | None,
    center: float | None,
) -> dict[str, np.ndarray]:
    generated, iq_h0, iq_h1 = _generate_selected_iq(spec, physics)
    is_h1 = split == "h1_development"
    iq = iq_h1 if is_h1 else iq_h0
    ids = np.asarray(generated.base_scene_id).astype("<U48")
    condition_index = np.full(len(ids), int(spec["condition_index"]), dtype=np.uint8)
    log_power = np.log(robust_iq_power(iq)).astype(np.float32)
    prf_hz = float(config["radar"]["prf_hz"])
    resources = config["resources"]
    microbatch = int(resources["inference_microbatch"])
    fallbacks = resources["oom_microbatch_fallbacks"]

    if split == "h0_scale_fit":
        prediction = _student_predict(
            models["cnn_ssm"], iq, prf_hz=prf_hz, device=device,
            microbatch=microbatch, oom_fallbacks=fallbacks,
        )
        return {
            "scene_id": ids,
            "condition_index": condition_index,
            "log_power": log_power,
            "cnn_ssm_raw_logit": prediction["score"],
            "inference_microbatch_cnn_ssm": np.full(
                len(ids), int(prediction["microbatch_used"][0]), dtype=np.uint8
            ),
        }

    b3 = _b3_predict(iq, physics, prf_hz)
    predictions: dict[str, dict[str, np.ndarray]] = {"b3": b3}
    for arm in STUDENT_ARMS:
        predictions[arm] = _student_predict(
            models[arm], iq, prf_hz=prf_hz, device=device,
            microbatch=microbatch, oom_fallbacks=fallbacks,
        )
    cnn_raw = predictions["cnn_ssm"]["score"]
    if beta is None or center is None:
        raise RuntimeError("R6-I correction parameters must be frozen before score evaluation")
    cnn_corrected = correct_presence_logit(cnn_raw, log_power, beta=beta, center=center).astype(np.float32)
    arrays: dict[str, np.ndarray] = {
        "scene_id": ids,
        "condition_index": condition_index,
        "log_power": log_power,
        "score_b3": predictions["b3"]["score"],
        "score_ce_tcn": predictions["ce_tcn"]["score"],
        "score_cnn_ssm_raw": cnn_raw,
        "score_cnn_ssm_corrected": cnn_corrected,
        "inference_microbatch_ce_tcn": np.full(
            len(ids), int(predictions["ce_tcn"]["microbatch_used"][0]), dtype=np.uint8
        ),
        "inference_microbatch_cnn_ssm": np.full(
            len(ids), int(predictions["cnn_ssm"]["microbatch_used"][0]), dtype=np.uint8
        ),
    }
    if is_h1:
        for output_arm in ("b3", "ce_tcn", "cnn_ssm_raw", "cnn_ssm_corrected"):
            model_arm = "cnn_ssm" if output_arm.startswith("cnn_ssm") else output_arm
            arrays[f"predicted_range_{output_arm}"] = predictions[model_arm]["range_cell"]
            arrays[f"predicted_doppler_{output_arm}"] = predictions[model_arm]["doppler_hz"]
        arrays["truth_range_cell"] = np.asarray(generated.truth_target_range_cell, dtype=np.float32)
        arrays["truth_doppler_hz"] = np.asarray(generated.truth_target_doppler_hz, dtype=np.float32)
    if any(not np.isfinite(value).all() for key, value in arrays.items() if key not in {"scene_id"}):
        raise FloatingPointError(f"R6-I {split} generated nonfinite predictions")
    return arrays


def _evaluate_split(
    split: str,
    rows: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    models: Mapping[str, torch.nn.Module],
    *,
    config: Mapping[str, Any],
    device: torch.device,
    beta: float | None,
    center: float | None,
    git_sha: str,
) -> dict[str, np.ndarray]:
    final_path = STAGE / f"predictions_{split}.npz"
    partial_path = STAGE / f"predictions_{split}.partial.npz"
    expected_ids = _expected_ids_for_rows(rows)
    if final_path.is_file():
        existing = _load_prediction_npz(final_path)
        if not np.array_equal(existing.get("scene_id"), expected_ids):
            raise RuntimeError(f"existing R6-I {split} predictions do not match the frozen manifest")
        if str(existing.get("_git_sha", np.asarray([""]))[0]) != git_sha:
            raise RuntimeError(f"existing R6-I {split} predictions were created by another commit")
        return {key: value for key, value in existing.items() if not key.startswith("_")}

    chunks: list[dict[str, np.ndarray]] = []
    cursor = 0
    if partial_path.is_file():
        partial = _load_prediction_npz(partial_path)
        cursor = int(partial.pop("_cursor")[0])
        partial_sha = str(partial.pop("_git_sha")[0])
        if partial_sha != git_sha:
            raise RuntimeError(f"R6-I {split} partial predictions were created by another commit")
        if not 0 <= cursor <= len(rows):
            raise RuntimeError(f"R6-I {split} partial cursor is invalid")
        if cursor:
            chunks.append(partial)
    save_every = 512
    progress_path = STAGE / f"progress_{split}.json"
    for index in range(cursor, len(rows)):
        current = _process_eval_chunk(
            split, rows[index], physics, models,
            config=config, device=device, beta=beta, center=center,
        )
        chunks.append(current)
        next_cursor = index + 1
        if next_cursor % save_every == 0 or next_cursor == len(rows):
            _record_chunks(partial_path, chunks, next_cursor, git_sha)
            _write_json(progress_path, {
                "split": split,
                "completed_chunks": next_cursor,
                "total_chunks": len(rows),
                "completed_scenes": int(sum(int(rows[i]["count"]) for i in range(next_cursor))),
                "total_scenes": int(sum(int(row["count"]) for row in rows)),
                "last_checkpoint_unix_time": time.time(),
            })
    combined = {key: np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]}
    if not np.array_equal(combined["scene_id"], expected_ids):
        raise RuntimeError(f"R6-I {split} generated order differs from the preregistered plan")
    combined["_git_sha"] = np.asarray([git_sha])
    _save_prediction_npz(final_path, combined)
    partial_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)
    return {key: value for key, value in combined.items() if not key.startswith("_")}


def _stratified_bootstrap_contrast(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    threshold_baseline: float,
    threshold_candidate: float,
    conditions: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    ids = np.asarray(baseline["scene_id"]).astype(str)
    if not np.array_equal(ids, np.asarray(candidate["scene_id"]).astype(str)):
        raise ValueError("R6-I paired bootstrap requires identical base_scene IDs and order")
    first_condition = np.asarray(baseline["condition_index"], dtype=np.int64)
    second_condition = np.asarray(candidate["condition_index"], dtype=np.int64)
    if not np.array_equal(first_condition, second_condition) or first_condition.shape != ids.shape:
        raise ValueError("R6-I paired bootstrap condition assignments differ")
    if replicates <= 0 or seed < 0 or not 0 < confidence < 1:
        raise ValueError("invalid R6-I paired bootstrap parameters")
    courses = np.asarray([str(row["course"]) for row in conditions])
    group_conditions = {
        "c3_pfa_difference": np.flatnonzero(courses == "C3"),
        "pd_difference": np.arange(len(conditions), dtype=np.int64),
        "c2_c3_joint_difference": np.flatnonzero(np.isin(courses, ("C2", "C3"))),
    }
    for group_name, indices in group_conditions.items():
        if len(indices) == 0 or any(not np.any(first_condition == index) for index in indices):
            raise ValueError(f"R6-I bootstrap group {group_name} is empty")

    first_detected = np.asarray(baseline["score"], dtype=np.float64) >= threshold_baseline
    second_detected = np.asarray(candidate["score"], dtype=np.float64) >= threshold_candidate
    first_range = np.asarray(baseline["predicted_range_cell"], dtype=np.float64)
    second_range = np.asarray(candidate["predicted_range_cell"], dtype=np.float64)
    truth_range = np.asarray(baseline["truth_range_cell"], dtype=np.float64)
    first_doppler = wrapped_error(
        baseline["predicted_doppler_hz"], baseline["truth_doppler_hz"], 1300.0
    )
    second_doppler = wrapped_error(
        candidate["predicted_doppler_hz"], candidate["truth_doppler_hz"], 1300.0
    )
    if any(len(value) != len(ids) for value in (first_range, second_range, truth_range, first_doppler, second_doppler)):
        raise ValueError("R6-I paired bootstrap parameter arrays are misaligned")
    first_joint = first_detected & (np.abs(first_range - truth_range) <= 1.0) & (np.abs(first_doppler) <= 20.0)
    second_joint = second_detected & (np.abs(second_range - truth_range) <= 1.0) & (np.abs(second_doppler) <= 20.0)
    values = {
        "c3_pfa_difference": np.empty(replicates, dtype=np.float64),
        "pd_difference": np.empty(replicates, dtype=np.float64),
        "c2_c3_joint_difference": np.empty(replicates, dtype=np.float64),
        "range_rmse_ratio": np.empty(replicates, dtype=np.float64),
        "doppler_rmse_ratio": np.empty(replicates, dtype=np.float64),
    }
    rng = np.random.default_rng(seed)
    grouped_rows = {
        int(condition): np.flatnonzero(first_condition == condition)
        for condition in np.unique(first_condition)
    }
    for replicate in range(replicates):
        sampled: dict[int, np.ndarray] = {}
        for condition in np.arange(len(conditions)):
            group = grouped_rows.get(int(condition))
            if group is not None and len(group):
                sampled[int(condition)] = group[rng.integers(0, len(group), size=len(group))]
        c3_indices = np.concatenate([sampled[int(index)] for index in group_conditions["c3_pfa_difference"]])
        all_indices = np.concatenate([sampled[int(index)] for index in group_conditions["pd_difference"]])
        c2c3_indices = np.concatenate([sampled[int(index)] for index in group_conditions["c2_c3_joint_difference"]])
        values["c3_pfa_difference"][replicate] = np.mean(
            second_detected[c3_indices].astype(np.float64) - first_detected[c3_indices].astype(np.float64)
        )
        values["pd_difference"][replicate] = np.mean(
            second_detected[all_indices].astype(np.float64) - first_detected[all_indices].astype(np.float64)
        )
        values["c2_c3_joint_difference"][replicate] = np.mean(
            second_joint[c2c3_indices].astype(np.float64) - first_joint[c2c3_indices].astype(np.float64)
        )
        values["range_rmse_ratio"][replicate] = math.sqrt(
            float(np.mean((second_range[all_indices] - truth_range[all_indices]) ** 2)) /
            max(float(np.mean((first_range[all_indices] - truth_range[all_indices]) ** 2)), 1e-24)
        )
        values["doppler_rmse_ratio"][replicate] = math.sqrt(
            float(np.mean(second_doppler[all_indices] ** 2)) /
            max(float(np.mean(first_doppler[all_indices] ** 2)), 1e-24)
        )
    tail = (1.0 - confidence) / 2.0
    result: dict[str, Any] = {
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "bootstrap_unit": "base_scene_id_stratified_by_condition",
        "contrasts": {},
    }
    # Point estimands use the same condition-balanced subsets as the bootstrap.
    c3_mask = np.isin(first_condition, group_conditions["c3_pfa_difference"])
    pd_mask = np.ones(len(ids), dtype=bool)
    c2c3_mask = np.isin(first_condition, group_conditions["c2_c3_joint_difference"])
    first_range_rmse = math.sqrt(float(np.mean((first_range - truth_range) ** 2)))
    first_doppler_rmse = math.sqrt(float(np.mean(first_doppler ** 2)))
    point_values = {
        "c3_pfa_difference": float(np.mean(second_detected[c3_mask]) - np.mean(first_detected[c3_mask])),
        "pd_difference": float(np.mean(second_detected[pd_mask]) - np.mean(first_detected[pd_mask])),
        "c2_c3_joint_difference": float(np.mean(second_joint[c2c3_mask]) - np.mean(first_joint[c2c3_mask])),
        "range_rmse_ratio": math.sqrt(float(np.mean((second_range - truth_range) ** 2)) / max(first_range_rmse**2, 1e-24)),
        "doppler_rmse_ratio": math.sqrt(float(np.mean(second_doppler**2)) / max(first_doppler_rmse**2, 1e-24)),
    }
    for name, samples in values.items():
        result["contrasts"][name] = {
            "estimate": point_values[name],
            "ci95": [float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))],
        }
    return result


def _paired_h0_pfa_bootstrap(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    threshold_baseline: float,
    threshold_candidate: float,
    conditions: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, Any]:
    ids = np.asarray(baseline["scene_id"]).astype(str)
    if not np.array_equal(ids, np.asarray(candidate["scene_id"]).astype(str)):
        raise ValueError("R6-I H0 bootstrap requires paired scene IDs in the same order")
    condition_index = np.asarray(baseline["condition_index"], dtype=np.int64)
    if not np.array_equal(condition_index, np.asarray(candidate["condition_index"], dtype=np.int64)):
        raise ValueError("R6-I H0 bootstrap requires matched conditions")
    courses = np.asarray([str(row["course"]) for row in conditions])
    c3_conditions = np.flatnonzero(courses == "C3")
    if len(c3_conditions) != 6:
        raise ValueError("R6-I H0 bootstrap requires the six registered C3 cells")
    if replicates <= 0 or seed < 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid R6-I H0 paired bootstrap parameters")
    first = np.asarray(baseline["score"], dtype=np.float64) >= threshold_baseline
    second = np.asarray(candidate["score"], dtype=np.float64) >= threshold_candidate
    groups = {int(condition): np.flatnonzero(condition_index == condition) for condition in c3_conditions}
    if any(not len(group) for group in groups.values()):
        raise ValueError("R6-I C3 bootstrap has an empty condition")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draws = [
            group[rng.integers(0, len(group), size=len(group))]
            for group in groups.values()
        ]
        indices = np.concatenate(draws)
        samples[replicate] = np.mean(
            second[indices].astype(np.float64) - first[indices].astype(np.float64)
        )
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(second[np.isin(condition_index, c3_conditions)]) -
                          np.mean(first[np.isin(condition_index, c3_conditions)])),
        "ci95": [float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))],
        "replicates": int(replicates),
        "seed": int(seed),
        "confidence": float(confidence),
        "unit": "base_scene_id_stratified_by_C3_condition",
    }


def _metric_view(
    predictions: Mapping[str, np.ndarray], arm: str, *, is_h1: bool
) -> dict[str, np.ndarray]:
    result = {
        "scene_id": np.asarray(predictions["scene_id"]),
        "condition_index": np.asarray(predictions["condition_index"]),
        "score": np.asarray(predictions[SCORE_KEYS[arm]]),
    }
    if is_h1:
        result.update({
            "predicted_range_cell": np.asarray(predictions[f"predicted_range_{arm}"]),
            "predicted_doppler_hz": np.asarray(predictions[f"predicted_doppler_{arm}"]),
            "truth_range_cell": np.asarray(predictions["truth_range_cell"]),
            "truth_doppler_hz": np.asarray(predictions["truth_doppler_hz"]),
        })
    return result


def _score_correlation_diagnostics(fit: Mapping[str, Any]) -> dict[str, Any]:
    x = np.asarray(fit["tail_log_power_medians"], dtype=np.float64)
    y = np.asarray(fit["tail_logit_quantiles"], dtype=np.float64)
    beta = float(fit["beta"])
    intercept = float(fit["intercept"])
    residual = y - (beta * x + intercept)
    total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = None if total <= 0 else float(1.0 - np.sum(residual**2) / total)
    return {
        "binned_tail_logit_ols_r_squared": r_squared,
        "tail_logit_change_per_unit_log_power": beta,
        "tail_logit_change_per_doubling_power": beta * math.log(2.0),
        "fit_split_scene_count": int(sum(fit["bin_counts"])),
    }


def _observable_scale_summary(
    datasets: Mapping[str, Mapping[str, np.ndarray]],
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for split, data in datasets.items():
        index = np.asarray(data["condition_index"], dtype=np.int64)
        values = np.asarray(data["log_power"], dtype=np.float64)
        if index.shape != values.shape or not np.isfinite(values).all():
            raise ValueError(f"R6-I {split} observable-scale values are invalid")
        by_condition: dict[str, Any] = {}
        for condition_index, condition in enumerate(conditions):
            selected = values[index == condition_index]
            if not len(selected):
                raise ValueError(f"R6-I {split} lacks condition {condition_index}")
            by_condition[str(condition_index)] = {
                "condition": dict(condition),
                "n": int(len(selected)),
                "log_power_median": float(np.median(selected)),
                "log_power_q05": float(np.quantile(selected, 0.05)),
                "log_power_q95": float(np.quantile(selected, 0.95)),
            }
        summaries[split] = {
            "n": int(len(values)),
            "log_power_median": float(np.median(values)),
            "by_condition": by_condition,
        }
    return summaries


def _compute_metrics(
    config: Mapping[str, Any],
    condition_config: Mapping[str, Any],
    fit: Mapping[str, Any],
    fit_observations: Mapping[str, np.ndarray],
    calibration: Mapping[str, np.ndarray],
    verification: Mapping[str, np.ndarray],
    development: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    conditions = condition_config["radar"]["conditions"]
    workpoints = [float(config["evaluation"]["primary_workpoint"]), *map(float, config["evaluation"]["secondary_workpoints"])]
    result: dict[str, Any] = {
        "status": "evaluated",
        "experiment_id": "D2-A-R6-I",
        "protocol_version": str(config["experiment"]["protocol_version"]),
        "git_sha": _git("rev-parse", "HEAD"),
        "training_performed": False,
        "final_blind_generated": False,
        "split_counts": {
            "h0_scale_fit": int(len(fit_observations["scene_id"])),
            "h0_threshold_calibration": int(len(calibration["scene_id"])),
            "h0_verification": int(len(verification["scene_id"])),
            "h1_development": int(len(development["scene_id"])),
        },
        "score_scale_fit": dict(fit),
        "score_scale_diagnostics": _score_correlation_diagnostics(fit),
        "observable_log_power_by_split_and_condition": _observable_scale_summary(
            {
                "h0_scale_fit": fit_observations,
                "h0_threshold_calibration": calibration,
                "h0_verification": verification,
                "h1_development": development,
            },
            conditions,
        ),
        "workpoints": {},
    }
    thresholds_by_workpoint: dict[str, dict[str, float]] = {}
    for target in workpoints:
        label = f"{target:g}"
        systems: dict[str, Any] = {}
        thresholds: dict[str, float] = {}
        for arm in ARMS:
            cal_view = _metric_view(calibration, arm, is_h1=False)
            ver_view = _metric_view(verification, arm, is_h1=False)
            threshold = _threshold(cal_view["score"], target)
            thresholds[arm] = threshold
            h0_result = h0_metrics(
                cal_view, ver_view, threshold=threshold, conditions=conditions
            )
            dev_view = _metric_view(development, arm, is_h1=True)
            h1_result = h1_metrics(
                dev_view, threshold=threshold, conditions=conditions,
                h0_verification=ver_view, prf_hz=float(config["radar"]["prf_hz"]),
            )
            systems[arm] = {"threshold": threshold, "h0": h0_result, "h1": h1_result}
        thresholds_by_workpoint[label] = thresholds
        result["workpoints"][label] = {"target_pfa": target, "systems": systems}

    primary_key = f"{float(config['evaluation']['primary_workpoint']):g}"
    primary_thresholds = thresholds_by_workpoint[primary_key]
    raw_cal = _metric_view(calibration, "cnn_ssm_raw", is_h1=False)
    corr_cal = _metric_view(calibration, "cnn_ssm_corrected", is_h1=False)
    raw_ver = _metric_view(verification, "cnn_ssm_raw", is_h1=False)
    corr_ver = _metric_view(verification, "cnn_ssm_corrected", is_h1=False)
    h0_contrast = _paired_h0_pfa_bootstrap(
        raw_ver, corr_ver,
        threshold_baseline=primary_thresholds["cnn_ssm_raw"],
        threshold_candidate=primary_thresholds["cnn_ssm_corrected"],
        conditions=conditions,
        replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
        seed=int(config["evaluation"]["paired_bootstrap_seed"]),
        confidence=float(config["evaluation"]["paired_ci_level"]),
    )
    h1_raw = _metric_view(development, "cnn_ssm_raw", is_h1=True)
    h1_corr = _metric_view(development, "cnn_ssm_corrected", is_h1=True)
    task_contrast = _stratified_bootstrap_contrast(
        h1_raw, h1_corr,
        threshold_baseline=primary_thresholds["cnn_ssm_raw"],
        threshold_candidate=primary_thresholds["cnn_ssm_corrected"],
        conditions=conditions,
        replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
        seed=int(config["evaluation"]["paired_bootstrap_seed"]) + 1,
        confidence=float(config["evaluation"]["paired_ci_level"]),
    )
    result["confirmatory_paired_contrast_corrected_minus_raw"] = {
        "c3_pfa": h0_contrast,
        "h1_task": task_contrast,
        "primary_workpoint": float(config["evaluation"]["primary_workpoint"]),
    }
    corrected_h0 = result["workpoints"][primary_key]["systems"]["cnn_ssm_corrected"]["h0"]
    c3_cells_pass = bool(corrected_h0["c3_six_cells_familywise_pass"])
    global_pfa_pass = bool(
        corrected_h0["verification"]["pfa_ci95_exact_upper"]
        <= float(config["evaluation"]["primary_pfa_global_upper_maximum"])
    )
    c3_delta = h0_contrast["estimate"]
    c3_delta_ci_upper = h0_contrast["ci95"][1]
    task_contrasts = task_contrast["contrasts"]
    paired_pfa_pass = bool(
        c3_delta <= -float(config["evaluation"]["required_c3_pooled_pfa_reduction_vs_raw"])
        and c3_delta_ci_upper < 0.0
    )
    pd_pass = bool(
        task_contrasts["pd_difference"]["ci95"][0]
        >= float(config["evaluation"]["pd_noninferiority_difference_lower_minimum"])
    )
    joint_pass = bool(
        task_contrasts["c2_c3_joint_difference"]["ci95"][0]
        >= float(config["evaluation"]["joint_noninferiority_difference_lower_minimum"])
    )
    range_pass = bool(
        task_contrasts["range_rmse_ratio"]["ci95"][1]
        <= float(config["evaluation"]["parameter_rmse_ratio_upper_maximum"])
    )
    doppler_pass = bool(
        task_contrasts["doppler_rmse_ratio"]["ci95"][1]
        <= float(config["evaluation"]["parameter_rmse_ratio_upper_maximum"])
    )
    gates = {
        "overall_h0_upper_bound": global_pfa_pass,
        "c3_six_cell_familywise_upper_bounds": c3_cells_pass,
        "paired_c3_pfa_reduction": paired_pfa_pass,
        "paired_pd_noninferiority": pd_pass,
        "paired_c2_c3_joint_noninferiority": joint_pass,
        "paired_range_rmse_noninferiority": range_pass,
        "paired_doppler_rmse_noninferiority": doppler_pass,
    }
    result["p7_readmission"] = {
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "all_gates_pass": bool(all(gates.values())),
        "interpretation": (
            "may propose a separately preregistered P7 multi-seed protocol"
            if all(gates.values()) else
            "stop this score correction; do not generate final blind"
        ),
    }
    return result


def _run_scale_intervention(
    config: Mapping[str, Any],
    condition_config: Mapping[str, Any],
    physics: Mapping[str, Any],
    models: Mapping[str, torch.nn.Module],
    *,
    device: torch.device,
    beta: float,
    center: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    conditions = condition_config["radar"]["conditions"]
    matching = [
        index for index, row in enumerate(conditions)
        if row["course"] == "C3" and row["scenario_id"] == "partial_overlap"
        and float(row["snr_db"]) == 0.0 and float(row["cnr_db"]) == 20.0
    ]
    if len(matching) != 1:
        raise RuntimeError("R6-I registered scale-intervention condition is not unique")
    condition_index = matching[0]
    plans = build_r6i_split_plans(config, condition_config)["h1_development"]
    intervention_rows = [row for row in plans if int(row["condition_index"]) == condition_index]
    factors = (1.0, *map(float, config["evaluation"]["scale_intervention"]))
    collected: dict[str, dict[str, dict[float, dict[str, list[np.ndarray]]]]] = {}
    for presence in ("H0", "H1"):
        collected[presence] = {}
        for arm in ARMS:
            collected[presence][arm] = {
                factor: {key: [] for key in ("score", "range_cell", "doppler_hz", "log_power", "power_ratio_error")}
                for factor in factors
            }
    prf_hz = float(config["radar"]["prf_hz"])
    for spec in intervention_rows:
        generated, iq_h0, iq_h1 = _generate_selected_iq(spec, physics)
        del generated
        for presence, base_iq in (("H0", iq_h0), ("H1", iq_h1)):
            base_power = robust_iq_power(base_iq)
            for factor in factors:
                current_iq = base_iq if factor == 1.0 else factor * base_iq
                power = robust_iq_power(current_iq)
                log_power = np.log(power)
                relative_error = np.zeros_like(power) if factor == 1.0 else (
                    np.abs(power / base_power - factor**2) / factor**2
                )
                classical = _b3_predict(current_iq, physics, prf_hz)
                predictions: dict[str, dict[str, np.ndarray]] = {"b3": classical}
                for arm in STUDENT_ARMS:
                    predictions[arm] = _student_predict(
                        models[arm], current_iq,
                        prf_hz=prf_hz, device=device,
                        microbatch=int(config["resources"]["inference_microbatch"]),
                        oom_fallbacks=config["resources"]["oom_microbatch_fallbacks"],
                    )
                corrected = correct_presence_logit(
                    predictions["cnn_ssm"]["score"], log_power, beta=beta, center=center
                )
                for arm in ARMS:
                    source = "cnn_ssm" if arm.startswith("cnn_ssm") else arm
                    score = corrected if arm == "cnn_ssm_corrected" else predictions[source]["score"]
                    for key, value in (
                        ("score", score),
                        ("range_cell", predictions[source]["range_cell"]),
                        ("doppler_hz", predictions[source]["doppler_hz"]),
                        ("log_power", log_power),
                        ("power_ratio_error", relative_error),
                    ):
                        collected[presence][arm][factor][key].append(np.asarray(value))

    summary: dict[str, Any] = {
        "condition_index": int(condition_index),
        "condition": dict(conditions[condition_index]),
        "factors": {},
        "diagnostic_only": True,
        "thresholds_from_primary_h0_calibration": dict(thresholds),
    }
    for presence in ("H0", "H1"):
        summary["factors"][presence] = {}
        for arm in ARMS:
            merged = {
                factor: {
                    key: np.concatenate(parts)
                    for key, parts in collected[presence][arm][factor].items()
                }
                for factor in factors
            }
            base = merged[1.0]
            base_decision = base["score"] >= float(thresholds[arm])
            summary["factors"][presence][arm] = {}
            for factor in factors:
                current = merged[factor]
                decision = current["score"] >= float(thresholds[arm])
                entry: dict[str, Any] = {
                    "n": int(len(current["score"])),
                    "score_mean": float(np.mean(current["score"])),
                    "fixed_threshold_event_rate": float(np.mean(decision)),
                    "decision_consistency_vs_unscaled": float(np.mean(decision == base_decision)),
                    "mean_log_power": float(np.mean(current["log_power"])),
                }
                if factor == 1.0:
                    entry["power_ratio_relative_error_max"] = 0.0
                    entry["range_abs_change_mean_cells"] = 0.0
                    entry["doppler_abs_change_mean_hz"] = 0.0
                else:
                    entry["power_ratio_relative_error_max"] = float(np.max(current["power_ratio_error"]))
                    range_change = np.abs(current["range_cell"] - base["range_cell"])
                    doppler_change = np.abs(wrapped_error(current["doppler_hz"], base["doppler_hz"], prf_hz))
                    entry["range_abs_change_mean_cells"] = float(np.mean(range_change))
                    entry["range_abs_change_p95_cells"] = float(np.quantile(range_change, 0.95))
                    entry["doppler_abs_change_mean_hz"] = float(np.mean(doppler_change))
                    entry["doppler_abs_change_p95_hz"] = float(np.quantile(doppler_change, 0.95))
                summary["factors"][presence][arm][str(factor)] = entry
    return summary


def _pfa_series(metrics: Mapping[str, Any], workpoint: str) -> dict[str, list[float | None]]:
    result: dict[str, list[float | None]] = {}
    systems = metrics["workpoints"][workpoint]["systems"]
    for arm in ARMS:
        result[arm] = [
            systems[arm]["h0"]["verification_by_condition"][str(index)]["pfa"]
            for index in range(12)
        ]
    return result


def r6i_analyze() -> dict[str, Any]:
    """Write concise analysis and plots from the one frozen metrics package."""
    metrics_path = STAGE / "metrics.json"
    intervention_path = STAGE / "scale_intervention.json"
    if not metrics_path.is_file() or not intervention_path.is_file():
        raise RuntimeError("R6-I analyze requires metrics and scale-intervention results")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
    config, condition_config, _ = r6i_configs()
    primary = f"{float(config['evaluation']['primary_workpoint']):g}"
    systems = metrics["workpoints"][primary]["systems"]
    paired = metrics["confirmatory_paired_contrast_corrected_minus_raw"]
    contrasts = paired["h1_task"]["contrasts"]
    pfa_delta = paired["c3_pfa"]
    gate_lines = [
        f"- `{name}`：{'PASS' if passed else 'FAIL'}"
        for name, passed in metrics["p7_readmission"]["gates"].items()
    ]
    lines = [
        "# D2-A-R6-I 分析",
        "",
        f"**状态：** {metrics['p7_readmission']['status']}  ",
        f"**提交：** `{metrics['git_sha']}`  ",
        f"**设备：** {metrics['device']['name']} / {metrics['device']['device']}，FP32  ",
        "**训练：** 未训练；模型固定为R6-H seed808最佳检查点。**final blind：** 未生成。",
        "",
        "## 研究问题与方法",
        "",
        "本阶段检验CNN-SSM H0高分尾部是否随每CPI可观测稳健IQ功率系统变化，以及一条仅由独立H0拟合的右尾斜率校正能否改善C3虚警并保护检测/参数任务。校正仅使用原始presence logit与IQ-derived log-power；H0拟合、阈值校准、H0验证和H1开发场景分开。CE-TCN、raw/corrected CNN-SSM和B3使用相同生成场景。",
        "",
        "## H0独立验证（主工作点）",
        "",
        "| 系统 | 全局Pfa | 单侧exact上界 | C3 pooled Pfa | C3六格familywise门 | 最差C3格Pfa |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for arm in ARMS:
        h0 = systems[arm]["h0"]
        worst_c3 = max(
            entry["pfa"] for entry in h0["verification_by_condition"].values()
            if entry["condition"]["course"] == "C3"
        )
        lines.append(
            f"| {arm} | {h0['verification']['pfa']:.5f} | {h0['verification']['pfa_ci95_exact_upper']:.5f} "
            f"| {h0['c3_pooled']['pfa']:.5f} | {'PASS' if h0['c3_six_cells_familywise_pass'] else 'FAIL'} | {worst_c3:.5f} |"
        )
    lines.extend([
        "",
        "## H1开发任务指标（主工作点）",
        "",
        "| 系统 | Pd | C2+C3 joint | 全H1 Range RMSE (cell) | 全H1 Doppler RMSE (Hz) |",
        "|---|---:|---:|---:|---:|",
    ])
    for arm in ARMS:
        h1 = systems[arm]["h1"]
        all_group = h1["by_group"]["all"]
        c2c3 = h1["by_group"]["C2_C3"]
        lines.append(
            f"| {arm} | {h1['pd']:.5f} | {c2c3['joint_success']:.5f} "
            f"| {all_group['all_h1']['range_rmse_cells']:.5f} "
            f"| {all_group['all_h1']['doppler_rmse_hz']:.5f} |"
        )
    lines.extend([
        "",
        "## Corrected CNN-SSM − raw CNN-SSM 配对差",
        "",
        "| 指标 | 点估计 | 95%配对bootstrap区间 | 冻结门 |",
        "|---|---:|---:|---|",
        f"| C3 pooled Pfa差 | {pfa_delta['estimate']:+.5f} | [{pfa_delta['ci95'][0]:+.5f}, {pfa_delta['ci95'][1]:+.5f}] | 至少下降0.01且上界<0 |",
        f"| Overall Pd差 | {contrasts['pd_difference']['estimate']:+.5f} | [{contrasts['pd_difference']['ci95'][0]:+.5f}, {contrasts['pd_difference']['ci95'][1]:+.5f}] | 下界≥−0.02 |",
        f"| C2+C3 joint差 | {contrasts['c2_c3_joint_difference']['estimate']:+.5f} | [{contrasts['c2_c3_joint_difference']['ci95'][0]:+.5f}, {contrasts['c2_c3_joint_difference']['ci95'][1]:+.5f}] | 下界≥−0.02 |",
        f"| Range RMSE比 | {contrasts['range_rmse_ratio']['estimate']:.5f} | [{contrasts['range_rmse_ratio']['ci95'][0]:.5f}, {contrasts['range_rmse_ratio']['ci95'][1]:.5f}] | 上界≤1.05 |",
        f"| Doppler RMSE比 | {contrasts['doppler_rmse_ratio']['estimate']:.5f} | [{contrasts['doppler_rmse_ratio']['ci95'][0]:.5f}, {contrasts['doppler_rmse_ratio']['ci95'][1]:.5f}] | 上界≤1.05 |",
        "",
        "## H0右尾与可观测尺度关系",
        "",
        f"H0-only 20等频数桶拟合得到斜率 `beta={metrics['score_scale_fit']['beta']:.6g}`，中心 `x={metrics['score_scale_fit']['center']:.6g}`；分桶右尾OLS R²为 `{metrics['score_scale_diagnostics']['binned_tail_logit_ols_r_squared']:.5f}`。斜率表示log-power每增加1时，H0 logit 95%分位约变化 `{metrics['score_scale_diagnostics']['tail_logit_change_per_unit_log_power']:.5f}`。这描述本仿真/固定模型，不代表普适校准关系。",
        "",
        "## 公共幅度缩放诊断",
        "",
        "对C3 partial-overlap / SNR=0 dB / CNR=20 dB的512个H1开发配对场景及同场景H0，报告固定0.05阈值下 `aX` 与 `X` 的决策一致率、事件率、参数输出变化和IQ功率缩放误差。此项为受控诊断，不表示真实硬件AGC。详细数据见 `scale_intervention.json`。",
        "",
        "## P7再准入门",
        "",
        *gate_lines,
        "",
        f"冻结决定：**{metrics['p7_readmission']['status']}**。{metrics['p7_readmission']['interpretation']}。即使通过，也须另行冻结P7多训练种子协议；本结果不授权final blind。",
        "",
        "## 12格分层结果",
        "",
        "| 条件索引 | 课程 | 场景 | SNR (dB) | CNR (dB) | 校正CNN-SSM Pfa | familywise上界 | 校正CNN-SSM Pd | joint success |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    corrected = systems["cnn_ssm_corrected"]
    condition_rows = condition_config["radar"]["conditions"]
    for index, condition in enumerate(condition_rows):
        h0 = corrected["h0"]["verification_by_condition"][str(index)]
        h1 = corrected["h1"]["by_condition"][str(index)]
        upper = h0.get("pfa_familywise_exact_upper", h0["pfa_ci95_exact_upper"])
        lines.append(
            f"| {index} | {condition['course']} | {condition['scenario_id']} | {condition['snr_db']:g} | {condition['cnr_db']:g} "
            f"| {h0['pfa']:.5f} | {upper:.5f} | {h1['pd']:.5f} | {h1['joint_success']:.5f} |"
        )
    lines.extend([
        "",
        "## 解释边界与下一步",
        "",
        "这是固定seed808检查点的单次新场景开发评价。阈值由独立H0 calibration设定；H0 verification用于虚警门，H1 development已用于本次研究判断，不能称为盲测。若校正未通过冻结门，停止本校正，不在这些数据上搜索其它公式；后续是否调整学生训练、重新研究教师教学或结束该路线，按R6阶段总目标和冻结停止规则决策。",
        "",
    ])
    scale_summary = metrics["observable_log_power_by_split_and_condition"]["h0_verification"]["by_condition"]
    scale_table = [
        "### 独立H0验证的可观测log-power分布",
        "",
        "| 条件索引 | 课程 | 场景 | log-power中位数 | Q05 | Q95 |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for index, condition in enumerate(condition_config["radar"]["conditions"]):
        summary = scale_summary[str(index)]
        scale_table.append(
            f"| {index} | {condition['course']} | {condition['scenario_id']} | "
            f"{summary['log_power_median']:.5f} | {summary['log_power_q05']:.5f} | {summary['log_power_q95']:.5f} |"
        )
    insert_at = lines.index("## 公共幅度缩放诊断")
    lines[insert_at:insert_at] = scale_table + [""]
    scale_lines = [
        "### 固定阈值缩放结果",
        "",
        "| 存在性 | 系统 | a | 事件率 | 与未缩放决策一致率 | Range变化均值 (cell) | Doppler变化均值 (Hz) | 功率比例误差最大值 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for presence in ("H0", "H1"):
        for arm in ARMS:
            for factor in config["evaluation"]["scale_intervention"]:
                entry = intervention["factors"][presence][arm][str(float(factor))]
                scale_lines.append(
                    f"| {presence} | {arm} | {float(factor):g} | {entry['fixed_threshold_event_rate']:.5f} "
                    f"| {entry['decision_consistency_vs_unscaled']:.5f} | {entry['range_abs_change_mean_cells']:.5f} "
                    f"| {entry['doppler_abs_change_mean_hz']:.5f} | {entry['power_ratio_relative_error_max']:.3e} |"
                )
    insert_at = lines.index("## P7再准入门")
    lines[insert_at:insert_at] = scale_lines + [""]
    analysis_path = STAGE / "analysis.md"
    analysis_path.write_text("\n".join(lines), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "written_without_plots", "analysis": str(analysis_path.relative_to(ROOT))}
    pfa = _pfa_series(metrics, primary)
    fig, axis = plt.subplots(figsize=(9.5, 4.8))
    x = np.arange(12)
    for arm in ARMS:
        axis.plot(x, pfa[arm], marker="o", label=arm)
    axis.axhline(0.10, color="black", linestyle="--", linewidth=1, label="C3 limit")
    axis.set(xlabel="R6-A condition index", ylabel="H0 verification Pfa", title="R6-I H0 Pfa by condition (primary workpoint)")
    axis.set_xticks(x)
    axis.grid(True, alpha=0.3)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(STAGE / "pfa_by_condition.png", dpi=150)
    plt.close(fig)

    tail_x = np.asarray(metrics["score_scale_fit"]["tail_log_power_medians"], dtype=float)
    tail_y = np.asarray(metrics["score_scale_fit"]["tail_logit_quantiles"], dtype=float)
    beta = float(metrics["score_scale_fit"]["beta"])
    intercept = float(metrics["score_scale_fit"]["intercept"])
    fig, axis = plt.subplots(figsize=(7.0, 4.6))
    axis.scatter(tail_x, tail_y, s=24, label="H0 bin 95th percentile")
    axis.plot(tail_x, beta * tail_x + intercept, color="tab:red", label="fitted OLS line")
    axis.set(xlabel="Median log IQ power", ylabel="CNN-SSM presence logit", title="H0 right-tail scale fit")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(STAGE / "score_scale_fit.png", dpi=150)
    plt.close(fig)
    report_path = ROOT / "docs/stage_reviews/D2-A-R6I_review.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "# D2-A-R6-I 阶段审查\n\n"
        f"**阶段判断：** {metrics['p7_readmission']['status']}  \n"
        f"**代码提交：** `{metrics['git_sha']}`  \n"
        "**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.3（本阶段未修改）。\n\n"
        "## 工程与实验科学性\n\n"
        "使用AISAR环境和RTX 3050 Laptop `cuda:0`对冻结R6-H模型作FP32推理；模型顺序驻留，未训练或生成final blind。新H0拟合、阈值校准、H0验证和H1开发scene IDs分离，完整结果和逐格指标见[`R6-I分析`](../../experiments/D2-A-R6/R6-I/analysis.md)及[`metrics`](../../experiments/D2-A-R6/R6-I/metrics.json)。\n\n"
        "## 阶段结论与主线影响\n\n"
        f"冻结P7再准入结果为 **{metrics['p7_readmission']['status']}**。{metrics['p7_readmission']['interpretation']}。本结果只涉及固定CNN-SSM检测分数，不证明任何知识蒸馏收益，也不把整体尺度诊断推广到真实硬件。R6-I没有生成盲测。\n\n"
        "## 下一步\n\n"
        "保持研究总纲和R6-H历史结果不变。依照`analysis.md`中的冻结门，决定是否另行冻结P7多seed确认；若R6-I校正失败则停止该校正，不在同一开发数据上继续搜索。整个R5-C/R6授权路线完成后，再编写唯一的ChatGPT综合审查请求。\n",
        encoding="utf-8",
    )
    return {
        "status": "written",
        "analysis": str(analysis_path.relative_to(ROOT)),
        "stage_review": str(report_path.relative_to(ROOT)),
        "plots": ["pfa_by_condition.png", "score_scale_fit.png"],
    }


def r6i_evaluate() -> dict[str, Any]:
    """Run the one frozen development evaluation on the registered independent splits."""
    config, condition_config, physics = r6i_configs()
    preflight_path = STAGE / "preflight.json"
    g1_path = STAGE / "g1.json"
    if not preflight_path.is_file() or not g1_path.is_file():
        raise RuntimeError("R6-I evaluation requires passing G0 and G1 records")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    g1 = json.loads(g1_path.read_text(encoding="utf-8"))
    if g1.get("status") != "PASS" or g1.get("git_sha") != _git("rev-parse", "HEAD"):
        raise RuntimeError("R6-I evaluation requires passing G1 for current HEAD")
    _check_runtime_freeze(preflight, config)
    environment = _require_aisar_environment(config)
    device_info = _require_cuda(config)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    evaluation_started = time.perf_counter()
    plans = build_r6i_split_plans(config, condition_config)
    models = {
        arm: _load_frozen_student(config, arm)
        for arm in STUDENT_ARMS
    }
    fit_predictions = _evaluate_split(
        "h0_scale_fit", plans["h0_scale_fit"], physics, models,
        config=config, device=device, beta=None, center=None,
        git_sha=preflight["commit"]["git_sha"],
    )
    fit = fit_tail_scale_slope(
        fit_predictions["cnn_ssm_raw_logit"], fit_predictions["log_power"],
        bin_count=int(config["score_correction"]["equal_count_bins"]),
        quantile=float(config["score_correction"]["tail_quantile"]),
        minimum_bin_count=int(config["score_correction"]["minimum_bin_count"]),
    )
    fit["fitting_split"] = "h0_scale_fit"
    fit["fitting_presence"] = "H0_only"
    fit["fitting_scenes"] = int(len(fit_predictions["scene_id"]))
    fit["fitting_scene_id_sha256"] = hashlib.sha256(
        "\n".join(np.asarray(fit_predictions["scene_id"]).astype(str)).encode("utf-8")
    ).hexdigest()
    fit_path = STAGE / "fit.json"
    _write_json(fit_path, fit)
    beta = float(fit["beta"])
    center = float(fit["center"])
    calibration = _evaluate_split(
        "h0_threshold_calibration", plans["h0_threshold_calibration"], physics, models,
        config=config, device=device, beta=beta, center=center,
        git_sha=preflight["commit"]["git_sha"],
    )
    verification = _evaluate_split(
        "h0_verification", plans["h0_verification"], physics, models,
        config=config, device=device, beta=beta, center=center,
        git_sha=preflight["commit"]["git_sha"],
    )
    development = _evaluate_split(
        "h1_development", plans["h1_development"], physics, models,
        config=config, device=device, beta=beta, center=center,
        git_sha=preflight["commit"]["git_sha"],
    )
    metrics = _compute_metrics(
        config, condition_config, fit, fit_predictions,
        calibration, verification, development,
    )
    primary_key = f"{float(config['evaluation']['primary_workpoint']):g}"
    primary_thresholds = {
        arm: float(metrics["workpoints"][primary_key]["systems"][arm]["threshold"])
        for arm in ARMS
    }
    intervention = _run_scale_intervention(
        config, condition_config, physics, models,
        device=device, beta=beta, center=center, thresholds=primary_thresholds,
    )
    _write_json(STAGE / "scale_intervention.json", intervention)
    metrics.update({
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "source_sha256": preflight["source_sha256"],
        "checkpoint_sha256": {
            name: preflight["checkpoints"][name]["checkpoint_sha256"] for name in STUDENT_ARMS
        },
        "device": device_info,
        "environment": environment,
        "precision": "FP32",
        "evaluation_wall_seconds": float(time.perf_counter() - evaluation_started),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "inference_microbatch_start": int(config["resources"]["inference_microbatch"]),
        "inference_microbatch_used": {
            split: {
                key.removeprefix("inference_microbatch_"): sorted(
                    int(value) for value in np.unique(arrays[key])
                )
                for key in arrays if key.startswith("inference_microbatch_")
            }
            for split, arrays in {
                "h0_scale_fit": fit_predictions,
                "h0_threshold_calibration": calibration,
                "h0_verification": verification,
                "h1_development": development,
            }.items()
        },
        "iq_persisted": False,
        "scale_intervention": "scale_intervention.json",
    })
    _write_json(STAGE / "metrics.json", metrics)
    return metrics
