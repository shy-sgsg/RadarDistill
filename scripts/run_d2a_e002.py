#!/usr/bin/env python3
"""Run the preregistered D2-A-E002 curriculum and K2 pilot in stages."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
sys.path.insert(0, str(ROOT / "src"))

from radardistill.d2a_baselines import run_b0, run_b1, run_b2, run_b3
from radardistill.d2a_r3_baselines import run_t1_global_normalized
from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_e002 import (
    calibrate_student_threshold,
    e0_energy_score,
    evaluate_detection_scores,
    evaluate_paired_predictions,
    exact_binomial_upper,
    paired_difference_interval,
    summarize_h0_verification,
)
from radardistill.distillation.task_losses import decode_task_outputs
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.task_student import TaskStudent
from radardistill.teachers.d2a_k2 import audit_k2_cache, summarize_k2_target_windows
from radardistill.training.d2a_e002 import (
    PROTOCOL_VERSION,
    require_cuda_device,
    run_micro_overfit_sanity,
    train_d2a_e002_pilot,
    validate_e002_config,
)
from radardistill.training.d2a_e002_data import (
    check_cache_disk_space,
    materialize_e002_splits,
    validate_disjoint_scene_ids,
)


METHODS = ("E0", "B0", "B1", "B2", "B3", "T1", "A", "B", "C")
FROZEN_INPUTS = (
    "experiments/D2-A-E001/config.yaml",
    "experiments/D2-A-R3/config.yaml",
    "experiments/D2-A-E002/config.yaml",
    "experiments/D2-A-E002/protocol.md",
    "scripts/run_d2a_e002.py",
    "src/radardistill/d2a_baselines.py",
    "src/radardistill/d2a_r3_baselines.py",
    "src/radardistill/distillation/d2a_e002_losses.py",
    "src/radardistill/distillation/task_losses.py",
    "src/radardistill/evaluation/d2a_e001.py",
    "src/radardistill/evaluation/d2a_e002.py",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/students/task_student.py",
    "src/radardistill/teachers/d2a_k2.py",
    "src/radardistill/training/d2a_e002.py",
    "src/radardistill/training/d2a_e002_data.py",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return value if not isinstance(value, float) or np.isfinite(value) else None
    if isinstance(value, (bool, str)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _require_frozen_commit() -> str:
    branch = _git("branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"E002 preflight requires the authorized main branch, found {branch!r}")
    for item in FROZEN_INPUTS:
        _git("ls-files", "--error-unmatch", item)
        changed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", item], cwd=ROOT
        ).returncode
        if changed != 0:
            raise RuntimeError(f"preregistered E002 input has uncommitted changes: {item}")
    head = _git("rev-parse", "HEAD")
    remote = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True, capture_output=True
    )
    if remote.returncode != 0:
        raise RuntimeError("E002 preflight could not read origin/main; commit and push before running")
    if remote.stdout.strip() != head:
        raise RuntimeError("E002 preflight requires the committed protocol/code to be pushed to origin/main")
    return head


def _nvidia_smi() -> dict[str, Any]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader",
        ],
        text=True,
        capture_output=True,
    )
    return {
        "returncode": int(query.returncode),
        "query": query.stdout.strip(),
        "stderr": query.stderr.strip(),
        "full": subprocess.run(
            ["nvidia-smi"], text=True, capture_output=True
        ).stdout.strip()
        if query.returncode == 0
        else None,
    }


def _environment_record(*, git_sha: str, disk_free_bytes: int) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu = None
    memory = None
    if cuda_available:
        gpu = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        memory = {"free_bytes": int(free), "total_bytes": int(total)}
    return {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_sha": git_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": bool(cuda_available),
        "cuda_device_count": int(torch.cuda.device_count()),
        "gpu_name": gpu,
        "cuda_memory": memory,
        "disk_free_bytes": int(disk_free_bytes),
        "nvidia_smi": _nvidia_smi(),
    }


def _preflight(config_path: Path) -> dict[str, Any]:
    config = _read_yaml(config_path)
    validate_e002_config(config)
    if float(config["experiment"]["protocol_version"]) != PROTOCOL_VERSION:
        raise RuntimeError("E002 protocol/config version mismatch")
    git_sha = _require_frozen_commit()
    device = require_cuda_device("cuda:0")
    disk = check_cache_disk_space(config["data"]["train"]["cache_root"], config, reserve_gib=3.0)
    physics_config = _read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    if int(physics_config["radar"]["pulse_count"]) != 130 or float(physics_config["radar"]["prf_hz"]) != 1300.0:
        raise RuntimeError("E002 physics config no longer matches PRF=1300 Hz, N=130")
    root = ROOT / "experiments/D2-A-E002"
    (root / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    environment = _environment_record(git_sha=git_sha, disk_free_bytes=int(disk["free_bytes_before_generation"]))
    _write_json(root / "environment.json", environment)
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "git_sha": git_sha,
        "branch": "main",
        "pushed_to_origin_main": True,
        "cuda_device": str(device),
        "gpu_name": environment["gpu_name"],
        "disk": disk,
        "estimated_cache_gib": disk["cache_bytes_estimated"] / 1024**3,
        "environment_record": "experiments/D2-A-E002/environment.json",
        "passed": True,
    }
    _write_json(root / "preflight.json", result)
    return result


def _require_preflight() -> dict[str, Any]:
    root = ROOT / "experiments/D2-A-E002"
    path = root / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run --stage preflight before generating data or training")
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("passed") or result.get("git_sha") != _require_frozen_commit():
        raise RuntimeError("preflight no longer matches the committed E002 protocol/code")
    return result


def _generate_data(config_path: Path) -> dict[str, Any]:
    preflight = _require_preflight()
    config = _read_yaml(config_path)
    physics = _read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    started = time.perf_counter()
    before_free_bytes = int(shutil.disk_usage(ROOT).free)
    splits = materialize_e002_splits(config["data"]["train"]["cache_root"], config, physics)
    validate_disjoint_scene_ids(splits)
    elapsed = time.perf_counter() - started
    root = ROOT / "experiments/D2-A-E002"
    cache_root = ROOT / config["data"]["train"]["cache_root"]
    cache_bytes = sum(path.stat().st_size for path in cache_root.rglob("*.npy"))
    result = {
        "git_sha": preflight["git_sha"],
        "split_counts": {name: int(len(arrays["base_scene_id"])) for name, arrays in splits.items()},
        "split_ids_disjoint": True,
        "cache_bytes_estimated": int(preflight["disk"]["cache_bytes_estimated"]),
        "cache_bytes_actual_npy": int(cache_bytes),
        "disk_free_bytes_before": before_free_bytes,
        "disk_free_bytes_after": int(os.statvfs(cache_root).f_bavail * os.statvfs(cache_root).f_frsize),
        "generation_seconds": float(elapsed),
        "data_location": str(cache_root.relative_to(ROOT)),
        "data_is_gitignored": True,
    }
    _write_json(root / "data_generation.json", result)
    return result


def _run_k2_audit() -> dict[str, Any]:
    preflight = _require_preflight()
    config = _read_yaml(ROOT / "experiments/D2-A-E002/config.yaml")
    physics = _read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    physics["teacher"] = config["teacher"]
    cache = ROOT / config["data"]["train"]["cache_root"] / Path("teacher_audit")
    arrays = {path.stem: np.load(path, mmap_mode="r") for path in cache.glob("*.npy")}
    audit = audit_k2_cache(
        arrays,
        physics,
        bootstrap_replicates=int(config["statistics"]["development_bootstrap_replicates"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
        chunk_size=int(config["data"]["train"]["generation_chunk_size"]),
    )
    audit["git_sha"] = preflight["git_sha"]
    root = ROOT / "experiments/D2-A-E002"
    _write_json(root / "k2_audit.json", audit)
    c3 = np.asarray(arrays["course_level"]).astype(str) == "C3"
    scene_summary = summarize_k2_target_windows(
        arrays["teacher_response_h0"][c3],
        arrays["teacher_response"][c3],
        arrays["truth_range_cell"][c3],
        arrays["truth_doppler_hz"][c3],
        arrays["base_scene_id"][c3],
        prf_hz=float(config["radar"]["prf_hz"]),
        bootstrap_replicates=int(config["statistics"]["development_bootstrap_replicates"]),
        bootstrap_seed=int(config["statistics"]["bootstrap_seed"]),
    )
    with (root / "k2_audit_scenes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("base_scene_id", "scenario_id", "snr_db", "target_mass_h0", "target_mass_h1", "paired_mass_lift", "teacher_top_cell_hit"))
        for local_index, index in enumerate(np.flatnonzero(c3)):
            writer.writerow((
                str(arrays["base_scene_id"][index]),
                str(arrays["scenario_id"][index]),
                float(arrays["snr_db"][index]),
                float(scene_summary["per_scene_h0_mass"][local_index]),
                float(scene_summary["per_scene_h1_mass"][local_index]),
                float(scene_summary["per_scene_mass_lift"][local_index]),
                bool(scene_summary["per_scene_top_cell_hit"][local_index]),
            ))
    return audit


def _run_sanity_and_pilot() -> dict[str, Any]:
    preflight = _require_preflight()
    config = _read_yaml(ROOT / "experiments/D2-A-E002/config.yaml")
    device = require_cuda_device("cuda:0")
    root = ROOT / "experiments/D2-A-E002"
    cache_root = ROOT / config["data"]["train"]["cache_root"]
    audit_path = root / "k2_audit.json"
    if not audit_path.is_file():
        raise RuntimeError("complete the independent K2 audit before pilot training")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    train = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "train").glob("*.npy")}
    validation = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "validation").glob("*.npy")}
    if not audit.get("allow_C_arm", False):
        train.pop("teacher_response", None)
    sanity = run_micro_overfit_sanity(train, config, device=device)
    sanity["git_sha"] = preflight["git_sha"]
    _write_json(root / "micro_overfit_sanity.json", sanity)
    if not sanity["passed"]:
        raise RuntimeError("G1 micro-overfit/gradient sanity failed; A/B/C formal pilot was not started")
    arms = ("A", "B", "C") if audit.get("allow_C_arm", False) else ("A", "B")
    nvidia_before = _nvidia_smi()
    pilot = train_d2a_e002_pilot(
        train,
        validation,
        config,
        root / "runs",
        device_name="cuda:0",
        arms=arms,
    )
    torch.cuda.synchronize(device)
    pilot["git_sha"] = preflight["git_sha"]
    pilot["arms_run"] = list(arms)
    pilot["micro_overfit_sanity"] = sanity
    pilot["nvidia_smi_before"] = nvidia_before
    pilot["nvidia_smi_after"] = _nvidia_smi()
    _write_json(root / "pilot.json", pilot)
    return pilot


def _one_dimensional(value: Any, dtype: Any = np.float64) -> np.ndarray:
    return np.atleast_1d(np.asarray(value, dtype=dtype)).reshape(-1)


def _collect_classical(
    iq: np.ndarray,
    physics: dict[str, Any],
    r3_options: dict[str, Any],
    *,
    batch_size: int = 16,
) -> dict[str, dict[str, np.ndarray | None]]:
    """Stream classical detection and parameter outputs for one IQ split."""
    count = len(iq)
    pieces: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"score": [], "range_cell": [], "doppler_hz": []}
        for method in ("E0", "B0", "B1", "B2", "B3", "T1")
    }
    prf_hz = float(physics["radar"]["prf_hz"])
    for start in range(0, count, batch_size):
        current = np.asarray(iq[start : start + batch_size])
        raw_outputs = {
            "B0": run_b0(current, prf_hz),
            "B1": run_b1(current, prf_hz),
            "B2": run_b2(current, prf_hz, physics),
            "B3": run_b3(current, prf_hz, physics),
            "T1": run_t1_global_normalized(current, prf_hz, physics, r3_options),
        }
        pieces["E0"]["score"].append(e0_energy_score(current))
        for method, output in raw_outputs.items():
            pieces[method]["score"].append(_one_dimensional(output.score))
            pieces[method]["range_cell"].append(_one_dimensional(output.range_cell))
            pieces[method]["doppler_hz"].append(_one_dimensional(output.doppler_hz))
    results: dict[str, dict[str, np.ndarray | None]] = {}
    for method, method_parts in pieces.items():
        results[method] = {
            "score": np.concatenate(method_parts["score"]),
            "range_cell": None if method == "E0" else np.concatenate(method_parts["range_cell"]),
            "doppler_hz": None if method == "E0" else np.concatenate(method_parts["doppler_hz"]),
        }
    return results


def _predict_student_scenes(
    iq: np.ndarray,
    model: TaskStudent,
    device: torch.device,
    *,
    batch_size: int = 16,
) -> dict[str, np.ndarray]:
    """Move one inference batch at a time to the required CUDA device."""
    count = len(iq)
    scores: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    dopplers: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            current = np.asarray(iq[start : start + batch_size], dtype=np.complex64)
            planes = np.concatenate((current.real, current.imag), axis=1).astype(np.float32, copy=False)
            outputs = model(torch.from_numpy(np.ascontiguousarray(planes)).to(device), compute_spectrum=False)
            decoded = decode_task_outputs(outputs, prf_hz=1300.0)
            scores.append(outputs["presence_logit"].cpu().numpy())
            ranges.append(decoded["range_cell"].cpu().numpy())
            dopplers.append(decoded["doppler_hz"].cpu().numpy())
    return {
        "score": np.concatenate(scores),
        "range_cell": np.concatenate(ranges),
        "doppler_hz": np.concatenate(dopplers),
    }


def _load_inference_model(checkpoint_path: Path, config: dict[str, Any], device: torch.device) -> TaskStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("has_k2_head") is not False:
        raise ValueError(f"inference checkpoint contains an unexpected K2 head: {checkpoint_path}")
    model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model.to(device)


def _amplitude_quartiles(amplitude: np.ndarray) -> np.ndarray:
    values = np.asarray(amplitude, dtype=np.float64)
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    return np.digitize(values, edges, right=True).astype(np.int8) + 1


def _h0_groups(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    amplitude_q = _amplitude_quartiles(arrays["truth_amplitude_diagnostic"])
    course = np.asarray(arrays["course_level"]).astype(str)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    cnr = np.asarray(arrays["requested_cnr_db"], dtype=np.float64)
    return {
        "course_level": course,
        "snr_db": np.asarray([f"{value:g}" for value in snr]),
        "requested_cnr_db": np.asarray([f"{value:g}" for value in cnr]),
        "amplitude_quartile": np.asarray([f"Q{value}" for value in amplitude_q]),
        "course_level_x_snr_db": np.asarray([f"{c}/SNR{v:g}" for c, v in zip(course, snr, strict=True)]),
    }


def _verification_group_rows(
    scores_by_method: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, Any]],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = _h0_groups(arrays)
    for method, scores in scores_by_method.items():
        threshold = float(thresholds[method]["threshold"])
        for group_name, labels in groups.items():
            for label in np.unique(labels):
                selected = labels == label
                false_alarms = int(np.sum(scores[selected] >= threshold))
                total = int(np.sum(selected))
                rows.append({
                    "method": method,
                    "group": group_name,
                    "label": str(label),
                    "n": total,
                    "false_alarm_count": false_alarms,
                    "pfa_ge": false_alarms / total,
                    "pfa_ci95_exact_upper": exact_binomial_upper(false_alarms, total),
                    "threshold": threshold,
                    "tie_count": int(np.sum(scores[selected] == threshold)),
                })
    return rows


def _development_group_rows(
    method: str,
    h0_scores: np.ndarray,
    h1_scores: np.ndarray,
    ranges: np.ndarray | None,
    dopplers: np.ndarray | None,
    threshold: float,
    arrays: dict[str, np.ndarray],
    prf_hz: float,
    range_cell_m: float,
) -> list[dict[str, Any]]:
    truth_range = np.asarray(arrays["truth_range_cell"], dtype=np.float64)
    truth_doppler = np.asarray(arrays["truth_doppler_hz"], dtype=np.float64)
    labels: dict[str, np.ndarray] = {
        "course_level": np.asarray(arrays["course_level"]).astype(str),
        "scenario": np.asarray(arrays["scenario_id"]).astype(str),
        "snr_db": np.asarray([f"{value:g}" for value in arrays["snr_db"]]),
        "requested_cnr_db": np.asarray([f"{value:g}" for value in arrays["requested_cnr_db"]]),
        "amplitude_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(arrays["truth_amplitude"])]),
        "range_truth_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(truth_range)]),
        "doppler_truth_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(truth_doppler)]),
        "phase_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(np.asarray(arrays["truth_phase_rad"]))]),
        "measured_scnr_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(arrays["measured_scnr_db"])]),
        "measured_snr_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(arrays["measured_snr_db"])]),
        "measured_cnr_quartile": np.asarray([f"Q{x}" for x in _amplitude_quartiles(arrays["measured_cnr_db"])]),
    }
    rows: list[dict[str, Any]] = []
    detected = h1_scores >= threshold
    for group_name, group_labels in labels.items():
        for label in np.unique(group_labels):
            selected = group_labels == label
            if not np.any(selected):
                continue
            range_error = None if ranges is None else np.asarray(ranges)[selected] - truth_range[selected]
            doppler_error = None
            if dopplers is not None:
                raw_error = np.asarray(dopplers)[selected] - truth_doppler[selected]
                doppler_error = (raw_error + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
            row: dict[str, Any] = {
                "method": method,
                "group": group_name,
                "label": str(label),
                "n": int(np.sum(selected)),
                "pd": float(np.mean(detected[selected])),
                "auc": float(empirical_auc(h0_scores[selected], h1_scores[selected])),
                "joint_success": None,
                "range_rmse_cells": None,
                "range_rmse_m": None,
                "doppler_rmse_hz": None,
            }
            if range_error is not None and doppler_error is not None:
                row.update({
                    "range_rmse_cells": float(np.sqrt(np.mean(range_error**2))),
                    "range_rmse_m": float(np.sqrt(np.mean(range_error**2)) * range_cell_m),
                    "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error**2))),
                    "joint_success": float(np.mean(
                        detected[selected]
                        & (np.abs(range_error) <= 1.0)
                        & (np.abs(doppler_error) <= 20.0)
                    )),
                })
            rows.append(row)
    return rows


def _circular_mean_hz(values: np.ndarray, prf_hz: float) -> float:
    angles = np.asarray(values, dtype=np.float64) * (2.0 * math.pi / prf_hz)
    angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    return float((angle * prf_hz / (2.0 * math.pi) + prf_hz / 2.0) % prf_hz - prf_hz / 2.0)


def _paired_auc_difference_ci(
    h0_a: np.ndarray,
    h1_a: np.ndarray,
    h0_b: np.ndarray,
    h1_b: np.ndarray,
    ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    values = [np.asarray(item, dtype=np.float64).reshape(-1) for item in (h0_a, h1_a, h0_b, h1_b)]
    clusters = np.asarray(ids).reshape(-1)
    if any(item.size != clusters.size for item in values):
        raise ValueError("paired AUC inputs must align to shared scene IDs")
    unique = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    differences = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([members[cluster] for cluster in sampled])
        differences[index] = empirical_auc(values[0][selected], values[1][selected]) - empirical_auc(
            values[2][selected], values[3][selected]
        )
    return [float(value) for value in np.quantile(differences, [0.025, 0.975])]


def _one_sided_cluster_lower(
    values: np.ndarray,
    ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    clusters = np.asarray(ids).reshape(-1)
    unique = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in unique}
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        selected = np.concatenate([members[cluster] for cluster in sampled])
        estimates[index] = float(np.mean(values[selected]))
    return float(np.quantile(estimates, 0.05))


def _g5_gate_summary(course_signal: dict[str, Any]) -> dict[str, Any]:
    """Derive G5 gate booleans while leaving numeric comparisons intact."""
    summaries: dict[str, Any] = {}
    convergence = course_signal.get("convergence", {})
    for comparison_name, right_arm in (("B_minus_A", "B"), ("C_minus_B", "C")):
        if comparison_name not in course_signal:
            continue
        signal = course_signal[comparison_name]
        saving = convergence.get(right_arm, {}).get("update_saving_fraction_vs_A_best_update")
        joint_signal = (
            signal["joint_success_gain"] >= 0.05
            and signal["joint_success_gain_ci95"][0] > 0.0
        )
        convergence_signal = (
            saving is not None
            and saving >= 0.20
            and signal["joint_success_gain_ci95"][0] > -0.05
        )
        summaries[comparison_name] = {
            "joint_success_gain_point_ge_0_05_and_ci_lower_gt_0": bool(joint_signal),
            "update_saving_ge_0_20_and_joint_noninferiority_ci_lower_gt_minus_0_05": bool(
                convergence_signal
            ),
            "pilot_signal_present": bool(joint_signal or convergence_signal),
        }
    return summaries


def _wrap_hz(values: np.ndarray, prf_hz: float) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) + prf_hz / 2.0) % prf_hz - prf_hz / 2.0


def _physical_diagnostics(
    model: TaskStudent,
    arrays: dict[str, np.ndarray],
    threshold: float,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = int(config["statistics"]["physical_diagnostic_pairs"])
    per_course = count // 4
    course = np.asarray(arrays["course_level"]).astype(str)
    selected_ids = np.concatenate([
        np.flatnonzero(course == level)[:per_course]
        for level in ("C0", "C1", "C2", "C3")
    ])
    h0 = np.asarray(arrays["iq_h0"][selected_ids], dtype=np.complex64)
    h1 = np.asarray(arrays["iq_h1"][selected_ids], dtype=np.complex64)
    prf_hz = float(config["radar"]["prf_hz"])
    baseline_h0 = _predict_student_scenes(h0, model, device)
    baseline_h1 = _predict_student_scenes(h1, model, device)
    phase_scores_agree: list[np.ndarray] = []
    phase_auc: list[float] = []
    phase_range_change: list[np.ndarray] = []
    phase_doppler_change: list[np.ndarray] = []
    for angle in 2.0 * np.pi * np.arange(32, dtype=np.float64) / 32.0:
        factor = np.exp(1j * angle)
        rotated_h0 = _predict_student_scenes((h0 * factor).astype(np.complex64), model, device)
        rotated_h1 = _predict_student_scenes((h1 * factor).astype(np.complex64), model, device)
        phase_scores_agree.append(np.concatenate((
            (rotated_h0["score"] >= threshold) == (baseline_h0["score"] >= threshold),
            (rotated_h1["score"] >= threshold) == (baseline_h1["score"] >= threshold),
        )))
        phase_auc.append(empirical_auc(rotated_h0["score"], rotated_h1["score"]))
        phase_range_change.append(rotated_h1["range_cell"] - baseline_h1["range_cell"])
        phase_doppler_change.append(_wrap_hz(rotated_h1["doppler_hz"] - baseline_h1["doppler_hz"], prf_hz))

    scale_results: dict[str, Any] = {}
    for factor in (0.5, 2.0):
        scaled_h0 = _predict_student_scenes((h0 * factor).astype(np.complex64), model, device)
        scaled_h1 = _predict_student_scenes((h1 * factor).astype(np.complex64), model, device)
        scale_results[str(factor)] = {
            "pfa_at_original_threshold": float(np.mean(scaled_h0["score"] >= threshold)),
            "pd_at_original_threshold": float(np.mean(scaled_h1["score"] >= threshold)),
            "pfa_change_from_unscaled": float(
                np.mean(scaled_h0["score"] >= threshold) - np.mean(baseline_h0["score"] >= threshold)
            ),
            "pd_change_from_unscaled": float(
                np.mean(scaled_h1["score"] >= threshold) - np.mean(baseline_h1["score"] >= threshold)
            ),
        }

    frequency_results: dict[str, Any] = {}
    pulse_index = np.arange(130, dtype=np.float64)[None, None, :, None]
    target_component = h1.astype(np.complex128) - h0.astype(np.complex128)
    for shift_hz in (-25.0, -10.0, 10.0, 25.0):
        modulation = np.exp(2j * np.pi * shift_hz * pulse_index / prf_hz)
        shifted_h1 = (h0.astype(np.complex128) + target_component * modulation).astype(np.complex64)
        shifted_prediction = _predict_student_scenes(shifted_h1, model, device)
        residual = _wrap_hz(
            shifted_prediction["doppler_hz"] - baseline_h1["doppler_hz"] - shift_hz,
            prf_hz,
        )
        frequency_results[str(shift_hz)] = {
            "doppler_equivariance_residual_p95_hz": float(np.quantile(np.abs(residual), 0.95)),
            "doppler_equivariance_residual_median_hz": float(np.median(np.abs(residual))),
            "range_change_median_bins": float(np.median(np.abs(shifted_prediction["range_cell"] - baseline_h1["range_cell"]))),
        }

    rng = np.random.default_rng(int(config["statistics"]["physical_diagnostic_seed"]))
    channel_phase = rng.uniform(0.0, 2.0 * np.pi, size=(len(h0), 3, 1, 1))
    channel_factor = np.exp(1j * channel_phase)
    incoherent_h0 = _predict_student_scenes((h0 * channel_factor).astype(np.complex64), model, device)
    incoherent_h1 = _predict_student_scenes((h1 * channel_factor).astype(np.complex64), model, device)
    baseline_auc = float(empirical_auc(baseline_h0["score"], baseline_h1["score"]))
    return {
        "n_paired_scenes": int(len(selected_ids)),
        "course_level_counts": {level: int(np.sum(course[selected_ids] == level)) for level in ("C0", "C1", "C2", "C3")},
        "public_common_phase_rotations": 32,
        "phase_decision_agreement_rate": float(np.mean(np.concatenate(phase_scores_agree))),
        "phase_range_change_p95_bins": float(np.quantile(np.abs(np.concatenate(phase_range_change)), 0.95)),
        "phase_doppler_change_p95_hz": float(np.quantile(np.abs(np.concatenate(phase_doppler_change)), 0.95)),
        "phase_auc_unrotated": baseline_auc,
        "phase_auc_change_max_abs": float(np.max(np.abs(np.asarray(phase_auc) - baseline_auc))),
        "amplitude_scales_no_recalibration": scale_results,
        "target_frequency_shift_hz": frequency_results,
        "independent_channel_phase_ablation": {
            "auc": float(empirical_auc(incoherent_h0["score"], incoherent_h1["score"])),
            "auc_change": float(empirical_auc(incoherent_h0["score"], incoherent_h1["score"]) - empirical_auc(baseline_h0["score"], baseline_h1["score"])),
            "pfa_at_original_threshold": float(np.mean(incoherent_h0["score"] >= threshold)),
            "pd_at_original_threshold": float(np.mean(incoherent_h1["score"] >= threshold)),
        },
        "used_original_calibration_threshold": True,
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(_json_safe(row))


def _evaluate_pilot() -> dict[str, Any]:
    preflight = _require_preflight()
    root = ROOT / "experiments/D2-A-E002"
    config = _read_yaml(root / "config.yaml")
    pilot_path = root / "pilot.json"
    if not pilot_path.is_file():
        raise RuntimeError("complete the authorized CUDA pilot before evaluation")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("git_sha") != preflight["git_sha"]:
        raise RuntimeError("pilot and frozen preflight commit do not match")
    device = require_cuda_device("cuda:0")
    cache_root = ROOT / config["data"]["train"]["cache_root"]
    physics = _read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    r3 = _read_yaml(ROOT / "experiments/D2-A-R3/config.yaml")
    t1_options = r3["detection"]["t1_global_robust_background"]

    calibration_arrays = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "h0_calibration").glob("*.npy")}
    verification_arrays = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "h0_verification").glob("*.npy")}
    development = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "development").glob("*.npy")}
    boundary = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "identifiability_boundary").glob("*.npy")}
    if len(calibration_arrays["base_scene_id"]) != 15000 or len(verification_arrays["base_scene_id"]) != 15000:
        raise RuntimeError("H0 calibration and verification caches must each contain 15,000 scenes")
    if len(development["base_scene_id"]) != 5000:
        raise RuntimeError("development cache must contain 5,000 paired scenes")

    started = time.perf_counter()
    classical_cal = _collect_classical(calibration_arrays["iq_h0"], physics, t1_options)
    classical_ver = _collect_classical(verification_arrays["iq_h0"], physics, t1_options)
    classical_dev_h0 = _collect_classical(development["iq_h0"], physics, t1_options)
    classical_dev_h1 = _collect_classical(development["iq_h1"], physics, t1_options)
    classical_boundary_h0 = _collect_classical(boundary["iq_h0"], physics, t1_options)
    classical_boundary_h1 = _collect_classical(boundary["iq_h1"], physics, t1_options)

    predictions: dict[str, dict[str, np.ndarray | None]] = {}
    boundary_predictions: dict[str, dict[str, np.ndarray | None]] = {}
    for method in ("E0", "B0", "B1", "B2", "B3", "T1"):
        predictions[method] = {
            "calibration_scores": np.asarray(classical_cal[method]["score"]),
            "verification_scores": np.asarray(classical_ver[method]["score"]),
            "development_h0_scores": np.asarray(classical_dev_h0[method]["score"]),
            "development_h1_scores": np.asarray(classical_dev_h1[method]["score"]),
            "range_cell": classical_dev_h1[method]["range_cell"],
            "doppler_hz": classical_dev_h1[method]["doppler_hz"],
        }
        boundary_predictions[method] = {
            "h0_scores": np.asarray(classical_boundary_h0[method]["score"]),
            "h1_scores": np.asarray(classical_boundary_h1[method]["score"]),
            "range_cell": classical_boundary_h1[method]["range_cell"],
            "doppler_hz": classical_boundary_h1[method]["doppler_hz"],
        }

    physical_by_arm: dict[str, Any] = {}
    nvidia_before = _nvidia_smi()
    for arm in pilot["arms_run"]:
        arm_result = pilot["arms"][arm]
        model = _load_inference_model(Path(arm_result["best_inference_checkpoint"]), config, device)
        scores_cal = _predict_student_scenes(calibration_arrays["iq_h0"], model, device)
        scores_ver = _predict_student_scenes(verification_arrays["iq_h0"], model, device)
        dev_h0 = _predict_student_scenes(development["iq_h0"], model, device)
        dev_h1 = _predict_student_scenes(development["iq_h1"], model, device)
        boundary_h0 = _predict_student_scenes(boundary["iq_h0"], model, device)
        boundary_h1 = _predict_student_scenes(boundary["iq_h1"], model, device)
        predictions[arm] = {
            "calibration_scores": scores_cal["score"],
            "verification_scores": scores_ver["score"],
            "development_h0_scores": dev_h0["score"],
            "development_h1_scores": dev_h1["score"],
            "range_cell": dev_h1["range_cell"],
            "doppler_hz": dev_h1["doppler_hz"],
        }
        boundary_predictions[arm] = {
            "h0_scores": boundary_h0["score"],
            "h1_scores": boundary_h1["score"],
            "range_cell": boundary_h1["range_cell"],
            "doppler_hz": boundary_h1["doppler_hz"],
        }
        threshold = calibrate_student_threshold(
            scores_cal["score"], target_pfa=float(config["detection"]["target_pfa_cpi"])
        )
        physical_by_arm[arm] = _physical_diagnostics(
            model,
            development,
            float(threshold["threshold"]),
            config,
            device,
        )
        del model
        torch.cuda.empty_cache()

    threshold_by_method: dict[str, dict[str, Any]] = {}
    verification_by_method: dict[str, dict[str, Any]] = {}
    development_by_method: dict[str, dict[str, Any]] = {}
    threshold_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    verification_scores: dict[str, np.ndarray] = {}
    development_group_rows: list[dict[str, Any]] = []
    bootstrap_replicates = int(config["statistics"]["development_bootstrap_replicates"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    dev_ids = np.asarray(development["base_scene_id"]).astype(str)
    truth_range = np.asarray(development["truth_range_cell"], dtype=np.float64)
    truth_doppler = np.asarray(development["truth_doppler_hz"], dtype=np.float64)
    prf_hz = float(config["radar"]["prf_hz"])

    for method, data in predictions.items():
        threshold = calibrate_student_threshold(
            np.asarray(data["calibration_scores"]),
            target_pfa=float(config["detection"]["target_pfa_cpi"]),
        )
        threshold_by_method[method] = threshold
        threshold_rows.append({"method": method, **threshold})
        verification = summarize_h0_verification(
            np.asarray(data["verification_scores"]), float(threshold["threshold"])
        )
        verification_by_method[method] = verification
        verification_scores[method] = np.asarray(data["verification_scores"])
        h0_scores = np.asarray(data["development_h0_scores"])
        h1_scores = np.asarray(data["development_h1_scores"])
        if method == "E0":
            development_metrics = evaluate_detection_scores(
                h0_scores=h0_scores,
                h1_scores=h1_scores,
                threshold=float(threshold["threshold"]),
                base_scene_ids=dev_ids,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
        else:
            development_metrics = evaluate_paired_predictions(
                h0_scores=h0_scores,
                h1_scores=h1_scores,
                threshold=float(threshold["threshold"]),
                truth_range=truth_range,
                predicted_range=np.asarray(data["range_cell"]),
                truth_doppler=truth_doppler,
                predicted_doppler=np.asarray(data["doppler_hz"]),
                base_scene_ids=dev_ids,
                prf_hz=prf_hz,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed,
            )
        common_threshold = float(np.quantile(h0_scores, 0.95, method="higher"))
        development_metrics["common_roc_fpr_target"] = 0.05
        development_metrics["common_roc_fpr_empirical"] = float(np.mean(h0_scores >= common_threshold))
        development_metrics["pd_at_common_roc_fpr_005"] = float(np.mean(h1_scores >= common_threshold))
        development_metrics["common_roc_h0_threshold"] = common_threshold
        development_metrics["verification_pfa"] = verification["pfa_ge"]
        development_metrics["verification_pfa_ci95_exact_upper"] = verification["pfa_ci95_exact_upper"]
        development_by_method[method] = development_metrics
        if method != "E0":
            development_group_rows.extend(_development_group_rows(
                method,
                h0_scores,
                h1_scores,
                np.asarray(data["range_cell"]),
                np.asarray(data["doppler_hz"]),
                float(threshold["threshold"]),
                development,
                prf_hz,
                float(physics["waveform"]["range_cell_m"]),
            ))
        row = {
            "method": method,
            "threshold": threshold["threshold"],
            "calibration_pfa_ge": threshold["pfa_ge"],
            "verification_pfa_ge": verification["pfa_ge"],
            "verification_pfa_exact_upper95": verification["pfa_ci95_exact_upper"],
            "development_pd": development_metrics["pd"],
            "development_auc": development_metrics["auc"],
            "development_pd_at_common_fpr_005": development_metrics["pd_at_common_roc_fpr_005"],
            "range_rmse_cells": development_metrics.get("range_rmse_all_h1_cells"),
            "doppler_rmse_hz": development_metrics.get("doppler_rmse_all_h1_hz"),
            "joint_success": development_metrics.get("joint_success"),
        }
        result_rows.append(row)

    boundary_metrics: dict[str, Any] = {}
    boundary_ids = np.asarray(boundary["base_scene_id"]).astype(str)
    boundary_truth_range = np.asarray(boundary["truth_range_cell"], dtype=np.float64)
    boundary_truth_doppler = np.asarray(boundary["truth_doppler_hz"], dtype=np.float64)
    boundary_rows: list[dict[str, Any]] = []
    for method, values in boundary_predictions.items():
        threshold = float(threshold_by_method[method]["threshold"])
        if values["range_cell"] is None:
            summary = evaluate_detection_scores(
                h0_scores=np.asarray(values["h0_scores"]),
                h1_scores=np.asarray(values["h1_scores"]),
                threshold=threshold,
                base_scene_ids=boundary_ids,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 900,
            )
        else:
            summary = evaluate_paired_predictions(
                h0_scores=np.asarray(values["h0_scores"]),
                h1_scores=np.asarray(values["h1_scores"]),
                threshold=threshold,
                truth_range=boundary_truth_range,
                predicted_range=np.asarray(values["range_cell"]),
                truth_doppler=boundary_truth_doppler,
                predicted_doppler=np.asarray(values["doppler_hz"]),
                base_scene_ids=boundary_ids,
                prf_hz=prf_hz,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + 900,
            )
        boundary_metrics[method] = summary
        boundary_rows.append({
            "method": method,
            "n_paired_scenes": int(len(boundary_ids)),
            "auc": summary["auc"],
            "pd_at_own_calibration_threshold": summary["pd"],
            "joint_success": summary.get("joint_success"),
            "range_rmse_cells": summary.get("range_rmse_all_h1_cells"),
            "doppler_rmse_hz": summary.get("doppler_rmse_all_h1_hz"),
            "descriptive_only": True,
        })

    h0_group_rows = _verification_group_rows(
        verification_scores,
        threshold_by_method,
        verification_arrays,
    )
    train_arrays = {path.stem: np.load(path, mmap_mode="r") for path in (cache_root / "train").glob("*.npy")}
    constant_range = float(np.median(np.asarray(train_arrays["truth_range_cell"], dtype=np.float64)))
    constant_doppler = _circular_mean_hz(np.asarray(train_arrays["truth_doppler_hz"], dtype=np.float64), prf_hz)
    constant_range_error = np.full(len(dev_ids), constant_range, dtype=np.float64) - truth_range
    constant_doppler_error = _wrap_hz(constant_doppler - truth_doppler, prf_hz)
    constant_baseline = {
        "range_prediction_train_median_cell": constant_range,
        "doppler_prediction_train_circular_mean_hz": constant_doppler,
        "range_rmse_all_h1_cells": float(np.sqrt(np.mean(constant_range_error**2))),
        "doppler_rmse_all_h1_hz": float(np.sqrt(np.mean(constant_doppler_error**2))),
    }

    paired_comparisons: dict[str, Any] = {}
    e0_h0 = np.asarray(predictions["E0"]["development_h0_scores"])
    e0_h1 = np.asarray(predictions["E0"]["development_h1_scores"])
    e0_threshold = float(threshold_by_method["E0"]["threshold"])
    e0_common_threshold = float(np.quantile(e0_h0, 0.95, method="higher"))
    e0_common_detection = e0_h1 >= e0_common_threshold
    for arm in pilot["arms_run"]:
        data = predictions[arm]
        h0_scores = np.asarray(data["development_h0_scores"])
        h1_scores = np.asarray(data["development_h1_scores"])
        threshold = float(threshold_by_method[arm]["threshold"])
        method_range_error = np.asarray(data["range_cell"]) - truth_range
        method_doppler_error = _wrap_hz(np.asarray(data["doppler_hz"]) - truth_doppler, prf_hz)
        paired_comparisons[arm] = {
            "auc_gain_vs_E0": float(development_by_method[arm]["auc"] - development_by_method["E0"]["auc"]),
            "auc_gain_vs_E0_ci95_paired_bootstrap": _paired_auc_difference_ci(
                h0_scores, h1_scores, e0_h0, e0_h1, dev_ids,
                replicates=bootstrap_replicates, seed=bootstrap_seed + ord(arm),
            ),
            "pd_difference_vs_E0_at_common_fpr_005": float(
                np.mean(h1_scores >= np.quantile(h0_scores, 0.95, method="higher")) - np.mean(e0_common_detection)
            ),
            "pd_difference_vs_E0_at_common_fpr_005_ci95": paired_difference_interval(
                (h1_scores >= np.quantile(h0_scores, 0.95, method="higher")).astype(np.float64),
                e0_common_detection.astype(np.float64),
                dev_ids,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + ord(arm) + 100,
            ),
            "range_rmse_difference_vs_train_constant": float(
                np.sqrt(np.mean(method_range_error**2)) - constant_baseline["range_rmse_all_h1_cells"]
            ),
            "range_rmse_difference_vs_train_constant_ci95": paired_difference_interval(
                method_range_error,
                constant_range_error,
                dev_ids,
                statistic=lambda a, b: float(np.sqrt(np.mean(a**2)) - np.sqrt(np.mean(b**2))),
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + ord(arm) + 200,
            ),
            "doppler_rmse_difference_vs_train_constant": float(
                np.sqrt(np.mean(method_doppler_error**2)) - constant_baseline["doppler_rmse_all_h1_hz"]
            ),
            "doppler_rmse_difference_vs_train_constant_ci95": paired_difference_interval(
                method_doppler_error,
                constant_doppler_error,
                dev_ids,
                statistic=lambda a, b: float(np.sqrt(np.mean(a**2)) - np.sqrt(np.mean(b**2))),
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + ord(arm) + 300,
            ),
        }
        main = (
            (np.asarray(development["scenario_id"]).astype(str) == "separated")
            & np.isclose(np.asarray(development["snr_db"], dtype=np.float64), 10.0)
            & np.isclose(np.asarray(development["requested_cnr_db"], dtype=np.float64), 10.0)
        )
        main_detected = h1_scores[main] >= threshold
        main_joint = (
            main_detected
            & (np.abs(method_range_error[main]) <= 1.0)
            & (np.abs(method_doppler_error[main]) <= 20.0)
        )
        paired_comparisons[arm]["primary_A_10_10_n"] = int(np.sum(main))
        paired_comparisons[arm]["primary_A_10_10_joint_success"] = float(np.mean(main_joint))
        paired_comparisons[arm]["primary_A_10_10_joint_success_one_sided_ci95_lower"] = _one_sided_cluster_lower(
            main_joint.astype(np.float64),
            dev_ids[main],
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + ord(arm) + 400,
        )

    course_signal: dict[str, Any] = {}
    for left, right, name in (("A", "B", "B_minus_A"), ("B", "C", "C_minus_B")):
        if left not in pilot["arms_run"] or right not in pilot["arms_run"]:
            continue
        left_metrics = predictions[left]
        right_metrics = predictions[right]
        lscore = np.asarray(left_metrics["development_h1_scores"])
        rscore = np.asarray(right_metrics["development_h1_scores"])
        lthreshold = float(threshold_by_method[left]["threshold"])
        rthreshold = float(threshold_by_method[right]["threshold"])
        lrange_error = np.asarray(left_metrics["range_cell"]) - truth_range
        rrange_error = np.asarray(right_metrics["range_cell"]) - truth_range
        ldoppler_error = _wrap_hz(np.asarray(left_metrics["doppler_hz"]) - truth_doppler, prf_hz)
        rdoppler_error = _wrap_hz(np.asarray(right_metrics["doppler_hz"]) - truth_doppler, prf_hz)
        ljoint = (lscore >= lthreshold) & (np.abs(lrange_error) <= 1.0) & (np.abs(ldoppler_error) <= 20.0)
        rjoint = (rscore >= rthreshold) & (np.abs(rrange_error) <= 1.0) & (np.abs(rdoppler_error) <= 20.0)
        difference_ci = paired_difference_interval(
            rjoint.astype(np.float64), ljoint.astype(np.float64), dev_ids,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + len(course_signal) + 500,
        )
        course_signal[name] = {
            "joint_success_gain": float(np.mean(rjoint) - np.mean(ljoint)),
            "joint_success_gain_ci95": difference_ci,
            "left_updates_to_A_best_validation_loss": pilot["arms"][left].get("updates_to_A_best_validation_loss"),
            "right_updates_to_A_best_validation_loss": pilot["arms"][right].get("updates_to_A_best_validation_loss"),
        }
    if "A" in pilot["arms_run"]:
        a_best_update = int(pilot["arms"]["A"]["best_update"])
        for arm in ("B", "C"):
            if arm in pilot["arms_run"]:
                reached = pilot["arms"][arm].get("updates_to_A_best_validation_loss")
                course_signal.setdefault("convergence", {})[arm] = {
                    "A_best_update": a_best_update,
                    "updates_to_A_best_validation_loss": reached,
                    "update_saving_fraction_vs_A_best_update": None if reached is None or a_best_update == 0 else 1.0 - float(reached) / a_best_update,
                }

    gates: dict[str, Any] = {
        "G1_engineering": {
            "passed": bool(pilot.get("micro_overfit_sanity", {}).get("passed")),
            "micro_overfit_sanity": pilot.get("micro_overfit_sanity"),
            "cuda_gpu": preflight.get("gpu_name"),
        },
        "G2_k2_quality": {},
        "G3_h0_detection": {},
        "G4_task_learning": {},
        "G5_course_k2_signal": course_signal,
        "G6_physical_diagnostics": physical_by_arm,
    }
    k2_result = json.loads((root / "k2_audit.json").read_text(encoding="utf-8"))
    gates["G2_k2_quality"] = {
        "numeric_valid": bool(k2_result["numeric_valid"]),
        "scale_invariance_valid": bool(k2_result["scale_invariance_valid"]),
        "information_isolation_valid": bool(k2_result["information_isolation_valid"]),
        "severe_misleading_rule_triggered": bool(k2_result["severe_misleading_rule_triggered"]),
        "allow_C_arm": bool(k2_result["allow_C_arm"]),
    }
    for arm in pilot["arms_run"]:
        verification = verification_by_method[arm]
        subgroup_rows = [row for row in h0_group_rows if row["method"] == arm]
        gates["G3_h0_detection"][arm] = {
            "pfa_point_le_0_06": verification["pfa_ge"] <= 0.06,
            "pfa_exact_upper_le_0_06": verification["pfa_ci95_exact_upper"] <= 0.06,
            "all_subgroup_exact_upper_le_0_10": all(row["pfa_ci95_exact_upper"] <= 0.10 for row in subgroup_rows),
            "max_subgroup_exact_upper": max((row["pfa_ci95_exact_upper"] for row in subgroup_rows), default=None),
        }
        metrics = development_by_method[arm]
        comparison = paired_comparisons[arm]
        gates["G4_task_learning"][arm] = {
            "auc_ci_lower_gt_0_5": metrics["auc_ci95_paired_bootstrap"][0] > 0.5,
            "auc_point_gain_vs_E0_ge_0_02": comparison["auc_gain_vs_E0"] >= 0.02,
            "auc_gain_ci_lower_gt_0": comparison["auc_gain_vs_E0_ci95_paired_bootstrap"][0] > 0.0,
            "common_fpr_pd_difference_ci_lower_gt_minus_0_05": comparison["pd_difference_vs_E0_at_common_fpr_005_ci95"][0] > -0.05,
            "range_rmse_better_than_train_constant_ci_upper_lt_0": comparison["range_rmse_difference_vs_train_constant_ci95"][1] < 0.0,
            "doppler_rmse_better_than_train_constant_ci_upper_lt_0": comparison["doppler_rmse_difference_vs_train_constant_ci95"][1] < 0.0,
            "primary_joint_success_ge_0_50": comparison["primary_A_10_10_joint_success"] >= 0.50,
            "primary_joint_success_lower_gt_0_40": comparison["primary_A_10_10_joint_success_one_sided_ci95_lower"] > 0.40,
        }
    g5_comparisons = _g5_gate_summary(course_signal)
    gates["G5_course_k2_signal"] = g5_comparisons
    gates["G6_physical_diagnostics"] = {
        arm: {
            **diagnostic,
            "public_phase_gate_passed": (
                diagnostic["phase_decision_agreement_rate"] >= 0.99
                and diagnostic["phase_range_change_p95_bins"] <= 0.25
                and diagnostic["phase_doppler_change_p95_hz"] <= 5.0
                and diagnostic["phase_auc_change_max_abs"] <= 0.01
            ),
            "frequency_shift_gate_passed": (
                max(item["doppler_equivariance_residual_p95_hz"] for item in diagnostic["target_frequency_shift_hz"].values()) <= 10.0
                and max(item["range_change_median_bins"] for item in diagnostic["target_frequency_shift_hz"].values()) <= 0.25
            ),
            "amplitude_scale_sensitive": any(
                item["pfa_at_original_threshold"] > 0.10
                or item["pd_change_from_unscaled"] < -0.10
                for item in diagnostic["amplitude_scales_no_recalibration"].values()
            ),
        }
        for arm, diagnostic in physical_by_arm.items()
    }

    common_range_m = float(physics["waveform"]["range_cell_m"])
    g3_pass = {
        arm: bool(
            result["pfa_point_le_0_06"]
            and result["pfa_exact_upper_le_0_06"]
            and result["all_subgroup_exact_upper_le_0_10"]
        )
        for arm, result in gates["G3_h0_detection"].items()
    }
    result_rows = [
        {
            **row,
            "range_rmse_m": None if row["range_rmse_cells"] is None else row["range_rmse_cells"] * common_range_m,
            "G3_pass": None if row["method"] not in pilot["arms_run"] else g3_pass[row["method"]],
            "G4_pass": None if row["method"] not in pilot["arms_run"] else all(gates["G4_task_learning"][row["method"]].values()),
        }
        for row in result_rows
    ]
    g4_pass = {
        arm: bool(all(values.values()))
        for arm, values in gates["G4_task_learning"].items()
    }
    g6_pass = {
        arm: bool(values["public_phase_gate_passed"] and values["frequency_shift_gate_passed"])
        for arm, values in gates["G6_physical_diagnostics"].items()
    }
    if not gates["G1_engineering"]["passed"]:
        stage_decision = "STOP"
    elif any(g4_pass.values()) and any(g3_pass[arm] and g6_pass[arm] for arm in g4_pass if arm in g3_pass and arm in g6_pass):
        stage_decision = "CONTINUE"
    elif any(g4_pass.values()):
        stage_decision = "REFINE"
    else:
        stage_decision = "REDIRECT"
    _write_rows(root / "calibration_thresholds.csv", threshold_rows)
    _write_rows(root / "h0_verification_by_group.csv", h0_group_rows)
    _write_rows(root / "development_by_group.csv", development_group_rows)
    _write_rows(root / "result.csv", result_rows)
    _write_rows(root / "D_boundary.csv", boundary_rows)
    _write_rows(root / "development_per_scene.csv", [
        {
            "base_scene_id": str(dev_ids[index]),
            "scenario_id": str(development["scenario_id"][index]),
            "course_level": str(development["course_level"][index]),
            "snr_db": float(development["snr_db"][index]),
            "cnr_db": float(development["requested_cnr_db"][index]),
            "truth_range_cell": float(truth_range[index]),
            "truth_doppler_hz": float(truth_doppler[index]),
            "truth_amplitude": float(development["truth_amplitude"][index]),
            "truth_phase_rad": float(development["truth_phase_rad"][index]),
            "method": method,
            "h0_score": float(predictions[method]["development_h0_scores"][index]),
            "h1_score": float(predictions[method]["development_h1_scores"][index]),
            "threshold": float(threshold_by_method[method]["threshold"]),
            "detected": bool(predictions[method]["development_h1_scores"][index] >= threshold_by_method[method]["threshold"]),
            "range_estimate_cell": None if predictions[method]["range_cell"] is None else float(predictions[method]["range_cell"][index]),
            "doppler_estimate_hz": None if predictions[method]["doppler_hz"] is None else float(predictions[method]["doppler_hz"][index]),
        }
        for method in predictions
        for index in range(len(dev_ids))
    ])

    data_generation = json.loads((root / "data_generation.json").read_text(encoding="utf-8"))
    metrics = {
        "experiment": "D2-A-E002",
        "protocol_version": PROTOCOL_VERSION,
        "git_sha": preflight["git_sha"],
        "synthetic_data": True,
        "final_blind_generated": False,
        "training_arms_run": pilot["arms_run"],
        "data_generation": data_generation,
        "thresholds": threshold_by_method,
        "h0_verification": verification_by_method,
        "h0_verification_subgroups": h0_group_rows,
        "development": development_by_method,
        "exact_overlap_D_boundary_descriptive_only": boundary_metrics,
        "development_subgroups": development_group_rows,
        "train_constant_parameter_baseline": constant_baseline,
        "paired_comparisons": paired_comparisons,
        "course_comparison_estimates": course_signal,
        "k2_audit": json.loads((root / "k2_audit.json").read_text(encoding="utf-8")),
        "gates": gates,
        "stage_decision": stage_decision,
        "physical_diagnostics": physical_by_arm,
        "evaluation_seconds": float(time.perf_counter() - started),
        "nvidia_smi_before": nvidia_before,
        "nvidia_smi_after": _nvidia_smi(),
    }
    _write_json(root / "metrics.json", metrics)
    _write_plots(root, predictions, development, pilot, config, prf_hz)
    _write_analysis(root, metrics, pilot)
    _write_stage_review(metrics, pilot)
    return metrics


def _write_plots(
    root: Path,
    predictions: dict[str, dict[str, np.ndarray | None]],
    development: dict[str, np.ndarray],
    pilot: dict[str, Any],
    config: dict[str, Any],
    prf_hz: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    history_fig, history_ax = plt.subplots(figsize=(8, 5))
    for arm in pilot["arms_run"]:
        info = pilot["arms"][arm]
        path = Path(info["training_history"])
        table = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
        if table.size:
            history_ax.plot(table["update"], table["train_total"], label=f"{arm} train")
            if "validation_task_loss" in table.dtype.names:
                valid = np.isfinite(table["validation_task_loss"])
                history_ax.scatter(table["update"][valid], table["validation_task_loss"][valid], s=10, label=f"{arm} validation")
    history_ax.set(xlabel="Optimizer update", ylabel="Truth-task / training loss", title="D2-A-E002 pilot convergence")
    history_ax.legend(ncol=2, fontsize=8)
    history_fig.tight_layout()
    history_fig.savefig(figures / "training_curves.png", dpi=160)
    plt.close(history_fig)

    snr = np.asarray(development["snr_db"], dtype=np.float64)
    methods = [name for name in predictions if predictions[name]["range_cell"] is not None]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    truth_range = np.asarray(development["truth_range_cell"], dtype=np.float64)
    truth_doppler = np.asarray(development["truth_doppler_hz"], dtype=np.float64)
    for method in methods:
        range_values: list[float] = []
        doppler_values: list[float] = []
        snr_values: list[float] = []
        for level in np.unique(snr):
            selected = np.isclose(snr, level)
            range_error = np.asarray(predictions[method]["range_cell"])[selected] - truth_range[selected]
            doppler_error = (np.asarray(predictions[method]["doppler_hz"])[selected] - truth_doppler[selected] + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
            snr_values.append(float(level))
            range_values.append(float(np.sqrt(np.mean(range_error**2))))
            doppler_values.append(float(np.sqrt(np.mean(doppler_error**2))))
        axes[0].plot(snr_values, range_values, marker="o", label=method)
        axes[1].plot(snr_values, doppler_values, marker="o", label=method)
    axes[0].set(xlabel="SNR (dB)", ylabel="Range RMSE (cell)", title="Development range error")
    axes[1].set(xlabel="SNR (dB)", ylabel="Doppler RMSE (Hz)", title="Development Doppler error")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "rmse_snr.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    for method, values in predictions.items():
        h0 = np.asarray(values["development_h0_scores"])
        h1 = np.asarray(values["development_h1_scores"])
        fprs: list[float] = []
        tprs: list[float] = []
        for quantile in np.linspace(0.0, 1.0, 101):
            threshold = float(np.quantile(h0, quantile, method="higher"))
            fprs.append(float(np.mean(h0 >= threshold)))
            tprs.append(float(np.mean(h1 >= threshold)))
        axis.plot(fprs, tprs, label=method)
    axis.plot([0.0, 1.0], [0.0, 1.0], color="black", linestyle="--", linewidth=1)
    axis.set(xlabel="Empirical false alarm probability", ylabel="Empirical detection probability", title="Development ROC")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "roc.png", dpi=160)
    plt.close(fig)


def _write_analysis(root: Path, metrics: dict[str, Any], pilot: dict[str, Any]) -> None:
    dev = metrics["development"]
    gates = metrics["gates"]
    best_method = max(dev, key=lambda method: dev[method]["auc"])
    best_student = max(pilot["arms_run"], key=lambda method: dev[method]["auc"])
    lines = [
        "# D2-A-E002 阶段结果",
        "",
        f"- 协议：v{metrics['protocol_version']}；代码提交：`{metrics['git_sha']}`。",
        f"- 训练臂：{', '.join(metrics['training_arms_run'])}；每臂 1 个 seed；final blind 未生成。",
        f"- 数据：独立 H0 calibration/verification 各 15,000 场景，development 5,000 对；全为仿真数据。",
        f"- 数据生成耗时：{metrics['data_generation']['generation_seconds']:.1f} 秒；评价耗时：{metrics['evaluation_seconds']:.1f} 秒。",
        "",
        "## 总体结果",
        "",
        f"- 按 development AUC 最高的方法为 **{best_method}**（AUC={dev[best_method]['auc']:.4f}）。",
        f"- 学生臂中 AUC 最高的是 **{best_student}**（AUC={dev[best_student]['auc']:.4f}）。",
        f"- E0 AUC={dev['E0']['auc']:.4f}；E0 在自身校准阈值下 Pd={dev['E0']['pd']:.4f}。",
        f"- 训练 seed 101 的各臂 AUC、Pd、RMSE 与联合成功率见 [`result.csv`](result.csv) 和 [`metrics.json`](metrics.json)。",
        f"- K2 audit 是否允许 C 臂：**{gates['G2_k2_quality']['allow_C_arm']}**。该结果只描述软响应，不改变 R3 固定阈值门状态。",
        "",
        "## 预注册门",
        "",
        f"- G1 工程 sanity：{gates['G1_engineering']['passed']}。",
    ]
    for arm in pilot["arms_run"]:
        g3 = gates["G3_h0_detection"][arm]
        lines.append(f"- {arm} G3 H0 检测门：{g3['pfa_point_le_0_06'] and g3['pfa_exact_upper_le_0_06'] and g3['all_subgroup_exact_upper_le_0_10']}。")
        lines.append(f"- {arm} G4 任务学习门：{all(gates['G4_task_learning'][arm].values())}。")
    for comparison, signal in gates["G5_course_k2_signal"].items():
        lines.append(f"- G5 {comparison} pilot signal：{signal['pilot_signal_present']}。")
    lines.extend([
        f"- 阶段决策：**{metrics['stage_decision']}**。",
        f"- exact-overlap D boundary AUC（仅描述）：{metrics['exact_overlap_D_boundary_descriptive_only']['B3']['auc']:.4f}。",
        "",
        "## 解释边界",
        "",
        "本轮是单训练 seed 的开发 pilot。场景 bootstrap 反映场景抽样不确定性，不代表跨 seed 不确定性；结果不能证明课程学习或蒸馏的总体收益。模型选择未使用 final blind，且没有生成 final blind。R3 的独立固定阈值结果、P0/D1 结果和历史负结果保持原样。",
        "",
        "方法分组、阈值平局、固定阈值 H0 verification、相位/频移/尺度/通道物理诊断及逐场景预测分别见 CSV、JSON 和 `figures/`。",
    ])
    (root / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_stage_review(metrics: dict[str, Any], pilot: dict[str, Any]) -> None:
    target = ROOT / "docs/stage_reviews/D2-A-E002_review.md"
    dev = metrics["development"]
    best_student = max(pilot["arms_run"], key=lambda arm: dev[arm]["auc"])
    lines = [
        "# D2-A-E002 阶段审查材料",
        "",
        "**请首先读取本次Git提交对应版本的 `docs/RESEARCH_CHARTER.md`，理解RadarDistill的整体研究主题、研究目标、研究内容、科学问题和技术路线，然后结合这些内容审查本阶段的代码、实验、数据和研究结论。**",
        "",
        f"- 结果与代码冻结提交 SHA：`{metrics['git_sha']}`。",
        "- 研究总纲：`docs/RESEARCH_CHARTER.md` v1.1；本阶段未修改总纲。",
        "- 本阶段协议：`experiments/D2-A-E002/protocol.md` v2.1。",
        "",
        "## 与总体研究目标的关系",
        "",
        "本阶段检验物理仿真真值监督的任务级学生能否直接从三通道复数 IQ 输出目标存在、距离和 Doppler，并在相同数据多重集与 1,600 次更新预算下比较全分布随机混合、由易到难课程顺序及课程加 IQ-only K2 软响应。工作对应研究总纲 Q1（从复数观测学习物理参数）、Q2（教师是否转移有效信息）、Q3（仿真真值与传统教师如何支持训练）和 Q4（三通道联合参数任务的可行性）。本阶段不评价后期部署计算开销 Q6。",
        "",
        "## 协议、代码、数据和结果",
        "",
        "- 协议/配置：`experiments/D2-A-E002/protocol.md`、`experiments/D2-A-E002/config.yaml`。",
        "- 数据生成及训练入口：`scripts/run_d2a_e002.py`；配对缓存、IQ-only K2、损失、CUDA 训练器和评价模块位于 `src/radardistill/`。",
        "- 汇总：`experiments/D2-A-E002/metrics.json`、`result.csv`、`D_boundary.csv`、`development_by_group.csv`、`h0_verification_by_group.csv`、`development_per_scene.csv`。",
        "- 阈值校准和验证：`calibration_thresholds.csv` 与 `metrics.json` 中的精确单侧 Clopper–Pearson 上限。",
        "- K2 与物理诊断：`k2_audit.json`、`k2_audit_scenes.csv` 及 `metrics.json`。",
        "- 阶段分析和图表：`experiments/D2-A-E002/analysis.md`、`experiments/D2-A-E002/figures/`。",
        "",
        "## 观察到的证据",
        "",
        f"- 实际训练臂：{', '.join(metrics['training_arms_run'])}；每臂只有训练 seed 101。",
        f"- development AUC 最好的学生臂：{best_student}（AUC={dev[best_student]['auc']:.4f}）。",
        f"- E0 能量检测基线：AUC={dev['E0']['auc']:.4f}，Pd={dev['E0']['pd']:.4f}；完整基线对比见 `result.csv`。",
        f"- K2 质量审查是否允许 C 臂：{metrics['gates']['G2_k2_quality']['allow_C_arm']}。",
        f"- 阶段门与决策：**{metrics['stage_decision']}**；具体 G1–G6 数值见 `metrics.json`。",
        "- exact-overlap D 只作为不可辨识参考，不用于来源恢复门或模型选择。",
        "",
        "## 对整体研究方向的审查问题",
        "",
        "请判断本阶段是否真正推进三通道任务级学生目标，而不是只优化中间教师统计量；真值监督是否足以学习可辨识物理量；课程顺序和 K2 相对 B 臂是否给出有价值的 pilot 信号；学生是否优于常值参数预测与 E0 检测基线；R3 固定阈值结果是否与 K2 软响应资格正确区分；物理一致性证据是否支持学生使用通道相干信息；普通实值 TaskStudent 是否仍适合作为后续学生基线，或证据是否支持引入显式复数/几何归纳偏置。",
        "",
        "## 统计边界与下一步建议",
        "",
        "本轮为单 seed 开发 pilot；按 base_scene_id 的 bootstrap 区间只反映场景抽样不确定性，不代表跨 seed 不确定性。验证集用于选 checkpoint，development 用于有限的阶段门诊断；未生成 final blind。不得据此声称课程或蒸馏已跨 seed 证实。请结合总体研究总纲和预注册门决定下一阶段；如进入正式确认，应先冻结独立训练 seed、统计功效、模型选择以及全新的盲测协议。",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("experiments/D2-A-E002/config.yaml"))
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument(
        "--stage",
        choices=("preflight", "generate-data", "k2-audit", "pilot", "evaluate", "all"),
        required=True,
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if args.stage in {"preflight", "all"}:
        print(json.dumps(_json_safe(_preflight(config_path)), ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"generate-data", "all"}:
        print(json.dumps(_json_safe(_generate_data(config_path)), ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"k2-audit", "all"}:
        print(json.dumps(_json_safe(_run_k2_audit()), ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"pilot", "all"}:
        print(json.dumps(_json_safe(_run_sanity_and_pilot()), ensure_ascii=False, indent=2), flush=True)
    if args.stage in {"evaluate", "all"}:
        print(json.dumps(_json_safe(_evaluate_pilot()), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
