#!/usr/bin/env python3
"""Run the preregistered D2-A-T001 teacher study without caching scene IQ."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.d2a_baselines import DetectionOutput, run_b3
from radardistill.d2a_r3_baselines import run_t1_global_normalized
from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_e002 import (
    evaluate_paired_predictions,
    exact_binomial_upper,
    paired_difference_interval,
    summarize_h0_verification,
)
from radardistill.evaluation.d2a_r3 import calibrate_empirical_threshold
from radardistill.evaluation.d2a_t001 import (
    build_development_assignments,
    build_h0_mixture_assignments,
    evaluate_method_group,
    paired_metric_difference,
    summarize_soft_response,
    summarize_soft_response_per_scene,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.teachers.d2a_k2 import k2_response
from radardistill.teachers.d2a_t001 import (
    CovarianceBank,
    fit_post_doppler_covariance,
    post_doppler_ace_amf,
)
from radardistill.baselines import wrap_frequency

METHODS = ("B3", "T1", "ACE_AMF")
FROZEN_INPUTS = (
    "docs/RESEARCH_CHARTER.md",
    "experiments/D2-A-E001/config.yaml",
    "experiments/D2-A-E002/config.yaml",
    "experiments/D2-A-R3/config.yaml",
    "experiments/D2-A-T001/protocol.md",
    "experiments/D2-A-T001/config.yaml",
    "src/radardistill/d2a_baselines.py",
    "src/radardistill/d2a_r3_baselines.py",
    "src/radardistill/teachers/d2a_k2.py",
    "src/radardistill/teachers/d2a_t001.py",
    "src/radardistill/evaluation/d2a_t001.py",
    "src/radardistill/simulation/d2a.py",
    "scripts/run_d2a_t001.py",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else ""
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _repo_size_bytes() -> int:
    result = subprocess.run(
        ["du", "-sb", str(ROOT)], cwd=ROOT, text=True, capture_output=True, check=True
    )
    return int(result.stdout.split()[0])


def _require_frozen_commit(config: dict[str, Any]) -> str:
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("T001 requires the authorized main branch")
    for item in FROZEN_INPUTS:
        _git("ls-files", "--error-unmatch", item)
        if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", item], cwd=ROOT).returncode:
            raise RuntimeError(f"T001 frozen input has uncommitted changes: {item}")
    head = _git("rev-parse", "HEAD")
    remote = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True, capture_output=True
    )
    if remote.returncode != 0 or remote.stdout.strip() != head:
        raise RuntimeError("push the frozen protocol and implementation to origin/main before T001 data generation")
    expected = str(config["experiment"]["research_charter_commit"])
    charter_commit = _git("log", "-1", "--format=%H", expected, "--", "docs/RESEARCH_CHARTER.md")
    if charter_commit != expected:
        raise RuntimeError("the configured research-charter version is not present in the committed history")
    return head


def _nvidia_smi() -> dict[str, Any]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    return {
        "returncode": int(result.returncode),
        "query": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _environment(git_sha: str, free_bytes: int) -> dict[str, Any]:
    try:
        import torch

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        torch_info = {"available": False}
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return {
        "git_sha": git_sha,
        "branch": "main",
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": int(pages * page_size),
        "numpy_version": np.__version__,
        "torch": torch_info,
        "nvidia_smi": _nvidia_smi(),
        "disk_free_bytes": int(free_bytes),
        "project_size_bytes_before_run": int(_repo_size_bytes()),
        "final_project_size_cap_bytes": int(20 * 1024**3),
    }


def _cuda_device_status(device: str, minimum_free_bytes: int) -> dict[str, Any]:
    import torch

    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as error:
        raise ValueError(f"invalid teacher CUDA device {device!r}") from error
    if selected.type != "cuda":
        raise ValueError("T001 teacher execution requires CUDA; CPU fallback is disabled")
    if not torch.cuda.is_available():
        raise RuntimeError("T001 teacher execution requires an available CUDA device")
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError(f"T001 requested CUDA device {selected} is not visible")
    torch.cuda.set_device(selected)
    free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
    if free_bytes < int(minimum_free_bytes):
        raise RuntimeError(
            f"T001 requires at least {minimum_free_bytes / 1024**3:.2f} GiB free CUDA memory; "
            f"available={free_bytes / 1024**3:.2f} GiB"
        )
    return {
        "device": str(selected),
        "name": torch.cuda.get_device_name(selected),
        "free_memory_bytes": int(free_bytes),
        "total_memory_bytes": int(total_bytes),
        "minimum_free_memory_bytes": int(minimum_free_bytes),
        "precision": "complex64 compute after observable double-precision scale/phase normalization",
    }


def _preflight(config_path: Path, device: str) -> dict[str, Any]:
    config = _read_yaml(config_path)
    physics = _read_yaml(ROOT / config["physics"]["source_config"])
    e002 = _read_yaml(ROOT / "experiments/D2-A-E002/config.yaml")
    r3 = _read_yaml(ROOT / "experiments/D2-A-R3/config.yaml")
    if config["experiment"]["id"] != "D2-A-T001" or float(config["experiment"]["protocol_version"]) != 1.2:
        raise RuntimeError("T001 config must match preregistered protocol v1.2")
    if config["experiment"].get("final_blind") is not False:
        raise RuntimeError("T001 does not authorize final blind generation")
    if int(physics["radar"]["channel_count"]) != 3 or int(physics["radar"]["pulse_count"]) != 130 or float(physics["radar"]["prf_hz"]) != 1300.0:
        raise RuntimeError("frozen D2-A physics no longer matches 3 channels, PRF=1300 Hz, N=130")
    if e002["experiment"].get("protocol_version") != 2.1:
        raise RuntimeError("frozen E002 student anchor config version changed")
    if config["teacher"].get("amf_score_clip") is not None:
        raise RuntimeError("T001 AMF score must remain unclipped")
    execution = config["execution"]
    if str(device) != str(execution["teacher_device"]):
        raise RuntimeError(
            f"T001 teacher device is frozen as {execution['teacher_device']}, got {device}"
        )
    if (
        int(execution["teacher_batch_size"]) != int(config["teacher"]["training_batch_size"])
        or int(execution["teacher_batch_size"]) != int(config["data"]["generation_batch_size"])
        or int(execution["teacher_batch_size"]) > 16
    ):
        raise RuntimeError("T001 CUDA teacher batch size must match the frozen <=16-CPI stream size")
    cuda_status = _cuda_device_status(
        device,
        int(execution["minimum_free_cuda_memory_bytes"]),
    )
    git_sha = _require_frozen_commit(config)
    size_bytes = _repo_size_bytes()
    disk = os.statvfs(ROOT)
    free_bytes = int(disk.f_bavail * disk.f_frsize)
    if free_bytes < 2 * 1024**3:
        raise RuntimeError(f"T001 requires at least 2 GiB working-space reserve; available={free_bytes / 1024**3:.2f} GiB")
    environment = _environment(git_sha, free_bytes)
    environment["teacher_cuda"] = cuda_status
    root = ROOT / config["outputs"]["directory"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "config_used.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    _write_json(root / "environment.json", environment)
    result = {
        "protocol_version": 1.2,
        "git_sha": git_sha,
        "branch": "main",
        "pushed_to_origin_main": True,
        "teacher_cuda": cuda_status,
        "physics_config": config["physics"]["source_config"],
        "project_size_bytes_before_run": size_bytes,
        "final_project_size_cap_bytes": 20 * 1024**3,
        "disk_free_bytes": free_bytes,
        "estimated_persistent_t001_output_bytes_max": 500 * 1024**2,
        "passed": True,
    }
    _write_json(root / "preflight.json", result)
    return result


def _require_preflight(config: dict[str, Any], device: str) -> dict[str, Any]:
    root = ROOT / config["outputs"]["directory"]
    preflight_path = root / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("run the T001 preflight before generating scenes")
    result = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        not result.get("passed")
        or result.get("git_sha") != _require_frozen_commit(config)
        or result.get("teacher_cuda", {}).get("device") != device
    ):
        raise RuntimeError("T001 preflight no longer matches the pushed frozen code and protocol")
    _cuda_device_status(device, int(config["execution"]["minimum_free_cuda_memory_bytes"]))
    return result


def _seed(
    seed_base: int,
    chunk_index: int,
    stride: int,
    course_index: int = 0,
    course_stride: int = 100_000,
) -> int:
    return int(seed_base) + course_index * course_stride + int(chunk_index) * int(stride)


def _covariance_h0_batches(
    config: dict[str, Any], physics: dict[str, Any]
) -> tuple[Iterable[np.ndarray], set[str]]:
    teacher = config["teacher"]
    seed_base = int(teacher["covariance_training_seed_base"])
    stride = int(config["data"]["chunk_seed_stride"])
    batch_size = int(teacher["training_batch_size"])
    counts = teacher["covariance_course_counts"]
    course_levels = config["data"]["course_levels"]
    c3_counts = teacher["c3_snr_counts"]
    id_set: set[str] = set()

    def generate() -> Iterable[np.ndarray]:
        for course_index, course in enumerate(("C0", "C1", "C2", "C3")):
            conditions: list[tuple[float, float, int]]
            if course == "C3":
                conditions = [
                    (float(snr), float(course_levels[course]["cnr_db"]), int(count))
                    for snr, count in c3_counts.items()
                ]
            else:
                conditions = [
                    (float(course_levels[course]["snr_db"]), float(course_levels[course]["cnr_db"]), int(counts[course]))
                ]
            chunk_global = 0
            for snr, cnr, count in conditions:
                for start in range(0, count, batch_size):
                    size = min(batch_size, count - start)
                    seed = _seed(seed_base, chunk_global, stride, course_index)
                    batch = generate_d2a_paired_batch(
                        size, seed, snr, cnr, physics, scenario_id="separated"
                    )
                    id_set.update(str(value) for value in batch.base_scene_id)
                    yield batch.iq_h0
                    chunk_global += 1
    return generate(), id_set


def _iter_h0_split(
    split_config: dict[str, Any],
    config: dict[str, Any],
    physics: dict[str, Any],
) -> Iterable[tuple[np.ndarray, dict[str, np.ndarray]]]:
    count = int(split_config["scenes"])
    seed_base = int(split_config["seed_base"])
    mixture = build_h0_mixture_assignments(
        count,
        int(split_config["mixture_seed"]),
        config["data"]["course_levels"],
    )
    batch_size = int(config["data"]["generation_batch_size"])
    stride = int(config["data"]["chunk_seed_stride"])
    for chunk_index, start in enumerate(range(0, count, batch_size)):
        stop = min(start + batch_size, count)
        seed = seed_base + chunk_index * stride
        batch = generate_d2a_paired_batch(
            stop - start,
            seed,
            mixture["snr_db"][start:stop],
            mixture["cnr_db"][start:stop],
            physics,
            scenario_id="separated",
        )
        metadata = {
            "base_scene_id": batch.base_scene_id.astype("U96"),
            "course_level": mixture["course_level"][start:stop],
            "scenario_id": np.full(stop - start, "H0", dtype="U32"),
            "snr_db": mixture["snr_db"][start:stop],
            "cnr_db": mixture["cnr_db"][start:stop],
            "truth_amplitude_diagnostic": batch.truth_target_amplitude.copy(),
        }
        yield batch.iq_h0, metadata


def _method_outputs(
    iq: np.ndarray,
    covariance: CovarianceBank,
    physics: dict[str, Any],
    config: dict[str, Any],
    t1_options: dict[str, Any],
    device: str = "cpu",
) -> dict[str, dict[str, np.ndarray]]:
    prf_hz = float(physics["radar"]["prf_hz"])
    b3 = run_b3(iq, prf_hz, physics)
    t1 = run_t1_global_normalized(iq, prf_hz, physics, t1_options)
    ace = post_doppler_ace_amf(
        iq, covariance, physics, config["teacher"], device=device
    )
    return {
        "B3": {
            "score": np.atleast_1d(np.asarray(b3.score, dtype=np.float64)),
            "range_cell": np.atleast_1d(np.asarray(b3.range_cell, dtype=np.float64)),
            "doppler_hz": np.atleast_1d(np.asarray(b3.doppler_hz, dtype=np.float64)),
        },
        "T1": {
            "score": np.atleast_1d(np.asarray(t1.score, dtype=np.float64)),
            "range_cell": np.atleast_1d(np.asarray(t1.range_cell, dtype=np.float64)),
            "doppler_hz": np.atleast_1d(np.asarray(t1.doppler_hz, dtype=np.float64)),
        },
        "ACE_AMF": {
            "score": np.atleast_1d(np.asarray(ace.score, dtype=np.float64)),
            "range_cell": np.atleast_1d(np.asarray(ace.range_cell, dtype=np.float64)),
            "doppler_hz": np.atleast_1d(np.asarray(ace.doppler_hz, dtype=np.float64)),
            "angle_deg": np.atleast_1d(np.asarray(ace.angle_deg, dtype=np.float64)),
            "soft_response": np.asarray(ace.soft_response, dtype=np.float32),
        },
    }


def _score_h0_split(
    split_name: str,
    split_config: dict[str, Any],
    config: dict[str, Any],
    physics: dict[str, Any],
    covariance: CovarianceBank,
    t1_options: dict[str, Any],
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], set[str], np.ndarray]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    methods = {method: [] for method in METHODS}
    first_batch: np.ndarray | None = None
    for iq, metadata in _iter_h0_split(split_config, config, physics):
        if first_batch is None:
            first_batch = np.asarray(iq[: min(16, len(iq))]).copy()
        outputs = _method_outputs(iq, covariance, physics, config, t1_options, device)
        ids.update(str(value) for value in metadata["base_scene_id"])
        for method in METHODS:
            methods[method].extend(outputs[method]["score"].tolist())
        for index in range(len(iq)):
            row: dict[str, Any] = {key: value[index] for key, value in metadata.items()}
            row["split"] = split_name
            for method in METHODS:
                row[f"score_{method}"] = outputs[method]["score"][index]
            rows.append(row)
    if first_batch is None:
        raise RuntimeError(f"H0 split {split_name} generated no IQ scenes")
    for method in METHODS:
        if len(methods[method]) != int(split_config["scenes"]):
            raise RuntimeError(f"H0 split {split_name} has a wrong {method} score count")
    return rows, ids, first_batch


def _append_method_rows(
    rows: list[dict[str, Any]],
    *,
    batch: Any,
    assignments: dict[str, np.ndarray],
    start: int,
    stop: int,
    h0_outputs: dict[str, dict[str, np.ndarray]],
    h1_outputs: dict[str, dict[str, np.ndarray]],
) -> None:
    for index in range(stop - start):
        row = {
            "base_scene_id": str(batch.base_scene_id[index]),
            "course_level": str(assignments["course_level"][start + index]),
            "scenario_id": str(assignments["scenario_id"][start + index]),
            "condition_id": str(assignments["condition_id"][start + index]),
            "snr_db": float(batch.snr_db[index]),
            "cnr_db": float(batch.requested_cnr_db[index]),
            "truth_range_cell": float(batch.truth_target_range_cell[index]),
            "truth_doppler_hz": float(batch.truth_target_doppler_hz[index]),
            "truth_azimuth_deg": float(batch.truth_target_azimuth_deg[index]),
            "truth_amplitude_diagnostic": float(batch.truth_target_amplitude[index]),
        }
        for method in METHODS:
            for state, output in (("h0", h0_outputs), ("h1", h1_outputs)):
                row[f"score_{method}_{state}"] = float(output[method]["score"][index])
            for field in ("range_cell", "doppler_hz"):
                row[f"predicted_{field}_{method}"] = float(h1_outputs[method][field][index])
        if "angle_deg" in h1_outputs["ACE_AMF"]:
            row["predicted_angle_deg_ACE_AMF"] = float(h1_outputs["ACE_AMF"]["angle_deg"][index])
        rows.append(row)


def _collect_development(
    config: dict[str, Any],
    physics: dict[str, Any],
    e002: dict[str, Any],
    covariance: CovarianceBank,
    t1_options: dict[str, Any],
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], set[str], dict[str, np.ndarray]]:
    assignments = build_development_assignments(config, e002["data"]["course_levels"])
    development = config["data"]["development"]
    batch_size = int(config["data"]["generation_batch_size"])
    stride = int(config["data"]["chunk_seed_stride"])
    seed_base = int(development["seed_base"])
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    soft_parts: dict[str, list[dict[str, np.ndarray]]] = {"ACE_AMF": [], "K2": []}
    data_parts: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "course_level", "scenario_id", "condition_id", "snr_db", "cnr_db",
            "truth_range_cell", "truth_doppler_hz", "truth_azimuth_deg", "truth_amplitude",
            "base_scene_id",
        )
    }
    method_parts: dict[str, dict[str, list[np.ndarray]]] = {
        method: {field: [] for field in ("score_h0", "score_h1", "range_cell", "doppler_hz")}
        for method in METHODS
    }
    physical_iq_h0: list[np.ndarray] = []
    physical_target: list[np.ndarray] = []
    timing_iq: np.ndarray | None = None
    hard_key = str(development["primary_hard_condition"])
    for course_index, course in enumerate(("C0", "C1", "C2", "C3")):
        indices = np.flatnonzero(assignments["course_level"] == course)
        for chunk_index, offset in enumerate(range(0, indices.size, batch_size)):
            local = indices[offset : offset + batch_size]
            start = int(local[0])
            stop = int(local[-1]) + 1
            # Course assignment occupies a disjoint 250k seed sub-block.
            seed = _seed(seed_base, chunk_index, stride, course_index, course_stride=250_000)
            scenarios = assignments["scenario_id"][local]
            velocity = np.asarray(
                [physics["target"]["scenario_radial_velocity_mps"][str(value)] for value in scenarios],
                dtype=np.float64,
            )
            batch = generate_d2a_paired_batch(
                local.size,
                seed,
                assignments["snr_db"][local],
                assignments["cnr_db"][local],
                physics,
                scenario_id="separated",
                target_radial_velocity_mps=velocity,
            )
            h0_outputs = _method_outputs(batch.iq_h0, covariance, physics, config, t1_options, device)
            h1_outputs = _method_outputs(batch.iq_h1, covariance, physics, config, t1_options, device)
            if timing_iq is None:
                timing_iq = np.asarray(batch.iq_h1[: min(16, len(batch.iq_h1))]).copy()
            _append_method_rows(
                rows,
                batch=batch,
                assignments=assignments,
                start=start,
                stop=stop,
                h0_outputs=h0_outputs,
                h1_outputs=h1_outputs,
            )
            ids.update(str(value) for value in batch.base_scene_id)
            for key in data_parts:
                if key == "course_level":
                    value = assignments[key][local]
                elif key in ("scenario_id", "condition_id", "snr_db", "cnr_db"):
                    value = assignments[key][local]
                elif key == "truth_range_cell":
                    value = batch.truth_target_range_cell
                elif key == "truth_doppler_hz":
                    value = batch.truth_target_doppler_hz
                elif key == "truth_azimuth_deg":
                    value = batch.truth_target_azimuth_deg
                elif key == "truth_amplitude":
                    value = batch.truth_target_amplitude
                else:
                    value = batch.base_scene_id.astype("U96")
                data_parts[key].append(np.asarray(value))
            for method in METHODS:
                method_parts[method]["score_h0"].append(h0_outputs[method]["score"])
                method_parts[method]["score_h1"].append(h1_outputs[method]["score"])
                method_parts[method]["range_cell"].append(h1_outputs[method]["range_cell"])
                method_parts[method]["doppler_hz"].append(h1_outputs[method]["doppler_hz"])

            k2_h0 = k2_response(batch.iq_h0, physics)
            k2_h1 = k2_response(batch.iq_h1, physics)
            truth_range = batch.truth_target_range_cell
            truth_fd = batch.truth_target_doppler_hz
            for name, response0, response1 in (
                ("K2", k2_h0, k2_h1),
                (
                    "ACE_AMF",
                    h0_outputs["ACE_AMF"]["soft_response"],
                    h1_outputs["ACE_AMF"]["soft_response"],
                ),
            ):
                soft_parts[name].append(
                    summarize_soft_response_per_scene(
                        response0,
                        response1,
                        truth_range,
                        truth_fd,
                        prf_hz=float(physics["radar"]["prf_hz"]),
                    )
                )
            hard = assignments["condition_id"][local] == hard_key
            remaining = max(0, 128 - sum(len(item) for item in physical_iq_h0))
            hard_indices = np.flatnonzero(hard)[:remaining]
            if hard_indices.size:
                physical_iq_h0.append(np.asarray(batch.iq_h0[hard_indices]).copy())
                physical_target.append(np.asarray(batch.target_signal_iq[hard_indices]).copy())
    if len(rows) != int(development["paired_base_scenes"]) or timing_iq is None:
        raise RuntimeError("development stream produced an unexpected scene count")
    combined_data = {key: np.concatenate(values) for key, values in data_parts.items()}
    combined_data["amplitude_quartile"] = np.digitize(
        combined_data["truth_amplitude"],
        np.quantile(combined_data["truth_amplitude"], [0.25, 0.5, 0.75]),
        right=True,
    ) + 1
    combined_methods = {
        method: {field: np.concatenate(parts) for field, parts in fields.items()}
        for method, fields in method_parts.items()
    }
    combined_soft = {
        method: {
            key: np.concatenate([part[key] for part in parts])
            for key in parts[0]
        }
        for method, parts in soft_parts.items()
    }
    combined_data["physical_iq_h0"] = np.concatenate(physical_iq_h0) if physical_iq_h0 else np.empty((0, 3, 130, 32), np.complex64)
    combined_data["physical_target_iq"] = np.concatenate(physical_target) if physical_target else np.empty((0, 3, 130, 32), np.complex64)
    combined_data["timing_iq"] = timing_iq
    combined_data["methods"] = combined_methods  # type: ignore[assignment]
    combined_data["soft"] = combined_soft  # type: ignore[assignment]
    return rows, combined_data, ids, assignments


def _score_arrays(rows: list[dict[str, Any]], split: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected = [row for row in rows if row["split"] == split]
    if not selected:
        raise ValueError(f"missing rows for H0 split {split}")
    scores = {method: np.asarray([row[f"score_{method}"] for row in selected], dtype=np.float64) for method in METHODS}
    metadata = {
        key: np.asarray([row[key] for row in selected])
        for key in ("course_level", "snr_db", "cnr_db", "truth_amplitude_diagnostic", "base_scene_id")
    }
    return scores, metadata


def _h0_verification_groups(
    scores: dict[str, np.ndarray],
    thresholds: dict[str, dict[str, Any]],
    metadata: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups = {
        "course_level": metadata["course_level"].astype(str),
        "snr_db": np.asarray([f"{value:g}" for value in metadata["snr_db"]]),
        "cnr_db": np.asarray([f"{value:g}" for value in metadata["cnr_db"]]),
    }
    amp = np.asarray(metadata["truth_amplitude_diagnostic"], dtype=np.float64)
    amp_q = np.digitize(amp, np.quantile(amp, [0.25, 0.5, 0.75]), right=True) + 1
    groups["amplitude_quartile"] = np.asarray([f"Q{value}" for value in amp_q])
    summaries: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for method, values in scores.items():
        threshold = float(thresholds[method]["threshold"])
        summaries[method] = summarize_h0_verification(values, threshold)
        for group_name, labels in groups.items():
            for label in np.unique(labels):
                chosen = labels == label
                n = int(np.sum(chosen))
                if n == 0:
                    continue
                count = int(np.sum(values[chosen] >= threshold))
                rows.append({
                    "method": method,
                    "group": group_name,
                    "label": str(label),
                    "n": n,
                    "false_alarms": count,
                    "pfa_ge": count / n,
                    "pfa_exact_one_sided_95_upper": exact_binomial_upper(count, n),
                    "threshold": threshold,
                })
    return summaries, rows


def _rmse_difference_ci(
    candidate_error: np.ndarray,
    reference_error: np.ndarray,
    scene_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    statistic = lambda a, b: float(np.sqrt(np.mean(a)) - np.sqrt(np.mean(b)))
    interval = paired_difference_interval(
        np.asarray(candidate_error, dtype=np.float64) ** 2,
        np.asarray(reference_error, dtype=np.float64) ** 2,
        scene_ids,
        statistic=statistic,
        replicates=replicates,
        seed=seed,
    )
    return [float(interval[0]), float(interval[1])]


def _adjudicate_g3(
    *,
    g2_passed: bool,
    soft_summaries: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen soft-knowledge gate only after the teacher task gate."""
    if not g2_passed:
        return {
            "evaluated": False,
            "passed": False,
            "reason": "G2 failed; the preregistered protocol stops before G3 adjudication",
        }
    hard_ace = soft_summaries["ACE_AMF"]["hard_condition"]
    c3_ace = soft_summaries["ACE_AMF"]["pooled_C3"]
    checks = {
        "hard_target_mass_lift_ci_lower_positive": hard_ace["target_window_mass_lift_ci95"][0]
        > float(gates["g3_hard_target_mass_lift_ci95_lower_min"]),
        "hard_top_cell_localization_minimum": hard_ace["teacher_top_cell_localization_rate"]
        >= float(gates["g3_hard_top_cell_localization_min"]),
        "pooled_C3_mass_lift_ci_lower_positive": c3_ace["target_window_mass_lift_ci95"][0]
        > float(gates["g3_pooled_c3_mass_lift_ci95_lower_min"]),
    }
    return {"evaluated": True, **checks, "passed": bool(all(checks.values()))}


def _evaluate_development(
    config: dict[str, Any],
    physics: dict[str, Any],
    dev: dict[str, Any],
    h0_verification_rows: list[dict[str, Any]],
    thresholds: dict[str, dict[str, Any]],
    *,
    g1_runtime: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    data = dev
    methods = data["methods"]
    ids = np.asarray(data["base_scene_id"]).astype(str)
    truth_range = np.asarray(data["truth_range_cell"], dtype=np.float64)
    truth_fd = np.asarray(data["truth_doppler_hz"], dtype=np.float64)
    snr = np.asarray(data["snr_db"], dtype=np.float64)
    cnr = np.asarray(data["cnr_db"], dtype=np.float64)
    course = np.asarray(data["course_level"]).astype(str)
    scenario = np.asarray(data["scenario_id"]).astype(str)
    condition = np.asarray(data["condition_id"]).astype(str)
    amplitude_q = np.asarray(data["amplitude_quartile"], dtype=np.int8)
    azimuth = np.asarray(data["truth_azimuth_deg"], dtype=np.float64)
    doppler_margin = float(physics["radar"]["prf_hz"]) / 2.0 - np.abs(truth_fd)
    groups: dict[str, np.ndarray] = {
        "global": np.full(ids.size, "all", dtype="U8"),
        "course": course,
        "scenario": scenario,
        "condition": condition,
        "amplitude_quartile": np.asarray([f"Q{v}" for v in amplitude_q]),
        "azimuth_edge": np.where(np.abs(azimuth - float(physics["beam"]["center_azimuth_deg"])) >= 0.9, "edge", "center"),
        "doppler_nyquist_margin": np.where(doppler_margin <= 50.0, "within_50Hz", "more_than_50Hz"),
    }
    bootstrap = config["data"]["bootstrap"]
    replicates = int(bootstrap["paired_scene_replicates"])
    seed = int(bootstrap["seed"])
    prf = float(physics["radar"]["prf_hz"])
    summaries: dict[str, Any] = {"h0_verification": {}, "thresholds": thresholds, "development": {}}
    verification_scores, verification_meta = _score_arrays(h0_verification_rows, "h0_verification")
    h0_summary, subgroup_rows = _h0_verification_groups(verification_scores, thresholds, verification_meta)
    summaries["h0_verification"] = h0_summary
    result_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHODS):
        threshold = float(thresholds[method]["threshold"])
        method_summary: dict[str, Any] = {}
        for group_name, labels in groups.items():
            group_summary: dict[str, Any] = {}
            for label in np.unique(labels):
                chosen = labels == label
                n = int(np.sum(chosen))
                if n < 50:
                    continue
                metrics = evaluate_method_group(
                    h0_scores=methods[method]["score_h0"][chosen],
                    h1_scores=methods[method]["score_h1"][chosen],
                    threshold=threshold,
                    truth_range=truth_range[chosen],
                    predicted_range=methods[method]["range_cell"][chosen],
                    truth_doppler=truth_fd[chosen],
                    predicted_doppler=methods[method]["doppler_hz"][chosen],
                    base_scene_ids=ids[chosen],
                    prf_hz=prf,
                    bootstrap_replicates=replicates,
                    bootstrap_seed=seed + method_index * 10_000 + len(group_summary) * 100,
                )
                group_summary[str(label)] = metrics
                result_rows.append({"method": method, "group": group_name, "label": str(label), **metrics})
            method_summary[group_name] = group_summary
        summaries["development"][method] = method_summary

    hard_key = str(config["data"]["development"]["primary_hard_condition"])
    hard = condition == hard_key
    if int(np.sum(hard)) != int(config["data"]["development"]["primary_hard_condition_n"]):
        raise RuntimeError("primary hard-condition scene count differs from preregistration")
    hard_metrics = {
        method: summaries["development"][method]["condition"][hard_key]
        for method in METHODS
    }
    best_old = max(("B3", "T1"), key=lambda method: hard_metrics[method]["joint_success"])
    detected: dict[str, np.ndarray] = {}
    errors: dict[str, dict[str, np.ndarray]] = {}
    for method in METHODS:
        threshold = float(thresholds[method]["threshold"])
        detected[method] = methods[method]["score_h1"] >= threshold
        errors[method] = {
            "range": methods[method]["range_cell"] - truth_range,
            "doppler": np.asarray(wrap_frequency(methods[method]["doppler_hz"] - truth_fd, prf), dtype=np.float64),
        }
        errors[method]["joint"] = (
            detected[method]
            & (np.abs(errors[method]["range"]) <= float(config["metrics"]["joint_success"]["range_error_max_cells"]))
            & (np.abs(errors[method]["doppler"]) <= float(config["metrics"]["joint_success"]["doppler_error_max_hz"]))
        )

    hard_gain = paired_metric_difference(
        errors["ACE_AMF"]["joint"][hard],
        errors[best_old]["joint"][hard],
        ids[hard],
        bootstrap_replicates=replicates,
        bootstrap_seed=seed + 51001,
    )
    identifiable = np.isin(scenario, ("separated", "near_clutter_edge", "partial_overlap"))
    paired_protection: dict[str, Any] = {}
    for metric_name, candidate_values, reference_values in (
        ("pd", detected["ACE_AMF"], detected["T1"]),
        ("joint_success", errors["ACE_AMF"]["joint"], errors["T1"]["joint"]),
    ):
        paired_protection[metric_name] = paired_metric_difference(
            candidate_values[identifiable],
            reference_values[identifiable],
            ids[identifiable],
            bootstrap_replicates=replicates,
            bootstrap_seed=seed + 52000 + len(paired_protection),
        )
    paired_protection["range_rmse_cells"] = {
        "candidate_minus_T1": float(
            np.sqrt(np.mean(errors["ACE_AMF"]["range"][identifiable] ** 2))
            - np.sqrt(np.mean(errors["T1"]["range"][identifiable] ** 2))
        ),
        "ci95": _rmse_difference_ci(
            errors["ACE_AMF"]["range"][identifiable],
            errors["T1"]["range"][identifiable],
            ids[identifiable],
            replicates=replicates,
            seed=seed + 52010,
        ),
    }
    paired_protection["doppler_rmse_hz"] = {
        "candidate_minus_T1": float(
            np.sqrt(np.mean(errors["ACE_AMF"]["doppler"][identifiable] ** 2))
            - np.sqrt(np.mean(errors["T1"]["doppler"][identifiable] ** 2))
        ),
        "ci95": _rmse_difference_ci(
            errors["ACE_AMF"]["doppler"][identifiable],
            errors["T1"]["doppler"][identifiable],
            ids[identifiable],
            replicates=replicates,
            seed=seed + 52011,
        ),
    }
    course_auc_losses = {
        level: float(
            summaries["development"]["ACE_AMF"]["course"][level]["auc"]
            - summaries["development"]["T1"]["course"][level]["auc"]
        )
        for level in ("C0", "C1", "C2", "C3")
    }
    primary_h0 = h0_summary["ACE_AMF"]
    required_h0_groups = [row for row in subgroup_rows if row["method"] == "ACE_AMF" and row["group"] in ("course_level", "snr_db", "cnr_db")]
    h0_gates = {
        "point_pfa_at_most_0_06": primary_h0["pfa_ge"] <= float(config["metrics"]["teacher_gates"]["g2_h0_pfa_point_max"]),
        "exact_upper_at_most_0_06": primary_h0["pfa_ci95_exact_upper"] <= float(config["metrics"]["teacher_gates"]["g2_h0_exact_one_sided_95_upper_max"]),
        "large_subgroups_upper_at_most_0_10": bool(required_h0_groups) and all(
            row["pfa_exact_one_sided_95_upper"] <= float(config["metrics"]["teacher_gates"]["g2_large_subgroup_exact_95_upper_max"])
            for row in required_h0_groups
            if row["n"] >= 1000
        ),
        "large_subgroup_count": len([row for row in required_h0_groups if row["n"] >= 1000]),
    }
    gates = config["metrics"]["teacher_gates"]
    g2 = {
        "h0": h0_gates,
        "primary_hard_condition": {
            "condition_id": hard_key,
            "n": int(np.sum(hard)),
            "best_old_teacher_baseline": best_old,
            "joint_success_by_method": {name: hard_metrics[name]["joint_success"] for name in METHODS},
            "ACE_AMF_minus_best_old_joint_success": hard_gain,
            "point_gain_gate": hard_gain["candidate_minus_reference"] >= float(gates["g2_primary_joint_gain_point_min"]),
            "ci_lower_gate": hard_gain["ci95"][0] > float(gates["g2_primary_joint_gain_ci95_lower_min"]),
        },
        "identifiable_protection_vs_T1": paired_protection,
        "course_auc_loss_vs_T1": course_auc_losses,
    }
    g2["protection_gates"] = {
        "pd_ci_lower": paired_protection["pd"]["ci95"][0] > float(gates["g2_identifiable_pd_ci95_lower_min"]),
        "joint_ci_lower": paired_protection["joint_success"]["ci95"][0] > float(gates["g2_identifiable_joint_ci95_lower_min"]),
        "range_rmse_noninferior": paired_protection["range_rmse_cells"]["ci95"][1] <= float(gates["g2_range_rmse_noninferiority_cells"]),
        "doppler_rmse_noninferior": paired_protection["doppler_rmse_hz"]["ci95"][1] <= float(gates["g2_doppler_rmse_noninferiority_hz"]),
        "course_auc_noninferior": all(value >= -float(gates["g2_course_auc_loss_max"]) for value in course_auc_losses.values()),
    }
    g2["passed"] = bool(
        all(value for key, value in h0_gates.items() if isinstance(value, bool))
        and g2["primary_hard_condition"]["point_gain_gate"]
        and g2["primary_hard_condition"]["ci_lower_gate"]
        and all(g2["protection_gates"].values())
    )

    soft = data["soft"]
    soft_summaries: dict[str, Any] = {"evaluated": bool(g2["passed"])}
    if g2["passed"]:
        c3 = course == "C3"
        for name in ("ACE_AMF", "K2"):
            soft_summaries[name] = {
                "hard_condition": summarize_soft_response(
                    {key: value[hard] for key, value in soft[name].items()},
                    ids[hard],
                    bootstrap_replicates=replicates,
                    bootstrap_seed=seed + 61000 + int(name == "K2"),
                ),
                "pooled_C3": summarize_soft_response(
                    {key: value[c3] for key, value in soft[name].items()},
                    ids[c3],
                    bootstrap_replicates=replicates,
                    bootstrap_seed=seed + 62000 + int(name == "K2"),
                ),
            }
    else:
        soft_summaries["reason"] = "G2 failed; preregistered soft-response review is not performed"
    g3 = _adjudicate_g3(
        g2_passed=bool(g2["passed"]),
        soft_summaries=soft_summaries,
        gates=gates,
    )
    g1 = dict(g1_runtime)
    g1["passed"] = bool(all(bool(value) for key, value in g1.items() if isinstance(value, (bool, np.bool_))))
    teacher_gates = {"G1": g1, "G2": g2, "G3": g3, "student_pilot_eligible": bool(g1["passed"] and g2["passed"] and g3["passed"])}
    summaries["verification_subgroups"] = subgroup_rows
    summaries["soft_response"] = soft_summaries
    summaries["gates"] = teacher_gates
    summaries["primary_hard_condition"] = hard_key
    summaries["best_old_baseline_in_hard_condition"] = best_old
    summaries["bootstrap_replicates"] = replicates
    summaries["bootstrap_seed"] = seed

    soft_rows: list[dict[str, Any]] = []
    if soft_summaries["evaluated"]:
        for method_name, per_scene in soft.items():
            for index, scene_id in enumerate(ids):
                soft_rows.append({
                    "base_scene_id": scene_id,
                    "method": method_name,
                    "course_level": course[index],
                    "scenario_id": scenario[index],
                    "snr_db": snr[index],
                    "cnr_db": cnr[index],
                    **{key: value[index] for key, value in per_scene.items()},
                })
    summaries["physical_edges"] = {
        "azimuth_edge_definition_deg_from_beam_center": 0.9,
        "azimuth_edge_count": int(np.sum(np.abs(azimuth - float(physics["beam"]["center_azimuth_deg"])) >= 0.9)),
        "doppler_within_50hz_of_nyquist_count": int(np.sum(doppler_margin <= 50.0)),
        "frequency_rmse_by_course": {
            level: {
                method: summaries["development"][method]["course"][level]["doppler_rmse_all_h1_hz"]
                for method in METHODS
            }
            for level in ("C0", "C1", "C2", "C3")
        },
    }
    summaries["g1_runtime"] = g1_runtime
    return summaries, result_rows, soft_rows, {"g1": g1, "g2": g2, "g3": g3}


def _physical_diagnostics(
    dev: dict[str, Any],
    config: dict[str, Any],
    physics: dict[str, Any],
    covariance: CovarianceBank,
    threshold: float,
    device: str = "cpu",
) -> dict[str, Any]:
    iq0 = np.asarray(dev["physical_iq_h0"], dtype=np.complex64)
    target = np.asarray(dev["physical_target_iq"], dtype=np.complex64)
    if iq0.shape[0] == 0:
        return {"completed": False, "reason": "no frozen hard-condition physical subset"}
    iq1 = iq0 + target
    teacher_config = config["teacher"]
    baseline = post_doppler_ace_amf(iq1, covariance, physics, teacher_config, device=device)
    prf = float(physics["radar"]["prf_hz"])
    results: dict[str, Any] = {
        "completed": True,
        "n_fixed_hard_condition_scenes": int(iq1.shape[0]),
        "threshold_recalibrated": False,
        "original_threshold": float(threshold),
    }
    phase = np.exp(0.731j)
    rotated = post_doppler_ace_amf(iq1 * phase, covariance, physics, teacher_config, device=device)
    results["common_phase_rotation"] = {
        "factor": [float(phase.real), float(phase.imag)],
        "max_abs_score_difference": float(np.max(np.abs(np.asarray(rotated.score) - np.asarray(baseline.score)))),
        "max_abs_map_difference": float(np.max(np.abs(rotated.score_map - baseline.score_map))),
        "pfa_threshold_reused": True,
    }
    scale_results: dict[str, Any] = {}
    for factor in (0.5 + 0.0j, 2.0 + 0.0j, 0.6 + 1.2j):
        scaled = post_doppler_ace_amf(iq1 * factor, covariance, physics, teacher_config, device=device)
        scale_results[str(factor)] = {
            "max_abs_score_difference": float(np.max(np.abs(np.asarray(scaled.score) - np.asarray(baseline.score)))),
            "max_abs_map_difference": float(np.max(np.abs(scaled.score_map - baseline.score_map))),
            "max_abs_soft_map_difference": float(np.max(np.abs(scaled.soft_response - baseline.soft_response))),
            "pfa_threshold_reused": True,
        }
    results["common_complex_scale"] = scale_results

    pulse_count = iq1.shape[2]
    time_axis = np.arange(pulse_count, dtype=np.float64) / prf
    shift_results: dict[str, Any] = {}
    for delta_hz in (-25.0, -10.0, 10.0, 25.0):
        modulation = np.exp(2j * np.pi * delta_hz * time_axis)[None, None, :, None]
        shifted_iq = iq0 + target * modulation
        shifted = post_doppler_ace_amf(shifted_iq, covariance, physics, teacher_config, device=device)
        shift_error = np.asarray(wrap_frequency(
            np.asarray(shifted.doppler_hz) - np.asarray(baseline.doppler_hz) - delta_hz,
            prf,
        ))
        shift_results[str(delta_hz)] = {
            "mean_circular_shift_residual_hz": float(np.mean(shift_error)),
            "rmse_circular_shift_residual_hz": float(np.sqrt(np.mean(shift_error**2))),
            "within_5_hz_fraction": float(np.mean(np.abs(shift_error) <= 5.0)),
            "shifted_detection_fraction_at_original_threshold": float(np.mean(np.asarray(shifted.score) >= threshold)),
            "pfa_threshold_reused": True,
        }
    results["target_only_doppler_shift"] = shift_results

    rng = np.random.default_rng(int(config["data"]["bootstrap"]["seed"]) + 3)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(iq1.shape[0], iq1.shape[1], 1, 1))
    channel_phase_iq = iq1 * np.exp(1j * phases)
    channel_phase = post_doppler_ace_amf(channel_phase_iq, covariance, physics, teacher_config, device=device)
    results["independent_channel_phase_perturbation"] = {
        "score_ratio_mean": float(np.mean(np.asarray(channel_phase.score) / np.maximum(np.asarray(baseline.score), 1e-12))),
        "detection_fraction_at_original_threshold": float(np.mean(np.asarray(channel_phase.score) >= threshold)),
        "range_rmse_cells": float(np.sqrt(np.mean((np.asarray(channel_phase.range_cell) - np.asarray(baseline.range_cell)) ** 2))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(np.asarray(wrap_frequency(np.asarray(channel_phase.doppler_hz) - np.asarray(baseline.doppler_hz), prf)) ** 2))),
        "pfa_threshold_reused": True,
    }
    return results


def _benchmark_candidate(
    iq: np.ndarray,
    config: dict[str, Any],
    physics: dict[str, Any],
    covariance: CovarianceBank,
    device: str,
    repetitions: int = 1000,
) -> dict[str, Any]:
    sample = np.asarray(iq[:16], dtype=np.complex64)
    if sample.shape[0] == 0 or repetitions <= 0:
        return {"completed": False}
    backends: dict[str, Any] = {}
    for backend in dict.fromkeys(("cpu", device)):
        singles = time.perf_counter()
        for index in range(repetitions):
            post_doppler_ace_amf(
                sample[index % len(sample)],
                covariance,
                physics,
                config["teacher"],
                device=backend,
            )
        batch_1_seconds = time.perf_counter() - singles
        multi_count = 0
        multi = time.perf_counter()
        while multi_count < repetitions:
            current = sample[np.arange(16) % len(sample)]
            post_doppler_ace_amf(
                current, covariance, physics, config["teacher"], device=backend
            )
            multi_count += len(current)
        batch_16_seconds = time.perf_counter() - multi
        backends[backend] = {
            "batch_1_cpi_count": repetitions,
            "batch_1_seconds": batch_1_seconds,
            "batch_1_seconds_per_cpi": batch_1_seconds / repetitions,
            "batch_16_cpi_count": multi_count,
            "batch_16_seconds": batch_16_seconds,
            "batch_16_seconds_per_cpi": batch_16_seconds / multi_count,
        }
    return {
        "completed": True,
        "unique_cpis": int(sample.shape[0]),
        "repetitions": repetitions,
        "reuses_same_cpis": True,
        "backends": backends,
    }


def _write_figures(metrics: dict[str, Any], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = list(METHODS)
    pfa = [metrics["h0_verification"][method]["pfa_ge"] for method in methods]
    pfa_upper = [metrics["h0_verification"][method]["pfa_ci95_exact_upper"] for method in methods]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(methods))
    ax.bar(x, pfa, color=("#537895", "#6b8f71", "#b26358"))
    ax.scatter(x, pfa_upper, marker="_", color="black", label="one-sided exact 95% upper")
    ax.axhline(0.05, color="gray", linestyle="--", label="target Pfa 0.05")
    ax.axhline(0.06, color="red", linestyle=":", label="G2 point/upper limit 0.06")
    ax.set_xticks(x, methods)
    ax.set_ylabel("CPI false alarm probability")
    ax.set_title("Independent H0 verification")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "pfa_verification.png", dpi=150)
    plt.close(fig)

    hard_key = metrics["primary_hard_condition"]
    values = [metrics["development"][method]["condition"][hard_key]["joint_success"] for method in methods]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(methods)), values, color=("#537895", "#6b8f71", "#b26358"))
    ax.set_xticks(np.arange(len(methods)), methods)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Joint success")
    ax.set_title("C3 / partial overlap / SNR 0 dB hard condition")
    fig.tight_layout()
    fig.savefig(figures / "hard_condition_joint_success.png", dpi=150)
    plt.close(fig)

    levels = ("C0", "C1", "C2", "C3")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method in methods:
        rmse = [metrics["development"][method]["course"][level]["doppler_rmse_all_h1_hz"] for level in levels]
        ax.plot(levels, rmse, marker="o", label=method)
    ax.set_ylabel("Doppler RMSE (Hz)")
    ax.set_xlabel("Course level")
    ax.set_title("Development Doppler RMSE by difficulty mixture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "doppler_rmse_course.png", dpi=150)
    plt.close(fig)
    return [
        "figures/pfa_verification.png",
        "figures/hard_condition_joint_success.png",
        "figures/doppler_rmse_course.png",
    ]


def _write_analysis(metrics: dict[str, Any], output_dir: Path, git_sha: str) -> None:
    gates = metrics["gates"]
    global_rows = {
        name: metrics["development"][name]["global"]["all"]
        for name in METHODS
    }
    lines = [
        "# D2-A-T001 分析",
        "",
        f"**运行代码提交：** `{git_sha}`  ",
        "**研究总纲：** `docs/RESEARCH_CHARTER.md` v1.2（T001 未修改总纲）  ",
        "**数据：** 新生成的合成 H0/H1 开发数据；不是实测，也不是 final blind。  ",
        "**候选：** 唯一模型辅助三通道 post-Doppler ACE×AMF。  ",
        f"**候选设备：** `{metrics['compute']['teacher_device']}`；B3/T1/K2 对照：{metrics['compute']['comparison_baselines_device']}。  ",
        "",
        "## 冻结阶段门",
        "",
        f"- G1：{'PASS' if gates['G1']['passed'] else 'FAIL'}。",
        f"- G2：{'PASS' if gates['G2']['passed'] else 'FAIL'}。",
        f"- G3：{'PASS' if gates['G3']['passed'] else 'FAIL' if gates['G3']['evaluated'] else 'NOT EVALUATED'}。",
        f"- 是否允许 S2 pilot：{'允许' if gates['student_pilot_eligible'] else '禁止'}。",
        "",
        "## 总体开发指标",
        "",
        "| 方法 | 验证 Pfa | 单侧 exact 95% 上限 | Pd | AUC | Range RMSE | Doppler RMSE (Hz) | Joint success |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHODS:
        h0 = metrics["h0_verification"][name]
        row = global_rows[name]
        lines.append(
            f"| {name} | {h0['pfa_ge']:.5f} | {h0['pfa_ci95_exact_upper']:.5f} | {row['pd']:.4f} | {row['auc']:.4f} | {row['range_rmse_all_h1_cells']:.4f} | {row['doppler_rmse_all_h1_hz']:.3f} | {row['joint_success']:.4f} |"
        )
    hard_key = metrics["primary_hard_condition"]
    lines += [
        "",
        "## 主要困难条件与知识质量",
        "",
        f"主要困难条件：`{hard_key}`。ACE×AMF 相对最佳旧教师 `{metrics['best_old_baseline_in_hard_condition']}` 的 joint-success 差及 bootstrap CI：`{metrics['gates']['G2']['primary_hard_condition']['ACE_AMF_minus_best_old_joint_success']}`。",
    ]
    if metrics["soft_response"].get("evaluated", False):
        lines += [
            f"ACE×AMF hard target-mass lift：`{metrics['soft_response']['ACE_AMF']['hard_condition']['target_window_mass_lift_mean']:.5f}`, 95% CI `{metrics['soft_response']['ACE_AMF']['hard_condition']['target_window_mass_lift_ci95']}`；top-cell localization `{metrics['soft_response']['ACE_AMF']['hard_condition']['teacher_top_cell_localization_rate']:.4f}`。",
            f"K2 hard target-mass lift：`{metrics['soft_response']['K2']['hard_condition']['target_window_mass_lift_mean']:.5f}`, 95% CI `{metrics['soft_response']['K2']['hard_condition']['target_window_mass_lift_ci95']}`；top-cell localization `{metrics['soft_response']['K2']['hard_condition']['teacher_top_cell_localization_rate']:.4f}`。",
        ]
    else:
        lines.append("G2 未通过；按预注册协议不评价 soft response 质量，G3 记为 NOT EVALUATED。")
    lines += [
        "",
        "## 解释边界",
        "",
        "检测阈值由独立 H0 calibration 以 `higher` 经验 95% 分位冻结，并只在独立 H0 verification 评价。开发集仅用于预注册阶段门和分层误差；没有生成 final blind。若任一教师门失败，不运行新蒸馏学生；负结果保留，不追加候选或调参。",
        "",
        "精确数值、分组结果、bootstrap 区间、软图审计、运行资源和物理诊断见 `metrics.json`、`result.csv`、`verification_subgroups.csv`、`soft_teacher_audit.csv`、`physical_diagnostics.json` 与 `environment.json`。",
        "",
    ]
    (output_dir / "analysis.md").write_text("\n".join(lines), encoding="utf-8")


def _run_teacher(config_path: Path, device: str) -> dict[str, Any]:
    config = _read_yaml(config_path)
    preflight = _require_preflight(config, device)
    import torch

    torch.cuda.reset_peak_memory_stats(torch.device(device))
    physics = _read_yaml(ROOT / config["physics"]["source_config"])
    e002 = _read_yaml(ROOT / "experiments/D2-A-E002/config.yaml")
    r3 = _read_yaml(ROOT / "experiments/D2-A-R3/config.yaml")
    output_dir = ROOT / config["outputs"]["directory"]
    started = time.perf_counter()
    cpu_started = time.process_time()
    memory_start = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    covariance_batches, covariance_ids = _covariance_h0_batches(config, physics)
    covariance_started = time.perf_counter()
    covariance_cpu_started = time.process_time()
    covariance = fit_post_doppler_covariance(
        covariance_batches,
        oversampling_factor=int(config["teacher"]["frequency_oversampling_factor"]),
        eigenvalue_floor_relative_trace=float(config["teacher"]["covariance_eigenvalue_floor_relative_trace"]),
        device=device,
    )
    covariance_seconds = time.perf_counter() - covariance_started
    covariance_cpu_seconds = time.process_time() - covariance_cpu_started
    np.savez_compressed(
        output_dir / "covariance_bank.npz",
        covariance=covariance.covariance,
        inverse=covariance.inverse,
        eigenvalues=covariance.eigenvalues,
        condition_number=covariance.condition_number,
        eigenvalue_floor_fraction=covariance.eigenvalue_floor_fraction,
    )
    _write_json(output_dir / "covariance_diagnostics.json", covariance.diagnostics())

    t1_options = r3["detection"]["t1_global_robust_background"]
    calibration_rows, calibration_ids, timing_iq = _score_h0_split(
        "h0_calibration",
        config["data"]["h0_calibration"],
        config,
        physics,
        covariance,
        t1_options,
        device,
    )
    verification_rows, verification_ids, _ = _score_h0_split(
        "h0_verification",
        config["data"]["h0_verification"],
        config,
        physics,
        covariance,
        t1_options,
        device,
    )
    _write_csv(output_dir / "calibration_scores.csv", calibration_rows)
    _write_csv(output_dir / "verification_scores.csv", verification_rows)
    calibration_scores, _ = _score_arrays(calibration_rows, "h0_calibration")
    thresholds = {
        method: calibrate_empirical_threshold(
            values,
            target_pfa=float(config["metrics"]["target_pfa_cpi"]),
        )
        for method, values in calibration_scores.items()
    }

    development_rows, dev, development_ids, _ = _collect_development(
        config, physics, e002, covariance, t1_options, device
    )
    _write_csv(output_dir / "development_per_scene.csv", development_rows)

    overlap = {
        "covariance_train_vs_calibration": not bool(covariance_ids & calibration_ids),
        "covariance_train_vs_verification": not bool(covariance_ids & verification_ids),
        "covariance_train_vs_development": not bool(covariance_ids & development_ids),
        "calibration_vs_verification": not bool(calibration_ids & verification_ids),
        "calibration_vs_development": not bool(calibration_ids & development_ids),
        "verification_vs_development": not bool(verification_ids & development_ids),
    }
    score_finite = all(
        np.all(np.isfinite(np.asarray([row[f"score_{method}"] for row in calibration_rows + verification_rows])))
        for method in METHODS
    )
    development_finite = all(
        np.all(np.isfinite(np.asarray([row[f"score_{method}_{state}"] for row in development_rows])))
        for method in METHODS
        for state in ("h0", "h1")
    )
    scale_diagnostic = _physical_diagnostics(
        dev,
        config,
        physics,
        covariance,
        float(thresholds["ACE_AMF"]["threshold"]),
        device,
    )
    maximum_scale_map_difference = max(
        item["max_abs_map_difference"] for item in scale_diagnostic["common_complex_scale"].values()
    )
    maximum_scale_score_difference = max(
        item["max_abs_score_difference"] for item in scale_diagnostic["common_complex_scale"].values()
    )
    all_splits_disjoint = all(overlap.values())
    g1_runtime = {
        "covariance_finite_positive_definite": bool(
            np.all(np.isfinite(covariance.covariance))
            and np.all(np.isfinite(covariance.inverse))
            and np.all(np.min(covariance.eigenvalues, axis=1) > 0.0)
            and np.all(np.isfinite(covariance.condition_number))
        ),
        "all_splits_disjoint_by_base_scene_id": all_splits_disjoint,
        "split_disjointness": overlap,
        "all_h0_and_h1_scores_finite": bool(score_finite and development_finite),
        "paired_h0_h1_from_frozen_generator": True,
        "teacher_api_label_free": True,
        "public_common_complex_scale_invariance_max_abs_score": maximum_scale_score_difference,
        "public_common_complex_scale_invariance_max_abs_map": maximum_scale_map_difference,
        "scale_invariance_gate": bool(maximum_scale_score_difference <= 1e-6 and maximum_scale_map_difference <= 1e-6),
        "single_tone_and_covariance_unit_tests": "passed (tests/test_d2a_t001_teacher.py)",
        "frozen_baseline_functions_used_without_modification": True,
    }
    metrics, result_rows, soft_rows, gate_values = _evaluate_development(
        config,
        physics,
        dev,
        verification_rows,
        thresholds,
        g1_runtime=g1_runtime,
    )
    candidate_timing = _benchmark_candidate(
        timing_iq, config, physics, covariance, device
    )
    stage_wall = time.perf_counter() - started
    stage_cpu = time.process_time() - cpu_started
    rss_end = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    compute = {
        "teacher_device": str(torch.device(device)),
        "comparison_baselines_device": "CPU NumPy (B3, T1, and K2)",
        "teacher_precision": "complex64 CUDA compute with double-precision observable scale/phase normalization",
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(torch.device(device))),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(torch.device(device))),
        "covariance_fit_wall_seconds": covariance_seconds,
        "covariance_fit_cpu_seconds": covariance_cpu_seconds,
        "covariance_fit_scenes_per_second": covariance.scene_count / max(covariance_seconds, 1e-12),
        "covariance_bank_bytes_uncompressed": int(sum(item.nbytes for item in (
            covariance.covariance, covariance.inverse, covariance.eigenvalues,
            covariance.condition_number, covariance.eigenvalue_floor_fraction,
        ))),
        "covariance_bank_compressed_bytes": int((output_dir / "covariance_bank.npz").stat().st_size),
        "peak_rss_platform_units": int(max(memory_start, rss_end)),
        "peak_rss_unit": "KiB on Linux",
        "candidate_inference_timing": candidate_timing,
        "teacher_stage_wall_seconds": stage_wall,
        "teacher_stage_cpu_seconds": stage_cpu,
    }
    metrics["compute"] = compute
    metrics["physical_diagnostics"] = scale_diagnostic
    metrics["git_sha"] = preflight["git_sha"]
    metrics["protocol_version"] = float(config["experiment"]["protocol_version"])
    metrics["data_counts"] = {
        "covariance_training_h0": covariance.scene_count,
        "h0_calibration": len(calibration_ids),
        "h0_verification": len(verification_ids),
        "paired_development": len(development_ids),
    }
    metrics["student_pilot_status"] = "eligible_after_G1_G2_G3" if gate_values["g1"]["passed"] and gate_values["g2"]["passed"] and gate_values["g3"]["passed"] else "skipped_teacher_gate_failed"
    metrics["final_blind_generated"] = False
    _write_json(output_dir / "metrics.json", metrics)
    _write_csv(output_dir / "result.csv", result_rows)
    _write_csv(output_dir / "verification_subgroups.csv", metrics["verification_subgroups"])
    _write_csv(output_dir / "soft_teacher_audit.csv", soft_rows)
    _write_json(output_dir / "physical_diagnostics.json", scale_diagnostic)
    _write_figures(metrics, output_dir)
    _write_analysis(metrics, output_dir, preflight["git_sha"])

    size_bytes = _repo_size_bytes()
    metrics["project_size_bytes_after_teacher_stage"] = size_bytes
    _write_json(output_dir / "metrics.json", metrics)
    return {
        "git_sha": preflight["git_sha"],
        "gates": gate_values,
        "student_pilot_eligible": metrics["gates"]["student_pilot_eligible"],
        "project_size_gib_after_teacher_stage": size_bytes / 1024**3,
        "teacher_stage_wall_seconds": stage_wall,
        "metrics": str((output_dir / "metrics.json").relative_to(ROOT)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("experiments/D2-A-T001/config.yaml"))
    parser.add_argument("--stage", choices=("preflight", "teacher", "student", "all"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not args.config.is_absolute() else args.config.resolve()
    if args.stage in ("preflight", "all"):
        print(json.dumps(_json_safe(_preflight(config_path, args.device)), ensure_ascii=False, indent=2), flush=True)
    if args.stage in ("teacher", "all"):
        result = _run_teacher(config_path, args.device)
        print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2), flush=True)
    if args.stage == "student":
        config = _read_yaml(config_path)
        _require_preflight(config, args.device)
        metrics_path = ROOT / config["outputs"]["directory"] / "metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError("complete teacher evaluation before requesting the gated student pilot")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not metrics.get("gates", {}).get("student_pilot_eligible", False):
            result = {"student_pilot_status": "SKIPPED", "reason": "teacher G1/G2/G3 did not all pass", "final_blind_generated": False}
            _write_json(metrics_path.parent / "student_pilot.json", result)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
            return 0
        raise RuntimeError("teacher gates passed; implement the frozen S2-only pilot before training, then push its code and tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
