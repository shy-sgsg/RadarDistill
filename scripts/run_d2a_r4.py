#!/usr/bin/env python3
"""Run the preregistered, exploratory D2-A-R4 physics-bottleneck diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
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

from radardistill.baselines import wrap_frequency
from radardistill.d2a_baselines import dpca_time_shift_s
from radardistill.evaluation.d2a_e002 import exact_binomial_upper
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.task_student import TaskStudent
from radardistill.teachers.d2a_t001 import (
    ACEAMFOutput,
    CovarianceBank,
    post_doppler_ace_amf,
)

BATCH = 16
def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))
                if isinstance(value, (dict, list, tuple, np.ndarray))
                else json_safe(value)
                for key, value in row.items()
            })


def git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def scale_complex128(iq: np.ndarray, factor: complex) -> np.ndarray:
    """Apply a diagnostic complex scale before any complex64 quantization."""
    values = np.asarray(iq, dtype=np.complex128)
    return values * np.complex128(factor)


def compare_numeric_outputs(
    reference_score: np.ndarray,
    candidate_score: np.ndarray,
    reference_map: np.ndarray,
    candidate_map: np.ndarray,
    *,
    threshold: float,
    reference_soft_map: np.ndarray | None = None,
    candidate_soft_map: np.ndarray | None = None,
) -> dict[str, Any]:
    reference_score = np.asarray(reference_score, dtype=np.float64).reshape(-1)
    candidate_score = np.asarray(candidate_score, dtype=np.float64).reshape(-1)
    reference_map = np.asarray(reference_map, dtype=np.float64)
    candidate_map = np.asarray(candidate_map, dtype=np.float64)
    if reference_score.shape != candidate_score.shape or reference_map.shape != candidate_map.shape:
        raise ValueError("reference and candidate score/map shapes must match")
    score_delta = np.abs(candidate_score - reference_score)
    map_delta = np.abs(candidate_map - reference_map)
    relative = score_delta / np.maximum(np.abs(reference_score), 1e-12)
    map_relative = map_delta / np.maximum(np.abs(reference_map), 1e-12)
    if reference_map.ndim == 2:
        reference_map = reference_map[None, ...]
        candidate_map = candidate_map[None, ...]
    reference_peak = np.argmax(reference_map.reshape(reference_map.shape[0], -1), axis=1)
    candidate_peak = np.argmax(candidate_map.reshape(candidate_map.shape[0], -1), axis=1)
    reference_decision = reference_score >= float(threshold)
    candidate_decision = candidate_score >= float(threshold)
    result = {
        "n": int(reference_score.size),
        "max_abs_score_difference": float(np.max(score_delta)),
        "mean_abs_score_difference": float(np.mean(score_delta)),
        "max_relative_score_difference": float(np.max(relative)),
        "max_abs_map_difference": float(np.max(map_delta)),
        "mean_abs_map_difference": float(np.mean(map_delta)),
        "max_relative_map_difference": float(np.max(map_relative)),
        "changed_peak_count": int(np.sum(reference_peak != candidate_peak)),
        "changed_threshold_decision_count": int(np.sum(reference_decision != candidate_decision)),
        "threshold": float(threshold),
        "threshold_reused": True,
    }
    if reference_soft_map is not None or candidate_soft_map is not None:
        if reference_soft_map is None or candidate_soft_map is None:
            raise ValueError("both soft maps must be provided together")
        reference_soft_map = np.asarray(reference_soft_map, dtype=np.float64)
        candidate_soft_map = np.asarray(candidate_soft_map, dtype=np.float64)
        if reference_soft_map.shape != candidate_soft_map.shape:
            raise ValueError("reference and candidate soft map shapes must match")
        soft_delta = np.abs(candidate_soft_map - reference_soft_map)
        result["max_abs_soft_map_difference"] = float(np.max(soft_delta))
        result["mean_abs_soft_map_difference"] = float(np.mean(soft_delta))
        result["max_relative_soft_map_difference"] = float(
            np.max(soft_delta / np.maximum(np.abs(reference_soft_map), 1e-12))
        )
    return result


def record_gate_lineage(
    *,
    t001_gate_passed: bool,
    t001_tolerance: float,
    r4_tolerance: float,
    measured_difference: float,
) -> dict[str, Any]:
    return {
        "t001_gate_passed": bool(t001_gate_passed),
        "t001_tolerance": float(t001_tolerance),
        "r4_tolerance": float(r4_tolerance),
        "historical_value": float(measured_difference),
        "historical_value_within_r4_tolerance_counterfactual_only": bool(
            measured_difference <= r4_tolerance
        ),
        "r4_new_data_gate_result": "reported_separately",
        "exploratory_only": True,
        "historical_t001_result_modified": False,
    }


def s1_pilot_trigger(
    *,
    engineering_gate_passed: bool,
    public_phase_agreement: float,
    doppler_rotation_p95_hz: float,
) -> bool:
    return bool(
        engineering_gate_passed
        and (
            public_phase_agreement < 0.99
            or doppler_rotation_p95_hz > 10.0
        )
    )


def _cuda_device(device: str) -> torch.device:
    selected = torch.device(device)
    if selected.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-A-R4 requires CUDA; CPU fallback is disabled")
    if selected.index is not None and selected.index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA device not visible: {selected}")
    torch.cuda.set_device(selected)
    return selected


def _load_covariance() -> CovarianceBank:
    root = ROOT / "experiments/D2-A-T001"
    with np.load(root / "covariance_bank.npz") as data:
        diag = json.loads((root / "covariance_diagnostics.json").read_text(encoding="utf-8"))
        return CovarianceBank(
            covariance=np.asarray(data["covariance"]),
            inverse=np.asarray(data["inverse"]),
            eigenvalues=np.asarray(data["eigenvalues"]),
            condition_number=np.asarray(data["condition_number"]),
            eigenvalue_floor_fraction=np.asarray(data["eigenvalue_floor_fraction"]),
            scene_count=int(diag["scene_count"]),
            pulse_count=int(diag["pulse_count"]),
            channel_count=int(diag["channel_count"]),
            oversampling_factor=int(diag["oversampling_factor"]),
        )


def _historical_gate() -> dict[str, Any]:
    metrics = json.loads((ROOT / "experiments/D2-A-T001/metrics.json").read_text(encoding="utf-8"))
    gates = metrics["gates"]
    if gates["G1"]["passed"] or gates["G2"]["passed"] or gates["G3"]["evaluated"]:
        raise RuntimeError("the frozen T001 status no longer matches this R4 protocol")
    return {
        "G1": "FAIL",
        "G2": "FAIL",
        "G3": "NOT_EVALUATED",
        "candidate": "STOP",
        "historical_t001_absolute_tolerance": 1e-6,
        "historical_observed_absolute_difference": gates["G1"]["public_common_complex_scale_invariance_max_abs_score"],
    }


def _environment(device: torch.device) -> dict[str, Any]:
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
    )
    return {
        "git_sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_device": str(device),
        "gpu_name": props.name,
        "gpu_total_bytes": int(total_bytes),
        "gpu_free_bytes_before": int(free_bytes),
        "nvidia_smi": query.stdout.strip(),
        "nvidia_smi_returncode": query.returncode,
    }


def _require_pushed_protocol() -> str:
    head = git("rev-parse", "HEAD")
    if git("branch", "--show-current") != "main":
        raise RuntimeError("R4 data generation requires authorized main branch")
    remote = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True, capture_output=True
    )
    if remote.returncode != 0 or remote.stdout.strip() != head:
        raise RuntimeError("commit and push the frozen R4 protocol/code to origin/main before generating data")
    return head


def preflight(device: str) -> dict[str, Any]:
    selected = _cuda_device(device)
    historical = _historical_gate()
    free, total = torch.cuda.mem_get_info(selected)
    if free < 1 * 1024**3:
        raise RuntimeError(f"less than 1 GiB free CUDA memory: {free}")
    physics = read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    if (
        physics["radar"]["channel_count"] != 3
        or physics["radar"]["pulse_count"] != 130
        or physics["radar"]["range_bin_count"] != 32
        or float(physics["radar"]["prf_hz"]) != 1300.0
    ):
        raise RuntimeError("frozen D2-A physical parameters changed")
    covariance = _load_covariance()
    if covariance.inverse.shape != (520, 3, 3):
        raise RuntimeError("T001 frozen covariance bank has unexpected dimensions")
    environment = _environment(selected)
    if environment["gpu_name"] != "NVIDIA GeForce RTX 3050 Laptop GPU":
        raise RuntimeError(f"unexpected required GPU: {environment['gpu_name']}")
    return {
        "passed": True,
        "git_sha": git("rev-parse", "HEAD"),
        "exploratory_only": True,
        "historical_t001_status": historical,
        "device": str(selected),
        "free_gpu_bytes": int(free),
        "total_gpu_bytes": int(total),
        "environment": environment,
    }


def _make_h0_assignments(count: int, seed: int) -> list[tuple[str, float, float]]:
    if count != 4096:
        raise ValueError("R4 H0 split size is frozen at 4096")
    rows: list[tuple[str, float, float]] = []
    specs = [
        ("C0", 1024, 20.0, -5.0),
        ("C1", 1024, 15.0, 0.0),
        ("C2", 1024, 10.0, 10.0),
        ("C3", 512, 0.0, 20.0),
        ("C3", 512, 10.0, 20.0),
    ]
    for course, n, snr, cnr in specs:
        rows.extend([(course, snr, cnr)] * n)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(rows))
    return [rows[int(index)] for index in permutation]


def generate_h0_split(count: int, seed_base: int, assignment_seed: int, physics: dict[str, Any]) -> dict[str, np.ndarray]:
    assignments = _make_h0_assignments(count, assignment_seed)
    iq = np.empty((count, 3, 130, 32), dtype=np.complex64)
    course = np.empty(count, dtype="U2")
    snr = np.empty(count, dtype=np.float32)
    cnr = np.empty(count, dtype=np.float32)
    amplitude = np.empty(count, dtype=np.float32)
    noise_power = np.empty(count, dtype=np.float32)
    for start in range(0, count, BATCH):
        stop = min(start + BATCH, count)
        cell = assignments[start:stop]
        batch = generate_d2a_paired_batch(
            stop - start,
            seed_base + (start // BATCH) * 1009,
            np.asarray([item[1] for item in cell], dtype=np.float64),
            np.asarray([item[2] for item in cell], dtype=np.float64),
            physics,
            scenario_id="separated",
        )
        iq[start:stop] = batch.iq_h0
        course[start:stop] = np.asarray([item[0] for item in cell], dtype="U2")
        snr[start:stop] = batch.snr_db
        cnr[start:stop] = batch.requested_cnr_db
        amplitude[start:stop] = batch.truth_target_amplitude
        noise_power[start:stop] = batch.noise_power
    return {
        "iq": iq,
        "course": course,
        "snr_db": snr,
        "cnr_db": cnr,
        "amplitude": amplitude,
        "noise_power": noise_power,
        "base_scene_id": np.asarray([f"r4-h0-{seed_base}-{i}" for i in range(count)], dtype="U64"),
    }


def generate_h1_cells(physics: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    cell_specs = config["data"]["h1_development"]["cells"]
    total = int(config["data"]["h1_development"]["count"])
    iq0 = np.empty((total, 3, 130, 32), dtype=np.complex64)
    iq1 = np.empty_like(iq0)
    course = np.empty(total, dtype="U2")
    scenario = np.empty(total, dtype="U32")
    truth_range = np.empty(total, dtype=np.float32)
    truth_doppler = np.empty(total, dtype=np.float32)
    truth_velocity = np.empty(total, dtype=np.float32)
    snr = np.empty(total, dtype=np.float32)
    cnr = np.empty(total, dtype=np.float32)
    amplitude = np.empty(total, dtype=np.float32)
    ids = np.empty(total, dtype="U64")
    start = 0
    for cell in cell_specs:
        count = int(cell["count"])
        for batch_start in range(0, count, BATCH):
            batch_count = min(BATCH, count - batch_start)
            chunk_index = batch_start // BATCH
            batch = generate_d2a_paired_batch(
                batch_count,
                int(cell["seed_base"]) + chunk_index * 1009,
                float(cell["snr_db"]),
                float(cell["cnr_db"]),
                physics,
                scenario_id=str(cell["scenario"]),
            )
            sl = slice(start + batch_start, start + batch_start + batch_count)
            iq0[sl] = batch.iq_h0
            iq1[sl] = batch.iq_h1
            course[sl] = str(cell["course"])
            scenario[sl] = str(cell["scenario"])
            truth_range[sl] = batch.truth_target_range_cell
            truth_doppler[sl] = batch.truth_target_doppler_hz
            truth_velocity[sl] = batch.truth_target_radial_velocity_mps
            snr[sl] = batch.snr_db
            cnr[sl] = batch.requested_cnr_db
            amplitude[sl] = batch.truth_target_amplitude
            ids[sl] = np.asarray([f"r4-h1-{cell['seed_base']}-{batch_start + i}" for i in range(batch_count)])
        start += count
    if start != total:
        raise ValueError(f"H1 cell counts sum to {start}, expected {total}")
    return {
        "iq_h0": iq0,
        "iq_h1": iq1,
        "course": course,
        "scenario": scenario,
        "truth_range_cell": truth_range,
        "truth_doppler_hz": truth_doppler,
        "truth_radial_velocity_mps": truth_velocity,
        "snr_db": snr,
        "cnr_db": cnr,
        "amplitude": amplitude,
        "base_scene_id": ids,
    }


def _b3_t1_cuda(iq: np.ndarray, physics: dict[str, Any], options: dict[str, Any], device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    """CUDA implementation of frozen B3 and T1 formulas; CPU parity is audited."""
    values = np.asarray(iq)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32):
        raise ValueError("R4 CUDA baselines require [B,3,130,32] IQ")
    samples = torch.as_tensor(values, dtype=torch.complex64, device=device)
    pulse_count = samples.shape[2]
    prf = float(physics["radar"]["prf_hz"])
    shift_samples = dpca_time_shift_s(physics) * prf
    frequencies = torch.fft.fftfreq(pulse_count, device=device, dtype=torch.float32)
    phase = torch.exp((2j * np.pi * frequencies * shift_samples).to(torch.complex64))
    second_spectrum = torch.fft.fft(samples[:, 1], dim=1)
    shifted_second = torch.fft.ifft(second_spectrum * phase[None, :, None], dim=1)
    residual = samples[:, 0] - shifted_second
    rd = torch.fft.fftshift(torch.fft.fft(residual, dim=1), dim=1)
    power = torch.abs(rd).square().to(torch.float32)
    flat_index = torch.argmax(power.reshape(power.shape[0], -1), dim=1)
    d_count, r_count = power.shape[1:]
    d_index = torch.div(flat_index, r_count, rounding_mode="floor")
    r_index = flat_index % r_count
    row = torch.arange(power.shape[0], device=device)
    b3_score = power[row, d_index, r_index]
    f_axis = torch.as_tensor(np.fft.fftshift(np.fft.fftfreq(pulse_count, d=1.0 / prf)), device=device)
    b3_doppler = f_axis[d_index]
    b3_range = r_index.to(torch.float32)

    guard_d = int(options["peak_guard_doppler_radius_cells"])
    guard_r = int(options["peak_guard_range_radius_cells"])
    t1_background = torch.empty(power.shape[0], dtype=power.dtype, device=device)
    d_offsets = torch.arange(-guard_d, guard_d + 1, device=device)
    for index in range(power.shape[0]):
        mask = torch.ones((d_count, r_count), dtype=torch.bool, device=device)
        d_guard = (d_index[index] + d_offsets) % d_count
        r_lo = max(0, int(r_index[index].item()) - guard_r)
        r_hi = min(r_count, int(r_index[index].item()) + guard_r + 1)
        mask[d_guard[:, None], torch.arange(r_lo, r_hi, device=device)[None, :]] = False
        t1_background[index] = torch.quantile(power[index][mask], 0.5)
    t1_map = power / t1_background[:, None, None]
    t1_flat = torch.argmax(t1_map.reshape(t1_map.shape[0], -1), dim=1)
    t1_d = torch.div(t1_flat, r_count, rounding_mode="floor")
    t1_r = t1_flat % r_count
    t1_score = t1_map[row, t1_d, t1_r]
    t1_doppler = f_axis[t1_d]
    torch.cuda.synchronize(device)
    return {
        "B3": {
            "score": b3_score.cpu().numpy().astype(np.float64),
            "doppler_hz": b3_doppler.cpu().numpy().astype(np.float64),
            "range_cell": b3_range.cpu().numpy().astype(np.float64),
            "score_map": power.cpu().numpy().astype(np.float64),
        },
        "T1": {
            "score": t1_score.cpu().numpy().astype(np.float64),
            "doppler_hz": t1_doppler.cpu().numpy().astype(np.float64),
            "range_cell": t1_r.cpu().numpy().astype(np.float64),
            "score_map": t1_map.cpu().numpy().astype(np.float64),
        },
    }


def _score_ace(
    iq: np.ndarray,
    covariance: CovarianceBank,
    physics: dict[str, Any],
    teacher: dict[str, Any],
    device: torch.device,
    *,
    include_maps: bool,
) -> dict[str, np.ndarray]:
    outputs: list[ACEAMFOutput] = []
    for start in range(0, len(iq), BATCH):
        current = np.asarray(iq[start : start + BATCH], dtype=np.complex128)
        outputs.append(post_doppler_ace_amf(current, covariance, physics, teacher, device=str(device)))
    result = {
        "score": np.concatenate([np.atleast_1d(item.score) for item in outputs]).astype(np.float64),
        "doppler_hz": np.concatenate([np.atleast_1d(item.doppler_hz) for item in outputs]).astype(np.float64),
        "range_cell": np.concatenate([np.atleast_1d(item.range_cell) for item in outputs]).astype(np.float64),
    }
    if include_maps:
        result["score_map"] = np.concatenate([np.atleast_3d(item.score_map) for item in outputs], axis=0)
        result["soft_response"] = np.concatenate([np.atleast_3d(item.soft_response) for item in outputs], axis=0)
    return result


def _score_baselines(
    iq: np.ndarray,
    physics: dict[str, Any],
    options: dict[str, Any],
    device: torch.device,
    *,
    include_maps: bool,
) -> dict[str, dict[str, np.ndarray]]:
    parts: dict[str, list[np.ndarray]] = {
        method: [] for method in ("B3", "T1")
    }
    estimates: dict[str, dict[str, list[np.ndarray]]] = {
        method: {"doppler_hz": [], "range_cell": [], "score_map": []}
        for method in ("B3", "T1")
    }
    for start in range(0, len(iq), BATCH):
        outputs = _b3_t1_cuda(iq[start : start + BATCH], physics, options, device)
        for method in ("B3", "T1"):
            parts[method].append(outputs[method]["score"])
            for field in estimates[method]:
                if include_maps or field != "score_map":
                    estimates[method][field].append(outputs[method][field])
    result: dict[str, dict[str, np.ndarray]] = {}
    for method in ("B3", "T1"):
        result[method] = {"score": np.concatenate(parts[method])}
        result[method]["doppler_hz"] = np.concatenate(estimates[method]["doppler_hz"])
        result[method]["range_cell"] = np.concatenate(estimates[method]["range_cell"])
        if include_maps:
            result[method]["score_map"] = np.concatenate(estimates[method]["score_map"])
    return result


def _evaluate_detector(
    h0_scores: np.ndarray,
    h1_scores: np.ndarray,
    threshold: float,
    h0_meta: dict[str, np.ndarray],
) -> dict[str, Any]:
    h0_scores = np.asarray(h0_scores, dtype=np.float64)
    h1_scores = np.asarray(h1_scores, dtype=np.float64)
    false_alarm = h0_scores >= threshold
    result: dict[str, Any] = {
        "n_h0": int(h0_scores.size),
        "n_h1": int(h1_scores.size),
        "threshold": float(threshold),
        "threshold_comparator": ">=",
        "pfa_gt": float(np.mean(h0_scores > threshold)),
        "pfa_ge": float(np.mean(false_alarm)),
        "pfa_ci95_exact_upper": exact_binomial_upper(int(false_alarm.sum()), int(false_alarm.size)),
        "h0_ties": int(np.sum(h0_scores == threshold)),
        "pd": float(np.mean(h1_scores >= threshold)),
    }
    result["pfa_by_course"] = {}
    for course in np.unique(h0_meta["course"]):
        mask = h0_meta["course"] == course
        alarms = h0_scores[mask] >= threshold
        result["pfa_by_course"][str(course)] = {
            "n": int(mask.sum()),
            "pfa": float(alarms.mean()),
            "exact_upper95": exact_binomial_upper(int(alarms.sum()), int(mask.sum())),
        }
    result["pfa_by_condition"] = {}
    condition_keys = sorted({
        (
            str(h0_meta["course"][index]),
            float(h0_meta["snr_db"][index]),
            float(h0_meta["cnr_db"][index]),
        )
        for index in range(len(h0_scores))
    })
    for course, snr, cnr in condition_keys:
        mask = (
            (h0_meta["course"] == course)
            & np.isclose(h0_meta["snr_db"], snr)
            & np.isclose(h0_meta["cnr_db"], cnr)
        )
        alarms = h0_scores[mask] >= threshold
        result["pfa_by_condition"][f"{course}/SNR{snr:g}/CNR{cnr:g}"] = {
            "n": int(mask.sum()),
            "pfa": float(alarms.mean()),
            "exact_upper95": exact_binomial_upper(int(alarms.sum()), int(mask.sum())),
        }
    result["pfa_by_amplitude_quartile"] = {}
    edges = np.quantile(h0_meta["amplitude"], [0.25, 0.5, 0.75])
    quartiles = np.digitize(h0_meta["amplitude"], edges, right=True) + 1
    for q in range(1, 5):
        mask = quartiles == q
        alarms = h0_scores[mask] >= threshold
        result["pfa_by_amplitude_quartile"][str(q)] = {
            "n": int(mask.sum()),
            "pfa": float(alarms.mean()),
            "median_observable_power": float(np.median(h0_meta["observable_power"][mask])),
            "median_noise_power": float(np.median(h0_meta["noise_power"][mask])),
            "exact_upper95": exact_binomial_upper(int(alarms.sum()), int(mask.sum())),
        }
    from scipy.stats import spearmanr

    result["score_vs_observable_power_spearman"] = float(
        spearmanr(h0_scores, h0_meta["observable_power"]).statistic
    )
    return result


def check_h0_scenario_independence(physics: dict[str, Any], seed: int) -> dict[str, Any]:
    outputs = {}
    reference = None
    reference_noise_power = None
    for scenario in ("separated", "near_clutter_edge", "partial_overlap", "exact_overlap"):
        batch = generate_d2a_paired_batch(
            16, seed, 10.0, 10.0, physics, scenario_id=scenario
        )
        if reference is None:
            reference = batch.iq_h0
            reference_noise_power = batch.noise_power
        outputs[scenario] = {
            "iq_h0_exactly_equal_to_reference": bool(np.array_equal(batch.iq_h0, reference)),
            "max_abs_iq_h0_difference": float(np.max(np.abs(
                batch.iq_h0.astype(np.complex128) - reference.astype(np.complex128)
            ))),
            "noise_power_equal": bool(np.array_equal(batch.noise_power, reference_noise_power)),
        }
    return {
        "seed": int(seed),
        "snr_db": 10.0,
        "cnr_db": 10.0,
        "scenario_labels": list(outputs),
        "per_scenario": outputs,
        "h0_exactly_independent_of_target_scenario_label": all(
            item["iq_h0_exactly_equal_to_reference"] for item in outputs.values()
        ),
        "detector_input_contains_no_scenario_label": True,
    }


def _paired_h1_summary(
    method: str,
    h0_scores: np.ndarray,
    h1: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    truth_r = h1["truth_range_cell"].astype(np.float64)
    truth_f = h1["truth_doppler_hz"].astype(np.float64)
    score = np.asarray(predictions["score"], dtype=np.float64)
    detected = score >= threshold
    r_error = np.asarray(predictions["range_cell"], dtype=np.float64) - truth_r
    f_error = np.asarray(wrap_frequency(
        np.asarray(predictions["doppler_hz"], dtype=np.float64) - truth_f, 1300.0
    ))
    joint = detected & (np.abs(r_error) <= 1.0) & (np.abs(f_error) <= 20.0)
    from radardistill.evaluation.d2a_e001 import empirical_auc

    auc = float(empirical_auc(h0_scores, score)) if len(h0_scores) else float("nan")
    rng = np.random.default_rng(2026102801)
    draws = rng.integers(0, len(score), size=(2000, len(score)))
    pd_boot = np.mean(detected[draws], axis=1)
    joint_boot = np.mean(joint[draws], axis=1)
    range_rmse_boot = np.sqrt(np.mean(r_error[draws] ** 2, axis=1))
    doppler_rmse_boot = np.sqrt(np.mean(f_error[draws] ** 2, axis=1))

    def ci(values: np.ndarray) -> list[float]:
        return [float(item) for item in np.quantile(values, [0.025, 0.975])]

    return {
        "n": int(len(score)),
        "pd": float(np.mean(detected)),
        "auc": auc,
        "range_mae_cells": float(np.mean(np.abs(r_error))),
        "range_rmse_cells": float(np.sqrt(np.mean(r_error**2))),
        "doppler_mae_hz": float(np.mean(np.abs(f_error))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(f_error**2))),
        "doppler_bias_hz": float(np.mean(f_error)),
        "range_rmse_detected_h1": float(np.sqrt(np.mean(r_error[detected] ** 2))) if np.any(detected) else None,
        "doppler_rmse_detected_h1": float(np.sqrt(np.mean(f_error[detected] ** 2))) if np.any(detected) else None,
        "joint_success": float(np.mean(joint)),
        "scene_bootstrap_95_ci": {
            "pd": ci(pd_boot),
            "range_rmse_cells": ci(range_rmse_boot),
            "doppler_rmse_hz": ci(doppler_rmse_boot),
            "joint_success": ci(joint_boot),
            "replicates": 2000,
            "bootstrap_unit": "H1 base_scene_id",
            "seed_base": 2026102801,
        },
        "misses_count_as_joint_failure": True,
        "scenario_summary": {},
    }


def _paired_h1_method_difference(
    first: dict[str, np.ndarray],
    first_threshold: float,
    second: dict[str, np.ndarray],
    second_threshold: float,
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    *,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap paired scene-level differences: first method minus second."""
    n = len(truth_range)
    if len(first["score"]) != n or len(second["score"]) != n:
        raise ValueError("paired methods must use the same H1 scene set")
    range_truth = np.asarray(truth_range, dtype=np.float64)
    doppler_truth = np.asarray(truth_doppler, dtype=np.float64)

    def vectors(prediction: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
        detected = np.asarray(prediction["score"]) >= threshold
        range_error = np.asarray(prediction["range_cell"]) - range_truth
        doppler_error = np.asarray(wrap_frequency(
            np.asarray(prediction["doppler_hz"]) - doppler_truth, 1300.0
        ))
        return {
            "pd": detected.astype(np.float64),
            "range_squared": range_error**2,
            "doppler_squared": doppler_error**2,
            "joint": (
                detected & (np.abs(range_error) <= 1.0) & (np.abs(doppler_error) <= 20.0)
            ).astype(np.float64),
        }

    first_v = vectors(first, first_threshold)
    second_v = vectors(second, second_threshold)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(2000, n))
    result: dict[str, Any] = {}
    for name in ("pd", "joint"):
        delta = first_v[name][draws].mean(axis=1) - second_v[name][draws].mean(axis=1)
        point = float(first_v[name].mean() - second_v[name].mean())
        result[name] = {
            "difference": point,
            "paired_bootstrap_95_ci": [float(item) for item in np.quantile(delta, [0.025, 0.975])],
        }
    for name in ("range_squared", "doppler_squared"):
        delta = (
            np.sqrt(first_v[name][draws].mean(axis=1))
            - np.sqrt(second_v[name][draws].mean(axis=1))
        )
        point = float(np.sqrt(first_v[name].mean()) - np.sqrt(second_v[name].mean()))
        result[name.replace("_squared", "_rmse")] = {
            "difference": point,
            "paired_bootstrap_95_ci": [float(item) for item in np.quantile(delta, [0.025, 0.975])],
        }
    result["bootstrap_unit"] = "paired H1 base_scene_id"
    result["replicates"] = 2000
    result["seed"] = int(seed)
    return result


def _student_from_checkpoint(checkpoint_path: Path, config: dict[str, Any], device: torch.device) -> TaskStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("has_k2_head") is not False:
        raise ValueError(f"frozen inference checkpoint is not stripped of K2 head: {checkpoint_path}")
    model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def _predict_student(iq: np.ndarray, model: TaskStudent, device: torch.device) -> dict[str, np.ndarray]:
    from radardistill.distillation.task_losses import decode_task_outputs

    scores: list[np.ndarray] = []
    ranges: list[np.ndarray] = []
    dopplers: list[np.ndarray] = []
    true_gate_dopplers: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(iq), BATCH):
            current = np.asarray(iq[start : start + BATCH], dtype=np.complex64)
            planes = np.concatenate((current.real, current.imag), axis=1).astype(np.float32, copy=False)
            outputs = model(torch.from_numpy(np.ascontiguousarray(planes)).to(device), compute_spectrum=False)
            decoded = decode_task_outputs(outputs, prf_hz=1300.0)
            scores.append(outputs["presence_logit"].cpu().numpy())
            ranges.append(decoded["range_cell"].cpu().numpy())
            dopplers.append(decoded["doppler_hz"].cpu().numpy())
    result = {
        "score": np.concatenate(scores),
        "range_cell": np.concatenate(ranges),
        "doppler_hz": np.concatenate(dopplers),
    }
    return result


def _predict_student_with_true_range(
    iq: np.ndarray,
    truth_range_cell: np.ndarray,
    model: TaskStudent,
    device: torch.device,
) -> np.ndarray:
    """Decode the Doppler head at the training-time truth range cell."""
    dopplers: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(iq), BATCH):
            current = np.asarray(iq[start : start + BATCH], dtype=np.complex64)
            planes = np.concatenate((current.real, current.imag), axis=1).astype(np.float32, copy=False)
            outputs = model(torch.from_numpy(np.ascontiguousarray(planes)).to(device), compute_spectrum=False)
            indices = np.floor(
                np.asarray(truth_range_cell[start : start + BATCH], dtype=np.float64)
            ).astype(np.int64)
            indices = np.clip(indices, 0, 31)
            rows = torch.arange(len(current), device=device)
            selected = outputs["doppler_vector"][rows, torch.as_tensor(indices, device=device)]
            angle = torch.atan2(selected[:, 1], selected[:, 0])
            frequency = torch.remainder(angle * (1300.0 / (2.0 * np.pi)) + 650.0, 1300.0) - 650.0
            dopplers.append(frequency.cpu().numpy())
    return np.concatenate(dopplers)


def _map_alignment(
    score_map: np.ndarray,
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    *,
    prf_hz: float = 1300.0,
    range_gate: float = 1.0,
    doppler_gate_hz: float = 20.0,
) -> dict[str, Any]:
    maps = np.asarray(score_map, dtype=np.float64)
    if maps.ndim != 3:
        raise ValueError("score maps must have [scene,frequency,range] layout")
    bins = maps.shape[1]
    frequencies = np.fft.fftshift(np.fft.fftfreq(bins, d=1.0 / prf_hz))
    flat = np.argmax(maps.reshape(len(maps), -1), axis=1)
    peak_d = flat // maps.shape[2]
    peak_r = flat % maps.shape[2]
    peak_f = frequencies[peak_d]
    range_error = peak_r.astype(np.float64) - np.asarray(truth_range, dtype=np.float64)
    doppler_error = np.asarray(wrap_frequency(
        peak_f - np.asarray(truth_doppler, dtype=np.float64), prf_hz
    ))
    correct = (np.abs(range_error) <= range_gate) & (np.abs(doppler_error) <= doppler_gate_hz)
    margins = []
    for index, scene_map in enumerate(maps):
        range_mask = np.abs(np.arange(scene_map.shape[1]) - truth_range[index]) <= range_gate
        doppler_mask = np.abs(wrap_frequency(frequencies - truth_doppler[index], prf_hz)) <= doppler_gate_hz
        window = doppler_mask[:, None] & range_mask[None, :]
        inside = float(np.max(scene_map[window])) if np.any(window) else 0.0
        outside = float(np.max(scene_map[~window])) if np.any(~window) else 0.0
        margins.append(inside / max(outside, np.finfo(np.float64).tiny))
    return {
        "top_cell_localization_rate": float(np.mean(correct)),
        "correct_location_count": int(np.sum(correct)),
        "incorrect_location_count": int(np.sum(~correct)),
        "range_peak_mae_cells": float(np.mean(np.abs(range_error))),
        "doppler_peak_mae_hz": float(np.mean(np.abs(doppler_error))),
        "target_window_peak_to_outside_peak_ratio_mean": float(np.mean(margins)),
        "n": int(len(maps)),
    }


def _map_correct_mask(
    score_map: np.ndarray,
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    *,
    range_gate: float = 1.0,
    doppler_gate_hz: float = 20.0,
) -> np.ndarray:
    maps = np.asarray(score_map, dtype=np.float64)
    frequencies = np.fft.fftshift(np.fft.fftfreq(maps.shape[1], d=1.0 / 1300.0))
    flat = np.argmax(maps.reshape(len(maps), -1), axis=1)
    peak_d = flat // maps.shape[2]
    peak_r = flat % maps.shape[2]
    range_error = peak_r - np.asarray(truth_range, dtype=np.float64)
    doppler_error = np.asarray(wrap_frequency(
        frequencies[peak_d] - np.asarray(truth_doppler, dtype=np.float64), 1300.0
    ))
    return (np.abs(range_error) <= range_gate) & (np.abs(doppler_error) <= doppler_gate_hz)


def _teacher_frequency_shift_diagnostic(
    method: str,
    h1: dict[str, np.ndarray],
    baseline: dict[str, np.ndarray],
    threshold: float,
    covariance: CovarianceBank,
    physics: dict[str, Any],
    teacher_config: dict[str, Any],
    t1_options: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = min(256, len(h1["iq_h1"]))
    iq0 = np.asarray(h1["iq_h0"][:count], dtype=np.complex128)
    iq1 = np.asarray(h1["iq_h1"][:count], dtype=np.complex128)
    target = iq1 - iq0
    correct = _map_correct_mask(
        baseline["score_map"][:count],
        h1["truth_range_cell"][:count],
        h1["truth_doppler_hz"][:count],
    )
    pulse = np.arange(iq1.shape[2], dtype=np.float64) / 1300.0
    results: dict[str, Any] = {}
    for delta in (-25.0, -10.0, 10.0, 25.0):
        modulation = np.exp(2j * np.pi * delta * pulse)[None, None, :, None]
        shifted_iq = iq0 + target * modulation
        shifted_all = _score_all_teachers(
            shifted_iq.astype(np.complex64),
            covariance, physics, teacher_config, t1_options, device, include_maps=False,
        )
        shifted = shifted_all[method]
        residual = np.asarray(wrap_frequency(
            shifted["doppler_hz"] - baseline["doppler_hz"][:count] - delta, 1300.0
        ))
        shifted_detection = shifted["score"] >= threshold

        def summary(mask: np.ndarray) -> dict[str, Any]:
            if not np.any(mask):
                return {"n": 0, "residual_p95_abs_hz": None, "detection_fraction": None}
            return {
                "n": int(mask.sum()),
                "residual_mean_hz": float(np.mean(residual[mask])),
                "residual_p95_abs_hz": float(np.quantile(np.abs(residual[mask]), 0.95)),
                "detection_fraction_at_fixed_threshold": float(np.mean(shifted_detection[mask])),
            }

        results[str(delta)] = {
            "all_base_locations": summary(np.ones(count, dtype=bool)),
            "base_location_correct": summary(correct),
            "base_location_incorrect": summary(~correct),
            "threshold_recalibrated": False,
            "intervention": "frequency shift applied to H1-H0 target component only",
        }
    return {
        "n_scenes": int(count),
        "base_location_correct_count": int(correct.sum()),
        "base_location_incorrect_count": int((~correct).sum()),
        "shifts": results,
        "regenerated_physical_motion_evaluated_separately": True,
    }


def _covariance_at_peak_summary(
    score_map: np.ndarray,
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    covariance: CovarianceBank,
) -> dict[str, Any]:
    maps = np.asarray(score_map, dtype=np.float64)
    flat = np.argmax(maps.reshape(len(maps), -1), axis=1)
    peak_d = flat // maps.shape[2]
    correct = _map_correct_mask(maps, truth_range, truth_doppler)
    covariance_bin = np.rint(
        peak_d * len(covariance.condition_number) / maps.shape[1]
    ).astype(np.int64)
    condition = covariance.condition_number[
        np.clip(covariance_bin, 0, len(covariance.condition_number) - 1)
    ]
    return {
        "peak_bin_condition_number_median": float(np.median(condition)),
        "correct_peak_condition_number_median": float(np.median(condition[correct])) if np.any(correct) else None,
        "incorrect_peak_condition_number_median": float(np.median(condition[~correct])) if np.any(~correct) else None,
        "covariance_bank_bins": int(len(covariance.condition_number)),
        "covariance_bank_channel_dimension": int(covariance.channel_count),
    }


def _ace_soft_summary(
    soft_response: np.ndarray,
    truth_range: np.ndarray,
    truth_doppler: np.ndarray,
    *,
    prf_hz: float = 1300.0,
) -> dict[str, Any]:
    maps = np.asarray(soft_response, dtype=np.float64)
    frequencies = np.fft.fftshift(np.fft.fftfreq(maps.shape[1], d=1.0 / prf_hz))
    masses: list[float] = []
    entropies: list[float] = []
    for index, response in enumerate(maps):
        range_mask = np.abs(np.arange(response.shape[1]) - truth_range[index]) <= 1.0
        doppler_mask = np.abs(wrap_frequency(frequencies - truth_doppler[index], prf_hz)) <= 20.0
        window = doppler_mask[:, None] & range_mask[None, :]
        masses.append(float(np.sum(response[window])))
        positive = response[response > 0.0]
        entropies.append(float(-np.sum(positive * np.log(positive))))
    return {
        "target_window_mass_mean": float(np.mean(masses)),
        "normalized_entropy_mean": float(np.mean(entropies) / np.log(maps.shape[1] * maps.shape[2])),
        "n": int(len(maps)),
        "saliency_not_posterior": True,
    }


def _score_all_teachers(
    iq: np.ndarray,
    covariance: CovarianceBank,
    physics: dict[str, Any],
    teacher: dict[str, Any],
    t1_options: dict[str, Any],
    device: torch.device,
    *,
    include_maps: bool,
) -> dict[str, dict[str, np.ndarray]]:
    outputs = _score_baselines(iq, physics, t1_options, device, include_maps=include_maps)
    outputs["ACE_AMF"] = _score_ace(
        iq, covariance, physics, teacher, device, include_maps=include_maps
    )
    return outputs


def _h1_group_masks(h1: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    rows = sorted({
        (
            str(h1["course"][index]),
            str(h1["scenario"][index]),
            float(h1["snr_db"][index]),
            float(h1["cnr_db"][index]),
        )
        for index in range(len(h1["scenario"]))
    })
    groups = []
    for course, scenario, snr, cnr in rows:
        mask = (
            (h1["course"] == course)
            & (h1["scenario"] == scenario)
            & np.isclose(h1["snr_db"], snr)
            & np.isclose(h1["cnr_db"], cnr)
        )
        groups.append((f"{course}/{scenario}/SNR{snr:g}/CNR{cnr:g}", mask))
    return groups


def _read_thresholds(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        return {str(row["method"]): float(row["threshold"]) for row in rows}


def _student_phase_diagnostic(
    iq: np.ndarray,
    threshold: float,
    model: TaskStudent,
    device: torch.device,
) -> dict[str, Any]:
    count = min(256, len(iq))
    subset = np.asarray(iq[:count], dtype=np.complex64)
    baseline = _predict_student(subset, model, device)
    baseline_detection = baseline["score"] >= threshold
    phase_values = np.arange(32, dtype=np.float64) * (2.0 * np.pi / 32.0)
    agreements: list[np.ndarray] = []
    range_changes: list[np.ndarray] = []
    doppler_changes: list[np.ndarray] = []
    score_changes: list[np.ndarray] = []
    for phase in phase_values:
        rotated = scale_complex128(subset, np.exp(1j * phase)).astype(np.complex64)
        prediction = _predict_student(rotated, model, device)
        agreements.append((prediction["score"] >= threshold) == baseline_detection)
        range_changes.append(np.abs(prediction["range_cell"] - baseline["range_cell"]))
        doppler_changes.append(np.abs(wrap_frequency(
            prediction["doppler_hz"] - baseline["doppler_hz"], 1300.0
        )))
        score_changes.append(np.abs(prediction["score"] - baseline["score"]))
    agreement = np.concatenate(agreements)
    return {
        "n_scenes": int(count),
        "n_rotations": int(len(phase_values)),
        "decision_agreement": float(np.mean(agreement)),
        "decision_agreement_by_scene_min": float(np.min(np.mean(np.asarray(agreements), axis=0))),
        "range_change_p95_bins": float(np.quantile(np.concatenate(range_changes), 0.95)),
        "doppler_change_p95_hz": float(np.quantile(np.concatenate(doppler_changes), 0.95)),
        "score_abs_change_p95": float(np.quantile(np.concatenate(score_changes), 0.95)),
        "threshold_reused": True,
        "rotation_values_rad": phase_values.tolist(),
    }


def _student_frequency_shift_diagnostic(
    iq_h0: np.ndarray,
    iq_h1: np.ndarray,
    model: TaskStudent,
    device: torch.device,
) -> dict[str, Any]:
    count = min(256, len(iq_h1))
    h0 = np.asarray(iq_h0[:count], dtype=np.complex128)
    h1 = np.asarray(iq_h1[:count], dtype=np.complex128)
    base = _predict_student(h1, model, device)
    target_component = h1 - h0
    pulse = np.arange(h1.shape[2], dtype=np.float64) / 1300.0
    results: dict[str, Any] = {}
    for delta in (-25.0, -10.0, 10.0, 25.0):
        modulation = np.exp(2j * np.pi * delta * pulse)[None, None, :, None]
        target_shifted = h0 + target_component * modulation
        target_prediction = _predict_student(target_shifted.astype(np.complex64), model, device)
        full_shifted = h1 * modulation
        full_prediction = _predict_student(full_shifted.astype(np.complex64), model, device)

        def residual(prediction: dict[str, np.ndarray]) -> np.ndarray:
            return np.asarray(wrap_frequency(
                prediction["doppler_hz"] - base["doppler_hz"] - delta, 1300.0
            ))

        results[str(delta)] = {
            "target_component_only_shift_residual_mean_hz": float(np.mean(residual(target_prediction))),
            "target_component_only_shift_residual_p95_abs_hz": float(np.quantile(np.abs(residual(target_prediction)), 0.95)),
            "whole_observation_shift_residual_mean_hz": float(np.mean(residual(full_prediction))),
            "whole_observation_shift_residual_p95_abs_hz": float(np.quantile(np.abs(residual(full_prediction)), 0.95)),
            "target_only_is_synthetic_intervention": True,
            "whole_observation_shift_is_not_regenerated_motion": True,
        }
    return {
        "n_scenes": int(count),
        "physical_motion_comparison": "regenerated H1 cells summarized separately by scenario and velocity",
        "shifts": results,
    }


def _student_h1_metrics(
    h1: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, Any]:
    score = prediction["score"]
    detection = score >= threshold
    truth_range = h1["truth_range_cell"].astype(np.float64)
    truth_doppler = h1["truth_doppler_hz"].astype(np.float64)
    range_error = prediction["range_cell"] - truth_range
    doppler_error = np.asarray(wrap_frequency(
        prediction["doppler_hz"] - truth_doppler, 1300.0
    ))
    true_gate_error = np.asarray(wrap_frequency(
        prediction["true_gate_doppler_hz"] - truth_doppler, 1300.0
    ))
    range_index_error = np.abs(np.floor(prediction["range_cell"]) - np.floor(truth_range))
    metrics: dict[str, Any] = {
        "n": int(len(score)),
        "threshold": float(threshold),
        "pd": float(np.mean(detection)),
        "range_mae_cells": float(np.mean(np.abs(range_error))),
        "range_rmse_cells": float(np.sqrt(np.mean(range_error**2))),
        "doppler_mae_hz": float(np.mean(np.abs(doppler_error))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(doppler_error**2))),
        "doppler_rmse_at_truth_range_cell_hz": float(np.sqrt(np.mean(true_gate_error**2))),
        "true_range_cell_doppler_rmse_detected_hz": float(np.sqrt(np.mean(true_gate_error[detection]**2))) if np.any(detection) else None,
        "predicted_range_gate_error_gt_1_cell_fraction": float(np.mean(range_index_error > 1.0)),
        "doppler_gate_error_p95_hz": float(np.quantile(np.abs(doppler_error), 0.95)),
        "joint_success": float(np.mean(
            detection & (np.abs(range_error) <= 1.0) & (np.abs(doppler_error) <= 20.0)
        )),
        "range_gate_mismatch_diagnostic": {
            "decoded_doppler_uses_predicted_range_argmax": True,
            "training_supervision_selects_truth_range_cell": True,
            "true_range_cell_doppler_prediction_measured": True,
            "interpretation": "difference compares the inference range gate with the training-time truth gate",
        },
        "scenario_summary": {},
    }
    for group_name, mask in _h1_group_masks(h1):
        error = doppler_error[mask]
        metrics["scenario_summary"][group_name] = {
            "n": int(mask.sum()),
            "target_radial_velocity_mps": float(np.median(h1["truth_radial_velocity_mps"][mask])) if "truth_radial_velocity_mps" in h1 else None,
            "snr_db": float(np.median(h1["snr_db"][mask])),
            "cnr_db": float(np.median(h1["cnr_db"][mask])),
            "pd": float(np.mean(detection[mask])),
            "doppler_rmse_hz": float(np.sqrt(np.mean(error**2))),
        }
    return metrics


def run_numeric_audit(
    config: dict[str, Any],
    physics: dict[str, Any],
    teacher: dict[str, Any],
    covariance: CovarianceBank,
    device: torch.device,
) -> dict[str, Any]:
    sections: list[np.ndarray] = []
    scenario_ids = ("separated", "near_clutter_edge", "partial_overlap", "exact_overlap")
    seed = int(config["data"]["numerical_precision"]["seed_base"])
    for index, scenario in enumerate(scenario_ids):
        batch = generate_d2a_paired_batch(
            16, seed + index * 1009, 10.0, 10.0, physics, scenario_id=scenario
        )
        sections.append(batch.iq_h1)
    iq = np.concatenate(sections, axis=0).astype(np.complex128)
    old_metrics = json.loads((ROOT / "experiments/D2-A-T001/metrics.json").read_text(encoding="utf-8"))
    old_threshold = float(old_metrics["thresholds"]["ACE_AMF"]["threshold"])
    baseline_cpu = post_doppler_ace_amf(iq, covariance, physics, teacher, device="cpu")
    baseline_cuda = post_doppler_ace_amf(iq, covariance, physics, teacher, device=str(device))
    cpu_cuda = compare_numeric_outputs(
        baseline_cpu.score, baseline_cuda.score, baseline_cpu.score_map,
        baseline_cuda.score_map, threshold=old_threshold,
        reference_soft_map=baseline_cpu.soft_response,
        candidate_soft_map=baseline_cuda.soft_response,
    )
    r4_tolerance = float(config["data"]["numerical_precision"]["new_absolute_difference_gate"])
    output: dict[str, Any] = {
        "n_paired_scenes": int(len(iq)),
        "scenarios": list(scenario_ids),
        "cpu_reference_precision": "complex128",
        "cuda_precision": "complex64",
        "input_dtype_before_backend": str(iq.dtype),
        "t001_frozen_gate": record_gate_lineage(
            t001_gate_passed=False,
            t001_tolerance=1e-6,
            r4_tolerance=float(config["data"]["numerical_precision"]["new_absolute_difference_gate"]),
            measured_difference=float(old_metrics["gates"]["G1"]["public_common_complex_scale_invariance_max_abs_score"]),
        ),
        "cpu_vs_cuda": cpu_cuda,
        "public_phase_rotation": {},
        "public_complex_scales": {},
        "legacy_complex64_scale_construction": {},
        "cpu_covariance": {
            "max_hermitian_residual": float(np.max(np.abs(covariance.inverse - covariance.inverse.conj().transpose(0, 2, 1)))),
            "min_recorded_eigenvalue": float(np.min(covariance.eigenvalues)),
            "max_condition_number": float(np.max(covariance.condition_number)),
            "shape": list(covariance.inverse.shape),
        },
        "frozen_t001_threshold_reused": old_threshold,
    }
    phase = np.exp(0.731j)
    for backend, baseline in (("cpu_complex128", baseline_cpu), ("cuda_complex64", baseline_cuda)):
        rotated = post_doppler_ace_amf(scale_complex128(iq, phase), covariance, physics, teacher, device="cpu" if backend.startswith("cpu") else str(device))
        output["public_phase_rotation"][backend] = compare_numeric_outputs(
            baseline.score, rotated.score, baseline.score_map, rotated.score_map,
            threshold=old_threshold,
            reference_soft_map=baseline.soft_response,
            candidate_soft_map=rotated.soft_response,
        )
    for factor in (0.5 + 0.0j, 2.0 + 0.0j, 0.6 + 1.2j):
        key = f"{factor.real:+.1f}{factor.imag:+.1f}j"
        for backend, baseline in (("cpu_complex128", baseline_cpu), ("cuda_complex64", baseline_cuda)):
            backend_device = "cpu" if backend.startswith("cpu") else str(device)
            scaled = post_doppler_ace_amf(
                scale_complex128(iq, factor), covariance, physics, teacher, device=backend_device
            )
            scale_comparison = compare_numeric_outputs(
                baseline.score, scaled.score, baseline.score_map, scaled.score_map,
                threshold=old_threshold,
                reference_soft_map=baseline.soft_response,
                candidate_soft_map=scaled.soft_response,
            )
            scale_comparison["r4_scale_invariance_gate_passed"] = bool(
                scale_comparison["max_abs_score_difference"] <= r4_tolerance
                and scale_comparison["max_abs_map_difference"] <= r4_tolerance
                and scale_comparison["max_abs_soft_map_difference"] <= r4_tolerance
            )
            output["public_complex_scales"].setdefault(key, {})[backend] = scale_comparison
        legacy_iq = (iq.astype(np.complex64) * np.complex64(factor)).astype(np.complex64)
        legacy = post_doppler_ace_amf(legacy_iq, covariance, physics, teacher, device=str(device))
        output["legacy_complex64_scale_construction"][key] = compare_numeric_outputs(
            baseline_cuda.score, legacy.score, baseline_cuda.score_map, legacy.score_map,
            threshold=old_threshold,
            reference_soft_map=baseline_cuda.soft_response,
            candidate_soft_map=legacy.soft_response,
        )
    output["r4_scale_invariance_gate"] = {
        "absolute_tolerance": r4_tolerance,
        "per_scale_backend_pass": {
            factor: {
                backend: result["r4_scale_invariance_gate_passed"]
                for backend, result in backend_results.items()
            }
            for factor, backend_results in output["public_complex_scales"].items()
        },
        "passed": bool(all(
            result["r4_scale_invariance_gate_passed"]
            for backend_results in output["public_complex_scales"].values()
            for result in backend_results.values()
        )),
        "applies_only_to_r4": True,
        "historical_t001_gate_unchanged": True,
    }
    return output


def run_diagnostics(device_name: str) -> dict[str, Any]:
    device = _cuda_device(device_name)
    _require_pushed_protocol()
    out_dir = ROOT / "experiments/D2-A-R4"
    preflight_record = json.loads((out_dir / "preflight.json").read_text(encoding="utf-8"))
    if not preflight_record.get("passed") or preflight_record["git_sha"] != git("rev-parse", "HEAD"):
        raise RuntimeError("R4 preflight is missing or stale")
    config = read_yaml(out_dir / "config.yaml")
    physics = read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    r3 = read_yaml(ROOT / "experiments/D2-A-R3/config.yaml")
    e002 = read_yaml(ROOT / "experiments/D2-A-E002/config.yaml")
    teacher_config = read_yaml(ROOT / "experiments/D2-A-T001/config.yaml")["teacher"]
    covariance = _load_covariance()
    historical = _historical_gate()
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    print("R4: running same-IQ CPU/CUDA precision audit", flush=True)
    numerical = run_numeric_audit(config, physics, teacher_config, covariance, device)
    write_json(out_dir / "numeric_precision.json", numerical)

    print("R4: generating independent H0 calibration scenes", flush=True)
    cal = generate_h0_split(4096, 41_000_000, 2026102802, physics)
    print("R4: generating independent H0 verification scenes", flush=True)
    ver = generate_h0_split(4096, 42_000_000, 2026102803, physics)
    print("R4: generating paired H1 development scenes", flush=True)
    h1 = generate_h1_cells(physics, config)
    h0_scenario_independence = check_h0_scenario_independence(
        physics, int(config["data"]["h0_scenario_label_independence"]["seed"])
    )
    t1_options = r3["detection"]["t1_global_robust_background"]

    # The new CUDA B3/T1 implementation is checked against the frozen CPU
    # implementation on a fixed subset before its scores are used.
    from radardistill.d2a_baselines import run_b3
    from radardistill.d2a_r3_baselines import run_t1_global_normalized
    parity_iq = np.asarray(ver["iq"][:16], dtype=np.complex64)
    parity_cuda = _b3_t1_cuda(parity_iq, physics, t1_options, device)
    parity_cpu_b3 = run_b3(parity_iq, 1300.0, physics)
    parity_cpu_t1 = run_t1_global_normalized(parity_iq, 1300.0, physics, t1_options)
    baseline_parity: dict[str, Any] = {}
    for method, cpu_output in (("B3", parity_cpu_b3), ("T1", parity_cpu_t1)):
        cpu_map = cpu_output.rd_power if method == "B3" else cpu_output.score_map
        baseline_parity[method] = compare_numeric_outputs(
            np.atleast_1d(cpu_output.score), parity_cuda[method]["score"],
            np.asarray(cpu_map), parity_cuda[method]["score_map"],
            threshold=float(np.median(np.atleast_1d(cpu_output.score))),
        )
    baseline_parity_passed = all(
        item["changed_peak_count"] == 0
        and item["max_relative_score_difference"] <= 1e-3
        for item in baseline_parity.values()
    )

    print("R4: CUDA teacher scoring on calibration, verification and development", flush=True)
    cal_teacher = _score_all_teachers(
        cal["iq"], covariance, physics, teacher_config, t1_options, device, include_maps=False
    )
    ver_teacher = _score_all_teachers(
        ver["iq"], covariance, physics, teacher_config, t1_options, device, include_maps=False
    )
    h1_teacher = _score_all_teachers(
        h1["iq_h1"], covariance, physics, teacher_config, t1_options, device, include_maps=True
    )
    teacher_thresholds = {
        method: float(np.quantile(values["score"], 0.95, method="higher"))
        for method, values in cal_teacher.items()
    }
    ver_meta = {
        **ver,
        "observable_power": np.mean(np.abs(ver["iq"].astype(np.complex128)) ** 2, axis=(1, 2, 3)),
    }
    teacher_metrics: dict[str, Any] = {}
    for method in ("B3", "T1", "ACE_AMF"):
        threshold = teacher_thresholds[method]
        summary = _evaluate_detector(
            ver_teacher[method]["score"], h1_teacher[method]["score"], threshold, ver_meta
        )
        summary["calibration"] = {
            "n": int(len(cal_teacher[method]["score"])),
            "pfa_gt": float(np.mean(cal_teacher[method]["score"] > threshold)),
            "pfa_ge": float(np.mean(cal_teacher[method]["score"] >= threshold)),
            "tie_count": int(np.sum(cal_teacher[method]["score"] == threshold)),
            "quantile_method": "higher",
        }
        summary["parameter_and_map"] = _paired_h1_summary(
            method, ver_teacher[method]["score"], h1, h1_teacher[method], threshold
        )
        summary["parameter_and_map"]["peak_map_alignment"] = _map_alignment(
            h1_teacher[method]["score_map"], h1["truth_range_cell"], h1["truth_doppler_hz"]
        )
        if method == "ACE_AMF":
            summary["parameter_and_map"]["soft_response"] = _ace_soft_summary(
                h1_teacher[method]["soft_response"],
                h1["truth_range_cell"], h1["truth_doppler_hz"],
            )
        for group_name, mask in _h1_group_masks(h1):
            first_index = int(np.flatnonzero(mask)[0])
            h0_mask = (
                (ver["course"] == h1["course"][first_index])
                & np.isclose(ver["snr_db"], h1["snr_db"][first_index])
                & np.isclose(ver["cnr_db"], h1["cnr_db"][first_index])
            )
            item = _paired_h1_summary(
                method,
                ver_teacher[method]["score"][h0_mask],
                {
                    "truth_range_cell": h1["truth_range_cell"][mask],
                    "truth_doppler_hz": h1["truth_doppler_hz"][mask],
                },
                {
                    key: h1_teacher[method][key][mask]
                    for key in ("score", "range_cell", "doppler_hz")
                },
                threshold,
            )
            summary["parameter_and_map"]["scenario_summary"][group_name] = item
        summary["covariance_condition_at_peak"] = _covariance_at_peak_summary(
            h1_teacher[method]["score_map"],
            h1["truth_range_cell"],
            h1["truth_doppler_hz"],
            covariance,
        )
        teacher_metrics[method] = summary

    teacher_paired_differences = {}
    for first, second in (("ACE_AMF", "B3"), ("ACE_AMF", "T1"), ("T1", "B3")):
        teacher_paired_differences[f"{first}_minus_{second}"] = _paired_h1_method_difference(
            h1_teacher[first],
            teacher_thresholds[first],
            h1_teacher[second],
            teacher_thresholds[second],
            h1["truth_range_cell"],
            h1["truth_doppler_hz"],
            seed=2026102801,
        )

    print("R4: teacher target-frequency shift diagnostics", flush=True)
    for method in ("B3", "T1", "ACE_AMF"):
        teacher_metrics[method]["target_only_frequency_shift"] = _teacher_frequency_shift_diagnostic(
            method,
            h1,
            h1_teacher[method],
            teacher_thresholds[method],
            covariance,
            physics,
            teacher_config,
            t1_options,
            device,
        )

    # Oracle target-only response is a post-hoc analysis upper reference and is
    # never used for a detector, threshold, teacher input or student input.
    oracle = np.asarray(h1["iq_h1"], dtype=np.complex128) - np.asarray(h1["iq_h0"], dtype=np.complex128)
    oracle_teachers = _score_all_teachers(
        oracle.astype(np.complex64), covariance, physics, teacher_config, t1_options, device, include_maps=True
    )
    for method in ("B3", "T1", "ACE_AMF"):
        teacher_metrics[method]["oracle_target_only_peak_alignment"] = _map_alignment(
            oracle_teachers[method]["score_map"], h1["truth_range_cell"], h1["truth_doppler_hz"]
        )
        teacher_metrics[method]["oracle_target_only_restrictions"] = {
            "analysis_only": True,
            "used_for_detector": False,
            "used_for_threshold": False,
            "used_for_student": False,
            "definition": "H1 IQ minus paired H0 IQ",
        }

    print("R4: evaluating frozen E002 student A/B/C on CUDA", flush=True)
    frozen_thresholds = _read_thresholds(ROOT / "experiments/D2-A-E002/calibration_thresholds.csv")
    student_metrics: dict[str, Any] = {}
    student_predictions: dict[str, dict[str, np.ndarray]] = {}
    phase_trigger_candidates: list[dict[str, Any]] = []
    student_config = e002
    for spec in config["students"]:
        name = str(spec["name"])
        model = _student_from_checkpoint(ROOT / str(spec["checkpoint"]), student_config, device)
        torch.cuda.reset_peak_memory_stats(device)
        h0_prediction = _predict_student(ver["iq"], model, device)
        h1_prediction = _predict_student(h1["iq_h1"], model, device)
        h1_prediction["true_gate_doppler_hz"] = _predict_student_with_true_range(
            h1["iq_h1"], h1["truth_range_cell"], model, device
        )
        threshold = frozen_thresholds[name]
        student_summary = _evaluate_detector(
            h0_prediction["score"], h1_prediction["score"], threshold, ver_meta
        )
        student_summary["frozen_threshold_source"] = "D2-A-E002/calibration_thresholds.csv"
        student_summary["h1_task"] = _student_h1_metrics(h1, h1_prediction, threshold)
        h1_bootstrap = _paired_h1_summary(
            name, h0_prediction["score"], h1, h1_prediction, threshold
        )
        student_summary["h1_task"]["auc_vs_h0_verification"] = h1_bootstrap["auc"]
        student_summary["h1_task"]["scene_bootstrap_95_ci"] = h1_bootstrap["scene_bootstrap_95_ci"]
        phase_diagnostic = _student_phase_diagnostic(
            h1["iq_h1"], threshold, model, device
        )
        shift_diagnostic = _student_frequency_shift_diagnostic(
            h1["iq_h0"], h1["iq_h1"], model, device
        )
        student_summary["public_phase_rotation"] = phase_diagnostic
        student_summary["signal_level_frequency_shift"] = shift_diagnostic
        student_summary["cuda_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        student_summary["cuda_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        student_metrics[name] = student_summary
        student_predictions[name] = {
            key: value for key, value in h1_prediction.items()
            if key in ("score", "range_cell", "doppler_hz")
        }
        phase_trigger_candidates.append({
            "name": name,
            "phase_agreement": phase_diagnostic["decision_agreement"],
            "doppler_rotation_p95_hz": phase_diagnostic["doppler_change_p95_hz"],
        })
        del model
        torch.cuda.empty_cache()

    student_paired_differences = {}
    for first, second in (("B", "A"), ("C", "A"), ("C", "B")):
        student_paired_differences[f"{first}_minus_{second}"] = _paired_h1_method_difference(
            student_predictions[first],
            frozen_thresholds[first],
            student_predictions[second],
            frozen_thresholds[second],
            h1["truth_range_cell"],
            h1["truth_doppler_hz"],
            seed=2026102801,
        )

    cal_ids = set(np.asarray(cal["base_scene_id"]).astype(str))
    ver_ids = set(np.asarray(ver["base_scene_id"]).astype(str))
    h1_ids = set(np.asarray(h1["base_scene_id"]).astype(str))
    split_disjointness = {
        "h0_calibration_vs_verification": not bool(cal_ids & ver_ids),
        "h0_calibration_vs_h1_development": not bool(cal_ids & h1_ids),
        "h0_verification_vs_h1_development": not bool(ver_ids & h1_ids),
        "all_within_split_ids_unique": all(
            len(values) == len(ids)
            for values, ids in (
                (cal_ids, cal["base_scene_id"]),
                (ver_ids, ver["base_scene_id"]),
                (h1_ids, h1["base_scene_id"]),
            )
        ),
    }
    engineering_gate = bool(
        np.isfinite(numerical["cpu_vs_cuda"]["max_abs_score_difference"])
        and baseline_parity_passed
        and all(split_disjointness.values())
        and all(np.all(np.isfinite(values["score"])) for values in ver_teacher.values())
        and all(np.isfinite(value["pfa_ge"]) for value in student_metrics.values())
        and len(np.unique(h1["base_scene_id"])) == len(h1["base_scene_id"])
    )
    most_affected = min(phase_trigger_candidates, key=lambda item: item["phase_agreement"])
    largest_doppler_change = max(
        phase_trigger_candidates, key=lambda item: item["doppler_rotation_p95_hz"]
    )
    phase_trigger = s1_pilot_trigger(
        engineering_gate_passed=engineering_gate,
        public_phase_agreement=float(most_affected["phase_agreement"]),
        doppler_rotation_p95_hz=float(largest_doppler_change["doppler_rotation_p95_hz"]),
    )
    peak = {
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    metrics = {
        "experiment": "D2-A-R4",
        "protocol_version": 1.0,
        "git_sha": git("rev-parse", "HEAD"),
        "evidence_class": "exploratory_only_due_to_T001_G1_G2_fail",
        "historical_t001_status": historical,
        "historical_t001_modified": False,
        "data_counts": {
            "h0_calibration": 4096,
            "h0_verification": 4096,
            "h1_development": int(len(h1["iq_h1"])),
            "range_cells_per_cpi": 32,
            "range_cells_counted_as_independent_scenes": False,
        },
        "split_disjointness": {
            **split_disjointness,
            "seed_blocks": "R4-only 41M through 48.5M including conditional pilot evaluation",
            "disjoint_from_frozen_e001_r3_e002_t001_seed_blocks": True,
            "final_blind_generated": False,
        },
        "numerical_precision": numerical,
        "h0_scenario_label_independence": h0_scenario_independence,
        "cuda_b3_t1_cpu_parity": baseline_parity,
        "teacher_thresholds_calibrated_on_r4_h0_calibration": teacher_thresholds,
        "teachers": teacher_metrics,
        "teacher_paired_differences": teacher_paired_differences,
        "frozen_students": student_metrics,
        "student_paired_differences": student_paired_differences,
        "stage_gate": {
            "R4_engineering_gate_passed": engineering_gate,
            "R4_scale_invariance_gate_passed": numerical["r4_scale_invariance_gate"]["passed"],
            "CUDA_B3_T1_frozen_formula_parity_passed": baseline_parity_passed,
            "phase_augmentation_triggered": phase_trigger,
            "trigger_worst_phase_agreement_model": most_affected,
            "trigger_largest_doppler_rotation_change_model": largest_doppler_change,
            "S1_E1_truth_only_pilot": "ELIGIBLE_EXPLORATORY" if phase_trigger else "NOT_RUN_PHASE_TRIGGER_NOT_MET",
            "S2_or_KD_run": False,
            "final_blind_generated": False,
            "formal_T001_G1_G2_reclassified": False,
        },
        "resources": {
            **peak,
            "wall_seconds": time.perf_counter() - start_time,
            "gpu_name": torch.cuda.get_device_name(device),
            "inference_device": str(device),
            "teacher_precision": "complex64 CUDA",
            "student_precision": "FP32 CUDA",
            "cpu_teacher_fallback": False,
        },
    }
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "environment.json", _environment(device))
    return metrics


def _augment_pair_phase(batch: dict[str, np.ndarray], rng: np.random.Generator) -> None:
    """Apply one public phase rotation jointly to each paired H0/H1 example."""
    planes = np.asarray(batch["iq_ri"], dtype=np.float32)
    if planes.ndim != 4 or planes.shape[1:] != (6, 130, 32) or len(planes) % 2:
        raise ValueError("paired phase augmentation requires interleaved [H0,H1] IQ planes")
    pairs = len(planes) // 2
    real = planes[:, :3]
    imag = planes[:, 3:]
    complex_iq = real.astype(np.float64) + 1j * imag.astype(np.float64)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=pairs)
    factor = np.exp(1j * phase).astype(np.complex128)
    complex_iq = complex_iq.reshape(pairs, 2, 3, 130, 32) * factor[:, None, None, None, None]
    flat = complex_iq.reshape(2 * pairs, 3, 130, 32)
    batch["iq_ri"] = np.ascontiguousarray(
        np.concatenate((flat.real, flat.imag), axis=1).astype(np.float32)
    )


def _r4_train_s1_attempt(
    *,
    microbatch: int,
    pair_plan: np.ndarray,
    train_arrays: dict[str, np.ndarray],
    validation_arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    initial_state: dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
    phase_seed: int,
) -> dict[str, Any]:
    import math
    import csv as csv_module
    from radardistill.distillation.d2a_e002_losses import d2a_e002_loss
    from radardistill.training.d2a_e002 import (
        _paired_validation,
        make_paired_examples,
        scale_d2a_e002_microbatch_loss,
    )

    training = config["training"]
    effective_batch = int(training["effective_batch_size"])
    if microbatch <= 0 or effective_batch % microbatch or microbatch % 2:
        raise ValueError("microbatch must be an even divisor of effective batch")
    accumulation = effective_batch // microbatch
    model = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=True,
    ).to(device)
    model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["optimizer"]["learning_rate_initial"]),
        weight_decay=float(training["optimizer"]["weight_decay"]),
    )
    update_count = int(training["optimizer_updates_per_arm"])
    lr_initial = float(training["optimizer"]["learning_rate_initial"])
    lr_final = float(training["optimizer"]["learning_rate_final"])
    clip = float(training["optimizer"]["gradient_clip_norm"])
    validation_interval = int(training["validation_interval_updates"])
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "training_history.csv"
    history_fields = [
        "update", "learning_rate", "train_total", "train_presence",
        "train_range_cell", "train_range_offset", "train_doppler",
        "validation_task_loss", "validation_auc",
    ]
    phase_rng = np.random.default_rng(int(phase_seed))
    model.eval()
    initial_validation = _paired_validation(model, validation_arrays, config, device, microbatch)
    best_loss = float(initial_validation["task_loss"])
    best_update = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()

    def save_r4_checkpoint(
        path: Path,
        *,
        update: int,
        validation: dict[str, float],
        optimizer_state: torch.optim.Optimizer | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "experiment": "D2-A-R4-S1",
            "protocol_version": 1.0,
            "arm": "S1_public_phase_augmented_E1_truth_only",
            "update": int(update),
            "microbatch_size": int(microbatch),
            "model_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            },
            "validation": dict(validation),
            "teacher_distillation": False,
        }
        if optimizer_state is not None:
            payload["optimizer_state_dict"] = optimizer_state.state_dict()
        torch.save(payload, path)

    model.train()
    with history_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv_module.DictWriter(stream, fieldnames=history_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow({"update": 0, "validation_task_loss": best_loss, "validation_auc": initial_validation.get("auc")})
        stream.flush()
        for update_index, pair_ids in enumerate(pair_plan):
            fraction = update_index / max(update_count - 1, 1)
            learning_rate = lr_final + 0.5 * (lr_initial - lr_final) * (1.0 + math.cos(math.pi * fraction))
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            batch = make_paired_examples(train_arrays, pair_ids, include_teacher=False)
            _augment_pair_phase(batch, phase_rng)
            optimizer.zero_grad(set_to_none=True)
            terms = {name: 0.0 for name in ("presence", "range_cell", "range_offset", "doppler", "total")}
            for micro_start in range(0, effective_batch, microbatch):
                micro_stop = micro_start + microbatch
                x = torch.from_numpy(batch["iq_ri"][micro_start:micro_stop]).to(device)
                truth = {
                    key: torch.from_numpy(batch[key][micro_start:micro_stop]).to(device)
                    for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
                }
                outputs = model(x, compute_spectrum=False)
                losses = d2a_e002_loss(
                    outputs, truth, config, teacher_response=None, teacher_weight=0.0
                )
                if any(not torch.isfinite(value).all() for value in losses.values()):
                    raise FloatingPointError(f"non-finite S1 loss at update {update_index + 1}")
                micro_h1 = int(truth["truth_presence"].sum().item())
                scaled = scale_d2a_e002_microbatch_loss(
                    losses,
                    config,
                    microbatch_size=microbatch,
                    effective_batch_size=effective_batch,
                    microbatch_h1=micro_h1,
                    effective_h1=effective_batch // 2,
                    teacher_weight=0.0,
                )
                scaled.backward()
                for name in ("presence", "range_cell", "range_offset", "doppler", "total"):
                    terms[name] += float(losses["task_total" if name == "total" else name].detach().item()) / accumulation
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite S1 gradient at update {update_index + 1}")
            optimizer.step()
            row: dict[str, Any] = {
                "update": update_index + 1,
                "learning_rate": learning_rate,
                "train_total": terms["total"],
                "train_presence": terms["presence"],
                "train_range_cell": terms["range_cell"],
                "train_range_offset": terms["range_offset"],
                "train_doppler": terms["doppler"],
            }
            if (update_index + 1) % validation_interval == 0 or update_index + 1 == update_count:
                model.eval()
                validation = _paired_validation(model, validation_arrays, config, device, microbatch)
                row["validation_task_loss"] = float(validation["task_loss"])
                row["validation_auc"] = float(validation.get("auc", float("nan")))
                if float(validation["task_loss"]) < best_loss:
                    best_loss = float(validation["task_loss"])
                    best_update = update_index + 1
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                save_r4_checkpoint(
                    out_dir / "progress_checkpoint.pt",
                    update=update_index + 1,
                    validation=validation,
                    optimizer_state=optimizer,
                )
                model.train()
            writer.writerow(row)
            stream.flush()
    model.load_state_dict(best_state, strict=True)
    model.eval()
    save_r4_checkpoint(
        out_dir / "best_checkpoint.pt",
        update=best_update,
        validation=_paired_validation(model, validation_arrays, config, device, microbatch),
    )
    inference = TaskStudent(
        width=int(config["student"]["width"]),
        temporal_dilations=config["student"]["temporal_dilations"],
        include_spectrum_head=False,
    )
    inference.load_state_dict(
        {key: value for key, value in best_state.items() if not key.startswith("spectrum_head.")},
        strict=True,
    )
    torch.save(
        {
            "experiment": "D2-A-R4-S1",
            "protocol_version": 1.0,
            "has_k2_head": False,
            "model_state_dict": inference.state_dict(),
            "teacher_distillation": False,
        },
        out_dir / "best_inference.pt",
    )
    return {
        "microbatch_size": microbatch,
        "effective_batch_size": effective_batch,
        "optimizer_updates": update_count,
        "best_update": best_update,
        "initial_validation": initial_validation,
        "best_validation_task_loss": best_loss,
        "train_seconds": time.perf_counter() - start_time,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "checkpoint": str(out_dir / "best_inference.pt"),
        "history": str(history_path),
        "supervision": "E1 truth only",
        "phase_augmentation": "shared uniform phase per paired H0/H1 example",
        "distillation": False,
    }


def _pilot_eval_h1_cells(physics: dict[str, Any], config: dict[str, Any]) -> dict[str, np.ndarray]:
    pilot_config = json.loads(json.dumps(config))
    for cell in pilot_config["data"]["h1_development"]["cells"]:
        cell["seed_base"] = int(cell["seed_base"]) + 5_000_000
    return generate_h1_cells(physics, pilot_config)


def run_pilot(device_name: str) -> dict[str, Any]:
    device = _cuda_device(device_name)
    _require_pushed_protocol()
    out_dir = ROOT / "experiments/D2-A-R4"
    metrics_path = out_dir / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("run R4 diagnostics before requesting the conditional S1 pilot")
    diagnostics = json.loads(metrics_path.read_text(encoding="utf-8"))
    trigger = diagnostics["stage_gate"]["phase_augmentation_triggered"]
    if not trigger:
        raise RuntimeError("R4 phase-augmentation trigger was not met; do not train S1")
    if not diagnostics["stage_gate"]["R4_engineering_gate_passed"]:
        raise RuntimeError("R4 engineering gate did not pass; do not train S1")
    if diagnostics["stage_gate"]["S2_or_KD_run"] or diagnostics["stage_gate"]["final_blind_generated"]:
        raise RuntimeError("R4 diagnostic record violates T001 hard gate lineage")
    preflight_record = json.loads((out_dir / "preflight.json").read_text(encoding="utf-8"))
    if not preflight_record.get("passed") or preflight_record["git_sha"] != git("rev-parse", "HEAD"):
        raise RuntimeError("R4 preflight is missing or stale")
    config = read_yaml(out_dir / "config.yaml")
    physics = read_yaml(ROOT / "experiments/D2-A-E001/config.yaml")
    e002 = read_yaml(ROOT / "experiments/D2-A-E002/config.yaml")
    cache_root = out_dir / "data" / "r4_pilot_cache"
    if cache_root.exists():
        allowed = {"pilot_train", "pilot_validation"}
        unexpected = {path.name for path in cache_root.iterdir()} - allowed
        if unexpected:
            raise FileExistsError(
                f"preserving unexpected user data in R4 pilot cache root: {sorted(unexpected)}"
            )
    cache_root.mkdir(parents=True, exist_ok=True)
    levels = e002["data"]["course_levels"]
    train = e002["data"]["train"]
    validation = e002["data"]["validation"]
    train_counts = {
        level: spec.get("scenario_counts", spec.get("scenario_snr_counts"))
        for level, spec in levels.items()
    }
    from radardistill.training.d2a_e002_data import materialize_paired_split

    train_arrays = materialize_paired_split(
        cache_root,
        "pilot_train",
        physics,
        levels,
        train_counts,
        train["seed_bases_by_course"],
        assignment_seed=int(train["assignment_seed"]),
        chunk_size=int(train["generation_chunk_size"]),
        chunk_seed_stride=int(e002["data"]["chunk_seed_stride"]),
        include_k2=False,
    )
    validation_arrays = materialize_paired_split(
        cache_root,
        "pilot_validation",
        physics,
        levels,
        validation["cell_counts"],
        validation["seed_bases_by_course"],
        assignment_seed=int(validation["assignment_seed"]),
        chunk_size=int(validation["generation_chunk_size"]),
        chunk_seed_stride=int(e002["data"]["chunk_seed_stride"]),
        include_k2=False,
    )
    from radardistill.training.d2a_e002 import build_matched_pair_plans

    course_ids = {
        level: np.flatnonzero(np.asarray(train_arrays["course_level"]).astype(str) == level)
        for level in ("C0", "C1", "C2", "C3")
    }
    plans = build_matched_pair_plans(
        course_ids,
        updates_per_course=int(e002["training"]["curriculum_updates"]["C0"]),
        mixed_updates=int(e002["training"]["curriculum_updates"]["C4"]),
        batch_pairs=int(e002["training"]["paired_base_scenes_per_update"]),
        seed=int(e002["data"]["training_seeds"]["batch_plan"]),
    )
    initial_path = ROOT / "experiments/D2-A-E002/runs/seed_101/A/microbatch_16/initial_checkpoint.pt"
    initial_checkpoint = torch.load(initial_path, map_location="cpu", weights_only=True)
    initial_state = initial_checkpoint["model_state_dict"]
    seed = int(e002["training"]["training_seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    attempts: list[dict[str, Any]] = []
    training_result = None
    for microbatch in (
        int(config["data"]["pilot"]["microbatch_initial"]),
        *[int(value) for value in config["data"]["pilot"]["microbatch_fallbacks"]],
    ):
        attempt_dir = out_dir / "runs" / "seed_101" / "S1" / f"microbatch_{microbatch}"
        try:
            training_result = _r4_train_s1_attempt(
                microbatch=microbatch,
                pair_plan=plans["A"],
                train_arrays=train_arrays,
                validation_arrays=validation_arrays,
                config=e002,
                initial_state=initial_state,
                device=device,
                out_dir=attempt_dir,
                phase_seed=int(config["data"]["pilot"]["phase_augmentation_rng_seed"]),
            )
            break
        except torch.cuda.OutOfMemoryError:
            attempts.append({"microbatch": microbatch, "status": "OOM"})
            torch.cuda.empty_cache()
            if microbatch == int(config["data"]["pilot"]["microbatch_fallbacks"][-1]):
                raise
            shutil.rmtree(attempt_dir, ignore_errors=True)
    if training_result is None:
        raise RuntimeError("R4 S1 pilot failed to complete")
    attempts.append({"microbatch": training_result["microbatch_size"], "status": "completed"})
    torch.cuda.empty_cache()

    print("R4 S1 pilot: generating independent evaluation scenes", flush=True)
    pilot_eval = config["data"]["pilot"]["evaluation"]
    h0_cal = generate_h0_split(
        4096, int(pilot_eval["h0_calibration_seed_base"]),
        int(pilot_eval["h0_calibration_assignment_seed"]), physics,
    )
    h0_ver = generate_h0_split(
        4096, int(pilot_eval["h0_verification_seed_base"]),
        int(pilot_eval["h0_verification_assignment_seed"]), physics,
    )
    h1 = _pilot_eval_h1_cells(physics, config)
    pilot_checkpoints = {
        "S0_E002_A_frozen": ROOT / "experiments/D2-A-E002/runs/seed_101/A/microbatch_16/best_inference.pt",
        "S1_public_phase_augmented": Path(training_result["checkpoint"]),
    }
    evaluation: dict[str, Any] = {}
    for name, checkpoint_path in pilot_checkpoints.items():
        model = _student_from_checkpoint(checkpoint_path, e002, device)
        cal_prediction = _predict_student(h0_cal["iq"], model, device)
        threshold = float(np.quantile(cal_prediction["score"], 0.95, method="higher"))
        ver_prediction = _predict_student(h0_ver["iq"], model, device)
        h1_prediction = _predict_student(h1["iq_h1"], model, device)
        h1_prediction["true_gate_doppler_hz"] = _predict_student_with_true_range(
            h1["iq_h1"], h1["truth_range_cell"], model, device
        )
        meta = {
            **h0_ver,
            "observable_power": np.mean(np.abs(h0_ver["iq"].astype(np.complex128)) ** 2, axis=(1, 2, 3)),
        }
        summary = _evaluate_detector(ver_prediction["score"], h1_prediction["score"], threshold, meta)
        summary["h1_task"] = _student_h1_metrics(h1, h1_prediction, threshold)
        h1_bootstrap = _paired_h1_summary(
            name, cal_prediction["score"], h1, h1_prediction, threshold
        )
        summary["h1_task"]["auc_vs_h0_calibration"] = h1_bootstrap["auc"]
        summary["h1_task"]["scene_bootstrap_95_ci"] = h1_bootstrap["scene_bootstrap_95_ci"]
        summary["public_phase_rotation"] = _student_phase_diagnostic(
            h1["iq_h1"], threshold, model, device
        )
        evaluation[name] = summary
        del model
        torch.cuda.empty_cache()
    train_range = np.asarray(train_arrays["truth_range_cell"], dtype=np.float64)
    train_doppler = np.asarray(train_arrays["truth_doppler_hz"], dtype=np.float64)
    pilot_truth_baseline = {
        "range_constant": float(np.median(train_range)),
        "range_rmse_cells": float(np.sqrt(np.mean((h1["truth_range_cell"] - np.median(train_range)) ** 2))),
        "doppler_constant_circular_mean_hz": float(np.angle(np.mean(np.exp(2j * np.pi * train_doppler / 1300.0))) * 1300.0 / (2.0 * np.pi)),
    }
    baseline_f = pilot_truth_baseline["doppler_constant_circular_mean_hz"]
    pilot_truth_baseline["doppler_rmse_hz"] = float(np.sqrt(np.mean(
        wrap_frequency(h1["truth_doppler_hz"] - baseline_f, 1300.0) ** 2
    )))
    s1 = evaluation["S1_public_phase_augmented"]
    trigger_gate = (
        s1["public_phase_rotation"]["decision_agreement"] >= 0.99
        and s1["h1_task"]["auc_vs_h0_calibration"] > 0.5
        and s1["pfa_ge"] <= evaluation["S0_E002_A_frozen"]["pfa_ge"] + 0.02
        and s1["h1_task"]["doppler_rmse_hz"] < pilot_truth_baseline["doppler_rmse_hz"]
        and s1["h1_task"]["range_rmse_cells"] < pilot_truth_baseline["range_rmse_cells"]
    )
    result = {
        "experiment": "D2-A-R4-S1",
        "status": "exploratory_one_seed_truth_only",
        "git_sha": git("rev-parse", "HEAD"),
        "training_seed": seed,
        "data_reconstruction": {
            "train_scenes": int(len(train_arrays["base_scene_id"])),
            "validation_scenes": int(len(validation_arrays["base_scene_id"])),
            "same_seed_blocks_as_E002": True,
            "same_A_arm_pair_plan": True,
            "all_training_samples_cached_on_gpu": False,
        },
        "training_attempts": attempts,
        "training": training_result,
        "pilot_truth_constant_baseline": pilot_truth_baseline,
        "evaluation": evaluation,
        "S1_exploratory_gate_passed": bool(trigger_gate),
        "formal_stage_gate_effect": "none; T001 G1/G2 remain FAIL and T001 G3 remains NOT_EVALUATED",
        "S2_or_KD_run": False,
        "final_blind_generated": False,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    write_json(out_dir / "pilot.json", result)
    # Remove only the dedicated R4 pilot cache after its seed/config manifest
    # has been recorded; historical E002 caches and user data are untouched.
    write_json(out_dir / "pilot_data_generation.json", result["data_reconstruction"])
    for split in ("pilot_train", "pilot_validation"):
        manifest = cache_root / split / "manifest.json"
        if not manifest.is_file():
            raise RuntimeError(f"cannot clean R4 pilot cache without its manifest: {manifest}")
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_data.get("split") != split:
            raise RuntimeError(f"R4 pilot cache manifest mismatch: {manifest}")
    shutil.rmtree(cache_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "diagnostics", "pilot"), required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out_dir = ROOT / "experiments/D2-A-R4"
    if args.stage == "preflight":
        result = preflight(args.device)
        result["environment"] = _environment(_cuda_device(args.device))
        write_json(out_dir / "preflight.json", result)
        print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    elif args.stage == "diagnostics":
        result = run_diagnostics(args.device)
        print(json.dumps(json_safe(result["stage_gate"]), ensure_ascii=False, indent=2))
    else:
        result = run_pilot(args.device)
        print(json.dumps(json_safe({
            "S1_exploratory_gate_passed": result["S1_exploratory_gate_passed"],
            "checkpoint": result["training"]["checkpoint"],
            "cuda_device": result["cuda_device"],
        }), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
