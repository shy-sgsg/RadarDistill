#!/usr/bin/env python3
"""Run the frozen D0 simulator and APX15 teacher correctness evaluation."""

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
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.simulation.d0 import (  # noqa: E402
    channel_steering_vector,
    generate_complex_awgn,
    generate_d0_batch,
    synthesize_target_echo,
    verify_lfm_range_psf,
)
from radardistill.teachers.apx15_doppler import APX15DopplerTeacher  # noqa: E402


def _wrapped_error(estimate: np.ndarray, truth: np.ndarray, prf_hz: float) -> np.ndarray:
    return (estimate - truth + prf_hz / 2.0) % prf_hz - prf_hz / 2.0


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    negative_sorted = np.sort(np.asarray(negative, dtype=np.float64))
    positive = np.asarray(positive, dtype=np.float64)
    below = np.searchsorted(negative_sorted, positive, side="left")
    through = np.searchsorted(negative_sorted, positive, side="right")
    wins = 0.5 * (below + through)
    return float(wins.sum() / (positive.size * negative_sorted.size))


def _auc_bootstrap(
    positive: np.ndarray,
    negative: np.ndarray,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        resampled_positive = positive[rng.integers(0, positive.size, positive.size)]
        resampled_negative = negative[rng.integers(0, negative.size, negative.size)]
        samples[index] = _auc(resampled_positive, resampled_negative)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(low), float(high)]


def _same_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(left, right, equal_nan=True))


def _run_noiseless_teacher(
    config: dict[str, Any], teacher: APX15DopplerTeacher
) -> tuple[dict[str, float], dict[str, Any]]:
    count = int(config["correctness_evaluation"]["noiseless_teacher_examples"])
    batch_size = int(config["execution"]["batch_size_cpu_teacher"])
    seed_base = int(config["seeds"]["noiseless_teacher"])
    range_errors: list[np.ndarray] = []
    doppler_errors: list[np.ndarray] = []
    doppler_by_range: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, count, batch_size)):
        size = min(batch_size, count - start)
        batch = generate_d0_batch(
            size,
            seed_base + batch_index,
            np.inf,
            1.0,
            config,
        )
        prediction = teacher.predict(batch.iq)
        range_errors.append(prediction.teacher_range_cell - batch.truth_range_cell)
        doppler_errors.append(
            _wrapped_error(
                prediction.teacher_doppler_hz,
                batch.truth_doppler_hz,
                float(config["radar"]["prf_hz"]),
            )
        )
    rerr = np.concatenate(range_errors)
    derr = np.concatenate(doppler_errors)
    summary = {
        "count": int(count),
        "range_rmse_cells": float(np.sqrt(np.mean(rerr**2))),
        "range_max_abs_error_cells": float(np.max(np.abs(rerr))),
        "range_bias_cells": float(np.mean(rerr)),
        "doppler_rmse_hz": float(np.sqrt(np.mean(derr**2))),
        "doppler_max_abs_error_hz": float(np.max(np.abs(derr))),
        "doppler_bias_hz": float(np.mean(derr)),
        "doppler_p99_abs_error_hz": float(np.quantile(np.abs(derr), 0.99)),
    }
    thresholds = config["correctness_evaluation"]
    gates = {
        "range_rmse": summary["range_rmse_cells"]
        <= thresholds["noiseless_teacher_range_rmse_cells_max"],
        "range_max": summary["range_max_abs_error_cells"]
        <= thresholds["noiseless_teacher_range_abs_error_cells_max"],
        "doppler_rmse": summary["doppler_rmse_hz"]
        <= thresholds["noiseless_teacher_doppler_rmse_hz_max"],
        "doppler_max": summary["doppler_max_abs_error_hz"]
        <= thresholds["noiseless_teacher_doppler_abs_error_hz_max"],
    }
    return summary, gates


def _run_boundary_checks(
    config: dict[str, Any], teacher: APX15DopplerTeacher
) -> dict[str, Any]:
    eval_cfg = config["correctness_evaluation"]
    range_values = eval_cfg["range_boundary_cells"]
    frequency_values = eval_cfg["boundary_doppler_hz"]
    azimuth_interval = config["channels"]["azimuth_deg_interval"]
    elevation_interval = config["channels"]["elevation_deg_interval"]
    azimuth = float(sum(azimuth_interval) / 2.0)
    elevation = float(sum(elevation_interval) / 2.0)

    signals = []
    truth_range = []
    truth_doppler = []
    for range_cell in range_values:
        for frequency in frequency_values:
            signals.append(
                synthesize_target_echo(
                    float(range_cell),
                    float(frequency),
                    0.37,
                    1.0,
                    azimuth,
                    elevation,
                    config,
                )
            )
            truth_range.append(float(range_cell))
            truth_doppler.append(float(frequency))

    prediction = teacher.predict(np.stack(signals).astype(np.complex64))
    rerr = prediction.teacher_range_cell - np.asarray(truth_range)
    derr = _wrapped_error(
        prediction.teacher_doppler_hz,
        np.asarray(truth_doppler),
        float(config["radar"]["prf_hz"]),
    )
    passed = bool(
        np.all(np.isfinite(prediction.teacher_range_cell))
        and np.all(np.isfinite(prediction.teacher_doppler_hz))
        and np.all(np.abs(rerr) <= 1e-5)
        and np.all(np.abs(derr) <= 0.06)
        and np.all((prediction.teacher_range_cell >= 0.0)
                   & (prediction.teacher_range_cell <= config["radar"]["range_bin_count"] - 1))
    )
    return {
        "count": len(signals),
        "range_max_abs_error_cells": float(np.max(np.abs(rerr))),
        "doppler_max_abs_error_hz": float(np.max(np.abs(derr))),
        "range_outputs_inside_window": bool(
            np.all((prediction.teacher_range_cell >= 0.0)
                   & (prediction.teacher_range_cell <= config["radar"]["range_bin_count"] - 1))
        ),
        "passed": passed,
    }


def _run_native_fft_axis(
    config: dict[str, Any], teacher: APX15DopplerTeacher
) -> dict[str, Any]:
    prf = float(config["radar"]["prf_hz"])
    pulse_count = int(config["radar"]["pulse_count"])
    channels = int(config["radar"]["channel_count"])
    ranges = int(config["radar"]["range_bin_count"])
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(pulse_count, d=1.0 / prf))
    times = np.arange(pulse_count, dtype=np.float64) / prf
    tones = np.exp(2j * np.pi * frequency_axis[:, None] * times[None, :])
    iq = np.broadcast_to(
        tones[:, None, :, None], (pulse_count, channels, pulse_count, ranges)
    ).copy().astype(np.complex64)
    transformed = teacher.transform(iq, nfft=pulse_count)
    peak_bins = np.argmax(np.abs(transformed[:, 0, :, 0]) ** 2, axis=1)
    expected_bins = np.arange(pulse_count, dtype=np.int64)
    return {
        "tested_tones": pulse_count,
        "passed_tones": int(np.count_nonzero(peak_bins == expected_bins)),
        "max_bin_error": int(np.max(np.abs(peak_bins - expected_bins))),
        "frequency_axis_hz": [float(frequency_axis[0]), float(frequency_axis[-1])],
        "passed": bool(np.array_equal(peak_bins, expected_bins)),
    }


def _run_noisy_teacher(
    config: dict[str, Any], teacher: APX15DopplerTeacher
) -> tuple[list[dict[str, Any]], bool]:
    eval_cfg = config["correctness_evaluation"]
    snr_grid = [float(x) for x in eval_cfg["noisy_teacher_snr_db"]]
    per_class = int(eval_cfg["noisy_teacher_examples_per_class_per_snr"])
    batch_size = int(config["execution"]["batch_size_cpu_teacher"])
    seed_base = int(config["seeds"]["noisy_teacher"])
    records = []

    for snr_index, snr in enumerate(snr_grid):
        score_by_class: dict[bool, np.ndarray] = {}
        errors: dict[str, np.ndarray] = {}
        for presence in (False, True):
            scores: list[np.ndarray] = []
            ranges: list[np.ndarray] = []
            dopplers: list[np.ndarray] = []
            for batch_index, start in enumerate(range(0, per_class, batch_size)):
                count = min(batch_size, per_class - start)
                seed = seed_base + snr_index * 100000 + int(presence) * 10000 + batch_index
                batch = generate_d0_batch(
                    count,
                    seed,
                    snr,
                    1.0 if presence else 0.0,
                    config,
                )
                prediction = teacher.predict(batch.iq)
                scores.append(prediction.teacher_detection_score)
                if presence:
                    ranges.append(prediction.teacher_range_cell - batch.truth_range_cell)
                    dopplers.append(
                        _wrapped_error(
                            prediction.teacher_doppler_hz,
                            batch.truth_doppler_hz,
                            float(config["radar"]["prf_hz"]),
                        )
                    )
            score_by_class[presence] = np.concatenate(scores)
            if presence:
                errors["range"] = np.concatenate(ranges)
                errors["doppler"] = np.concatenate(dopplers)

        negative = score_by_class[False]
        positive = score_by_class[True]
        auc = _auc(positive, negative)
        ci = _auc_bootstrap(
            positive,
            negative,
            int(eval_cfg["bootstrap_replicates"]),
            int(eval_cfg["bootstrap_seed"]) + snr_index,
        )
        record = {
            "snr_db": snr,
            "n_h0": int(negative.size),
            "n_h1": int(positive.size),
            "teacher_score_auc": auc,
            "teacher_score_auc_ci95_low": ci[0],
            "teacher_score_auc_ci95_high": ci[1],
            "h0_score_median": float(np.median(negative)),
            "h0_score_p95": float(np.quantile(negative, 0.95)),
            "h0_score_p99": float(np.quantile(negative, 0.99)),
            "h1_range_rmse_cells": float(np.sqrt(np.mean(errors["range"] ** 2))),
            "h1_doppler_rmse_hz": float(np.sqrt(np.mean(errors["doppler"] ** 2))),
            "h1_doppler_bias_hz": float(np.mean(errors["doppler"])),
        }
        records.append(record)

    gate_cfg = eval_cfg["teacher_detection_score_auc_gate"]
    target_record = next(x for x in records if x["snr_db"] == float(gate_cfg["snr_db"]))
    return records, bool(target_record["teacher_score_auc"] >= float(gate_cfg["minimum_auc"]))


def _run_environment(root: Path, runtime_s: float) -> dict[str, Any]:
    def first_line(command: list[str]) -> str | None:
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return result.stdout.splitlines()[0] if result.stdout else None

    try:
        fftw_version = subprocess.run(
            ["pkg-config", "--modversion", "fftw3"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        fftw_version = None
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_changes = subprocess.run(
            ["git", "diff", "--quiet"], cwd=root, check=False
        ).returncode != 0 or subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=root, check=False
        ).returncode != 0
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_sha = None
        tracked_changes = None

    return {
        "git_sha": git_sha,
        "tracked_worktree_changes": tracked_changes,
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gxx": first_line(["g++", "--version"]),
        "fftw3": fftw_version,
        "cuda_training_used": False,
        "runtime_seconds": float(runtime_s),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def _write_analysis(path: Path, metrics: dict[str, Any]) -> None:
    gate_names = sorted(metrics["gates"])
    status = metrics["status"]
    lines = [
        "# D0-E001 仿真与 APX15 教师正确性结果",
        "",
        f"**结果：{status}**。本实验仅验证距离压缩简化模型及慢时间频谱子链，不训练学生，不代表完整 CoreX 生产链或实测数据。",
        "",
        "## 正确性门",
        "",
        "| 检查 | 结果 |",
        "|---|---|",
    ]
    lines.extend(f"| {name} | {'PASS' if metrics['gates'][name] else 'FAIL'} |" for name in gate_names)
    lfm = metrics["waveform_verifier"]
    noise = metrics["noise_moment"]
    noiseless = metrics["noiseless_teacher"]
    boundary = metrics["boundary_checks"]
    lines.extend(
        [
            "",
            "## 物理模型与教师核对",
            "",
            f"- 离散 LFM 匹配滤波 PSF 最大归一化幅度误差：`{lfm['max_normalized_magnitude_error']:.8g}`。",
            f"- 复 AWGN 经验功率：`{noise['measured_power']:.8g}`，相对误差 `{noise['relative_error']:.4%}`。",
            f"- 无噪声教师：range RMSE `{noiseless['range_rmse_cells']:.8g}` cell，Doppler RMSE `{noiseless['doppler_rmse_hz']:.8g}` Hz，最大 Doppler 误差 `{noiseless['doppler_max_abs_error_hz']:.8g}` Hz。",
            f"- Nyquist 邻域/范围端点检查：range 最大误差 `{boundary['range_max_abs_error_cells']:.8g}` cell，Doppler 最大环绕误差 `{boundary['doppler_max_abs_error_hz']:.8g}` Hz。",
            "",
            "## 含噪教师诊断",
            "",
            "教师检测分数为细网格峰值/全图中位功率。D0 不从这些样本选择工作阈值；D1 将使用独立 H0 校准集。",
            "",
            "| SNR (dB) | H0/H1 数量 | AUC (95% bootstrap CI) | H1 range RMSE (cell) | H1 Doppler RMSE (Hz) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for record in metrics["teacher_by_snr"]:
        lines.append(
            f"| {record['snr_db']:g} | {record['n_h0']}/{record['n_h1']} | "
            f"{record['teacher_score_auc']:.4f} [{record['teacher_score_auc_ci95_low']:.4f}, {record['teacher_score_auc_ci95_high']:.4f}] | "
            f"{record['h1_range_rmse_cells']:.5g} | {record['h1_doppler_rmse_hz']:.5g} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "通过只说明冻结的理想距离 PSF、噪声采样和 APX15 FFT 布局适配符合本协议。距离轴是理论 `c/(2B)` 分辨单元；本阶段没有验证实际 5 m 距离分辨率字段、完整快时间脉压链、杂波、多目标、通道误差或 CoreX 执行。教师 AUC 是 AWGN 单目标开发诊断，不是运行检测阈值或系统虚警结论。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_d0_e001(config_path: Path, output_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("experiment", {}).get("id") != "D0-E001":
        raise ValueError("configuration experiment.id must be D0-E001")
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "metrics.json").exists():
        raise FileExistsError(
            f"{output_dir / 'metrics.json'} exists; preserve this run and select a new --output directory"
        )
    effective = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    config_snapshot = output_dir / "config_used.yaml"
    config_snapshot.write_text(effective, encoding="utf-8")
    protocol_source = config_path.with_name("protocol.md")
    protocol_snapshot = output_dir / "protocol.md"
    if protocol_source.resolve() != protocol_snapshot.resolve():
        protocol_snapshot.write_text(protocol_source.read_text(encoding="utf-8"), encoding="utf-8")

    lfm = verify_lfm_range_psf(config)
    lfm_gate = lfm["max_normalized_magnitude_error"] <= float(
        config["waveform"]["verifier"]["max_normalized_magnitude_error"]
    )

    pair_count = int(config["correctness_evaluation"]["matched_noise_counterfactual_pairs"])
    pair_seed = int(config["seeds"]["matched_noise_counterfactual"])
    h0_pair = generate_d0_batch(pair_count, pair_seed, 0.0, 0.0, config)
    h1_pair = generate_d0_batch(pair_count, pair_seed, 0.0, 1.0, config)
    residual = (
        h1_pair.iq.astype(np.complex128)
        - h0_pair.iq.astype(np.complex128)
        - h1_pair.signal_iq.astype(np.complex128)
    )
    target_norm = float(np.linalg.norm(h1_pair.signal_iq.astype(np.complex128)))
    counterfactual_error = float(np.linalg.norm(residual) / max(target_norm, np.finfo(float).tiny))
    counterfactual_gate = bool(
        np.array_equal(h0_pair.noise_iq, h1_pair.noise_iq)
        and _same_with_nan(h0_pair.reference_amplitude, h1_pair.reference_amplitude)
        and _same_with_nan(h0_pair.noise_power, h1_pair.noise_power)
        and _same_with_nan(h0_pair.nuisance_initial_phase_rad, h1_pair.nuisance_initial_phase_rad)
        and _same_with_nan(h0_pair.nuisance_azimuth_deg, h1_pair.nuisance_azimuth_deg)
        and _same_with_nan(h0_pair.nuisance_elevation_deg, h1_pair.nuisance_elevation_deg)
        and counterfactual_error <= 1e-6
    )

    noise_count = int(config["correctness_evaluation"]["noise_moment_complex_samples"])
    noise_rng = np.random.default_rng(int(config["seeds"]["noise_moment"]))
    noise = generate_complex_awgn(noise_rng, (noise_count,), 1.0)
    measured_noise_power = float(np.mean(np.abs(noise) ** 2, dtype=np.float64))
    noise_relative_error = abs(measured_noise_power - 1.0)
    noise_gate = noise_relative_error <= float(
        config["correctness_evaluation"]["noise_power_relative_error_max"]
    )

    repeat_count = int(config["correctness_evaluation"]["reproducibility_examples"])
    repeat_seed = int(config["seeds"]["reproducibility"])
    repeat_a = generate_d0_batch(repeat_count, repeat_seed, (-5.0, 5.0), 0.5, config)
    repeat_b = generate_d0_batch(repeat_count, repeat_seed, (-5.0, 5.0), 0.5, config)
    reproducibility_fields = (
        "iq",
        "signal_iq",
        "noise_iq",
        "truth_presence",
        "truth_range_cell",
        "truth_doppler_hz",
        "truth_initial_phase_rad",
        "nuisance_initial_phase_rad",
        "truth_target_amplitude",
        "reference_amplitude",
        "noise_power",
        "snr_db",
        "truth_azimuth_deg",
        "truth_elevation_deg",
        "nuisance_azimuth_deg",
        "nuisance_elevation_deg",
    )
    reproducibility_gate = all(
        _same_with_nan(getattr(repeat_a, field), getattr(repeat_b, field))
        for field in reproducibility_fields
    ) and repeat_a.iq.shape == (repeat_count, 3, 130, 32)

    channel_count = int(config["correctness_evaluation"]["channel_steering_examples"])
    steering_rng = np.random.default_rng(int(config["seeds"]["channel_steering"]))
    az_interval = config["channels"]["azimuth_deg_interval"]
    el_interval = config["channels"]["elevation_deg_interval"]
    azimuths = steering_rng.uniform(*az_interval, size=channel_count)
    elevations = steering_rng.uniform(*el_interval, size=channel_count)
    positions = np.asarray(config["channels"]["positions_xyz_m"], dtype=np.float64)
    relative_positions = positions - positions.mean(axis=0, keepdims=True)
    wavelength = float(config["waveform"]["propagation_speed_mps"]) / float(
        config["waveform"]["carrier_hz"]
    )
    steering_error = 0.0
    steering_modulus_error = 0.0
    for azimuth, elevation in zip(azimuths, elevations):
        az = np.deg2rad(azimuth)
        el = np.deg2rad(elevation)
        look = np.asarray(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        expected = np.exp(-2j * np.pi * (relative_positions @ look) / wavelength)
        actual = channel_steering_vector(float(azimuth), float(elevation), config)
        steering_error = max(steering_error, float(np.max(np.abs(actual - expected))))
        steering_modulus_error = max(
            steering_modulus_error, float(np.max(np.abs(np.abs(actual) - 1.0)))
        )
    steering_gate = steering_error <= 1e-12 and steering_modulus_error <= 1e-12

    teacher = APX15DopplerTeacher(config)
    try:
        native_fft_axis = _run_native_fft_axis(config, teacher)
        noiseless_teacher, noiseless_gates = _run_noiseless_teacher(config, teacher)
        boundary_checks = _run_boundary_checks(config, teacher)
        teacher_by_snr, teacher_auc_gate = _run_noisy_teacher(config, teacher)
    finally:
        teacher.close()

    gates = {
        "lfm_matched_filter_psf": bool(lfm_gate),
        "counterfactual_h0_h1_background": bool(counterfactual_gate),
        "complex_awgn_power": bool(noise_gate),
        "seed_reproducibility_and_shape": bool(reproducibility_gate),
        "three_channel_steering": bool(steering_gate),
        "native_apx15_fft_axis": bool(native_fft_axis["passed"]),
        "noiseless_teacher_range_rmse": bool(noiseless_gates["range_rmse"]),
        "noiseless_teacher_range_max": bool(noiseless_gates["range_max"]),
        "noiseless_teacher_doppler_rmse": bool(noiseless_gates["doppler_rmse"]),
        "noiseless_teacher_doppler_max": bool(noiseless_gates["doppler_max"]),
        "nyquist_and_range_boundaries": bool(boundary_checks["passed"]),
        "teacher_auc_at_minus_10_db": bool(teacher_auc_gate),
    }
    metrics = {
        "experiment_id": "D0-E001",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol_status": "preregistered_before_D0_implementation",
        "waveform_verifier": lfm,
        "counterfactual_h0_h1": {
            "pairs": pair_count,
            "identical_noise_arrays": bool(np.array_equal(h0_pair.noise_iq, h1_pair.noise_iq)),
            "identical_reference_amplitude": bool(
                _same_with_nan(h0_pair.reference_amplitude, h1_pair.reference_amplitude)
            ),
            "identical_noise_power": bool(_same_with_nan(h0_pair.noise_power, h1_pair.noise_power)),
            "relative_target_residual_error": counterfactual_error,
        },
        "noise_moment": {
            "complex_samples": noise_count,
            "requested_power": 1.0,
            "measured_power": measured_noise_power,
            "relative_error": noise_relative_error,
        },
        "reproducibility": {
            "count": repeat_count,
            "identical_all_fields": bool(reproducibility_gate),
            "shape": list(repeat_a.iq.shape),
        },
        "channel_steering": {
            "count": channel_count,
            "max_relative_phase_response_error": steering_error,
            "max_unit_modulus_error": steering_modulus_error,
        },
        "native_fft_axis": native_fft_axis,
        "noiseless_teacher": noiseless_teacher,
        "boundary_checks": boundary_checks,
        "teacher_by_snr": teacher_by_snr,
        "gates": gates,
    }
    runtime_s = time.perf_counter() - started
    environment = _run_environment(ROOT, runtime_s)
    metrics["environment"] = environment
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "snr_db",
        "n_h0",
        "n_h1",
        "teacher_score_auc",
        "teacher_score_auc_ci95_low",
        "teacher_score_auc_ci95_high",
        "h0_score_median",
        "h0_score_p95",
        "h0_score_p99",
        "h1_range_rmse_cells",
        "h1_doppler_rmse_hz",
        "h1_doppler_bias_hz",
    ]
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(teacher_by_snr)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_analysis(output_dir / "analysis.md", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D0-E001" / "config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "D0-E001",
    )
    args = parser.parse_args()
    metrics = run_d0_e001(args.config, args.output)
    print(
        json.dumps(
            {
                "experiment_id": metrics["experiment_id"],
                "status": metrics["status"],
                "gates_passed": sum(bool(x) for x in metrics["gates"].values()),
                "gate_count": len(metrics["gates"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    if metrics["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
