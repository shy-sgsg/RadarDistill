#!/usr/bin/env python3
"""Run the preregistered D2-A R0/R1 development simulation."""

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
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.d2a_baselines import DetectionOutput, dpca_difference, run_b0, run_b1, run_b2, run_b3
from radardistill.evaluation.d2a_e001 import cluster_bootstrap_interval, summarize_detector
from radardistill.simulation.d2a import compute_scene_power_metrics, generate_d2a_paired_batch


METHODS = ("B0", "B1", "B2", "B3")


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
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _run_methods(iq: np.ndarray, config: dict[str, Any]) -> tuple[dict[str, DetectionOutput], dict[str, float]]:
    prf = float(config["radar"]["prf_hz"])
    methods = {
        "B0": lambda: run_b0(iq, prf),
        "B1": lambda: run_b1(iq, prf),
        "B2": lambda: run_b2(iq, prf, config),
        "B3": lambda: run_b3(iq, prf, config),
    }
    outputs: dict[str, DetectionOutput] = {}
    seconds_per_scene: dict[str, float] = {}
    scene_count = int(iq.shape[0]) if iq.ndim == 4 else 1
    for method, run in methods.items():
        started = time.perf_counter()
        outputs[method] = run()
        seconds_per_scene[method] = (time.perf_counter() - started) / scene_count
    return outputs, seconds_per_scene


def _combine_outputs(parts: list[DetectionOutput]) -> DetectionOutput:
    return DetectionOutput(
        score=np.concatenate([np.atleast_1d(np.asarray(part.score)) for part in parts]),
        doppler_hz=np.concatenate([np.atleast_1d(np.asarray(part.doppler_hz)) for part in parts]),
        range_cell=np.concatenate([np.atleast_1d(np.asarray(part.range_cell)) for part in parts]),
        angle_deg=(
            None
            if parts[0].angle_deg is None
            else np.concatenate([np.atleast_1d(np.asarray(part.angle_deg)) for part in parts])
        ),
        rd_power=np.concatenate(
            [part.rd_power[None, ...] if part.rd_power.ndim == 2 else part.rd_power for part in parts],
            axis=0,
        ),
    )


def _iter_paired_batches(
    count: int,
    seed_start: int,
    snr_db: float,
    cnr_db: float,
    scenario_id: str,
    config: dict[str, Any],
    batch_size: int,
):
    for start in range(0, count, batch_size):
        current_count = min(batch_size, count - start)
        yield generate_d2a_paired_batch(
            current_count,
            seed_start + start,
            snr_db,
            cnr_db,
            config,
            scenario_id=scenario_id,
        )


def _calibrate_thresholds(
    config: dict[str, Any], batch_size: int
) -> tuple[dict[tuple[float, float], dict[str, float]], dict[str, Any]]:
    evaluation_config = config["development_evaluation"]
    count = int(evaluation_config["calibration_h0_scenes_per_snr_cnr_condition"])
    target_pfa = float(evaluation_config["target_false_alarm_probability_per_cpi"])
    seed_start = int(evaluation_config["calibration_scene_seed_start"])
    unique_conditions = evaluation_config["unique_h0_calibration_conditions"]
    thresholds: dict[tuple[float, float], dict[str, float]] = {}
    calibration_records: dict[str, Any] = {}

    for condition_index, condition in enumerate(unique_conditions):
        snr_db = float(condition["snr_db"])
        cnr_db = float(condition["cnr_db"])
        score_parts: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
        for batch in _iter_paired_batches(
            count,
            seed_start + condition_index * 10000,
            snr_db,
            cnr_db,
            "separated",
            config,
            batch_size,
        ):
            outputs, _ = _run_methods(batch.iq_h0, config)
            for method, output in outputs.items():
                score_parts[method].append(np.atleast_1d(np.asarray(output.score, dtype=np.float64)))
        condition_key = (snr_db, cnr_db)
        thresholds[condition_key] = {}
        calibration_records[f"snr={snr_db:g},cnr={cnr_db:g}"] = {}
        for method in METHODS:
            scores = np.concatenate(score_parts[method])
            threshold = float(np.quantile(scores, 1.0 - target_pfa, method="higher"))
            thresholds[condition_key][method] = threshold
            calibration_records[f"snr={snr_db:g},cnr={cnr_db:g}"][method] = {
                "n_h0": int(scores.size),
                "threshold": threshold,
                "empirical_pfa_cpi": float(np.mean(scores >= threshold)),
                "quantile_method": "higher",
            }
    return thresholds, calibration_records


def _evaluate_conditions(
    config: dict[str, Any],
    thresholds: dict[tuple[float, float], dict[str, float]],
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    evaluation_config = config["development_evaluation"]
    count = int(evaluation_config["evaluation_paired_scenes_per_condition"])
    seed_start = int(evaluation_config["evaluation_scene_seed_start"])
    bootstrap_replicates = int(evaluation_config["cluster_bootstrap_replicates"])
    bootstrap_seed = int(evaluation_config["cluster_bootstrap_seed"])
    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "B3": {
            "separated_clutter_suppression_db": [],
            "separated_target_retention": [],
            "separated_scene_ids": [],
        }
    }

    for condition_index, condition in enumerate(evaluation_config["conditions"]):
        scenario_id = str(condition["scenario_id"])
        snr_db = float(condition["snr_db"])
        cnr_db = float(condition["cnr_db"])
        condition_key = (snr_db, cnr_db)
        outputs_h0: dict[str, list[DetectionOutput]] = {method: [] for method in METHODS}
        outputs_h1: dict[str, list[DetectionOutput]] = {method: [] for method in METHODS}
        scene_ids: list[np.ndarray] = []
        range_truth: list[np.ndarray] = []
        doppler_truth: list[np.ndarray] = []
        scene_snr: list[np.ndarray] = []
        scene_cnr: list[np.ndarray] = []
        scene_scnr: list[np.ndarray] = []
        runtime_parts: dict[str, list[float]] = {method: [] for method in METHODS}
        clutter_suppression_parts: list[np.ndarray] = []
        target_retention_parts: list[np.ndarray] = []

        for batch in _iter_paired_batches(
            count,
            seed_start + condition_index * 10000,
            snr_db,
            cnr_db,
            scenario_id,
            config,
            batch_size,
        ):
            h0_outputs, h0_runtime = _run_methods(batch.iq_h0, config)
            h1_outputs, h1_runtime = _run_methods(batch.iq_h1, config)
            for method in METHODS:
                outputs_h0[method].append(h0_outputs[method])
                outputs_h1[method].append(h1_outputs[method])
                runtime_parts[method].append((h0_runtime[method] + h1_runtime[method]) / 2.0)
            scene_ids.append(batch.base_scene_id)
            range_truth.append(batch.truth_target_range_cell)
            doppler_truth.append(batch.truth_target_doppler_hz)

            power_metrics = compute_scene_power_metrics(
                batch.target_signal_iq, batch.clutter_iq, batch.noise_iq
            )
            scene_snr.append(np.asarray(power_metrics["snr_db"]))
            scene_cnr.append(np.asarray(power_metrics["cnr_db"]))
            scene_scnr.append(np.asarray(power_metrics["scnr_db"]))

            if scenario_id == "separated":
                clutter_after = dpca_difference(batch.clutter_iq, config)
                target_after = dpca_difference(batch.target_signal_iq, config)
                clutter_before_power = np.mean(np.abs(batch.clutter_iq[:, 0]) ** 2, axis=(1, 2))
                clutter_after_power = np.mean(np.abs(clutter_after) ** 2, axis=(1, 2))
                target_before_power = np.mean(np.abs(batch.target_signal_iq[:, 0]) ** 2, axis=(1, 2))
                target_after_power = np.mean(np.abs(target_after) ** 2, axis=(1, 2))
                clutter_retention = clutter_after_power / np.maximum(clutter_before_power, 1e-30)
                target_retention = target_after_power / np.maximum(target_before_power, 1e-30)
                clutter_suppression_db = -10.0 * np.log10(np.maximum(clutter_retention, 1e-30))
                clutter_suppression_parts.append(clutter_suppression_db)
                target_retention_parts.append(target_retention)

        ids = np.concatenate(scene_ids)
        truth_range = np.concatenate(range_truth)
        truth_doppler = np.concatenate(doppler_truth)
        measured_snr = np.concatenate(scene_snr)
        measured_cnr = np.concatenate(scene_cnr)
        measured_scnr = np.concatenate(scene_scnr)
        for method in METHODS:
            h0 = _combine_outputs(outputs_h0[method])
            h1 = _combine_outputs(outputs_h1[method])
            summary = summarize_detector(
                h0,
                h1,
                thresholds[condition_key][method],
                truth_range,
                truth_doppler,
                ids,
                prf_hz=float(config["radar"]["prf_hz"]),
                doppler_gate_hz=20.0,
                range_gate_bins=1.0,
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + condition_index * 100 + METHODS.index(method),
            )
            summary.update(
                {
                    "scenario_id": scenario_id,
                    "group": str(condition["group"]),
                    "snr_db_input": snr_db,
                    "cnr_db_requested": cnr_db,
                    "mean_cpi_snr_db": float(np.mean(measured_snr)),
                    "mean_cpi_cnr_db": float(np.mean(measured_cnr)),
                    "mean_cpi_scnr_db": float(np.mean(measured_scnr)),
                    "mean_detector_seconds_per_cpi": float(np.mean(runtime_parts[method])),
                    "method": method,
                }
            )
            rows.append(summary)

        if scenario_id == "separated":
            diagnostics["B3"]["separated_clutter_suppression_db"].extend(
                np.concatenate(clutter_suppression_parts).tolist()
            )
            diagnostics["B3"]["separated_target_retention"].extend(
                np.concatenate(target_retention_parts).tolist()
            )
            diagnostics["B3"]["separated_scene_ids"].extend(ids.tolist())

    clutter_values = np.asarray(diagnostics["B3"]["separated_clutter_suppression_db"], dtype=np.float64)
    target_values = np.asarray(diagnostics["B3"]["separated_target_retention"], dtype=np.float64)
    diagnostic_ids = np.asarray(diagnostics["B3"]["separated_scene_ids"])
    bootstrap_replicates = int(evaluation_config["cluster_bootstrap_replicates"])
    bootstrap_seed = int(evaluation_config["cluster_bootstrap_seed"])
    clutter_ci = cluster_bootstrap_interval(
        clutter_values,
        diagnostic_ids,
        statistic=np.median,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 901,
    )
    target_ci = cluster_bootstrap_interval(
        target_values,
        diagnostic_ids,
        statistic=np.median,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed + 902,
    )
    median_clutter = float(np.median(clutter_values))
    median_target = float(np.median(target_values))
    diagnostics["B3"] = {
        "diagnostic_scenes": int(diagnostic_ids.size),
        "median_separated_clutter_suppression_db": median_clutter,
        "median_separated_clutter_suppression_db_ci95": list(clutter_ci),
        "median_separated_target_retention": median_target,
        "median_separated_target_retention_ci95": list(target_ci),
        "eligible_under_preregistered_gate": bool(median_clutter > 0.0 and median_target >= 0.5),
        "target_retention_reference": "raw_unscaled_difference_power_relative_to_input_channel_0; may exceed 1",
    }
    return rows, diagnostics


def _write_plot(rows: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenarios = ("separated", "partial_overlap")
    methods = METHODS
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {method: color for method, color in zip(methods, ("#334e68", "#2f855a", "#d69e2e", "#805ad5"))}
    for scenario in scenarios:
        for method in methods:
            points = sorted(
                [row for row in rows if row["scenario_id"] == scenario and row["cnr_db_requested"] == 10.0 and row["method"] == method],
                key=lambda row: row["snr_db_input"],
            )
            if len(points) < 2:
                continue
            label = f"{method} / {scenario}"
            axes[0].plot(
                [row["snr_db_input"] for row in points],
                [row["pd"] for row in points],
                marker="o",
                color=colors[method],
                linestyle="-" if scenario == "separated" else "--",
                label=label,
            )
            rmse = [row["doppler_rmse_hz_detected"] for row in points]
            axes[1].plot(
                [row["snr_db_input"] for row in points],
                [np.nan if value is None else value for value in rmse],
                marker="o",
                color=colors[method],
                linestyle="-" if scenario == "separated" else "--",
                label=label,
            )
    axes[0].set(title="Detection probability at calibrated Pfa_CPI=0.05", xlabel="Nominal input SNR (dB)", ylabel="Pd", ylim=(0, 1.02))
    axes[1].set(title="Doppler RMSE among detections", xlabel="Nominal input SNR (dB)", ylabel="RMSE (Hz)")
    axes[0].grid(True, alpha=0.25)
    axes[1].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].legend(fontsize=7, ncol=2)
    figure.suptitle("D2-A-E001 synthetic development baselines (CNR=10 dB)")
    figure.savefig(output_dir / "rmse_pd_snr.png", dpi=170)
    plt.close(figure)


def _environment_record() -> dict[str, Any]:
    gpu_info: str | None = None
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pytorch_available": False,
        "device_used": "CPU for deterministic simulation and classical baselines; no neural network training",
        "nvidia_smi": gpu_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "experiments" / "D2-A-E001" / "config.yaml")
    parser.add_argument("--output-dir", default=ROOT / "experiments" / "D2-A-E001")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    with (output_dir / "config_used.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False, allow_unicode=True)
    with (output_dir / "environment.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(_environment_record()), stream, ensure_ascii=False, indent=2)

    started = time.perf_counter()
    thresholds, calibration = _calibrate_thresholds(config, args.batch_size)
    rows, diagnostics = _evaluate_conditions(config, thresholds, args.batch_size)
    elapsed = time.perf_counter() - started
    b3_eligible = diagnostics["B3"]["eligible_under_preregistered_gate"]
    for row in rows:
        row["method_eligible_for_ranking"] = bool(row["method"] != "B3" or b3_eligible)

    metrics = {
        "experiment_id": config["experiment"]["id"],
        "data_are_synthetic": True,
        "formal_student_training_run": False,
        "final_blind_generated": False,
        "target_pfa_cpi": config["development_evaluation"]["target_false_alarm_probability_per_cpi"],
        "threshold_calibration": calibration,
        "dpca_diagnostics": diagnostics,
        "condition_results": rows,
        "elapsed_seconds": elapsed,
        "bootstrap_replicates": config["development_evaluation"]["cluster_bootstrap_replicates"],
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(metrics), stream, ensure_ascii=False, indent=2, allow_nan=False)

    csv_fields = [
        "scenario_id", "group", "method", "method_eligible_for_ranking", "snr_db_input",
        "cnr_db_requested", "mean_cpi_snr_db", "mean_cpi_cnr_db", "mean_cpi_scnr_db",
        "n_scenes", "threshold", "pfa_cpi", "pfa_cpi_ci95", "pfa_cut",
        "false_alarms_per_cpi", "suprathreshold_rd_cells_per_cpi", "pd", "pd_ci95", "roc_auc", "roc_auc_ci95",
        "range_rmse_cells_detected", "range_rmse_cells_detected_ci95",
        "doppler_rmse_hz_detected", "doppler_rmse_hz_detected_ci95", "joint_success",
        "joint_success_ci95", "mean_detector_seconds_per_cpi",
    ]
    with (output_dir / "result.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_safe(row.get(key)), ensure_ascii=False) if key.endswith("ci95") else row.get(key) for key in csv_fields})

    _write_plot(rows, output_dir)
    primary = [row for row in rows if row["group"] == "primary"]
    separated_primary = [row for row in primary if row["scenario_id"] == "separated"]
    best = max(separated_primary, key=lambda row: row["roc_auc"])
    analysis_lines = [
        "# D2-A-E001 开发实验分析",
        "",
        f"运行耗时：{elapsed:.1f} s；全部数据为合成开发场景。正式学生训练和 final blind 均未运行。",
        "",
        f"分离场景 A 的主条件中最高点估计 ROC AUC：{best['method']}，AUC={best['roc_auc']:.4f}，Pd={best['pd']:.4f}，Pfa_CPI={best['pfa_cpi']:.4f}。此处仅作快速索引，按场景解读见下表和阶段审查。",
        "",
        f"DPCA 门：静止杂波中位抑制 {diagnostics['B3']['median_separated_clutter_suppression_db']:.2f} dB {diagnostics['B3']['median_separated_clutter_suppression_db_ci95']}；分离场景目标功率比中位数 {diagnostics['B3']['median_separated_target_retention']:.3f} {diagnostics['B3']['median_separated_target_retention_ci95']}；eligible={b3_eligible}。目标功率比相对单输入通道，未归一化差分可使其超过 1；几何仍依赖共址 Tx 假设。",
        "",
        "| 场景 | 方法 | SNR/CNR (dB) | Pd (95% CI) | Pfa_CPI (95% CI) | AUC (95% CI) | Doppler RMSE (Hz) | joint success |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        analysis_lines.append(
            f"| {row['scenario_id']} | {row['method']} | {row['snr_db_input']:.0f}/{row['cnr_db_requested']:.0f} | "
            f"{row['pd']:.3f} {row['pd_ci95']} | {row['pfa_cpi']:.3f} {row['pfa_cpi_ci95']} | "
            f"{row['roc_auc']:.3f} {row['roc_auc_ci95']} | {row['doppler_rmse_hz_detected']} | {row['joint_success']:.3f} |"
        )
    analysis_lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 阈值由独立 H0 calibration scenes 按 Pfa_CPI=0.05 设定；开发 H0 未用于重校准。256 场景开发集不支持 1e-6 级 CUT 虚警结论。",
            "- A-D 共享配对背景/噪声；D 的相同流形目标只能用于源归因不可辨识边界分析。",
            "- SNR 输入采用 D0 连续 range-PSF 峰值定义；CNR/SCNR 同时从完整 CPI component mean 独立计算。",
            "- B2 是显式几何相干前端；不能被描述为纯学习基线。B3 仅在所注明 Tx 参考假设下有效。",
            "- 下一步阶段审查需要结合测试、bootstrap 区间、A-D 场景及物理参数未知项决定 GO / CONDITIONAL GO / STOP / REDIRECT；表中 AUC 点估计不能单独替代阶段门。",
        ]
    )
    (output_dir / "analysis.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    print(f"Wrote D2-A results under {output_dir}; elapsed={elapsed:.1f}s; B3 eligible={b3_eligible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
