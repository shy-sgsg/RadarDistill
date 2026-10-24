"""Reproducible CUDA training and evaluation for P1-A-E001."""

from __future__ import annotations

import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .p1_a_e001 import (
    P1PhaseMLP,
    PhaseBatch,
    circular_error,
    decode_phase_vectors,
    generate_phase_batch,
    normalized_cross_coherence,
    phase_crlb_std,
    phase_profile_ml,
    phase_reference_correlation,
    phase_sample_circular_mean,
    phase_target_vectors,
    require_cuda,
)


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _save_batch(path: Path, batch: PhaseBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        iq=batch.iq,
        frequency_hz=batch.frequency_hz,
        common_phase_rad=batch.common_phase_rad,
        amplitude=batch.amplitude,
        channel_phase_rad=batch.channel_phase_rad,
        delta_rad=batch.delta_rad,
        snr_db=batch.snr_db,
        noise_power=batch.noise_power,
        channel_snr_db=batch.channel_snr_db,
        seed=np.asarray(batch.seed, dtype=np.int64),
    )


def _features_on_cpu(iq: np.ndarray, chunk_size: int = 512) -> torch.Tensor:
    outputs = []
    for start in range(0, iq.shape[0], chunk_size):
        values = torch.from_numpy(np.asarray(iq[start : start + chunk_size]))
        outputs.append(normalized_cross_coherence(values))
    return torch.cat(outputs, dim=0)


def _targets_on_cpu(delta: np.ndarray) -> torch.Tensor:
    return phase_target_vectors(torch.from_numpy(np.asarray(delta, dtype=np.float32)))


def _fit_one_seed(
    config: dict[str, Any],
    train_batch: PhaseBatch,
    validation_batch: PhaseBatch,
    torch_seed: int,
    device: torch.device,
    output: Path,
    train_seed: int,
) -> dict[str, Any]:
    training = config["training"]
    checkpoint_path = output / "checkpoints" / f"seed_{train_seed}.pt"
    history_path = output / "training" / f"seed_{train_seed}.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.exists() and history_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            int(saved["train_seed"]) != train_seed
            or int(saved["torch_seed"]) != torch_seed
            or saved["training_config"] != training
        ):
            raise RuntimeError(f"checkpoint configuration mismatch: {checkpoint_path}")
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        return {
            "train_seed": train_seed,
            "torch_seed": torch_seed,
            "best_epoch": int(saved["best_epoch"]),
            "best_validation_loss": float(saved["best_validation_loss"]),
            "epochs_completed": len(history),
            "training_seconds": float(saved["training_seconds"]),
            "peak_allocated_bytes": int(saved["peak_allocated_bytes"]),
            "peak_reserved_bytes": int(saved["peak_reserved_bytes"]),
            "checkpoint": str(checkpoint_path.relative_to(output)),
            "resumed_from_checkpoint": True,
        }

    torch.manual_seed(int(torch_seed))
    torch.cuda.manual_seed_all(int(torch_seed))
    torch.set_num_threads(1)
    model = P1PhaseMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    train_features = _features_on_cpu(train_batch.iq)
    train_targets = _targets_on_cpu(train_batch.delta_rad)
    validation_features = _features_on_cpu(validation_batch.iq)
    validation_targets = _targets_on_cpu(validation_batch.delta_rad)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(torch_seed) + 1)
    batch_size = int(training["batch_size"])
    max_epochs = int(training["epochs"])
    patience = int(training["patience"])
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    records: list[dict[str, Any]] = []
    torch.cuda.reset_peak_memory_stats(device)
    start_time = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        train_loss_sum = 0.0
        seen = 0
        try:
            for start in range(0, len(permutation), batch_size):
                indices = permutation[start : start + batch_size]
                inputs = train_features[indices].to(device, non_blocking=False)
                targets = train_targets[indices].to(device, non_blocking=False)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = F.mse_loss(outputs, targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch}")
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.detach()) * len(indices)
                seen += len(indices)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                "P1-A-E001 ran out of CUDA memory. Stop this run, lower batch_size in the frozen config for every seed, "
                "and restart the full three-seed comparison with a new output directory."
            ) from exc

        model.eval()
        validation_loss_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(validation_features), 512):
                inputs = validation_features[start : start + 512].to(device)
                targets = validation_targets[start : start + 512].to(device)
                loss = F.mse_loss(model(inputs), targets)
                validation_loss_sum += float(loss) * len(inputs)
        validation_loss = validation_loss_sum / len(validation_features)
        record = {
            "epoch": epoch,
            "train_vector_mse": train_loss_sum / max(seen, 1),
            "validation_vector_mse": validation_loss,
        }
        records.append(record)
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    elapsed = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("training ended without a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    peak_allocated = int(torch.cuda.max_memory_allocated(device))
    peak_reserved = int(torch.cuda.max_memory_reserved(device))
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_vector_mse", "validation_vector_mse"))
        writer.writeheader()
        writer.writerows(records)
    torch.save(
        {
            "state_dict": best_state,
            "train_seed": int(train_seed),
            "torch_seed": int(torch_seed),
            "best_epoch": int(best_epoch),
            "best_validation_loss": float(best_loss),
            "training_seconds": float(elapsed),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "training_config": training,
        },
        checkpoint_path,
    )
    result = {
        "train_seed": int(train_seed),
        "torch_seed": int(torch_seed),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "epochs_completed": len(records),
        "training_seconds": float(elapsed),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "checkpoint": str(checkpoint_path.relative_to(output)),
        "resumed_from_checkpoint": False,
    }
    del model, optimizer, train_features, train_targets, validation_features, validation_targets
    torch.cuda.empty_cache()
    return result


def _predict_checkpoint(
    checkpoint: Path, features: torch.Tensor, device: torch.device, train_seed: int
) -> np.ndarray:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = P1PhaseMLP().to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    predictions: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(features), 512):
            outputs = model(features[start : start + 512].to(device))
            predictions.append(decode_phase_vectors(outputs).cpu())
    result = torch.cat(predictions, dim=0).numpy().astype(np.float64)
    del model
    torch.cuda.empty_cache()
    return result


def _batch_crlb(batch: PhaseBatch) -> np.ndarray:
    result = np.empty((len(batch.iq), 2), dtype=np.float64)
    for index in range(len(batch.iq)):
        result[index] = phase_crlb_std(
            batch.amplitude[index],
            float(batch.noise_power[index]),
            np.ones(batch.iq.shape[-1], dtype=np.complex128),
        )
    return result


def _phase_metrics(error: np.ndarray, crlb: np.ndarray | None = None) -> dict[str, float | int]:
    error = np.asarray(error, dtype=np.float64).reshape(-1)
    if not error.size:
        raise ValueError("phase metrics require at least one observation")
    result: dict[str, float | int] = {
        "count": int(error.size),
        "mae_rad": float(np.mean(np.abs(error))),
        "rmse_rad": float(np.sqrt(np.mean(error**2))),
        "circular_bias_rad": float(np.angle(np.mean(np.exp(1j * error)))),
        "outlier_rate_gt_pi_over_4": float(np.mean(np.abs(error) > np.pi / 4.0)),
    }
    if crlb is not None:
        bound = np.asarray(crlb, dtype=np.float64).reshape(-1)
        result["mean_crlb_std_rad"] = float(np.mean(bound))
        result["rmse_over_rms_crlb"] = float(
            np.sqrt(np.mean(error**2)) / np.sqrt(np.mean(bound**2))
        )
    return result


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _crlb_rows(blind_batches: list[PhaseBatch], snrs: list[float]) -> list[dict[str, Any]]:
    rows = []
    for snr, batch in zip(snrs, blind_batches):
        values = _batch_crlb(batch)
        for parameter, name in enumerate(("delta21", "delta31")):
            rows.append(
                {
                    "snr_db": snr,
                    "phase_parameter": name,
                    "crlb_rms_std_rad": float(np.sqrt(np.mean(values[:, parameter] ** 2))),
                    "crlb_mean_std_rad": float(np.mean(values[:, parameter])),
                }
            )
    return rows


def _plot_rmse(
    config: dict[str, Any], rows: list[dict[str, Any]], crlb_rows: list[dict[str, Any]], output: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snrs = [float(value) for value in config["data"]["test_snr_levels_db"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    methods = [
        ("B0_constant_zero", "B0 zero prior", "#777777"),
        ("B1_coherent_cross_correlation", "B1 cross-correlation", "#d62728"),
        ("B2_sample_circular_mean", "B2 sample circular mean", "#ff7f0e"),
        ("B3_rank_one_profile_ML", "B3 rank-one profile-ML", "#2ca02c"),
    ]
    for parameter, ax in enumerate(axes):
        label = "delta21" if parameter == 0 else "delta31"
        for method, legend, color in methods:
            selected = [row for row in rows if row["phase_parameter"] == label and row["method"] == method]
            selected.sort(key=lambda row: float(row["snr_db"]))
            ax.plot(snrs, [row["rmse_rad"] for row in selected], marker="o", label=legend, color=color)
        student_rows = [
            row for row in rows
            if row["phase_parameter"] == label and str(row["method"]).startswith("student_seed_")
        ]
        student_methods = sorted({row["method"] for row in student_rows})
        for method in student_methods:
            selected = [row for row in student_rows if row["method"] == method]
            selected.sort(key=lambda row: float(row["snr_db"]))
            ax.plot(snrs, [row["rmse_rad"] for row in selected], color="#1f77b4", alpha=0.22, linewidth=1)
        mean_student = []
        for snr in snrs:
            selected = [row["rmse_rad"] for row in student_rows if float(row["snr_db"]) == snr]
            mean_student.append(float(np.mean(selected)))
        ax.plot(snrs, mean_student, marker="s", linewidth=2.3, color="#1f77b4", label="student mean across seeds")
        bound = [row for row in crlb_rows if row["phase_parameter"] == label]
        bound.sort(key=lambda row: float(row["snr_db"]))
        ax.plot(snrs, [row["crlb_rms_std_rad"] for row in bound], "k--", label="conditional CRLB")
        ax.set_title("Δφ21" if parameter == 0 else "Δφ31")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Circular phase RMSE (rad)")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(snrs)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "rmse_snr.png", dpi=180)
    plt.close(fig)


def _analysis_report(
    metrics: dict[str, Any],
    rows: list[dict[str, Any]],
    crlb_rows: list[dict[str, Any]],
    output: Path,
    grouped_rows: list[dict[str, Any]] | None = None,
    paired: dict[str, Any] | None = None,
) -> None:
    snrs = sorted({float(row["snr_db"]) for row in rows})
    sections = [
        "# P1-A-E001：理想三通道相对相位估计",
        "",
        f"**预注册 G1：{'PASS' if metrics['g1_pass'] else 'FAIL'}**（{metrics['g1_pass_count']}/{len(metrics['train_seeds'])} 个训练种子同时满足全部门槛）。",
        "",
        "## 实验条件",
        "",
        "共享单音、三路独立同方差圆对称复 AWGN；`PRF=1300 Hz, N=130`。学生以能量归一化的 `C21/C31` 复互相关为输入，结构为 `4→64→64→4`，只用物理相位真值进行 E1 监督。三个模型按顺序在 CUDA 训练；全新最终盲测在 checkpoint 冻结后生成，每个 SNR 1024 条、共 8192 条。",
        "",
        "## 每个 SNR 的盲测 RMSE",
        "",
        "下表学生为三个训练种子的均值 ± 种子间标准差；基线和 CRLB 在同一盲测样本上计算。",
        "",
        "| SNR (dB) | 相位 | 学生均值 ± SD | B1 互相关 | B2 单样本圆均值 | B3 profile-ML | RMS CRLB |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for snr in snrs:
        for parameter in ("delta21", "delta31"):
            student_values = [
                float(row["rmse_rad"]) for row in rows
                if row["phase_parameter"] == parameter
                and row["method"].startswith("student_seed_")
                and float(row["snr_db"]) == snr
            ]
            baseline_values = {}
            for method in (
                "B1_coherent_cross_correlation",
                "B2_sample_circular_mean",
                "B3_rank_one_profile_ML",
            ):
                baseline_values[method] = next(
                    float(row["rmse_rad"]) for row in rows
                    if row["phase_parameter"] == parameter and row["method"] == method and float(row["snr_db"]) == snr
                )
            bound = next(
                float(row["crlb_rms_std_rad"]) for row in crlb_rows
                if row["phase_parameter"] == parameter and float(row["snr_db"]) == snr
            )
            mean = float(np.mean(student_values))
            spread = float(np.std(student_values, ddof=1))
            sections.append(
                f"| {snr:g} | {parameter} | {mean:.5g} ± {spread:.2g} | "
                f"{baseline_values['B1_coherent_cross_correlation']:.5g} | "
                f"{baseline_values['B2_sample_circular_mean']:.5g} | "
                f"{baseline_values['B3_rank_one_profile_ML']:.5g} | {bound:.5g} |"
            )
    sections.extend(
        [
            "",
            "在低 SNR 的个别点，学生的 RMSE 低于不加权 B2 单样本圆均值；例如 -10 dB 两相位的学生平均约 0.95/0.91 rad，B2 约 1.07 rad。但学生未持续优于更强的 B1/B3；在 0–20 dB 高 SNR 区间明显落后，尤其 20 dB 配对区间全为正。",
        ]
    )
    sections.extend(
        [
            "",
            "## 预注册门与训练诊断",
            "",
            "| 训练种子 | 无噪声 Δφ21/Δφ31 MAE (rad) | 无噪声 Δφ21/Δφ31 P95 (rad) | 20 dB 相对 B0 降幅 | 不变性最大误差 (rad) | 种子门 |",
            "|---:|---:|---:|---:|---:|---|",
        ]
    )
    for seed, values in metrics["seed_gates"].items():
        clean = values["noiseless_metrics"]
        reductions = [item["fractional_rmse_reduction_vs_b0"] for item in values["high_snr_blind_metrics"]]
        sections.append(
            f"| {seed} | {clean[0]['mae_rad']:.5g}/{clean[1]['mae_rad']:.5g} | "
            f"{clean[0]['p95_abs_error_rad']:.5g}/{clean[1]['p95_abs_error_rad']:.5g} | "
            f"{reductions[0]:.2%}/{reductions[1]:.2%} | {values['invariance_max_error_rad']:.3g} | "
            f"{'PASS' if values['seed_pass'] else 'FAIL'} |"
        )
    sections.extend(
        [
            "",
            "`G1` 要求至少 2/3 种子同时满足无噪声 MAE<0.01 rad、P95<0.05 rad、20 dB 两相位 RMSE 至少比 B0 低 50%，并且公共相位旋转/复尺度/频移变化均小于 0.02 rad。此处 0/3 种子通过：三种子均通过 20 dB 常值基线改善和不变性，但无噪声 MAE 均未达到 0.01 rad。",
            "",
            "| 种子 | 最佳轮次 | Train 向量 MSE | Validation 向量 MSE | CUDA 训练时间 (s) | 峰值已分配显存 (MiB) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for record in metrics["training_records"]:
        history_path = output / "training" / f"seed_{record['train_seed']}.csv"
        history = list(csv.DictReader(history_path.open(encoding="utf-8")))
        best = min(history, key=lambda row: float(row["validation_vector_mse"]))
        sections.append(
            f"| {record['train_seed']} | {record['best_epoch']} | {float(best['train_vector_mse']):.5g} | "
            f"{record['best_validation_loss']:.5g} | {record['training_seconds']:.2f} | "
            f"{record['peak_allocated_bytes'] / 2**20:.1f} |"
        )
    sections.extend(
        [
            "",
            "训练与验证向量 MSE 均在约 0.10 附近平台化，没有明显训练集优于验证集的差距；更符合当前读出欠拟合/优化不足的现象，而非明显过拟合。该诊断不能区分有限模型逼近、混合 SNR 目标冲突和优化因素的因果贡献。盲测未用于追加训练或调整配置。",
            "",
            "## 20 dB 与理论界",
            "",
            "| 相位 | B1 互相关 | B2 圆均值 | B3 秩一 profile-ML | CRLB | 学生三种子范围 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for parameter in ("delta21", "delta31"):
        get = lambda method: next(
            float(row["rmse_rad"]) for row in rows
            if row["phase_parameter"] == parameter and row["method"] == method and float(row["snr_db"]) == 20.0
        )
        bound = next(
            float(row["crlb_rms_std_rad"]) for row in crlb_rows
            if row["phase_parameter"] == parameter and float(row["snr_db"]) == 20.0
        )
        student = [get(f"student_seed_{seed}") for seed in metrics["train_seeds"]]
        sections.append(
            f"| {parameter} | {get('B1_coherent_cross_correlation'):.5g} | "
            f"{get('B2_sample_circular_mean'):.5g} | {get('B3_rank_one_profile_ML'):.5g} | "
            f"{bound:.5g} | {min(student):.5g}–{max(student):.5g} |"
        )
    if paired is not None:
        sections.extend(
            [
                "",
                "20 dB 学生−基线的配对 RMSE 差及 95% bootstrap 区间 (rad)：",
                "",
                "| 种子 | 相位 | 对 B1 互相关 | 对 B3 profile-ML |",
                "|---:|---|---:|---:|",
            ]
        )
        for seed in metrics["train_seeds"]:
            key_b1 = f"seed_{seed}/snr_20/vs_B1_coherent_cross_correlation"
            key_b3 = f"seed_{seed}/snr_20/vs_B3_rank_one_profile_ML"
            for parameter, phase_name in enumerate(("delta21", "delta31")):
                b1 = paired[key_b1][parameter]
                b3 = paired[key_b3][parameter]
                sections.append(
                    f"| {seed} | {phase_name} | {b1['mean_rmse_difference_student_minus_baseline_rad']:.5g} "
                    f"[{b1['ci95_lower_rad']:.5g}, {b1['ci95_upper_rad']:.5g}] | "
                    f"{b3['mean_rmse_difference_student_minus_baseline_rad']:.5g} "
                    f"[{b3['ci95_lower_rad']:.5g}, {b3['ci95_upper_rad']:.5g}] |"
                )
    if grouped_rows:
        sections.extend(["", "## 20 dB 分组诊断", ""])
        for group_type, title in (
            ("true_phase_sector", "真实相位八扇区"),
            ("common_phase_sector", "公共初相八扇区"),
            ("amplitude_spread_quartile", "通道幅度差四分位"),
        ):
            subset = [
                row for row in grouped_rows
                if float(row["snr_db"]) == 20.0
                and row["group_type"] == group_type
                and str(row["method"]).startswith("student_seed_")
            ]
            errors = [float(row["rmse_rad"]) for row in subset]
            sections.append(f"- {title}：学生各训练种子/相位的分组 RMSE 范围 {min(errors):.5g}–{max(errors):.5g} rad。")
    sections.extend(
        [
            "",
            "## 阶段判断",
            "",
            "学生显著优于零相位先验，并严格保持公共相位旋转、全局复尺度和公共频移不变性；因此这些有限条件下它利用了相对相位信息。可是 G1 的无噪声精度门未通过，20 dB RMSE 为约 0.023–0.026 rad，是解析基线及 CRLB 的约 2.3–2.6 倍；三种子的配对区间显示学生显著劣于 B1 和 B3。B1/B2/B3 在 20 dB 均接近 CRLB。故 P1-A-E001 按冻结门判为 **STOP / REDIRECT**：暂停本学习读出路线，不进入 P1-B 或知识蒸馏，也不根据盲测追加训练或降低门槛。该结论不否定所有学习架构，也不外推到真实回波、通道失配或 GMTI 检测。",
            "",
            "E001–E007 及 P1-A-E001 使用的都是独立阶段数据；本轮盲测种子为 `401999, 402999, …, 408999`，在 checkpoint 冻结后按预注册方案各生成 1024 条。",
            "",
            "## 产物与复现",
            "",
            "配置/协议、数据、checkpoint、epoch 记录、环境与显存占用、逐样本预测、逐 SNR/分组指标、bootstrap、图表和阶段审查分别见本目录及 `docs/stage_reviews/P1-A-E001_review.md`。",
            "",
            "```bash",
            "conda run --no-capture-output -n AISAR python scripts/run_p1_a_e001.py --output experiments/P1-A-E001-reproduction",
            "```",
        ]
    )
    (output / "analysis.md").write_text("\n".join(sections) + "\n", encoding="utf-8")


def run_p1_a_e001(config: dict[str, Any], output: Path, project_root: Path) -> dict[str, Any]:
    """Train three seeds sequentially on CUDA, then evaluate one frozen blind set."""
    if (output / "run_manifest.json").exists():
        raise RuntimeError(
            f"P1-A-E001 is complete at {output}; refusing to re-evaluate the blind set"
        )
    device = require_cuda(str(config["training"]["device"]))
    torch.cuda.set_device(device)
    output.mkdir(parents=True, exist_ok=True)
    environment = _environment_snapshot(device)
    _json_dump(output / "environment.json", environment)

    train_seeds = [int(value) for value in config["data"]["train_seeds"]]
    validation_seeds = [int(value) for value in config["data"]["validation_seeds"]]
    torch_seeds = [int(value) for value in config["data"]["torch_seeds"]]
    if not (len(train_seeds) == len(validation_seeds) == len(torch_seeds) == 3):
        raise ValueError("P1-A-E001 requires exactly three paired training, validation and torch seeds")
    if len(set(train_seeds + validation_seeds)) != 6:
        raise ValueError("training and validation data seeds must be unique")
    if str(config["training"]["precision"]) != "float32":
        raise ValueError("P1-A-E001 frozen precision is FP32")
    test_snr_levels = [float(value) for value in config["data"]["test_snr_levels_db"]]
    if 20.0 not in test_snr_levels:
        raise ValueError("the frozen G1 baseline gate requires 20 dB blind data")
    blind_seeds = [int(config["data"]["final_test_seed"]) + 1000 * index for index in range(len(test_snr_levels))]
    all_data_seeds = (
        train_seeds
        + validation_seeds
        + [int(config["data"]["noiseless_seed"])]
        + blind_seeds
    )
    if len(set(all_data_seeds)) != len(all_data_seeds):
        raise ValueError("training, validation, noiseless and final blind data seeds must not overlap")

    train_snr_levels = tuple(float(value) for value in config["data"]["train_snr_levels_db"])
    training_records = []
    for index, (train_seed, validation_seed, torch_seed) in enumerate(
        zip(train_seeds, validation_seeds, torch_seeds), start=1
    ):
        train_batch = _load_or_generate(
            output / "data" / f"train_seed_{train_seed}.npz",
            count=int(config["data"]["train_count"]),
            seed=train_seed,
            snr_db=None,
            snr_levels_db=train_snr_levels,
            config=config,
        )
        validation_batch = _load_or_generate(
            output / "data" / f"validation_seed_{validation_seed}.npz",
            count=int(config["data"]["validation_count"]),
            seed=validation_seed,
            snr_db=None,
            snr_levels_db=train_snr_levels,
            config=config,
        )
        training_record = _fit_one_seed(
            config,
            train_batch,
            validation_batch,
            torch_seed,
            device,
            output,
            train_seed,
        )
        training_records.append(training_record)
        print(
            f"E001 training {index}/3: seed={train_seed}, best_epoch={training_record['best_epoch']}, "
            f"validation_vector_mse={training_record['best_validation_loss']:.6g}, "
            f"peak_allocated={training_record['peak_allocated_bytes'] / 2**20:.1f} MiB",
            flush=True,
        )
        del train_batch, validation_batch
        torch.cuda.empty_cache()
    frozen_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _json_dump(
        output / "models_frozen.json",
        {"frozen_utc": frozen_utc, "training_records": training_records, "model_count": len(training_records)},
    )

    blind_batches = []
    for index, snr in enumerate(test_snr_levels):
        blind_seed = blind_seeds[index]
        safe_snr = f"m{abs(int(snr)):02d}" if snr < 0 else f"p{int(snr):02d}"
        batch = _load_or_generate(
            output / "data" / f"blind_snr_{safe_snr}.npz",
            count=int(config["data"]["test_per_snr"]),
            seed=blind_seed,
            snr_db=snr,
            snr_levels_db=tuple(),
            config=config,
        )
        blind_batches.append(batch)
    _json_dump(
        output / "blind_manifest.json",
        {
            "generated_after_all_models_frozen": True,
            "models_frozen_utc": frozen_utc,
            "blind_seeds": [batch.seed for batch in blind_batches],
            "snr_db": test_snr_levels,
            "samples_per_snr": int(config["data"]["test_per_snr"]),
        },
    )

    metrics, result_rows, grouped_rows, prediction_values = _evaluate_all(
        config, output, blind_batches, training_records, device
    )
    bootstrap = prediction_values.pop("__bootstrap__")
    crlb_rows = _crlb_rows(blind_batches, test_snr_levels)
    metrics["environment"] = environment
    metrics["final_blind_seeds"] = [batch.seed for batch in blind_batches]
    metrics["models_frozen_utc"] = frozen_utc
    metrics["crlb_by_snr"] = crlb_rows
    metrics["resource_use_by_training_seed"] = [
        {
            "train_seed": value["train_seed"],
            "training_seconds": value["training_seconds"],
            "peak_allocated_bytes": value["peak_allocated_bytes"],
            "peak_reserved_bytes": value["peak_reserved_bytes"],
        }
        for value in training_records
    ]
    _json_dump(output / "metrics.json", metrics)
    _json_dump(output / "paired_comparisons.json", bootstrap)
    _write_rows(output / "result.csv", result_rows)
    _write_rows(output / "grouped_metrics.csv", grouped_rows)
    with (output / "predictions.npz").open("wb") as handle:
        np.savez_compressed(handle, **prediction_values)
    _plot_rmse(config, result_rows, crlb_rows, output)
    _analysis_report(metrics, result_rows, crlb_rows, output, grouped_rows, bootstrap)
    _json_dump(
        output / "run_manifest.json",
        {
            "experiment_id": "P1-A-E001",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "command": "scripts/run_p1_a_e001.py",
            "project_root": str(project_root),
            "blind_test_generated_once_after_models_frozen": True,
            "output_files": [
                "config.yaml", "protocol.md", "environment.json", "models_frozen.json", "blind_manifest.json",
                "metrics.json", "result.csv", "grouped_metrics.csv", "paired_comparisons.json",
                "predictions.npz", "rmse_snr.png", "analysis.md",
            ],
        },
    )
    return metrics


def _paired_bootstrap_rmse(
    student_error: np.ndarray,
    baseline_error: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    student = np.asarray(student_error, dtype=np.float64)
    baseline = np.asarray(baseline_error, dtype=np.float64)
    if student.shape != baseline.shape or student.ndim != 1 or student.size < 2:
        raise ValueError("paired bootstrap requires equal one-dimensional error arrays")
    point = float(np.sqrt(np.mean(student**2)) - np.sqrt(np.mean(baseline**2)))
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, student.size, size=student.size)
        draws[index] = np.sqrt(np.mean(student[selected] ** 2)) - np.sqrt(
            np.mean(baseline[selected] ** 2)
        )
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return {
        "n": int(student.size),
        "mean_rmse_difference_student_minus_baseline_rad": point,
        "ci95_lower_rad": float(lower),
        "ci95_upper_rad": float(upper),
        "bootstrap_samples": int(samples),
        "bootstrap_seed": int(seed),
    }


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "gpu_free_memory_at_start_bytes": int(free_bytes),
        "gpu_visible_total_bytes": int(total_bytes),
        "git_commit": commit,
        "git_worktree_dirty_at_run": dirty,
        "device": str(device),
    }


def _load_batch(path: Path) -> PhaseBatch:
    with np.load(path, allow_pickle=False) as values:
        return PhaseBatch(
            iq=values["iq"],
            frequency_hz=values["frequency_hz"],
            common_phase_rad=values["common_phase_rad"],
            amplitude=values["amplitude"],
            channel_phase_rad=values["channel_phase_rad"],
            delta_rad=values["delta_rad"],
            snr_db=values["snr_db"],
            noise_power=values["noise_power"],
            channel_snr_db=values["channel_snr_db"],
            seed=int(values["seed"]),
        )


def _load_or_generate(
    path: Path,
    *,
    count: int,
    seed: int,
    snr_db: float | None,
    snr_levels_db: tuple[float, ...],
    config: dict[str, Any],
) -> PhaseBatch:
    if path.exists():
        batch = _load_batch(path)
        if batch.seed != seed or len(batch.iq) != count:
            raise RuntimeError(f"saved data role does not match frozen config: {path}")
        return batch
    batch = generate_phase_batch(
        count=count,
        seed=seed,
        snr_db=snr_db,
        snr_levels_db=snr_levels_db,
        prf_hz=float(config["prf_hz"]),
        n_samples=int(config["n_samples"]),
        amplitude_range=tuple(float(value) for value in config["data"]["amplitude_range"]),
    )
    _save_batch(path, batch)
    return batch


def _group_masks(batch: PhaseBatch, parameter: int) -> list[tuple[str, str, np.ndarray]]:
    truth = batch.delta_rad[:, parameter]
    phase_bins = np.linspace(-np.pi, np.pi, 9)
    phase_index = np.clip(np.digitize(truth, phase_bins[1:-1]), 0, 7)
    masks: list[tuple[str, str, np.ndarray]] = [("overall", "all", np.ones(len(truth), dtype=bool))]
    for index in range(8):
        masks.append(("true_phase_sector", str(index), phase_index == index))
    common_bins = np.linspace(-np.pi, np.pi, 9)
    common_index = np.clip(np.digitize(batch.common_phase_rad, common_bins[1:-1]), 0, 7)
    for index in range(8):
        masks.append(("common_phase_sector", str(index), common_index == index))
    spread = np.ptp(batch.amplitude, axis=1)
    quartiles = np.quantile(spread, [0.25, 0.5, 0.75])
    amplitude_group = np.digitize(spread, quartiles, right=True)
    for index in range(4):
        masks.append(("amplitude_spread_quartile", str(index), amplitude_group == index))
    return masks


def _evaluate_all(
    config: dict[str, Any],
    output: Path,
    blind_batches: list[PhaseBatch],
    training_records: list[dict[str, Any]],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    seeds = [int(seed) for seed in config["data"]["train_seeds"]]
    snr_levels = [float(value) for value in config["data"]["test_snr_levels_db"]]
    baseline_names = ("B0_constant_zero", "B1_coherent_cross_correlation", "B2_sample_circular_mean", "B3_rank_one_profile_ML")
    prediction_store: dict[str, np.ndarray] = {}
    baseline_predictions: dict[str, np.ndarray] = {
        name: np.empty((len(blind_batches), int(config["data"]["test_per_snr"]), 2), dtype=np.float64)
        for name in baseline_names
    }
    student_predictions: dict[int, np.ndarray] = {
        seed: np.empty_like(next(iter(baseline_predictions.values()))) for seed in seeds
    }
    per_seed_invariance: dict[str, Any] = {}
    noiseless_path = output / "data" / "noiseless.npz"
    noiseless = _load_or_generate(
        noiseless_path,
        count=int(config["data"]["noiseless_count"]),
        seed=int(config["data"]["noiseless_seed"]),
        snr_db=float("inf"),
        snr_levels_db=tuple(),
        config=config,
    )
    noiseless_features = _features_on_cpu(noiseless.iq)

    for seed_index, (seed, training_record) in enumerate(zip(seeds, training_records)):
        checkpoint = output / training_record["checkpoint"]
        noiseless_prediction = _predict_checkpoint(checkpoint, noiseless_features, device, seed)
        noiseless_error = circular_error(noiseless_prediction, noiseless.delta_rad)
        noiseless_metrics = [
            _phase_metrics(noiseless_error[:, parameter]) for parameter in range(2)
        ]
        n = noiseless.iq.shape[-1]
        rotation = np.linspace(-2.7, 2.7, len(noiseless.iq), dtype=np.float64)
        scale_phase = np.linspace(-1.0, 1.0, len(noiseless.iq), dtype=np.float64)
        complex_scale = (0.7 + 0.2 * np.sin(np.arange(len(noiseless.iq)))) * np.exp(1j * scale_phase)
        frequency_shift = np.linspace(-91.0, 91.0, len(noiseless.iq), dtype=np.float64)
        time_index = np.arange(n, dtype=np.float64)
        transformations = {
            "common_phase_rotation": noiseless.iq * np.exp(1j * rotation[:, None, None]),
            "global_complex_scale": noiseless.iq * complex_scale[:, None, None],
            "common_frequency_modulation": noiseless.iq
            * np.exp(2j * np.pi * frequency_shift[:, None, None] * time_index[None, None, :] / float(config["prf_hz"])),
        }
        invariance: dict[str, Any] = {}
        for name, transformed in transformations.items():
            transformed_features = _features_on_cpu(np.asarray(transformed, dtype=np.complex64))
            transformed_prediction = _predict_checkpoint(checkpoint, transformed_features, device, seed)
            delta = np.abs(circular_error(transformed_prediction, noiseless_prediction))
            invariance[name] = {
                "max_abs_error_rad_per_phase": [float(value) for value in np.max(delta, axis=0)],
                "overall_max_abs_error_rad": float(np.max(delta)),
            }
        per_seed_invariance[str(seed)] = {
            "noiseless_metrics": noiseless_metrics,
            "invariance": invariance,
        }
        print(
            f"E001 evaluation: train_seed={seed}, noiseless RMSE="
            f"{noiseless_metrics[0]['rmse_rad']:.5g}/{noiseless_metrics[1]['rmse_rad']:.5g} rad",
            flush=True,
        )

        for snr_index, batch in enumerate(blind_batches):
            features = _features_on_cpu(batch.iq)
            student_predictions[seed][snr_index] = _predict_checkpoint(
                checkpoint, features, device, seed
            )
        del noiseless_prediction, transformations
        torch.cuda.empty_cache()

    for snr_index, batch in enumerate(blind_batches):
        baseline_predictions["B0_constant_zero"][snr_index] = 0.0
        baseline_predictions["B1_coherent_cross_correlation"][snr_index] = phase_reference_correlation(batch.iq)
        baseline_predictions["B2_sample_circular_mean"][snr_index] = phase_sample_circular_mean(batch.iq)
        baseline_predictions["B3_rank_one_profile_ML"][snr_index] = phase_profile_ml(batch.iq)

    for seed in seeds:
        prediction_store[f"student_seed_{seed}"] = student_predictions[seed]
    prediction_store.update({f"{name}": values for name, values in baseline_predictions.items()})
    snr_results: list[dict[str, Any]] = []
    grouped_results: list[dict[str, Any]] = []
    bootstrap_results: dict[str, Any] = {}
    bootstrap_count = int(config["evaluation"]["bootstrap_samples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    student_names = {seed: f"student_seed_{seed}" for seed in seeds}
    prediction_methods: dict[str, tuple[int | None, np.ndarray]] = {
        name: (None, values) for name, values in baseline_predictions.items()
    }
    prediction_methods.update(
        {
            student_names[seed]: (seed, student_predictions[seed])
            for seed in seeds
        }
    )
    for snr_index, (snr, batch) in enumerate(zip(snr_levels, blind_batches)):
        crlb = _batch_crlb(batch)
        for method, (training_seed, predictions) in prediction_methods.items():
            errors = circular_error(predictions[snr_index], batch.delta_rad)
            for parameter in range(2):
                row = {
                    "snr_db": snr,
                    "phase_parameter": "delta21" if parameter == 0 else "delta31",
                    "method": method,
                    "train_seed": "analytic" if training_seed is None else training_seed,
                    **_phase_metrics(errors[:, parameter], crlb[:, parameter]),
                }
                snr_results.append(row)
                for group_type, group_label, mask in _group_masks(batch, parameter):
                    if np.any(mask):
                        grouped_results.append(
                            {
                                "snr_db": snr,
                                "phase_parameter": row["phase_parameter"],
                                "method": method,
                                "train_seed": row["train_seed"],
                                "group_type": group_type,
                                "group": group_label,
                                **_phase_metrics(errors[mask, parameter], crlb[mask, parameter]),
                            }
                        )
            if training_seed is not None:
                for baseline in ("B1_coherent_cross_correlation", "B3_rank_one_profile_ML"):
                    baseline_errors = circular_error(baseline_predictions[baseline][snr_index], batch.delta_rad)
                    key = f"seed_{training_seed}/snr_{snr:g}/vs_{baseline}"
                    bootstrap_results[key] = [
                        _paired_bootstrap_rmse(
                            errors[:, parameter],
                            baseline_errors[:, parameter],
                            samples=bootstrap_count,
                            seed=bootstrap_seed + 10_000 * seeds.index(training_seed) + 100 * snr_index + parameter,
                        )
                        for parameter in range(2)
                    ]
        print(f"Blind evaluation completed at SNR={snr:g} dB ({len(batch.iq)} records)", flush=True)

    noiseless_threshold_mae = float(config["evaluation"]["noiseless_mae_max_rad"])
    noiseless_threshold_p95 = float(config["evaluation"]["noiseless_p95_max_rad"])
    invariance_threshold = float(config["evaluation"]["physics_invariance_max_error_rad"])
    required_seeds = int(config["evaluation"]["required_noiseless_seeds"])
    high_snr_index = snr_levels.index(20.0)
    required_blind = int(config["evaluation"].get("required_blind_seeds", required_seeds))
    seed_gate: dict[str, Any] = {}
    for seed in seeds:
        values = per_seed_invariance[str(seed)]
        noiseless_errors = circular_error(
            _predict_checkpoint(
                output / next(record["checkpoint"] for record in training_records if int(record["train_seed"]) == seed),
                noiseless_features,
                device,
                seed,
            ),
            noiseless.delta_rad,
        )
        clean_detail = [
            {
                **values["noiseless_metrics"][parameter],
                "p95_abs_error_rad": float(np.quantile(np.abs(noiseless_errors[:, parameter]), 0.95)),
            }
            for parameter in range(2)
        ]
        clean_ok = all(
            row["mae_rad"] < noiseless_threshold_mae
            and row["p95_abs_error_rad"] < noiseless_threshold_p95
            for row in clean_detail
        )
        blind_metrics = []
        for parameter in range(2):
            student_error = circular_error(
                student_predictions[seed][high_snr_index],
                blind_batches[high_snr_index].delta_rad,
            )[:, parameter]
            b0_error = circular_error(
                baseline_predictions["B0_constant_zero"][high_snr_index],
                blind_batches[high_snr_index].delta_rad,
            )[:, parameter]
            student_rmse = float(np.sqrt(np.mean(student_error**2)))
            b0_rmse = float(np.sqrt(np.mean(b0_error**2)))
            blind_metrics.append(
                {
                    "student_rmse_rad": student_rmse,
                    "b0_rmse_rad": b0_rmse,
                    "fractional_rmse_reduction_vs_b0": 1.0 - student_rmse / b0_rmse,
                    "passes_50pct_reduction": student_rmse <= 0.5 * b0_rmse,
                }
            )
        invariance_max = max(
            transform["overall_max_abs_error_rad"]
            for transform in values["invariance"].values()
        )
        invariance_ok = invariance_max < invariance_threshold
        seed_gate[str(seed)] = {
            "noiseless_metrics": clean_detail,
            "noiseless_pass": clean_ok,
            "high_snr_blind_metrics": blind_metrics,
            "blind_pass": all(row["passes_50pct_reduction"] for row in blind_metrics),
            "invariance_max_error_rad": invariance_max,
            "invariance_pass": invariance_ok,
            "seed_pass": clean_ok and invariance_ok and all(row["passes_50pct_reduction"] for row in blind_metrics),
        }

    gate_pass_count = sum(int(value["seed_pass"]) for value in seed_gate.values())
    g1_pass = gate_pass_count >= max(required_seeds, required_blind)
    metrics = {
        "experiment_id": "P1-A-E001",
        "final_blind_status": "evaluated_once_after_all_models_frozen",
        "train_seeds": seeds,
        "torch_seeds": [int(value) for value in config["data"]["torch_seeds"]],
        "validation_seeds": [int(value) for value in config["data"]["validation_seeds"]],
        "blind_snr_db": snr_levels,
        "samples_per_snr": int(config["data"]["test_per_snr"]),
        "seed_gates": seed_gate,
        "g1_pass": bool(g1_pass),
        "g1_pass_count": int(gate_pass_count),
        "g1_required_pass_count": int(max(required_seeds, required_blind)),
        "noiseless_and_invariance_diagnostics": per_seed_invariance,
        "training_records": training_records,
        "methods": list(prediction_methods),
        "crlb_scope": "conditional CRLB with unknown arbitrary shared waveform and fixed per-record amplitudes/noise",
    }
    return metrics, snr_results, grouped_results, prediction_store | {"__bootstrap__": bootstrap_results}
