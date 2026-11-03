#!/usr/bin/env python3
"""Create the frozen R6-E training, condition, and physical-diagnostic figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "j1_joint": "J1-J",
    "ce_tcn": "CE-TCN",
    "rax_transformer": "RAX-Transformer",
}
COLORS = {
    "j1_joint": "#365F91",
    "ce_tcn": "#C58B16",
    "rax_transformer": "#B95E2C",
}


def _condition_label(condition: dict[str, object]) -> str:
    course = str(condition["course"])
    scenario = {
        "separated": "sep",
        "near_clutter_edge": "edge",
        "partial_overlap": "partial",
    }[str(condition["scenario_id"])]
    snr = float(condition["snr_db"])
    return f"{course}-{scenario}\n{snr:g} dB"


def _load_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def plot_training(metrics: dict[str, object], runs_dir: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11), sharex=True, constrained_layout=True)
    for ax, arm in zip(axes, LABELS):
        rows = _load_history(runs_dir / arm / "training_history.csv")
        x = np.asarray([row["update"] for row in rows])
        train = np.asarray([row["interval_train_total"] for row in rows])
        val = np.asarray([row["validation_total"] for row in rows])
        color = COLORS[arm]
        training = metrics["arms"][arm]["training"]
        best = int(training["best_update"])
        ax.plot(x, train, color=color, linewidth=1.6, label="训练损失")
        ax.plot(x, val, color=color, linewidth=2.0, linestyle="--", label="验证损失")
        ax.axvline(best, color="#555555", linewidth=1.0, linestyle=":")
        ax.scatter([best], [min(val)], color=color, marker="D", zorder=4, s=34,
                   label=f"最佳验证点：{best:,} update")
        ax.set_ylabel("任务损失")
        ax.set_title(f"{LABELS[arm]} · {int(training['updates']):,} updates")
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        ax.legend(loc="upper right", frameon=False, ncols=3)
    axes[-1].set_xlabel("优化器更新次数")
    fig.suptitle("D2-A-R6-E 训练与验证曲线（每 400 updates 记录）", fontsize=15)
    fig.savefig(out_dir / "training_curves.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _condition_entries(metrics: dict[str, object], arm: str, split: str) -> list[dict[str, object]]:
    if split == "h0":
        values = metrics["arms"][arm]["h0"]["verification_by_condition"]
        return [values[str(i)] for i in range(len(values))]
    values = metrics["arms"][arm]["h1_development"]["by_condition"]
    return [values[str(i)] for i in range(len(values))]


def plot_h0(metrics: dict[str, object], out_dir: Path) -> None:
    first = _condition_entries(metrics, "j1_joint", "h0")
    labels = [_condition_label(entry["condition"]) for entry in first]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 6.2), constrained_layout=True)
    ax.axvspan(5.5, 11.5, color="#F0E5D8", alpha=0.65, zorder=0)
    for arm in LABELS:
        entries = _condition_entries(metrics, arm, "h0")
        values = np.asarray([entry["pfa"] for entry in entries], dtype=float)
        ax.plot(x, values, marker="o", linewidth=1.8, markersize=5,
                color=COLORS[arm], label=LABELS[arm])
        c3_x = np.asarray([i for i, entry in enumerate(entries)
                           if entry["condition"]["course"] == "C3"], dtype=int)
        upper = np.asarray([entries[i]["pfa_familywise_exact_upper"] for i in c3_x], dtype=float)
        lower = values[c3_x]
        ax.errorbar(c3_x, lower, yerr=np.vstack((np.zeros_like(upper), upper - lower)),
                    fmt="none", ecolor=COLORS[arm], elinewidth=1.1, capsize=3, alpha=0.9)
        ax.scatter(c3_x, upper, marker="s", s=22, color=COLORS[arm], zorder=3)
    ax.axhline(0.10, color="#555555", linewidth=1.2, linestyle="--", label="C3 0.10 资格线")
    ax.axvline(5.5, color="#777777", linewidth=0.9, linestyle=":")
    ax.set_xticks(x, labels, rotation=42, ha="right")
    ax.set_ylim(bottom=0)
    ax.set_ylabel("独立 H0 验证 Pfa_CPI")
    ax.set_title("固定全局 Pfa=0.05 校准阈值后的条件虚警率\nC3 方块为六格 Bonferroni 单侧 95% 上界")
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    ax.legend(frameon=False, ncols=4, loc="upper left")
    fig.savefig(out_dir / "h0_pfa_by_condition.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_h1(metrics: dict[str, object], out_dir: Path) -> None:
    entries = _condition_entries(metrics, "j1_joint", "h1")
    labels = [_condition_label(entry["condition"]) for entry in entries]
    x = np.arange(len(labels), dtype=float)
    metrics_to_plot = [
        ("pd", "检测概率 Pd", "pd_ci95_exact", (0, 1)),
        ("joint_success", "联合成功率", "joint_success_ci95_exact", (0, 1)),
        ("range_rmse", "距离 RMSE（cell）", None, None),
        ("doppler_rmse", "Doppler RMSE（Hz）", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True, sharex=True)
    for ax, (metric, title, ci_key, ylim) in zip(axes.flat, metrics_to_plot):
        for arm_index, arm in enumerate(LABELS):
            arm_entries = _condition_entries(metrics, arm, "h1")
            offset = (arm_index - 1) * 0.075
            vals: list[float] = []
            lows: list[float] = []
            highs: list[float] = []
            for entry in arm_entries:
                if metric == "range_rmse":
                    value = float(entry["all_h1"]["range_rmse_cells"])
                elif metric == "doppler_rmse":
                    value = float(entry["all_h1"]["doppler_rmse_hz"])
                else:
                    value = float(entry[metric])
                vals.append(value)
                if ci_key:
                    lo, hi = entry[ci_key]
                    lows.append(value - float(lo))
                    highs.append(float(hi) - value)
            if ci_key:
                ax.errorbar(x + offset, vals, yerr=np.vstack((lows, highs)), fmt="o-",
                            color=COLORS[arm], linewidth=1.5, markersize=4, capsize=2,
                            label=LABELS[arm])
            else:
                ax.plot(x + offset, vals, marker="o", linewidth=1.6, markersize=4,
                        color=COLORS[arm], label=LABELS[arm])
        ax.set_title(title)
        ax.grid(axis="y", color="#D8D8D8", linewidth=0.7)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xticks(x, labels, rotation=45, ha="right")
    axes[0, 0].legend(frameon=False, ncols=3, loc="lower left")
    fig.suptitle("H1 development：各条件检测与参数估计（每格 n=512）", fontsize=15)
    fig.savefig(out_dir / "h1_metrics_by_condition.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_physical(metrics: dict[str, object], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    arms = list(LABELS)
    phase_x: np.ndarray | None = None
    for arm in arms:
        physical = metrics["arms"][arm]["physical_consistency"]
        phase = physical["common_phase_rotations"]
        phase_x = np.asarray([entry["phase_rad"] for entry in phase], dtype=float)
        agreement = [entry["summary"]["decision_agreement"] for entry in phase]
        doppler_change = [entry["summary"]["doppler_wrapped_change_p95_hz"] for entry in phase]
        score_change = [entry["summary"]["score_absolute_change_p95"] for entry in phase]
        axes[0, 0].plot(phase_x, agreement, marker="o", markersize=3, linewidth=1.5,
                        color=COLORS[arm], label=LABELS[arm])
        axes[0, 1].plot(phase_x, doppler_change, marker="o", markersize=3, linewidth=1.5,
                        color=COLORS[arm], label=LABELS[arm])
        axes[1, 0].plot(phase_x, score_change, marker="o", markersize=3, linewidth=1.5,
                        color=COLORS[arm], label=LABELS[arm])
        shifts = physical["frequency_shift_interventions"]
        axes[1, 1].plot([-130, 130], [shifts["-130"]["doppler_shift_response_residual_p95_absolute_hz"],
                                      shifts["130"]["doppler_shift_response_residual_p95_absolute_hz"]],
                        marker="o", linewidth=1.5, color=COLORS[arm], label=LABELS[arm])
    axes[0, 0].set(title="公共相位旋转下决策一致率", ylabel="一致率")
    axes[0, 1].set(title="公共相位旋转下 Doppler 变化 p95", ylabel="绝对变化（Hz）")
    axes[1, 0].set(title="公共相位旋转下分数变化 p95", ylabel="绝对变化")
    axes[1, 1].set(title="频移干预后 Doppler 残差 p95", ylabel="残差绝对值（Hz）", xlabel="注入频移（Hz）")
    for ax in axes.flat[:3]:
        ax.set_xlabel("相位旋转（弧度）")
        ax.set_xticks(np.linspace(0, 2 * np.pi, 5))
    for ax in axes.flat:
        ax.grid(color="#D8D8D8", linewidth=0.7)
    axes[0, 0].set_ylim(0.9, 1.01)
    axes[0, 0].legend(frameon=False, ncols=3, loc="lower left")
    fig.suptitle("物理一致性受控诊断（同一 H1 development scenes）", fontsize=14)
    fig.savefig(out_dir / "physical_consistency_diagnostics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "experiments/D2-A-R6/R6-E")
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    metrics = json.loads((experiment_dir / "metrics.json").read_text(encoding="utf-8"))
    out_dir = experiment_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = experiment_dir / "runs/seed_606"
    plot_training(metrics, runs_dir, out_dir)
    plot_h0(metrics, out_dir)
    plot_h1(metrics, out_dir)
    plot_physical(metrics, out_dir)
    print("Wrote:")
    for path in sorted(out_dir.glob("*.png")):
        print(f"  {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
