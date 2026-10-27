#!/usr/bin/env python3
"""Create compact R5-A learning-curve and task-summary figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "experiments/D2-A-R5/R5-A"


def _read_validation_history(arm: str) -> dict[str, list[float]]:
    path = ARTIFACTS / "runs/seed_101" / arm / "training_history.csv"
    points: dict[str, list[float]] = {"update": []}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if not row["validation_task_loss"]:
                continue
            points["update"].append(float(row["update"]))
            for key in (
                "validation_task_loss",
                "validation_auc",
                "validation_range_rmse_cells",
                "validation_doppler_rmse_hz",
            ):
                points.setdefault(key, []).append(float(row[key]))
    if not points["update"]:
        raise ValueError(f"no validation records in {path}")
    return points


def plot_learning_curves(output_dir: Path) -> Path:
    fields = (
        ("validation_task_loss", "Validation task loss", "lower is better"),
        ("validation_auc", "Validation AUC", "higher is better"),
        ("validation_range_rmse_cells", "Validation range RMSE", "range cells"),
        ("validation_doppler_rmse_hz", "Validation Doppler RMSE", "Hz"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    colors = {"S0": "#1769aa", "S2": "#d1495b"}
    for ax, (field, title, ylabel) in zip(axes.flat, fields, strict=True):
        for arm in ("S0", "S2"):
            history = _read_validation_history(arm)
            ax.plot(history["update"], history[field], label=arm, color=colors[arm], linewidth=1.8)
        ax.axvline(6400, color="#555555", linestyle="--", linewidth=1, label="extension point")
        ax.set_title(title)
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=False)
    fig.suptitle("D2-A-R5-A: seed 101 validation learning curves", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    output = output_dir / "training_curves.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_development_by_course(output_dir: Path) -> Path:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    courses = ("C0", "C1", "C2", "C3")
    fields = (
        ("joint_success", "Joint success", "rate"),
        ("pd", "Probability of detection", "rate"),
        ("range_rmse_all_h1_cells", "Range RMSE, all H1", "range cells"),
        ("doppler_rmse_all_h1_hz", "Doppler RMSE, all H1", "Hz"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2))
    positions = list(range(len(courses)))
    width = 0.36
    colors = {"S0": "#1769aa", "S2": "#d1495b"}
    for ax, (field, title, ylabel) in zip(axes.flat, fields, strict=True):
        for index, arm in enumerate(("S0", "S2")):
            values = [
                metrics["arms"][arm]["development"]["by_course"][course][field]
                for course in courses
            ]
            offsets = [position + (index - 0.5) * width for position in positions]
            ax.bar(offsets, values, width=width, label=arm, color=colors[arm])
        ax.set_xticks(positions, courses)
        ax.set_title(title)
        ax.set_xlabel("Course level")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    fig.suptitle("D2-A-R5-A: reused development set (not blind)", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    output = output_dir / "development_by_course.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_c3_rmse_by_snr(output_dir: Path) -> Path:
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    scenarios = ("separated", "near_clutter_edge", "partial_overlap")
    fields = (
        ("range_rmse_all_h1_cells", "Range RMSE", "range cells"),
        ("doppler_rmse_all_h1_hz", "Doppler RMSE", "Hz"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    colors = {"S0": "#1769aa", "S2": "#d1495b"}
    for row_index, (field, title, ylabel) in enumerate(fields):
        for column_index, scenario in enumerate(scenarios):
            ax = axes[row_index, column_index]
            for arm in ("S0", "S2"):
                grouped = metrics["arms"][arm]["development"]["by_scenario_snr"]
                values = [
                    grouped[f"C3/{scenario}/SNR{snr}/CNR20"][field]
                    for snr in (0, 10)
                ]
                ax.plot((0, 10), values, marker="o", color=colors[arm], label=arm, linewidth=1.8)
            ax.set_title(f"C3 {scenario.replace('_', ' ')}")
            ax.set_xlabel("SNR (dB), CNR fixed at 20 dB")
            ax.set_ylabel(ylabel if column_index == 0 else "")
            ax.grid(alpha=0.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, frameon=False)
    fig.suptitle("D2-A-R5-A: C3 development RMSE by SNR (two points)", y=1.035)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    output = output_dir / "c3_rmse_by_snr.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for output in (
        plot_learning_curves(args.output_dir),
        plot_development_by_course(args.output_dir),
        plot_c3_rmse_by_snr(args.output_dir),
    ):
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
