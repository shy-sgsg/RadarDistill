#!/usr/bin/env python3
"""Render compact figures from the completed R5-A capacity artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = ROOT / "experiments/D2-A-R5/R5-A/capacity"
SMALL_HISTORY = ROOT / "experiments/D2-A-R5/R5-A/runs/seed_101/S0/training_history.csv"
MEDIUM_HISTORY = CAPACITY / "runs/seed_101/Medium/training_history.csv"
COLORS = {"Small": "#3569a8", "Medium": "#d87928"}


def _history(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    metrics = json.loads((CAPACITY / "metrics.json").read_text(encoding="utf-8"))
    training = json.loads((CAPACITY / "training.json").read_text(encoding="utf-8"))
    histories = {"Small": _history(SMALL_HISTORY), "Medium": _history(MEDIUM_HISTORY)}
    output = CAPACITY / "figures"
    output.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    curves = (
        ("validation_task_loss", "Validation truth-task loss"),
        ("validation_range_rmse_cells", "Validation range RMSE (cells)"),
        ("validation_doppler_rmse_hz", "Validation Doppler RMSE (Hz)"),
        ("validation_auc", "Validation AUC"),
    )
    for axis, (field, title) in zip(axes.flat, curves, strict=True):
        for label in ("Small", "Medium"):
            points = [row for row in histories[label] if row.get(field, "") not in ("", "nan")]
            axis.plot(
                [int(float(row["update"])) for row in points],
                [float(row[field]) for row in points],
                color=COLORS[label],
                label=label,
                linewidth=1.7,
            )
            selected = int(training["arms"][label]["selected_update"])
            axis.axvline(selected, color=COLORS[label], linestyle=":", alpha=0.65, linewidth=1)
        axis.set(title=title, xlabel="Optimizer update")
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("D2-A-R5-A capacity: matched 12,800-update validation trajectories")
    fig.tight_layout()
    fig.savefig(output / "capacity_training_curves.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(2)
    for index, label in enumerate(("Small", "Medium")):
        h0 = metrics["arms"][label]["c3_h0_verification"]["overall"]
        point = float(h0["pfa_ge"])
        upper = float(h0["pfa_ci95_exact_upper"])
        axes[0].errorbar(
            index,
            point,
            yerr=[[0.0], [upper - point]],
            fmt="o",
            color=COLORS[label],
            capsize=4,
            label=f"{label}: {point:.3f}, upper95 {upper:.3f}",
        )
    axes[0].axhline(0.05, color="black", linestyle="--", linewidth=1, label="Calibration target 0.05")
    axes[0].axhline(0.10, color="#a23b3b", linestyle=":", linewidth=1.5, label="C3 limit 0.10")
    axes[0].set(xticks=x, xticklabels=("Small", "Medium"), ylabel="C3 H0 false-alarm probability", title="Fixed-threshold C3 H0 verification")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    conditions = (
        "C3/separated/SNR0/CNR20",
        "C3/separated/SNR10/CNR20",
        "C3/near_clutter_edge/SNR0/CNR20",
        "C3/near_clutter_edge/SNR10/CNR20",
        "C3/partial_overlap/SNR0/CNR20",
        "C3/partial_overlap/SNR10/CNR20",
    )
    positions = np.arange(len(conditions))
    for label in ("Small", "Medium"):
        values = [metrics["arms"][label]["development"]["by_scenario_snr"][key]["joint_success"] for key in conditions]
        axes[1].plot(positions, values, marker="o", color=COLORS[label], label=label)
    axes[1].set(
        xticks=positions,
        xticklabels=("sep\n0", "sep\n10", "edge\n0", "edge\n10", "overlap\n0", "overlap\n10"),
        ylim=(0, 1.02),
        ylabel="Joint detection and parameter success",
        title="C3 development task success by scenario/SNR",
    )
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    fig.suptitle("Capacity gain coexists with a shared C3 false-alarm failure")
    fig.tight_layout()
    fig.savefig(output / "c3_capacity_protection.png", dpi=170)
    plt.close(fig)
    print("\n".join(str(path.relative_to(ROOT)) for path in sorted(output.glob("*.png"))))


if __name__ == "__main__":
    main()
