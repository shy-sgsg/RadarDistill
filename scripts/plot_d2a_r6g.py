#!/usr/bin/env python3
"""Render the frozen R6-G validation-loss and task-metric trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "experiments/D2-A-R6/R6-G"
ARMS = ("T0", "T1", "T3")
ARM_LABELS = {"T0": "Truth only", "T1": "Truth + K3", "T3": "Truth + truth response"}
ARM_COLORS = {"T0": "#365F91", "T1": "#C58B16", "T3": "#B95E2C"}
CHECKPOINTS = (0, 12_800, 25_600, 51_200)


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"R6-G plot metric {label} must be finite")
    return number


def build_plot_series(
    metrics: Mapping[str, Any], runs_dir: Path
) -> dict[str, dict[str, list[float] | list[int]]]:
    """Load comparable validation loss and registered task snapshots for all arms."""
    if metrics.get("status") != "development_evaluation_complete":
        raise ValueError("R6-G plots require a completed development evaluation")
    if metrics.get("experiment") != "D2-A-R6-G":
        raise ValueError("metrics file is not the registered D2-A-R6-G result")
    trajectories = metrics.get("validation_trajectories")
    if not isinstance(trajectories, Mapping):
        raise ValueError("R6-G metrics lack validation trajectories")

    series: dict[str, dict[str, list[float] | list[int]]] = {}
    for arm in ARMS:
        trajectory = trajectories.get(arm)
        if not isinstance(trajectory, Mapping):
            raise ValueError(f"R6-G metrics lack validation trajectory for {arm}")
        history_path = Path(runs_dir) / arm / "training_history.csv"
        with history_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or "update" not in rows[0] or not (
            {"validation_truth_total", "validation_total"} & rows[0].keys()
        ):
            raise ValueError(f"R6-G {arm} history lacks truth-task validation loss")

        updates = [int(row["update"]) for row in rows]
        losses = []
        for row in rows:
            # The initial row uses the explicit alias; later validation rows log `total`.
            value = row.get("validation_truth_total") or row.get("validation_total")
            losses.append(_finite_float(value, f"{arm}.validation_truth_total"))
        if updates[0] != 0 or updates[-1] != 51_200 or any(
            right <= left for left, right in zip(updates, updates[1:])
        ):
            raise ValueError(f"R6-G {arm} history must increase from update 0 to 51,200")
        if [int(update) for update in trajectory.get("updates", [])] != updates:
            raise ValueError(f"R6-G {arm} metrics and CSV history update lists differ")

        task_points = trajectory.get("task_metrics_at_registered_updates")
        if not isinstance(task_points, Mapping) or tuple(
            sorted(int(key) for key in task_points)
        ) != CHECKPOINTS:
            raise ValueError(f"R6-G {arm} task metrics lack registered checkpoint updates")
        snapshots = [task_points[str(update)] for update in CHECKPOINTS]
        fields = {
            "pd": "pd",
            "joint_success": "joint_success_all_H1",
            "range_rmse_cells": "range_rmse_all_H1_cells",
            "doppler_rmse_hz": "doppler_rmse_all_H1_hz",
        }
        values: dict[str, list[float] | list[int]] = {
            "validation_updates": updates,
            "validation_truth_loss": losses,
            "checkpoint_updates": list(CHECKPOINTS),
        }
        for output_key, metric_key in fields.items():
            values[output_key] = [
                _finite_float(snapshot[metric_key], f"{arm}.{metric_key}")
                for snapshot in snapshots
            ]
        series[arm] = values
    return series


def plot_curves(
    series: Mapping[str, Mapping[str, list[float] | list[int]]], output_path: Path
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(11.5, 15.5), constrained_layout=True)
    panels = (
        ("pd", "Validation detection probability Pd", (0.0, 1.0)),
        ("joint_success", "Validation joint success", (0.0, 1.0)),
        ("range_rmse_cells", "Validation range RMSE (cells)", None),
        ("doppler_rmse_hz", "Validation Doppler RMSE (Hz)", None),
    )

    for arm in ARMS:
        values = series[arm]
        axes[0].plot(
            np.asarray(values["validation_updates"]) / 1_000.0,
            values["validation_truth_loss"],
            color=ARM_COLORS[arm],
            linewidth=1.7,
            label=ARM_LABELS[arm],
        )
    axes[0].set_title("Validation truth-task loss (logged every 400 updates)")
    axes[0].set_ylabel("loss")
    axes[0].legend(frameon=False, ncols=3)

    for axis, (key, title, limits) in zip(axes[1:], panels):
        for arm in ARMS:
            values = series[arm]
            axis.plot(
                np.asarray(values["checkpoint_updates"]) / 1_000.0,
                values[key],
                marker="o",
                linewidth=1.7,
                markersize=4,
                color=ARM_COLORS[arm],
                label=ARM_LABELS[arm],
            )
        axis.set_title(title)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.legend(frameon=False, ncols=3, fontsize=8)

    for axis in axes:
        axis.set_xticks([0.0, 12.8, 25.6, 51.2], ["0", "12.8k", "25.6k", "51.2k"])
        axis.set_xlabel("optimizer updates")
        axis.grid(axis="y", color="#D8D8D8", linewidth=0.7)
    fig.suptitle("D2-A-R6-G validation learning curves", fontsize=14)
    fig.text(
        0.5,
        -0.005,
        "Task metrics are validation checkpoint diagnostics; final H0/H1 development results are reported separately.",
        ha="center",
        fontsize=9,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=STAGE / "metrics.json")
    parser.add_argument(
        "--runs",
        type=Path,
        default=STAGE / "runs/seed_707_v1.0.1",
        help="directory containing the three registered training histories",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE / "figures/training_task_curves.png",
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    plot_curves(build_plot_series(metrics, args.runs), args.output)


if __name__ == "__main__":
    main()
