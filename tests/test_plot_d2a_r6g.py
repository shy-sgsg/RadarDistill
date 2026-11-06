from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from runpy import run_path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/plot_d2a_r6g.py"
ARMS = ("T0", "T1", "T3")
CHECKPOINTS = (0, 12_800, 25_600, 51_200)


def _r6g_metrics_fixture(runs_dir: Path) -> dict[str, object]:
    trajectories: dict[str, object] = {}
    for arm_index, arm in enumerate(ARMS):
        history_dir = runs_dir / arm
        history_dir.mkdir(parents=True)
        with (history_dir / "training_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("update", "validation_truth_total")
            )
            writer.writeheader()
            writer.writerows(
                {
                    "update": update,
                    "validation_truth_total": 3.0 - arm_index - update / 51_200,
                }
                for update in range(0, 51_201, 400)
            )

        task_points = {
            str(update): {
                "pd": 0.50 + checkpoint_index * 0.10 + arm_index * 0.01,
                "joint_success_all_H1": 0.20 + checkpoint_index * 0.10 + arm_index * 0.01,
                "range_rmse_all_H1_cells": 2.0 - checkpoint_index * 0.1,
                "doppler_rmse_all_H1_hz": 50.0 - checkpoint_index * 10.0,
            }
            for checkpoint_index, update in enumerate(CHECKPOINTS)
        }
        trajectories[arm] = {
            "updates": list(range(0, 51_201, 400)),
            "validation_truth_loss_points": {
                str(update): {"validation_truth_total": 3.0 - arm_index - update / 51_200}
                for update in CHECKPOINTS
            },
            "task_metrics_at_registered_updates": task_points,
        }
    return {
        "status": "development_evaluation_complete",
        "experiment": "D2-A-R6-G",
        "validation_trajectories": trajectories,
    }


def test_r6g_plot_data_uses_truth_validation_loss_and_registered_task_points(tmp_path):
    runs_dir = tmp_path / "runs"
    metrics = _r6g_metrics_fixture(runs_dir)
    module = run_path(str(SCRIPT)) if SCRIPT.is_file() else {}
    build_plot_series = module.get("build_plot_series")

    assert callable(build_plot_series), "R6-G plotting data loader is missing"
    series = build_plot_series(metrics, runs_dir)

    assert series["T1"]["validation_updates"] == list(range(0, 51_201, 400))
    assert series["T1"]["validation_truth_loss"][:3] == pytest.approx(
        [2.0, 1.9921875, 1.984375]
    )
    assert series["T1"]["checkpoint_updates"] == list(CHECKPOINTS)
    assert series["T1"]["pd"] == pytest.approx([0.51, 0.61, 0.71, 0.81])
    assert series["T1"]["joint_success"] == pytest.approx([0.21, 0.31, 0.41, 0.51])
    assert series["T1"]["range_rmse_cells"] == pytest.approx([2.0, 1.9, 1.8, 1.7])
    assert series["T1"]["doppler_rmse_hz"] == [50.0, 40.0, 30.0, 20.0]


def test_r6g_plot_data_rejects_incomplete_registered_task_points(tmp_path):
    runs_dir = tmp_path / "runs"
    metrics = _r6g_metrics_fixture(runs_dir)
    metrics["validation_trajectories"]["T1"]["task_metrics_at_registered_updates"].pop("25600")
    module = run_path(str(SCRIPT))

    with pytest.raises(ValueError, match="registered checkpoint updates"):
        module["build_plot_series"](metrics, runs_dir)


def test_r6g_plot_cli_writes_png_from_completed_metrics_and_histories(tmp_path):
    runs_dir = tmp_path / "runs"
    metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "figures/r6g_curves.png"
    metrics_path.write_text(
        json.dumps(_r6g_metrics_fixture(runs_dir)), encoding="utf-8"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--metrics",
            str(metrics_path),
            "--runs",
            str(runs_dir),
            "--output",
            str(output_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert output_path.stat().st_size > 10_000
