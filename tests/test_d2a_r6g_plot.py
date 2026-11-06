import csv

from scripts.plot_d2a_r6g import build_plot_series


def test_plot_reads_truth_validation_loss_from_validation_total_when_alias_is_blank(tmp_path):
    updates = [0, 12_800, 25_600, 51_200]
    loss_values = [5.0, 0.5, 0.3, 0.2]
    metrics = {
        "status": "development_evaluation_complete",
        "experiment": "D2-A-R6-G",
        "validation_trajectories": {},
    }

    for arm in ("T0", "T1", "T3"):
        run_dir = tmp_path / arm
        run_dir.mkdir()
        history_path = run_dir / "training_history.csv"
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=["update", "validation_total", "validation_truth_total"],
            )
            writer.writeheader()
            for index, update in enumerate(updates):
                writer.writerow(
                    {
                        "update": update,
                        "validation_total": loss_values[index],
                        "validation_truth_total": loss_values[index] if update == 0 else "",
                    }
                )

        metrics["validation_trajectories"][arm] = {
            "updates": updates,
            "task_metrics_at_registered_updates": {
                str(update): {
                    "pd": 0.8,
                    "joint_success_all_H1": 0.7,
                    "range_rmse_all_H1_cells": 1.0,
                    "doppler_rmse_all_H1_hz": 10.0,
                }
                for update in updates
            },
        }

    series = build_plot_series(metrics, tmp_path)

    assert series["T1"]["validation_truth_loss"] == loss_values
