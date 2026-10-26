"""Export compact D2-A-R4 metric tables and a condition-wise RMSE figure."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/D2-A-R4"
FIELDS = [
    "record_type", "method", "condition", "n", "threshold", "pfa_cpi",
    "pfa_upper95", "pd", "auc", "joint_success", "range_rmse_cells",
    "doppler_rmse_hz", "top_cell_localization_rate", "median_observable_power",
    "phase_agreement", "phase_doppler_change_p95_hz", "estimate", "ci95_lower", "ci95_upper",
]


def _row(record_type: str, method: str, condition: str, **values: Any) -> dict[str, Any]:
    return {field: values.get(field) for field in FIELDS} | {
        "record_type": record_type,
        "method": method,
        "condition": condition,
    }


def build_rows(metrics: dict[str, Any], pilot: dict[str, Any], sensitivity: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for section, items in (("teachers", metrics["teachers"]), ("frozen_students", metrics["frozen_students"])):
        for method, result in items.items():
            task = result.get("parameter_and_map", result.get("h1_task", {}))
            row = _row(
                "h1_overall", method, "all four H1 groups", n=result.get("n_h1"),
                threshold=result.get("threshold"), pfa_cpi=result.get("pfa_ge"),
                pfa_upper95=result.get("pfa_ci95_exact_upper"), pd=task.get("pd", result.get("pd")),
                auc=task.get("auc", result.get("auc_vs_h0_verification")),
                joint_success=task.get("joint_success"),
                range_rmse_cells=task.get("range_rmse_cells"),
                doppler_rmse_hz=task.get("doppler_rmse_hz"),
            )
            if section == "teachers":
                row["top_cell_localization_rate"] = task.get("peak_map_alignment", {}).get("top_cell_localization_rate")
            rows.append(row)

            for condition, value in result.get("pfa_by_condition", {}).items():
                rows.append(_row(
                    "h0_condition", method, condition, n=value.get("n"),
                    threshold=result.get("threshold"), pfa_cpi=value.get("pfa"),
                    pfa_upper95=value.get("exact_upper95"),
                ))
            for condition, value in result.get("pfa_by_amplitude_quartile", {}).items():
                rows.append(_row(
                    "h0_amplitude_quartile", method, f"quartile {condition}", n=value.get("n"),
                    threshold=result.get("threshold"), pfa_cpi=value.get("pfa"),
                    pfa_upper95=value.get("exact_upper95"), median_observable_power=value.get("median_observable_power"),
                ))
            for condition, value in task.get("scenario_summary", {}).items():
                rows.append(_row(
                    "h1_condition", method, condition, n=value.get("n"),
                    pd=value.get("pd"), auc=value.get("auc"), joint_success=value.get("joint_success"),
                    range_rmse_cells=value.get("range_rmse_cells"),
                    doppler_rmse_hz=value.get("doppler_rmse_hz"),
                ))

    for method, result in pilot["evaluation"].items():
        task = result["h1_task"]
        phase = sensitivity["pilot_public_phase_rotation"][method]["balanced_sample"]
        rows.append(_row(
            "pilot_overall", method, "independent pilot H1", n=task.get("n"),
            threshold=result.get("threshold"), pfa_cpi=result.get("pfa_ge"),
            pfa_upper95=result.get("pfa_ci95_exact_upper"), pd=result.get("pd"),
            auc=task.get("auc_vs_h0_calibration"), joint_success=task.get("joint_success"),
            range_rmse_cells=task.get("range_rmse_cells"), doppler_rmse_hz=task.get("doppler_rmse_hz"),
            phase_agreement=phase.get("decision_agreement"),
            phase_doppler_change_p95_hz=phase.get("doppler_change_p95_hz"),
        ))
        for condition, value in result.get("pfa_by_condition", {}).items():
            rows.append(_row(
                "pilot_h0_condition", method, condition, n=value.get("n"),
                threshold=result.get("threshold"), pfa_cpi=value.get("pfa"),
                pfa_upper95=value.get("exact_upper95"),
            ))
        for condition, value in task.get("scenario_summary", {}).items():
            rows.append(_row(
                "pilot_h1_condition", method, condition, n=value.get("n"),
                pd=value.get("pd"), doppler_rmse_hz=value.get("doppler_rmse_hz"),
            ))

    for method, values in sensitivity["frozen_students"].items():
        phase = values["public_phase_rotation"]
        rows.append(_row(
            "balanced_phase_sensitivity", method, "64 scenes per H1 group", n=phase.get("n_scenes"),
            phase_agreement=phase.get("decision_agreement"),
            phase_doppler_change_p95_hz=phase.get("doppler_change_p95_hz"),
        ))
    for method, values in sensitivity["pilot_public_phase_rotation"].items():
        for sample, phase in (("initial first 256", values["original_first_256"]), ("balanced 64 per group", values["balanced_sample"])):
            rows.append(_row(
                "pilot_phase_sensitivity", method, sample, n=phase.get("n_scenes"),
                phase_agreement=phase.get("decision_agreement"),
                phase_doppler_change_p95_hz=phase.get("doppler_change_p95_hz"),
            ))
    paired = sensitivity["pilot_s1_minus_s0_paired_task_difference"]
    for metric, value in paired.items():
        if not isinstance(value, dict) or "difference" not in value:
            continue
        lower, upper = value["paired_bootstrap_95_ci"]
        rows.append(_row(
            "pilot_paired_difference", "S1 minus S0", metric, n=2048,
            estimate=value["difference"], ci95_lower=lower, ci95_upper=upper,
        ))
    return rows


def write_figure(metrics: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    methods = ["B3", "T1", "ACE_AMF", "A", "B", "C"]
    colors = {
        "B3": "#365f91", "T1": "#21867a", "ACE_AMF": "#d17a22",
        "A": "#6b7280", "B": "#9c78b5", "C": "#c45a71",
    }
    markers = {"B3": "o", "T1": "s", "ACE_AMF": "^", "A": "D", "B": "v", "C": "P"}
    groups = list(metrics["teachers"]["B3"]["parameter_and_map"]["scenario_summary"])
    centers = np.arange(len(groups), dtype=float)
    offsets = np.linspace(-0.25, 0.25, len(methods))

    fig, ax = plt.subplots(figsize=(10.4, 5.4), constrained_layout=True)
    for offset, method in zip(offsets, methods):
        section = "teachers" if method in metrics["teachers"] else "frozen_students"
        result = metrics[section][method]
        task = result.get("parameter_and_map", result.get("h1_task", {}))
        summaries = task["scenario_summary"]
        values = [summaries[group].get("doppler_rmse_hz") for group in groups]
        ax.scatter(
            centers + offset, values, s=56, color=colors[method], marker=markers[method],
            edgecolors="#263238", linewidths=0.45, label=method, zorder=3,
        )

    ax.set_yscale("log")
    ax.set_ylim(0.03, 220)
    ax.set_xticks(centers)
    ax.set_xticklabels([group.split("/")[0] + "\n" + group.split("/")[2] for group in groups])
    ax.set_ylabel("Doppler RMSE (Hz, log scale)")
    ax.set_xlabel("Frozen H1 condition group")
    ax.set_title("Doppler RMSE across the four frozen H1 conditions")
    ax.grid(axis="y", which="major", color="#d8dde3", linewidth=0.8)
    ax.grid(axis="y", which="minor", color="#edf0f3", linewidth=0.45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=3, frameon=False, loc="upper left")
    fig.text(
        0.01, -0.035,
        "Each point uses n=512 H1 scenes. SNR, CNR, geometry and course vary together across groups; this is not an isolated SNR sweep.",
        fontsize=8.5, color="#4b5563",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    pilot = json.loads((OUT / "pilot.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((OUT / "sampling_sensitivity.json").read_text(encoding="utf-8"))
    rows = build_rows(metrics, pilot, sensitivity)
    with (OUT / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_figure(metrics, OUT / "figures/doppler_rmse_by_condition.png")
    print(f"wrote {len(rows)} rows and {OUT / 'figures/doppler_rmse_by_condition.png'}")


if __name__ == "__main__":
    main()
