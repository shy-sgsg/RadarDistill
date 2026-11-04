"""Render the frozen R6-F full-chain development comparison from metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "experiments/D2-A-R6/R6-F"
LABELS = {
    "A": "B3 + FFT4",
    "B": "CE-TCN",
    "C": "CE + profile-ML",
    "D": "B3 + CE",
    "E": "J1-J",
    "F": "CE × fit + profile-ML",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        type=Path,
        default=STAGE / "metrics.json",
        help="R6-F metrics.json path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE / "figures/full_chain_comparison.png",
        help="output PNG path",
    )
    args = parser.parse_args()
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    names = tuple(metrics["systems"])
    costs = metrics["full_chain_cost"]["systems"]

    joint = [metrics["systems"][name]["h1_development"]["by_group"]["C2_C3"]["joint_success"] for name in names]
    c3_pfa = [metrics["systems"][name]["h0"]["c3_pooled"]["pfa"] for name in names]
    c3_upper = [metrics["systems"][name]["h0"]["c3_pooled"]["pfa_ci95_exact_upper"] for name in names]
    latency_ms = [1000.0 * costs[name]["mean_seconds_per_cpi"] for name in names]
    colors = ["#637381" if costs[name]["hardware"]["device"] is None else "#2474a6" for name in names]
    x = np.arange(len(names))
    tick_labels = [f"{name}\n{LABELS[name]}" for name in names]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    axes[0].bar(x, joint, color="#2474a6")
    axes[0].set_title("C2+C3 joint success")
    axes[0].set_ylabel("fraction of H1 scenes")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x, tick_labels, rotation=25, ha="right", fontsize=8)
    for i, value in enumerate(joint):
        axes[0].text(i, value + 0.018, f"{value:.3f}", ha="center", fontsize=8)

    axes[1].bar(x, c3_pfa, color="#d97d32")
    axes[1].errorbar(x, c3_pfa, yerr=[np.zeros(len(x)), np.asarray(c3_upper) - np.asarray(c3_pfa)], fmt="none", ecolor="#303030", capsize=3)
    axes[1].axhline(0.10, color="#a32323", linestyle="--", linewidth=1.2, label="C3 pooled limit 0.10")
    axes[1].set_title("C3 pooled H0 false-alarm rate")
    axes[1].set_ylabel("Pfa per CPI")
    axes[1].set_ylim(0, max(0.22, max(c3_upper) + 0.02))
    axes[1].set_xticks(x, tick_labels, rotation=25, ha="right", fontsize=8)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x, latency_ms, color=colors)
    axes[2].set_title("Complete-chain benchmark")
    axes[2].set_ylabel("mean ms per CPI, batch 16")
    axes[2].set_xticks(x, tick_labels, rotation=25, ha="right", fontsize=8)
    for i, value in enumerate(latency_ms):
        axes[2].text(i, value + max(latency_ms) * 0.025, f"{value:.2f}", ha="center", fontsize=8)
    axes[2].text(0.02, 0.97, "gray: CPU   blue: CUDA", transform=axes[2].transAxes, va="top", fontsize=8)

    fig.suptitle("D2-A-R6-F full-chain development comparison", fontsize=13)
    fig.text(
        0.5,
        -0.02,
        "All six chains fail the C3 six-cell familywise Pfa gate. Timing is a matched pipeline benchmark, not a real-time claim.",
        ha="center",
        fontsize=9,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
