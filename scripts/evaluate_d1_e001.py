#!/usr/bin/env python3
"""Calibrate D1 detection scores or run the frozen final blind evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d1_evaluation import (  # noqa: E402
    run_d1_blind_evaluation,
    run_d1_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E001" / "config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "D1-E001",
    )
    parser.add_argument("--phase", choices=("calibration", "blind"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--replay-existing-predictions",
        action="store_true",
        help="recompute metrics from a complete saved blind run without regenerating samples",
    )
    arguments = parser.parse_args()
    if arguments.replay_existing_predictions and arguments.phase != "blind":
        parser.error("--replay-existing-predictions requires --phase blind")
    config = yaml.safe_load(arguments.config.resolve().read_text(encoding="utf-8"))
    if arguments.phase == "calibration":
        result = run_d1_calibration(
            config,
            root=ROOT,
            output=arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
        )
        summary = {
            "experiment_id": result["experiment_id"],
            "phase": result["phase"],
            "status": result["status"],
            "calibrated_methods": list(result["methods"]),
            "output": str(arguments.output),
        }
    else:
        result = run_d1_blind_evaluation(
            config,
            root=ROOT,
            output=arguments.output,
            device=arguments.device,
            batch_size=arguments.batch_size,
            replay_existing_predictions=arguments.replay_existing_predictions,
        )
        summary = {
            "experiment_id": result["experiment_id"],
            "phase": result["phase"],
            "status": result["status"],
            "stage_decision": result["gates"]["stage_decision"],
            "gates": {
                name: data["passed"]
                for name, data in result["gates"].items()
                if name != "stage_decision"
            },
            "analysis_replay": result.get("analysis_replay"),
            "output": str(arguments.output),
        }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
