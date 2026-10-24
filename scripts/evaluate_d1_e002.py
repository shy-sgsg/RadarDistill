#!/usr/bin/env python3
"""Freeze D1-E002 checkpoints, calibrate thresholds, or run final blind evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d1_e002 import (  # noqa: E402
    freeze_d1_e002_models,
    run_d1_e002_blind,
    run_d1_e002_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "calibration", "blind"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E002" / "config.yaml",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.phase == "freeze":
        result = freeze_d1_e002_models(args.root, args.config)
    elif args.phase == "calibration":
        result = run_d1_e002_calibration(args.root, args.config, batch_size=args.batch_size)
    else:
        result = run_d1_e002_blind(args.root, args.config, batch_size=args.batch_size)
    summary = {"experiment_id": "D1-E002", "phase": args.phase, "output": str(result)}
    if args.phase == "blind":
        metrics = json.loads((Path(result) / "metrics.json").read_text(encoding="utf-8"))
        summary["stage_decision"] = metrics["stage_decision"]
        summary["gates"] = {name: item["passed"] for name, item in metrics["gates"].items()}
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
