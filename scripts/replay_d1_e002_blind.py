#!/usr/bin/env python3
"""Complete a failed D1-E002 blind evaluation from saved blind artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d1_e002 import replay_d1_e002_blind  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E002" / "config.yaml",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    output = replay_d1_e002_blind(args.root, args.config, batch_size=args.batch_size)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    summary = {
        "experiment_id": "D1-E002",
        "phase": "blind_replay",
        "output": str(output),
        "stage_decision": metrics["stage_decision"],
        "gates": {name: item["passed"] for name, item in metrics["gates"].items()},
        "blind_data_regenerated": metrics["evaluation_replay"]["blind_data_regenerated"],
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
