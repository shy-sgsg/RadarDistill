#!/usr/bin/env python3
"""Run the frozen D2-A-R6-F full-chain hybrid comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d2a_r6f import r6f_evaluate, r6f_g1, r6f_preflight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "g1", "evaluate", "summarize", "all"))
    args = parser.parse_args()
    if args.command == "preflight":
        result = r6f_preflight()
    elif args.command == "g1":
        result = r6f_g1()
    elif args.command == "evaluate":
        result = r6f_evaluate()
    elif args.command == "summarize":
        result = r6f_evaluate(reuse_predictions=True)
    else:
        preflight = r6f_preflight()
        g1 = r6f_g1()
        result = {
            "preflight": preflight,
            "g1": g1,
            "evaluation": r6f_evaluate(),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
