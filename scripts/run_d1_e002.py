#!/usr/bin/env python3
"""Run the frozen D1-E002 pilot or formal CUDA phase."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.training.d1_e002 import run_d1_e002


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("pilot", "formal"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E002" / "config.yaml",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = run_d1_e002(args.root, args.config, args.phase)
    print(f"D1-E002 {args.phase} output: {output}")


if __name__ == "__main__":
    main()
