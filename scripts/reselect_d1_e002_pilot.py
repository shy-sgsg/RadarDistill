#!/usr/bin/env python3
"""Recompute D1-E002 pilot eligibility from saved validation metrics only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.training.d1_e002 import reselect_d1_e002_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E002" / "config.yaml",
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    output = reselect_d1_e002_pilot(args.root, args.config)
    print(f"D1-E002 corrected pilot selection: {output}")


if __name__ == "__main__":
    main()
