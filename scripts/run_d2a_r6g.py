#!/usr/bin/env python3
"""Run the frozen R6-G preflight and owner-scoped data preparation steps."""

from __future__ import annotations

import argparse
import json
from typing import Any

from radardistill.training import d2a_r6g


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="run G0 frozen-source, scene, disk, and CUDA checks")
    commands.add_parser("g1", help="run the physical and CUDA student/teacher smoke test")
    paired = commands.add_parser("prepare-data", help="generate or resume paired train/validation IQ")
    paired.add_argument("--split", choices=("train", "validation"), required=True)
    targets = commands.add_parser("prepare-targets", help="generate or resume CUDA K3/T3 target maps")
    targets.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    if args.command == "preflight":
        result = d2a_r6g.r6g_preflight()
    elif args.command == "g1":
        result = d2a_r6g.r6g_run_g1()
    elif args.command == "prepare-data":
        result = d2a_r6g.generate_r6g_pair_cache(args.split)
    else:
        result = d2a_r6g.generate_r6g_target_cache(batch_size=args.batch_size)
    _print(result)


if __name__ == "__main__":
    main()
