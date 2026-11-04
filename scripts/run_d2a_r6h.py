#!/usr/bin/env python
"""Run the gated D2-A-R6-H SSM architecture screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.training.d2a_r6h import (
    R6H_ARMS,
    cleanup_owned_r6h_training_cache,
    generate_r6h_paired_cache,
    r6h_evaluate,
    r6h_preflight,
    r6h_run_g1,
    r6h_train_all,
    r6h_train_arm,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "g1", "generate-train", "generate-validation", "train", "train-all", "evaluate", "cleanup"),
    )
    parser.add_argument("--arm", choices=R6H_ARMS)
    args = parser.parse_args()
    if args.command == "preflight":
        result = r6h_preflight()
    elif args.command == "g1":
        result = r6h_run_g1()
    elif args.command == "generate-train":
        result = generate_r6h_paired_cache("train")
    elif args.command == "generate-validation":
        result = generate_r6h_paired_cache("validation")
    elif args.command == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        result = r6h_train_arm(args.arm)
    elif args.command == "train-all":
        result = r6h_train_all()
    elif args.command == "evaluate":
        result = r6h_evaluate()
    else:
        result = cleanup_owned_r6h_training_cache()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
