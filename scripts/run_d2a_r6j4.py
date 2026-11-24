#!/usr/bin/env python3
"""Run the frozen D2-A-R6-J4 preflight, training and development evaluation."""

from __future__ import annotations

import argparse
import json

from radardistill.evaluation.d2a_r6j4 import r6j4_analyze, r6j4_evaluate, r6j4_g1, r6j4_preflight
from radardistill.training.d2a_r6j4 import generate_r6j4_pair_cache, r6j4_train_all


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "g1", "cache-train", "cache-validation", "train-all", "evaluate", "analyze"),
    )
    args = parser.parse_args()
    handlers = {
        "preflight": r6j4_preflight,
        "g1": r6j4_g1,
        "cache-train": lambda: generate_r6j4_pair_cache("train"),
        "cache-validation": lambda: generate_r6j4_pair_cache("validation"),
        "train-all": r6j4_train_all,
        "evaluate": r6j4_evaluate,
        "analyze": r6j4_analyze,
    }
    result = handlers[args.command]()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
