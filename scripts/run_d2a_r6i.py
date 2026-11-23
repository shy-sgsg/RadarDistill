#!/usr/bin/env python3
"""Run one frozen D2-A-R6-I preflight, G1, evaluation, or analysis command."""

from __future__ import annotations

import argparse
import json
import sys

from radardistill.evaluation.d2a_r6i import r6i_analyze, r6i_evaluate, r6i_g1, r6i_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "g1", "evaluate", "analyze"))
    args = parser.parse_args()
    commands = {
        "preflight": r6i_preflight,
        "g1": r6i_g1,
        "evaluate": r6i_evaluate,
        "analyze": r6i_analyze,
    }
    try:
        result = commands[args.command]()
    except Exception as exc:
        print(f"R6-I {args.command} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
