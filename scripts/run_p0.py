#!/usr/bin/env python3
"""Run the configured P0 experiment from the repository root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.experiment import run_p0  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "P0-E001" / "config.yaml",
        help="experiment YAML configuration",
    )
    parser.add_argument("--output", type=Path, help="artifact directory (defaults to experiments/<experiment_id>)")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output or ROOT / "experiments" / str(config["experiment_id"])
    summary = run_p0(config, output)
    print(f"Completed {summary['experiment_id']} at {output}")
    print(f"Analysis: {output / 'analysis.md'}")


if __name__ == "__main__":
    main()
