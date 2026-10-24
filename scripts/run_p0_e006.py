#!/usr/bin/env python3
"""Run the preregistered P0-E006 study on CUDA in the AISAR environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.torch_experiment import run_p0_e006  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "P0-E006" / "config.yaml",
        help="frozen experiment YAML configuration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "P0-E006",
        help="artifact directory",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SystemExit("CUDA is required for P0-E006; no training was started.")
    print(
        f"CUDA preflight: {torch.cuda.get_device_name(0)}; "
        f"PyTorch {torch.__version__}; CUDA build {torch.version.cuda}",
        flush=True,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    result = run_p0_e006(config, args.output, ROOT)
    print(f"Completed {result['experiment_id']} on {result['environment']['gpu']}.")
    print(f"Metrics: {args.output / 'metrics.json'}")
    print(f"Analysis: {args.output / 'analysis.md'}")


if __name__ == "__main__":
    main()
