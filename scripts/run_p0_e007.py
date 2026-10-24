#!/usr/bin/env python3
"""Run the preregistered P0-E007 model screen, formal training and evaluation."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.p0_e007 import (  # noqa: E402
    run_final,
    run_formal_e1,
    run_formal_kd,
    run_screen,
    write_analysis_report,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("screen", "formal-e1", "formal-kd", "final", "summarize", "all"),
        default="screen",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "P0-E007" / "config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "P0-E007",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    config = copy.deepcopy(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise SystemExit("--batch-size must be positive")
        config["training"]["batch_size"] = args.batch_size
    args.output.mkdir(parents=True, exist_ok=True)
    effective_config = args.output / "config_used.yaml"
    encoded_config = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    if effective_config.exists() and effective_config.read_text(encoding="utf-8") != encoded_config:
        if args.phase != "summarize":
            raise SystemExit(
                "effective config differs from the existing run; preserve this run and use a new output directory"
            )
    else:
        effective_config.write_text(encoded_config, encoding="utf-8")

    def screen() -> None:
        if not (args.output / "screening.json").exists():
            result = run_screen(config, ROOT, args.output)
            print(f"Screen selection: {result['selection']['selected']}", flush=True)

    def formal_e1() -> None:
        if not (args.output / "formal_e1_metrics.json").exists():
            result = run_formal_e1(config, ROOT, args.output)
            print(
                "Formal E1 KD-eligible models: "
                f"{result.get('kd_selection', {}).get('selected', [])}",
                flush=True,
            )

    def formal_kd() -> None:
        if not (args.output / "formal_kd_metrics.json").exists():
            result = run_formal_kd(config, ROOT, args.output)
            print(f"Formal KD status: {result['status']}", flush=True)

    def final() -> None:
        if not (args.output / "final_metrics.json").exists():
            result = run_final(config, ROOT, args.output)
            print(f"Final status: {result['status']}", flush=True)

    if args.phase == "screen":
        screen()
    elif args.phase == "formal-e1":
        formal_e1()
    elif args.phase == "formal-kd":
        formal_kd()
    elif args.phase == "final":
        final()
    elif args.phase == "summarize":
        write_analysis_report(config, args.output)
    else:
        screen()
        formal_e1()
        formal_kd()
        final()
        write_analysis_report(config, args.output)

    if args.phase != "summarize":
        write_analysis_report(config, args.output)
    print(f"Artifacts: {args.output}", flush=True)


if __name__ == "__main__":
    main()
