#!/usr/bin/env python3
"""Run the frozen P1-A-E001 three-channel relative phase experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.p1_a_e001_runner import run_p1_a_e001  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "P1-A-E001" / "config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "P1-A-E001",
    )
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    arguments.output.mkdir(parents=True, exist_ok=True)
    for source in (arguments.config, arguments.config.with_name("protocol.md")):
        destination = arguments.output / source.name
        if source.resolve() == destination.resolve():
            continue
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise SystemExit(f"{destination} already exists with different content; preserve it and use a new output directory")
        else:
            shutil.copy2(source, destination)
    encoded = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    effective_config = arguments.output / "config_used.yaml"
    if effective_config.exists() and effective_config.read_text(encoding="utf-8") != encoded:
        raise SystemExit(
            "effective config differs from the existing P1-A-E001 run; preserve the run and use a new output directory"
        )
    if not effective_config.exists():
        effective_config.write_text(encoded, encoding="utf-8")

    metrics = run_p1_a_e001(config, arguments.output, ROOT)
    print(
        json.dumps(
            {
                "experiment_id": metrics["experiment_id"],
                "g1_pass": metrics["g1_pass"],
                "g1_seed_pass_count": metrics["g1_pass_count"],
                "artifacts": str(arguments.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
