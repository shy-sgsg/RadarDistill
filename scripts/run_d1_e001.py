#!/usr/bin/env python3
"""Run the preregistered D1-E001 pilot or formal phase on CUDA."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.training.d1_runner import run_d1_e001  # noqa: E402


def _copy_frozen_inputs(config_path: Path, output: Path) -> dict:
    protocol_path = config_path.with_name("protocol.md")
    output.mkdir(parents=True, exist_ok=True)
    for source in (config_path, protocol_path):
        destination = output / source.name
        if source.resolve() == destination.resolve():
            continue
        if destination.exists():
            if destination.read_bytes() != source.read_bytes():
                raise SystemExit(
                    f"{destination} differs from the frozen D1 input; preserve it and choose a new --output"
                )
        else:
            shutil.copy2(source, destination)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    encoded = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    effective = output / "config_used.yaml"
    if effective.exists() and effective.read_text(encoding="utf-8") != encoded:
        raise SystemExit(
            f"{effective} differs from this run; preserve it and choose a new --output"
        )
    if not effective.exists():
        effective.write_text(encoded, encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E001" / "config.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments" / "D1-E001",
    )
    parser.add_argument("--phase", choices=("pilot", "formal"), default="pilot")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    config = _copy_frozen_inputs(arguments.config.resolve(), arguments.output.resolve())
    metrics = run_d1_e001(
        config,
        arguments.output,
        root=ROOT,
        phase=arguments.phase,
        device=arguments.device,
    )
    print(
        json.dumps(
            {
                "experiment_id": metrics["experiment_id"],
                "phase": metrics["phase"],
                "pilot_gate": metrics.get("pilot_gate"),
                "completed_seed_ids": metrics.get("completed_seed_ids"),
                "output": str(arguments.output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
