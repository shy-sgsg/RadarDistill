#!/usr/bin/env python3
"""Validate the committed D1-E002 protocol and all registered seed streams."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.training.d1_e002 import (  # noqa: E402
    assert_preregistered_files,
    prior_registered_seed_values,
    validate_e002_config,
    validate_e002_seed_disjointness,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "D1-E002" / "config.yaml",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_e002_config(config)
    seeds = validate_e002_seed_disjointness(
        config,
        existing_seed_values=prior_registered_seed_values(ROOT),
    )
    commit = assert_preregistered_files(ROOT, config_path)
    print(
        json.dumps(
            {
                "experiment_id": "D1-E002",
                "status": "VALID",
                "code_commit": commit,
                "registered_seed_values": len(seeds),
                "overlap_with_prior_experiments": 0,
                "candidate_order": config["training"]["candidate_order"],
                "blind_data_generated": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
