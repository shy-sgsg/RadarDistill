#!/usr/bin/env python3
"""Freeze provenance for formal historical checkpoints before R7-D evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/D2-A-R7/R7-D/config.yaml"
OUTPUT = ROOT / "experiments/D2-A-R7/R7-D/checkpoint_inventory.csv"
FIELDS = (
    "method_id",
    "experiment_id",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "checkpoint_update",
    "seed_id",
    "arm",
    "model_name",
    "inference_parameter_count",
    "task_scope",
    "status",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a dictionary")
    state = payload.get("model_state_dict", payload.get("state_dict"))
    if not isinstance(state, dict):
        raise ValueError("checkpoint has no model_state_dict/state_dict")
    return state


def discover_inventory(root: Path = ROOT, config_path: Path = CONFIG) -> list[dict[str, Any]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    patterns = config["checkpoint_inventory"]["include_families"]
    paths: set[Path] = set()
    for pattern in patterns:
        paths.update(path for path in (root / "experiments").glob(pattern) if path.is_file())

    inventory: list[dict[str, Any]] = []
    for path in sorted(paths):
        relative = path.relative_to(root)
        text_path = relative.as_posix()
        if "/pilot/" in text_path or "/attempts/" in text_path:
            continue
        if path.name not in {"best_checkpoint.pt", "best_inference.pt"}:
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = _payload_state(payload)
        params = sum(int(value.numel()) for value in state.values() if torch.is_tensor(value))
        experiment = payload.get("experiment")
        if isinstance(experiment, dict):
            experiment = experiment.get("id", "")
        experiment = experiment or next((part for part in relative.parts if part.startswith("D")), "")
        arm = str(payload.get("arm", payload.get("method", payload.get("model", ""))))
        seed = str(payload.get("seed_id", payload.get("seed", "")))
        if not seed:
            seed = next((part for part in relative.parts if part.startswith("seed_")), "")
        update = payload.get("global_update", payload.get("update", ""))
        task = "joint"
        lowered = f"{text_path} {arm}".lower()
        if "j1_detection" in lowered or "detection_only" in lowered:
            task = "detection_only"
        elif "j1_parameter" in lowered or "parameter_only" in lowered:
            task = "parameter_only"
        if "d1-e002" in lowered and "/s2/" in lowered:
            task = "joint_with_lag_one_feature"
        inventory.append(
            {
                "method_id": text_path.removesuffix(".pt").replace("/", "::"),
                "experiment_id": str(experiment),
                "checkpoint_path": text_path,
                "checkpoint_sha256": _sha256(path),
                "checkpoint_bytes": path.stat().st_size,
                "checkpoint_update": update,
                "seed_id": seed,
                "arm": arm,
                "model_name": str(payload.get("model", payload.get("student", payload.get("model_name", "")))),
                "inference_parameter_count": params,
                "task_scope": task,
                "status": "FROZEN_FOR_R7D",
            }
        )
    return inventory


def write_inventory(rows: list[dict[str, Any]], output: Path = OUTPUT) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output)


def main() -> int:
    rows = discover_inventory()
    if not rows:
        raise RuntimeError("no formal checkpoints matched the frozen R7-D inventory")
    write_inventory(rows)
    print(json.dumps({"checkpoint_count": len(rows), "output": str(OUTPUT.relative_to(ROOT))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
