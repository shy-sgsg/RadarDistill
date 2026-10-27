#!/usr/bin/env python3
"""Run the reviewed D2-A-R5-C teaching-content pilot in explicit stages."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_d2a_r5 as r5_runner
from radardistill.students.task_student import TaskStudent, export_inference_model
from radardistill.training.d2a_r5c_teaching_pilot import (
    FROZEN_INPUTS,
    _require_training_authorization,
    _require_training_cuda,
    _require_training_preflight,
    assert_split_seed_disjoint,
    build_initial_state,
    build_split_seed_specs,
    run_overfit_sanity,
    train_arms_sequentially,
    validate_r5c_teaching_config,
)

CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-C-teaching/config.yaml"
OUTPUT_ROOT = ROOT / "experiments/D2-A-R5/R5-C-teaching"
R5_CONFIG_PATH = ROOT / "experiments/D2-A-R5/R5-A/config.yaml"
PHYSICS_CONFIG_PATH = ROOT / "experiments/D2-A-E001/config.yaml"


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _require_committed_main() -> str:
    if _git("branch", "--show-current") != "main":
        raise RuntimeError("R5-C teaching requires the user-authorized main branch")
    head = _git("rev-parse", "HEAD")
    if head != _git("rev-parse", "origin/main"):
        raise RuntimeError("R5-C teaching requires reviewed code/config pushed to origin/main")
    for relative in FROZEN_INPUTS:
        _git("ls-files", "--error-unmatch", relative)
        check = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", relative], cwd=ROOT)
        if check.returncode != 0:
            raise RuntimeError(f"R5-C teaching frozen input has local changes: {relative}")
    return head


def _require_committed_config(config: dict[str, Any]) -> None:
    if config != _read_yaml(CONFIG_PATH):
        raise RuntimeError("R5-C teaching config must match the committed config on main")


def _existing_seed_collisions(specs: list[dict[str, Any]]) -> list[tuple[str, int, str]]:
    from run_d2a_r5c_teacher_audit import _existing_scene_seeds

    lower = min(int(spec["seed_block_start"]) for spec in specs)
    upper = max(int(spec["seed_block_end_exclusive"]) for spec in specs)
    return _existing_scene_seeds(lower, upper)


def _project_size_bytes() -> int:
    result = subprocess.run(
        ["du", "-sb", str(ROOT)], check=True, text=True, capture_output=True
    )
    return int(result.stdout.split()[0])


def _model_parameter_counts() -> tuple[int, int]:
    training_model = TaskStudent(width=96, spectrum_frequency_bins=520)
    training_count = sum(parameter.numel() for parameter in training_model.parameters())
    inference_model = export_inference_model(training_model)
    inference_count = sum(parameter.numel() for parameter in inference_model.parameters())
    del inference_model, training_model
    return training_count, inference_count


def run_preflight(config: dict[str, Any]) -> dict[str, Any]:
    """Check review/freeze, seed isolation, data inputs, CUDA and disk; generate nothing."""
    _require_training_authorization(config)
    head = _require_committed_main()
    _require_committed_config(config)
    specs = build_split_seed_specs(config)
    collisions = _existing_seed_collisions(specs)
    assert_split_seed_disjoint(specs, [seed for _, seed, _ in collisions])

    device = _require_training_cuda("cuda:0")
    free_vram, total_vram = torch.cuda.mem_get_info(device)
    minimum_free_vram = int(config["resources"]["minimum_free_vram_bytes"])
    if free_vram < minimum_free_vram:
        raise RuntimeError(f"R5-C teaching needs {minimum_free_vram} free VRAM bytes; found {free_vram}")
    free_disk = shutil.disk_usage(ROOT).free
    minimum_free_disk = int(config["resources"]["minimum_free_disk_bytes"])
    if free_disk < minimum_free_disk:
        raise RuntimeError(f"R5-C teaching needs {minimum_free_disk} free disk bytes; found {free_disk}")
    project_size = _project_size_bytes()

    r5_config = _read_yaml(R5_CONFIG_PATH)
    cache_root = ROOT / r5_config["data"]["cache_root"]
    train_arrays = r5_runner._load_split(cache_root, "train")
    validation_arrays = r5_runner._load_split(cache_root, "validation")
    if len(train_arrays["base_scene_id"]) != int(config["data"]["training_pairs"]):
        raise RuntimeError("R5-C teaching frozen training cache does not have 51,200 pairs")
    if len(validation_arrays["base_scene_id"]) != int(config["data"]["validation_pairs"]):
        raise RuntimeError("R5-C teaching frozen validation cache does not have 5,000 pairs")
    with np.load(ROOT / config["data"]["training_source"], allow_pickle=False) as archive:
        pair_plan = archive["plan"]
        pair_scene_ids = archive["scene_ids"].astype(str)
        seed_base = int(archive["seed_base"])
    if seed_base != int(config["data"]["pair_plan_seed_base"]):
        raise RuntimeError("R5-C teaching shared training pair plan seed differs from config")
    if not np.array_equal(pair_scene_ids, np.asarray(train_arrays["base_scene_id"]).astype(str)):
        raise RuntimeError("R5-C teaching pair plan IDs differ from the frozen R5-A training cache")
    if pair_plan.shape != (12800, 32):
        raise RuntimeError("R5-C teaching pair plan does not contain 12,800 fixed updates")

    initial_state = build_initial_state(int(config["training"]["initialization_seed"]))
    training_parameters, inference_parameters = _model_parameter_counts()
    if training_parameters != int(config["model"]["training_parameter_count_expected"]):
        raise RuntimeError(
            f"R5-C teaching training parameter count differs: {training_parameters}"
        )
    if inference_parameters != int(config["model"]["inference_parameter_count_expected"]):
        raise RuntimeError(
            f"R5-C teaching inference parameter count differs: {inference_parameters}"
        )
    environment = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "git_sha": head,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_free_bytes_before": int(free_vram),
        "cuda_total_bytes": int(total_vram),
        "disk_free_bytes": int(free_disk),
        "project_size_bytes_before": int(project_size),
        "training_pair_plan_shape": list(pair_plan.shape),
        "training_pair_plan_seed_base": seed_base,
        "training_parameter_count": training_parameters,
        "inference_parameter_count": inference_parameters,
        "initial_state_tensor_count": len(initial_state),
        "new_evaluation_chunks": len(specs),
        "new_evaluation_scenes": sum(int(spec["scene_count"]) for spec in specs),
        "config": config,
        "seed_collisions": 0,
        "final_blind_generated": False,
        "passed": True,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "preflight.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return environment


def _require_preflight_complete(
    config: dict[str, Any], *, require_g1: bool = False
) -> dict[str, Any]:
    head = _require_committed_main()
    _require_committed_config(config)
    path = OUTPUT_ROOT / "preflight.json"
    if not path.is_file():
        raise RuntimeError("R5-C teaching preflight must pass before overfit or training")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("git_sha") != head
        or record.get("passed") is not True
        or record.get("config") != config
    ):
        raise RuntimeError("R5-C teaching preflight is missing, failed, or from another commit")
    if require_g1:
        _require_training_preflight(config, require_g1=True)
    return record


def _load_training_inputs(config: dict[str, Any]):
    _require_training_authorization(config)
    r5_config = _read_yaml(R5_CONFIG_PATH)
    cache_root = ROOT / r5_config["data"]["cache_root"]
    train_arrays = r5_runner._load_split(cache_root, "train")
    validation_arrays = r5_runner._load_split(cache_root, "validation")
    with np.load(ROOT / config["data"]["training_source"], allow_pickle=False) as archive:
        pair_plan = archive["plan"].copy()
        pair_scene_ids = archive["scene_ids"].astype(str)
        seed_base = int(archive["seed_base"])
    if seed_base != int(config["data"]["pair_plan_seed_base"]):
        raise RuntimeError("R5-C teaching training pair-plan seed does not match config")
    if not np.array_equal(pair_scene_ids, np.asarray(train_arrays["base_scene_id"]).astype(str)):
        raise RuntimeError("R5-C teaching training pair-plan IDs differ from the R5-A cache")
    return train_arrays, validation_arrays, pair_plan, _read_yaml(PHYSICS_CONFIG_PATH)


def run_overfit(config: dict[str, Any]) -> dict[str, Any]:
    _require_preflight_complete(config)
    train_arrays, _, _, physics_config = _load_training_inputs(config)
    initial_state = build_initial_state(int(config["training"]["initialization_seed"]))
    initial_microbatch = int(config["training"]["initial_microbatch_size"])
    microbatches = [
        initial_microbatch,
        *[
            int(value)
            for value in config["training"]["oom_microbatch_fallbacks"]
            if int(value) < initial_microbatch
        ],
    ]
    attempts: list[int] = []
    for microbatch in microbatches:
        attempts.append(microbatch)
        try:
            result = run_overfit_sanity(
                initial_state=initial_state,
                train_arrays=train_arrays,
                config=config,
                device="cuda:0",
                physics_config=physics_config,
                microbatch=microbatch,
            )
            result["microbatch_attempts"] = attempts
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if microbatch == microbatches[-1]:
                raise
    result["git_sha"] = _require_committed_main()
    result["config"] = config
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "g1_overfit_sanity.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def run_training(config: dict[str, Any]) -> list[dict[str, Any]]:
    _require_preflight_complete(config, require_g1=True)
    sanity_path = OUTPUT_ROOT / "g1_overfit_sanity.json"
    if not sanity_path.is_file() or json.loads(sanity_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("R5-C teaching G1 overfit gate must pass all arms before training")
    train_arrays, validation_arrays, pair_plan, physics_config = _load_training_inputs(config)
    initial_state = build_initial_state(int(config["training"]["initialization_seed"]))
    results = train_arms_sequentially(
        initial_state=initial_state,
        train_arrays=train_arrays,
        validation_arrays=validation_arrays,
        pair_plan=pair_plan,
        config=config,
        device="cuda:0",
        microbatch=int(config["training"]["initial_microbatch_size"]),
        arm_root=OUTPUT_ROOT / "runs" / f"seed_{config['training']['initialization_seed']}",
        physics_config=physics_config,
    )
    (OUTPUT_ROOT / "training.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("preflight", "overfit", "train", "evaluate", "analyze")
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args(argv)
    config = _read_yaml(args.config)
    validate_r5c_teaching_config(config)

    if args.phase == "preflight":
        result = run_preflight(config)
    elif args.phase == "overfit":
        result = run_overfit(config)
    elif args.phase == "train":
        result = run_training(config)
    else:
        _require_training_authorization(config)
        raise RuntimeError(
            f"R5-C teaching {args.phase} phase is implemented in the evaluation task, not yet available"
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
