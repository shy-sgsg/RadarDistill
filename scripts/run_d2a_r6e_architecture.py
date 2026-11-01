#!/usr/bin/env python3
"""Run the preregistered D2-A-R6-E matched architecture screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d2a_r6a import calibrate_threshold
from radardistill.evaluation.d2a_r6e import (
    h0_metrics,
    h1_metrics,
    paired_quality_bootstrap,
    physical_diagnostics,
    predict_eval_split,
)
from radardistill.students.r6_architectures import R6E_ARMS, build_r6e_student, count_r6e_parameters
from radardistill.training import d2a_r6e as training


STAGE = ROOT / "experiments/D2-A-R6/R6-E"
PREDICTIONS = STAGE / "predictions"
RUNS = STAGE / "runs/seed_606"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _save_predictions(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def evaluate() -> dict[str, Any]:
    preflight, head = training._require_preflight()
    config, r6a, physics = training._configs()
    g1 = json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-E evaluation requires G1_PASS")
    plans = training._all_split_plans(config, r6a)
    conditions = r6a["radar"]["conditions"]
    device = torch.device("cuda:0")
    microbatch = int(g1["formal_microbatch"])
    prf_hz = float(config["radar"]["prf_hz"])
    completed: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    arm_metrics: dict[str, Any] = {}
    h1_by_arm: dict[str, dict[str, np.ndarray]] = {}
    thresholds: dict[str, float] = {}

    for arm in R6E_ARMS:
        run_dir = RUNS / arm
        record_path = run_dir / "training_record.json"
        checkpoint_path = run_dir / "best_checkpoint.pt"
        if not record_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"R6-E arm {arm} lacks a completed training record/checkpoint")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if record.get("status") != "complete" or int(record.get("updates", -1)) != int(config["training"]["fixed_updates_per_model"]):
            raise RuntimeError(f"R6-E arm {arm} did not complete its frozen update budget")
        completed[arm] = record

        model = build_r6e_student(arm, seed=int(config["data"]["training_seed"])).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("owner") != record.get("owner"):
            raise RuntimeError(f"R6-E {arm} best checkpoint owner mismatch")
        if int(checkpoint.get("update", -1)) != int(record["best_update"]):
            raise RuntimeError(f"R6-E {arm} checkpoint update differs from its training record")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

        calibration = predict_eval_split(
            model, "h0_calibration", plans["h0_calibration"], physics,
            device=device, microbatch=microbatch, prf_hz=prf_hz,
        )
        threshold_info = calibrate_threshold(
            calibration["score"], float(config["evaluation"]["target_pfa_cpi"])
        )
        threshold = float(threshold_info["threshold"])
        thresholds[arm] = threshold
        verification = predict_eval_split(
            model, "h0_verification", plans["h0_verification"], physics,
            device=device, microbatch=microbatch, prf_hz=prf_hz,
        )
        h1 = predict_eval_split(
            model, "h1_development", plans["h1_development"], physics,
            device=device, microbatch=microbatch, prf_hz=prf_hz,
        )
        h1_by_arm[arm] = h1
        h0_summary = h0_metrics(calibration, verification, threshold=threshold, conditions=conditions)
        h1_summary = h1_metrics(
            h1, threshold=threshold, conditions=conditions,
            h0_verification=verification, prf_hz=prf_hz,
        )
        physical = physical_diagnostics(
            model, plans["h1_development"], physics, h1,
            threshold=threshold, device=device, microbatch=microbatch, prf_hz=prf_hz,
            phase_count=int(config["evaluation"]["common_phase_rotation_count"]),
            frequency_shifts_hz=config["evaluation"]["frequency_shift_intervention_hz"],
        )
        _save_predictions(PREDICTIONS / arm / "h0_calibration.npz", calibration)
        _save_predictions(PREDICTIONS / arm / "h0_verification.npz", verification)
        _save_predictions(PREDICTIONS / arm / "h1_development.npz", h1)
        arm_metrics[arm] = {
            "training": record,
            "parameters": count_r6e_parameters(model),
            "best_checkpoint_update": int(checkpoint["update"]),
            "threshold_calibration": threshold_info,
            "h0": h0_summary,
            "h1_development": h1_summary,
            "physical_consistency": physical,
        }
        print(json.dumps({
            "stage": "evaluation_complete", "arm": arm, "threshold": threshold,
            "c2_c3_joint_success": h1_summary["by_group"]["C2_C3"]["joint_success"],
            "overall_pfa": h0_summary["verification"]["pfa"],
        }, ensure_ascii=False), flush=True)
        del model, checkpoint, calibration, verification, h1
        torch.cuda.empty_cache()

    bootstrap = int(config["evaluation"]["paired_bootstrap_replicates"])
    bootstrap_seed = int(config["evaluation"]["paired_bootstrap_seed"])
    baseline = "j1_joint"
    for candidate, key in (("ce_tcn", "ce_tcn_minus_j1_joint"),
                           ("rax_transformer", "rax_transformer_minus_j1_joint")):
        contrasts[key] = paired_quality_bootstrap(
            h1_by_arm[baseline], h1_by_arm[candidate],
            threshold_baseline=thresholds[baseline], threshold_candidate=thresholds[candidate],
            conditions=conditions, replicates=bootstrap, seed=bootstrap_seed,
            confidence=float(config["evaluation"]["candidate_contrast_ci_level"]),
        )

    result = {
        "status": "development_evaluation_complete",
        "experiment": "D2-A-R6-E",
        "git_sha": head,
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "development_only": True,
        "final_blind_generated": False,
        "independent_scene_counts": training._scene_counts(plans),
        "primary_contrasts": contrasts,
        "arms": arm_metrics,
        "gpu": torch.cuda.get_device_name(device),
        "device": str(device),
        "precision": "FP32",
        "microbatch": microbatch,
        "effective_batch_size": int(config["training"]["effective_batch_size_cpi"]),
    }
    _write_json(STAGE / "metrics.json", result)
    return result


def _valid_preflight() -> bool:
    path = STAGE / "preflight.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return (
            record.get("git_sha") == training._git("rev-parse", "HEAD")
            and record.get("protocol_sha256") == training._sha256(STAGE / "protocol.md")
            and record.get("config_sha256") == training._sha256(STAGE / "config.yaml")
            and record.get("source_sha256") == training._source_sha256()
        )
    except (OSError, json.JSONDecodeError, RuntimeError):
        return False


def run_all() -> dict[str, Any]:
    preflight = json.loads((STAGE / "preflight.json").read_text(encoding="utf-8")) if _valid_preflight() else training.r6e_preflight()
    _, head = training._require_preflight()
    g1_path = STAGE / "g1.json"
    g1 = json.loads(g1_path.read_text(encoding="utf-8")) if g1_path.is_file() else {}
    if g1.get("status") != "G1_PASS" or g1.get("git_sha") != head:
        g1 = training.r6e_run_g1()
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-E G1 failed; formal data generation and training were not started")
    for split in ("train", "validation"):
        progress_path = training.CACHE / split / "progress.json"
        if progress_path.is_file():
            arrays = training._load_pair_cache(split)
            print(json.dumps({"stage": "cache_ready", "split": split, "scenes": len(arrays["scene_id"])}), flush=True)
            del arrays
        else:
            print(json.dumps(training.generate_paired_cache(split), ensure_ascii=False), flush=True)
    for arm in R6E_ARMS:
        record_path = RUNS / arm / "training_record.json"
        if record_path.is_file():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") == "complete" and int(record.get("updates", 0)) == int(training._configs()[0]["training"]["fixed_updates_per_model"]):
                print(json.dumps({"stage": "training_already_complete", "arm": arm}), flush=True)
                continue
        print(json.dumps(training.r6e_train_arm(arm), ensure_ascii=False), flush=True)
    return evaluate()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "g1", "generate", "train", "train-all", "evaluate", "all", "cleanup"))
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--arm", choices=R6E_ARMS)
    args = parser.parse_args()
    if args.command == "preflight":
        result = training.r6e_preflight()
    elif args.command == "g1":
        result = training.r6e_run_g1()
    elif args.command == "generate":
        if args.split is None:
            parser.error("generate requires --split")
        result = training.generate_paired_cache(args.split)
    elif args.command == "train":
        if args.arm is None:
            parser.error("train requires --arm")
        result = training.r6e_train_arm(args.arm)
    elif args.command == "train-all":
        result = {arm: training.r6e_train_arm(arm) for arm in R6E_ARMS}
    elif args.command == "evaluate":
        result = evaluate()
    elif args.command == "cleanup":
        result = training.cleanup_owned_training_cache()
    else:
        result = run_all()
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
