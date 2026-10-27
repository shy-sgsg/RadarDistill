#!/usr/bin/env python3
"""Run the reviewed D2-A-R5-C teaching-content pilot in explicit stages."""

from __future__ import annotations

import argparse
import csv
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
from radardistill.evaluation.d2a_r5c_teaching_pilot import (
    iter_generated_split_chunks,
    paired_teaching_contrasts,
    predict_task_student_chunk,
    summarize_pilot_arm,
)
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


def _load_evaluation_models(config: dict[str, Any], head: str) -> dict[str, torch.nn.Module]:
    training_path = OUTPUT_ROOT / "training.json"
    if not training_path.is_file():
        raise RuntimeError("R5-C teaching evaluation requires completed four-arm training.json")
    records = json.loads(training_path.read_text(encoding="utf-8"))
    if (
        not isinstance(records, list)
        or len(records) != len(config["training"]["arms"])
        or {item.get("arm") for item in records} != set(config["training"]["arms"])
    ):
        raise RuntimeError("R5-C teaching training record does not contain exactly four arms")

    models: dict[str, torch.nn.Module] = {}
    for record in records:
        arm = str(record["arm"])
        if int(record.get("updates_completed", -1)) != int(config["training"]["updates"]):
            raise RuntimeError(f"R5-C teaching arm {arm} did not complete the frozen update budget")
        checkpoint_path = Path(record["best_checkpoint_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"R5-C teaching arm {arm} checkpoint is missing: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            payload.get("experiment") != "D2-A-R5"
            or payload.get("subphase") != "R5-C-teaching"
            or payload.get("arm") != arm
            or payload.get("source_git_sha") != head
            or payload.get("config") != config
            or int(payload.get("global_update", -1)) != int(record.get("best_update", -2))
        ):
            raise RuntimeError(f"R5-C teaching arm {arm} checkpoint provenance is inconsistent")
        training_model = TaskStudent(
            width=int(config["model"]["width"]),
            spectrum_frequency_bins=int(config["model"]["spectrum_frequency_bins"]),
        )
        training_model.load_state_dict(payload["model_state_dict"], strict=True)
        inference_model = export_inference_model(training_model).eval().cpu()
        parameter_count = sum(parameter.numel() for parameter in inference_model.parameters())
        if parameter_count != int(config["model"]["inference_parameter_count_expected"]):
            raise RuntimeError(f"R5-C teaching arm {arm} inference parameter count changed")
        models[arm] = inference_model
        del training_model, payload
    return models


def _append_arrays(target: dict[str, list[np.ndarray]], values: dict[str, Any]) -> None:
    for name, value in values.items():
        target.setdefault(name, []).append(np.asarray(value))


def _join_arrays(source: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    if not source:
        raise RuntimeError("R5-C teaching evaluation produced an empty split")
    result = {name: np.concatenate(parts, axis=0) for name, parts in source.items()}
    lengths = {values.shape[0] for values in result.values()}
    if len(lengths) != 1:
        raise RuntimeError("R5-C teaching streamed split arrays have inconsistent row counts")
    return result


def _write_prediction_csv(path: Path, rows: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows)
    count = len(rows[fieldnames[0]])
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            writer.writerow(
                {
                    name: (
                        rows[name][index].item()
                        if isinstance(rows[name][index], np.generic)
                        else rows[name][index]
                    )
                    for name in fieldnames
                }
            )
    temporary.replace(path)


def _save_arm_predictions(arm: str, payload: dict[str, Any]) -> None:
    directory = OUTPUT_ROOT / "predictions" / arm
    if directory.exists():
        raise FileExistsError(
            f"R5-C teaching prediction directory already exists; preserve it and inspect before rerun: {directory}"
        )
    _write_prediction_csv(directory / "h0_calibration.csv", payload["calibration"])
    _write_prediction_csv(directory / "h0_verification.csv", payload["verification"])
    _write_prediction_csv(directory / "c3_h0_verification.csv", payload["c3_h0_verification"])
    _write_prediction_csv(directory / "h1_development.csv", payload["development"])


def run_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Stream independent H0/H1 splits and persist per-scene predictions only."""
    _require_training_authorization(config)
    preflight = _require_preflight_complete(config, require_g1=True)
    head = _require_committed_main()
    if (OUTPUT_ROOT / "metrics.json").exists() or (OUTPUT_ROOT / "split_manifest.json").exists():
        raise FileExistsError(
            "R5-C teaching evaluation records already exist; preserve and inspect before rerun"
        )
    for arm in config["training"]["arms"]:
        if (OUTPUT_ROOT / "predictions" / arm).exists():
            raise FileExistsError(
                "R5-C teaching prediction output already exists; preserve and inspect: "
                f"{OUTPUT_ROOT / 'predictions' / arm}"
            )
    models = _load_evaluation_models(config, head)
    device = _require_training_cuda("cuda:0")
    physics_config = _read_yaml(PHYSICS_CONFIG_PATH)
    split_names = (
        "h0_calibration",
        "h0_verification",
        "c3_h0_verification",
        "h1_development",
    )
    per_arm: dict[str, dict[str, dict[str, Any]]] = {
        arm: {
            split: {}
            for split in ("calibration", "verification", "c3_h0_verification", "development")
        }
        for arm in config["training"]["arms"]
    }
    split_scene_ids: dict[str, set[str]] = {split: set() for split in split_names}
    split_manifest: dict[str, list[dict[str, Any]]] = {split: [] for split in split_names}
    cuda_peak_allocated = 0
    cuda_peak_reserved = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for split_name in split_names:
        for chunk in iter_generated_split_chunks(config, physics_config, split_name):
            batch = chunk["batch"]
            metadata = chunk["metadata"]
            split_manifest[split_name].append(chunk["spec"])
            ids = metadata["base_scene_ids"].astype(str)
            if split_scene_ids[split_name].intersection(ids.tolist()):
                raise RuntimeError(f"R5-C teaching split {split_name} generated duplicate scene IDs")
            split_scene_ids[split_name].update(ids.tolist())
            for arm in config["training"]["arms"]:
                model = models[arm].to(device)
                try:
                    if split_name == "h1_development":
                        h1 = predict_task_student_chunk(
                            model,
                            batch.iq_h1,
                            device,
                            truth_range_cell=batch.truth_target_range_cell,
                            prf_hz=float(config["radar"]["prf_hz"]),
                        )
                    else:
                        h0 = predict_task_student_chunk(
                            model,
                            batch.iq_h0,
                            device,
                            prf_hz=float(config["radar"]["prf_hz"]),
                        )
                    torch.cuda.synchronize(device)
                finally:
                    models[arm] = model.cpu()
                cuda_peak_allocated = max(cuda_peak_allocated, int(torch.cuda.max_memory_allocated(device)))
                cuda_peak_reserved = max(cuda_peak_reserved, int(torch.cuda.max_memory_reserved(device)))

                if split_name == "h0_calibration":
                    _append_arrays(
                        per_arm[arm]["calibration"],
                        {"base_scene_ids": ids, "h0_score": h0["score"]},
                    )
                elif split_name == "h0_verification":
                    _append_arrays(
                        per_arm[arm]["verification"],
                        {**metadata, "h0_score": h0["score"]},
                    )
                elif split_name == "c3_h0_verification":
                    _append_arrays(
                        per_arm[arm]["c3_h0_verification"],
                        {**metadata, "h0_score": h0["score"]},
                    )
                else:
                    _append_arrays(
                        per_arm[arm]["development"],
                        {
                            **metadata,
                            "h1_score": h1["score"],
                            "truth_range_cell": batch.truth_target_range_cell,
                            "predicted_range_cell": h1["predicted_range_cell"],
                            "predicted_range_bin": h1["predicted_range_bin"],
                            "truth_doppler_hz": batch.truth_target_doppler_hz,
                            "predicted_doppler_hz": h1["predicted_doppler_hz"],
                            "true_range_gated_doppler_hz": h1[
                                "true_range_gated_doppler_hz"
                            ],
                        },
                    )
                del model
                if split_name == "h1_development":
                    del h1
                else:
                    del h0

    expected_counts = {
        name: int(config["data"]["splits"][name]["scene_count"]) for name in split_names
    }
    if any(len(ids) != expected_counts[name] for name, ids in split_scene_ids.items()):
        raise RuntimeError("R5-C teaching generated split count differs from the preregistered config")
    if any(
        split_scene_ids[left].intersection(split_scene_ids[right])
        for index, left in enumerate(split_names)
        for right in split_names[index + 1 :]
    ):
        raise RuntimeError("R5-C teaching generated split IDs overlap")

    summarized: dict[str, dict[str, Any]] = {}
    paired_payloads: dict[str, dict[str, Any]] = {}
    for arm in config["training"]["arms"]:
        for split in per_arm[arm]:
            per_arm[arm][split] = _join_arrays(per_arm[arm][split])
        evaluation_payload = {
            "development": per_arm[arm]["development"],
        }
        verification_payload = {
            "overall": per_arm[arm]["verification"],
            "c3": per_arm[arm]["c3_h0_verification"],
        }
        summarized[arm] = summarize_pilot_arm(
            evaluation_payload,
            per_arm[arm]["calibration"],
            verification_payload,
            config,
        )
        paired_payload = {
            "calibration": per_arm[arm]["calibration"],
            "development": per_arm[arm]["development"],
            "c3_h0_verification": per_arm[arm]["c3_h0_verification"],
        }
        paired_payloads[arm] = paired_payload
        _save_arm_predictions(arm, per_arm[arm])

    contrasts = paired_teaching_contrasts(paired_payloads, config)
    elapsed = time.perf_counter() - started
    metrics = {
        "experiment": "D2-A-R5-C-teaching",
        "protocol_version": config["experiment"]["protocol_version"],
        "git_sha": head,
        "config": config,
        "preflight_git_sha": preflight["git_sha"],
        "per_arm": summarized,
        "paired_contrasts": contrasts,
        "evaluation_seconds": float(elapsed),
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_peak_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_reserved_bytes": cuda_peak_reserved,
        "split_scene_counts": {name: len(ids) for name, ids in split_scene_ids.items()},
        "full_iq_saved": False,
        "teacher_maps_saved": False,
        "final_blind": False,
    }
    manifest_path = OUTPUT_ROOT / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    metrics_path = OUTPUT_ROOT / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return metrics


def run_analysis(config: dict[str, Any]) -> dict[str, Any]:
    """Write a concise report from the frozen metrics without regenerating data."""
    _require_training_authorization(config)
    head = _require_committed_main()
    metrics_path = OUTPUT_ROOT / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("R5-C teaching analysis requires metrics.json from the evaluation phase")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("git_sha") != head or metrics.get("config") != config:
        raise RuntimeError("R5-C teaching metrics do not match the pushed code/config")
    rows = []
    c3_cell_rows = []
    for arm in config["training"]["arms"]:
        result = metrics["per_arm"][arm]
        dev = result["development"]["overall"]
        identifiable = result["development"]["identifiable_C2_C3"]
        overall_h0 = result["h0_verification"]["overall"]["overall"]
        c3_h0 = result["h0_verification"]["c3"]["overall"]
        rows.append(
            f"| {arm} | {overall_h0['pfa_ge']:.4f} | "
            f"{overall_h0['pfa_ci95_exact_upper']:.4f} | {c3_h0['pfa_ge']:.4f} | "
            f"{c3_h0['pfa_ci95_exact_upper']:.4f} | {dev['pd']:.4f} | {dev['auc']:.4f} | "
            f"{identifiable['joint_success']:.4f} | "
            f"{identifiable['range_rmse_all_h1_cells']:.3f} | "
            f"{identifiable['doppler_rmse_all_h1_hz']:.3f} |"
        )
        h0_c3_cells = result["h0_verification"]["c3"]["c3_cells"]
        for condition, cell in result["development"]["c3_cells"].items():
            h0_cell = h0_c3_cells.get(condition)
            h0_pfa = "n/a" if h0_cell is None else f"{h0_cell['pfa_ge']:.4f}"
            h0_upper = "n/a" if h0_cell is None else f"{h0_cell['pfa_ci95_exact_upper']:.4f}"
            c3_cell_rows.append(
                f"| {arm} | {condition} | {cell['pd']:.4f} | {cell['joint_success']:.4f} | "
                f"{cell['range_rmse_all_h1_cells']:.3f} | {cell['doppler_rmse_all_h1_hz']:.3f} | "
                f"{h0_pfa} | {h0_upper} |"
            )
    contrasts = metrics["paired_contrasts"]
    decision_rows = []
    for name in ("T1_minus_T0", "T2_minus_T0", "T3_minus_T0", "T2_minus_T3"):
        item = (
            contrasts["secondary"][name]
            if name in {"T3_minus_T0", "T2_minus_T3"}
            else contrasts["primary"][name]
        )
        gate = (
            "descriptive"
            if name in {"T3_minus_T0", "T2_minus_T3"}
            else str(item["passed"])
        )
        decision_rows.append(
            f"| {name} | {item['point_difference']:+.4f} | "
            f"[{item['nominal_95_ci'][0]:+.4f}, {item['nominal_95_ci'][1]:+.4f}] | "
            f"{gate} |"
        )
    passing = [
        arm
        for arm in ("T1", "T2")
        if contrasts["primary"][f"{arm}_minus_T0"]["passed"]
        and contrasts["task_protection"][arm]["passed"]
    ]
    recommendation = (
        f"仅 {', '.join(passing)} 同时通过本 pilot 的主收益与保护筛选；这仍需独立多种子确认。"
        if passing
        else "T1/T2 均未同时通过主收益和任务保护筛选；停止当前教师内容分支，不启动时机、可靠性或机制扩展。"
    )
    protection_rows = []
    for arm in ("T1", "T2"):
        gate = contrasts["task_protection"][arm]
        primary = contrasts["primary"][f"{arm}_minus_T0"]
        range_ratio = gate["C2_C3_range_rmse_ratio"]
        doppler_ratio = gate["C2_C3_doppler_rmse_ratio"]
        c3_delta = gate["C3_H0_pfa_difference"]
        protection_rows.append(
            f"| {arm} | {primary['point_difference']:+.4f} | "
            f"{primary['simultaneous_one_sided_97_5_lower_bound']:+.4f} | "
            f"{gate['overall_pd']['point_difference']:+.4f} | "
            f"{gate['overall_joint']['point_difference']:+.4f} | "
            f"{range_ratio['point_ratio']:.3f} [{range_ratio['nominal_95_ci'][0]:.3f}, {range_ratio['nominal_95_ci'][1]:.3f}] | "
            f"{doppler_ratio['point_ratio']:.3f} [{doppler_ratio['nominal_95_ci'][0]:.3f}, {doppler_ratio['nominal_95_ci'][1]:.3f}] | "
            f"{gate['C3_H0_pfa_absolute']['point_pfa']:.4f} | "
            f"{c3_delta['nominal_95_upper']:+.4f} | {gate['passed']} |"
        )
    analysis = "\n".join(
        [
            "# D2-A-R5-C Teaching Content Pilot 分析",
            "",
            f"Git SHA：`{head}`；总纲：`docs/RESEARCH_CHARTER.md` v1.3。",
            "",
            "本结果是单训练 seed 的新开发场景筛选，不是跨 seed 确认、最终盲测或检测器资格认证。",
            "AUC 采用独立 H0 verification 分数按冻结条件分层，并按 H1 development 条件样本数加权；C3 使用专用 H0 verification split。",
            "",
            "## 每臂指标",
            "",
            "| 臂 | overall H0 Pfa | overall exact 95% upper | C3 H0 Pfa | C3 exact 95% upper | Pd | AUC | C2+C3 joint | C2+C3 range RMSE | C2+C3 Doppler RMSE (Hz) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "## C3 分组",
            "",
            "| 臂 | 条件 | Pd | Joint | Range RMSE | Doppler RMSE (Hz) | 独立 C3 H0 Pfa | exact 95% upper |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
            *c3_cell_rows,
            "",
            "## 配对对比",
            "",
            "| 对比 | C2+C3 joint 差 | nominal 95% CI | 主门 |",
            "|---|---:|---:|---|",
            *decision_rows,
            "",
            "## T1/T2 任务保护",
            "",
            "| 臂 | C2+C3 joint 差 | simultaneous 97.5% lower | overall Pd 差 | overall joint 差 | range RMSE ratio [95% CI] | Doppler RMSE ratio [95% CI] | C3 H0 Pfa | C3 Pfa 差 95% upper | 全部保护 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            *protection_rows,
            "",
            "## 阶段筛选",
            "",
            recommendation,
            "",
            "T3 是 truth-assisted 对照，不是教师输出；T2−T3 是描述性对照，不设晋级门。T1/T2 的辅助响应由 IQ-only K3 计算；所有臂保留物理真值任务监督。场景 bootstrap 仅描述场景抽样，不代表训练 seed 方差。",
            "",
        ]
    )
    path = OUTPUT_ROOT / "analysis.md"
    if path.exists():
        raise FileExistsError(f"R5-C teaching analysis already exists; preserve and inspect: {path}")
    path.write_text(analysis, encoding="utf-8")
    return {"git_sha": head, "analysis_path": str(path), "passing_teacher_arms": passing}


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
    elif args.phase == "evaluate":
        result = run_evaluation(config)
    else:
        result = run_analysis(config)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
