"""Full-chain hybrid detection and parameter-estimation systems for D2-A-R6-F."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np
import torch
import yaml

from radardistill.baselines import estimate_mle_local, wrap_frequency
from radardistill.d2a_baselines import dpca_difference, run_b3
from radardistill.evaluation.d2a_r6a import (
    build_split_plan,
    calibrate_threshold,
    summarize_parameter_estimator,
    truth_range_to_bin,
    wrapped_error,
)
from radardistill.evaluation.d2a_r6e import (
    empirical_auc,
    exact_upper,
    h0_metrics,
    h1_metrics,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_architectures import build_r6e_student, count_r6e_parameters


SYSTEM_NAMES = ("A", "B", "C", "D", "E", "F")
MODEL_OUTPUT_KEYS = ("presence_logit", "range_logits", "range_offset", "doppler_vector")
ROOT = Path(__file__).resolve().parents[3]
STAGE = ROOT / "experiments/D2-A-R6/R6-F"
PREDICTIONS = STAGE / "predictions"
DEVICE_NAME = "NVIDIA GeForce RTX 3050 Laptop GPU"
SOURCE_PATHS = (
    "src/radardistill/evaluation/d2a_r6f.py",
    "scripts/run_d2a_r6f.py",
    "src/radardistill/students/r6_architectures.py",
    "src/radardistill/evaluation/d2a_r6a.py",
    "src/radardistill/evaluation/d2a_r6e.py",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/d2a_baselines.py",
    "src/radardistill/baselines.py",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    result = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"expected a YAML mapping in {path}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


def _configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _read_yaml(STAGE / "config.yaml")
    condition_source = _read_yaml(ROOT / config["radar"]["condition_source"])
    physics = _read_yaml(ROOT / config["radar"]["physics_source"])
    conditions = condition_source["radar"]["conditions"]
    if len(conditions) != int(config["radar"]["condition_count"]):
        raise ValueError("R6-F condition count differs from the frozen R6-A source")
    return config, condition_source, physics


def build_r6f_split_plans(
    config: Mapping[str, Any], conditions: Sequence[Mapping[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Use the registered R6-A condition table to construct each R6-F split."""
    data = config["data"]
    plan_config = {
        "data": {
            "condition_count": len(conditions),
            "condition_seed_stride": data["condition_seed_stride"],
            "chunk_size": data["chunk_size"],
            "chunk_seed_stride": data["chunk_seed_stride"],
            "splits": data["splits"],
        },
        "radar": {"conditions": conditions},
    }
    return {
        name: build_split_plan(plan_config, name)
        for name in ("h0_calibration", "h0_verification", "h1_development")
    }


def _prior_scene_ids() -> tuple[set[str], list[str]]:
    known: set[str] = set()
    files: list[str] = []
    experiments = ROOT / "experiments"
    for path in sorted(experiments.rglob("*.csv")):
        if path.is_relative_to(STAGE):
            continue
        import csv

        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            key = next(
                (name for name in ("scene_id", "base_scene_id", "base_scene_ids") if name in fields),
                None,
            )
            if key is None:
                continue
            for row in reader:
                value = str(row.get(key, ""))
                if value:
                    known.update(part for part in value.split(";") if part)
        files.append(str(path.relative_to(ROOT)))
    return known, files


def _source_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    paths = [ROOT / relative for relative in SOURCE_PATHS]
    paths.extend(
        ROOT / config["models"][key][entry]
        for key in ("ce_tcn", "j1_joint")
        for entry in ("checkpoint", "training_record")
    )
    paths.extend(
        ROOT / relative
        for relative in (config["radar"]["condition_source"], config["radar"]["physics_source"])
    )
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _tracked_worktree_clean() -> bool:
    return not _git("status", "--porcelain", "--untracked-files=no")


def r6f_preflight() -> dict[str, Any]:
    """Freeze code/data/checkpoint identity and validate the reserved scene IDs."""
    config, condition_config, _ = _configs()
    if not _tracked_worktree_clean():
        raise RuntimeError("R6-F preflight requires a committed clean tracked worktree")
    if not torch.cuda.is_available():
        raise RuntimeError("R6-F requires CUDA; CPU fallback is forbidden")
    device_name = torch.cuda.get_device_name(0)
    if device_name != DEVICE_NAME:
        raise RuntimeError(f"R6-F requires {DEVICE_NAME}, found {device_name}")
    plans = build_r6f_split_plans(config, condition_config["radar"]["conditions"])
    prior_ids, prior_files = _prior_scene_ids()
    scene_audit = audit_r6f_scene_ids(
        plans,
        g1_seed=int(config["data"]["g1_test_seed"]),
        g1_count=int(config["data"]["g1_test_count"]),
        prior_scene_ids=prior_ids,
    )
    for arm, expected in (("ce_tcn", 63429), ("j1_joint", 70277)):
        record_path = ROOT / config["models"][arm]["training_record"]
        checkpoint_path = ROOT / config["models"][arm]["checkpoint"]
        record = json.loads(record_path.read_text(encoding="utf-8"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if record.get("status") != "complete" or int(record.get("updates", -1)) != 51200:
            raise RuntimeError(f"R6-F frozen {arm} training record is incomplete")
        if int(checkpoint.get("update", -1)) != int(record.get("best_update", -2)):
            raise RuntimeError(f"R6-F frozen {arm} checkpoint update differs from its record")
        if checkpoint.get("owner") != record.get("owner"):
            raise RuntimeError(f"R6-F frozen {arm} checkpoint owner differs from its record")
        if int(config["models"][arm]["expected_parameter_count"]) != expected:
            raise RuntimeError(f"R6-F frozen {arm} parameter count changed in config")
    fingerprints = _source_fingerprints(config)
    result = {
        "status": "G0_PASS",
        "experiment": "D2-A-R6-F",
        "git_sha": _git("rev-parse", "HEAD"),
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_and_checkpoint_sha256": fingerprints,
        "condition_config_sha256": _sha256(ROOT / config["radar"]["condition_source"]),
        "physics_config_sha256": _sha256(ROOT / config["radar"]["physics_source"]),
        "gpu": device_name,
        "device": "cuda:0",
        "precision": "FP32",
        "scene_id_audit": scene_audit,
        "prior_scene_manifest_paths": prior_files,
        "split_scene_counts": {key: int(value["scene_count"]) for key, value in config["data"]["splits"].items()},
        "final_blind_generated": False,
    }
    _write_json(STAGE / "preflight.json", result)
    return result


def _require_preflight() -> dict[str, Any]:
    path = STAGE / "preflight.json"
    if not path.is_file():
        raise RuntimeError("R6-F G0 preflight is missing")
    config, _, _ = _configs()
    record = json.loads(path.read_text(encoding="utf-8"))
    valid = (
        record.get("status") == "G0_PASS"
        and record.get("git_sha") == _git("rev-parse", "HEAD")
        and record.get("protocol_sha256") == _sha256(STAGE / "protocol.md")
        and record.get("config_sha256") == _sha256(STAGE / "config.yaml")
        and record.get("source_and_checkpoint_sha256") == _source_fingerprints(config)
        and _tracked_worktree_clean()
    )
    if not valid:
        raise RuntimeError("R6-F G0 record is stale or committed sources changed")
    return record


def _load_r6e_model(arm: str, config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    if device != torch.device("cuda:0") or not torch.cuda.is_available():
        raise RuntimeError("R6-F student inference must use cuda:0")
    if torch.cuda.get_device_name(0) != DEVICE_NAME:
        raise RuntimeError(f"R6-F requires {DEVICE_NAME}")
    model_spec = config["models"][arm]
    record = json.loads((ROOT / model_spec["training_record"]).read_text(encoding="utf-8"))
    payload = torch.load(ROOT / model_spec["checkpoint"], map_location="cpu", weights_only=False)
    if payload.get("owner") != record.get("owner") or int(payload.get("update", -1)) != int(record.get("best_update", -2)):
        raise RuntimeError(f"R6-F {arm} checkpoint does not match its frozen training record")
    model = build_r6e_student(arm, seed=int(model_spec["training_seed"]))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    count = count_r6e_parameters(model)["parameters"]
    if count != int(model_spec["expected_parameter_count"]):
        raise RuntimeError(f"R6-F {arm} parameter count {count} differs from its registered value")
    return model.to(device).eval()


def _model_outputs(
    model: torch.nn.Module, iq: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("model IQ input must be complex [B,3,130,32]")
    iq_ri = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(iq_ri)).to(device, non_blocking=False)
    with torch.inference_mode():
        outputs = model(tensor)
    _validate_model_outputs(outputs, len(values))
    return {key: outputs[key].detach().cpu() for key in MODEL_OUTPUT_KEYS}


def r6f_g1() -> dict[str, Any]:
    """Run a fixed paired-simulation chain check at batch 16 and repeated batch 1."""
    preflight = _require_preflight()
    config, condition_config, physics = _configs()
    conditions = condition_config["radar"]["conditions"]
    data_config = config["data"]
    condition_index = int(data_config["g1_condition_index"])
    condition = conditions[condition_index]
    if str(condition["scenario_id"]) != "partial_overlap" or str(condition["course"]) != "C3":
        raise RuntimeError("R6-F G1 must use the frozen C3 partial-overlap condition")
    generated = generate_d2a_paired_batch(
        int(data_config["g1_test_count"]),
        int(data_config["g1_test_seed"]),
        float(condition["snr_db"]),
        float(condition["cnr_db"]),
        dict(physics),
        scenario_id=str(condition["scenario_id"]),
    )
    h0_iq = generated.iq_h0
    h1_iq = generated.iq_h1
    device = torch.device("cuda:0")
    prf_hz = float(config["radar"]["prf_hz"])
    system_batch_results: dict[str, dict[str, np.ndarray]] = {}
    model_consistency: dict[str, Any] = {}
    for arm, inference_system in (("ce_tcn", "CE"), ("j1_joint", "J1")):
        model = _load_r6e_model(arm, config, device)
        with torch.inference_mode():
            h0_batched_outputs = _model_outputs(model, h0_iq, device)
            h1_batched_outputs = _model_outputs(model, h1_iq, device)
            h0_single_outputs = [
                _model_outputs(model, h0_iq[index : index + 1], device)
                for index in range(len(h0_iq))
            ]
            h1_single_outputs = [
                _model_outputs(model, h1_iq[index : index + 1], device)
                for index in range(len(h1_iq))
            ]
        def concatenate_single(items: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
            return {key: torch.cat([item[key] for item in items], dim=0) for key in MODEL_OUTPUT_KEYS}

        h0_single = concatenate_single(h0_single_outputs)
        h1_single = concatenate_single(h1_single_outputs)
        if inference_system == "CE":
            h0_batch_systems = build_ce_system_predictions(
                h0_iq, h0_batched_outputs, physics, prf_hz=prf_hz, include_parameter_estimates=False
            )
            h0_single_systems = build_ce_system_predictions(
                h0_iq, h0_single, physics, prf_hz=prf_hz, include_parameter_estimates=False
            )
            h1_batch_systems = build_ce_system_predictions(
                h1_iq, h1_batched_outputs, physics, prf_hz=prf_hz, include_parameter_estimates=True
            )
            h1_single_systems = build_ce_system_predictions(
                h1_iq, h1_single, physics, prf_hz=prf_hz, include_parameter_estimates=True
            )
        else:
            h0_native_batch = decode_model_grid(h0_batched_outputs, prf_hz)
            h0_native_single = decode_model_grid(h0_single, prf_hz)
            h1_native_batch = decode_model_grid(h1_batched_outputs, prf_hz)
            h1_native_single = decode_model_grid(h1_single, prf_hz)
            h0_batch_systems = {"E": {"score": h0_native_batch["presence_probability"]}}
            h0_single_systems = {"E": {"score": h0_native_single["presence_probability"]}}
            h1_batch_systems = {"E": {
                "score": h1_native_batch["presence_probability"],
                "candidate_range_bin": h1_native_batch["candidate_range_bin"],
                "predicted_range_cell": h1_native_batch["range_cell"],
                "predicted_doppler_hz": h1_native_batch["doppler_hz"],
            }}
            h1_single_systems = {"E": {
                "score": h1_native_single["presence_probability"],
                "candidate_range_bin": h1_native_single["candidate_range_bin"],
                "predicted_range_cell": h1_native_single["range_cell"],
                "predicted_doppler_hz": h1_native_single["doppler_hz"],
            }}
        if inference_system == "CE":
            if set(h0_batch_systems) != {"A", "B", "C", "D", "F"}:
                raise RuntimeError("R6-F H0 CE path returned an unexpected system set")
            if any("predicted_range_cell" in item or "predicted_doppler_hz" in item for item in h0_batch_systems.values()):
                raise RuntimeError("R6-F H0 path must not return target parameter estimates")
            if not np.array_equal(
                h1_batch_systems["A"]["candidate_range_bin"],
                h1_batch_systems["D"]["detection_candidate_range_bin"],
            ):
                raise RuntimeError("R6-F B3 detection candidate was not propagated to system D")
            for system in ("B", "C", "D", "F"):
                if not np.array_equal(h1_batch_systems[system]["candidate_range_bin"], h1_batch_systems["B"]["candidate_range_bin"]):
                    raise RuntimeError(f"R6-F CE parameter candidate was not propagated to system {system}")
            np.testing.assert_allclose(
                h1_batch_systems["F"]["score"],
                h1_batch_systems["B"]["score"] * h1_batch_systems["F"]["fit_fraction"],
                rtol=1e-7,
                atol=1e-8,
            )
        compare = {"H0": (h0_batch_systems, h0_single_systems), "H1": (h1_batch_systems, h1_single_systems)}
        differences: dict[str, Any] = {}
        for split_name, (batch_result, single_result) in compare.items():
            for system, values in batch_result.items():
                differences[f"{split_name}_{system}"] = check_batch_consistency(
                    values, single_result[system], atol=1e-4
                )
                if not np.isfinite(np.asarray(values["score"])).all():
                    raise FloatingPointError(f"R6-F G1 {split_name} {system} score is not finite")
        if inference_system == "CE":
            system_batch_results.update(h1_batch_systems)
        else:
            system_batch_results.update(h1_batch_systems)
        model_consistency[arm] = differences
        del model, h0_batched_outputs, h1_batched_outputs, h0_single_outputs, h1_single_outputs
        torch.cuda.empty_cache()
    scores = np.concatenate([value["score"] for value in system_batch_results.values()])
    if not np.isfinite(scores).all():
        raise FloatingPointError("R6-F G1 generated a nonfinite system score")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != DEVICE_NAME:
        raise RuntimeError("R6-F G1 left the required CUDA device")
    abstention = decision_outputs(
        system_batch_results["E"]["score"],
        system_batch_results["E"]["predicted_range_cell"],
        system_batch_results["E"]["predicted_doppler_hz"],
        threshold=float(np.max(system_batch_results["E"]["score"]) + 1.0),
    )
    if abstention["detected"].any() or not np.isnan(abstention["output_range_cell"]).all():
        raise RuntimeError("R6-F null/abstain behavior failed in G1")
    result = {
        "status": "G1_PASS",
        "experiment": "D2-A-R6-F",
        "git_sha": preflight["git_sha"],
        "g0_status": preflight["status"],
        "condition_index": condition_index,
        "condition": dict(condition),
        "scene_seed": int(data_config["g1_test_seed"]),
        "paired_scene_count": int(len(h0_iq)),
        "batch_equivalence_absolute_tolerance": 1e-4,
        "batch_equivalence": model_consistency,
        "system_names_checked": list(SYSTEM_NAMES),
        "candidate_propagation": "pass",
        "h0_truth_parameter_leakage": "not_applicable_to_inference_interface; no parameter labels emitted",
        "null_abstention": "pass",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "final_blind_generated": False,
    }
    _write_json(STAGE / "g1.json", result)
    return result


def _append_arrays(buffer: dict[str, list[np.ndarray]], values: Mapping[str, np.ndarray]) -> None:
    for key, value in values.items():
        buffer.setdefault(key, []).append(np.asarray(value))


def _concatenate_arrays(buffer: Mapping[str, Sequence[np.ndarray]]) -> dict[str, np.ndarray]:
    if not buffer or any(not values for values in buffer.values()):
        raise ValueError("prediction split is empty or incomplete")
    result = {key: np.concatenate(values, axis=0) for key, values in buffer.items()}
    lengths = {len(value) for value in result.values()}
    if len(lengths) != 1:
        raise ValueError("prediction fields have inconsistent scene counts")
    return result


def _generate_split_batch(
    spec: Mapping[str, Any], physics: Mapping[str, Any]
):
    condition = spec["condition"]
    return generate_d2a_paired_batch(
        int(spec["count"]),
        int(spec["seed"]),
        float(condition["snr_db"]),
        float(condition["cnr_db"]),
        dict(physics),
        scenario_id=str(condition["scenario_id"]),
    )


def _run_ce_pass(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
) -> dict[str, int]:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    prf_hz = float(config["radar"]["prf_hz"])
    for split in ("h0_calibration", "h0_verification", "h1_development"):
        started = time.perf_counter()
        buffer: dict[str, list[np.ndarray]] = {}
        is_h1 = split == "h1_development"
        for spec in plans[split]:
            generated = _generate_split_batch(spec, physics)
            iq = generated.iq_h1 if is_h1 else generated.iq_h0
            ce_outputs = _model_outputs(model, iq, device)
            systems = build_ce_system_predictions(
                iq,
                ce_outputs,
                physics,
                prf_hz=prf_hz,
                include_parameter_estimates=is_h1,
            )
            chunk: dict[str, np.ndarray] = {
                "scene_id": np.asarray(generated.base_scene_id).astype("<U48"),
                "condition_index": np.full(int(spec["count"]), int(spec["condition_index"]), dtype=np.uint8),
            }
            for name, prediction in systems.items():
                for key, value in prediction.items():
                    chunk[f"{name}__{key}"] = np.asarray(value)
            if is_h1:
                truth_range = np.asarray(generated.truth_target_range_cell, dtype=np.float64)
                truth_doppler = np.asarray(generated.truth_target_doppler_hz, dtype=np.float64)
                truth_bins = truth_range_to_bin(truth_range, int(config["radar"]["range_bin_count"]))
                oracle_features = candidate_features_at_bins(
                    iq, truth_bins, physics, prf_hz=prf_hz, include_profile_ml=True
                )
                oracle_ce = decode_model_at_bins(ce_outputs, truth_bins, prf_hz)
                chunk.update({
                    "truth_range_cell": truth_range,
                    "truth_doppler_hz": truth_doppler,
                    "oracle_candidate_range_bin": truth_bins.astype(np.int16),
                    "oracle_fft4_doppler_hz": oracle_features["fft4_doppler_hz"],
                    "oracle_profile_ml_doppler_hz": oracle_features["profile_ml_doppler_hz"],
                    "oracle_ce_range_cell": oracle_ce["range_cell"],
                    "oracle_ce_doppler_hz": oracle_ce["doppler_hz"],
                })
            _append_arrays(buffer, chunk)
        arrays = _concatenate_arrays(buffer)
        path = PREDICTIONS / f"_ce_{split}.npz"
        _write_npz(path, arrays)
        counts[split] = len(arrays["scene_id"])
        print(json.dumps({
            "stage": "R6-F CE pass split complete",
            "split": split,
            "scenes": counts[split],
            "seconds": round(time.perf_counter() - started, 2),
        }, ensure_ascii=False), flush=True)
    return counts


def _run_j1_pass(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    physics: Mapping[str, Any],
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
    device: torch.device,
) -> dict[str, int]:
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    prf_hz = float(config["radar"]["prf_hz"])
    for split in ("h0_calibration", "h0_verification", "h1_development"):
        started = time.perf_counter()
        buffer: dict[str, list[np.ndarray]] = {}
        is_h1 = split == "h1_development"
        for spec in plans[split]:
            generated = _generate_split_batch(spec, physics)
            iq = generated.iq_h1 if is_h1 else generated.iq_h0
            outputs = _model_outputs(model, iq, device)
            decoded = decode_model_grid(outputs, prf_hz)
            chunk: dict[str, np.ndarray] = {
                "scene_id": np.asarray(generated.base_scene_id).astype("<U48"),
                "condition_index": np.full(int(spec["count"]), int(spec["condition_index"]), dtype=np.uint8),
                "E__score": decoded["presence_probability"],
                "E__candidate_range_bin": decoded["candidate_range_bin"].astype(np.int16),
            }
            if is_h1:
                truth_range = np.asarray(generated.truth_target_range_cell, dtype=np.float64)
                truth_doppler = np.asarray(generated.truth_target_doppler_hz, dtype=np.float64)
                truth_bins = truth_range_to_bin(truth_range, int(config["radar"]["range_bin_count"]))
                oracle = decode_model_at_bins(outputs, truth_bins, prf_hz)
                chunk.update({
                    "truth_range_cell": truth_range,
                    "truth_doppler_hz": truth_doppler,
                    "E__predicted_range_cell": decoded["range_cell"],
                    "E__predicted_doppler_hz": decoded["doppler_hz"],
                    "oracle_j1_range_cell": oracle["range_cell"],
                    "oracle_j1_doppler_hz": oracle["doppler_hz"],
                })
            _append_arrays(buffer, chunk)
        arrays = _concatenate_arrays(buffer)
        path = PREDICTIONS / f"_j1_{split}.npz"
        _write_npz(path, arrays)
        counts[split] = len(arrays["scene_id"])
        print(json.dumps({
            "stage": "R6-F J1 pass split complete",
            "split": split,
            "scenes": counts[split],
            "seconds": round(time.perf_counter() - started, 2),
        }, ensure_ascii=False), flush=True)
    return counts


def _finalize_predictions() -> dict[str, int]:
    counts: dict[str, int] = {}
    for split in ("h0_calibration", "h0_verification", "h1_development"):
        ce_path = PREDICTIONS / f"_ce_{split}.npz"
        j1_path = PREDICTIONS / f"_j1_{split}.npz"
        ce = _load_npz(ce_path)
        j1 = _load_npz(j1_path)
        if not np.array_equal(ce["scene_id"], j1["scene_id"]):
            raise ValueError(f"R6-F CE/J1 scene IDs do not match for {split}")
        if not np.array_equal(ce["condition_index"], j1["condition_index"]):
            raise ValueError(f"R6-F CE/J1 condition assignments do not match for {split}")
        if split == "h1_development":
            if not np.array_equal(ce["truth_range_cell"], j1["truth_range_cell"]):
                raise ValueError("R6-F CE/J1 range truths do not match")
            if not np.array_equal(ce["truth_doppler_hz"], j1["truth_doppler_hz"]):
                raise ValueError("R6-F CE/J1 Doppler truths do not match")
        combined = dict(ce)
        combined.update({key: value for key, value in j1.items() if key not in {"scene_id", "condition_index", "truth_range_cell", "truth_doppler_hz"}})
        score_map = compose_system_scores(
            combined["A__score"], combined["B__score"], combined["E__score"], combined["F__fit_fraction"]
        )
        for name, scores in score_map.items():
            combined[f"{name}__score"] = scores
        output = PREDICTIONS / f"{split}.npz"
        _write_npz(output, combined)
        counts[split] = len(combined["scene_id"])
        ce_path.unlink()
        j1_path.unlink()
    return counts


def paired_r6f_contrast(
    baseline: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    *,
    threshold_baseline: float,
    threshold_candidate: float,
    conditions: Sequence[Mapping[str, Any]],
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    """Compare a candidate with A using condition-stratified paired scene resampling."""
    ids = np.asarray(baseline["scene_id"]).astype(str)
    if not np.array_equal(ids, np.asarray(candidate["scene_id"]).astype(str)):
        raise ValueError("R6-F contrasts require the same scene IDs in the same order")
    condition_index = np.asarray(baseline["condition_index"], dtype=np.int64)
    if not np.array_equal(condition_index, np.asarray(candidate["condition_index"], dtype=np.int64)):
        raise ValueError("R6-F contrasts require identical condition assignments")
    for key in ("truth_range_cell", "truth_doppler_hz"):
        if not np.array_equal(np.asarray(baseline[key]), np.asarray(candidate[key])):
            raise ValueError(f"R6-F contrasts require matched {key}")
    if replicates <= 0 or seed < 0 or len(ids) == 0:
        raise ValueError("invalid R6-F bootstrap configuration")
    if np.any(condition_index < 0) or np.any(condition_index >= len(conditions)):
        raise ValueError("R6-F contrast condition index is outside the frozen table")
    base_score = np.asarray(baseline["score"], dtype=np.float64)
    cand_score = np.asarray(candidate["score"], dtype=np.float64)
    base_range_error = np.asarray(baseline["predicted_range_cell"], dtype=np.float64) - np.asarray(baseline["truth_range_cell"], dtype=np.float64)
    cand_range_error = np.asarray(candidate["predicted_range_cell"], dtype=np.float64) - np.asarray(candidate["truth_range_cell"], dtype=np.float64)
    base_doppler_error = np.asarray(wrapped_error(baseline["predicted_doppler_hz"], baseline["truth_doppler_hz"], 1300.0), dtype=np.float64)
    cand_doppler_error = np.asarray(wrapped_error(candidate["predicted_doppler_hz"], candidate["truth_doppler_hz"], 1300.0), dtype=np.float64)
    vectors = (base_score, cand_score, base_range_error, cand_range_error, base_doppler_error, cand_doppler_error)
    if any(value.shape != ids.shape or not np.isfinite(value).all() for value in vectors):
        raise ValueError("R6-F paired metrics must be finite aligned vectors")
    base_detected = base_score >= threshold_baseline
    cand_detected = cand_score >= threshold_candidate
    base_joint = base_detected & (np.abs(base_range_error) <= 1.0) & (np.abs(base_doppler_error) <= 20.0)
    cand_joint = cand_detected & (np.abs(cand_range_error) <= 1.0) & (np.abs(cand_doppler_error) <= 20.0)
    courses = np.asarray([str(item["course"]) for item in conditions])
    primary_conditions = {index for index, course in enumerate(courses) if course in {"C2", "C3"}}
    stratum_indices = [np.flatnonzero(condition_index == index) for index in range(len(conditions))]
    if any(len(indices) == 0 for indices in stratum_indices) or not primary_conditions:
        raise ValueError("R6-F bootstrap requires samples in every registered condition")
    primary_mask = np.isin(condition_index, list(primary_conditions))
    rng = np.random.default_rng(seed)
    joint_samples = np.empty(replicates, dtype=np.float64)
    pd_samples = np.empty(replicates, dtype=np.float64)
    range_ratio_samples = np.empty(replicates, dtype=np.float64)
    doppler_ratio_samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = [rng.choice(indices, size=len(indices), replace=True) for indices in stratum_indices]
        all_sampled = np.concatenate(sampled)
        primary_sampled = np.concatenate([
            sampled[index] for index in sorted(primary_conditions)
        ])
        joint_samples[replicate] = np.mean(
            cand_joint[primary_sampled].astype(np.float64) - base_joint[primary_sampled].astype(np.float64)
        )
        pd_samples[replicate] = np.mean(
            cand_detected[all_sampled].astype(np.float64) - base_detected[all_sampled].astype(np.float64)
        )
        range_ratio_samples[replicate] = math.sqrt(
            np.mean(cand_range_error[all_sampled] ** 2)
            / max(np.mean(base_range_error[all_sampled] ** 2), 1e-24)
        )
        doppler_ratio_samples[replicate] = math.sqrt(
            np.mean(cand_doppler_error[all_sampled] ** 2)
            / max(np.mean(base_doppler_error[all_sampled] ** 2), 1e-24)
        )
    joint_ci = [float(value) for value in np.quantile(joint_samples, [0.005, 0.995])]
    pd_ci = [float(value) for value in np.quantile(pd_samples, [0.0125, 0.9875])]
    range_ci = [float(value) for value in np.quantile(range_ratio_samples, [0.0125, 0.9875])]
    doppler_ci = [float(value) for value in np.quantile(doppler_ratio_samples, [0.0125, 0.9875])]
    joint_difference = float(np.mean(cand_joint[primary_mask].astype(np.float64) - base_joint[primary_mask].astype(np.float64)))
    range_ratio = float(math.sqrt(np.mean(cand_range_error**2) / max(np.mean(base_range_error**2), 1e-24)))
    doppler_ratio = float(math.sqrt(np.mean(cand_doppler_error**2) / max(np.mean(base_doppler_error**2), 1e-24)))
    pd_difference = float(np.mean(cand_detected.astype(np.float64) - base_detected.astype(np.float64)))
    protection = {
        "pd_difference": pd_difference,
        "pd_difference_ci97_5": pd_ci,
        "pd_lower_minimum": -0.02,
        "pd_pass": bool(pd_ci[0] >= -0.02),
        "range_rmse_ratio": range_ratio,
        "range_rmse_ratio_ci97_5": range_ci,
        "range_rmse_ratio_upper_maximum": 1.05,
        "range_pass": bool(range_ci[1] <= 1.05),
        "doppler_rmse_ratio": doppler_ratio,
        "doppler_rmse_ratio_ci97_5": doppler_ci,
        "doppler_rmse_ratio_upper_maximum": 1.05,
        "doppler_pass": bool(doppler_ci[1] <= 1.05),
    }
    return {
        "group": "identifiable_C2_C3",
        "n_paired_scenes": int(primary_mask.sum()),
        "joint_success_difference": joint_difference,
        "joint_success_ci99": joint_ci,
        "confidence_level": 0.99,
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "meaningful_gain_threshold": 0.02,
        "candidate_screening_pass": bool(joint_difference >= 0.02 and joint_ci[0] > 0.0),
        "protection": protection,
    }


def summarize_candidate_correctness(
    candidate_bins: np.ndarray,
    truth_range_cell: np.ndarray,
    condition_index: np.ndarray,
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report exact nearest-gate candidate accuracy without conflating it with joint success."""
    bins = np.asarray(candidate_bins)
    truth = np.asarray(truth_range_cell, dtype=np.float64)
    index = np.asarray(condition_index, dtype=np.int64)
    if bins.ndim != 1 or truth.shape != bins.shape or index.shape != bins.shape or bins.size == 0:
        raise ValueError("candidate, truth and condition arrays must be aligned nonempty vectors")
    if not np.isfinite(bins).all() or np.any(bins != np.floor(bins)) or not np.isfinite(truth).all():
        raise ValueError("candidate bins and truth ranges must be finite; candidate bins must be integers")
    if np.any(index < 0) or np.any(index >= len(conditions)):
        raise ValueError("candidate condition index is outside the registered table")
    correct = bins.astype(np.int64) == truth_range_to_bin(truth, 32)
    courses = np.asarray([str(item["course"]) for item in conditions])
    groups = {
        "all": np.ones(len(index), dtype=bool),
        "C2_C3": np.isin(courses[index], ("C2", "C3")),
        "C3": courses[index] == "C3",
    }

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        count = int(correct[mask].sum())
        return {"n": n, "correct": count, "accuracy": None if n == 0 else count / n}

    by_condition = {}
    for condition in range(len(conditions)):
        mask = index == condition
        by_condition[str(condition)] = {
            "condition": dict(conditions[condition]),
            **summarize(mask),
        }
    return {
        "overall": summarize(groups["all"]),
        "by_group": {name: summarize(mask) for name, mask in groups.items() if name != "all"},
        "by_condition": by_condition,
    }


def summarize_candidate_agreement(
    detection_bins: np.ndarray,
    parameter_bins: np.ndarray,
    condition_index: np.ndarray,
    conditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report whether a hybrid chain's detector and estimator choose the same range gate."""
    detector = np.asarray(detection_bins)
    estimator = np.asarray(parameter_bins)
    index = np.asarray(condition_index, dtype=np.int64)
    if (
        detector.ndim != 1
        or estimator.shape != detector.shape
        or index.shape != detector.shape
        or detector.size == 0
    ):
        raise ValueError("detector, estimator and condition arrays must be aligned nonempty vectors")
    if (
        not np.isfinite(detector).all()
        or not np.isfinite(estimator).all()
        or np.any(detector != np.floor(detector))
        or np.any(estimator != np.floor(estimator))
    ):
        raise ValueError("detector and estimator candidate bins must be finite integers")
    if np.any(index < 0) or np.any(index >= len(conditions)):
        raise ValueError("candidate condition index is outside the registered table")
    agreement = detector.astype(np.int64) == estimator.astype(np.int64)
    courses = np.asarray([str(item["course"]) for item in conditions])
    groups = {
        "all": np.ones(len(index), dtype=bool),
        "C2_C3": np.isin(courses[index], ("C2", "C3")),
        "C3": courses[index] == "C3",
    }

    def summarize(mask: np.ndarray) -> dict[str, Any]:
        n = int(mask.sum())
        count = int(agreement[mask].sum())
        return {"n": n, "agree": count, "agreement_rate": None if n == 0 else count / n}

    by_condition = {}
    for condition in range(len(conditions)):
        mask = index == condition
        by_condition[str(condition)] = {
            "condition": dict(conditions[condition]),
            **summarize(mask),
        }
    return {
        "overall": summarize(groups["all"]),
        "by_group": {name: summarize(mask) for name, mask in groups.items() if name != "all"},
        "by_condition": by_condition,
    }


def check_batch_consistency(
    batched: Mapping[str, np.ndarray], single_calls: Mapping[str, np.ndarray], *, atol: float
) -> dict[str, Any]:
    """Reject observable pipeline outputs that change with inference batch size."""
    if not math.isfinite(atol) or atol < 0 or set(batched) != set(single_calls) or not batched:
        raise ValueError("batch-size consistency inputs or tolerance are invalid")
    differences: dict[str, float] = {}
    for key in batched:
        first = np.asarray(batched[key], dtype=np.float64)
        second = np.asarray(single_calls[key], dtype=np.float64)
        if first.shape != second.shape or not np.isfinite(first).all() or not np.isfinite(second).all():
            raise ValueError(f"batch-size consistency shape or finite check failed for {key}")
        difference = float(np.max(np.abs(first - second))) if first.size else 0.0
        differences[key] = difference
        if difference > atol:
            raise ValueError(
                f"batch-size consistency failed for {key}: max difference {difference} > {atol}"
            )
    return {"max_abs_difference": max(differences.values(), default=0.0), "by_field": differences}


def _system_view(arrays: Mapping[str, np.ndarray], system: str) -> dict[str, np.ndarray]:
    result = {
        "scene_id": arrays["scene_id"],
        "condition_index": arrays["condition_index"],
        "score": arrays[f"{system}__score"],
    }
    for key in (
        "predicted_range_cell",
        "predicted_doppler_hz",
        "candidate_range_bin",
        "detection_candidate_range_bin",
    ):
        value_key = f"{system}__{key}"
        if value_key in arrays:
            result[key] = arrays[value_key]
    for key in ("truth_range_cell", "truth_doppler_hz"):
        if key in arrays:
            result[key] = arrays[key]
    return result


def _doppler_summary(predicted: np.ndarray, truth: np.ndarray, prf_hz: float) -> dict[str, Any]:
    error = np.asarray(wrapped_error(predicted, truth, prf_hz), dtype=np.float64)
    return {
        "n": int(len(error)),
        "doppler_mae_hz": float(np.mean(np.abs(error))),
        "doppler_rmse_hz": float(np.sqrt(np.mean(error**2))),
        "doppler_bias_hz": float(np.mean(error)),
    }


def _oracle_estimator_metrics(arrays: Mapping[str, np.ndarray], prf_hz: float) -> dict[str, Any]:
    truth_range = arrays["truth_range_cell"]
    truth_doppler = arrays["truth_doppler_hz"]
    result: dict[str, Any] = {
        "scope": "offline only; true nearest range gate is used only to select a Doppler profile",
        "fft4_at_true_gate": _doppler_summary(arrays["oracle_fft4_doppler_hz"], truth_doppler, prf_hz),
        "profile_ml_at_true_gate": _doppler_summary(arrays["oracle_profile_ml_doppler_hz"], truth_doppler, prf_hz),
        "ce_parameter_head_at_true_gate": {
            "range_rmse_cells": float(np.sqrt(np.mean((arrays["oracle_ce_range_cell"] - truth_range) ** 2))),
            **_doppler_summary(arrays["oracle_ce_doppler_hz"], truth_doppler, prf_hz),
        },
        "j1_parameter_head_at_true_gate": {
            "range_rmse_cells": float(np.sqrt(np.mean((arrays["oracle_j1_range_cell"] - truth_range) ** 2))),
            **_doppler_summary(arrays["oracle_j1_doppler_hz"], truth_doppler, prf_hz),
        },
    }
    return result


def _require_g1(preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = STAGE / "g1.json"
    if not path.is_file():
        raise RuntimeError("R6-F G1 chain test is missing")
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "G1_PASS" or record.get("git_sha") != preflight.get("git_sha"):
        raise RuntimeError("R6-F G1 did not pass on the current committed implementation")
    return record


def _ensure_prediction_owner(preflight: Mapping[str, Any]) -> dict[str, Any]:
    path = STAGE / "output_owner.json"
    owner = {
        "experiment": "D2-A-R6-F",
        "git_sha": preflight["git_sha"],
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "prediction_directory": str(PREDICTIONS.relative_to(ROOT)),
        "raw_iq_persisted": False,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != owner:
            raise RuntimeError("R6-F prediction directory belongs to a different run")
    elif PREDICTIONS.exists() and any(PREDICTIONS.iterdir()):
        raise RuntimeError("R6-F prediction path contains files without a matching owner record")
    else:
        _write_json(path, owner)
    return owner


def r6f_evaluate() -> dict[str, Any]:
    """Run frozen full-chain prediction, independent calibration and development evaluation."""
    preflight = _require_preflight()
    g1 = _require_g1(preflight)
    _ensure_prediction_owner(preflight)
    config, condition_config, physics = _configs()
    conditions = condition_config["radar"]["conditions"]
    plans = build_r6f_split_plans(config, conditions)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    ce_model = _load_r6e_model("ce_tcn", config, device)
    ce_peak_bytes = 0
    ce_counts = _run_ce_pass(ce_model, config, conditions, physics, plans, device)
    torch.cuda.synchronize(device)
    ce_peak_bytes = int(torch.cuda.max_memory_allocated(device))
    del ce_model
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(device)
    j1_model = _load_r6e_model("j1_joint", config, device)
    j1_counts = _run_j1_pass(j1_model, config, physics, plans, device)
    torch.cuda.synchronize(device)
    j1_peak_bytes = int(torch.cuda.max_memory_allocated(device))
    del j1_model
    torch.cuda.empty_cache()

    counts = _finalize_predictions()
    if counts != ce_counts or counts != j1_counts:
        raise RuntimeError("R6-F prediction pass scene counts do not match")
    arrays_by_split = {
        split: _load_npz(PREDICTIONS / f"{split}.npz")
        for split in ("h0_calibration", "h0_verification", "h1_development")
    }
    h0_cal = arrays_by_split["h0_calibration"]
    h0_ver = arrays_by_split["h0_verification"]
    h1 = arrays_by_split["h1_development"]
    thresholds: dict[str, float] = {}
    calibration_info: dict[str, Any] = {}
    system_metrics: dict[str, Any] = {}
    conditions_list = list(conditions)
    for system in SYSTEM_NAMES:
        cal_view = _system_view(h0_cal, system)
        ver_view = _system_view(h0_ver, system)
        threshold_info = calibrate_threshold(
            cal_view["score"], float(config["evaluation"]["all_systems"]["target_pfa_cpi"])
        )
        threshold = float(threshold_info["threshold"])
        thresholds[system] = threshold
        calibration_info[system] = threshold_info
        h0 = h0_metrics(cal_view, ver_view, threshold=threshold, conditions=conditions_list)
        global_upper = float(h0["verification"]["pfa_ci95_exact_upper"])
        absolute_h0_pass = bool(
            global_upper <= float(config["evaluation"]["global_verification_pfa_upper_maximum"])
            and h0["c3_six_cells_familywise_pass"]
        )
        h1_view = _system_view(h1, system)
        h1_summary = h1_metrics(
            h1_view,
            threshold=threshold,
            conditions=conditions_list,
            h0_verification=ver_view,
            prf_hz=float(config["radar"]["prf_hz"]),
        )
        detected = np.asarray(h1_view["score"], dtype=np.float64) >= threshold
        candidate_accuracy = summarize_candidate_correctness(
            h1_view["candidate_range_bin"],
            h1_view["truth_range_cell"],
            h1_view["condition_index"],
            conditions_list,
        )
        detector_estimator_agreement = None
        detected_detector_estimator_agreement = None
        if system == "D":
            detector_estimator_agreement = summarize_candidate_agreement(
                h1_view["detection_candidate_range_bin"],
                h1_view["candidate_range_bin"],
                h1_view["condition_index"],
                conditions_list,
            )
            if detected.any():
                detected_detector_estimator_agreement = summarize_candidate_agreement(
                    h1_view["detection_candidate_range_bin"][detected],
                    h1_view["candidate_range_bin"][detected],
                    h1_view["condition_index"][detected],
                    conditions_list,
                )
        detected_candidate_accuracy = (
            summarize_candidate_correctness(
                h1_view["candidate_range_bin"][detected],
                h1_view["truth_range_cell"][detected],
                h1_view["condition_index"][detected],
                conditions_list,
            )
            if detected.any() else None
        )
        detected_parameter_metrics = None
        if detected.any():
            detected_parameter_metrics = summarize_parameter_estimator(
                h1_view["predicted_range_cell"][detected],
                h1_view["truth_range_cell"][detected],
                h1_view["predicted_doppler_hz"][detected],
                h1_view["truth_doppler_hz"][detected],
                np.ones(int(detected.sum()), dtype=bool),
                prf_hz=float(config["radar"]["prf_hz"]),
                range_gate_cells=1.0,
                doppler_gate_hz=20.0,
            )
        abstention_by_condition = {}
        for condition_index, condition in enumerate(conditions_list):
            mask = h1_view["condition_index"] == condition_index
            abstention_by_condition[str(condition_index)] = {
                "condition": dict(condition),
                "n": int(mask.sum()),
                "null_outputs": int((~detected[mask]).sum()),
                "null_rate": float(np.mean(~detected[mask])),
            }
        system_metrics[system] = {
            "threshold_calibration": threshold_info,
            "h0": h0,
            "absolute_h0_qualification_pass": absolute_h0_pass,
            "h1_development": h1_summary,
            "candidate_correctness": candidate_accuracy,
            "detector_estimator_candidate_agreement": detector_estimator_agreement,
            "detected_detector_estimator_candidate_agreement": detected_detector_estimator_agreement,
            "detected_candidate_correctness": detected_candidate_accuracy,
            "abstention": {
                "null_outputs": int((~detected).sum()),
                "null_rate": float(np.mean(~detected)),
                "by_condition": abstention_by_condition,
            },
            "parameter_error_when_detected": detected_parameter_metrics,
        }

    contrasts: dict[str, Any] = {}
    baseline_view = _system_view(h1, "A")
    for system in SYSTEM_NAMES:
        if system == "A":
            continue
        candidate_view = _system_view(h1, system)
        contrast = paired_r6f_contrast(
            baseline_view,
            candidate_view,
            threshold_baseline=thresholds["A"],
            threshold_candidate=thresholds[system],
            conditions=conditions_list,
            replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
            seed=int(config["evaluation"]["paired_bootstrap_seed"]),
        )
        h0_pass = bool(system_metrics[system]["absolute_h0_qualification_pass"])
        protections = contrast["protection"]
        contrast["absolute_h0_qualification_pass"] = h0_pass
        contrast["full_candidate_pass"] = bool(
            contrast["candidate_screening_pass"]
            and protections["pd_pass"]
            and protections["range_pass"]
            and protections["doppler_pass"]
            and h0_pass
        )
        contrasts[f"{system}_minus_A"] = contrast

    cost_metrics = benchmark_r6f_system_costs(
        config, conditions_list, physics, plans["h1_development"], thresholds
    )

    result = {
        "status": "development_evaluation_complete",
        "experiment": "D2-A-R6-F",
        "protocol_version": config["experiment"]["protocol_version"],
        "git_sha": preflight["git_sha"],
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "source_and_checkpoint_sha256": preflight["source_and_checkpoint_sha256"],
        "g0_status": preflight["status"],
        "g1_status": g1["status"],
        "scene_counts": counts,
        "development_data_is_final_blind": False,
        "thresholds_by_system": thresholds,
        "systems": system_metrics,
        "primary_contrasts_vs_A": contrasts,
        "oracle_candidate_diagnostics": _oracle_estimator_metrics(h1, float(config["radar"]["prf_hz"])),
        "full_chain_cost": cost_metrics,
        "resources": {
            "gpu": torch.cuda.get_device_name(device),
            "device": str(device),
            "precision": "FP32",
            "simultaneous_models_resident": 1,
            "ce_peak_allocated_bytes": ce_peak_bytes,
            "j1_peak_allocated_bytes": j1_peak_bytes,
            "evaluation_seconds_without_cost_benchmark": float(time.perf_counter() - started),
        },
    }
    _write_json(STAGE / "metrics.json", result)
    return result


def run_cost_system_batch(
    system: str,
    iq: np.ndarray,
    physics: Mapping[str, Any],
    *,
    prf_hz: float,
    threshold: float,
    model: torch.nn.Module | None,
    device: torch.device,
) -> dict[str, Any]:
    """Run one deployed chain, including its threshold and detected-only estimator."""
    if system not in SYSTEM_NAMES:
        raise ValueError(f"unknown R6-F system {system}")
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("cost input must be complex [B,3,130,32]")
    if not math.isfinite(threshold):
        raise ValueError("system threshold must be finite")
    batch = len(values)
    ranges = np.zeros(batch, dtype=np.float64)
    dopplers = np.zeros(batch, dtype=np.float64)
    estimator_invocations = 0

    if system in {"A", "D"}:
        classical = run_b3(values, prf_hz, dict(physics))
        score = np.atleast_1d(np.asarray(classical.score, dtype=np.float64))
        bins_float = np.atleast_1d(np.asarray(classical.range_cell, dtype=np.float64))
        if np.any(bins_float != np.floor(bins_float)):
            raise ValueError("B3 candidate must be an integer range gate")
        bins = bins_float.astype(np.int64)
        parameter_bins = bins.copy() if system == "A" else np.full(batch, -1, dtype=np.int64)
    else:
        if model is None:
            raise ValueError(f"system {system} requires its registered neural model")
        model_outputs = _model_outputs(model, values, device)
        native = decode_model_grid(model_outputs, prf_hz)
        bins = native["candidate_range_bin"]
        parameter_bins = bins.copy()
        if system == "F":
            fit = candidate_features_at_bins(
                values, bins, physics, prf_hz=prf_hz, include_profile_ml=False
            )["fit_fraction"]
            score = native["presence_probability"] * fit
        else:
            score = native["presence_probability"]
        if system in {"B", "E"}:
            ranges[:] = native["range_cell"]
            dopplers[:] = native["doppler_hz"]
            estimator_invocations = batch

    detected = score >= threshold
    if system == "A" and detected.any():
        features = candidate_features_at_bins(
            values[detected], bins[detected], physics, prf_hz=prf_hz, include_profile_ml=False
        )
        ranges[detected] = bins[detected]
        dopplers[detected] = features["fft4_doppler_hz"]
        estimator_invocations = int(detected.sum())
    elif system == "D" and detected.any():
        if model is None:
            raise ValueError("system D requires its registered CE-TCN estimator")
        selected_outputs = _model_outputs(model, values[detected], device)
        native = decode_model_grid(selected_outputs, prf_hz)
        ranges[detected] = native["range_cell"]
        dopplers[detected] = native["doppler_hz"]
        parameter_bins[detected] = native["candidate_range_bin"]
        estimator_invocations = int(detected.sum())
    elif system in {"C", "F"} and detected.any():
        features = candidate_features_at_bins(
            values[detected], bins[detected], physics, prf_hz=prf_hz, include_profile_ml=True
        )
        ranges[detected] = native["range_cell"][detected]
        dopplers[detected] = features["profile_ml_doppler_hz"]
        estimator_invocations = int(detected.sum())
    elif system == "E" and detected.any():
        estimator_invocations = int(detected.sum())
    decision = decision_outputs(score, ranges, dopplers, threshold)
    result = {
        "score": score,
        "candidate_range_bin": parameter_bins,
        "detected_count": int(detected.sum()),
        "estimator_invocations": int(estimator_invocations),
        "decision": decision,
    }
    if system == "D":
        result["detection_candidate_range_bin"] = bins
    return result


def benchmark_r6f_system_costs(
    config: Mapping[str, Any],
    conditions: Sequence[Mapping[str, Any]],
    physics: Mapping[str, Any],
    h1_plan: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Measure each complete chain on identical condition-stratified H1 CPI batches."""
    total_cost_scenes = int(config["evaluation"]["cost_subset_count"])
    if total_cost_scenes % len(conditions):
        raise ValueError("R6-F cost scene count must divide evenly across conditions")
    count_per_condition = total_cost_scenes // len(conditions)
    batch_size = int(config["evaluation"]["cost_batch_size"])
    if batch_size != 16 or count_per_condition % batch_size:
        raise ValueError("R6-F cost subset must be condition-balanced and divisible by batch size 16")
    batches: list[np.ndarray] = []
    for condition_index in range(len(conditions)):
        rows = [row for row in h1_plan if int(row["condition_index"]) == condition_index]
        needed = count_per_condition
        selected = []
        for row in rows:
            if needed <= 0:
                break
            take = min(int(row["count"]), needed)
            generated = _generate_split_batch({**row, "count": take}, physics)
            selected.append(np.asarray(generated.iq_h1[:take]))
            needed -= take
        if needed:
            raise ValueError(f"R6-F cost subset lacks scenes for condition {condition_index}")
        condition_iq = np.concatenate(selected, axis=0)
        if len(condition_iq) != count_per_condition:
            raise RuntimeError("R6-F cost subset scene count changed")
        batches.extend(
            condition_iq[start : start + batch_size]
            for start in range(0, count_per_condition, batch_size)
        )
    if len(batches) != 24:
        raise RuntimeError("R6-F cost subset must contain 24 batches")
    device = torch.device("cuda:0")
    warmup_count = int(config["evaluation"]["cost_warmup_batches"])
    repeats = int(config["evaluation"]["cost_repeats"])
    result: dict[str, Any] = {
        "scope": "input IQ resident on CPU; timing includes conversion, transfer, complete chain and null output; excludes data generation and file writes",
        "scene_count": len(conditions) * count_per_condition,
        "scenes_per_condition": count_per_condition,
        "batch_size": batch_size,
        "warmup_batches": warmup_count,
        "measured_repeats": repeats,
        "systems": {},
    }
    model_by_system: dict[str, torch.nn.Module | None] = {name: None for name in SYSTEM_NAMES}
    loaded_arm: str | None = None
    for system in SYSTEM_NAMES:
        if system in {"B", "C", "D", "F"}:
            arm = "ce_tcn"
        elif system == "E":
            arm = "j1_joint"
        else:
            arm = None
        if arm is not None and arm != loaded_arm:
            for key in SYSTEM_NAMES:
                model_by_system[key] = None
            if "model" in locals():
                del model
            torch.cuda.empty_cache()
            model = _load_r6e_model(arm, config, device)
            for key in SYSTEM_NAMES:
                model_by_system[key] = model if key in ({"B", "C", "D", "F"} if arm == "ce_tcn" else {"E"}) else None
            loaded_arm = arm
            torch.cuda.reset_peak_memory_stats(device)
        model = model_by_system[system]

        def run_one(iq_batch: np.ndarray) -> dict[str, Any]:
            return run_cost_system_batch(
                system,
                iq_batch,
                physics,
                prf_hz=float(config["radar"]["prf_hz"]),
                threshold=float(thresholds[system]),
                model=model,
                device=device,
            )

        for batch in batches[:warmup_count]:
            run_one(batch)
        if model is not None:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        batch_seconds: list[float] = []
        detection_count = 0
        estimator_count = 0
        for _ in range(repeats):
            for batch in batches:
                if model is not None:
                    torch.cuda.synchronize(device)
                tick = time.perf_counter()
                outcome = run_one(batch)
                if model is not None:
                    torch.cuda.synchronize(device)
                batch_seconds.append(time.perf_counter() - tick)
                detection_count += int(outcome["detected_count"])
                estimator_count += int(outcome["estimator_invocations"])
        elapsed = float(sum(batch_seconds))
        per_cpi = np.asarray(batch_seconds, dtype=np.float64) / batch_size
        peak = int(torch.cuda.max_memory_allocated(device)) if model is not None else 0
        result["systems"][system] = {
            "hardware": {"cpu": "host CPU", "gpu": None if model is None else torch.cuda.get_device_name(device), "device": None if model is None else str(device)},
            "elapsed_seconds": elapsed,
            "measured_batches": int(len(batches) * repeats),
            "measured_cpi": int(len(batches) * repeats * batch_size),
            "mean_seconds_per_cpi": float(elapsed / (len(batches) * repeats * batch_size)),
            "median_seconds_per_cpi": float(np.median(per_cpi)),
            "p95_seconds_per_cpi": float(np.quantile(per_cpi, 0.95)),
            "throughput_cpi_per_second_for_this_batching": float(len(batches) * repeats * batch_size / elapsed),
            "detected_fraction_over_repeats": float(detection_count / (len(batches) * repeats * batch_size)),
            "estimator_invocations_over_repeats": estimator_count,
            "peak_cuda_allocated_bytes": peak,
        }
        print(json.dumps({
            "stage": "R6-F cost system complete",
            "system": system,
            "mean_ms_per_cpi": round(result["systems"][system]["mean_seconds_per_cpi"] * 1000.0, 4),
        }, ensure_ascii=False), flush=True)
    for key in SYSTEM_NAMES:
        model_by_system[key] = None
    torch.cuda.empty_cache()
    return result


def audit_r6f_scene_ids(
    plans: Mapping[str, Sequence[Mapping[str, int]]],
    *,
    g1_seed: int,
    g1_count: int,
    prior_scene_ids: set[str],
) -> dict[str, Any]:
    """Ensure split and G1 scene IDs are unique and absent from prior manifests."""
    if g1_seed < 0 or g1_count <= 0:
        raise ValueError("G1 seed must be nonnegative and G1 count must be positive")
    planned: list[set[str]] = []
    per_split: dict[str, int] = {}
    for split, rows in plans.items():
        ids = {
            f"d2a-scene-{int(row['seed'])}-{index}"
            for row in rows
            for index in range(int(row["count"]))
        }
        expected = sum(int(row["count"]) for row in rows)
        if len(ids) != expected:
            raise ValueError(f"scene IDs are duplicated within {split}")
        per_split[str(split)] = len(ids)
        planned.append(ids)
    combined = set().union(*planned) if planned else set()
    if sum(per_split.values()) != len(combined):
        raise ValueError("scene IDs overlap between registered splits")
    g1_ids = {f"d2a-scene-{g1_seed}-{index}" for index in range(g1_count)}
    if combined.intersection(g1_ids):
        raise ValueError("G1 scene IDs overlap a registered evaluation split")
    overlap = combined.union(g1_ids).intersection(prior_scene_ids)
    if overlap:
        raise ValueError(f"prior scene ID overlap: {sorted(overlap)[:3]}")
    return {
        "planned_scene_ids": len(combined),
        "g1_scene_ids": len(g1_ids),
        "known_scene_ids": len(prior_scene_ids),
        "overlap": 0,
        "scene_counts_by_split": per_split,
    }


def _validate_model_outputs(outputs: Mapping[str, torch.Tensor], batch: int) -> None:
    missing = [key for key in MODEL_OUTPUT_KEYS if key not in outputs]
    if missing:
        raise ValueError(f"model output is missing fields: {missing}")
    shapes = {
        "presence_logit": (batch,),
        "range_logits": (batch, 32),
        "range_offset": (batch, 32),
        "doppler_vector": (batch, 32, 2),
    }
    for key, expected in shapes.items():
        value = outputs[key]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            raise ValueError(f"model output {key} must have shape {expected}")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"model output {key} contains nonfinite values")


def _decode_selected(
    outputs: Mapping[str, torch.Tensor], candidate_bins: np.ndarray, prf_hz: float
) -> dict[str, np.ndarray]:
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    bins = np.asarray(candidate_bins)
    batch = int(outputs["presence_logit"].shape[0])
    if bins.shape != (batch,) or not np.issubdtype(bins.dtype, np.number):
        raise ValueError("candidate bins must be one numeric value per CPI")
    if not np.isfinite(bins).all() or np.any(bins != np.floor(bins)):
        raise ValueError("candidate bins must be finite integers")
    bins = bins.astype(np.int64)
    if np.any((bins < 0) | (bins >= 32)):
        raise ValueError("candidate bin outside 32-bin range axis")
    _validate_model_outputs(outputs, batch)
    device_bins = torch.as_tensor(bins, device=outputs["range_logits"].device)
    rows = torch.arange(batch, device=device_bins.device)
    offset = outputs["range_offset"][rows, device_bins]
    vectors = outputs["doppler_vector"][rows, device_bins]
    angle = torch.atan2(vectors[:, 1], vectors[:, 0])
    doppler = torch.remainder(angle * (prf_hz / (2.0 * math.pi)) + prf_hz / 2.0, prf_hz) - prf_hz / 2.0
    return {
        "presence_probability": torch.sigmoid(outputs["presence_logit"]).detach().cpu().numpy().astype(np.float64),
        "candidate_range_bin": bins,
        "range_cell": (device_bins.to(offset.dtype) + offset).detach().cpu().numpy().astype(np.float64),
        "doppler_hz": doppler.detach().cpu().numpy().astype(np.float64),
    }


def decode_model_grid(outputs: Mapping[str, torch.Tensor], prf_hz: float) -> dict[str, np.ndarray]:
    """Decode a native neural candidate and its conditional circular Doppler head."""
    if "range_logits" not in outputs or outputs["range_logits"].ndim != 2:
        raise ValueError("range_logits must be a [B,32] tensor")
    if outputs["range_logits"].shape[1] != 32:
        raise ValueError("range_logits must contain all 32 range gates")
    bins = outputs["range_logits"].argmax(dim=-1).detach().cpu().numpy()
    return _decode_selected(outputs, bins, prf_hz)


def decode_model_at_bins(
    outputs: Mapping[str, torch.Tensor], candidate_bins: np.ndarray, prf_hz: float
) -> dict[str, np.ndarray]:
    """Read a neural parameter estimate at externally selected observable range gates."""
    return _decode_selected(outputs, candidate_bins, prf_hz)


def normalized_fit_fraction(samples: np.ndarray, prf_hz: float, fft_factor: int = 4) -> np.ndarray:
    """Return normalized maximum single-tone projection power for each complex CPI."""
    values = np.asarray(samples)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2 or not np.iscomplexobj(values):
        raise ValueError("samples must be a nonempty complex [B,N] array")
    if not np.isfinite(values).all() or not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("samples and PRF must be finite, with positive PRF")
    if not isinstance(fft_factor, int) or fft_factor < 1:
        raise ValueError("fft_factor must be a positive integer")
    pulse_count = values.shape[1]
    spectrum = np.fft.fft(values, n=fft_factor * pulse_count, axis=-1)
    maximum = np.max(np.abs(spectrum) ** 2, axis=-1)
    energy = np.sum(np.abs(values) ** 2, axis=-1)
    fraction = np.divide(
        maximum,
        pulse_count * energy,
        out=np.zeros_like(maximum, dtype=np.float64),
        where=energy > 0.0,
    )
    return np.clip(fraction, 0.0, 1.0)


def candidate_features_at_bins(
    iq: np.ndarray,
    candidate_bins: np.ndarray,
    physics: Mapping[str, Any],
    *,
    prf_hz: float,
    include_profile_ml: bool,
) -> dict[str, np.ndarray]:
    """Measure DPCA Doppler features at observable candidate range gates."""
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("iq must be a complex [B,3,130,32] array")
    if not np.isfinite(values).all():
        raise ValueError("iq must contain finite values")
    bins = np.asarray(candidate_bins)
    if bins.shape != (len(values),) or not np.issubdtype(bins.dtype, np.number):
        raise ValueError("candidate_bins must contain one numeric gate per CPI")
    if not np.isfinite(bins).all() or np.any(bins != np.floor(bins)):
        raise ValueError("candidate bins must be finite integers")
    bins = bins.astype(np.int64)
    if np.any((bins < 0) | (bins >= values.shape[-1])):
        raise ValueError("candidate bin outside the IQ range axis")
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    residual = dpca_difference(values, dict(physics))
    samples = residual[np.arange(len(values)), :, bins]
    pulse_count = samples.shape[-1]
    spectrum = np.fft.fft(samples, n=4 * pulse_count, axis=-1)
    power = np.abs(spectrum) ** 2
    peak_indices = np.argmax(power, axis=-1)
    frequencies = np.fft.fftfreq(4 * pulse_count, d=1.0 / prf_hz)
    maximum = power[np.arange(len(values)), peak_indices]
    energy = np.sum(np.abs(samples) ** 2, axis=-1)
    fit = np.divide(
        maximum,
        pulse_count * energy,
        out=np.zeros_like(maximum, dtype=np.float64),
        where=energy > 0.0,
    )
    result = {
        "fft4_doppler_hz": np.asarray(wrap_frequency(frequencies[peak_indices], prf_hz), dtype=np.float64),
        "fit_fraction": np.clip(fit, 0.0, 1.0),
    }
    if include_profile_ml:
        result["profile_ml_doppler_hz"] = np.asarray(
            estimate_mle_local(samples, prf_hz), dtype=np.float64
        )
    return result


def build_ce_system_predictions(
    iq: np.ndarray,
    ce_outputs: Mapping[str, torch.Tensor],
    physics: Mapping[str, Any],
    *,
    prf_hz: float,
    include_parameter_estimates: bool,
) -> dict[str, dict[str, np.ndarray]]:
    """Construct the B3/CE hybrid chains without passing truth into inference."""
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("iq must be complex [B,3,130,32]")
    _validate_model_outputs(ce_outputs, len(values))
    classical = run_b3(values, prf_hz, dict(physics))
    b3_score = np.atleast_1d(np.asarray(classical.score, dtype=np.float64))
    b3_bins_float = np.atleast_1d(np.asarray(classical.range_cell, dtype=np.float64))
    if np.any(b3_bins_float != np.floor(b3_bins_float)):
        raise ValueError("B3 must return an integer range candidate")
    b3_bins = b3_bins_float.astype(np.int64)
    ce = decode_model_grid(ce_outputs, prf_hz)
    ce_bins = ce["candidate_range_bin"]
    ce_fit = candidate_features_at_bins(
        values, ce_bins, physics, prf_hz=prf_hz, include_profile_ml=include_parameter_estimates
    )
    scores = {
        "A": b3_score,
        "B": ce["presence_probability"],
        "C": ce["presence_probability"],
        "D": b3_score,
        "F": ce["presence_probability"] * ce_fit["fit_fraction"],
    }
    result: dict[str, dict[str, np.ndarray]] = {
        "A": {"score": scores["A"], "candidate_range_bin": b3_bins},
        "B": {"score": scores["B"], "candidate_range_bin": ce_bins},
        "C": {"score": scores["C"], "candidate_range_bin": ce_bins},
        "D": {
            "score": scores["D"],
            "candidate_range_bin": ce_bins,
            "detection_candidate_range_bin": b3_bins,
        },
        "F": {
            "score": scores["F"],
            "candidate_range_bin": ce_bins,
            "fit_fraction": ce_fit["fit_fraction"],
        },
    }
    if include_parameter_estimates:
        b3_features = candidate_features_at_bins(
            values, b3_bins, physics, prf_hz=prf_hz, include_profile_ml=False
        )
        for name, range_values, doppler_values in (
            ("A", b3_bins.astype(np.float64), b3_features["fft4_doppler_hz"]),
            ("B", ce["range_cell"], ce["doppler_hz"]),
            ("C", ce["range_cell"], ce_fit["profile_ml_doppler_hz"]),
            ("D", ce["range_cell"], ce["doppler_hz"]),
            ("F", ce["range_cell"], ce_fit["profile_ml_doppler_hz"]),
        ):
            result[name]["predicted_range_cell"] = np.asarray(range_values, dtype=np.float64)
            result[name]["predicted_doppler_hz"] = np.asarray(doppler_values, dtype=np.float64)
    return result


def compose_system_scores(
    b3_score: np.ndarray,
    ce_probability: np.ndarray,
    j1_probability: np.ndarray,
    fit_fraction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the six frozen system scores from their observable score components."""
    values = {
        "A": np.asarray(b3_score, dtype=np.float64),
        "B": np.asarray(ce_probability, dtype=np.float64),
        "C": np.asarray(ce_probability, dtype=np.float64),
        "D": np.asarray(b3_score, dtype=np.float64),
        "E": np.asarray(j1_probability, dtype=np.float64),
    }
    fit = np.asarray(fit_fraction, dtype=np.float64)
    shapes = {value.shape for value in (*values.values(), fit)}
    if len(shapes) != 1 or next(iter(shapes)) == ():
        raise ValueError("all system score inputs must have the same non-scalar shape")
    if any(not np.isfinite(value).all() for value in (*values.values(), fit)):
        raise FloatingPointError("system score components must be finite")
    if np.any((values["B"] < 0.0) | (values["B"] > 1.0)) or np.any((values["E"] < 0.0) | (values["E"] > 1.0)):
        raise ValueError("neural presence scores must be probabilities")
    if np.any((fit < 0.0) | (fit > 1.0)):
        raise ValueError("fit_fraction must lie in [0,1]")
    values["F"] = values["B"] * fit
    return values


def decision_outputs(
    scores: np.ndarray, predicted_range: np.ndarray, predicted_doppler: np.ndarray, threshold: float
) -> dict[str, np.ndarray]:
    """Apply the frozen >= threshold rule and emit null-valued parameters for abstentions."""
    score = np.asarray(scores, dtype=np.float64)
    ranges = np.asarray(predicted_range, dtype=np.float64)
    dopplers = np.asarray(predicted_doppler, dtype=np.float64)
    if score.ndim != 1 or ranges.shape != score.shape or dopplers.shape != score.shape or score.size == 0:
        raise ValueError("scores and parameter estimates must be aligned nonempty vectors")
    if not np.isfinite(score).all() or not np.isfinite(threshold):
        raise ValueError("scores and threshold must be finite")
    if not np.isfinite(ranges).all() or not np.isfinite(dopplers).all():
        raise ValueError("raw parameter estimates must be finite")
    detected = score >= threshold
    output_range = ranges.copy()
    output_doppler = dopplers.copy()
    output_range[~detected] = np.nan
    output_doppler[~detected] = np.nan
    return {
        "detected": detected,
        "output_range_cell": output_range,
        "output_doppler_hz": output_doppler,
    }
