"""Preflight, physical gates and frozen evaluation for D2-A-R6-J4."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from radardistill.d2a_baselines import dpca_time_shift_s
from radardistill.evaluation.d2a_r6a import build_split_plan, calibrate_threshold
from radardistill.evaluation.d2a_r6e import (
    empirical_auc,
    exact_upper,
    h0_metrics,
    h1_metrics,
    paired_quality_bootstrap,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_j4 import R6J4Student, r6j4_dpca_fft, r6j4_loss
from radardistill.students.r6_joint import R6J1Student, wrap_r6d_doppler
from radardistill.training.d2a_r6d_joint import r6d_loss
from radardistill.training.d2a_r6j4 import (
    ARMS,
    DEVICE_NAME,
    ROOT,
    STAGE,
    RUNS,
    _atomic_json,
    _atomic_torch_save,
    _cache_paths,
    _expected_cache_shapes,
    _git_sha,
    _loss,
    _loss_batch,
    _model,
    _paired_batch,
    audit_r6j4_scene_plans,
    build_r6j4_pair_order,
    build_r6j4_scene_plans,
    checkpoint_owner_matches,
    read_r6j4_yaml,
    r6j4_configs,
    r6j4_presence_labels,
    validate_r6j4_cuda_contract,
)


PREDICTIONS = STAGE / "predictions"
SOURCE_PATHS = (
    "src/radardistill/evaluation/d2a_r6j4.py",
    "src/radardistill/training/d2a_r6j4.py",
    "src/radardistill/students/r6_j4.py",
    "src/radardistill/students/r6_joint.py",
    "src/radardistill/training/d2a_r6d_joint.py",
    "src/radardistill/evaluation/d2a_r6a.py",
    "src/radardistill/evaluation/d2a_r6e.py",
    "src/radardistill/simulation/d2a.py",
    "src/radardistill/d2a_baselines.py",
    "scripts/run_d2a_r6j4.py",
    "experiments/D2-A-E001/config.yaml",
    "experiments/D2-A-R6/R6-A/config.yaml",
)


def ensure_r6j4_gates_passed(g0: Mapping[str, Any], g1: Mapping[str, Any]) -> None:
    if g0.get("status") != "G0_PASS":
        raise RuntimeError("R6-J4 evaluation requires G0_PASS")
    if g1.get("status") != "G1_PASS":
        raise RuntimeError("R6-J4 evaluation requires G1_PASS")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _tracked_worktree_clean() -> bool:
    return not _git("status", "--porcelain", "--untracked-files=no")


def _source_fingerprints(config: Mapping[str, Any]) -> dict[str, str]:
    paths = [ROOT / value for value in SOURCE_PATHS]
    paths.extend((STAGE / "protocol.md", STAGE / "config.yaml"))
    condition_source = ROOT / config["radar"]["condition_source"]
    paths.append(condition_source)
    return {str(path.relative_to(ROOT)): _sha256(path) for path in paths}


def _prior_manifest_scene_ids() -> tuple[set[str], list[str]]:
    known: set[str] = set()
    sources: list[str] = []
    for path in sorted((ROOT / "experiments").rglob("*.csv")):
        if path.is_relative_to(STAGE):
            continue
        try:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames or []
                key = next((name for name in ("scene_id", "base_scene_id", "base_scene_ids") if name in fields), None)
                if key is None:
                    continue
                for row in reader:
                    value = str(row.get(key, ""))
                    if value:
                        known.update(part for part in value.split(";") if part)
            sources.append(str(path.relative_to(ROOT)))
        except (OSError, UnicodeError, csv.Error) as exc:
            raise RuntimeError(f"failed to read prior scene manifest {path}") from exc
    return known, sources


def _prior_registered_scene_ids(config: Mapping[str, Any], condition_config: Mapping[str, Any]) -> tuple[set[str], dict[str, int]]:
    # R6-I's frozen scanner reconstructs R6-D/E/F/G/H registered plans. Older
    # R6-A/B/C scene IDs are read from their retained manifests above.
    from radardistill.evaluation.d2a_r6i import _registered_prior_scene_ids

    r6i_config = read_r6j4_yaml(ROOT / "experiments/D2-A-R6/R6-I/config.yaml")
    return _registered_prior_scene_ids(r6i_config, condition_config)


def _expected_counts(config: Mapping[str, Any]) -> dict[str, int]:
    splits = config["data"]["splits"]
    return {
        "train": int(splits["train"]["paired_scenes"]),
        "validation": int(splits["validation"]["paired_scenes"]),
        "h0_calibration": int(splits["h0_calibration"]["scene_count"]),
        "h0_verification": int(splits["h0_verification"]["scene_count"]),
        "h1_development": int(splits["h1_development"]["scene_count"]),
        "g1": int(config["data"]["g1_test_count"]),
    }


def _cuda_info(config: Mapping[str, Any]) -> dict[str, Any]:
    resources = config["resources"]
    available = bool(torch.cuda.is_available())
    index = torch.cuda.current_device() if available else None
    name = torch.cuda.get_device_name(index) if available else None
    free, total = torch.cuda.mem_get_info(index) if available else (0, 0)
    contract = validate_r6j4_cuda_contract(
        available=available,
        device_name=name,
        device_index=index,
        free_bytes=int(free),
        required_name=str(resources["required_gpu_name"]),
        required_index=0,
        minimum_free_bytes=int(resources["minimum_free_vram_bytes"]),
    )
    return {
        **contract,
        "total_vram_bytes": int(total),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable,
    }


def _require_aisar() -> None:
    env = os.environ.get("CONDA_DEFAULT_ENV", "")
    exe = sys.executable
    if env != "AISAR" and "/envs/AISAR/" not in exe:
        raise RuntimeError(f"R6-J4 requires the AISAR Conda environment, found {env or exe}")


def _write_plan_manifest(name: str, plan: Sequence[Mapping[str, Any]], owner: Mapping[str, Any]) -> None:
    from radardistill.training.d2a_r6j4 import _write_scene_manifest

    path = STAGE / f"registered_scene_plan_{name}.csv"
    owner_path = STAGE / f"registered_scene_plan_{name}.owner.json"
    if path.exists() or owner_path.exists():
        if not (path.is_file() and owner_path.is_file()):
            raise RuntimeError(f"R6-J4 {name} manifest has an incomplete owner binding")
        if json.loads(owner_path.read_text(encoding="utf-8")) != dict(owner):
            raise RuntimeError(f"R6-J4 {name} manifest owner mismatch")
        return
    _write_scene_manifest(path, name, plan)
    _atomic_json(owner_path, dict(owner))


def r6j4_preflight() -> dict[str, Any]:
    """Run G0 on pushed main before any science cache or full training data exists."""
    _require_aisar()
    config, condition_config, _ = r6j4_configs()
    if config["experiment"].get("id") != "D2-A-R6-J4" or config["experiment"].get("status") != "preregistered":
        raise RuntimeError("R6-J4 must remain in preregistered state during G0")
    if bool(config["experiment"].get("final_blind_authorized", True)):
        raise RuntimeError("R6-J4 does not authorize a final blind")
    radar = config["radar"]
    for key, expected in (("channel_count", 3), ("pulse_count", 130), ("range_bin_count", 32), ("condition_count", 12)):
        if int(radar[key]) != expected:
            raise RuntimeError(f"R6-J4 radar invariant {key} changed")
    if float(radar["prf_hz"]) != 1300.0:
        raise RuntimeError("R6-J4 PRF changed")
    branch = _git("branch", "--show-current")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if branch != "main" or head != remote:
        raise RuntimeError("R6-J4 G0 requires the frozen implementation at pushed main HEAD")
    if not _tracked_worktree_clean():
        raise RuntimeError("R6-J4 G0 requires a clean tracked tree")
    source_sha = _source_fingerprints(config)
    plans = build_r6j4_scene_plans(config, condition_config["radar"]["conditions"])
    manifest_ids, manifest_sources = _prior_manifest_scene_ids()
    registered_ids, registered_counts = _prior_registered_scene_ids(config, condition_config)
    prior_ids = manifest_ids | registered_ids
    audit = audit_r6j4_scene_plans(plans, expected_counts=_expected_counts(config), prior_scene_ids=prior_ids)
    free_disk = int(shutil.disk_usage(STAGE).free)
    required_disk = int(config["data"]["minimum_free_disk_bytes_before_generation"])
    if free_disk < required_disk:
        raise RuntimeError(f"R6-J4 requires {required_disk} free disk bytes before cache generation; found {free_disk}")
    gpu = _cuda_info(config)
    owner = {
        "owner": "D2-A-R6-J4-scene-plan",
        "git_sha": head,
        "protocol_sha256": _sha256(STAGE / "protocol.md"),
        "config_sha256": _sha256(STAGE / "config.yaml"),
        "source_sha256": source_sha,
    }
    for split, plan in plans.items():
        _write_plan_manifest(split, plan, {**owner, "split": split, "count": audit["scene_counts"][split]})
    result = {
        "status": "G0_PASS",
        "git_sha": head,
        "branch": branch,
        "remote_main_sha": remote,
        "tracked_worktree_clean": True,
        "protocol_sha256": owner["protocol_sha256"],
        "config_sha256": owner["config_sha256"],
        "source_sha256": source_sha,
        "condition_count": len(condition_config["radar"]["conditions"]),
        "scene_counts": audit["scene_counts"],
        "split_overlap_count": audit["split_overlap_count"],
        "prior_overlap_count": audit["prior_overlap_count"],
        "prior_manifest_count": len(manifest_sources),
        "prior_manifest_scene_id_count": len(manifest_ids),
        "prior_registered_scene_id_count": len(registered_ids),
        "prior_registered_plan_counts": registered_counts,
        "prior_manifest_sources": manifest_sources,
        "gpu": gpu,
        "free_disk_bytes_before_cache": free_disk,
        "minimum_free_disk_bytes": required_disk,
        "train_validation_cache_estimate_bytes": int(config["data"]["estimated_train_validation_cache_bytes"]),
        "full_science_cache_generated": False,
        "final_blind_generated": False,
    }
    _atomic_json(STAGE / "preflight.json", result)
    return result


def _numpy_j4_frontend(iq_ri: np.ndarray, *, delay_s: float, prf_hz: float, nfft: int) -> np.ndarray:
    iq = np.asarray(iq_ri, dtype=np.float64)
    complex_iq = iq[:, :3] + 1j * iq[:, 3:]
    shift = np.exp(2j * np.pi * np.fft.fftfreq(130)[None, :, None] * delay_s * prf_hz)
    second = np.fft.ifft(np.fft.fft(complex_iq[:, 1], axis=1) * shift, axis=1)
    return np.fft.fftshift(np.fft.fft(complex_iq[:, 0] - second, n=nfft, axis=1), axes=1)


def _microbatch_probe(config: Mapping[str, Any], physics: Mapping[str, Any], iq: torch.Tensor,
                      truth: Mapping[str, torch.Tensor], device: torch.device) -> tuple[int, dict[str, Any]]:
    training = config["training"]
    options = [int(training["microbatch_initial"]), *map(int, training["microbatch_candidates_after_oom"])]
    seen: set[int] = set()
    reports: dict[str, Any] = {}
    for size in options:
        if size in seen:
            continue
        seen.add(size)
        reports[str(size)] = {}
        all_fit = True
        for arm in ARMS:
            model = _model(arm, config, physics, int(training["training_seed_by_arm"][arm])).to(device)
            model.zero_grad(set_to_none=True)
            try:
                outputs = model(iq[:size].to(device))
                truth_part = {key: value[:size].to(device) for key, value in truth.items()}
                terms = _loss(arm, outputs, truth_part, config)
                terms["total"].backward()
                grads = [p.grad for p in model.parameters() if p.grad is not None]
                if not grads or any(not torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError(f"R6-J4 G1 {arm} has invalid gradients")
                reports[str(size)][arm] = {
                    "status": "pass",
                    "gradient_parameter_count": len(grads),
                    "gradient_norm": float(torch.linalg.vector_norm(torch.stack([g.float().norm() for g in grads]))),
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            except torch.cuda.OutOfMemoryError:
                all_fit = False
                reports[str(size)][arm] = {"status": "oom"}
                torch.cuda.empty_cache()
            finally:
                del model
                torch.cuda.empty_cache()
        if all_fit:
            return size, reports
    raise RuntimeError("R6-J4 G1 cannot fit a registered microbatch on CUDA")


def _overfit_g1_arm(arm: str, config: Mapping[str, Any], physics: Mapping[str, Any],
                    iq_np: np.ndarray, truth_np: Mapping[str, np.ndarray], *, microbatch: int,
                    device: torch.device) -> dict[str, Any]:
    training = config["training"]
    model = _model(arm, config, physics, int(training["training_seed_by_arm"][arm])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate_initial"]),
                                  weight_decay=float(training["weight_decay"]))
    iq = torch.from_numpy(iq_np)
    truth = {key: torch.from_numpy(value) for key, value in truth_np.items()}
    initial = _loss_batch(model, iq.to(device), {key: value.to(device) for key, value in truth.items()},
                          arm, config, microbatch)["total"]
    updates = []
    for _ in range(int(training["g1_micro_overfit_updates"])):
        terms = __import__("radardistill.training.d2a_r6j4", fromlist=["_training_update"])._training_update(
            model, optimizer, iq, truth, arm=arm, config=config, microbatch=microbatch, device=device
        )
        updates.append(float(terms["total"]))
    last = float(np.mean(updates[-20:]))
    reduction = (initial - last) / max(abs(initial), 1e-12)
    record = {
        "initial_loss": initial,
        "last20_mean_loss": last,
        "relative_reduction": reduction,
        "updates": len(updates),
        "unique_paired_scenes": 16,
        "minimum_reduction": float(training["g1_min_relative_loss_reduction"]),
        "passed": bool(np.isfinite(updates).all() and reduction >= float(training["g1_min_relative_loss_reduction"])),
    }
    del model, optimizer
    torch.cuda.empty_cache()
    return record


def r6j4_g1() -> dict[str, Any]:
    """Run physical parity, CUDA gradient and fixed 16-pair overfit gates."""
    _require_aisar()
    config, condition_config, physics = r6j4_configs()
    preflight_path = STAGE / "preflight.json"
    if not preflight_path.is_file():
        raise RuntimeError("R6-J4 G1 requires a G0 record")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "G0_PASS" or _git_sha() != preflight.get("git_sha"):
        raise RuntimeError("R6-J4 G1 requires G0_PASS on the frozen pushed HEAD")
    gpu = _cuda_info(config)
    device = torch.device("cuda:0")
    plans = build_r6j4_scene_plans(config, condition_config["radar"]["conditions"])
    g1_spec = plans["g1"][0]
    condition = g1_spec["condition"]
    generated = generate_d2a_paired_batch(
        int(g1_spec["count"]), int(g1_spec["seed"]), float(condition["snr_db"]),
        float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
    )
    h0 = np.asarray(generated.iq_h0, dtype=np.complex64)
    h1 = np.asarray(generated.iq_h1, dtype=np.complex64)
    if h0.shape != (16, 3, 130, 32) or h1.shape != h0.shape:
        raise RuntimeError("R6-J4 G1 paired simulator returned an unexpected IQ shape")
    if len(set(str(v) for v in generated.base_scene_id)) != 16:
        raise RuntimeError("R6-J4 G1 generated duplicate scene IDs")
    iq_h1_ri = np.concatenate((h1.real, h1.imag), axis=1).astype(np.float32)
    delay = dpca_time_shift_s(dict(physics))
    reference = _numpy_j4_frontend(iq_h1_ri[:4], delay_s=delay, prf_hz=float(config["radar"]["prf_hz"]),
                                   nfft=int(config["models"]["j4_frontend"]["doppler_fft_bins"]))
    torch_iq = torch.from_numpy(iq_h1_ri[:4]).to(device)
    torch_response = r6j4_dpca_fft(torch_iq, dpca_delay_s=delay, prf_hz=float(config["radar"]["prf_hz"]), nfft=520)
    cuda_response = torch_response.detach().cpu().numpy()
    relative_error = float(np.max(np.abs(cuda_response - reference) / np.maximum(np.abs(reference), 1e-12)))
    if not np.allclose(cuda_response, reference, rtol=1e-4, atol=1e-6):
        raise RuntimeError("R6-J4 fixed CUDA DPCA/FFT differs from complex128 reference")
    phase = torch.exp(torch.tensor(0.73j, dtype=torch.complex64, device=device))
    rotated = torch_iq.clone()
    rotated_complex = torch.complex(rotated[:, :3], rotated[:, 3:]) * phase
    rotated_ri = torch.cat((rotated_complex.real, rotated_complex.imag), dim=1)
    rotated_response = r6j4_dpca_fft(rotated_ri, dpca_delay_s=delay, prf_hz=float(config["radar"]["prf_hz"]), nfft=520)
    phase_power_error = float(torch.max(torch.abs(rotated_response.abs().square() - torch_response.abs().square())).item())
    scaled_response = r6j4_dpca_fft(torch_iq * 2.0, dpca_delay_s=delay, prf_hz=float(config["radar"]["prf_hz"]), nfft=520)
    scale_power_error = float(torch.max(torch.abs(scaled_response.abs().square() / torch_response.abs().square().clamp_min(1e-12) - 4.0)).item())
    del torch_response, rotated_response, scaled_response
    torch.cuda.empty_cache()

    pair_batch, pair_truth = _paired_batch({
        "iq_h0": h0, "iq_h1": h1,
        "truth_range_cell": np.asarray(generated.truth_target_range_cell, dtype=np.float32),
        "truth_doppler_hz": np.asarray(generated.truth_target_doppler_hz, dtype=np.float32),
    }, np.arange(16))
    pair_truth_t = {key: torch.from_numpy(value) for key, value in pair_truth.items()}
    selected_microbatch, resource_probe = _microbatch_probe(
        config, physics, torch.from_numpy(pair_batch), pair_truth_t, device
    )
    repeated = np.tile(np.arange(16, dtype=np.int64), 2)
    overfit_batch, overfit_truth = _paired_batch({
        "iq_h0": h0, "iq_h1": h1,
        "truth_range_cell": np.asarray(generated.truth_target_range_cell, dtype=np.float32),
        "truth_doppler_hz": np.asarray(generated.truth_target_doppler_hz, dtype=np.float32),
    }, repeated)
    overfit = {
        arm: _overfit_g1_arm(arm, config, physics, overfit_batch, overfit_truth,
                             microbatch=selected_microbatch, device=device)
        for arm in ARMS
    }
    passed = all(row["passed"] for row in overfit.values())
    record = {
        "status": "G1_PASS" if passed else "G1_FAIL",
        "git_sha": preflight["git_sha"],
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "gpu": gpu,
        "device": "cuda:0",
        "precision": "FP32",
        "fixed_frontend_precision": {
            "dpca_shift_accumulation": "complex128",
            "doppler_fft": "complex64",
            "learned_readout": "FP32",
        },
        "microbatch_probe": resource_probe,
        "formal_microbatch": selected_microbatch,
        "frontend_complex128_reference_max_relative_error": relative_error,
        "public_phase_power_max_absolute_error": phase_power_error,
        "input_scale_two_power_ratio_max_absolute_error_from_four": scale_power_error,
        "physical_scene_ids": [str(value) for value in generated.base_scene_id],
        "physical_condition": dict(condition),
        "overfit_arms": overfit,
        "formal_train_validation_data_generated": any((STAGE / "cache" / "training" / split / "progress.json").exists() for split in ("train", "validation")),
        "final_blind_generated": False,
    }
    _atomic_json(STAGE / "g1.json", record)
    return record


def _load_checkpoint(arm: str, config: Mapping[str, Any], physics: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    directory = RUNS / arm
    record_path = directory / "training_record.json"
    checkpoint_path = directory / "best_checkpoint.pt"
    owner_path = directory / "owner.json"
    if not all(path.is_file() for path in (record_path, checkpoint_path, owner_path)):
        raise RuntimeError(f"R6-J4 {arm} checkpoint package is incomplete")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if record.get("status") != "complete" or record.get("owner") != owner or not checkpoint_owner_matches(state, owner):
        raise RuntimeError(f"R6-J4 {arm} checkpoint/record owner mismatch")
    model = _model(arm, config, physics, int(config["training"]["training_seed_by_arm"][arm])).to(device)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    return model


def _prediction_arrays() -> dict[str, list[np.ndarray]]:
    return {key: [] for key in ("scene_id", "condition_index", "score", "predicted_range_cell",
                               "predicted_doppler_hz", "truth_range_cell", "truth_doppler_hz")}


def _append_prediction(arrays: dict[str, list[np.ndarray]], *, scene_ids: np.ndarray,
                       condition_index: int, scores: np.ndarray, pred_range: np.ndarray | None = None,
                       pred_doppler: np.ndarray | None = None, truth_range: np.ndarray | None = None,
                       truth_doppler: np.ndarray | None = None) -> None:
    count = len(scene_ids)
    arrays["scene_id"].append(np.asarray(scene_ids).astype("<U48"))
    arrays["condition_index"].append(np.full(count, condition_index, dtype=np.uint8))
    arrays["score"].append(np.asarray(scores, dtype=np.float32))
    for key, value in (("predicted_range_cell", pred_range), ("predicted_doppler_hz", pred_doppler),
                       ("truth_range_cell", truth_range), ("truth_doppler_hz", truth_doppler)):
        if value is not None:
            arrays[key].append(np.asarray(value, dtype=np.float32))


def _finalize_predictions(arrays: Mapping[str, Sequence[np.ndarray]], *, h1: bool) -> dict[str, np.ndarray]:
    keys = ["scene_id", "condition_index", "score"]
    if h1:
        keys += ["predicted_range_cell", "predicted_doppler_hz", "truth_range_cell", "truth_doppler_hz"]
    return {key: np.concatenate(arrays[key]) for key in keys}


def _predict_model_split(model: torch.nn.Module, arm: str, split: str, plan: Sequence[Mapping[str, Any]],
                         physics: Mapping[str, Any], *, device: torch.device, microbatch: int,
                         prf_hz: float) -> dict[str, np.ndarray]:
    from radardistill.distillation.task_losses import decode_task_outputs

    h1 = split == "h1_development"
    arrays = _prediction_arrays()
    model.eval()
    with torch.inference_mode():
        for spec in plan:
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            iq = np.asarray(generated.iq_h1 if h1 else generated.iq_h0, dtype=np.complex64)
            ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
            score_parts: list[np.ndarray] = []
            range_parts: list[np.ndarray] = []
            doppler_parts: list[np.ndarray] = []
            for start in range(0, len(ri), microbatch):
                tensor = torch.from_numpy(np.ascontiguousarray(ri[start:start + microbatch])).to(device)
                output = model(tensor)
                if arm == "j1_joint":
                    decoded = decode_task_outputs(output, prf_hz)
                    score_parts.append(output["presence_logit"].cpu().numpy())
                    if h1:
                        range_parts.append(decoded["range_cell"].cpu().numpy())
                        doppler_parts.append(decoded["doppler_hz"].cpu().numpy())
                else:
                    score_parts.append(output["presence_logit"].cpu().numpy())
                    if h1:
                        vector = output["doppler_vector_pooled"]
                        doppler = torch.atan2(vector[:, 1], vector[:, 0]) * (prf_hz / (2.0 * math.pi))
                        doppler = wrap_r6d_doppler(doppler, prf_hz)
                        range_parts.append(output["range_cell_estimate"].cpu().numpy())
                        doppler_parts.append(doppler.cpu().numpy())
            _append_prediction(
                arrays, scene_ids=np.asarray(generated.base_scene_id), condition_index=int(spec["condition_index"]),
                scores=np.concatenate(score_parts),
                pred_range=np.concatenate(range_parts) if h1 else None,
                pred_doppler=np.concatenate(doppler_parts) if h1 else None,
                truth_range=np.asarray(generated.truth_target_range_cell, dtype=np.float32) if h1 else None,
                truth_doppler=np.asarray(generated.truth_target_doppler_hz, dtype=np.float32) if h1 else None,
            )
    result = _finalize_predictions(arrays, h1=h1)
    if not np.isfinite(result["score"]).all():
        raise FloatingPointError(f"R6-J4 {arm} {split} scores are nonfinite")
    return result


def _fixed_peak_batch(iq: np.ndarray, *, delay_s: float, prf_hz: float, nfft: int,
                      device: torch.device) -> dict[str, np.ndarray]:
    ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32, copy=False)
    response = r6j4_dpca_fft(torch.from_numpy(ri).to(device), dpca_delay_s=delay_s, prf_hz=prf_hz, nfft=nfft)
    power = response.abs().square()
    batch, _, ranges = power.shape
    flat = power.reshape(batch, -1).argmax(dim=1)
    doppler_index = torch.div(flat, ranges, rounding_mode="floor")
    range_index = flat % ranges
    rows = torch.arange(batch, device=device)
    center = torch.log(power[rows, doppler_index, range_index].clamp_min(1e-20))
    left = torch.log(power[rows, (doppler_index - 1) % nfft, range_index].clamp_min(1e-20))
    right = torch.log(power[rows, (doppler_index + 1) % nfft, range_index].clamp_min(1e-20))
    denominator = left - 2.0 * center + right
    delta = torch.where(denominator.abs() > 1e-12, 0.5 * (left - right) / denominator, torch.zeros_like(center)).clamp(-0.5, 0.5)
    frequency = torch.fft.fftshift(torch.fft.fftfreq(nfft, d=1.0 / prf_hz, device=device))
    doppler = wrap_r6d_doppler(frequency[doppler_index] + delta * (prf_hz / nfft), prf_hz)
    return {
        "score": power.reshape(batch, -1).max(dim=1).values.cpu().numpy(),
        "range": range_index.float().cpu().numpy(),
        "doppler": doppler.cpu().numpy(),
    }


def _predict_fixed_split(split: str, plan: Sequence[Mapping[str, Any]], physics: Mapping[str, Any], *,
                         device: torch.device, microbatch: int, prf_hz: float,
                         delay_s: float) -> dict[str, dict[str, np.ndarray]]:
    is_h1 = split == "h1_development"
    method_arrays = {"fixed_dpca_130_fft_peak": _prediction_arrays(),
                     "fixed_dpca_520_fft_peak": _prediction_arrays()}
    with torch.inference_mode():
        for spec in plan:
            condition = spec["condition"]
            generated = generate_d2a_paired_batch(
                int(spec["count"]), int(spec["seed"]), float(condition["snr_db"]),
                float(condition["cnr_db"]), dict(physics), scenario_id=str(condition["scenario_id"]),
            )
            iq = np.asarray(generated.iq_h1 if is_h1 else generated.iq_h0, dtype=np.complex64)
            for start in range(0, len(iq), microbatch):
                stop = min(len(iq), start + microbatch)
                sample = np.ascontiguousarray(iq[start:stop])
                for name, nfft in (("fixed_dpca_130_fft_peak", 130), ("fixed_dpca_520_fft_peak", 520)):
                    prediction = _fixed_peak_batch(sample, delay_s=delay_s, prf_hz=prf_hz, nfft=nfft, device=device)
                    sl = slice(start, stop)
                    _append_prediction(
                        method_arrays[name], scene_ids=np.asarray(generated.base_scene_id)[sl],
                        condition_index=int(spec["condition_index"]), scores=prediction["score"],
                        pred_range=prediction["range"] if is_h1 else None,
                        pred_doppler=prediction["doppler"] if is_h1 else None,
                        truth_range=np.asarray(generated.truth_target_range_cell)[sl] if is_h1 else None,
                        truth_doppler=np.asarray(generated.truth_target_doppler_hz)[sl] if is_h1 else None,
                    )
    result = {name: _finalize_predictions(arrays, h1=is_h1) for name, arrays in method_arrays.items()}
    # B3 is the same frozen DPCA + native 130-bin maximum-power detector.
    result["b3_130_fft"] = {
        key: result["fixed_dpca_130_fft_peak"][key].copy()
        for key in ("scene_id", "condition_index", "score")
    }
    return result


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def _paired_joint_bootstrap(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray], *,
    threshold_first: float, threshold_second: float, conditions: Sequence[Mapping[str, Any]],
    replicates: int, seed: int, confidence: float = 0.95,
) -> dict[str, Any]:
    from radardistill.evaluation.d2a_r6a import wrapped_error

    ids = np.asarray(first["scene_id"]).astype(str)
    if not np.array_equal(ids, np.asarray(second["scene_id"]).astype(str)):
        raise ValueError("R6-J4 paired predictions have different scene IDs/order")
    cond_a = np.asarray(first["condition_index"], dtype=np.int64)
    cond_b = np.asarray(second["condition_index"], dtype=np.int64)
    if not np.array_equal(cond_a, cond_b):
        raise ValueError("R6-J4 paired condition assignments differ")
    course = np.asarray([str(item["course"]) for item in conditions])
    primary = np.isin(course[cond_a], ("C2", "C3"))
    first_detected = np.asarray(first["score"]) >= threshold_first
    second_detected = np.asarray(second["score"]) >= threshold_second
    r1 = np.asarray(first["predicted_range_cell"]) - np.asarray(first["truth_range_cell"])
    r2 = np.asarray(second["predicted_range_cell"]) - np.asarray(second["truth_range_cell"])
    d1 = wrapped_error(first["predicted_doppler_hz"], first["truth_doppler_hz"], 1300.0)
    d2 = wrapped_error(second["predicted_doppler_hz"], second["truth_doppler_hz"], 1300.0)
    j1 = first_detected & (np.abs(r1) <= 1.0) & (np.abs(d1) <= 20.0)
    j2 = second_detected & (np.abs(r2) <= 1.0) & (np.abs(d2) <= 20.0)
    primary_indices = np.flatnonzero(primary)
    rng = np.random.default_rng(seed)
    samples_joint = np.empty(replicates)
    samples_pd = np.empty(replicates)
    samples_range = np.empty(replicates)
    samples_doppler = np.empty(replicates)
    for index in range(replicates):
        selected = rng.choice(primary_indices, size=len(primary_indices), replace=True)
        all_ids = rng.integers(0, len(ids), size=len(ids))
        samples_joint[index] = np.mean(j2[selected].astype(float) - j1[selected].astype(float))
        samples_pd[index] = np.mean(second_detected[all_ids].astype(float) - first_detected[all_ids].astype(float))
        samples_range[index] = np.sqrt(np.mean(r2[all_ids] ** 2) / max(np.mean(r1[all_ids] ** 2), 1e-24))
        samples_doppler[index] = np.sqrt(np.mean(d2[all_ids] ** 2) / max(np.mean(d1[all_ids] ** 2), 1e-24))
    tail = (1.0 - confidence) / 2.0
    ci = lambda values: [float(v) for v in np.quantile(values, [tail, 1.0 - tail])]
    estimate = float(np.mean(j2[primary].astype(float) - j1[primary].astype(float)))
    joint_ci = ci(samples_joint)
    pd_ci, range_ci, doppler_ci = ci(samples_pd), ci(samples_range), ci(samples_doppler)
    return {
        "primary": "identifiable_C2_C3_paired_joint_success",
        "n_paired_scenes": int(primary.sum()),
        "joint_difference": estimate,
        "joint_ci95": joint_ci,
        "meaningful_gain_minimum": 0.02,
        "primary_pass": bool(estimate >= 0.02 and joint_ci[0] > 0),
        "overall_pd_difference_ci95": pd_ci,
        "pd_protection_pass": bool(pd_ci[0] >= -0.02),
        "range_rmse_ratio_ci95": range_ci,
        "range_protection_pass": bool(range_ci[1] <= 1.05),
        "doppler_rmse_ratio_ci95": doppler_ci,
        "doppler_protection_pass": bool(doppler_ci[1] <= 1.05),
        "bootstrap_replicates": int(replicates),
        "seed": int(seed),
    }


def _metrics_for_predictions(
    predictions: Mapping[str, Mapping[str, np.ndarray]], *,
    conditions: Sequence[Mapping[str, Any]], config: Mapping[str, Any],
) -> dict[str, Any]:
    methods = list(predictions["h0_calibration"])
    metrics: dict[str, Any] = {"methods": {}, "contrasts": {}, "baselines": {}}
    thresholds: dict[str, float] = {}
    for method in methods:
        cal = predictions["h0_calibration"][method]
        ver = predictions["h0_verification"][method]
        h1 = predictions["h1_development"][method]
        threshold = float(np.quantile(cal["score"], 0.95, method="higher"))
        thresholds[method] = threshold
        h0_summary = h0_metrics(cal, ver, threshold=threshold, conditions=conditions)
        if "predicted_range_cell" in h1:
            h1_summary = h1_metrics(h1, threshold=threshold, conditions=conditions,
                                    h0_verification=ver, prf_hz=float(config["radar"]["prf_hz"]))
        else:
            detected = np.asarray(h1["score"]) >= threshold
            cond = np.asarray(h1["condition_index"], dtype=np.int64)
            courses = np.asarray([str(value["course"]) for value in conditions])[cond]
            h1_summary = {
                "n": len(detected), "pd": float(detected.mean()),
                "pd_by_condition": {str(i): float(detected[cond == i].mean()) for i in range(len(conditions))},
                "roc_auc_vs_h0_verification": float(empirical_auc(ver["score"], h1["score"])),
                "by_group": {group: {"n": int((courses == group).sum()), "pd": float(detected[courses == group].mean())}
                             for group in ("C0", "C1", "C2", "C3")},
                "parameter_metrics": "not provided by the B3 detector",
            }
        metrics["methods"][method] = {
            "threshold": threshold,
            "h0": h0_summary,
            "h1": h1_summary,
        }
    j1 = predictions["h1_development"]["j1_joint"]
    j4 = predictions["h1_development"]["j4_joint"]
    metrics["contrasts"]["j4_minus_j1"] = _paired_joint_bootstrap(
        j1, j4, threshold_first=thresholds["j1_joint"], threshold_second=thresholds["j4_joint"],
        conditions=conditions, replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
        seed=int(config["evaluation"]["paired_bootstrap_seed"]), confidence=float(config["evaluation"]["ci_level"]),
    )
    fixed130 = predictions["h1_development"]["fixed_dpca_130_fft_peak"]
    fixed520 = predictions["h1_development"]["fixed_dpca_520_fft_peak"]
    metrics["contrasts"]["fixed520_minus_fixed130"] = _paired_joint_bootstrap(
        fixed130, fixed520,
        threshold_first=thresholds["fixed_dpca_130_fft_peak"],
        threshold_second=thresholds["fixed_dpca_520_fft_peak"],
        conditions=conditions,
        replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
        seed=int(config["evaluation"]["paired_bootstrap_seed"]) + 1,
        confidence=float(config["evaluation"]["ci_level"]),
    )
    metrics["contrasts"]["j4_minus_fixed520"] = _paired_joint_bootstrap(
        predictions["h1_development"]["fixed_dpca_520_fft_peak"],
        predictions["h1_development"]["j4_joint"],
        threshold_first=thresholds["fixed_dpca_520_fft_peak"],
        threshold_second=thresholds["j4_joint"],
        conditions=conditions,
        replicates=int(config["evaluation"]["paired_bootstrap_replicates"]),
        seed=int(config["evaluation"]["paired_bootstrap_seed"]) + 2,
        confidence=float(config["evaluation"]["ci_level"]),
    )
    metrics["baselines"]["b3_equals_fixed_dpca_130_detector"] = bool(
        np.array_equal(predictions["h0_calibration"]["b3_130_fft"]["score"],
                       predictions["h0_calibration"]["fixed_dpca_130_fft_peak"]["score"])
        and np.array_equal(predictions["h1_development"]["b3_130_fft"]["scene_id"],
                           predictions["h1_development"]["fixed_dpca_130_fft_peak"]["scene_id"])
    )
    j4_c3_pass = bool(metrics["methods"]["j4_joint"]["h0"]["c3_six_cells_familywise_pass"])
    metrics["screening"] = {
        "j4_primary_and_task_protections_pass": bool(
            metrics["contrasts"]["j4_minus_j1"]["primary_pass"]
            and metrics["contrasts"]["j4_minus_j1"]["pd_protection_pass"]
            and metrics["contrasts"]["j4_minus_j1"]["range_protection_pass"]
            and metrics["contrasts"]["j4_minus_j1"]["doppler_protection_pass"]
        ),
        "j4_c3_six_cell_pfa_pass": j4_c3_pass,
        "eligible_for_new_confirmation_proposal": bool(
            metrics["contrasts"]["j4_minus_j1"]["primary_pass"]
            and metrics["contrasts"]["j4_minus_j1"]["pd_protection_pass"]
            and metrics["contrasts"]["j4_minus_j1"]["range_protection_pass"]
            and metrics["contrasts"]["j4_minus_j1"]["doppler_protection_pass"]
            and j4_c3_pass
        ),
    }
    metrics["thresholds"] = thresholds
    metrics["final_blind_generated"] = False
    return metrics


def r6j4_evaluate() -> dict[str, Any]:
    """Run the registered single development evaluation exactly once per split/method."""
    _require_aisar()
    config, condition_config, physics = r6j4_configs()
    ensure_r6j4_gates_passed(
        json.loads((STAGE / "preflight.json").read_text(encoding="utf-8")),
        json.loads((STAGE / "g1.json").read_text(encoding="utf-8")),
    )
    preflight = json.loads((STAGE / "preflight.json").read_text(encoding="utf-8"))
    if _git_sha() != preflight["git_sha"]:
        raise RuntimeError("R6-J4 evaluation code changed after G0")
    summary = json.loads((STAGE / "training_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or set(summary.get("arms", {})) != set(ARMS):
        raise RuntimeError("R6-J4 evaluation requires both matched arms to complete the frozen budget")
    plans = build_r6j4_scene_plans(config, condition_config["radar"]["conditions"])
    device = torch.device("cuda:0")
    gpu = _cuda_info(config)
    microbatch = int(json.loads((STAGE / "g1.json").read_text(encoding="utf-8"))["formal_microbatch"])
    delay = dpca_time_shift_s(dict(physics))
    methods: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    # Deterministic physical baselines use the exact same scene IDs and cached prediction artifacts.
    for split in ("h0_calibration", "h0_verification", "h1_development"):
        fixed = _predict_fixed_split(split, plans[split], physics, device=device, microbatch=microbatch,
                                     prf_hz=float(config["radar"]["prf_hz"]), delay_s=delay)
        for method, arrays in fixed.items():
            methods.setdefault(method, {})[split] = arrays
            _write_npz(PREDICTIONS / f"{method}_{split}.npz", arrays)
    torch.cuda.empty_cache()
    for arm in ARMS:
        model = _load_checkpoint(arm, config, physics, device)
        for split in ("h0_calibration", "h0_verification", "h1_development"):
            result = _predict_model_split(model, arm, split, plans[split], physics, device=device,
                                          microbatch=microbatch, prf_hz=float(config["radar"]["prf_hz"]))
            methods.setdefault(arm, {})[split] = result
            _write_npz(PREDICTIONS / f"{arm}_{split}.npz", result)
        del model
        torch.cuda.empty_cache()
    ordered = {split: {method: methods[method][split] for method in methods}
               for split in ("h0_calibration", "h0_verification", "h1_development")}
    metrics = _metrics_for_predictions(ordered, conditions=condition_config["radar"]["conditions"], config=config)
    metrics.update({
        "status": "evaluated",
        "git_sha": preflight["git_sha"],
        "protocol_sha256": preflight["protocol_sha256"],
        "config_sha256": preflight["config_sha256"],
        "gpu": gpu,
        "device": "cuda:0",
        "precision": "FP32",
        "microbatch": microbatch,
        "training_arms": {arm: summary["arms"][arm] for arm in ARMS},
        "registered_scene_counts": preflight["scene_counts"],
        "prediction_files": [str(path.relative_to(ROOT)) for path in sorted(PREDICTIONS.glob("*.npz"))],
    })
    _atomic_json(STAGE / "metrics.json", metrics)
    return metrics


def _plot_results(metrics: Mapping[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = STAGE / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    h1 = metrics["methods"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    pfa_labels = ["B3/130", "DPCA 520", "J1-J", "J4"]
    pfa_names = ["b3_130_fft", "fixed_dpca_520_fft_peak", "j1_joint", "j4_joint"]
    joint_labels = ["DPCA 130", "DPCA 520", "J1-J", "J4"]
    joint_names = ["fixed_dpca_130_fft_peak", "fixed_dpca_520_fft_peak", "j1_joint", "j4_joint"]
    colors = ["#777777", "#4682b4", "#52796f", "#d08c60"]
    c3_pfa = [h1[name]["h0"]["c3_pooled"]["pfa"] for name in pfa_names]
    joint = [h1[name]["h1"]["by_group"]["C2_C3"]["joint_success"] for name in joint_names]
    axes[0].bar(pfa_labels, c3_pfa, color=colors)
    axes[0].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylim(0, max(0.11, max(c3_pfa) * 1.10))
    axes[0].set_ylabel("C3 pooled H0 Pfa")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(joint_labels, joint, color=colors)
    axes[1].set_ylabel("C2+C3 joint success")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(figure_dir / "r6j4_metrics.png", dpi=160)
    plt.close(fig)


def r6j4_analyze() -> dict[str, Any]:
    metrics_path = STAGE / "metrics.json"
    if not metrics_path.is_file():
        raise RuntimeError("R6-J4 analyze requires metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    _plot_results(metrics)
    contrast = metrics["contrasts"]["j4_minus_j1"]
    grid = metrics["contrasts"]["fixed520_minus_fixed130"]
    readout = metrics["contrasts"]["j4_minus_fixed520"]
    screening = metrics["screening"]
    lines = [
        "# D2-A-R6-J4 分析",
        "",
        "**状态：** 单 seed development pilot 已完成。**最终盲测：** 未生成。",
        "**协议：** `experiments/D2-A-R6/R6-J4/protocol.md` v1.0；配置 SHA：`" + str(metrics["config_sha256"]) + "`。",
        "",
        "## 预注册主要对比",
        "",
        f"J4−J1-J 的 C2+C3 joint 差为 `{contrast['joint_difference']:+.5f}`，paired 95% CI `{contrast['joint_ci95']}`；主门：`{contrast['primary_pass']}`。",
        f"保护门：Pd `{contrast['pd_protection_pass']}`，range `{contrast['range_protection_pass']}`，Doppler `{contrast['doppler_protection_pass']}`。",
        f"固定 520−130 网格 joint 差 `{grid['joint_difference']:+.5f}`，95% CI `{grid['joint_ci95']}`；J4−固定 520 峰值 joint 差 `{readout['joint_difference']:+.5f}`，95% CI `{readout['joint_ci95']}`。",
        f"C3 六格 familywise Pfa 门：`{screening['j4_c3_six_cell_pfa_pass']}`；进入新确认实验候选资格：`{screening['eligible_for_new_confirmation_proposal']}`。",
        "",
        "## 评价边界",
        "",
        "固定 DPCA 130/520 峰值分别保留物理频率网格和学习读出的参照；B3 与 130 点固定 DPCA 峰值在本配置下为同一检测公式，报告时不当作独立证据。J4 与 J1-J 的差异是完整表示/学习系统对比，不单独归因于某个物理操作。零填充只加密频率网格，不增加观测信息。",
        "",
        "所有Pfa门均需看C3六个独立条件格familywise exact上界；总体或pooled C3通过不能替代逐格保护。单训练seed的development只作架构筛选，不代表seed稳定性、最终泛化、实测性能或实时优势。",
        "",
        "## 决策",
        "",
        "只有在主要收益门与Pd、距离、Doppler、C3六格Pfa保护同时满足时才可提名新的确认实验；否则保留所有负结果，按冻结停止规则停止当前J4参数化或仅记录局部收益。",
        "",
    ]
    (STAGE / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
    return {"analysis": str((STAGE / "analysis.md").relative_to(ROOT)), "figure": str((STAGE / "figures/r6j4_metrics.png").relative_to(ROOT))}
