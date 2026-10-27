#!/usr/bin/env python3
"""Run the preregistered Small/Medium width-only R5-A capacity comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_d2a_e003 as e003_runner
import run_d2a_r5 as r5_runner
from radardistill.distillation.task_losses import task_loss
from radardistill.students.task_student import TaskStudent
from radardistill.training.d2a_e002 import make_paired_examples
from radardistill.training.d2a_r5 import build_r5_capacity_specs

CAPACITY_ROOT = ROOT / "experiments/D2-A-R5/R5-A/capacity"
CAPACITY_CONFIG_PATH = CAPACITY_ROOT / "config.yaml"
CAPACITY_SOURCE_FILES = (
    "experiments/D2-A-R5/R5-A/capacity/config.yaml",
    "scripts/run_d2a_r5_capacity.py",
    "src/radardistill/training/d2a_r5.py",
    "tests/test_d2a_r5_training.py",
)


def _read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    r5_runner._write_json(path, value)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _microbatch_candidates(training: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the frozen descending OOM fallback sequence without duplicates."""
    effective_batch = int(training["effective_batch_size"])
    raw = [int(training["initial_microbatch_size"])]
    raw.extend(int(value) for value in training["oom_microbatch_fallbacks"])
    candidates = tuple(dict.fromkeys(raw))
    if effective_batch <= 0 or not candidates:
        raise ValueError("effective batch and microbatch candidates must be positive")
    for value in candidates:
        if value <= 0 or value % 2 or effective_batch % value:
            raise ValueError("each microbatch must be a positive even divisor of the effective batch")
    if any(left <= right for left, right in zip(candidates, candidates[1:])):
        raise ValueError("microbatch OOM fallbacks must strictly decrease")
    return candidates


def _capacity_summary(comparison: Mapping[str, Any]) -> dict[str, Any]:
    """Rename inherited E003 S2−S0 statistics to the actual Medium−Small contrast."""
    required = {"G3_task_learning", "G4_task_protection", "G4_primary_C2_C3_joint_gain"}
    missing = required - set(comparison)
    if missing:
        raise KeyError(f"E003 comparison lacks capacity-gate fields: {sorted(missing)}")
    basic = comparison["G3_task_learning"]
    protection = comparison["G4_task_protection"]
    primary = comparison["G4_primary_C2_C3_joint_gain"]
    if not isinstance(basic, Mapping) or not all(key in basic for key in ("S0", "S2")):
        raise KeyError("E003 basic-learning screen must contain S0 and S2 results")
    gain = {
        "point_difference": float(primary["point_difference_s2_minus_s0"]),
        "ci95": [float(value) for value in primary["paired_scene_bootstrap_ci95"]],
        "passed": bool(primary["passed"]),
    }
    basic_by_model = {"Small": basic["S0"], "Medium": basic["S2"]}
    candidate_eligible = bool(
        all(bool(result.get("passed")) for result in basic_by_model.values())
        and bool(protection.get("passed"))
        and gain["passed"]
    )
    return {
        "baseline": "Small",
        "candidate": "Medium",
        "basic_learning": basic_by_model,
        "task_protection": protection,
        "primary_gain_medium_minus_small": gain,
        "candidate_eligible": candidate_eligible,
        "screen": "both models pass basic-learning; Medium passes inherited task protection and preregistered primary gain",
    }


def _require_capacity_frozen_main() -> str:
    # The base runner checks all already-frozen R5/E003 source and protocol inputs.
    r5_runner._require_frozen_main()
    for path in CAPACITY_SOURCE_FILES:
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        if subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", path], cwd=ROOT
        ).returncode != 0:
            raise RuntimeError(f"capacity source is not committed cleanly: {path}")
    head = _git("rev-parse", "HEAD")
    remote = _git("rev-parse", "origin/main")
    if head != remote:
        raise RuntimeError("R5 capacity requires the frozen source and config pushed to origin/main")
    return head


def _load_capacity_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    capacity_config = _read_yaml(CAPACITY_CONFIG_PATH)
    r5_config, e003_config, physics = r5_runner._load_configs()
    if (
        capacity_config.get("experiment", {}).get("status") != "preregistered"
        or capacity_config["experiment"].get("final_blind_generation_authorized") is not False
        or capacity_config["experiment"].get("distillation_authorized") is not False
    ):
        raise ValueError("capacity config must remain preregistered with blind/KD disabled")
    if capacity_config["training"].get("device") != "cuda:0" or capacity_config["training"].get("cpu_fallback") != "forbidden":
        raise ValueError("capacity training must remain CUDA-only")
    if capacity_config["training"].get("target_updates") != 12800:
        raise ValueError("capacity comparison must match the frozen 12,800-update R5-A budget")
    if capacity_config["models"].get("width") != {"Small": 32, "Medium": 96}:
        raise ValueError("capacity comparison widths must remain Small=32 and Medium=96")
    if capacity_config["models"].get("vary_only") != "trunk_width":
        raise ValueError("capacity comparison may vary trunk width only")
    expected_parameters = r5_config["capacity"]["parameter_count_expected"]
    if capacity_config["models"].get("expected_parameters") != expected_parameters:
        raise ValueError("capacity config parameter counts must match the frozen R5-A specification")
    training = capacity_config["training"]
    base_training = r5_config["training"]
    for field in (
        "optimizer",
        "learning_rate_initial",
        "learning_rate_final",
        "extension_learning_rate_final",
        "paired_scenes_per_update",
        "effective_batch_size",
        "validation_interval_updates",
        "precision",
        "device",
        "cpu_fallback",
        "amp",
        "simultaneous_models_on_gpu",
        "initial_microbatch_size",
        "oom_microbatch_fallbacks",
        "preserve_effective_batch_on_oom",
    ):
        if training.get(field) != base_training.get(field):
            raise ValueError(f"capacity config may not change frozen R5-A training setting: {field}")
    if training.get("supervision") != base_training.get("supervision"):
        raise ValueError("capacity comparison must remain truth-only E1")
    evaluation = capacity_config["evaluation"]
    if (
        evaluation.get("target_pfa_cpi") != e003_config["detection"]["target_pfa_cpi"]
        or evaluation.get("threshold_comparator") != ">="
        or evaluation.get("paired_bootstrap_replicates") != e003_config["statistics"]["bootstrap_replicates"]
        or evaluation.get("paired_bootstrap_unit") != "base_scene_id"
        or evaluation.get("calibrate_each_model_separately") is not True
    ):
        raise ValueError("capacity evaluation must inherit the frozen E003 calibration/bootstrap definitions")
    source = (CAPACITY_ROOT / capacity_config["data"]["source_config"]).resolve()
    cache = (CAPACITY_ROOT / capacity_config["data"]["cache_root"]).resolve()
    if source != r5_runner.E003_CONFIG_PATH.resolve() or cache != (ROOT / r5_config["data"]["cache_root"]).resolve():
        raise ValueError("capacity config must reuse the frozen E003 protocol and R5 cache")
    return capacity_config, r5_config, e003_config, physics


def _capacity_preflight_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    current_sha = _require_capacity_frozen_main()
    base_preflight, base_training, base_evaluation_sha = r5_runner._require_posttraining_provenance(
        require_cuda=True
    )
    capacity_config, r5_config, e003_config, physics = _load_capacity_inputs()
    small_record = base_training["arms"]["S0"]
    extension = base_training["continuation_decision"]
    specs = build_r5_capacity_specs(
        r5_config,
        extension_decision=extension,
        small_training_record=small_record,
    )
    expected = capacity_config["training"]
    if (
        int(capacity_config["experiment"]["training_seed"]) != specs["seed"]
        or int(expected["target_updates"]) != specs["target_updates"]
        or int(expected["effective_batch_size"]) != int(r5_config["training"]["effective_batch_size"])
        or expected["supervision"] != specs["supervision"]
    ):
        raise ValueError("capacity training config differs from the frozen R5-A Small baseline/schedule")
    checkpoint = ROOT / specs["small_checkpoint_path"]
    if not checkpoint.is_file():
        raise FileNotFoundError(f"selected R5-A Small checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("experiment") != "D2-A-R5"
        or payload.get("subphase") != "R5-A"
        or payload.get("arm") != "S0"
        or int(payload.get("width", -1)) != specs["arms"]["Small"]["width"]
        or int(payload.get("global_update", -1)) != int(small_record["selected_update"])
        or payload.get("source_git_sha") != base_training["git_sha"]
    ):
        raise ValueError("selected Small checkpoint does not match the completed R5-A training record")
    return (
        capacity_config,
        r5_config,
        e003_config,
        physics,
        current_sha,
        {
            "base_preflight": base_preflight,
            "base_training": base_training,
            "base_evaluation_sha": base_evaluation_sha,
            "specs": specs,
            "small_record": small_record,
        },
    )


def run_preflight() -> dict[str, Any]:
    inputs = _capacity_preflight_inputs()
    capacity_config, _, _, _, current_sha, context = inputs
    path = CAPACITY_ROOT / "preflight.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("capacity_git_sha") != current_sha:
            raise RuntimeError("capacity preflight belongs to another source commit; preserve and review it")
        return existing
    result = {
        "experiment": "D2-A-R5/R5-A-capacity",
        "capacity_git_sha": current_sha,
        "base_training_git_sha": context["base_training"]["git_sha"],
        "base_evaluation_git_sha": context["base_evaluation_sha"],
        "protocol_version": float(capacity_config["experiment"]["protocol_version"]),
        "specs": context["specs"],
        "environment": r5_runner._environment_record(current_sha),
        "passed": True,
        "same_frozen_e003_train_validation_ids": True,
        "final_blind_generated": False,
        "distillation_run": False,
    }
    _write_json(path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def _require_capacity_preflight(*, require_cuda: bool = True) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    inputs = _capacity_preflight_inputs()
    capacity_config, _, _, _, current_sha, context = inputs
    path = CAPACITY_ROOT / "preflight.json"
    if not path.is_file():
        raise RuntimeError("run capacity --phase preflight before capacity sanity, training, or evaluation")
    preflight = json.loads(path.read_text(encoding="utf-8"))
    if preflight.get("passed") is not True or preflight.get("capacity_git_sha") != current_sha:
        raise RuntimeError("capacity preflight is stale or did not pass")
    if require_cuda:
        r5_runner._require_cuda()
    return (*inputs[:4], current_sha, {**context, "capacity_preflight": preflight})


def run_overfit_sanity() -> dict[str, Any]:
    capacity_config, r5_config, e003_config, _, commit, context = _require_capacity_preflight()
    cache_root = ROOT / r5_config["data"]["cache_root"]
    train = r5_runner._load_split(cache_root, "train")
    sanity_config = r5_config["gates"]["G1_overfit"]
    pair_ids = np.arange(int(sanity_config["paired_scenes"]), dtype=np.int64)
    batch = make_paired_examples(train, pair_ids, include_teacher=False)
    x_cpu = torch.from_numpy(batch["iq_ri"])
    truth_cpu = {
        key: torch.from_numpy(batch[key])
        for key in ("truth_presence", "truth_range_cell", "truth_doppler_hz")
    }
    loss_config = r5_runner._training_loss_config(e003_config)
    device = r5_runner._require_cuda()
    seed = int(capacity_config["experiment"]["training_seed"])
    initial_state = r5_runner._initial_student_state(seed, width=96)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = r5_runner._build_model("S0", initial_state, width=96, seed=seed).to(device)
    x = x_cpu.to(device)
    truth = {key: value.to(device) for key, value in truth_cpu.items()}
    model.eval()
    initial_output = model(x, compute_spectrum=False)
    initial_terms = task_loss(initial_output, truth, loss_config, teacher_weight=0.0)
    task_modules = {"range_cell": model.range_head, "doppler": model.doppler_head}
    gradient_norms: dict[str, float] = {}
    for name, module in task_modules.items():
        gradients = torch.autograd.grad(
            initial_terms[name], tuple(module.parameters()), retain_graph=True, allow_unused=True
        )
        norms = [gradient.detach().double().square().sum() for gradient in gradients if gradient is not None]
        gradient_norms[name] = float(torch.sqrt(torch.stack(norms).sum()).item()) if norms else 0.0
    gradients_passed = all(np.isfinite(value) and value > 0 for value in gradient_norms.values())
    initial_loss = float(initial_terms["task_total"].detach().item())
    del initial_output, initial_terms
    torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(e003_config["training"]["learning_rate_initial"]),
        weight_decay=float(e003_config["training"]["weight_decay"]),
    )
    updates = int(sanity_config["updates"])
    model.train()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(updates):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, compute_spectrum=False)
        loss = task_loss(output, truth, loss_config, teacher_weight=0.0)["task_total"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite Medium capacity overfit loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(e003_config["training"]["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("non-finite Medium capacity overfit gradient")
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        final_output = model(x, compute_spectrum=False)
        final_terms = task_loss(final_output, truth, loss_config, teacher_weight=0.0)
        presence_accuracy = float(
            (((final_output["presence_logit"] >= 0) == truth["truth_presence"].bool()).float().mean()).item()
        )
        h1 = truth["truth_presence"].bool()
        range_accuracy = float(
            (final_output["range_logits"][h1].argmax(-1) == torch.floor(truth["truth_range_cell"][h1]).long())
            .float().mean().item()
        )
    final_loss = float(final_terms["task_total"].item())
    loss_ratio = final_loss / max(initial_loss, np.finfo(np.float64).tiny)
    passed = bool(
        gradients_passed
        and loss_ratio <= 1.0 - float(sanity_config["required_train_truth_task_loss_relative_drop"])
        and presence_accuracy >= 1.0
        and range_accuracy >= 0.95
    )
    result = {
        "git_sha": commit,
        "model": "Medium",
        "width": 96,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "updates": updates,
        "paired_scenes": int(len(pair_ids)),
        "initial_truth_task_loss": initial_loss,
        "final_truth_task_loss": final_loss,
        "loss_ratio_final_to_initial": loss_ratio,
        "presence_accuracy_on_overfit_batch": presence_accuracy,
        "h1_range_cell_accuracy_on_overfit_batch": range_accuracy,
        "task_gradient_norms": gradient_norms,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "passed": passed,
        "scientific_metrics": False,
        "final_blind_generated": False,
    }
    _write_json(CAPACITY_ROOT / "g1_overfit_sanity.json", result)
    del optimizer, model, x, truth
    torch.cuda.empty_cache()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def run_training() -> dict[str, Any]:
    capacity_config, r5_config, e003_config, _, commit, context = _require_capacity_preflight()
    sanity_path = CAPACITY_ROOT / "g1_overfit_sanity.json"
    if not sanity_path.is_file() or json.loads(sanity_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("Medium G1 overfit sanity must pass before the formal capacity training")
    cache_root = ROOT / r5_config["data"]["cache_root"]
    train = r5_runner._load_split(cache_root, "train")
    validation = r5_runner._load_split(cache_root, "validation")
    if len(train["base_scene_id"]) != 51_200 or len(validation["base_scene_id"]) != 5_000:
        raise RuntimeError("capacity train/validation IDs must match the frozen E003 cohorts")
    plan_path = r5_runner.ARTIFACTS / "training_pair_plan.npz"
    if not plan_path.is_file():
        raise FileNotFoundError("the completed R5-A shared pair plan is required for capacity training")
    plan = r5_runner._load_plan(train, r5_config)
    target_updates = int(context["specs"]["target_updates"])
    if plan.shape[0] != 12800 or target_updates != 12800:
        raise RuntimeError("capacity training must use the frozen 12,800-update pair plan")
    seed = int(capacity_config["experiment"]["training_seed"])
    initial_state = r5_runner._initial_student_state(seed, width=96)
    medium_root = CAPACITY_ROOT / "runs/seed_101/Medium"
    existing_record_path = medium_root / "training_record.json"
    prior_record = json.loads(existing_record_path.read_text(encoding="utf-8")) if existing_record_path.is_file() else None
    if prior_record is not None and int(prior_record.get("updates_completed", -1)) >= target_updates:
        medium_record = prior_record
    else:
        effective_batch = int(r5_config["training"]["effective_batch_size"])
        oom_attempts: list[dict[str, Any]] = []
        started = time.perf_counter()
        medium_record = None
        for microbatch in _microbatch_candidates(capacity_config["training"]):
            torch.cuda.empty_cache()
            try:
                medium_record = r5_runner._train_attempt(
                    arm="S0",
                    width=96,
                    initial_state=initial_state,
                    train_arrays=train,
                    validation_arrays=validation,
                    plan=plan,
                    e003_config=e003_config,
                    r5_config=r5_config,
                    device=r5_runner._require_cuda(),
                    microbatch=microbatch,
                    target_updates=target_updates,
                    arm_root=medium_root,
                )
                break
            except torch.cuda.OutOfMemoryError as exc:
                oom_attempts.append({"microbatch_size": microbatch, "error": str(exc)[:1000]})
                path = medium_root / "oom_attempts.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(oom_attempts[-1], ensure_ascii=False) + "\n")
                torch.cuda.empty_cache()
                print(f"[R5-A capacity] Medium OOM at microbatch={microbatch}; retrying smaller", flush=True)
        if medium_record is None:
            _write_json(medium_root / "failed.json", {"model": "Medium", "oom_attempts": oom_attempts, "cpu_fallback": False})
            raise RuntimeError("Medium OOMed at microbatch 4; stopped without CPU, AMP or effective-batch fallback")
        medium_record["model"] = "Medium"
        medium_record["arm"] = "S0"
        medium_record["width"] = 96
        medium_record["parameter_count"] = int(medium_record["parameter_count"])
        medium_record["updates_completed"] = int(medium_record["updates_completed"])
        medium_record["training_seconds_cumulative"] = float(time.perf_counter() - started)
        medium_record["prior_oom_attempts"] = oom_attempts
        _write_json(existing_record_path, medium_record)

    if int(medium_record.get("width", -1)) != 96 or int(medium_record.get("parameter_count", -1)) != 82887 or int(medium_record.get("updates_completed", -1)) != target_updates:
        raise ValueError("Medium training record does not match the frozen width/parameter/update budget")
    small_record = dict(context["small_record"])
    small_record["model"] = "Small"
    small_record["checkpoint_reused_from_r5a"] = True
    result = {
        "experiment": "D2-A-R5/R5-A-capacity",
        "capacity_git_sha": commit,
        "base_training_git_sha": context["base_training"]["git_sha"],
        "protocol_version": float(capacity_config["experiment"]["protocol_version"]),
        "training_seed": seed,
        "supervision": "physical_truth_only_E1",
        "teacher_losses_used": False,
        "same_frozen_train_validation_ids_as_E003": True,
        "same_pair_plan_as_R5A": True,
        "effective_batch_size": int(capacity_config["training"]["effective_batch_size"]),
        "target_updates": target_updates,
        "device": "cuda:0",
        "precision": "FP32",
        "arms": {"Small": small_record, "Medium": medium_record},
        "final_blind_generated": False,
        "distillation_run": False,
        "training_seed_uncertainty_estimated": False,
    }
    _write_json(CAPACITY_ROOT / "training.json", result)
    print(json.dumps({"git_sha": commit, "arms": {name: {key: result["arms"][name].get(key) for key in ("width", "parameter_count", "updates_completed", "selected_update", "microbatch_size", "training_seconds_cumulative", "cuda_peak_allocated_bytes")} for name in ("Small", "Medium")}, "final_blind_generated": False}, ensure_ascii=False, indent=2), flush=True)
    return result


def _load_capacity_model(label: str, record: Mapping[str, Any], *, expected_source_sha: str) -> torch.nn.Module:
    width = int(record["width"])
    checkpoint_path = ROOT / str(record["best_checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        payload.get("experiment") != "D2-A-R5"
        or payload.get("subphase") != "R5-A"
        or payload.get("arm") != "S0"
        or int(payload.get("width", -1)) != width
        or payload.get("source_git_sha") != expected_source_sha
    ):
        raise ValueError(f"{label} selected checkpoint identity/provenance mismatch: {checkpoint_path}")
    model = TaskStudent(width=width, include_spectrum_head=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model


def _relabel_evaluation(label: str, arm_data: dict[str, Any], physical: dict[str, Any], rows: list[dict[str, Any]], compute: list[dict[str, Any]]) -> None:
    arm_data["metrics"]["arm"] = label
    if "phase" in physical:
        physical["phase"]["arm"] = label
    if "frequency" in physical:
        physical["frequency"]["arm"] = label
    for row in rows:
        row["arm"] = label
    for row in compute:
        row["arm"] = label


def run_evaluate() -> dict[str, Any]:
    capacity_config, r5_config, e003_config, physics, commit, context = _require_capacity_preflight()
    training_path = CAPACITY_ROOT / "training.json"
    if not training_path.is_file():
        raise RuntimeError("Small/Medium capacity training must complete before evaluation")
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("capacity_git_sha") != commit or training.get("distillation_run") is not False or training.get("final_blind_generated") is not False:
        raise RuntimeError("capacity training provenance or no-blind/no-KD declaration is invalid")
    cache_root = ROOT / r5_config["data"]["cache_root"]
    arrays = {
        name: r5_runner._load_split(cache_root, name)
        for name in ("h0_calibration", "h0_verification", "c3_h0_verification", "development", "exact_overlap_boundary")
    }
    train_arrays = r5_runner._load_split(cache_root, "train")
    selected_h0_by_course, h0_rows = e003_runner._selected_course_indices(arrays["h0_verification"], e003_config)
    selected_h1_by_course, h1_rows = e003_runner._selected_course_indices(arrays["development"], e003_config)
    selected_h0 = np.concatenate([selected_h0_by_course[level] for level in ("C0", "C1", "C2", "C3")])
    selected_h1 = np.concatenate([selected_h1_by_course[level] for level in ("C0", "C1", "C2", "C3")])
    e003_runner._write_csv(
        CAPACITY_ROOT / "selected_scene_ids.csv",
        [{"split": "h0_verification", **row} for row in h0_rows]
        + [{"split": "development", **row} for row in h1_rows],
    )
    model_records = training["arms"]
    expected_sources = {
        "Small": context["base_training"]["git_sha"],
        "Medium": commit,
    }
    evaluated: dict[str, dict[str, Any]] = {}
    physical_results: dict[str, Any] = {}
    roc_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    compute_rows: list[dict[str, Any]] = []
    predictions_for_archive: dict[str, Any] = {}
    device = r5_runner._require_cuda()
    for label in ("Small", "Medium"):
        print(f"[R5-A capacity] evaluating {label} on frozen independent H0/H1 development splits", flush=True)
        model = _load_capacity_model(label, model_records[label], expected_source_sha=expected_sources[label]).to(device).eval()
        arm_data, physical, arm_roc, arm_rows, arm_compute = e003_runner._run_arm_evaluation(
            "S0", model, device, e003_config, arrays, selected_h0, selected_h1, physics
        )
        _relabel_evaluation(label, arm_data, physical, arm_rows, arm_compute)
        for row in arm_roc:
            row["arm"] = label
        evaluated[label] = arm_data
        physical_results[label] = physical
        roc_rows.extend(arm_roc)
        result_rows.extend(arm_rows)
        compute_rows.extend(arm_compute)
        predictions_for_archive[label] = arm_data["predictions"]
        del model
        torch.cuda.empty_cache()

    # The inherited E003 comparator's S0/S2 slots are deliberately mapped to Small/Medium.
    comparison_inputs = {"S0": evaluated["Small"], "S2": evaluated["Medium"]}
    raw_comparison = e003_runner._compare_arms(comparison_inputs, arrays, train_arrays, e003_config)
    capacity_comparison = _capacity_summary(raw_comparison)
    metrics = {
        "experiment": "D2-A-R5/R5-A-capacity",
        "capacity_git_sha": commit,
        "base_training_git_sha": context["base_training"]["git_sha"],
        "training_seed": int(capacity_config["experiment"]["training_seed"]),
        "same_frozen_e003_calibration_verification_development_splits": True,
        "thresholds_calibrated_separately_per_model": True,
        "threshold_comparator": ">=",
        "arms": {label: evaluated[label]["metrics"] for label in ("Small", "Medium")},
        "capacity_comparison": capacity_comparison,
        "inherited_e003_comparison": {
            "slot_mapping": {"S0": "Small", "S2": "Medium"},
            "values": raw_comparison,
        },
        "physical_consistency": physical_results,
        "compute": compute_rows,
        "scene_predictions": "experiments/D2-A-R5/R5-A/capacity/scene_predictions.csv",
        "prediction_archive": "experiments/D2-A-R5/R5-A/capacity/predictions.npz",
        "bootstrap_unit": "paired base_scene_id; single training seed does not estimate seed uncertainty",
        "development_is_final_blind": False,
        "final_blind_generated": False,
        "distillation_run": False,
    }
    _write_json(CAPACITY_ROOT / "metrics.json", metrics)
    e003_runner._write_csv(CAPACITY_ROOT / "result.csv", result_rows)
    e003_runner._write_csv(CAPACITY_ROOT / "roc.csv", roc_rows)
    r5_runner._write_scene_predictions(CAPACITY_ROOT / "scene_predictions.csv", arrays, predictions_for_archive)
    r5_runner._save_prediction_archive(CAPACITY_ROOT / "predictions.npz", evaluated)
    result = {
        "git_sha": commit,
        "Small_width": int(training["arms"]["Small"]["width"]),
        "Medium_width": int(training["arms"]["Medium"]["width"]),
        "primary_gain_medium_minus_small": capacity_comparison["primary_gain_medium_minus_small"],
        "task_protection_passed": bool(capacity_comparison["task_protection"].get("passed")),
        "candidate_eligible": capacity_comparison["candidate_eligible"],
        "final_blind_generated": False,
        "distillation_run": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "overfit", "train", "evaluate"))
    args = parser.parse_args(argv)
    actions = {
        "preflight": run_preflight,
        "overfit": run_overfit_sanity,
        "train": run_training,
        "evaluate": run_evaluate,
    }
    actions[args.phase]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
