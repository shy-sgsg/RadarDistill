#!/usr/bin/env python3
"""Build a source-linked L0/L1 capability table, CSV views, and figures for R7-S."""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "experiments/D2-A-R7/R7-D/metrics.json"
INVENTORY_PATH = ROOT / "experiments/D2-A-R7/R7-D/checkpoint_inventory.csv"
OUT = ROOT / "experiments/D2-A-R7-S/S2"

CAPABILITY_FIELDS = [
    "study", "experiment", "method_id", "method_name", "method_role", "model_family",
    "task_scope", "training_seed", "parameter_count", "trainable_parameter_count",
    "checkpoint_update", "checkpoint_path", "checkpoint_sha256", "source_git_sha",
    "dataset", "snr_db", "split", "n_h0_calibration", "n_h0_verification", "n_h1",
    "false_alarm_unit", "target_pfa_cpi", "threshold", "pfa_cpi", "pfa_ci95",
    "false_alarm_cpi_count", "pd", "pd_ci95", "auc", "range_rmse_cells_all_h1",
    "range_mae_cells_all_h1", "range_bias_cells_all_h1", "range_p95_abs_error_cells_all_h1",
    "range_rmse_ci95_all_h1", "n_h1_detected", "range_rmse_cells_detected_h1",
    "range_mae_cells_detected_h1", "range_bias_cells_detected_h1",
    "range_p95_abs_error_cells_detected_h1", "range_rmse_ci95_detected_h1",
    "doppler_rmse_hz_all_h1", "doppler_mae_hz_all_h1", "doppler_bias_hz_all_h1",
    "doppler_p95_abs_error_hz_all_h1", "doppler_rmse_ci95_hz_all_h1",
    "doppler_rmse_hz_detected_h1", "doppler_mae_hz_detected_h1",
    "doppler_bias_hz_detected_h1", "doppler_p95_abs_error_hz_detected_h1",
    "doppler_rmse_ci95_hz_detected_h1", "joint_success_count", "joint_success",
    "joint_success_ci95", "metric_status", "comparability_status", "source_metrics",
    "source_protocol", "source_analysis", "source_pointer", "metric_payload_json",
]

IDENTITY_FIELDS = [
    "study", "method_id", "experiment_id", "model_name", "architecture_metadata",
    "task_scope", "parameter_count", "trainable_parameter_count", "training_seed",
    "checkpoint_selected_update", "training_updates_completed", "unique_training_scenes",
    "training_scene_exposure", "effective_batch_size", "teacher_knowledge",
    "fixed_physical_frontend", "checkpoint_path", "checkpoint_sha256",
    "training_metadata_source", "training_record_status",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _get(mapping: Any, *keys: str, default: Any = "") -> Any:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def _ci(value: Any) -> Any:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) if value not in (None, "") else ""


def _metric_slice(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return item.get("detection") or {}, item.get("parameters") or {}


def build_capability_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten actual R7-D results, preserving L0/L1 and all-H1/detected-H1 grains."""
    rows: list[dict[str, Any]] = []
    source_commit = metrics.get("source_commit", "NOT_RECORDED")

    def emit(method: dict[str, Any], item: dict[str, Any], dataset: str, pointer: str, category: str) -> None:
        detection, parameters = _metric_slice(item)
        all_h1 = parameters.get("all_h1") or {}
        detected = parameters.get("detected_h1") or {}
        range_all = all_h1.get("range_error_cells") or {}
        range_detected = detected.get("range_error_cells") or {}
        doppler_all = all_h1.get("doppler_error_hz") or {}
        doppler_detected = detected.get("doppler_error_hz") or {}
        is_l0 = dataset == "L0_TARGET_ONLY"
        method_id = str(method.get("method_id", ""))
        role = str(method.get("task_scope", method.get("method_role", ""))).lower()
        parameter_only = category == "traditional_reference" and ("parameter_only" in role or "estimator" in method_id.lower() or "profile_ml" in method_id.lower())
        detector_na = "NOT_APPLICABLE_PARAMETER_ONLY" if parameter_only else "NOT_REPORTED"
        def detector_value(value: Any, unavailable: str = detector_na) -> Any:
            return unavailable if value is None or value == "" else value
        row = {key: "" for key in CAPABILITY_FIELDS}
        row.update({
            "study": method.get("experiment_id", ""),
            "experiment": "R7-D",
            "method_id": method_id,
            "method_name": method.get("model_label", method.get("method_label", method.get("method_id", ""))),
            "method_role": method.get("task_scope", method.get("method_role", category)),
            "model_family": _family(method),
            "task_scope": method.get("task_scope", method.get("method_role", "")),
            "training_seed": method.get("seed_id", "NOT_APPLICABLE_NONLEARNED" if category == "traditional_reference" else "NOT_RECORDED"),
            "parameter_count": method.get("parameter_count", "NOT_APPLICABLE_NONLEARNED" if category == "traditional_reference" else "NOT_RECORDED"),
            "trainable_parameter_count": method.get("trainable_parameter_count", "NOT_APPLICABLE_NONLEARNED" if category == "traditional_reference" else "NOT_RECORDED"),
            "checkpoint_update": method.get("checkpoint_update", "NOT_APPLICABLE_NONLEARNED" if category == "traditional_reference" else "NOT_RECORDED"),
            "checkpoint_path": method.get("checkpoint_path", "NOT_APPLICABLE_NONLEARNED" if category == "traditional_reference" else "NOT_RECORDED"),
            "checkpoint_sha256": method.get("checkpoint_sha256", "NOT_APPLICABLE_NONLEARNED" if category == "traditional_reference" else "NOT_RECORDED"),
            "source_git_sha": source_commit,
            "dataset": dataset,
            "snr_db": "NOT_APPLICABLE_NO_NOISE" if is_l0 else item.get("snr_db", ""),
            "split": "noiseless target-only diagnostic" if is_l0 else "new independent R7-D H0/H1",
            "n_h0_calibration": "NOT_APPLICABLE_L0_NO_RANDOM_H0" if is_l0 else item.get("h0_calibration_n", ""),
            "n_h0_verification": "NOT_APPLICABLE_L0_NO_RANDOM_H0" if is_l0 else detection.get("verification_scenes", item.get("h0_verification_n", "")),
            "n_h1": all_h1.get("n", item.get("h1_n", detection.get("h1_scenes", ""))),
            "false_alarm_unit": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(detection.get("false_alarm_unit", "")),
            "target_pfa_cpi": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(detection.get("target_pfa_cpi")),
            "threshold": "NOT_APPLICABLE_L0_NO_THRESHOLD" if is_l0 else detector_value(detection.get("threshold")),
            "pfa_cpi": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(detection.get("verification_pfa")),
            "pfa_ci95": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else (_ci(detection.get("verification_pfa_ci95")) or detector_na),
            "false_alarm_cpi_count": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(detection.get("verification_false_alarms")),
            "pd": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(detection.get("h1_pd")),
            "pd_ci95": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else (_ci(detection.get("h1_pd_ci95")) or detector_na),
            "auc": detector_value(detection.get("auc")),
            "range_rmse_cells_all_h1": all_h1.get("range_rmse_cells", ""),
            "range_mae_cells_all_h1": all_h1.get("range_mae_cells", ""),
            "range_bias_cells_all_h1": range_all.get("bias", ""),
            "range_p95_abs_error_cells_all_h1": range_all.get("p95_abs_error", ""),
            "range_rmse_ci95_all_h1": _ci(_get(all_h1, "range_bootstrap_ci95", "rmse_ci95", default="")),
            "n_h1_detected": detected.get("n", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "range_rmse_cells_detected_h1": detected.get("range_rmse_cells", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "range_mae_cells_detected_h1": detected.get("range_mae_cells", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "range_bias_cells_detected_h1": range_detected.get("bias", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "range_p95_abs_error_cells_detected_h1": range_detected.get("p95_abs_error", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "range_rmse_ci95_detected_h1": _ci(_get(detected, "range_bootstrap_ci95", "rmse_ci95", default="")),
            "doppler_rmse_hz_all_h1": all_h1.get("doppler_rmse_hz", ""),
            "doppler_mae_hz_all_h1": all_h1.get("doppler_mae_hz", ""),
            "doppler_bias_hz_all_h1": doppler_all.get("bias", ""),
            "doppler_p95_abs_error_hz_all_h1": doppler_all.get("p95_abs_error", ""),
            "doppler_rmse_ci95_hz_all_h1": _ci(_get(all_h1, "doppler_bootstrap_ci95", "rmse_ci95", default="")),
            "doppler_rmse_hz_detected_h1": detected.get("doppler_rmse_hz", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "doppler_mae_hz_detected_h1": detected.get("doppler_mae_hz", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "doppler_bias_hz_detected_h1": doppler_detected.get("bias", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "doppler_p95_abs_error_hz_detected_h1": doppler_detected.get("p95_abs_error", "NOT_APPLICABLE_L0_DIAGNOSTIC" if is_l0 else "NOT_APPLICABLE_PARAMETER_ONLY"),
            "doppler_rmse_ci95_hz_detected_h1": _ci(_get(detected, "doppler_bootstrap_ci95", "rmse_ci95", default="")),
            "joint_success_count": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(parameters.get("joint_success_count")),
            "joint_success": "NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else detector_value(parameters.get("joint_success")),
            "joint_success_ci95": ("NOT_APPLICABLE_NO_THRESHOLD" if is_l0 else (_ci(parameters.get("joint_success_ci95")) or detector_na)),
            "metric_status": detection.get("status", method.get("status", "NOT_REPORTED")),
            "comparability_status": "R7D_SHARED_SCENES; profile_ML uses fixed 256-H1 subset" if method.get("method_id") == "profile_ML_finite_grid_256_H1" else "R7D_SHARED_SCENES; see protocol for method-specific applicability",
            "source_metrics": "experiments/D2-A-R7/R7-D/metrics.json",
            "source_protocol": "experiments/D2-A-R7/R7-D/protocol.md",
            "source_analysis": "experiments/D2-A-R7/R7-D/analysis.md",
            "source_pointer": pointer,
            "metric_payload_json": json.dumps(item, ensure_ascii=False, separators=(",", ":")),
        })
        rows.append(row)

    for index, method in enumerate(metrics.get("models", [])):
        datasets = method.get("datasets", {})
        if datasets.get("L0_TARGET_ONLY"):
            emit(method, datasets["L0_TARGET_ONLY"], "L0_TARGET_ONLY", f"/models/{index}/datasets/L0_TARGET_ONLY", "historical_checkpoint")
        for snr_index, item in enumerate(datasets.get("L1_AWGN_NO_CLUTTER", [])):
            emit(method, item, "L1_AWGN_NO_CLUTTER", f"/models/{index}/datasets/L1_AWGN_NO_CLUTTER/{snr_index}", "historical_checkpoint")

    for index, method in enumerate(metrics.get("classical", [])):
        datasets = method.get("datasets", {})
        l0 = datasets.get("L0_TARGET_ONLY")
        if isinstance(l0, dict):
            emit(method, l0, "L0_TARGET_ONLY", f"/classical/{index}/datasets/L0_TARGET_ONLY", "traditional_reference")
        for snr_index, item in enumerate(datasets.get("L1_AWGN_NO_CLUTTER", [])):
            emit(method, item, "L1_AWGN_NO_CLUTTER", f"/classical/{index}/datasets/L1_AWGN_NO_CLUTTER/{snr_index}", "traditional_reference")

    rows.sort(key=lambda row: (row["method_id"], row["dataset"], float(row["snr_db"]) if row["snr_db"] not in ("", "NOT_APPLICABLE_NO_NOISE") else -999.0))
    return rows


def _family(method: dict[str, Any]) -> str:
    label = str(method.get("model_label", method.get("method_label", method.get("method_id", ""))))
    path = str(method.get("checkpoint_path", ""))
    method_id = str(method.get("method_id", ""))
    if not path:
        if "profile_ML" in method_id:
            return "traditional profile-ML estimator"
        if "lag_one" in method_id:
            return "traditional lag-one estimator"
        if "zero_pad" in method_id:
            return "traditional zero-padded FFT"
        if "parabolic" in method_id:
            return "traditional FFT interpolation"
        return "traditional FFT baseline"
    if "R6-J4" in path and "j4_joint" in path:
        return "fixed DPCA/FFT + learned reader"
    if "rax_transformer" in path or "TFAS" in path:
        return "attention/Transformer"
    if "complex_ssm" in path:
        return "complex SSM"
    if "cnn_ssm" in path:
        return "CNN-SSM"
    if "ce_tcn" in path:
        return "CE-TCN"
    if "R6-J4" in path:
        return "raw-IQ J1-J"
    if "lag_one_feature" in method_id or "joint_with_lag_one_feature" in str(method.get("task_scope", "")):
        return "raw-IQ TCN + lag-one feature"
    if "lightweight_2d_depthwise_separable_tcn" in label:
        return "depthwise-separable TCN"
    if "D1-E002 S" in label:
        return "D1-E002 TCN"
    if "S2 fixed-coherence" in label:
        return "physics-guided TaskStudent"
    if "TaskStudent" in label or "Medium" in label or "Small" in label or "Large" in label or "XLarge" in label:
        return "TaskStudent CNN"
    return "neural checkpoint; family metadata incomplete"


def _metadata_dict(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return {"metadata_display": raw}
    return value if isinstance(value, dict) else {"metadata_display": raw}


def _architecture_label(model: dict[str, Any], source_model: dict[str, Any], path: str) -> str:
    architecture = source_model.get("architecture")
    if architecture:
        return str(architecture)
    variant = source_model.get("variant")
    if variant:
        return f"TaskStudent TCN variant {variant}; width={source_model.get('width','NOT_RECORDED')}"
    label = str(model.get("model_label", ""))
    for marker, result in (
        ("TemporalFrequencyAttentionStudent", "TemporalFrequencyAttentionStudent"),
        ("CE-TCN", "CE-TCN"), ("RAX-Transformer", "RAX-Transformer"),
        ("CNN-SSM", "CNN-SSM"), ("complex_ssm", "complex SSM"),
        ("TaskStudent", "TaskStudent; detailed block architecture not serialized in R7-D inventory"),
    ):
        if marker.lower() in label.lower() or marker.lower() in path.lower():
            return result
    if "j4_joint" in path:
        return "fixed DPCA/FFT candidate frontend + learned reader"
    return "NOT_RECORDED_IN_CHECKPOINT_INVENTORY"


def _training_cohort(model: dict[str, Any], path: str) -> tuple[str, str]:
    """Return only training cohort sizes stated in source protocols/configs."""
    if "D1-E001" in path:
        return "4096 H0 + 4096 H1 per training seed", "experiments/D1-E001/protocol.md"
    if "D1-E002" in path:
        return "4096 H0 + 4096 H1 per formal seed", "experiments/D1-E002/protocol.md"
    if "R5-A" in path:
        return "51,200 paired base scenes", "experiments/D2-A-R5/R5-A/protocol.md"
    if "R5-B" in path:
        return "51,200 paired base scenes", "experiments/D2-A-R5/R5-B/protocol.md"
    if "R5-C-teaching" in path:
        return "51,200 paired base scenes", "experiments/D2-A-R5/R5-C-teaching/protocol.md"
    if "R6-C" in path and "train_extended" in path:
        return "102,400 unique training scene pairs", "experiments/D2-A-R6/R6-C/config.yaml"
    for stage in ("R6-B", "R6-C", "R6-D", "R6-E", "R6-G", "R6-H", "R6-J4"):
        if stage in path:
            return "51,200 paired training scenes", f"experiments/D2-A-R6/{stage}/protocol.md"
    return "NOT_RECORDED", "NOT_RECORDED"


def _teacher_training_label(model: dict[str, Any], path: str) -> str:
    arm = str(model.get("arm", ""))
    stage = str(model.get("experiment_id", ""))
    provenance = f"{path} {model.get('method_id', '')}"
    if stage == "D1-E001":
        return {
            "A_truth_only": "NO_TEACHER_TRUTH_ONLY",
            "B_teacher_pretrain_then_truth": "YES_TEACHER_PRETRAIN_THEN_TRUTH",
            "C_joint_truth_and_teacher": "YES_JOINT_TEACHER_LOSS",
            "D_progressive_distillation": "YES_PROGRESSIVE_DISTILLATION",
        }.get(arm, "NOT_RECORDED")
    if stage == "D1-E002":
        return "NO_TEACHER; S2 uses lag-one input feature" if "lag_one" in str(model.get("task_scope", "")) else "NO_TEACHER_TRUTH_SUPERVISION"
    if "R5-C-teaching" in provenance:
        return {"T0": "NO_TEACHER_TRUTH_ONLY", "T1": "YES_K3_SOFT_RESPONSE", "T2": "YES_K2_LOCAL_SOFT_RESPONSE", "T3": "TRUTH_ASSISTED_CONTROL_NO_TEACHER"}.get(arm, "NOT_RECORDED")
    if "R6-G" in provenance:
        return {"T0": "NO_TEACHER_TRUTH_ONLY", "T1": "YES_K3_SOFT_RESPONSE", "T3": "TRUTH_ASSISTED_CONTROL_NO_TEACHER"}.get(arm, "NOT_RECORDED")
    if "R5-A" in path and "S2" in path:
        return "NO_TEACHER; FIXED_COHERENCE_PHYSICS_FEATURE"
    if "HardNegative" in path:
        return "NO_TEACHER; H0_HARD_NEGATIVE_RESAMPLING"
    return "NO_TEACHER_KNOWLEDGE_IDENTIFIED_IN_SELECTED_STAGE"


def build_identity_rows(metrics: dict[str, Any], inventory_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    inventory = {row.get("method_id", ""): row for row in inventory_rows}
    output = []
    for model in metrics.get("models", []):
        method_id = model.get("method_id", "")
        record = inventory.get(method_id, {})
        source_model = _metadata_dict(record.get("model_name", ""))
        checkpoint_path = model.get("checkpoint_path", record.get("checkpoint_path", ""))
        architecture = _architecture_label(model, source_model, checkpoint_path)
        cohort, cohort_source = _training_cohort(model, checkpoint_path)
        run_dir = ROOT / Path(checkpoint_path).parent if checkpoint_path else None
        train_record: dict[str, Any] = {}
        training_status = "NOT_RECORDED"
        if run_dir and (run_dir / "training_record.json").is_file():
            try:
                train_record = json.loads((run_dir / "training_record.json").read_text(encoding="utf-8"))
                training_status = "TRAINING_RECORD_READ"
            except json.JSONDecodeError:
                training_status = "TRAINING_RECORD_UNREADABLE"
        history_completed = "NOT_RECORDED"
        if run_dir and (run_dir / "training_history.csv").is_file():
            try:
                history = read_csv(run_dir / "training_history.csv")
                updates = [int(row["update"]) for row in history if row.get("update", "").isdigit()]
                if updates:
                    history_completed = max(updates)
                    training_status = "HISTORY_READ" if training_status == "NOT_RECORDED" else training_status + "+HISTORY_READ"
            except (OSError, csv.Error, KeyError):
                pass
        selected_update = model.get("checkpoint_update") or "NOT_RECORDED_IN_CHECKPOINT_INVENTORY"
        unique_scenes = train_record.get("unique_train_paired_scenes", "NOT_RECORDED")
        exposure = train_record.get("total_cpi_exposures_per_presence", train_record.get("cpi_exposures_per_presence", "NOT_RECORDED"))
        batch = train_record.get("effective_batch_size", "NOT_RECORDED")
        if not unique_scenes:
            unique_scenes = "NOT_RECORDED"
        if unique_scenes == "NOT_RECORDED" and cohort != "NOT_RECORDED":
            unique_scenes = cohort
        if not exposure:
            exposure = "NOT_RECORDED"
        if exposure == "NOT_RECORDED" and "paired base scenes" in cohort and str(selected_update).isdigit():
            paired_presentations = int(selected_update) * 32
            exposure = f"derived_from_protocol: paired_scene_presentations={paired_presentations}; CPI_presentations={paired_presentations * 2}"
        frontend = "NOT_RECORDED_IN_CHECKPOINT_METADATA"
        if source_model.get("is_hybrid") is True:
            frontend = "YES_FIXED_HYBRID_FRONTEND"
        elif source_model.get("is_hybrid") is False:
            frontend = "NO_FIXED_HYBRID_FRONTEND"
        if "R6-J4" in checkpoint_path and "j4_joint" in checkpoint_path:
            frontend = "YES_FIXED_DPCA_FFT_FRONTEND"
        if "R5-A" in checkpoint_path and "S2" in checkpoint_path:
            frontend = "FIXED_COHERENCE_PHYSICS_FEATURE"
        if "lag_one_feature" in str(model.get("task_scope", "")):
            frontend = "LAG_ONE_INPUT_FEATURE; NO_FIXED_DETECTOR"
        output.append({
            "study": model.get("experiment_id", ""), "method_id": method_id,
            "experiment_id": model.get("experiment_id", ""),
            "model_name": model.get("model_label", record.get("model_name", "")),
            "architecture_metadata": architecture,
            "task_scope": model.get("task_scope", record.get("task_scope", "NOT_RECORDED")),
            "parameter_count": model.get("parameter_count", "NOT_RECORDED"),
            "trainable_parameter_count": model.get("trainable_parameter_count", "NOT_RECORDED"),
            "training_seed": model.get("seed_id", record.get("seed_id", "NOT_RECORDED")),
            "checkpoint_selected_update": selected_update,
            "training_updates_completed": train_record.get("updates", train_record.get("updates_completed", history_completed)),
            "unique_training_scenes": unique_scenes,
            "training_scene_exposure": exposure,
            "effective_batch_size": batch,
            "teacher_knowledge": _teacher_training_label(model, checkpoint_path),
            "fixed_physical_frontend": frontend,
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": model.get("checkpoint_sha256", record.get("checkpoint_sha256", "NOT_RECORDED")),
            "training_metadata_source": "; ".join(item for item in ("R7-D checkpoint_inventory.csv", (str(run_dir.relative_to(ROOT) / "training_record.json") if run_dir and (run_dir / "training_record.json").is_file() else ""), cohort_source) if item and item != "NOT_RECORDED"),
            "training_record_status": training_status,
        })
    return output


def _representative(method: dict[str, Any]) -> bool:
    path = str(method.get("checkpoint_path", ""))
    targets = [
        "D1-E001/formal/seed_1/A_truth_only/", "D1-E001/formal/seed_1/B_teacher_pretrain_then_truth/",
        "D1-E002/formal/seed_1/S1/", "D1-E002/formal/seed_1/S2/",
        "R5-A/runs/seed_101/S0/", "R5-A/capacity/runs/seed_101/Medium/", "R5-A/runs/seed_101/S2/",
        "R5-B/runs/seed_101/TFAS/", "R5-C-teaching/runs/seed_303/T0/", "R5-C-teaching/runs/seed_303/T1/",
        "R5-C-teaching/runs/seed_303/T2/", "R5-C-teaching/runs/seed_303/T3/",
        "R6-C/runs/seed_404/train_core/Medium/", "R6-D/runs/seed_505/j1_joint/", "R6-E/runs/seed_606/ce_tcn/",
        "R6-E/runs/seed_606/rax_transformer/", "R6-G/runs/seed_707_v1.0.1/T0/", "R6-G/runs/seed_707_v1.0.1/T1/",
        "R6-H/runs/seed_808/ce_tcn/", "R6-H/runs/seed_808/cnn_ssm/", "R6-H/runs/seed_808/complex_ssm/",
        "R6-J4/runs/seed_909/j1_joint/", "R6-J4/runs/seed_909/j4_joint/",
    ]
    return any(target in path for target in targets)


def _short_label(row: dict[str, Any]) -> str:
    method_id = str(row.get("method_id", ""))
    classical_names = {
        "B0_single_channel_FFT_peak": "B0 FFT",
        "B1_three_channel_noncoherent_FFT_peak": "B1 3-ch FFT",
        "B1_detection_plus_native_FFT_parameters": "FFT + native grid",
        "B1_detection_plus_log_parabolic_parameters": "FFT + parabolic",
        "B1_detection_plus_4x_zero_pad_parameters": "FFT + 4x zero pad",
        "lag_one_phase_difference_estimator": "lag-one phase",
        "profile_ML_finite_grid_256_H1": "profile-ML (256 H1)",
    }
    if method_id in classical_names:
        return classical_names[method_id]
    path = str(row.get("checkpoint_path", ""))
    if path and not path.startswith("NOT_APPLICABLE"):
        parts = Path(path).parts
        stage = next((part for part in parts if part.startswith(("D1-", "R5-", "R6-"))), "model")
        if stage == "R5-C-teaching":
            stage = "R5-C"
        arm = parts[-2] if len(parts) >= 2 else "model"
        seed = str(row.get("training_seed", "")).replace("seed_", "s")
        if seed and not seed.startswith("s"):
            seed = f"s{seed}"
        return f"{stage}:{arm}:{seed}".rstrip(":")
    return str(row.get("method_name", method_id)).replace("_", " ")


def _save_figures(capability_rows: list[dict[str, Any]], identity_rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150, "savefig.dpi": 180})
    models = json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("models", [])
    checkpoint_ids = {method["method_id"] for method in models}
    chosen = {method["method_id"] for method in models if _representative(method)}
    chosen.update(method["method_id"] for method in json.loads(METRICS_PATH.read_text(encoding="utf-8")).get("classical", []))
    plotted = [row for row in capability_rows if row["dataset"] == "L1_AWGN_NO_CLUTTER" and row["method_id"] in chosen]
    snrs = sorted({float(row["snr_db"]) for row in plotted if row["snr_db"] not in ("", "NOT_APPLICABLE_NO_NOISE")})
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in plotted:
        by_method.setdefault(row["method_id"], []).append(row)
    styles = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    metrics = [
        ("pd", "Pd", "Pd at SNR-specific calibrated Pfa_CPI=0.01"),
        ("pfa_cpi", "Pfa_CPI", "Independent H0 verification Pfa_CPI"),
        ("range_rmse_cells_all_h1", "Range RMSE (cells)", "All-H1 range RMSE"),
        ("doppler_rmse_hz_all_h1", "Doppler RMSE (Hz)", "All-H1 circular Doppler RMSE"),
        ("joint_success", "Joint success", "Joint success; misses count as failures"),
    ]
    for figure_number, (field, ylabel, title) in enumerate(metrics, start=1):
        fig, ax = plt.subplots(figsize=(12.8, 7.3))
        for color_index, (method_id, rows) in enumerate(sorted(by_method.items())):
            ordered = sorted(rows, key=lambda item: float(item["snr_db"]))
            xs, ys = [], []
            for row in ordered:
                value = row.get(field, "")
                try:
                    number = float(value)
                except (ValueError, TypeError):
                    continue
                if math.isfinite(number):
                    xs.append(float(row["snr_db"]))
                    ys.append(number)
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=1.25, markersize=3.5,
                        color=styles[color_index % len(styles)], alpha=0.82,
                        label=_short_label(ordered[0]))
        ax.set(title=title, xlabel="SNR (dB)", ylabel=ylabel, xticks=snrs)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=6.3, frameon=False, ncol=1)
        fig.tight_layout()
        fig.savefig(OUT / f"figure_{figure_number}_{field}.png", bbox_inches="tight")
        plt.close(fig)

    # L0 to L1 @ 0 dB paired descriptive comparison; no deployment claim for L0 Pfa.
    l0 = {row["method_id"]: row for row in capability_rows if row["dataset"] == "L0_TARGET_ONLY"}
    l1_0 = {row["method_id"]: row for row in capability_rows if row["dataset"] == "L1_AWGN_NO_CLUTTER" and str(row["snr_db"]) in {"0", "0.0"}}
    common = sorted(l0.keys() & l1_0.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for ax, metric, label in zip(axes, ["range_rmse_cells_all_h1", "doppler_rmse_hz_all_h1"], ["Range RMSE (cells)", "Doppler RMSE (Hz)"]):
        for method_id in common:
            x, y = l0[method_id].get(metric, ""), l1_0[method_id].get(metric, "")
            try:
                x, y = float(x), float(y)
            except (TypeError, ValueError):
                continue
            learned = method_id in checkpoint_ids
            color = "#2f6b9a" if learned else "#b9503c"
            ax.scatter(x, y, s=30, marker="o" if learned else "s", alpha=.78, color=color)
        ax.set(xlabel=f"L0 {label}", ylabel=f"L1 @ 0 dB {label}", title=f"{label}: noiseless to AWGN")
    from matplotlib.lines import Line2D
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2f6b9a", label="historical student checkpoint", markersize=6),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#b9503c", label="traditional reference", markersize=6),
    ], loc="upper left", fontsize=7)
    fig.suptitle("Same method identities; L0 is a noiseless diagnostic, not a Pfa workpoint")
    fig.tight_layout()
    fig.savefig(OUT / "figure_6_l0_vs_l1.png", bbox_inches="tight")
    plt.close(fig)

    # Descriptive parameter-count relation at the same R7-D operating points.
    ident = {row["method_id"]: row for row in identity_rows}
    points = {row["method_id"]: row for row in capability_rows if row["dataset"] == "L1_AWGN_NO_CLUTTER" and str(row["snr_db"]) in {"10", "10.0"}}
    stage_keys = sorted({meta.get("study", "") for meta in ident.values()})
    stage_color = {stage: styles[index % len(styles)] for index, stage in enumerate(stage_keys)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    for method_id, row in points.items():
        meta = ident.get(method_id, {})
        try:
            size = float(meta.get("parameter_count", ""))
        except (TypeError, ValueError):
            continue
        color = stage_color.get(meta.get("study", ""), styles[0])
        for ax, metric, ylabel in zip(axes, ["pd", "doppler_rmse_hz_all_h1"], ["Pd", "Doppler RMSE (Hz)"]):
            try:
                val = float(row.get(metric, ""))
            except (TypeError, ValueError):
                continue
            ax.scatter(size, val, s=26, alpha=.72, color=color)
            ax.set_xscale("log")
            ax.set(xlabel="Trainable parameter count (log scale)", ylabel=ylabel, title=f"L1 @ 10 dB: {ylabel}")
    fig.suptitle("Descriptive across-checkpoint scatter; training conditions differ, not a capacity-causal test")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=stage_color[stage], label=stage, markersize=6) for stage in stage_keys],
               loc="lower center", ncol=min(len(stage_keys), 5), fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(OUT / "figure_7_parameter_count_vs_performance.png", bbox_inches="tight")
    plt.close(fig)

    # Full-method overview; each column is normalized only for color, cell text remains raw.
    metrics_for_heatmap = [
        ("L0 range", lambda m: l0.get(m, {}).get("range_rmse_cells_all_h1", ""), "low"),
        ("L0 Doppler", lambda m: l0.get(m, {}).get("doppler_rmse_hz_all_h1", ""), "low"),
        ("L1 Pd 0 dB", lambda m: l1_0.get(m, {}).get("pd", ""), "high"),
        ("L1 Doppler 10 dB", lambda m: points.get(m, {}).get("doppler_rmse_hz_all_h1", ""), "low"),
        ("L1 Joint 10 dB", lambda m: points.get(m, {}).get("joint_success", ""), "high"),
    ]
    all_ids = sorted({row["method_id"] for row in capability_rows}, key=lambda mid: (ident.get(mid, {}).get("study", "traditional"), mid))
    raw = np.full((len(all_ids), len(metrics_for_heatmap)), np.nan, dtype=float)
    scores = np.full_like(raw, np.nan)
    for col, (_, getter, direction) in enumerate(metrics_for_heatmap):
        for row_idx, method_id in enumerate(all_ids):
            try:
                raw[row_idx, col] = float(getter(method_id))
            except (TypeError, ValueError):
                pass
        finite = np.isfinite(raw[:, col])
        if np.any(finite):
            vals = raw[finite, col]
            lo, hi = float(np.min(vals)), float(np.max(vals))
            normalized = np.ones_like(vals) if hi == lo else (vals - lo) / (hi - lo)
            if direction == "low":
                normalized = 1.0 - normalized
            scores[finite, col] = normalized
    fig, ax = plt.subplots(figsize=(11, max(10, len(all_ids) * .22)))
    image = ax.imshow(scores, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(metrics_for_heatmap)), [item[0] for item in metrics_for_heatmap], rotation=25, ha="right")
    labels = []
    capability_by_method = {row["method_id"]: row for row in capability_rows}
    for method_id in all_ids:
        meta = ident.get(method_id, capability_by_method.get(method_id, {}))
        labels.append(_short_label({**meta, "method_id": method_id}))
    ax.set_yticks(range(len(all_ids)), labels, fontsize=5.6)
    for row_idx in range(len(all_ids)):
        for col in range(len(metrics_for_heatmap)):
            value = raw[row_idx, col]
            label = "—" if not np.isfinite(value) else f"{value:.3g}"
            ax.text(col, row_idx, label, ha="center", va="center", fontsize=5.4)
    fig.colorbar(image, ax=ax, label="Within-column normalized descriptive score (green=better)")
    ax.set_title("R7-D capability across 47 historical checkpoints and 7 classical references\nEach row is a checkpoint/method record, not an independent architecture")
    fig.tight_layout()
    fig.savefig(OUT / "figure_8_model_family_capability_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def _table_views(capability_rows: list[dict[str, Any]], identity_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    l0 = [row for row in capability_rows if row["dataset"] == "L0_TARGET_ONLY"]
    l1 = [row for row in capability_rows if row["dataset"] == "L1_AWGN_NO_CLUTTER"]
    return {
        "table_1_model_identity_training.csv": identity_rows,
        "table_2_l0_all_methods.csv": l0,
        "table_3_l1_detection_by_snr.csv": [{k: r.get(k, "") for k in ("method_id", "method_name", "method_role", "model_family", "parameter_count", "dataset", "snr_db", "n_h0_calibration", "n_h0_verification", "n_h1", "target_pfa_cpi", "pfa_cpi", "pfa_ci95", "pd", "pd_ci95", "auc", "false_alarm_unit", "metric_status", "source_pointer")} for r in l1],
        "table_4_l1_range_estimation_by_snr.csv": [{k: r.get(k, "") for k in ("method_id", "method_name", "method_role", "model_family", "dataset", "snr_db", "n_h1", "n_h1_detected", "range_rmse_cells_all_h1", "range_rmse_ci95_all_h1", "range_rmse_cells_detected_h1", "range_rmse_ci95_detected_h1", "metric_status", "source_pointer")} for r in l1],
        "table_5_l1_doppler_estimation_by_snr.csv": [{k: r.get(k, "") for k in ("method_id", "method_name", "method_role", "model_family", "dataset", "snr_db", "n_h1", "n_h1_detected", "doppler_rmse_hz_all_h1", "doppler_rmse_ci95_hz_all_h1", "doppler_rmse_hz_detected_h1", "doppler_rmse_ci95_hz_detected_h1", "metric_status", "source_pointer")} for r in l1],
        "table_6_l1_joint_by_snr.csv": [{k: r.get(k, "") for k in ("method_id", "method_name", "method_role", "model_family", "dataset", "snr_db", "n_h1", "joint_success_count", "joint_success", "joint_success_ci95", "metric_status", "source_pointer")} for r in l1],
        "table_7_l1_all_vs_detected_errors.csv": [{k: r.get(k, "") for k in ("method_id", "method_name", "model_family", "dataset", "snr_db", "n_h1", "n_h1_detected", "range_rmse_cells_all_h1", "range_rmse_cells_detected_h1", "doppler_rmse_hz_all_h1", "doppler_rmse_hz_detected_h1", "joint_success", "source_pointer")} for r in l1],
        "table_8_l1_traditional_references.csv": [r for r in l1 if not r.get("checkpoint_path")],
    }


def build_comparability_rows(capability_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in capability_rows:
        by_method.setdefault(row["method_id"], []).append(row)
    output = []
    for method_id, rows in sorted(by_method.items()):
        first = rows[0]
        l0_n = next((row["n_h1"] for row in rows if row["dataset"] == "L0_TARGET_ONLY"), "NOT_REPORTED")
        l1_ns = sorted({str(row["n_h1"]) for row in rows if row["dataset"] == "L1_AWGN_NO_CLUTTER"})
        output.append({
            "method_id": method_id, "method_name": first["method_name"], "method_role": first["method_role"],
            "model_family": first["model_family"], "parameter_count": first["parameter_count"],
            "L0_result_status": "MEASURED" if l0_n not in ("NOT_REPORTED", "") else "NOT_REPORTED_BY_R7D_PROTOCOL",
            "L0_n_h1": l0_n, "L1_snr_points": len([r for r in rows if r["dataset"] == "L1_AWGN_NO_CLUTTER"]),
            "L1_h1_n_values": ";".join(l1_ns),
            "detection_output": "YES" if first["task_scope"] not in ("parameter_only", "parameter_estimator") and "estimator" not in method_id else "NOT_APPLICABLE_PARAMETER_ONLY",
            "parameter_output": "YES" if any(r["range_rmse_cells_all_h1"] != "" for r in rows) else "NOT_REPORTED",
            "false_alarm_unit": first["false_alarm_unit"], "comparability_boundary": first["comparability_status"],
            "source_metrics": first["source_metrics"],
        })
    return output


def write_index(capability_rows: list[dict[str, Any]], identity_rows: list[dict[str, Any]]) -> None:
    views = _table_views(capability_rows, identity_rows)
    for filename, rows in views.items():
        fields = IDENTITY_FIELDS if filename.startswith("table_1") else list(rows[0]) if rows else []
        write_csv(OUT / filename, rows, fields)
    write_csv(OUT / "table_9_method_comparability.csv", build_comparability_rows(capability_rows), list(build_comparability_rows(capability_rows)[0]))
    write_csv(OUT / "all_models_l0_l1_capability_matrix.csv", capability_rows, CAPABILITY_FIELDS)
    index = [
        "# D2-A-R7-S S2：无杂波基础能力总表与图表\n",
        f"\n主表记录 **{len(capability_rows)} 条实际结果**：L0 {sum(r['dataset']=='L0_TARGET_ONLY' for r in capability_rows)} 条，L1 {sum(r['dataset']=='L1_AWGN_NO_CLUTTER' for r in capability_rows)} 条；覆盖47个历史checkpoint和7种传统方法。R7-D原 protocol 没有为 profile-ML 生成L0结果，故不补造该行。\n",
        "\n这张CSV是用户要求的逐方法、逐数据集能力主表；检测、参数估计、全H1/检测命中子集、样本数、区间、训练/checkpoint身份和来源均保留。跨阶段原协议结果请同时参阅[完整统一性能矩阵](../../D2-A-R7/unified_performance_matrix.csv)。\n",
        "\nL0为无噪声、无杂波目标单独场景（每方法2,048 H1），仅作映射诊断，不报告Pfa/Pd工作点。L1为无杂波复AWGN；每SNR每方法有2,048 H0校准、2,048独立H0验证和2,048 H1，Pfa目标按每SNR校准为0.01。profile-ML固定使用每SNR前256个H1；检测专用或参数专用方法的非适用指标留空/写明状态。\n",
        "\n## 固定方法实例快览\n",
        "\n以下列出同一方法在0和10 dB的行，避免把不同模型的单点最佳值拼成一个虚构方法。完整模型/教师结果以主CSV和跨阶段矩阵为准。\n",
    ]
    compare_ids = [
        "experiments::D2-A-R6::R6-J4::runs::seed_909::j1_joint::best_checkpoint",
        "experiments::D2-A-R6::R6-J4::runs::seed_909::j4_joint::best_checkpoint",
        "B1_detection_plus_4x_zero_pad_parameters",
        "lag_one_phase_difference_estimator",
        "profile_ML_finite_grid_256_H1",
    ]
    compare_names = {
        compare_ids[0]: "R6-J4 J1-J raw-IQ student",
        compare_ids[1]: "R6-J4 J4 fixed-DPCA/FFT hybrid student",
        compare_ids[2]: "B1 FFT detector + 4× zero-pad estimator",
        compare_ids[3]: "lag-one phase estimator (parameter-only)",
        compare_ids[4]: "profile-ML (parameter-only; 256 H1)",
    }
    by_key = {(row["method_id"], float(row["snr_db"])): row for row in capability_rows
              if row["dataset"] == "L1_AWGN_NO_CLUTTER" and row["snr_db"] not in ("", "NOT_APPLICABLE_NO_NOISE")}
    index.append("\n| 方法 | SNR | n H1 | Pfa_CPI | Pd | AUC | Range RMSE (cell) | Doppler RMSE (Hz) | Joint success |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for method_id in compare_ids:
        for snr in (0.0, 10.0):
            row = by_key.get((method_id, snr))
            if row is None:
                continue
            values = []
            for key in ("pfa_cpi", "pd", "auc", "range_rmse_cells_all_h1", "doppler_rmse_hz_all_h1", "joint_success"):
                value = row.get(key, "")
                if value in ("", "NOT_APPLICABLE_PARAMETER_ONLY", "NOT_APPLICABLE_NO_THRESHOLD", "NOT_REPORTED"):
                    values.append("—")
                    continue
                try:
                    value = f"{float(value):.4g}"
                except (TypeError, ValueError):
                    value = str(value)
                values.append(value)
            index.append("| " + " | ".join([compare_names[method_id], str(row["snr_db"]), str(row["n_h1"]), *values]) + " |\n")
    index.append("\n参数RMSE针对全部H1；检测命中H1子集误差另见表7。检测专用/参数专用方法无关指标不参与排名。\n")
    index.append("\n## 九个表格\n")
    for number, filename in enumerate([*views.keys(), "table_9_method_comparability.csv"], start=1):
        index.append(f"\n{number}. [{filename}](./{filename})\n")
    index += [
        "\n## 八幅图\n",
        "\n- [Pd–SNR](./figure_1_pd.png)\n- [Pfa_CPI–SNR](./figure_2_pfa_cpi.png)\n- [Range RMSE–SNR](./figure_3_range_rmse_cells_all_h1.png)\n- [Doppler RMSE–SNR](./figure_4_doppler_rmse_hz_all_h1.png)\n- [Joint success–SNR](./figure_5_joint_success.png)\n- [L0与L1对照](./figure_6_l0_vs_l1.png)\n- [参数量与性能](./figure_7_parameter_count_vs_performance.png)\n- [全方法能力热图](./figure_8_model_family_capability_heatmap.png)\n",
        "\n代表性曲线按阶段/方法类别预先固定，不按本图的最高分挑选；完整逐行结果见主CSV。图7是跨训练预算、数据分布和任务臂的描述性散点，不能作参数量因果结论。热图颜色只在各列内部归一化，格中数字为原始指标。\n",
    ]
    (OUT / "model_capabilities.md").write_text("".join(index), encoding="utf-8")


def main() -> None:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    inventory = read_csv(INVENTORY_PATH)
    capability_rows = build_capability_rows(metrics)
    identity_rows = build_identity_rows(metrics, inventory)
    write_index(capability_rows, identity_rows)
    _save_figures(capability_rows, identity_rows)
    print(json.dumps({
        "capability_rows": len(capability_rows),
        "l0_rows": sum(row["dataset"] == "L0_TARGET_ONLY" for row in capability_rows),
        "l1_rows": sum(row["dataset"] == "L1_AWGN_NO_CLUTTER" for row in capability_rows),
        "checkpoint_identity_rows": len(identity_rows),
        "outputs": sorted(path.name for path in OUT.iterdir()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
