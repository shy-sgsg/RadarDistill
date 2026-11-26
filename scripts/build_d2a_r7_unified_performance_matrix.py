#!/usr/bin/env python3
"""Build one source-linked, row-level comparison table for D0/D1/R5/R6/R7."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments/D2-A-R7/unified_performance_matrix.csv"
LEGACY_TABLE = ROOT / "docs/research/D2-A_model_performance_matrix.md"
R7_METRICS = ROOT / "experiments/D2-A-R7/R7-D/metrics.json"
R7E_REFERENCE_MATRIX = ROOT / "docs/research/D2-A-R7_CLASSICAL_REFERENCE_MATRIX.md"

FIELDS = [
    "record_grain", "study", "experiment", "dataset", "split", "metric_group",
    "condition_key", "scenario_level", "course", "scenario_id", "snr_db", "snr_definition", "cnr_db", "cnr_definition",
    "method_id", "method_name", "method_role", "task_scope", "seed", "checkpoint_role",
    "checkpoint_path", "checkpoint_sha256", "parameter_count",
    "trainable_parameter_count", "training_updates", "training_scene_exposure",
    "n_scenes", "n_h0_calibration", "n_h0_verification", "n_h1", "training_seed_count", "false_alarm_unit", "detection_workpoint",
    "target_pfa_cpi", "pfa_cpi", "false_alarm_cpi_count", "n_valid_h0_cpi", "pfa_ci95_lower", "pfa_ci95_upper",
    "valid_cuts_per_cpi", "valid_h0_cut_count", "false_cut_count", "mean_false_cut_exceedances_per_cpi",
    "pfa_cell", "pfa_cell_ci95_upper", "pfa_cell_verification_status", "pd", "pd_ci95_lower", "pd_ci95_upper", "auc",
    "auc_ci95_lower", "auc_ci95_upper", "range_rmse_cells",
    "range_rmse_detected_cells", "doppler_rmse_hz",
    "doppler_rmse_detected_hz", "joint_success", "teacher_localization_rate",
    "teacher_response_lift", "coherence_before_abs_median", "coherence_after_abs_median",
    "phase_residual_p95_rad", "aligned_residual_power_ratio", "clutter_suppression_db",
    "inference_wall_seconds", "gpu_peak_allocated_mib", "source_git_sha", "applicability_status", "comparability_status", "metric_status", "source_metrics", "source_protocol",
    "source_analysis", "source_pointer", "legacy_record_json", "metric_payload_json",
]

CONDITION_GRID = {
    "0": ("C0", "separated", 20.0, -5.0),
    "1": ("C1", "separated", 15.0, 0.0),
    "2": ("C1", "near_clutter_edge", 15.0, 0.0),
    "3": ("C2", "separated", 10.0, 10.0),
    "4": ("C2", "near_clutter_edge", 10.0, 10.0),
    "5": ("C2", "partial_overlap", 10.0, 10.0),
    "6": ("C3", "separated", 0.0, 20.0),
    "7": ("C3", "separated", 10.0, 20.0),
    "8": ("C3", "near_clutter_edge", 0.0, 20.0),
    "9": ("C3", "near_clutter_edge", 10.0, 20.0),
    "10": ("C3", "partial_overlap", 0.0, 20.0),
    "11": ("C3", "partial_overlap", 10.0, 20.0),
}
COLLECTION_KEYS = {
    "arms", "methods", "models", "systems", "candidate_summaries", "per_arm",
    "validation_trajectories", "comparisons", "capacity_comparison",
    "comparison_and_gates", "oracle_candidate_diagnostics", "thresholds_by_arm", "strategies",
    "values", "G3_task_learning", "basic_learning",
}
SIGNALS = {
    "pfa", "pfa_cpi", "pfa_gt", "pfa_ge", "verification_pfa", "pfa_1e_2",
    "pd", "h1_pd", "pd_at_pfa_1e_2", "auc", "auc_global", "roc_auc",
    "roc_auc_global", "range_rmse_cells", "range_rmse_all_h1_cells",
    "range_rmse", "doppler_rmse_hz", "doppler_rmse_all_h1_hz",
    "doppler_rmse", "joint_success", "teacher_top_cell_localization_rate",
    "target_localization_rate", "response_lift", "lift_mean", "target_response_lift",
    "target_window_mass_lift_mean",
}
METHOD_PATH_IDS = {
    "a", "b", "c", "s0", "s2", "e0", "b0", "b1", "b2", "b3", "t0", "t1", "t2", "t3",
    "k1", "k2", "k3", "k2_local", "k2-local", "k2_audit", "ace_amf", "train_constant_parameter_baseline",
}
METHOD_PATH_ALIASES = {
    "k2_audit": "K2",
    "train_constant_parameter_baseline": "CONSTANT_PARAMETER_BASELINE",
}


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def json_pointer(segments: tuple[str, ...] | list[str]) -> str:
    """Encode object keys as RFC 6901 JSON Pointer path segments."""
    if not segments:
        return ""
    encoded = (str(segment).replace("~", "~0").replace("/", "~1") for segment in segments)
    return "/" + "/".join(encoded)


def legacy_parameter_counts(value: str) -> tuple[str, str]:
    """Normalize the old summary's display-only parameter text into numeric/status fields."""
    text = (value or "").strip()
    if not text:
        return "NOT_RECORDED_LEGACY_SUMMARY", "NOT_RECORDED_LEGACY_SUMMARY"
    if "非学习" in text or "无可训练参数" in text:
        return "NOT_APPLICABLE_NONLEARNED", "NOT_APPLICABLE_NONLEARNED"
    if "训练" in text and "推理" in text:
        values = re.findall(r"\d[\d,]*", text)
        if len(values) >= 2:
            training = int(values[0].replace(",", ""))
            inference = int(values[1].replace(",", ""))
            return str(inference), str(training)
    match = re.search(r"\d[\d,]*", text)
    if match:
        return str(int(match.group(0).replace(",", ""))), "NOT_RECORDED_LEGACY_SUMMARY"
    return "NOT_RECORDED_LEGACY_SUMMARY", "NOT_RECORDED_LEGACY_SUMMARY"


def markdown_line_pointer(path: Path, target: str) -> str:
    """Return a source-linked pointer to exactly one Markdown line."""
    matches = [
        line_number
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if target in line
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Markdown line containing {target!r} in {path}, found {len(matches)}")
    return f"markdown table row at line {matches[0]}: {target}"


def related_doc(metrics_path: Path, name: str) -> str:
    for parent in (metrics_path.parent, *metrics_path.parent.parents):
        candidate = parent / name
        if candidate.exists() and (candidate == ROOT / name or ROOT in candidate.parents):
            return rel(candidate)
        if parent == ROOT:
            break
    return "NOT_RECORDED"


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def text_json(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def first(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return ""


def first_metric(record: dict[str, Any], *keys: str) -> Any:
    """Keep explicit applicability/availability labels in the metric columns."""
    for key in keys:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        if isinstance(value, str) and value.startswith(("NOT_", "MISSING_", "UNAVAILABLE", "UNVERIFIED")):
            return value
    return ""


def interval(record: dict[str, Any], list_keys: tuple[str, ...], low_keys: tuple[str, ...] = (), high_keys: tuple[str, ...] = ()) -> tuple[Any, Any]:
    for key in list_keys:
        value = record.get(key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return value[0], value[1]
    return first(record, *low_keys), first(record, *high_keys)


def base_row(**kwargs: Any) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update(kwargs)
    return row


def as_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def stage_from_path(path: Path) -> tuple[str, str]:
    parts = path.relative_to(ROOT).parts
    for i, part in enumerate(parts):
        if part.startswith("D0-") or part.startswith("D1-") or part.startswith("D2-A-"):
            experiment = parts[i + 1] if part == "D2-A-R7" and i + 1 < len(parts) else part
            return part, experiment
    return "UNKNOWN", path.parent.name


def condition_from_key(key: str) -> dict[str, Any]:
    match = re.search(r"(C[0-3])[_/ -](separated|near_clutter_edge|partial_overlap).*?(?:snr|SNR)[_/-]?(-?\d+(?:\.\d+)?).*?(?:cnr|CNR)[_/-]?(-?\d+(?:\.\d+)?)", key, re.IGNORECASE)
    if not match:
        return {}
    return {"course": match.group(1), "scenario_id": match.group(2), "snr_db": float(match.group(3)), "cnr_db": float(match.group(4))}


def direct_payload(node: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in node.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
        elif isinstance(value, (list, tuple)) and len(value) <= 20 and all(
            item is None or isinstance(item, (str, int, float, bool)) for item in value
        ):
            payload[key] = value
        elif key == "condition" and isinstance(value, dict):
            payload[key] = value
    return payload


def model_metadata(node: Any) -> dict[str, Any]:
    """Read only explicit scalar training/model metadata from a model record."""
    aliases = {
        "parameter_count": ("parameter_count", "parameter_count_inference_model", "inference_parameter_count"),
        "trainable_parameter_count": ("trainable_parameter_count", "parameter_count_training_model", "training_parameter_count"),
        "training_updates": ("optimizer_updates", "training_updates", "selected_update", "best_update", "best_checkpoint_update", "checkpoint_update"),
        "training_scene_exposure": ("training_scene_exposure", "train_scene_exposure", "n_train_scenes", "n_train_pairs", "training_scene_count"),
    }
    found: dict[str, Any] = {}
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for out_key, names in aliases.items():
                if out_key in found:
                    continue
                for name in names:
                    candidate = value.get(name)
                    if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
                        found[out_key] = candidate
                        break
            for child in value.values():
                if len(found) == len(aliases):
                    return
                walk(child)
        elif isinstance(value, list) and len(value) <= 512:
            for child in value:
                if len(found) == len(aliases):
                    return
                walk(child)
    walk(node)
    return found


def normalized_row(node: dict[str, Any], *, stage: str, experiment: str, path: Path,
                   pointer: str, context: dict[str, Any], method_id: str,
                   metric_group: str = "") -> dict[str, Any]:
    all_keys = set(node)
    pfa_low, pfa_high = interval(node, ("pfa_ci95", "pfa_exact95_ci", "pfa_ci95_exact"), (), ("pfa_ci95_exact_upper", "pfa_exact95_upper", "pfa_ci95_upper", "pfa_bonferroni_familywise_upper"))
    pd_low, pd_high = interval(node, ("pd_ci95", "pd_bootstrap_ci95", "pd_ci95_exact", "pd_wilson_95"), ("pd_wilson_95_lower",), ("pd_wilson_95_upper",))
    auc_low, auc_high = interval(node, ("auc_ci95", "auc_ci95_paired_bootstrap"))
    range_rmse = first_metric(node, "range_rmse_all_h1_cells", "range_rmse_cells", "range_rmse")
    doppler_rmse = first_metric(node, "doppler_rmse_all_h1_hz", "doppler_rmse_hz", "doppler_rmse")
    group = metric_group or ("H0_DETECTION" if all_keys.intersection({"pfa", "pfa_cpi", "pfa_gt", "pfa_ge", "verification_pfa"}) else "H1_DETECTION_ESTIMATION")
    h0_group = group.startswith("H0") or group == "THRESHOLD_CALIBRATION"
    cond = dict(context)
    if isinstance(node.get("condition"), dict):
        cond.update(node["condition"])
    for k in ("course", "scenario_id", "snr_db", "cnr_db"):
        if k in node:
            cond[k] = node[k]
    dataset_name = cond.get("dataset", node.get("dataset", experiment))
    if not isinstance(dataset_name, str):
        dataset_name = experiment
    row = base_row(
        record_grain="CONDITION" if any(k in cond for k in ("course", "scenario_id", "snr_db", "cnr_db")) else "SUMMARY",
        study=stage, experiment=experiment, dataset=dataset_name,
        split=cond.get("split", ""), metric_group=group, scenario_level=cond.get("scenario_level", ""),
        condition_key=cond.get("condition_key", ""), course=cond.get("course", ""),
        scenario_id=cond.get("scenario_id", ""), snr_db=cond.get("snr_db", ""),
        snr_definition=cond.get("snr_definition", node.get("snr_definition", "")),
        cnr_db=cond.get("cnr_db", ""), cnr_definition=cond.get("cnr_definition", node.get("cnr_definition", "")),
        method_id=method_id, method_name=cond.get("method_name", method_id),
        method_role=cond.get("method_role", ""), task_scope=cond.get("task_scope", node.get("task_scope", "")),
        seed=cond.get("seed", ""), checkpoint_path=cond.get("checkpoint_path", ""),
        checkpoint_sha256=cond.get("checkpoint_sha256", ""),
        parameter_count=cond.get("parameter_count", first(node, "parameter_count", "inference_parameter_count", "parameter_count_inference_model", "inference_parameters")),
        trainable_parameter_count=cond.get("trainable_parameter_count", first(node, "trainable_parameter_count", "parameter_count_training_model", "training_parameter_count", "training_parameters")),
        training_updates=cond.get("training_updates", first(node, "training_updates", "optimizer_updates", "selected_update", "best_update", "best_checkpoint_update", "checkpoint_update")),
        training_scene_exposure=cond.get("training_scene_exposure", first(node, "training_scene_exposure", "train_scene_exposure", "n_train_scenes", "n_train_pairs", "training_scene_count")),
        training_seed_count=cond.get("training_seed_count", first(node, "training_seed_count", "n_training_seeds")),
        n_h0_calibration=first(node, "h0_calibration_n", "calibration_n", "n_h0_calibration"),
        n_h0_verification=first(node, "verification_n", "h0_verification_n", "n_h0_verification", "n_h0", *(('n',) if h0_group else ())),
        n_h1=first(node, "h1_scenes", "h1_n", "n_h1", "n_paired_scenes", *(('n',) if not h0_group else ())),
        false_alarm_unit=node.get(
            "false_alarm_unit",
            "NOT_APPLICABLE_SOFT_TEACHER_RESPONSE" if group == "TEACHER_SOFT_RESPONSE_QUALITY"
            else (
                "Pfa_CPI" if stage.startswith("D2-A") and all_keys.intersection(
                    {"pfa", "pfa_cpi", "pfa_gt", "pfa_ge", "verification_pfa", "h1_pd", "pd"}
                ) else ""
            ),
        ),
        detection_workpoint=first_metric(node, "detection_workpoint", "workpoint", "target_pfa_cpi", "target_pfa"),
        target_pfa_cpi=first(node, "target_pfa_cpi", "target_pfa", "workpoint_pfa"),
        pfa_cpi=first_metric(node, "pfa", "pfa_cpi", "verification_pfa", "pfa_gt", "pfa_ge", "pfa_1e_2"),
        false_alarm_cpi_count=first(node, "verification_false_alarms", "false_alarm_cpi_count", "false_alarms", "false_scene_count"),
        n_valid_h0_cpi=first(node, "verification_scenes", "n_h0_verification", "n_h0"),
        pfa_ci95_lower=pfa_low, pfa_ci95_upper=pfa_high,
        valid_cuts_per_cpi=first(node, "valid_cuts_per_cpi", "n_valid_cuts_per_cpi"),
        valid_h0_cut_count=first(node, "valid_h0_cut_count", "n_valid_h0_cuts"),
        false_cut_count=first(node, "false_cut_count", "n_false_cuts"),
        mean_false_cut_exceedances_per_cpi=first(node, "mean_false_cut_exceedances_per_cpi"),
        pfa_cell=first_metric(node, "pfa_cell"),
        pfa_cell_ci95_upper=first_metric(node, "pfa_cell_ci95_upper", "pfa_cell_upper_95"),
        pfa_cell_verification_status=first_metric(node, "pfa_cell_verification_status"),
        pd=first_metric(node, "pd", "h1_pd", "pd_at_pfa_1e_2"), pd_ci95_lower=pd_low, pd_ci95_upper=pd_high,
        auc=first_metric(node, "auc", "auc_global", "roc_auc", "roc_auc_global"),
        auc_ci95_lower=auc_low, auc_ci95_upper=auc_high,
        range_rmse_cells=range_rmse, range_rmse_detected_cells=first(node, "range_rmse_detected_h1_cells"),
        doppler_rmse_hz=doppler_rmse, doppler_rmse_detected_hz=first(node, "doppler_rmse_detected_h1_hz"),
        joint_success=first_metric(node, "joint_success"),
        teacher_localization_rate=first_metric(node, "teacher_top_cell_localization_rate", "target_localization_rate"),
        teacher_response_lift=first_metric(node, "target_window_mass_lift_mean", "response_lift", "lift_mean", "target_response_lift"),
        coherence_before_abs_median=first(node, "coherence_before_magnitude_median"),
        coherence_after_abs_median=first(node, "coherence_after_magnitude_median"),
        phase_residual_p95_rad=first(node, "phase_residual_after_p95_abs_rad"),
        aligned_residual_power_ratio=first(node, "aligned_residual_power_ratio_median"),
        clutter_suppression_db=first(node, "suppression_db_relative_to_channel_a_median"),
        inference_wall_seconds=first(node, "inference_wall_seconds", "wall_seconds"),
        gpu_peak_allocated_mib=first(node, "gpu_peak_allocated_mib"),
        source_git_sha=cond.get("source_git_sha", "NOT_RECORDED"),
        metric_status=node.get("status") or "STATUS_NOT_RECORDED_IN_SOURCE", source_metrics=rel(path),
        source_protocol=related_doc(path, "protocol.md"), source_analysis=related_doc(path, "analysis.md"),
        source_pointer=pointer, metric_payload_json=text_json(direct_payload(node)),
    )
    if row["condition_key"] == "" and row["record_grain"] == "CONDITION":
        row["condition_key"] = "/".join(pointer.strip("/").split("/")[-3:])
    pointer_lower = pointer.lower()
    for checkpoint_role in ("best_validation", "initial_validation", "best_checkpoint", "inference_checkpoint", "teacher_validation_baseline"):
        if checkpoint_role in pointer_lower:
            row["checkpoint_role"] = checkpoint_role
            break
    if not row["pfa_cell"]:
        if group == "TEACHER_SOFT_RESPONSE_QUALITY":
            row["pfa_cell"] = "NOT_APPLICABLE_TEACHER_SCORE"
            row["pfa_cell_verification_status"] = "NOT_APPLICABLE_NO_THRESHOLD_DETECTOR"
        elif row["false_alarm_unit"] == "Pfa_CPI":
            row["pfa_cell"] = "NOT_REPORTED_FOR_THIS_SOURCE_RECORD"
            row["pfa_cell_ci95_upper"] = "NOT_VERIFIED"
            row["pfa_cell_verification_status"] = "NOT_VERIFIED_NO_DENSE_CELL_METRIC"
        elif row["task_scope"] == "parameter_only":
            row["pfa_cell"] = "NOT_APPLICABLE_PARAMETER_ONLY"
            row["pfa_cell_verification_status"] = "NOT_APPLICABLE_NO_DETECTION_OUTPUT"
    if row["snr_db"] != "" and not row["snr_definition"]:
        row["snr_definition"] = "NOT_EMBEDDED_IN_METRIC; SEE_SOURCE_PROTOCOL"
    if row["cnr_db"] != "" and not row["cnr_definition"]:
        row["cnr_definition"] = "NOT_EMBEDDED_IN_METRIC; SEE_SOURCE_PROTOCOL"
    if not row["training_scene_exposure"] and row["method_role"] not in {"CLASSICAL", "TRADITIONAL_TEACHER_PHYSICS_DIAGNOSTIC"}:
        row["training_scene_exposure"] = "NOT_RECORDED_IN_SOURCE_METRIC"
    row["comparability_status"] = "CROSS_STAGE_NOT_DIRECTLY_RANKED"
    return row


def add_json_metric_file(rows: list[dict[str, Any]], path: Path) -> None:
    data = json.loads(path.read_text())
    stage, experiment = stage_from_path(path)
    if stage == "D0-E001":
        return
    default_method = ""
    if isinstance(data.get("model_metrics"), dict):
        default_method = str(data["model_metrics"].get("arm", ""))
    def visit(node: Any, pointer: tuple[str, ...] = (), context: dict[str, Any] | None = None,
              current_model: str = "", metric_group: str = "") -> None:
        if isinstance(node, list):
            if len(node) <= 512:
                for index, item in enumerate(node):
                    if isinstance(item, dict):
                        visit(item, pointer + (str(index),), context, current_model, metric_group)
            return
        if not isinstance(node, dict):
            return
        context = dict(context or {})
        if isinstance(node.get("condition"), dict):
            context.update(node["condition"])
        context_aliases = {
            "seed_id": "seed",
            "parameter_count_training_model": "trainable_parameter_count",
            "parameter_count_inference_model": "parameter_count",
            "inference_parameter_count": "parameter_count",
            "optimizer_updates": "training_updates",
            "selected_update": "training_updates",
            "best_update": "training_updates",
            "n_train_scenes": "training_scene_exposure",
            "n_train_pairs": "training_scene_exposure",
            "training_scene_count": "training_scene_exposure",
            "train_scene_exposure": "training_scene_exposure",
        }
        context_keys = (
            "course", "scenario_id", "snr_db", "cnr_db", "split", "dataset", "task_scope", "seed", "seed_id",
            "checkpoint_path", "checkpoint_sha256", "parameter_count", "trainable_parameter_count",
            "training_updates", "training_scene_exposure", "training_seed_count", "method_role",
            *context_aliases.keys(),
        )
        for key in context_keys:
            if key in node and isinstance(node[key], (str, int, float)):
                mapped = context_aliases.get(key, key)
                context[mapped] = node[key]
        if pointer and pointer[-1] in {"by_condition", "verification_by_condition", "pd_by_condition", "by_scenario_snr", "by_scenario", "c3_cells", "by_course"}:
            context["condition_collection"] = pointer[-1]
        if pointer and context.get("condition_collection") and pointer[-1].isdigit():
            idx = pointer[-1]
            context["condition_key"] = f"{context['condition_collection']}:{idx}"
            if idx in CONDITION_GRID and not any(k in context for k in ("course", "scenario_id", "snr_db", "cnr_db")):
                course, scenario, snr, cnr = CONDITION_GRID[idx]
                context.update(course=course, scenario_id=scenario, snr_db=snr, cnr_db=cnr)
        if len(pointer) >= 2 and pointer[-2] in {"by_snr_db", "by_snr"}:
            try:
                context["snr_db"] = float(pointer[-1])
            except ValueError:
                pass
        if len(pointer) >= 2 and pointer[-2] in {"by_course", "by_course_level"} and pointer[-1].startswith("C"):
            context["course"] = pointer[-1]
        if len(pointer) >= 2 and pointer[-2] in {"by_scenario", "by_scenario_id"}:
            context["scenario_id"] = pointer[-1]
        for split_name in ("blind", "formal", "pilot", "development", "validation", "verification", "calibration"):
            if split_name in pointer:
                context.setdefault("split", split_name)
        pathstr = "/".join(pointer).lower()
        if len(pointer) >= 2 and pointer[-2] == "workpoints":
            try:
                context["target_pfa_cpi"] = float(pointer[-1])
            except ValueError:
                pass
        if pointer:
            parsed = condition_from_key("/".join(pointer))
            context.update({k:v for k,v in parsed.items() if k not in context})
        if "teacher_validation_baseline" in pointer and not current_model:
            current_model = "teacher_validation_baseline"
            context["method_name"] = current_model
        if "arm" in node and isinstance(node["arm"], str) and not current_model:
            current_model = node["arm"]
            context["method_name"] = current_model
        if "method" in node and isinstance(node["method"], str) and not current_model:
            current_model = node["method"]
            context["method_name"] = current_model
        if not current_model:
            path_segments = [segment.casefold() for segment in pointer]
            method_segment = next((segment for segment in reversed(path_segments) if segment in METHOD_PATH_IDS), "")
            if method_segment:
                current_model = METHOD_PATH_ALIASES.get(method_segment, method_segment.upper() if len(method_segment) <= 3 else method_segment)
                context["method_name"] = current_model
            elif "legacy_b3" in pathstr:
                current_model = "B3"
                context["method_name"] = current_model
        if context.get("condition_collection"):
            context["condition_key"] = context.get("condition_key", "/".join(pointer[-2:]))
        group = metric_group
        if "teacher" in pathstr or "candidate_summaries" in pathstr:
            group = "TEACHER_SOFT_RESPONSE_QUALITY"
        elif any(s in pathstr for s in ("calibration", "verification", "h0")):
            group = "H0_DETECTION"
        elif any(s in pathstr for s in ("h1", "development", "parameters", "estimators")):
            group = "H1_DETECTION_ESTIMATION"
        if SIGNALS.intersection(node):
            rows.append(normalized_row(node, stage=stage, experiment=experiment, path=path,
                                       pointer=json_pointer(pointer), context=context,
                                       method_id=current_model or "STAGE_AGGREGATE", metric_group=group))
            if not current_model:
                rows[-1]["method_name"] = "Stage-level aggregate; method not specified by source"
                rows[-1]["method_role"] = "AGGREGATE_NOT_METHOD_SPECIFIC"
                rows[-1]["metric_status"] = rows[-1]["metric_status"] or "AGGREGATE_NO_METHOD_LABEL_IN_SOURCE"
        for key, value in node.items():
            next_model = current_model
            if key in COLLECTION_KEYS and isinstance(value, dict):
                for model_id, model_data in value.items():
                    model_context = dict(context)
                    model_context.update({k: v for k, v in model_metadata(model_data).items() if k not in model_context})
                    id_text = str(model_id)
                    seed_match = re.search(r"(?:^|/)seed[_-]?(\d+)$", id_text)
                    if seed_match:
                        model_context["seed"] = seed_match.group(1)
                    model_context["method_name"] = id_text
                    visit(model_data, pointer + (key, id_text), model_context, id_text, group)
                continue
            visit(value, pointer + (str(key),), context, next_model, group)
    source_git_sha = data.get("git_sha", data.get("source_commit", data.get("commit_sha", "NOT_RECORDED")))
    visit(data, current_model=default_method, context={"source_git_sha": source_git_sha})


def add_csv_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for item in csv.DictReader(handle):
            stage, experiment = stage_from_path(path)
            context = dict(item)
            rows.append(base_row(
                record_grain="SNR", study=stage, experiment=experiment, dataset=experiment,
                split="blind" if "blind" in path.parts else "formal", metric_group="H1_DETECTION_ESTIMATION",
                snr_db=item.get("snr_db", ""), method_id=item.get("method", ""), method_name=item.get("method", ""),
                n_h0_verification=item.get("n_h0", ""), n_h1=item.get("n_h1", ""),
                false_alarm_unit="Pfa_CPI" if stage.startswith("D1") else "",
                target_pfa_cpi=0.01 if stage == "D1-E002" else 0.01,
                pfa_cpi=item.get("pfa_at_pfa_1e_2", item.get("pfa_1e_2", "")),
                pd=item.get("pd_at_pfa_1e_2", item.get("pd_pfa_1e_2", "")),
                auc=item.get("auc", ""), range_rmse_cells=item.get("range_rmse_cells", ""),
                doppler_rmse_hz=item.get("doppler_rmse_hz", ""), metric_status="EXTRACTED_FROM_RESULT_CSV",
                comparability_status="SAME_EXPERIMENT_RESULT_CSV_BY_METHOD_AND_SNR",
                source_metrics=rel(path),
                source_protocol=related_doc(path, "protocol.md"), source_analysis=related_doc(path, "analysis.md"),
                source_pointer=f"csv row method={item.get('method')}, snr_db={item.get('snr_db')}",
                metric_payload_json=text_json(item),
            ))


def add_d0(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "experiments/D0-E001/metrics.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for index, item in enumerate(data.get("teacher_by_snr", [])):
        rows.append(base_row(
            record_grain="SNR", study="D0-E001", experiment="D0-E001", dataset="D0 single-target complex AWGN",
            split="teacher diagnostic", metric_group="TEACHER_SCORE_AND_ESTIMATION",
            snr_db=item.get("snr_db", ""), method_id="D0_TEACHER", method_name="D0 fixed spectral teacher",
            method_role="TEACHER_SCORE_NOT_THRESHOLD_DETECTOR", n_h0_verification=item.get("n_h0", ""),
            n_h1=item.get("n_h1", ""), auc=item.get("teacher_score_auc", ""),
            range_rmse_cells=item.get("h1_range_rmse_cells", ""), doppler_rmse_hz=item.get("h1_doppler_rmse_hz", ""),
            false_alarm_unit="NOT_APPLICABLE_NO_THRESHOLD", pfa_cpi="NOT_APPLICABLE", pd="NOT_APPLICABLE",
            metric_status="EXTRACTED", source_metrics=rel(path),
            source_protocol="experiments/D0-E001/protocol.md", source_analysis="experiments/D0-E001/analysis.md",
            source_pointer=json_pointer(("teacher_by_snr", str(index))), metric_payload_json=text_json(item),
        ))


def add_r7b_physics(rows: list[dict[str, Any]]) -> None:
    path = ROOT / "experiments/D2-A-R7/R7-B/metrics.json"
    if not path.exists():
        return
    metrics = json.loads(path.read_text())
    component_names = ("static_clutter", "noise_only", "target_only", "mixed_h0", "mixed_h1")
    for condition_key, condition_metrics in metrics.get("conditions", {}).items():
        condition_label = condition_key.replace("snr_", "SNR=").replace("_cnr_", "/CNR=") + " dB"
        snr_match = re.search(r"snr_(-?\d+(?:\.\d+)?)_cnr_(-?\d+(?:\.\d+)?)", condition_key)
        snr = float(snr_match.group(1)) if snr_match else ""
        cnr = float(snr_match.group(2)) if snr_match else ""
        for component in component_names:
            component_metrics = condition_metrics.get(component)
            if not isinstance(component_metrics, dict):
                continue
            rows.append(base_row(
                record_grain="PHYSICS_DIAGNOSTIC", study="D2-A-R7", experiment="R7-B",
                dataset="D2-A synthetic stationary-clutter diagnostic", split="new independent diagnostic scenes",
                metric_group="DPCA_TEACHER_PHYSICS", condition_key=condition_label, snr_db=snr, cnr_db=cnr, scenario_id=component,
                method_id="B3_GEOMETRIC_DPCA", method_name="B3 geometric DPCA", method_role="TRADITIONAL_TEACHER_PHYSICS_DIAGNOSTIC",
                n_scenes=component_metrics.get("n_scenes", ""), false_alarm_unit="NOT_APPLICABLE_PHYSICS_DIAGNOSTIC",
                pfa_cpi="NOT_APPLICABLE", pd="NOT_APPLICABLE", auc="NOT_APPLICABLE",
                joint_success="NOT_APPLICABLE", coherence_before_abs_median=component_metrics.get("coherence_before_magnitude_median", ""),
                coherence_after_abs_median=component_metrics.get("coherence_after_magnitude_median", ""),
                phase_residual_p95_rad=component_metrics.get("phase_residual_after_p95_abs_rad", ""),
                aligned_residual_power_ratio=component_metrics.get("aligned_residual_power_ratio_median", ""),
                clutter_suppression_db=component_metrics.get("suppression_db_relative_to_channel_a_median", ""),
                comparability_status="R7-B_COMPONENT_DIAGNOSTIC_ONLY", metric_status="RECOMPUTED_COMPONENT_DIAGNOSTIC",
                source_metrics=rel(path), source_protocol="experiments/D2-A-R7/R7-B/protocol.md",
                source_analysis="experiments/D2-A-R7/R7-B/analysis.md", source_pointer=json_pointer(("conditions", condition_key, component)),
                metric_payload_json=text_json(component_metrics),
            ))
    for component, component_metrics in metrics.get("pooled_components", {}).items():
        rows.append(base_row(
            record_grain="PHYSICS_DIAGNOSTIC", study="D2-A-R7", experiment="R7-B",
            dataset="D2-A synthetic stationary-clutter diagnostic", split="pooled across five declared SNR/CNR groups",
            metric_group="DPCA_TEACHER_PHYSICS", condition_key="POOLED_1280_SCENES", scenario_id=component,
            method_id="B3_GEOMETRIC_DPCA", method_name="B3 geometric DPCA", method_role="TRADITIONAL_TEACHER_PHYSICS_DIAGNOSTIC",
            n_scenes=component_metrics.get("n_scenes", ""), false_alarm_unit="NOT_APPLICABLE_PHYSICS_DIAGNOSTIC",
            pfa_cpi="NOT_APPLICABLE", pd="NOT_APPLICABLE", auc="NOT_APPLICABLE", joint_success="NOT_APPLICABLE",
            coherence_before_abs_median=component_metrics.get("coherence_before_magnitude_median", ""),
            coherence_after_abs_median=component_metrics.get("coherence_after_magnitude_median", ""),
            phase_residual_p95_rad=component_metrics.get("phase_residual_after_p95_abs_rad", ""),
            aligned_residual_power_ratio=component_metrics.get("aligned_residual_power_ratio_median", ""),
            clutter_suppression_db=component_metrics.get("suppression_db_relative_to_channel_a_median", ""),
            comparability_status="R7-B_POOLED_COMPONENT_DIAGNOSTIC", metric_status="RECOMPUTED_COMPONENT_DIAGNOSTIC",
            source_metrics=rel(path), source_protocol="experiments/D2-A-R7/R7-B/protocol.md",
            source_analysis="experiments/D2-A-R7/R7-B/analysis.md", source_pointer=json_pointer(("pooled_components", component)),
            metric_payload_json=text_json(component_metrics),
        ))
    old = metrics.get("historical_e001_recomputation", {})
    if old:
        rows.append(base_row(
            record_grain="HISTORICAL_RECOMPUTATION", study="D2-A-R7", experiment="D2-A-E001 to R7-B",
            dataset="D2-A-E001 historical stationary-clutter diagnostic", split="old E001 scenes compared with new R7-B scenes",
            metric_group="DPCA_TEACHER_PHYSICS", method_id="B3_GEOMETRIC_DPCA", method_name="B3 geometric DPCA",
            method_role="TRADITIONAL_TEACHER_PHYSICS_DIAGNOSTIC", n_scenes=old.get("old_scene_count", ""),
            false_alarm_unit="NOT_APPLICABLE_PHYSICS_DIAGNOSTIC", pfa_cpi="NOT_APPLICABLE", pd="NOT_APPLICABLE",
            clutter_suppression_db=old.get("new_median_clutter_suppression_db", ""),
            comparability_status="HISTORICAL_RECOMPUTATION_NOT_MODEL_RANKING", metric_status="RECOMPUTED_HISTORICAL_DEFINITION",
            source_metrics=rel(path), source_protocol="experiments/D2-A-R7/R7-B/protocol.md",
            source_analysis="experiments/D2-A-R7/R7-B/analysis.md", source_pointer=json_pointer(("historical_e001_recomputation",)),
            metric_payload_json=text_json(old),
        ))


def add_legacy_rows(rows: list[dict[str, Any]]) -> None:
    lines = LEGACY_TABLE.read_text(encoding="utf-8").splitlines()
    headers: list[str] | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        record = dict(zip(headers, cells))
        if len(record) < 4:
            continue
        parameter_count, trainable_parameter_count = legacy_parameter_counts(record.get("参数量（训练/推理；如不同）", ""))
        rows.append(base_row(
            record_grain="HISTORICAL_SUMMARY", study="D2-A R5/R6", experiment=record.get("实验 / 数据集与样本", ""),
            dataset=record.get("实验 / 数据集与样本", ""), method_name=record.get("模型 / 方法", ""),
            parameter_count=parameter_count, trainable_parameter_count=trainable_parameter_count,
            method_role="HISTORICAL_SUMMARY_ROW", metric_group="DETECTION_AND_ESTIMATION_SUMMARY",
            comparability_status="CROSS_STAGE_NOT_DIRECTLY_RANKED",
            metric_status="VERBATIM_EXISTING_60_ROW_SUMMARY", source_metrics=rel(LEGACY_TABLE),
            source_pointer=f"markdown table row at line {line_number}: {record.get('模型 / 方法','')}", legacy_record_json=text_json(record),
        ))


def add_r7_rows(rows: list[dict[str, Any]]) -> None:
    if not R7_METRICS.exists():
        return
    metrics = json.loads(R7_METRICS.read_text())
    if metrics.get("status") != "COMPLETE":
        raise SystemExit(f"R7-D result is not COMPLETE: {metrics.get('status')}")
    path = R7_METRICS
    for model_index, model in enumerate(metrics.get("models", [])):
        common = dict(
            study="D2-A-R7", experiment="R7-D", method_id=model.get("method_id", ""),
            method_name=model.get("model_label", model.get("arm", "")), method_role=model.get("task_scope", ""),
            task_scope=model.get("task_scope", ""), seed=model.get("seed_id", ""), checkpoint_role="best_checkpoint",
            source_git_sha=metrics.get("source_commit", "NOT_RECORDED"),
            checkpoint_path=model.get("checkpoint_path", ""), checkpoint_sha256=model.get("checkpoint_sha256", ""),
            parameter_count=model.get("parameter_count", "NOT_RECORDED"), trainable_parameter_count=model.get("trainable_parameter_count", "NOT_RECORDED"),
            training_updates=model.get("checkpoint_update", ""), training_seed_count=1,
            training_scene_exposure="NOT_RECORDED_IN_CHECKPOINT_INVENTORY",
            inference_wall_seconds=model.get("wall_seconds", ""), gpu_peak_allocated_mib=model.get("cuda_peak_allocated_mib", ""),
            source_metrics=rel(path),
            source_protocol="experiments/D2-A-R7/R7-D/protocol.md", source_analysis="experiments/D2-A-R7/R7-D/analysis.md",
            comparability_status="PAIRED_ON_SHARED_R7D_L0_L1_SCENE_IDS",
        )
        datasets = model.get("datasets", {})
        l0 = datasets.get("L0_TARGET_ONLY", {})
        if l0:
            det = l0.get("detection", {})
            param = (l0.get("parameters") or {}).get("all_h1", {})
            row = base_row(**common, record_grain="DIAGNOSTIC", dataset="L0_TARGET_ONLY", scenario_level="L0", split="noiseless target-only diagnostic",
                           metric_group="H1_DIAGNOSTIC", n_h1=param.get("n", ""), false_alarm_unit="NOT_APPLICABLE_NO_THRESHOLD",
                           snr_definition="NO_NOISE_TARGET_ONLY_DIAGNOSTIC", detection_workpoint="NOT_APPLICABLE_NO_THRESHOLD",
                           valid_cuts_per_cpi=4160, valid_h0_cut_count="NOT_APPLICABLE_ZERO_IQ_H0",
                           false_cut_count="NOT_APPLICABLE_NO_THRESHOLD", pfa_cpi="NOT_APPLICABLE", pd="NOT_APPLICABLE", auc=det.get("auc", ""),
                           range_rmse_cells=param.get("range_rmse_cells", ""), doppler_rmse_hz=param.get("doppler_rmse_hz", ""),
                           joint_success="NOT_APPLICABLE", pfa_cell="NOT_APPLICABLE_SPARSE_CPI_SCORE",
                           pfa_cell_ci95_upper="NOT_APPLICABLE", pfa_cell_verification_status="NOT_APPLICABLE_NO_DENSE_SCORE_MAP",
                           metric_status=det.get("status", model.get("status", "")),
                           source_pointer=json_pointer(("models", str(model_index), "datasets", "L0_TARGET_ONLY")), metric_payload_json=text_json(l0))
            rows.append(row)
        for snr_index, item in enumerate(datasets.get("L1_AWGN_NO_CLUTTER", [])):
            det = item.get("detection", {})
            params = item.get("parameters") or {}
            all_h1 = params.get("all_h1", {})
            detected = params.get("detected_h1") or {}
            row = base_row(**common, record_grain="SNR", dataset="L1_AWGN_NO_CLUTTER", scenario_level="L1", split="new independent R7-D H0/H1",
                           metric_group="H0_DETECTION_AND_H1_ESTIMATION", snr_db=item.get("snr_db", ""),
                           snr_definition="target_reference_amplitude_squared / complex_noise_power_at_continuous_range_PSF_peak; clutter exactly zero",
                           cnr_db="NOT_APPLICABLE_NO_CLUTTER", cnr_definition="NOT_APPLICABLE_NO_CLUTTER",
                           n_h0_calibration=item.get("h0_calibration_n", ""), n_h0_verification=det.get("verification_scenes", ""),
                           n_h1=det.get("h1_scenes", all_h1.get("n", "")), false_alarm_unit=det.get("false_alarm_unit", "Pfa_CPI"),
                           detection_workpoint=det.get("target_pfa_cpi", ""), target_pfa_cpi=det.get("target_pfa_cpi", ""), pfa_cpi=det.get("verification_pfa", ""),
                           false_alarm_cpi_count=det.get("verification_false_alarms", ""), n_valid_h0_cpi=det.get("verification_scenes", ""),
                           pfa_ci95_lower=(det.get("verification_pfa_ci95") or ["", ""])[0],
                           pfa_ci95_upper=(det.get("verification_pfa_ci95") or ["", ""])[1],
                           valid_cuts_per_cpi=4160, valid_h0_cut_count="NOT_APPLICABLE_NO_DENSE_SCORE_MAP",
                           false_cut_count="NOT_APPLICABLE_NO_DENSE_SCORE_MAP", pfa_cell="NOT_APPLICABLE_SPARSE_CPI_SCORE",
                           pfa_cell_ci95_upper="NOT_APPLICABLE", pfa_cell_verification_status="NOT_VERIFIED_NO_DENSE_SCORE_MAP",
                           pd=det.get("h1_pd", ""),
                           pd_ci95_lower=(det.get("h1_pd_ci95") or ["", ""])[0], pd_ci95_upper=(det.get("h1_pd_ci95") or ["", ""])[1],
                           auc=det.get("auc", ""), range_rmse_cells=all_h1.get("range_rmse_cells", ""),
                           range_rmse_detected_cells=detected.get("range_rmse_cells", ""),
                           doppler_rmse_hz=all_h1.get("doppler_rmse_hz", ""),
                           doppler_rmse_detected_hz=detected.get("doppler_rmse_hz", ""),
                           joint_success=params.get("joint_success", ""), metric_status=det.get("status", ""),
                           source_pointer=json_pointer(("models", str(model_index), "datasets", "L1_AWGN_NO_CLUTTER", str(snr_index))),
                           metric_payload_json=text_json(item))
            rows.append(row)
    for model_index, model in enumerate(metrics.get("classical", [])):
        common = dict(study="D2-A-R7", experiment="R7-D", method_id=model.get("method_id", ""),
                      method_name=model.get("method_label", model.get("method_id", "")), method_role=model.get("method_role", "CLASSICAL"),
                      parameter_count="NOT_APPLICABLE_NONLEARNED", trainable_parameter_count="NOT_APPLICABLE_NONLEARNED",
                      training_updates="NOT_APPLICABLE_NONLEARNED", training_seed_count="NOT_APPLICABLE_NONLEARNED",
                      training_scene_exposure="NOT_APPLICABLE_NONLEARNED", inference_wall_seconds="NOT_RECORDED",
                      source_git_sha=metrics.get("source_commit", "NOT_RECORDED"),
                      comparability_status="PAIRED_ON_SHARED_R7D_L0_L1_SCENE_IDS",
                      source_metrics=rel(path), source_protocol="experiments/D2-A-R7/R7-D/protocol.md",
                      source_analysis="experiments/D2-A-R7/R7-D/analysis.md")
        l0_dataset = model.get("datasets", {}).get("L0_TARGET_ONLY", [])
        for item_index, item in enumerate(as_items(l0_dataset)):
            det = item.get("detection", {})
            param = item.get("parameters", {})
            all_h1 = param.get("all_h1", {})
            l0_pointer_segments = ["classical", str(model_index), "datasets", "L0_TARGET_ONLY"]
            if isinstance(l0_dataset, list):
                l0_pointer_segments.append(str(item_index))
            rows.append(base_row(**common, record_grain="DIAGNOSTIC", dataset="L0_TARGET_ONLY", scenario_level="L0", split="noiseless target-only diagnostic",
                                 metric_group="CLASSICAL_BASELINE", n_h1=all_h1.get("n", ""),
                                 false_alarm_unit=det.get("false_alarm_unit", "NOT_APPLICABLE_NO_THRESHOLD"),
                                 pfa_cpi="NOT_APPLICABLE", pd="NOT_APPLICABLE", auc=det.get("auc", ""),
                                 snr_definition="NO_NOISE_TARGET_ONLY_DIAGNOSTIC", detection_workpoint="NOT_APPLICABLE_NO_THRESHOLD",
                                 valid_cuts_per_cpi=4160, valid_h0_cut_count="NOT_APPLICABLE_ZERO_IQ_H0",
                                 false_cut_count="NOT_APPLICABLE_NO_THRESHOLD", pfa_cell="NOT_APPLICABLE_NO_DENSE_CUT_MAP",
                                 pfa_cell_ci95_upper="NOT_APPLICABLE", pfa_cell_verification_status="NOT_APPLICABLE_NO_DENSE_SCORE_MAP",
                                 range_rmse_cells=all_h1.get("range_rmse_cells", ""), doppler_rmse_hz=all_h1.get("doppler_rmse_hz", ""),
                                 joint_success=param.get("joint_success", "NOT_APPLICABLE"), metric_status=det.get("status", ""),
                                 source_pointer=json_pointer(l0_pointer_segments), metric_payload_json=text_json(item)))
        for item_index, item in enumerate(as_items(model.get("datasets", {}).get("L1_AWGN_NO_CLUTTER", []))):
            det = item.get("detection", {})
            param = item.get("parameters", {})
            rows.append(base_row(**common, record_grain="SNR", dataset="L1_AWGN_NO_CLUTTER", scenario_level="L1", split="new independent R7-D H0/H1",
                                 metric_group="CLASSICAL_BASELINE", snr_db=item.get("snr_db", ""),
                                 snr_definition="target_reference_amplitude_squared / complex_noise_power_at_continuous_range_PSF_peak; clutter exactly zero",
                                 cnr_db="NOT_APPLICABLE_NO_CLUTTER", cnr_definition="NOT_APPLICABLE_NO_CLUTTER",
                                 n_h0_calibration=item.get("h0_calibration_n", ""), n_h0_verification=det.get("verification_scenes", ""),
                                 n_h1=det.get("h1_scenes", item.get("h1_n", "")), false_alarm_unit=det.get("false_alarm_unit", "Pfa_CPI"),
                                 detection_workpoint=det.get("target_pfa_cpi", ""), target_pfa_cpi=det.get("target_pfa_cpi", ""), pfa_cpi=det.get("verification_pfa", ""),
                                 false_alarm_cpi_count=det.get("verification_false_alarms", ""), n_valid_h0_cpi=det.get("verification_scenes", ""),
                                 pfa_ci95_lower=(det.get("verification_pfa_ci95") or ["", ""])[0], pfa_ci95_upper=(det.get("verification_pfa_ci95") or ["", ""])[1],
                                 valid_cuts_per_cpi=4160, valid_h0_cut_count="NOT_APPLICABLE_NO_DENSE_SCORE_MAP",
                                 false_cut_count="NOT_APPLICABLE_NO_DENSE_SCORE_MAP", pfa_cell="NOT_APPLICABLE_NO_DENSE_CUT_MAP",
                                 pfa_cell_ci95_upper="NOT_APPLICABLE", pfa_cell_verification_status="NOT_VERIFIED_NO_DENSE_SCORE_MAP",
                                 pd=det.get("h1_pd", ""), pd_ci95_lower=(det.get("h1_pd_ci95") or ["", ""])[0],
                                 pd_ci95_upper=(det.get("h1_pd_ci95") or ["", ""])[1], auc=det.get("auc", ""),
                                 range_rmse_cells=param.get("range_rmse_cells", ""), doppler_rmse_hz=param.get("doppler_rmse_hz", ""),
                                 joint_success=param.get("joint_success", ""), metric_status=det.get("status", ""),
                                 source_pointer=json_pointer(("classical", str(model_index), "datasets", "L1_AWGN_NO_CLUTTER", str(item_index))), metric_payload_json=text_json(item)))


def add_r7e_nonrun_rows(rows: list[dict[str, Any]]) -> None:
    methods = [
        ("ATI_phase_interferometry", "ATI相位干涉", "NOT_RUN_REQUIRES_SEPARATE_GEOMETRY_PROTOCOL", "NOT_RUN", "ATI / 相位干涉"),
        ("CA_CFAR_dense_cell", "CA-CFAR逐CUT检测", "NOT_RUN_REQUIRES_DENSE_CELL_PROTOCOL", "NOT_RUN", "CA/OS-CFAR (Rohling)"),
        ("OS_CFAR_dense_cell", "OS-CFAR逐CUT检测", "NOT_RUN_REQUIRES_DENSE_CELL_PROTOCOL", "NOT_RUN", "CA/OS-CFAR (Rohling)"),
        ("Kelly_GLRT", "Kelly自适应GLRT", "NOT_RUN_SECONDARY_SAMPLE_AND_TARGET_STEERING_NOT_QUALIFIED", "NOT_RUN", "Kelly GLRT / AMF / ACE"),
        ("AMF_ACE", "AMF/ACE自适应检测", "NOT_RUN_COVARIANCE_AND_STEERING_ASSUMPTIONS_NOT_QUALIFIED", "NOT_RUN", "Kelly GLRT / AMF / ACE"),
        ("full_STAP_SMI_390D", "全秩390维STAP/SMI", "NOT_APPLICABLE_CURRENT_TRAINING_SUPPORT", "NOT_APPLICABLE", "full STAP / SMI"),
        ("reduced_rank_post_Doppler_STAP", "Post-Doppler降秩STAP", "DEFERRED_UNTIL_CLUTTER_SUBSPACE_IS_IDENTIFIABLE", "NOT_RUN", "post-Doppler / 降秩STAP"),
        ("knowledge_aided_covariance", "知识辅助协方差处理", "NOT_APPLICABLE_CURRENT_INPUTS", "NOT_RUN", "Knowledge-aided covariance"),
        ("APX15_7kg", "7kg APX15工程参考", "NOT_QUALIFIED_NOT_RUN", "NOT_RUN", "7kg APX15 engineering pipeline"),
    ]
    for method_id, method_name, applicability, status, source_target in methods:
        rows.append(base_row(
            record_grain="APPLICABILITY", study="D2-A-R7", experiment="R7-E",
            dataset="R7 declared methods and inputs", split="applicability inventory; no new performance evaluation",
            metric_group="CLASSICAL_METHOD_APPLICABILITY", method_id=method_id, method_name=method_name,
            method_role="CLASSICAL_REFERENCE_OR_ENGINEERING_REFERENCE", task_scope="detection_or_parameter_method",
            false_alarm_unit="NOT_APPLICABLE_NO_RUN", detection_workpoint="NOT_APPLICABLE_NO_RUN",
            pfa_cpi="NOT_RUN", pfa_cell="NOT_RUN", pfa_cell_verification_status="NOT_RUN",
            pd="NOT_RUN", auc="NOT_RUN", range_rmse_cells="NOT_RUN", doppler_rmse_hz="NOT_RUN",
            joint_success="NOT_RUN", applicability_status=applicability,
            comparability_status="NO_PERFORMANCE_RANKING_NO_EVALUATION",
            metric_status=status, source_protocol="experiments/D2-A-R7/R7-E/protocol.md",
            source_analysis=rel(R7E_REFERENCE_MATRIX),
            source_pointer=markdown_line_pointer(R7E_REFERENCE_MATRIX, source_target),
            metric_payload_json=text_json({"applicability_status": applicability, "performance_status": status}),
        ))


def main() -> None:
    rows: list[dict[str, Any]] = []
    add_legacy_rows(rows)
    add_d0(rows)
    for csv_path in (ROOT / "experiments/D1-E001/result.csv", ROOT / "experiments/D1-E002/blind/result.csv"):
        if csv_path.exists():
            add_csv_rows(rows, csv_path)
    for json_path in (ROOT / "experiments/D1-E001/formal/metrics.json", ROOT / "experiments/D1-E002/blind/metrics.json"):
        if json_path.exists():
            add_json_metric_file(rows, json_path)
    for folder in (
        ROOT / "experiments/D1-R1",
        ROOT / "experiments/D2-A-E001",
        ROOT / "experiments/D2-A-R3",
        ROOT / "experiments/D2-A-R4",
        ROOT / "experiments/D2-A-E002",
        ROOT / "experiments/D2-A-E003",
        ROOT / "experiments/D2-A-T001",
        ROOT / "experiments/D2-A-R5",
        ROOT / "experiments/D2-A-R6",
    ):
        for path in sorted(folder.glob("**/metrics.json")):
            add_json_metric_file(rows, path)
    add_r7b_physics(rows)
    add_r7_rows(rows)
    add_r7e_nonrun_rows(rows)
    rows.sort(key=lambda r: (str(r["study"]), str(r["experiment"]), str(r["method_id"]), str(r["record_grain"]), str(r["condition_key"]), str(r["snr_db"]), str(r["source_pointer"])))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {rel(OUTPUT)}")
    print("grain counts:", {k: sum(r['record_grain']==k for r in rows) for k in sorted({r['record_grain'] for r in rows})})
    print("stage counts:", {k: sum(r['study']==k for r in rows) for k in sorted({r['study'] for r in rows})})


if __name__ == "__main__":
    main()
