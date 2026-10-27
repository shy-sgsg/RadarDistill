"""Independent streamed evaluation for the D2-A-R5-C teaching pilot."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

from radardistill.baselines import wrap_frequency
from radardistill.distillation.task_losses import decode_task_outputs
from radardistill.evaluation.d2a_e001 import empirical_auc
from radardistill.evaluation.d2a_e002 import (
    calibrate_student_threshold,
    exact_binomial_upper,
    paired_difference_interval,
    summarize_h0_verification,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.training.d2a_r5c_teaching_pilot import (
    _require_training_authorization,
    build_split_seed_specs,
    validate_r5c_teaching_config,
)


_H1_FIELDS = (
    "h1_score",
    "truth_range_cell",
    "predicted_range_cell",
    "predicted_range_bin",
    "truth_doppler_hz",
    "predicted_doppler_hz",
    "true_range_gated_doppler_hz",
)
_METADATA_FIELDS = (
    "base_scene_ids",
    "course_level",
    "scenario_id",
    "snr_db",
    "requested_cnr_db",
)


def _finite_vector(data: Mapping[str, Any], key: str, *, dtype=np.float64) -> np.ndarray:
    if key not in data:
        raise ValueError(f"R5-C teaching prediction is missing {key}")
    values = np.asarray(data[key], dtype=dtype).reshape(-1)
    if values.size == 0:
        raise ValueError(f"R5-C teaching prediction {key} must be nonempty")
    if np.issubdtype(values.dtype, np.number) and np.any(~np.isfinite(values)):
        raise ValueError(f"R5-C teaching prediction {key} must be finite")
    return values


def _scene_ids(data: Mapping[str, Any], *, key: str = "base_scene_ids") -> np.ndarray:
    ids = _finite_vector(data, key, dtype=str).astype(str, copy=False)
    if np.any(ids == ""):
        raise ValueError("R5-C teaching base_scene_ids must be nonempty strings")
    if np.unique(ids).size != ids.size:
        raise ValueError("R5-C teaching requires a unique base_scene_id per prediction")
    return ids


def _validate_h1_payload(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    for name in _METADATA_FIELDS:
        values[name] = _scene_ids(data) if name == "base_scene_ids" else _finite_vector(
            data, name, dtype=str if name in {"course_level", "scenario_id"} else np.float64
        )
    for name in _H1_FIELDS:
        values[name] = _finite_vector(
            data, name, dtype=np.int64 if name == "predicted_range_bin" else np.float64
        )
    count = values["base_scene_ids"].size
    if any(array.size != count for array in values.values()):
        raise ValueError("R5-C teaching H1 predictions and metadata must have equal lengths")
    return values


def _condition_key(course: str, scenario: str, snr_db: float, cnr_db: float) -> str:
    return f"{course}|{scenario}|snr={float(snr_db):g}|cnr={float(cnr_db):g}"


def _condition_scores(split: Mapping[str, Any]) -> dict[str, np.ndarray]:
    scores = _finite_vector(split, "h0_score")
    metadata = {
        name: _finite_vector(
            split,
            name,
            dtype=str if name in {"course_level", "scenario_id"} else np.float64,
        )
        for name in ("course_level", "scenario_id", "snr_db", "requested_cnr_db")
    }
    if any(values.size != scores.size for values in metadata.values()):
        raise ValueError("R5-C teaching H0 metadata and scores must have equal lengths")
    keys = np.asarray(
        [
            _condition_key(course, scenario, snr_db, cnr_db)
            for course, scenario, snr_db, cnr_db in zip(
                metadata["course_level"],
                metadata["scenario_id"],
                metadata["snr_db"],
                metadata["requested_cnr_db"],
                strict=True,
            )
        ],
        dtype=str,
    )
    return {key: scores[keys == key] for key in sorted(set(keys.tolist()))}


def _condition_weighted_auc(
    positive_scores: np.ndarray,
    condition_keys: np.ndarray,
    mask: np.ndarray,
    negative_scores_by_condition: Mapping[str, np.ndarray],
) -> float:
    selected_keys = condition_keys[mask]
    selected_scores = positive_scores[mask]
    if selected_scores.size == 0:
        raise ValueError("R5-C teaching cannot compute AUC for an empty H1 group")
    weighted_auc = 0.0
    for key in sorted(set(selected_keys.tolist())):
        negative = negative_scores_by_condition.get(key)
        if negative is None:
            raise ValueError(f"R5-C teaching AUC lacks independent H0 verification for {key}")
        positive = selected_scores[selected_keys == key]
        weighted_auc += empirical_auc(negative, positive) * positive.size
    return float(weighted_auc / selected_scores.size)


def _mask_summary(
    values: Mapping[str, np.ndarray], mask: np.ndarray, threshold: float,
    *, prf_hz: float, range_gate_bins: float, doppler_gate_hz: float,
    condition_keys: np.ndarray,
    negative_scores_by_condition: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    if not np.any(mask):
        raise ValueError("R5-C teaching cannot summarize an empty evaluation group")
    h1_score = values["h1_score"][mask]
    detected = h1_score >= threshold
    range_error = values["predicted_range_cell"][mask] - values["truth_range_cell"][mask]
    truth_range_bin = np.floor(values["truth_range_cell"][mask]).astype(np.int64)
    doppler_error = np.asarray(
        wrap_frequency(
            values["predicted_doppler_hz"][mask] - values["truth_doppler_hz"][mask],
            prf_hz,
        ),
        dtype=np.float64,
    )
    true_range_doppler_error = np.asarray(
        wrap_frequency(
            values["true_range_gated_doppler_hz"][mask]
            - values["truth_doppler_hz"][mask],
            prf_hz,
        ),
        dtype=np.float64,
    )
    joint = (
        detected
        & (np.abs(range_error) <= range_gate_bins)
        & (np.abs(doppler_error) <= doppler_gate_hz)
    )
    return {
        "n_base_scenes": int(np.count_nonzero(mask)),
        "auc": _condition_weighted_auc(
            values["h1_score"], condition_keys, mask, negative_scores_by_condition
        ),
        "pd": float(np.mean(detected)),
        "range_mae_all_h1_cells": float(np.mean(np.abs(range_error))),
        "range_rmse_all_h1_cells": float(np.sqrt(np.mean(range_error**2))),
        "range_bias_all_h1_cells": float(np.mean(range_error)),
        "range_cell_mismatch_rate": float(
            np.mean(values["predicted_range_bin"][mask] != truth_range_bin)
        ),
        "doppler_mae_all_h1_hz": float(np.mean(np.abs(doppler_error))),
        "doppler_rmse_all_h1_hz": float(np.sqrt(np.mean(doppler_error**2))),
        "doppler_bias_all_h1_hz": float(np.mean(doppler_error)),
        "true_range_gated_doppler_mae_hz": float(np.mean(np.abs(true_range_doppler_error))),
        "true_range_gated_doppler_rmse_hz": float(
            np.sqrt(np.mean(true_range_doppler_error**2))
        ),
        "true_range_gated_doppler_bias_hz": float(np.mean(true_range_doppler_error)),
        "joint_success": float(np.mean(joint)),
        "joint_success_count": int(np.sum(joint)),
        "detected_count": int(np.sum(detected)),
        "missed_count": int(detected.size - np.sum(detected)),
        "misses_count_as_joint_failure": True,
        "range_gate_bins": float(range_gate_bins),
        "doppler_gate_hz": float(doppler_gate_hz),
    }


def _summarize_h0_groups(
    scores: np.ndarray, metadata: Mapping[str, Any], threshold: float
) -> dict[str, Any]:
    base = summarize_h0_verification(scores, threshold)
    if not metadata:
        return {"overall": base}
    values = {
        name: _finite_vector(metadata, name, dtype=str if name in {"course_level", "scenario_id"} else np.float64)
        for name in ("course_level", "scenario_id", "snr_db", "requested_cnr_db")
    }
    if any(array.size != scores.size for array in values.values()):
        raise ValueError("R5-C teaching H0 metadata and scores must have equal lengths")
    result: dict[str, Any] = {"overall": base, "by_course": {}, "c3_cells": {}}
    for course in sorted(set(values["course_level"].tolist())):
        mask = values["course_level"] == course
        result["by_course"][str(course)] = summarize_h0_verification(scores[mask], threshold)
    c3 = values["course_level"] == "C3"
    condition_keys = np.asarray(
        [
            _condition_key(course, scenario, snr, cnr)
            for course, scenario, snr, cnr in zip(
                values["course_level"],
                values["scenario_id"],
                values["snr_db"],
                values["requested_cnr_db"],
                strict=True,
            )
        ],
        dtype=str,
    )
    for key in sorted(set(condition_keys[c3].tolist())):
        mask = c3 & (condition_keys == key)
        result["c3_cells"][key] = summarize_h0_verification(scores[mask], threshold)
    return result


def summarize_pilot_arm(
    predictions: Mapping[str, Any],
    calibration: Mapping[str, Any],
    verification: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize one arm using its independent H0 calibration and verification."""
    evaluation = config["evaluation"]
    calibration_scores = _finite_vector(calibration, "h0_score")
    threshold_record = calibrate_student_threshold(
        calibration_scores, float(evaluation["threshold_target_pfa_cpi"])
    )
    threshold_record["score_type"] = str(evaluation["score_type"])
    threshold = float(threshold_record["threshold"])
    if "overall" not in verification or "c3" not in verification:
        raise ValueError("R5-C teaching verification requires overall and C3 H0 splits")
    verification_result: dict[str, Any] = {}
    for split_name in ("overall", "c3"):
        split = verification[split_name]
        scores = _finite_vector(split, "h0_score")
        metadata = {
            name: split[name]
            for name in ("course_level", "scenario_id", "snr_db", "requested_cnr_db")
            if name in split
        }
        verification_result[split_name] = _summarize_h0_groups(
            scores, metadata, threshold
        )
    h0_scores_by_condition = _condition_scores(verification["overall"])
    c3_h0_by_condition = _condition_scores(verification["c3"])
    if any(not key.startswith("C3|") for key in c3_h0_by_condition):
        raise ValueError("R5-C teaching C3 H0 verification contains a non-C3 condition")
    h0_scores_by_condition.update(c3_h0_by_condition)

    if "development" not in predictions:
        raise ValueError("R5-C teaching predictions require an H1 development split")
    development = _validate_h1_payload(predictions["development"])
    prf_hz = float(config["radar"]["prf_hz"])
    condition_keys = np.asarray(
        [
            _condition_key(course, scenario, snr, cnr)
            for course, scenario, snr, cnr in zip(
                development["course_level"],
                development["scenario_id"],
                development["snr_db"],
                development["requested_cnr_db"],
                strict=True,
            )
        ],
        dtype=str,
    )
    metric_kwargs = {
        "condition_keys": condition_keys,
        "negative_scores_by_condition": h0_scores_by_condition,
    }
    metrics = _mask_summary(
        development,
        np.ones(development["base_scene_ids"].size, dtype=bool),
        threshold,
        prf_hz=prf_hz,
        range_gate_bins=float(evaluation["range_gate_bins"]),
        doppler_gate_hz=float(evaluation["doppler_gate_hz"]),
        **metric_kwargs,
    )
    identifiable_mask = np.isin(development["course_level"], ("C2", "C3"))
    identifiable_metrics = _mask_summary(
        development,
        identifiable_mask,
        threshold,
        prf_hz=prf_hz,
        range_gate_bins=float(evaluation["range_gate_bins"]),
        doppler_gate_hz=float(evaluation["doppler_gate_hz"]),
        **metric_kwargs,
    )
    course_metrics: dict[str, Any] = {}
    courses = development["course_level"]
    for course in sorted(set(courses.tolist())):
        course_metrics[str(course)] = _mask_summary(
            development,
            courses == course,
            threshold,
            prf_hz=prf_hz,
            range_gate_bins=float(evaluation["range_gate_bins"]),
            doppler_gate_hz=float(evaluation["doppler_gate_hz"]),
            **metric_kwargs,
        )
    c3_cells: dict[str, Any] = {}
    c3_mask = courses == "C3"
    for key in sorted(set(condition_keys[c3_mask].tolist())):
        c3_cells[key] = _mask_summary(
            development,
            c3_mask & (condition_keys == key),
            threshold,
            prf_hz=prf_hz,
            range_gate_bins=float(evaluation["range_gate_bins"]),
            doppler_gate_hz=float(evaluation["doppler_gate_hz"]),
            **metric_kwargs,
        )

    return {
        "threshold_calibration": threshold_record,
        "h0_verification": verification_result,
        "development": {
            "overall": metrics,
            "identifiable_C2_C3": identifiable_metrics,
            "by_course": course_metrics,
            "c3_cells": c3_cells,
        },
        "development_split_is_blind": False,
    }


def iter_generated_split_chunks(
    config: Mapping[str, Any], physics_config: Mapping[str, Any], split_name: str
) -> Iterable[dict[str, Any]]:
    """Generate one frozen split in bounded chunks without retaining complete IQ."""
    _require_training_authorization(config)
    split_config = config["data"]["splits"].get(split_name)
    if split_config is None:
        raise ValueError(f"unknown R5-C teaching split: {split_name}")
    split_specs = [
        spec for spec in build_split_seed_specs(config) if spec["split"] == split_name
    ]
    if sum(int(spec["scene_count"]) for spec in split_specs) != int(
        split_config["scene_count"]
    ):
        raise RuntimeError(f"R5-C teaching split {split_name} scene count changed")
    for spec in split_specs:
        condition = spec["condition"]
        batch = generate_d2a_paired_batch(
            int(spec["scene_count"]),
            int(spec["seed"]),
            float(condition["snr_db"]),
            float(condition["cnr_db"]),
            dict(physics_config),
            scenario_id=str(condition["scenario_id"]),
        )
        if len(batch.base_scene_id) != int(spec["scene_count"]):
            raise RuntimeError("R5-C teaching generator returned a wrong chunk size")
        metadata = {
            "base_scene_ids": np.asarray(batch.base_scene_id).astype(str),
            "course_level": np.asarray(
                [condition["course"]] * len(batch.base_scene_id), dtype=str
            ),
            "scenario_id": np.asarray(
                [condition["scenario_id"]] * len(batch.base_scene_id), dtype=str
            ),
            "snr_db": np.asarray(batch.snr_db, dtype=np.float64),
            "requested_cnr_db": np.asarray(batch.requested_cnr_db, dtype=np.float64),
            "condition_index": np.full(
                len(batch.base_scene_id), spec["source_condition_index"], dtype=np.int64
            ),
        }
        yield {"spec": spec, "batch": batch, "metadata": metadata}


def predict_task_student_chunk(
    model: torch.nn.Module,
    iq: np.ndarray,
    device: torch.device,
    *,
    truth_range_cell: np.ndarray | None = None,
    prf_hz: float = 1300.0,
) -> dict[str, np.ndarray]:
    """Predict task outputs from a bounded complex-IQ chunk on the requested device."""
    values = np.asarray(iq)
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32) or not np.iscomplexobj(values):
        raise ValueError("R5-C teaching IQ must have shape [B,3,130,32] and be complex")
    if values.shape[0] == 0 or np.any(~np.isfinite(values)):
        raise ValueError("R5-C teaching IQ chunk must be nonempty and finite")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R5-C teaching evaluation requires the visible CUDA device")
    iq_ri = np.concatenate((values.real, values.imag), axis=1).astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(iq_ri)).to(device)
    model.eval()
    with torch.inference_mode():
        outputs = model(tensor, compute_spectrum=False)
        decoded = decode_task_outputs(outputs, prf_hz)
        predicted_bin = outputs["range_logits"].argmax(dim=-1)
        result = {
            "score": outputs["presence_logit"].float().cpu().numpy().reshape(-1),
            "predicted_range_cell": decoded["range_cell"].float().cpu().numpy().reshape(-1),
            "predicted_range_bin": predicted_bin.cpu().numpy().reshape(-1),
            "predicted_doppler_hz": decoded["doppler_hz"].float().cpu().numpy().reshape(-1),
        }
        if truth_range_cell is not None:
            truth_range = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
            if truth_range.size != values.shape[0] or np.any(~np.isfinite(truth_range)):
                raise ValueError("R5-C teaching true-range gate requires one finite truth per IQ")
            truth_bins = np.floor(truth_range).astype(np.int64)
            if np.any(truth_bins < 0) or np.any(truth_bins >= 32):
                raise ValueError("R5-C teaching truth range falls outside the 32-bin grid")
            rows = torch.arange(values.shape[0], device=device)
            cells = torch.as_tensor(truth_bins, dtype=torch.long, device=device)
            vectors = outputs["doppler_vector"][rows, cells]
            angles = torch.atan2(vectors[:, 1], vectors[:, 0])
            true_range_doppler = torch.remainder(
                angles * (prf_hz / (2.0 * torch.pi)) + prf_hz / 2.0, prf_hz
            ) - prf_hz / 2.0
            result["true_range_gated_doppler_hz"] = (
                true_range_doppler.float().cpu().numpy().reshape(-1)
            )
    return result


def _aligned_by_scene(
    reference_ids: np.ndarray, data: Mapping[str, Any], *, label: str
) -> dict[str, np.ndarray]:
    values = _validate_h1_payload(data)
    ids = values["base_scene_ids"]
    if set(ids.tolist()) != set(reference_ids.tolist()):
        raise ValueError(f"R5-C teaching arms must have the same base_scene_id set for {label}")
    positions = {scene_id: index for index, scene_id in enumerate(ids.tolist())}
    order = np.asarray([positions[scene_id] for scene_id in reference_ids.tolist()], dtype=np.int64)
    return {name: array[order] for name, array in values.items()}


def _metadata_equal(
    first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray], *, label: str
) -> None:
    for name in ("course_level", "scenario_id", "snr_db", "requested_cnr_db", "truth_range_cell", "truth_doppler_hz"):
        if not np.array_equal(first[name], second[name]):
            raise ValueError(f"R5-C teaching arms have mismatched {name} for {label}")


def _paired_bootstrap_summary(
    first: np.ndarray,
    second: np.ndarray,
    scene_ids: np.ndarray,
    *,
    replicates: int,
    seed: int,
    statistic=None,
) -> tuple[float, list[float]]:
    values_a = np.asarray(first, dtype=np.float64).reshape(-1)
    values_b = np.asarray(second, dtype=np.float64).reshape(-1)
    clusters = np.asarray(scene_ids).astype(str).reshape(-1)
    if values_a.size == 0 or values_a.shape != values_b.shape or values_a.size != clusters.size:
        raise ValueError("R5-C teaching paired bootstrap inputs must have equal nonzero size")
    if np.any(~np.isfinite(values_a)) or np.any(~np.isfinite(values_b)):
        raise ValueError("R5-C teaching paired bootstrap values must be finite")
    if np.unique(clusters).size != clusters.size:
        raise ValueError("R5-C teaching bootstrap requires unique base_scene_id rows")
    if statistic is None:
        statistic = lambda a, b: float(np.mean(a - b))
    point = float(statistic(values_a, values_b))
    interval = paired_difference_interval(
        values_a,
        values_b,
        clusters,
        statistic=statistic,
        replicates=replicates,
        seed=seed,
        confidence=0.95,
    )
    return point, [float(interval[0]), float(interval[1])]


def _rmse_ratio(candidate_error: np.ndarray, reference_error: np.ndarray) -> float:
    candidate = float(np.sqrt(np.mean(np.asarray(candidate_error, dtype=np.float64) ** 2)))
    reference = float(np.sqrt(np.mean(np.asarray(reference_error, dtype=np.float64) ** 2)))
    if reference == 0.0:
        return 1.0 if candidate == 0.0 else float("inf")
    return candidate / reference


def _arm_contrast_payload(
    arm: str,
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    candidate_c3_scores: np.ndarray,
    candidate_threshold: float,
    reference_c3_scores: np.ndarray,
    reference_threshold: float,
    c3_ids: np.ndarray,
    *,
    config: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    reps = int(evaluation["bootstrap_replicates"])
    seed = int(evaluation["bootstrap_seed"]) + seed_offset
    all_ids = candidate["base_scene_ids"]
    identifiable = np.isin(candidate["course_level"], ("C2", "C3"))
    if not np.any(identifiable):
        raise ValueError("R5-C teaching primary C2+C3 scope is empty")
    cand_detected = candidate["detected"]
    ref_detected = reference["detected"]
    cand_joint = candidate["joint_success"]
    ref_joint = reference["joint_success"]

    def contrast(
        name: str,
        a: np.ndarray,
        b: np.ndarray,
        offset: int,
        *,
        statistic=None,
        scene_ids: np.ndarray | None = None,
    ):
        point, interval = _paired_bootstrap_summary(
            a,
            b,
            all_ids if scene_ids is None else scene_ids,
            replicates=reps,
            seed=seed + offset,
            statistic=statistic,
        )
        return {"point_difference": point, "nominal_95_ci": interval}

    primary_scope = contrast(
        "primary",
        cand_joint[identifiable].astype(np.float64),
        ref_joint[identifiable].astype(np.float64),
        11,
        scene_ids=all_ids[identifiable],
    )
    primary_scope["n_paired_scenes"] = int(np.sum(identifiable))
    primary_scope["simultaneous_one_sided_97_5_lower_bound"] = primary_scope[
        "nominal_95_ci"
    ][0]
    primary_scope["minimum_gain"] = float(evaluation["minimum_primary_gain"])
    primary_scope["passed"] = bool(
        primary_scope["point_difference"] >= float(evaluation["minimum_primary_gain"])
        and primary_scope["simultaneous_one_sided_97_5_lower_bound"] > 0.0
    )

    overall_pd = contrast(
        "overall_pd", cand_detected.astype(float), ref_detected.astype(float), 23
    )
    overall_joint = contrast("overall_joint", cand_joint.astype(float), ref_joint.astype(float), 37)
    c2c3_joint = primary_scope

    candidate_range = candidate["range_error"][identifiable]
    reference_range = reference["range_error"][identifiable]
    candidate_doppler = candidate["doppler_error"][identifiable]
    reference_doppler = reference["doppler_error"][identifiable]
    range_ratio = _rmse_ratio(candidate_range, reference_range)
    range_ratio_stat = lambda a, b: _rmse_ratio(a, b)
    _, range_ratio_ci = _paired_bootstrap_summary(
        candidate_range,
        reference_range,
        all_ids[identifiable],
        replicates=reps,
        seed=seed + 53,
        statistic=range_ratio_stat,
    )
    doppler_ratio = _rmse_ratio(candidate_doppler, reference_doppler)
    doppler_ratio_stat = lambda a, b: _rmse_ratio(a, b)
    _, doppler_ratio_ci = _paired_bootstrap_summary(
        candidate_doppler,
        reference_doppler,
        all_ids[identifiable],
        replicates=reps,
        seed=seed + 67,
        statistic=doppler_ratio_stat,
    )

    cand_false_alarm = candidate_c3_scores >= candidate_threshold
    ref_false_alarm = reference_c3_scores >= reference_threshold
    c3_pfa_point, c3_pfa_ci = _paired_bootstrap_summary(
        cand_false_alarm.astype(float),
        ref_false_alarm.astype(float),
        c3_ids,
        replicates=reps,
        seed=seed + 79,
    )
    c3_pfa_diff = {
        "point_difference": c3_pfa_point,
        "nominal_95_ci": c3_pfa_ci,
        "nominal_95_upper": c3_pfa_ci[1],
    }

    protections = {
        "overall_pd": {
            **overall_pd,
            "minimum_point_difference": float(evaluation["minimum_pd_difference_point"]),
            "minimum_nominal_95_lower": float(evaluation["minimum_protection_ci_lower"]),
        },
        "overall_joint": {
            **overall_joint,
            "minimum_point_difference": float(evaluation["minimum_joint_difference_point"]),
            "minimum_nominal_95_lower": float(evaluation["minimum_protection_ci_lower"]),
        },
        "C2_C3_joint": {
            "point_difference": c2c3_joint["point_difference"],
            "nominal_95_ci": c2c3_joint["nominal_95_ci"],
            "minimum_nominal_95_lower": float(evaluation["minimum_c2_c3_joint_ci_lower"]),
        },
        "C2_C3_range_rmse_ratio": {
            "point_ratio": float(range_ratio),
            "nominal_95_ci": range_ratio_ci,
            "maximum_point_ratio": float(evaluation["maximum_c2_c3_rmse_ratio_point"]),
            "maximum_nominal_95_upper": float(
                evaluation["maximum_c2_c3_rmse_ratio_nominal_95_upper"]
            ),
        },
        "C2_C3_doppler_rmse_ratio": {
            "point_ratio": float(doppler_ratio),
            "nominal_95_ci": doppler_ratio_ci,
            "maximum_point_ratio": float(evaluation["maximum_c2_c3_rmse_ratio_point"]),
            "maximum_nominal_95_upper": float(
                evaluation["maximum_c2_c3_rmse_ratio_nominal_95_upper"]
            ),
        },
        "C3_H0_pfa_difference": {
            **c3_pfa_diff,
            "maximum_nominal_95_upper": float(
                evaluation["maximum_c3_h0_pfa_difference_nominal_95_upper"]
            ),
        },
        "C3_H0_pfa_absolute": {
            "point_pfa": float(np.mean(cand_false_alarm)),
            "maximum_point_pfa": float(evaluation["c3_h0_pfa_point_limit"]),
            "exact_one_sided_95_upper": exact_binomial_upper(
                int(np.sum(cand_false_alarm)), int(cand_false_alarm.size)
            ),
        },
    }
    protections["overall_pd"]["passed"] = bool(
        protections["overall_pd"]["point_difference"]
        >= protections["overall_pd"]["minimum_point_difference"]
        and protections["overall_pd"]["nominal_95_ci"][0]
        > protections["overall_pd"]["minimum_nominal_95_lower"]
    )
    protections["overall_joint"]["passed"] = bool(
        protections["overall_joint"]["point_difference"]
        >= protections["overall_joint"]["minimum_point_difference"]
        and protections["overall_joint"]["nominal_95_ci"][0]
        > protections["overall_joint"]["minimum_nominal_95_lower"]
    )
    protections["C2_C3_joint"]["passed"] = bool(
        protections["C2_C3_joint"]["nominal_95_ci"][0]
        > protections["C2_C3_joint"]["minimum_nominal_95_lower"]
    )
    for key in ("C2_C3_range_rmse_ratio", "C2_C3_doppler_rmse_ratio"):
        protections[key]["passed"] = bool(
            protections[key]["point_ratio"] <= protections[key]["maximum_point_ratio"]
            and protections[key]["nominal_95_ci"][1]
            <= protections[key]["maximum_nominal_95_upper"]
        )
    protections["C3_H0_pfa_difference"]["passed"] = bool(
        protections["C3_H0_pfa_difference"]["nominal_95_upper"]
        <= protections["C3_H0_pfa_difference"]["maximum_nominal_95_upper"]
    )
    protections["C3_H0_pfa_absolute"]["passed"] = bool(
        protections["C3_H0_pfa_absolute"]["point_pfa"]
        <= protections["C3_H0_pfa_absolute"]["maximum_point_pfa"]
    )
    protections["passed"] = bool(
        all(
            protections[key]["passed"]
            for key in (
                "overall_pd",
                "overall_joint",
                "C2_C3_joint",
                "C2_C3_range_rmse_ratio",
                "C2_C3_doppler_rmse_ratio",
                "C3_H0_pfa_difference",
                "C3_H0_pfa_absolute",
            )
        )
    )
    return {"primary": primary_scope, "task_protection": protections}


def paired_teaching_contrasts(
    arm_predictions: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Compute the preregistered paired-scene teaching contrasts and protections."""
    validate_r5c_teaching_config(config)
    arms = tuple(str(arm) for arm in config["training"]["arms"])
    if set(arm_predictions) != set(arms):
        raise ValueError("R5-C teaching contrasts require the exact registered arm set")
    thresholds: dict[str, dict[str, Any]] = {}
    development: dict[str, dict[str, np.ndarray]] = {}
    c3_scores: dict[str, np.ndarray] = {}
    c3_ids_by_arm: dict[str, np.ndarray] = {}
    c3_metadata: dict[str, dict[str, np.ndarray]] = {}
    for arm in arms:
        payload = arm_predictions[arm]
        if not all(key in payload for key in ("calibration", "development", "c3_h0_verification")):
            raise ValueError(f"R5-C teaching arm {arm} lacks a required split")
        thresholds[arm] = calibrate_student_threshold(
            _finite_vector(payload["calibration"], "h0_score"),
            float(config["evaluation"]["threshold_target_pfa_cpi"]),
        )
        thresholds[arm]["score_type"] = str(config["evaluation"]["score_type"])
        dev = _validate_h1_payload(payload["development"])
        threshold = float(thresholds[arm]["threshold"])
        truth_range_bin = np.floor(dev["truth_range_cell"]).astype(np.int64)
        range_error = dev["predicted_range_cell"] - dev["truth_range_cell"]
        doppler_error = np.asarray(
            wrap_frequency(
                dev["predicted_doppler_hz"] - dev["truth_doppler_hz"],
                float(config["radar"]["prf_hz"]),
            ),
            dtype=np.float64,
        )
        detected = dev["h1_score"] >= threshold
        joint = (
            detected
            & (np.abs(range_error) <= float(config["evaluation"]["range_gate_bins"]))
            & (np.abs(doppler_error) <= float(config["evaluation"]["doppler_gate_hz"]))
        )
        dev.update(
            {
                "threshold_value": threshold,
                "detected": detected,
                "range_error": range_error,
                "doppler_error": doppler_error,
                "joint_success": joint,
                "truth_range_bin": truth_range_bin,
            }
        )
        development[arm] = dev
        c3 = payload["c3_h0_verification"]
        c3_scores[arm] = _finite_vector(c3, "h0_score")
        c3_ids_by_arm[arm] = _scene_ids(c3)
        if c3_ids_by_arm[arm].size != c3_scores[arm].size:
            raise ValueError(f"R5-C teaching arm {arm} C3 H0 IDs and scores differ in size")
        c3_metadata[arm] = {
            name: _finite_vector(
                c3,
                name,
                dtype=str if name in {"course_level", "scenario_id"} else np.float64,
            )
            for name in ("course_level", "scenario_id", "snr_db", "requested_cnr_db")
        }
        if any(values.size != c3_scores[arm].size for values in c3_metadata[arm].values()):
            raise ValueError(f"R5-C teaching arm {arm} C3 H0 metadata and scores differ in size")

    reference_arm = "T0"
    reference = development[reference_arm]
    reference_ids = reference["base_scene_ids"]
    c3_reference_ids = c3_ids_by_arm[reference_arm]
    aligned: dict[str, dict[str, np.ndarray]] = {reference_arm: reference}
    aligned_c3: dict[str, np.ndarray] = {reference_arm: c3_scores[reference_arm]}
    aligned_c3_metadata: dict[str, dict[str, np.ndarray]] = {
        reference_arm: c3_metadata[reference_arm]
    }
    for arm in arms:
        if arm == reference_arm:
            continue
        aligned[arm] = _aligned_by_scene(
            reference_ids, arm_predictions[arm]["development"], label="H1 development"
        )
        _metadata_equal(reference, aligned[arm], label="H1 development")
        dev = aligned[arm]
        threshold = float(thresholds[arm]["threshold"])
        truth_range_bin = np.floor(dev["truth_range_cell"]).astype(np.int64)
        range_error = dev["predicted_range_cell"] - dev["truth_range_cell"]
        doppler_error = np.asarray(
            wrap_frequency(
                dev["predicted_doppler_hz"] - dev["truth_doppler_hz"],
                float(config["radar"]["prf_hz"]),
            ),
            dtype=np.float64,
        )
        detected = dev["h1_score"] >= threshold
        dev.update(
            {
                "threshold_value": threshold,
                "detected": detected,
                "range_error": range_error,
                "doppler_error": doppler_error,
                "joint_success": (
                    detected
                    & (np.abs(range_error) <= float(config["evaluation"]["range_gate_bins"]))
                    & (np.abs(doppler_error) <= float(config["evaluation"]["doppler_gate_hz"]))
                ),
                "truth_range_bin": truth_range_bin,
            }
        )
        ids = c3_ids_by_arm[arm]
        scores = c3_scores[arm]
        if np.unique(ids).size != ids.size:
            raise ValueError("R5-C teaching requires a unique base_scene_id per C3 H0 prediction")
        if set(ids.tolist()) != set(c3_reference_ids.tolist()):
            raise ValueError("R5-C teaching arms must have the same base_scene_id set for C3 H0 verification")
        lookup = {scene_id: index for index, scene_id in enumerate(ids.tolist())}
        order = np.asarray([lookup[scene_id] for scene_id in c3_reference_ids.tolist()], dtype=np.int64)
        aligned_c3[arm] = scores[order]
        aligned_c3_metadata[arm] = {
            name: values[order] for name, values in c3_metadata[arm].items()
        }
        for name, values in aligned_c3_metadata[arm].items():
            if not np.array_equal(values, aligned_c3_metadata[reference_arm][name]):
                raise ValueError(f"R5-C teaching arms have mismatched C3 H0 {name}")

    primary: dict[str, Any] = {}
    secondary: dict[str, Any] = {}
    protections: dict[str, Any] = {}
    for index, arm in enumerate(("T1", "T2", "T3")):
        if arm not in aligned:
            continue
        contrast = _arm_contrast_payload(
            arm,
            aligned[arm],
            aligned[reference_arm],
            aligned_c3[arm],
            float(thresholds[arm]["threshold"]),
            aligned_c3[reference_arm],
            float(thresholds[reference_arm]["threshold"]),
            c3_reference_ids,
            config=config,
            seed_offset=(index + 1) * 1000,
        )
        key = f"{arm}_minus_T0"
        if arm == "T3":
            secondary[key] = {
                name: value
                for name, value in contrast["primary"].items()
                if name != "passed"
            }
            secondary[key]["role"] = "secondary_descriptive"
        else:
            primary[key] = contrast["primary"]
        protections[arm] = contrast["task_protection"]

    candidate_decisions = {
        arm: bool(primary[f"{arm}_minus_T0"]["passed"] and protections[arm]["passed"])
        for arm in ("T1", "T2")
    }
    return {
        "bootstrap": {
            "unit": "base_scene_id",
            "replicates": int(config["evaluation"]["bootstrap_replicates"]),
            "seed": int(config["evaluation"]["bootstrap_seed"]),
            "nominal_interval": "two_sided_95_percentile",
            "primary_simultaneous_lower": "one_sided_97.5_percentile_lower_bound",
        },
        "thresholds_by_arm": thresholds,
        "primary": primary,
        "secondary": secondary,
        "task_protection": protections,
        "candidate_decisions": candidate_decisions,
        "training_seed_uncertainty_estimated": False,
    }
