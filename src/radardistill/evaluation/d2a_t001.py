"""Scene grouping, paired metrics, and soft-response audits for D2-A-T001."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from radardistill.baselines import wrap_frequency
from radardistill.evaluation.d2a_e001 import cluster_bootstrap_interval
from radardistill.evaluation.d2a_e002 import (
    evaluate_paired_predictions,
    paired_difference_interval,
)

COURSE_LEVELS = ("C0", "C1", "C2", "C3")
SCENARIOS = ("separated", "near_clutter_edge", "partial_overlap")


def build_h0_mixture_assignments(
    count: int,
    mixture_seed: int,
    course_levels: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Create an equal-probability C0-C3 H0 mixture with the frozen C3 SNR split."""
    if count <= 0 or mixture_seed < 0 or tuple(course_levels) != COURSE_LEVELS:
        raise ValueError("H0 mixture requires positive count, seed, and ordered C0-C3 levels")
    rng = np.random.default_rng(int(mixture_seed))
    course = rng.choice(np.asarray(COURSE_LEVELS), size=count)
    snr = np.empty(count, dtype=np.float64)
    cnr = np.empty(count, dtype=np.float64)
    for level in COURSE_LEVELS:
        indices = np.flatnonzero(course == level)
        level_spec = course_levels[level]
        values = np.asarray(level_spec["snr_db"], dtype=np.float64).reshape(-1)
        if level == "C3":
            if values.size != 2:
                raise ValueError("C3 must have exactly two preregistered SNR values")
            snr[indices] = rng.choice(values, size=indices.size)
        else:
            if values.size != 1:
                raise ValueError(f"{level} must have one preregistered SNR value")
            snr[indices] = values[0]
        cnr[indices] = float(level_spec["cnr_db"])
    return {"course_level": course, "snr_db": snr, "cnr_db": cnr}


def build_development_assignments(
    config: Mapping[str, Any],
    e002_course_levels: Mapping[str, Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    """Expand and deterministically shuffle the frozen T001 paired development cells."""
    development = config["data"]["development"]
    assignment_seed = int(development["assignment_seed"])
    rows: dict[str, list[tuple[str, float, float, str]]] = {
        key: [] for key in ("course_level", "scenario_id", "snr_db", "cnr_db", "condition_id")
    }
    for level_index, level in enumerate(COURSE_LEVELS):
        level_spec = e002_course_levels[level]
        counts = development["counts"][level]
        level_rows: list[tuple[str, float, float, str]] = []
        for cell_name, raw_count in counts.items():
            count = int(raw_count)
            if count < 0:
                raise ValueError("development cell counts must be nonnegative")
            if "_snr_" in cell_name:
                scenario, snr_text = cell_name.rsplit("_snr_", 1)
                snr_db = float(snr_text)
            else:
                scenario = str(cell_name)
                snr_values = np.asarray(level_spec["snr_db"], dtype=np.float64).reshape(-1)
                if snr_values.size != 1:
                    raise ValueError(f"development cell {cell_name} does not identify a unique SNR")
                snr_db = float(snr_values[0])
            if scenario not in SCENARIOS:
                raise ValueError(f"unknown development scenario {scenario!r}")
            cnr_db = float(level_spec["cnr_db"])
            condition = f"{level}_{scenario}_snr_{int(snr_db)}_cnr_{int(cnr_db)}"
            level_rows.extend([(scenario, snr_db, cnr_db, condition)] * count)
        if len(level_rows) != int(development["per_course"]):
            raise ValueError(f"{level} development count differs from the frozen per-course total")
        order = np.random.default_rng(assignment_seed + level_index).permutation(len(level_rows))
        for index in order:
            scenario, snr_db, cnr_db, condition = level_rows[int(index)]
            rows["course_level"].append(level)
            rows["scenario_id"].append(scenario)
            rows["snr_db"].append(snr_db)
            rows["cnr_db"].append(cnr_db)
            rows["condition_id"].append(condition)
    expected = int(development["paired_base_scenes"])
    if any(len(value) != expected for value in rows.values()):
        raise ValueError("expanded development assignments do not match the frozen total")
    return {
        "course_level": np.asarray(rows["course_level"], dtype="U2"),
        "scenario_id": np.asarray(rows["scenario_id"], dtype="U32"),
        "snr_db": np.asarray(rows["snr_db"], dtype=np.float64),
        "cnr_db": np.asarray(rows["cnr_db"], dtype=np.float64),
        "condition_id": np.asarray(rows["condition_id"], dtype="U96"),
    }


def summarize_soft_response_per_scene(
    response_h0: np.ndarray,
    response_h1: np.ndarray,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    *,
    prf_hz: float,
    range_error_bins_max: float = 1.0,
    doppler_error_hz_max: float = 20.0,
) -> dict[str, np.ndarray]:
    """Measure paired target-window mass, localization, entropy, and peak margins."""
    h0 = np.asarray(response_h0, dtype=np.float64)
    h1 = np.asarray(response_h1, dtype=np.float64)
    ranges = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    dopplers = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    if h0.ndim == 2:
        h0 = h0[None, ...]
    if h1.ndim == 2:
        h1 = h1[None, ...]
    if h0.shape != h1.shape or h0.shape != (ranges.size, 130, 32) or dopplers.size != ranges.size:
        raise ValueError("soft maps and truth must have matching [B,130,32] dimensions")
    if ranges.size == 0 or prf_hz <= 0.0 or not np.isfinite(prf_hz):
        raise ValueError("soft-response summary needs scenes and a positive finite PRF")
    if any(np.any(~np.isfinite(value)) for value in (h0, h1, ranges, dopplers)):
        raise ValueError("soft-response inputs must be finite")
    if np.any(h0 < 0.0) or np.any(h1 < 0.0):
        raise ValueError("soft response must be nonnegative")
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(130, d=1.0 / prf_hz))
    range_axis = np.arange(32, dtype=np.float64)
    output: dict[str, np.ndarray] = {
        "h0_target_mass": np.empty(ranges.size),
        "h1_target_mass": np.empty(ranges.size),
        "h1_target_peak_margin": np.empty(ranges.size),
        "h0_top_cell_localized": np.empty(ranges.size, dtype=np.float64),
        "h1_top_cell_localized": np.empty(ranges.size, dtype=np.float64),
        "h0_normalized_entropy": np.empty(ranges.size),
        "h1_normalized_entropy": np.empty(ranges.size),
        "h0_max_score": np.empty(ranges.size),
        "h1_max_score": np.empty(ranges.size),
    }
    for index in range(ranges.size):
        fd_error = np.remainder(
            frequency_axis - dopplers[index] + prf_hz / 2.0, prf_hz
        ) - prf_hz / 2.0
        mask = (
            (np.abs(range_axis[None, :] - ranges[index]) <= range_error_bins_max)
            & (np.abs(fd_error[:, None]) <= doppler_error_hz_max)
        )
        for name, response in (("h0", h0[index]), ("h1", h1[index])):
            output[f"{name}_target_mass"][index] = float(np.sum(response[mask]))
            peak_flat = int(np.argmax(response))
            peak_k, peak_r = divmod(peak_flat, 32)
            peak_fd_error = float(
                np.remainder(frequency_axis[peak_k] - dopplers[index] + prf_hz / 2.0, prf_hz)
                - prf_hz / 2.0
            )
            output[f"{name}_top_cell_localized"][index] = float(
                abs(peak_r - ranges[index]) <= range_error_bins_max
                and abs(peak_fd_error) <= doppler_error_hz_max
            )
            flat = response.reshape(-1)
            total = float(np.sum(flat))
            if total > 0.0:
                probability = flat / total
                nonzero = probability > 0.0
                entropy = -float(np.sum(probability[nonzero] * np.log(probability[nonzero])))
                output[f"{name}_normalized_entropy"][index] = entropy / np.log(flat.size)
            else:
                output[f"{name}_normalized_entropy"][index] = 0.0
            output[f"{name}_max_score"][index] = float(np.max(response))
        target_peak = float(np.max(h1[index][mask])) if np.any(mask) else 0.0
        false_peak = float(np.max(h1[index][~mask])) if np.any(~mask) else 0.0
        output["h1_target_peak_margin"][index] = target_peak - false_peak
    return output


def summarize_soft_response(
    per_scene: Mapping[str, np.ndarray],
    base_scene_ids: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    ids = np.asarray(base_scene_ids).reshape(-1)
    if ids.size == 0 or any(np.asarray(value).shape != ids.shape for value in per_scene.values()):
        raise ValueError("per-scene soft-response metrics must match nonempty base-scene IDs")
    lift = np.asarray(per_scene["h1_target_mass"]) - np.asarray(per_scene["h0_target_mass"])
    lift_ci = cluster_bootstrap_interval(
        lift,
        ids,
        statistic=np.mean,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "n_paired_scenes": int(ids.size),
        "target_window_mass_h0_mean": float(np.mean(per_scene["h0_target_mass"])),
        "target_window_mass_h1_mean": float(np.mean(per_scene["h1_target_mass"])),
        "target_window_mass_lift_mean": float(np.mean(lift)),
        "target_window_mass_lift_ci95": [float(value) for value in lift_ci],
        "teacher_top_cell_localization_rate": float(np.mean(per_scene["h1_top_cell_localized"])),
        "h0_top_cell_localization_rate": float(np.mean(per_scene["h0_top_cell_localized"])),
        "h0_normalized_entropy_mean": float(np.mean(per_scene["h0_normalized_entropy"])),
        "h1_normalized_entropy_mean": float(np.mean(per_scene["h1_normalized_entropy"])),
        "h0_max_score_mean": float(np.mean(per_scene["h0_max_score"])),
        "h1_max_score_mean": float(np.mean(per_scene["h1_max_score"])),
        "h1_target_peak_margin_mean": float(np.mean(per_scene["h1_target_peak_margin"])),
        "per_scene_target_mass_lift": lift,
    }


def evaluate_method_group(
    *,
    h0_scores: np.ndarray,
    h1_scores: np.ndarray,
    threshold: float,
    truth_range: np.ndarray,
    predicted_range: np.ndarray,
    truth_doppler: np.ndarray,
    predicted_doppler: np.ndarray,
    base_scene_ids: np.ndarray,
    prf_hz: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return evaluate_paired_predictions(
        h0_scores=h0_scores,
        h1_scores=h1_scores,
        threshold=threshold,
        truth_range=truth_range,
        predicted_range=predicted_range,
        truth_doppler=truth_doppler,
        predicted_doppler=predicted_doppler,
        base_scene_ids=base_scene_ids,
        prf_hz=prf_hz,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )


def paired_metric_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    base_scene_ids: np.ndarray,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values_a = np.asarray(candidate, dtype=np.float64).reshape(-1)
    values_b = np.asarray(reference, dtype=np.float64).reshape(-1)
    ids = np.asarray(base_scene_ids).reshape(-1)
    ci = paired_difference_interval(
        values_a,
        values_b,
        ids,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "candidate_minus_reference": float(np.mean(values_a - values_b)),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def circular_doppler_error(predicted_hz: np.ndarray, truth_hz: np.ndarray, prf_hz: float) -> np.ndarray:
    return np.asarray(wrap_frequency(np.asarray(predicted_hz) - np.asarray(truth_hz), prf_hz), dtype=np.float64)
