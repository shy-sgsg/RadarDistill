"""IQ-only K2 soft range-Doppler response for D2-A-E002."""

from __future__ import annotations

from typing import Any

import numpy as np

from radardistill.d2a_baselines import run_b3
from radardistill.evaluation.d2a_e001 import cluster_bootstrap_interval


def k2_response(iq: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    """Return the frozen B3 DPCA power map normalized to a per-CPI distribution.

    The function accepts only observable IQ and the frozen physical geometry.
    It deliberately has no arguments for labels, scenario IDs, target location,
    SNR/CNR, or other simulation metadata.
    """
    values = np.asarray(iq)
    if values.ndim == 3:
        values = values[None, ...]
    if values.ndim != 4 or values.shape[1:] != (3, 130, 32):
        raise ValueError("iq must have shape [B,3,130,32] or [3,130,32]")
    if not np.iscomplexobj(values) or np.any(~np.isfinite(values)):
        raise ValueError("iq must contain finite complex samples")
    prf_hz = float(config["radar"]["prf_hz"])
    result = run_b3(values, prf_hz, config)
    power = np.asarray(result.rd_power, dtype=np.float64)
    if power.shape != (values.shape[0], 130, 32):
        raise RuntimeError("frozen B3 returned an unexpected K2 map shape")
    if np.any(~np.isfinite(power)) or np.any(power < 0.0):
        raise RuntimeError("frozen B3 returned invalid K2 power")
    totals = power.sum(axis=(1, 2), keepdims=True)
    if np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("K2 cannot normalize a zero or invalid B3 response")
    return (power / totals).astype(np.float32)


def summarize_k2_target_windows(
    response_h0: np.ndarray,
    response_h1: np.ndarray,
    truth_range_cell: np.ndarray,
    truth_doppler_hz: np.ndarray,
    base_scene_id: np.ndarray,
    *,
    prf_hz: float,
    range_error_bins_max: float = 1.0,
    doppler_error_hz_max: float = 20.0,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Measure paired target-window mass lift and top-cell localization."""
    h0 = np.asarray(response_h0, dtype=np.float64)
    h1 = np.asarray(response_h1, dtype=np.float64)
    ranges = np.asarray(truth_range_cell, dtype=np.float64).reshape(-1)
    frequencies = np.asarray(truth_doppler_hz, dtype=np.float64).reshape(-1)
    ids = np.asarray(base_scene_id).reshape(-1)
    count = ranges.size
    if h0.shape != h1.shape or h0.shape != (count, 130, 32) or frequencies.size != count or ids.size != count:
        raise ValueError("K2 maps, truth arrays, and scene IDs must have matching [B,130,32] shapes")
    if count == 0 or prf_hz <= 0.0 or not np.isfinite(prf_hz):
        raise ValueError("K2 target-window audit needs scenes and a positive PRF")
    if any(np.any(~np.isfinite(value)) for value in (h0, h1, ranges, frequencies)):
        raise ValueError("K2 audit inputs must be finite")
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(130, d=1.0 / prf_hz))
    range_axis = np.arange(32, dtype=np.float64)
    h0_mass = np.empty(count, dtype=np.float64)
    h1_mass = np.empty(count, dtype=np.float64)
    top_hit = np.empty(count, dtype=np.bool_)
    for index in range(count):
        doppler_error = np.remainder(
            frequency_axis - frequencies[index] + prf_hz / 2.0, prf_hz
        ) - prf_hz / 2.0
        target_mask = (
            (np.abs(range_axis[None, :] - ranges[index]) <= range_error_bins_max)
            & (np.abs(doppler_error[:, None]) <= doppler_error_hz_max)
        )
        h0_mass[index] = float(np.sum(h0[index][target_mask]))
        h1_mass[index] = float(np.sum(h1[index][target_mask]))
        peak = int(np.argmax(h1[index]))
        peak_doppler, peak_range = divmod(peak, 32)
        peak_doppler_error = float(
            np.remainder(frequency_axis[peak_doppler] - frequencies[index] + prf_hz / 2.0, prf_hz)
            - prf_hz / 2.0
        )
        top_hit[index] = (
            abs(peak_range - ranges[index]) <= range_error_bins_max
            and abs(peak_doppler_error) <= doppler_error_hz_max
        )
    lift = h1_mass - h0_mass
    lift_ci = cluster_bootstrap_interval(
        lift,
        ids,
        statistic=np.mean,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return {
        "n_paired_scenes": int(count),
        "target_window_mass_h0_mean": float(np.mean(h0_mass)),
        "target_window_mass_h1_mean": float(np.mean(h1_mass)),
        "target_window_mass_lift_mean": float(np.mean(lift)),
        "target_window_mass_lift_ci95": [float(value) for value in lift_ci],
        "teacher_top_cell_localization_rate": float(np.mean(top_hit)),
        "per_scene_h0_mass": h0_mass,
        "per_scene_h1_mass": h1_mass,
        "per_scene_mass_lift": lift,
        "per_scene_top_cell_hit": top_hit,
    }


def audit_k2_cache(
    arrays: dict[str, np.ndarray],
    config: dict[str, Any],
    *,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 0,
    chunk_size: int = 16,
) -> dict[str, Any]:
    """Run the frozen numeric, scale-invariance, and target-response K2 audit."""
    required = (
        "iq_h0", "iq_h1", "teacher_response_h0", "teacher_response",
        "truth_range_cell", "truth_doppler_hz", "base_scene_id", "course_level", "scenario_id", "snr_db",
    )
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"K2 audit cache is missing required arrays: {missing}")
    count = len(arrays["base_scene_id"])
    if count == 0 or chunk_size <= 0:
        raise ValueError("K2 audit requires scenes and a positive chunk size")
    response_pairs = (
        ("H0", np.asarray(arrays["teacher_response_h0"])),
        ("H1", np.asarray(arrays["teacher_response"])),
    )
    map_summaries: dict[str, Any] = {}
    numeric_valid = True
    for name, response in response_pairs:
        if response.shape != (count, 130, 32):
            numeric_valid = False
            map_summaries[name] = {"shape": list(response.shape), "valid": False}
            continue
        totals = response.sum(axis=(1, 2), dtype=np.float64)
        valid = bool(
            np.all(np.isfinite(response))
            and np.all(response >= 0.0)
            and np.all(np.abs(totals - 1.0) <= 1e-6)
        )
        numeric_valid = numeric_valid and valid
        map_summaries[name] = {
            "shape": list(response.shape),
            "finite": bool(np.all(np.isfinite(response))),
            "nonnegative": bool(np.all(response >= 0.0)),
            "sum_min": float(np.min(totals)),
            "sum_max": float(np.max(totals)),
            "valid": valid,
        }

    scale_differences: dict[str, float] = {}
    factors: tuple[complex, ...] = (0.5 + 0.0j, 2.0 + 0.0j, 0.6 + 1.2j)
    for field in ("iq_h0", "iq_h1"):
        max_difference = 0.0
        for start in range(0, count, chunk_size):
            stop = min(start + chunk_size, count)
            original_iq = np.asarray(arrays[field][start:stop])
            reference = (
                np.asarray(arrays["teacher_response_h0"][start:stop])
                if field == "iq_h0"
                else np.asarray(arrays["teacher_response"][start:stop])
            )
            for factor in factors:
                scaled = k2_response(original_iq.astype(np.complex128) * factor, config)
                max_difference = max(
                    max_difference,
                    float(np.max(np.abs(scaled.astype(np.float64) - reference.astype(np.float64)))),
                )
        scale_differences[field] = max_difference
    scale_valid = all(value <= 1e-6 for value in scale_differences.values())
    teacher_config = config.get("teacher", {})
    information_isolation_valid = (
        tuple(teacher_config.get("input_fields", ())) == ("observable_iq_only",)
        and set(teacher_config.get("forbidden_fields", ()))
        == {"truth", "scenario_id", "target_location", "snr", "cnr", "metadata"}
    )

    levels = np.asarray(arrays["course_level"]).astype(str)
    scenarios = np.asarray(arrays["scenario_id"]).astype(str)
    snr = np.asarray(arrays["snr_db"], dtype=np.float64)
    c3 = levels == "C3"
    if not np.any(c3):
        raise ValueError("K2 audit split must contain its preregistered C3 scenes")
    summary = summarize_k2_target_windows(
        arrays["teacher_response_h0"][c3],
        arrays["teacher_response"][c3],
        arrays["truth_range_cell"][c3],
        arrays["truth_doppler_hz"][c3],
        arrays["base_scene_id"][c3],
        prf_hz=float(config["radar"]["prf_hz"]),
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    c3_ci_upper = float(summary["target_window_mass_lift_ci95"][1])
    localization = float(summary["teacher_top_cell_localization_rate"])
    severe_misleading = c3_ci_upper <= 0.0 and localization <= 0.05
    by_condition: dict[str, Any] = {}
    for scenario in ("separated", "near_clutter_edge", "partial_overlap"):
        for snr_db in (0.0, 10.0):
            selected = c3 & (scenarios == scenario) & np.isclose(snr, snr_db)
            if not np.any(selected):
                continue
            local = summarize_k2_target_windows(
                arrays["teacher_response_h0"][selected],
                arrays["teacher_response"][selected],
                arrays["truth_range_cell"][selected],
                arrays["truth_doppler_hz"][selected],
                arrays["base_scene_id"][selected],
                prf_hz=float(config["radar"]["prf_hz"]),
                bootstrap_replicates=bootstrap_replicates,
                bootstrap_seed=bootstrap_seed + len(by_condition) + 1,
            )
            by_condition[f"{scenario}_snr_{int(snr_db)}db"] = {
                key: value
                for key, value in local.items()
                if not key.startswith("per_scene_")
            }
    return {
        "n_paired_scenes": int(count),
        "numeric_valid": bool(numeric_valid),
        "map_checks": map_summaries,
        "scale_invariance_max_abs_map_difference": scale_differences,
        "scale_invariance_valid": bool(scale_valid),
        "information_isolation_valid": bool(information_isolation_valid),
        "c3_pooled_identifiable": {
            key: value
            for key, value in summary.items()
            if not key.startswith("per_scene_")
        },
        "c3_by_scenario_snr": by_condition,
        "severe_misleading_rule_triggered": bool(severe_misleading),
        "allow_C_arm": bool(numeric_valid and scale_valid and information_isolation_valid and not severe_misleading),
    }
