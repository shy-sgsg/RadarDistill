"""Cell-level false-alarm summaries for the D2-A-R7 evaluation protocol."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.stats import beta as beta_distribution


def native_cell_count(range_cells: int, doppler_cells: int) -> int:
    """Return the number of physical range-Doppler CUTs, excluding oversampling."""
    for name, value in (("range_cells", range_cells), ("doppler_cells", doppler_cells)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return int(range_cells) * int(doppler_cells)


def map_oversampled_frequency_to_native(
    frequencies_hz: Any, prf_hz: float, native_doppler_cells: int
) -> np.ndarray:
    """Map arbitrary Doppler samples to nearest periodic native-grid indices.

    Frequency samples are interpreted modulo PRF. This mapping is for relating
    oversampled outputs to physical cells; it does not increase the number of
    independent CUTs. Exact half-bin ties resolve to the lower shifted index.
    """
    values = np.asarray(frequencies_hz, dtype=np.float64)
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("frequencies_hz must be nonempty and finite")
    if not np.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if (
        isinstance(native_doppler_cells, (bool, np.bool_))
        or not isinstance(native_doppler_cells, (int, np.integer))
        or native_doppler_cells <= 0
    ):
        raise ValueError("native_doppler_cells must be a positive integer")

    native = np.fft.fftshift(
        np.fft.fftfreq(int(native_doppler_cells), d=1.0 / float(prf_hz))
    )
    wrapped_delta = (
        values[..., None] - native + float(prf_hz) / 2.0
    ) % float(prf_hz) - float(prf_hz) / 2.0
    return np.argmin(np.abs(wrapped_delta), axis=-1).astype(np.int64)


def exact_binomial_upper(successes: int, total: int, confidence: float = 0.95) -> float:
    """One-sided exact Clopper-Pearson upper confidence limit."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must define a valid binomial count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if successes == total:
        return 1.0
    if successes == 0:
        return float(-math.expm1(math.log1p(-confidence) / total))
    return float(beta_distribution.ppf(confidence, successes + 1, total - successes))


def zero_event_required_n(
    target_probability: float,
    confidence: float = 0.95,
    family_size: int = 1,
) -> int:
    """Independent zero-event trials needed for a one-sided upper bound.

    ``family_size`` applies Bonferroni correction, for example across fixed
    physical CUTs when a simultaneous guarantee is requested.
    """
    if not 0.0 < target_probability < 1.0:
        raise ValueError("target_probability must lie strictly between zero and one")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if (
        isinstance(family_size, (bool, np.bool_))
        or not isinstance(family_size, (int, np.integer))
        or family_size <= 0
    ):
        raise ValueError("family_size must be a positive integer")
    alpha_per_test = (1.0 - confidence) / int(family_size)
    return int(math.ceil(math.log(alpha_per_test) / math.log1p(-target_probability)))


def _validate_dense_inputs(
    scores: Any, threshold: float, valid_mask: Any, comparison: str
) -> tuple[np.ndarray, np.ndarray, bool]:
    values = np.asarray(scores, dtype=np.float64)
    scalar_scene = values.ndim == 2
    if scalar_scene:
        values = values[None, ...]
    if values.ndim < 2 or values.shape[0] == 0 or int(np.prod(values.shape[1:])) == 0:
        raise ValueError("scores must have shape [scenes, ...physical CUTs] or [...CUTs]")
    if np.any(~np.isfinite(values)):
        raise ValueError("scores must be finite")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if comparison not in {"greater_than", "greater_equal"}:
        raise ValueError("comparison must be 'greater_than' or 'greater_equal'")

    mask = np.asarray(valid_mask, dtype=bool)
    physical_shape = values.shape[1:]
    if mask.shape != physical_shape:
        raise ValueError("valid_mask must match the physical CUT dimensions")
    if not np.any(mask):
        raise ValueError("valid_mask must include at least one physical CUT")
    return values.reshape(values.shape[0], -1), mask.reshape(-1), scalar_scene


def dense_h0_summary(
    scores: Any,
    threshold: float,
    valid_mask: Any,
    *,
    random_cut_seed: int,
    comparison: str = "greater_than",
    confidence: float = 0.95,
    bootstrap_seed: int = 1707,
    bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Summarize dense H0 scores while preserving scene clustering.

    ``scores`` accepts a single physical map ``[D,R]`` or a batch
    ``[scene,D,R]`` (more generally, ``[scene,...CUTs]``). The exact binomial
    interval is based on one preselected uniformly random CUT per independent
    scene. The pooled CUT rate is descriptive because cells within one CPI can
    be dependent. All-zero scene bootstrap intervals are deliberately omitted.
    """
    if random_cut_seed < 0 or bootstrap_seed < 0:
        raise ValueError("seeds must be nonnegative")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    values, mask, _ = _validate_dense_inputs(scores, threshold, valid_mask, comparison)
    is_false = values >= threshold if comparison == "greater_equal" else values > threshold
    is_false &= mask[None, :]
    scene_false_counts = np.sum(is_false, axis=1).astype(np.int64)
    valid_per_scene = int(np.sum(mask))
    scene_rates = scene_false_counts / valid_per_scene
    scene_count = values.shape[0]
    false_per_cut = np.sum(is_false, axis=0).astype(np.int64)
    fixed_cut_pfa = false_per_cut[mask] / scene_count
    fixed_cut_upper = np.asarray(
        [exact_binomial_upper(int(count), scene_count, confidence) for count in false_per_cut[mask]],
        dtype=np.float64,
    )

    rng = np.random.default_rng(random_cut_seed)
    valid_indices = np.flatnonzero(mask)
    selected = valid_indices[rng.integers(0, valid_indices.size, size=scene_count)]
    random_cut_false = is_false[np.arange(scene_count), selected]
    random_cut_count = int(np.sum(random_cut_false))
    random_cut_upper = exact_binomial_upper(random_cut_count, scene_count, confidence)

    has_any_false = scene_false_counts > 0
    if np.any(has_any_false):
        boot_rng = np.random.default_rng(bootstrap_seed)
        samples = np.empty(bootstrap_replicates, dtype=np.float64)
        for start in range(0, bootstrap_replicates, 256):
            count = min(256, bootstrap_replicates - start)
            indexes = boot_rng.integers(0, scene_count, size=(count, scene_count))
            samples[start : start + count] = np.mean(scene_rates[indexes], axis=1)
        alpha = 1.0 - confidence
        cluster_ci: list[float] | None = [
            float(np.quantile(samples, alpha / 2.0)),
            float(np.quantile(samples, 1.0 - alpha / 2.0)),
        ]
    else:
        cluster_ci = None

    alpha_per_cut = (1.0 - confidence) / int(np.sum(mask))
    simultaneous_upper = np.where(
        false_per_cut[mask] == scene_count,
        1.0,
        beta_distribution.ppf(
            1.0 - alpha_per_cut,
            false_per_cut[mask] + 1,
            scene_count - false_per_cut[mask],
        ),
    )
    return {
        "n_h0_scenes": int(scene_count),
        "valid_cuts_per_scene": valid_per_scene,
        "valid_cut_count": int(scene_count * valid_per_scene),
        "false_cut_count": int(np.sum(scene_false_counts)),
        "pfa_cell_pooled": float(np.sum(scene_false_counts) / (scene_count * valid_per_scene)),
        "pfa_cell_scene_cluster_bootstrap_ci": cluster_ci,
        "false_cut_exceedances_per_cpi": scene_false_counts.tolist(),
        "mean_false_cut_exceedances_per_cpi": float(np.mean(scene_false_counts)),
        "pfa_cpi": float(np.mean(has_any_false)),
        "pfa_cpi_false_scene_count": int(np.sum(has_any_false)),
        "fixed_cut_pfa_min_median_max": [
            float(np.min(fixed_cut_pfa)),
            float(np.median(fixed_cut_pfa)),
            float(np.max(fixed_cut_pfa)),
        ],
        "fixed_cut_cp_upper_max": float(np.max(fixed_cut_upper)),
        "fixed_cut_bonferroni_cp_upper_max": float(np.max(simultaneous_upper)),
        "random_cut_seed": int(random_cut_seed),
        "independent_random_cut": {
            "n_scenes": int(scene_count),
            "n_false": random_cut_count,
            "pfa_cell": float(random_cut_count / scene_count),
            "one_sided_cp_upper": float(random_cut_upper),
            "confidence": float(confidence),
        },
        "threshold_comparison": comparison,
    }


def sparse_h0_summary(
    false_candidate_counts: Any,
    nominal_cells_per_cpi: int,
) -> dict[str, Any]:
    """Summarize sparse H0 outputs without calling their rate dense cell Pfa."""
    counts = np.asarray(false_candidate_counts)
    if counts.ndim != 1 or counts.size == 0 or np.any(~np.isfinite(counts)):
        raise ValueError("false_candidate_counts must be a nonempty finite vector")
    if np.any(counts < 0) or np.any(counts != np.floor(counts)):
        raise ValueError("false_candidate_counts must contain nonnegative integers")
    if (
        isinstance(nominal_cells_per_cpi, (bool, np.bool_))
        or not isinstance(nominal_cells_per_cpi, (int, np.integer))
        or nominal_cells_per_cpi <= 0
    ):
        raise ValueError("nominal_cells_per_cpi must be a positive integer")
    counts = counts.astype(np.int64)
    return {
        "n_h0_scenes": int(counts.size),
        "pfa_cell": "NOT_APPLICABLE",
        "pfa_cpi": float(np.mean(counts > 0)),
        "pfa_cpi_false_scene_count": int(np.sum(counts > 0)),
        "false_candidate_counts_per_cpi": counts.tolist(),
        "mean_false_candidates_per_cpi": float(np.mean(counts)),
        "sparse_output_rate_per_nominal_cell": float(np.sum(counts) / counts.size / nominal_cells_per_cpi),
        "nominal_cells_per_cpi": int(nominal_cells_per_cpi),
    }
