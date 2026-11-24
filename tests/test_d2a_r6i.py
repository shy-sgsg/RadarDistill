from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.evaluation.d2a_r6i import (
    audit_r6i_split_plans,
    build_r6i_split_plans,
    correct_presence_logit,
    fit_tail_scale_slope,
    _paired_h0_pfa_bootstrap,
    robust_iq_power,
    _stratified_bootstrap_contrast,
    _compute_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def test_robust_iq_power_is_per_scene_positive_and_common_scale_equivariant() -> None:
    rng = np.random.default_rng(17)
    iq = rng.normal(size=(3, 3, 5, 4)) + 1j * rng.normal(size=(3, 3, 5, 4))
    power = robust_iq_power(iq)

    assert power.shape == (3,)
    assert np.isfinite(power).all()
    assert (power > 0).all()
    np.testing.assert_allclose(robust_iq_power((-1.25j) * iq), 1.25**2 * power, rtol=1e-12)


@pytest.mark.parametrize(
    "iq",
    [
        np.ones((3, 3, 5, 4), dtype=np.float64),
        np.zeros((3, 3, 5, 4), dtype=np.complex128),
        np.full((3, 3, 5, 4), np.nan + 1j, dtype=np.complex128),
    ],
)
def test_robust_iq_power_rejects_invalid_observations(iq: np.ndarray) -> None:
    with pytest.raises(ValueError):
        robust_iq_power(iq)


def test_tail_scale_fit_recovers_linear_right_tail_slope() -> None:
    log_power = np.repeat(np.linspace(-4.0, 2.0, 20), 100)
    logits = 1.7 + 0.63 * log_power

    fit = fit_tail_scale_slope(logits, log_power, bin_count=20, quantile=0.95)

    assert fit["bin_count"] == 20
    assert fit["minimum_bin_count"] == 50
    assert fit["beta"] == pytest.approx(0.63, abs=1e-12)
    assert fit["center"] == pytest.approx(float(np.median(log_power)))


def test_tail_scale_fit_rejects_nonfinite_small_or_constant_scale_inputs() -> None:
    log_power = np.repeat(np.linspace(-4.0, 2.0, 20), 10)
    with pytest.raises(ValueError, match="minimum"):
        fit_tail_scale_slope(np.arange(len(log_power)), log_power, bin_count=20, quantile=0.95)

    adequate_power = np.ones(2000)
    with pytest.raises(ValueError, match="vary"):
        fit_tail_scale_slope(np.arange(2000), adequate_power, bin_count=20, quantile=0.95)

    logits = np.ones(2000)
    adequate_power = np.linspace(-5.0, 1.0, 2000)
    logits[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_tail_scale_slope(logits, adequate_power, bin_count=20, quantile=0.95)


def test_correct_presence_logit_uses_only_frozen_score_scale_equation() -> None:
    logits = np.asarray([1.0, -2.0, 4.0])
    log_power = np.asarray([-1.0, 0.0, 3.0])
    beta = 0.4
    center = 0.5

    corrected = correct_presence_logit(logits, log_power, beta=beta, center=center)

    np.testing.assert_allclose(corrected, logits - beta * (log_power - center))
    with pytest.raises(ValueError):
        correct_presence_logit(logits, log_power[:-1], beta=beta, center=center)
    with pytest.raises(ValueError):
        correct_presence_logit(logits, log_power, beta=float("nan"), center=center)
    with pytest.raises(TypeError):
        correct_presence_logit(logits, log_power, beta=beta, center=center, condition_index=np.zeros(3))


def test_r6i_split_plan_counts_and_scene_disjointness() -> None:
    config = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-I/config.yaml").read_text())
    conditions = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-A/config.yaml").read_text())
    plans = build_r6i_split_plans(config, conditions)

    counts = {name: sum(int(row["count"]) for row in rows) for name, rows in plans.items()}
    assert counts == {
        "h0_scale_fit": 60000,
        "h0_threshold_calibration": 60000,
        "h0_verification": 120000,
        "h1_development": 6144,
        "g1_test": 16,
    }
    audit = audit_r6i_split_plans(plans)
    assert audit["overlap_count"] == 0
    assert audit["condition_counts"]["h0_verification"] == {str(index): 10000 for index in range(12)}


def test_split_audit_rejects_cross_split_scene_id_overlap() -> None:
    row = {"seed": 7, "count": 2, "condition_index": 0}
    with pytest.raises(ValueError, match="overlap"):
        audit_r6i_split_plans({"first": [row], "second": [row]})


def test_paired_h0_bootstrap_uses_matched_scenes_and_registered_c3_cells() -> None:
    conditions = [{"course": "C3"} for _ in range(6)]
    count_per_condition = 8
    condition_index = np.repeat(np.arange(6), count_per_condition)
    baseline = {
        "scene_id": np.asarray([f"scene-{i}" for i in range(len(condition_index))]),
        "condition_index": condition_index,
        "score": np.zeros(len(condition_index)),
    }
    candidate = {**baseline, "score": np.ones(len(condition_index))}
    result = _paired_h0_pfa_bootstrap(
        baseline, candidate,
        threshold_baseline=0.5, threshold_candidate=0.5,
        conditions=conditions, replicates=100, seed=19,
    )
    assert result["estimate"] == 1.0
    assert result["ci95"] == [1.0, 1.0]


def test_task_bootstrap_is_paired_and_stratified_by_condition() -> None:
    conditions = [{"course": "C2"}, {"course": "C3"}]
    condition_index = np.repeat(np.arange(2), 12)
    ids = np.asarray([f"scene-{i}" for i in range(len(condition_index))])
    truth_range = np.zeros(len(ids))
    truth_doppler = np.zeros(len(ids))
    baseline = {
        "scene_id": ids,
        "condition_index": condition_index,
        "score": np.ones(len(ids)),
        "predicted_range_cell": np.ones(len(ids)),
        "predicted_doppler_hz": np.full(len(ids), 10.0),
        "truth_range_cell": truth_range,
        "truth_doppler_hz": truth_doppler,
    }
    candidate = {
        **baseline,
        "score": np.concatenate((np.ones(12), np.zeros(12))),
    }
    result = _stratified_bootstrap_contrast(
        baseline, candidate,
        threshold_baseline=0.5, threshold_candidate=0.5,
        conditions=conditions, replicates=100, seed=23,
    )
    contrasts = result["contrasts"]
    assert contrasts["pd_difference"]["estimate"] == pytest.approx(-0.5)
    assert contrasts["c2_c3_joint_difference"]["estimate"] == pytest.approx(-0.5)
    assert contrasts["range_rmse_ratio"]["estimate"] == pytest.approx(1.0)
    assert contrasts["doppler_rmse_ratio"]["estimate"] == pytest.approx(1.0)


def test_metrics_pipeline_emits_all_workpoints_and_frozen_gates() -> None:
    config = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-I/config.yaml").read_text())
    condition_config = yaml.safe_load((ROOT / "experiments/D2-A-R6/R6-A/config.yaml").read_text())
    config["evaluation"]["paired_bootstrap_replicates"] = 25
    conditions = condition_config["radar"]["conditions"]
    condition_index = np.repeat(np.arange(12), 4).astype(np.uint8)
    count = len(condition_index)
    fit_index = np.repeat(np.arange(12), 2).astype(np.uint8)
    fit = {
        "beta": 0.2,
        "center": 0.0,
        "intercept": 0.5,
        "bin_counts": [50] * 20,
        "tail_log_power_medians": np.linspace(-3.0, 2.0, 20).tolist(),
        "tail_logit_quantiles": np.linspace(-0.1, 0.9, 20).tolist(),
    }
    fit_observations = {
        "scene_id": np.asarray([f"fit-{i}" for i in range(len(fit_index))]),
        "condition_index": fit_index,
        "log_power": np.linspace(-3.0, 2.0, len(fit_index)),
    }

    def h0(prefix: str) -> dict[str, np.ndarray]:
        result = {
            "scene_id": np.asarray([f"{prefix}-{i}" for i in range(count)]),
            "condition_index": condition_index,
            "log_power": np.linspace(-2.0, 1.0, count),
        }
        for index, key in enumerate(("score_b3", "score_ce_tcn", "score_cnn_ssm_raw", "score_cnn_ssm_corrected")):
            result[key] = np.linspace(-2.0 + index * 0.1, 2.0 + index * 0.1, count).astype(np.float32)
        return result

    calibration = h0("cal")
    verification = h0("verify")
    development = {
        "scene_id": np.asarray([f"h1-{i}" for i in range(count)]),
        "condition_index": condition_index,
        "log_power": np.linspace(-2.0, 1.0, count),
        "truth_range_cell": np.zeros(count, dtype=np.float32),
        "truth_doppler_hz": np.zeros(count, dtype=np.float32),
    }
    for index, arm in enumerate(("b3", "ce_tcn", "cnn_ssm_raw", "cnn_ssm_corrected")):
        development[f"score_{arm}"] = np.linspace(-2.0 + 0.1 * index, 2.0 + 0.1 * index, count).astype(np.float32)
        development[f"predicted_range_{arm}"] = np.zeros(count, dtype=np.float32)
        development[f"predicted_doppler_{arm}"] = np.zeros(count, dtype=np.float32)

    result = _compute_metrics(
        config, condition_config, fit, fit_observations,
        calibration, verification, development,
    )

    assert set(result["workpoints"]) == {"0.05", "0.01", "0.001"}
    assert set(result["p7_readmission"]["gates"]) == {
        "overall_h0_upper_bound",
        "c3_six_cell_familywise_upper_bounds",
        "paired_c3_pfa_reduction",
        "paired_pd_noninferiority",
        "paired_c2_c3_joint_noninferiority",
        "paired_range_rmse_noninferiority",
        "paired_doppler_rmse_noninferiority",
    }
    assert result["observable_log_power_by_split_and_condition"]["h0_verification"]["n"] == count
    assert result["split_counts"]["h0_scale_fit"] == len(fit_index)
    assert result["score_scale_fit"] == fit
