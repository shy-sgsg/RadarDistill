import numpy as np

from radardistill.d2a_baselines import DetectionOutput
from radardistill.evaluation.d2a_e001 import (
    cluster_bootstrap_interval,
    empirical_auc,
    summarize_detector,
)


def test_empirical_auc_handles_perfect_random_and_tied_scores():
    assert empirical_auc(np.asarray([0.1, 0.2, 0.3]), np.asarray([0.7, 0.8, 0.9])) == 1.0
    assert empirical_auc(np.asarray([0.1, 0.2, 0.3]), np.asarray([0.1, 0.2, 0.3])) == 0.5
    assert empirical_auc(np.ones(4), np.ones(5)) == 0.5


def test_cluster_bootstrap_interval_is_reproducible_and_resamples_scene_groups():
    values = np.asarray([0.0, 0.0, 1.0, 1.0])
    cluster_ids = np.asarray(["a", "a", "b", "b"])
    first = cluster_bootstrap_interval(values, cluster_ids, replicates=500, seed=82)
    second = cluster_bootstrap_interval(values, cluster_ids, replicates=500, seed=82)
    assert first == second
    assert first[0] == 0.0
    assert first[1] == 1.0


def test_summarize_detector_counts_misses_in_joint_success():
    h0 = DetectionOutput(
        score=np.asarray([0.0, 1.0, 0.0, 1.0]),
        doppler_hz=np.zeros(4),
        range_cell=np.zeros(4),
        angle_deg=None,
        rd_power=np.zeros((4, 4, 2)),
    )
    h1 = DetectionOutput(
        score=np.asarray([2.0, 0.0, 2.0, 2.0]),
        doppler_hz=np.asarray([0.0, 0.0, 649.0, 0.0]),
        range_cell=np.asarray([1.0, 1.0, 1.0, 1.0]),
        angle_deg=None,
        rd_power=np.ones((4, 4, 2)),
    )
    result = summarize_detector(
        h0,
        h1,
        threshold=1.5,
        truth_range_cell=np.ones(4),
        truth_doppler_hz=np.zeros(4),
        cluster_ids=np.asarray(["a", "b", "c", "d"]),
        prf_hz=1300.0,
        bootstrap_replicates=100,
        bootstrap_seed=1,
    )
    assert result["pfa_cpi"] == 0.0
    assert result["pd"] == 0.75
    assert result["joint_success"] == 0.5
