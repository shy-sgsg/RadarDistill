import numpy as np
import pytest

from radardistill.evaluation.d2a_r7_metrics import (
    dense_h0_summary,
    exact_binomial_upper,
    map_oversampled_frequency_to_native,
    native_cell_count,
    sparse_h0_summary,
    zero_event_required_n,
)


def test_native_grid_and_zero_padding_map_to_same_physical_cells():
    assert native_cell_count(32, 130) == 4160

    prf_hz = 1300.0
    native = np.fft.fftshift(np.fft.fftfreq(130, d=1 / prf_hz))
    oversampled = np.fft.fftshift(np.fft.fftfreq(520, d=1 / prf_hz))
    mapped = map_oversampled_frequency_to_native(oversampled, prf_hz, 130)
    assert mapped.shape == (520,)
    assert np.array_equal(mapped[::4], np.arange(130))
    assert np.all((mapped >= 0) & (mapped < 130))
    assert native[mapped[0]] == pytest.approx(native[0])


def test_dense_summary_uses_valid_physical_cuts_and_strict_threshold():
    scores = np.array(
        [
            [[0.5, 0.6, 0.4], [0.8, 0.1, 0.9]],
            [[0.7, 0.5, 0.2], [0.3, 0.8, 0.1]],
        ]
    )
    valid = np.array([[True, True, True], [True, False, True]])

    result = dense_h0_summary(scores, 0.5, valid, random_cut_seed=11)

    assert result["valid_cut_count"] == 10
    assert result["false_cut_count"] == 4
    assert result["pfa_cell_pooled"] == pytest.approx(0.4)
    assert result["pfa_cpi"] == pytest.approx(1.0)
    assert result["false_cut_exceedances_per_cpi"] == [3, 1]
    assert result["mean_false_cut_exceedances_per_cpi"] == pytest.approx(2.0)
    assert result["independent_random_cut"]["n_scenes"] == 2
    assert result["independent_random_cut"]["n_false"] <= 2
    assert result == dense_h0_summary(scores, 0.5, valid, random_cut_seed=11)


def test_dense_summary_reports_ge_tie_sensitivity_separately():
    scores = np.array([[[0.5, 0.49], [0.5, 0.1]]])
    valid = np.ones((2, 2), dtype=bool)
    strict = dense_h0_summary(scores, 0.5, valid, random_cut_seed=3)
    inclusive = dense_h0_summary(
        scores, 0.5, valid, random_cut_seed=3, comparison="greater_equal"
    )
    assert strict["false_cut_count"] == 0
    assert strict["pfa_cell_scene_cluster_bootstrap_ci"] is None
    assert inclusive["false_cut_count"] == 2


def test_sparse_output_is_not_mislabeled_as_dense_cell_pfa():
    result = sparse_h0_summary(np.array([0, 2, 1, 0]), nominal_cells_per_cpi=4160)
    assert result["pfa_cell"] == "NOT_APPLICABLE"
    assert result["pfa_cpi"] == pytest.approx(0.5)
    assert result["mean_false_candidates_per_cpi"] == pytest.approx(0.75)
    assert result["sparse_output_rate_per_nominal_cell"] == pytest.approx(0.75 / 4160)


def test_zero_event_exact_bound_and_required_sample_counts():
    assert exact_binomial_upper(0, 100, confidence=0.95) == pytest.approx(
        1 - 0.05 ** (1 / 100)
    )
    n = zero_event_required_n(1e-6, confidence=0.95)
    assert n == 2_995_731
    assert exact_binomial_upper(0, n, confidence=0.95) <= 1e-6
    assert exact_binomial_upper(0, n - 1, confidence=0.95) > 1e-6


def test_fixed_cell_familywise_sample_count_accounts_for_4160_cells():
    n = zero_event_required_n(1e-6, confidence=0.95, family_size=4160)
    assert n == 11_328_997


@pytest.mark.parametrize("bad", [0, -1, 1.0, np.nan])
def test_native_grid_rejects_invalid_cell_counts(bad):
    with pytest.raises((TypeError, ValueError)):
        native_cell_count(bad, 130)
