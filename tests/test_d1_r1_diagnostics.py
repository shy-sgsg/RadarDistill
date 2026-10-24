import numpy as np
import pytest
import torch

from radardistill.evaluation.d1_r1_diagnostics import (
    circular_doppler_error_hz,
    decode_doppler_modes,
    fit_circular_ridge_probe,
    validate_diagnostic_cache_path,
)


def test_circular_doppler_error_wraps_across_nyquist():
    error = circular_doppler_error_hz(
        np.asarray([-649.0, 649.0]),
        np.asarray([649.0, -649.0]),
        prf_hz=1300.0,
    )
    np.testing.assert_allclose(error, np.asarray([2.0, -2.0]), atol=1e-12)


def test_oracle_range_is_a_separate_read_only_diagnostic():
    outputs = {
        "presence_logit": torch.tensor([3.0]),
        "range_logits": torch.tensor([[0.0, 5.0, 0.0]]),
        "range_offset": torch.tensor([[0.2, 0.4, 0.8]]),
        "doppler_vector": torch.tensor([[[ -1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]),
    }

    modes = decode_doppler_modes(outputs, torch.tensor([0]), prf_hz=1300.0)

    assert modes["predicted_range_bin"].tolist() == [1]
    assert modes["predicted_range_cell"].tolist() == pytest.approx([1.4])
    assert modes["predicted_range_doppler_hz"].tolist() == pytest.approx([325.0])
    assert modes["oracle_range_doppler_hz"].tolist() == pytest.approx([-650.0])


def test_diagnostic_loader_rejects_blind_prediction_paths(tmp_path):
    blind_path = tmp_path / "experiments" / "D1-E001" / "blind" / "predictions.npz"
    with pytest.raises(ValueError, match="blind"):
        validate_diagnostic_cache_path(blind_path)


def test_diagnostic_loader_accepts_development_cache_paths(tmp_path):
    cache_path = tmp_path / "experiments" / "D1-E001" / "data" / "cache" / "pilot_validation"
    assert validate_diagnostic_cache_path(cache_path) == cache_path


def test_linear_probe_fits_circular_frequency_targets_without_wrap_bias():
    train_frequency = np.linspace(-120.0, 120.0, 201)
    eval_frequency = np.linspace(-110.0, 110.0, 41)
    train_phase = 2.0 * np.pi * train_frequency / 1300.0
    eval_phase = 2.0 * np.pi * eval_frequency / 1300.0
    train_features = np.column_stack((np.cos(train_phase), np.sin(train_phase)))
    eval_features = np.column_stack((np.cos(eval_phase), np.sin(eval_phase)))

    result = fit_circular_ridge_probe(
        train_features,
        train_frequency,
        eval_features,
        eval_frequency,
        prf_hz=1300.0,
        alpha=1.0,
    )

    assert np.isfinite(result["rmse_hz"])
    assert result["rmse_hz"] < 2.0
    assert result["predicted_hz"].shape == eval_frequency.shape
