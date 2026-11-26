from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from radardistill.evaluation.d2a_r7d import (
    binary_auc,
    bootstrap_error_intervals,
    calibrate_score_threshold,
    make_no_clutter_pair,
    summarize_detection_scores,
    summarize_parameter_errors,
)


def test_no_clutter_pair_exactly_discards_clutter_component() -> None:
    noise = np.ones((2, 3, 130, 32), dtype=np.complex64) * (1.0 + 2.0j)
    target = np.ones_like(noise) * (3.0 - 1.0j)
    clutter = np.ones_like(noise) * (50.0 + 25.0j)
    paired = SimpleNamespace(noise_iq=noise, target_signal_iq=target, clutter_iq=clutter)

    h0, h1 = make_no_clutter_pair(paired)

    np.testing.assert_array_equal(h0, noise)
    np.testing.assert_array_equal(h1, noise + target)
    assert h0.dtype == np.complex64
    assert h1.dtype == np.complex64


def test_threshold_uses_higher_quantile_and_strict_detection_comparator() -> None:
    calibration = np.arange(100, dtype=np.float64)
    threshold = calibrate_score_threshold(calibration, target_pfa=0.10)
    assert threshold == 90.0

    summary = summarize_detection_scores(
        verification_scores=np.asarray([89.0, 90.0, 91.0]),
        h1_scores=np.asarray([90.0, 91.0]),
        threshold=threshold,
    )
    assert summary["verification_false_alarms"] == 1
    assert summary["verification_pfa"] == 1 / 3
    assert summary["h1_detected"] == 1
    assert summary["h1_pd"] == 0.5


def test_parameter_errors_wrap_doppler_and_keep_all_h1_in_denominator() -> None:
    summary = summarize_parameter_errors(
        truth_range=np.asarray([4.0, 10.0, 20.0]),
        predicted_range=np.asarray([4.5, 9.0, 22.0]),
        truth_doppler=np.asarray([649.0, -649.0, 0.0]),
        predicted_doppler=np.asarray([-649.0, 649.0, 10.0]),
        prf_hz=1300.0,
        detection_mask=np.asarray([True, False, True]),
    )
    assert summary["all_h1"]["n"] == 3
    assert summary["all_h1"]["doppler_rmse_hz"] == 6.0
    assert summary["detected_h1"]["n"] == 2
    assert summary["joint_success"] == 1 / 3


def test_task_decoder_uses_cos_sin_order() -> None:
    import torch

    from radardistill.evaluation.d2a_r7d import decode_task_outputs

    angle = torch.tensor(0.5, dtype=torch.float32)
    outputs = {
        "presence_logit": torch.tensor([0.0]),
        "range_logits": torch.tensor([[0.0, 1.0]]),
        "range_offset": torch.tensor([[0.2, 0.25]]),
        "doppler_vector": torch.tensor([[[1.0, 0.0], [torch.cos(angle), torch.sin(angle)]]]),
    }
    decoded = decode_task_outputs(outputs, prf_hz=1300.0)
    assert decoded["range_cell"][0] == 1.25
    assert decoded["doppler_hz"][0] == pytest.approx(0.5 * 1300.0 / (2.0 * np.pi))


def test_auc_and_bootstrap_are_finite_and_reproducible() -> None:
    assert binary_auc(np.asarray([0.0, 1.0]), np.asarray([2.0, 3.0])) == 1.0
    first = bootstrap_error_intervals(np.asarray([-1.0, 0.0, 2.0]), replicates=64, seed=9)
    second = bootstrap_error_intervals(np.asarray([-1.0, 0.0, 2.0]), replicates=64, seed=9)
    assert first == second
    assert first["rmse_ci95"][0] <= first["rmse_ci95"][1]
