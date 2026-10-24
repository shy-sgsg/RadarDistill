from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from radardistill.evaluation.d1_e002 import (
    _candidate_gate,
    _open_or_generate_evaluation_cache,
    _profile_ml_prediction,
    _summarize_prediction,
    energy_detection_score,
    estimate_fft_at_range,
    estimate_lag_one_at_range,
    estimate_pooled_fft,
    evaluation_chunk_seed,
)

ROOT = Path(__file__).resolve().parents[1]


def _tone(frequency_hz: float, prf_hz: float = 1300.0) -> np.ndarray:
    pulse = np.arange(130, dtype=np.float64)
    phase = 2.0 * np.pi * frequency_hz * pulse / prf_hz
    tone = np.exp(1j * phase)
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex128)
    iq[:, :, :, 12] = tone[None, None, :]
    return iq


def test_lag_one_baseline_uses_same_channel_coherence_phase() -> None:
    estimate = estimate_lag_one_at_range(_tone(137.25), np.array([12]), 1300.0)
    assert abs(float(estimate[0]) - 137.25) < 1e-9


def test_fft_baseline_resolves_positive_and_negative_nyquist_neighbors() -> None:
    positive = estimate_fft_at_range(_tone(647.5), np.array([12]), 1300.0)
    negative = estimate_fft_at_range(_tone(-647.5), np.array([12]), 1300.0)
    assert abs(float(positive[0]) - 647.5) < 1e-8
    assert abs(float(negative[0]) + 647.5) < 1e-8


def test_fft_five_bin_centroid_is_finite_and_periodic() -> None:
    estimate = estimate_fft_at_range(
        _tone(649.0), np.array([12]), 1300.0, estimator="five_bin_excess_centroid"
    )
    assert np.isfinite(estimate).all()
    assert -650.0 <= estimate[0] < 650.0


def test_energy_detector_is_noncoherent_and_uses_peak_range_energy() -> None:
    iq = np.zeros((2, 3, 130, 32), dtype=np.complex64)
    iq[0, :, :, 4] = 1.0 + 2.0j
    iq[1, :, :, 8] = 2.0 + 0.0j
    score = energy_detection_score(iq)
    assert score[0] == 3 * 130 * 5.0
    assert score[1] == 3 * 130 * 4.0


def test_evaluation_seed_formula_separates_snr_and_chunk_streams() -> None:
    values = {
        evaluation_chunk_seed(2026230001, snr, chunk, snr_stride=1009, snr_count=8)
        for snr in range(8)
        for chunk in range(98)
    }
    assert len(values) == 8 * 98
    assert evaluation_chunk_seed(2026230001, 3, 2, snr_stride=1009, snr_count=8) == (
        2026230001 + 3 * 1009 + 2 * 1009 * 8
    )


def test_pooled_fft_peak_and_interpolation_follow_periodic_frequency() -> None:
    iq = _tone(211.25)
    peak = estimate_pooled_fft(iq, 1300.0, nfft=520, parabolic=False)
    interpolated = estimate_pooled_fft(iq, 1300.0, nfft=520, parabolic=True)
    assert abs(float(peak[0]) - 210.0) <= 2.5
    assert abs(float(interpolated[0]) - 211.25) < 1e-8


def test_evaluation_cache_streams_small_split_once_with_stable_sample_ids(tmp_path) -> None:
    d0_target = tmp_path / "experiments/D0-E001/config.yaml"
    d0_target.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "experiments/D0-E001/config.yaml", d0_target)
    config = yaml.safe_load((ROOT / "experiments/D1-E002/config.yaml").read_text())
    config["data"]["evaluation_generation_chunk_size"] = 2
    config["seeds"]["stream_stride"] = 1009
    config["splits"]["calibration"] = {
        "h0_seed_base": 751000,
        "h0_examples_per_snr": 3,
        "snr_db": [0, 5],
    }
    arrays = _open_or_generate_evaluation_cache(
        root=tmp_path,
        config=config,
        d0_config=yaml.safe_load(d0_target.read_text()),
        kind="calibration",
    )
    assert arrays["iq"].shape == (6, 3, 130, 32)
    assert arrays["sample_id"].tolist() == list(range(6))
    assert arrays["snr_db"].tolist() == [0.0, 0.0, 0.0, 5.0, 5.0, 5.0]
    assert not arrays["truth_presence"].any()


def test_profile_ml_uses_range_interval_from_d0_waveform_config(monkeypatch) -> None:
    from radardistill.evaluation import d1_evaluation

    captured: dict[str, object] = {}

    def fake_profile_ml(iq: np.ndarray, **kwargs: object) -> dict[str, np.ndarray]:
        captured.update(kwargs)
        return {
            "range_cell": np.asarray([12.0]),
            "doppler_hz": np.asarray([15.0]),
        }

    monkeypatch.setattr(d1_evaluation, "estimate_profile_ml_reference", fake_profile_ml)
    config = yaml.safe_load((ROOT / "experiments/D1-E002/config.yaml").read_text())
    config["data"]["snr_db"] = [0]
    config["baselines"]["profile_ml"]["h1_examples_per_snr"] = 1
    arrays = {
        "iq": np.zeros((1, 3, 130, 32), dtype=np.complex64),
        "sample_id": np.asarray([42001]),
        "truth_presence": np.asarray([True]),
        "truth_range_cell": np.asarray([12.0]),
        "truth_doppler_hz": np.asarray([15.0]),
        "truth_initial_phase_rad": np.asarray([0.2]),
        "truth_target_amplitude": np.asarray([1.0]),
        "snr_db": np.asarray([0.0]),
    }

    prediction = _profile_ml_prediction(ROOT, arrays, config)

    assert captured["range_interval"] == (4.0, 28.0)
    assert captured["range_grid_step"] == 0.25
    assert captured["frequency_fft_factor"] == 16
    assert prediction["sample_id"].tolist() == [42001]


def test_blind_metric_summary_uses_wrapped_error_and_calibration_threshold() -> None:
    config = yaml.safe_load((ROOT / "experiments/D1-E002/config.yaml").read_text())
    prediction = {
        "sample_id": np.arange(4),
        "truth_presence": np.asarray([False, True, False, True]),
        "snr_db": np.asarray([0.0, 0.0, 5.0, 5.0]),
        "truth_range_cell": np.asarray([np.nan, 10.0, np.nan, 20.0]),
        "truth_doppler_hz": np.asarray([np.nan, 649.0, np.nan, -649.0]),
        "truth_initial_phase_rad": np.asarray([np.nan, 1.0, np.nan, 5.0]),
        "truth_target_amplitude": np.asarray([np.nan, 1.0, np.nan, 2.0]),
        "score": np.asarray([0.0, 1.0, 0.0, 1.0]),
        "pred_range_cell": np.asarray([0.0, 10.1, 0.0, 19.9]),
        "pred_doppler_hz": np.asarray([0.0, -649.0, 0.0, 649.0]),
    }
    calibration = {
        "methods": {
            "S2/seed_1": {"score_source": "presence_logit", "threshold": 0.5}
        }
    }
    summary = _summarize_prediction("S2/seed_1", prediction, calibration, config)
    assert summary["detection"]["by_snr_db"]["0"]["pd"] == 1.0
    assert summary["estimation"]["by_snr_db"]["0"]["doppler_rmse_hz"] == 2.0


def test_candidate_gate_uses_seed_and_sample_paired_bootstrap() -> None:
    config = yaml.safe_load((ROOT / "experiments/D1-E002/config.yaml").read_text())
    config["acceptance"]["uncertainty"]["bootstrap_replicates"] = 30
    snrs = config["data"]["snr_db"]
    predictions = {}
    summaries = {}
    calibration = {"methods": {}}
    for seed_index, split in enumerate(config["splits"]["formal"]):
        seed_id = split["id"]
        presence = np.tile([False, True], len(snrs))
        snr_db = np.repeat(np.asarray(snrs, dtype=np.float64), 2)
        truth_doppler = np.where(presence, 150.0, np.nan)
        truth_range = np.where(presence, 14.0, np.nan)
        common = {
            "sample_id": np.arange(len(presence)),
            "truth_presence": presence,
            "snr_db": snr_db,
            "truth_range_cell": truth_range,
            "truth_doppler_hz": truth_doppler,
            "truth_initial_phase_rad": np.where(presence, 1.0, np.nan),
            "truth_target_amplitude": np.where(presence, 1.0 + seed_index * 0.1, np.nan),
        }
        s0_name = f"S0/{seed_id}"
        candidate_name = f"S2/{seed_id}"
        comparator_name = f"S3_matched_three_point/{seed_id}"
        s0 = {
            **common,
            "score": np.where(presence, 1.0, 0.0),
            "pred_range_cell": np.where(presence, 14.2, 0.0),
            "pred_doppler_hz": np.where(presence, 152.0, 0.0),
        }
        candidate = {
            **common,
            "score": np.where(presence, 1.0, 0.0),
            "pred_range_cell": np.where(presence, 14.1, 0.0),
            "pred_doppler_hz": np.where(presence, 150.5, 0.0),
        }
        matched = {
            **common,
            "score": np.where(presence, 1.0, 0.0),
            "pred_range_cell": np.where(presence, 14.2, 0.0),
            "pred_doppler_hz": np.where(presence, 152.0, 0.0),
        }
        for name, values in ((s0_name, s0), (candidate_name, candidate), (comparator_name, matched)):
            predictions[name] = values
            summaries[name] = _summarize_prediction(
                name,
                values,
                {"methods": {name: {"score_source": "presence_logit", "threshold": 0.5}}},
                config,
            )
            calibration["methods"][name] = {"threshold": 0.5}
    gate = _candidate_gate(
        "S2",
        prediction_by_name=predictions,
        summaries=summaries,
        calibration=calibration,
        config=config,
    )
    assert gate["passed"]
    assert gate["doppler_delta_paired_hierarchical_bootstrap_95_ci_hz"][1] < 0.0
