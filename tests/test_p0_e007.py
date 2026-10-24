import numpy as np
import pytest
import torch

from radardistill.baselines import wrapped_error
from radardistill.torch_models import (
    CircularSpectralNet,
    FFTSubbinCorrector,
    IQTransformer,
    circular_frequency_targets,
    decode_circular_logits,
    local_spectrum_features,
    spectral_coherence_features,
)
from radardistill.p0_e007 import (
    e007_loss,
    estimate_fft_nfft_interpolated,
    estimate_fft_nfft_peak,
    _adjacent_gate,
    grouped_frequency_metrics,
    _make_final_blind_batch,
    _make_noiseless_batch,
    _baseline_method_names,
    make_teacher_targets,
    paired_bootstrap_rmse_difference,
    require_cuda,
    _recover_completed_training_record,
    screen_selection,
)
from radardistill.signal import generate_batch, generate_single_tone


def _random_iq(batch=3, n_samples=130, seed=81):
    generator = np.random.default_rng(seed)
    values = generator.normal(size=(batch, n_samples)) + 1j * generator.normal(
        size=(batch, n_samples)
    )
    return torch.as_tensor(values.astype(np.complex64))


def _modulate(iq, bins, n_fft, prf_hz=1300.0):
    n = torch.arange(iq.shape[-1], dtype=torch.float32)
    delta_hz = bins * prf_hz / n_fft
    return iq * torch.exp(2j * torch.pi * delta_hz * n / prf_hz)


def test_spectral_coherence_features_are_common_phase_and_scale_invariant():
    iq = _random_iq()
    baseline = spectral_coherence_features(iq, n_fft=2080)
    rotated = spectral_coherence_features(
        iq * torch.exp(torch.tensor(0.71j)), n_fft=2080
    )
    scaled = spectral_coherence_features(iq * (2.3 - 0.8j), n_fft=2080)

    assert baseline.shape == (3, 3, 2080)
    assert torch.isfinite(baseline).all()
    torch.testing.assert_close(rotated, baseline, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(scaled, baseline, rtol=2e-4, atol=2e-5)


def test_spectral_coherence_features_roll_exactly_under_one_fft_bin_shift():
    iq = _random_iq()
    baseline = spectral_coherence_features(iq, n_fft=2080)
    shifted_iq = _modulate(iq, bins=1, n_fft=2080)
    shifted = spectral_coherence_features(shifted_iq, n_fft=2080)

    torch.testing.assert_close(
        shifted, torch.roll(baseline, shifts=1, dims=-1), rtol=3e-4, atol=3e-5
    )


def test_local_spectrum_features_are_finite_for_zero_and_preserve_window_shape():
    iq = torch.zeros((2, 130), dtype=torch.complex64)
    features = local_spectrum_features(iq, n_fft=520, window_bins=9)

    assert features.shape == (2, 27)
    assert torch.isfinite(features).all()


def test_local_spectrum_features_are_invariant_to_common_phase_scale_and_grid_shift():
    iq = _random_iq()
    baseline = local_spectrum_features(iq, n_fft=520, window_bins=9)
    rotated_scaled = local_spectrum_features(
        iq * torch.exp(torch.tensor(-1.13j)) * (0.8 + 1.9j),
        n_fft=520,
        window_bins=9,
    )
    shifted = local_spectrum_features(
        _modulate(iq, bins=1, n_fft=520), n_fft=520, window_bins=9
    )

    torch.testing.assert_close(rotated_scaled, baseline, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(shifted, baseline, rtol=4e-4, atol=4e-5)


def test_circular_frequency_targets_wrap_and_normalize_across_nyquist():
    prf_hz = 1300.0
    n_bins = 520
    spacing = prf_hz / n_bins
    frequencies = torch.tensor([-prf_hz / 2 + 17 * spacing, prf_hz / 2 - 1e-4])
    targets = circular_frequency_targets(
        frequencies, prf_hz, n_bins=n_bins, sigma_bins=0.5
    )

    assert targets.shape == (2, n_bins)
    assert torch.isfinite(targets).all()
    torch.testing.assert_close(targets.sum(dim=-1), torch.ones(2))
    assert int(targets[0].argmax()) == 17
    assert int(targets[1].argmax()) == 0


def test_decode_circular_logits_uses_prf_period_and_half_open_interval():
    logits = torch.full((2, 520), -40.0)
    logits[0, 0] = 40.0
    logits[1, 519] = 40.0

    decoded = decode_circular_logits(logits, prf_hz=1300.0)

    assert decoded[0].item() == pytest.approx(-650.0, abs=1e-4)
    assert decoded[1].item() == pytest.approx(647.5, abs=1e-3)


def test_circular_spectral_net_outputs_roll_equivariant_frequency_logits():
    torch.manual_seed(17)
    model = CircularSpectralNet(
        input_channels=3, hidden_channels=(8, 8, 8), kernel_size=5
    ).eval()
    features = torch.randn(2, 3, 520)
    shift_bins = 13

    logits = model(features)
    shifted_logits = model(torch.roll(features, shifts=shift_bins, dims=-1))

    assert logits.shape == (2, 520)
    torch.testing.assert_close(
        shifted_logits,
        torch.roll(logits, shifts=shift_bins, dims=-1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_fft_subbin_corrector_returns_finite_bounded_correction_and_gradients():
    torch.manual_seed(19)
    model = FFTSubbinCorrector(input_features=27, hidden_units=(64, 32), half_bin=0.5)
    features = torch.randn(12, 27)

    correction = model(features)
    loss = correction.square().mean()
    loss.backward()

    assert correction.shape == (12,)
    assert torch.isfinite(correction).all()
    assert correction.abs().max().item() <= 0.5
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_iq_transformer_is_small_and_trains_on_direct_iq_tokens():
    torch.manual_seed(23)
    model = IQTransformer(
        n_samples=130,
        output_bins=2080,
        d_model=32,
        n_heads=4,
        n_layers=2,
        feedforward_dim=64,
        dropout=0.05,
    )
    inputs = torch.randn(3, 130, 2)

    logits = model(inputs)
    logits.square().mean().backward()

    assert logits.shape == (3, 2080)
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_teacher_targets_keep_truth_hard_point_and_soft_response_separate():
    batch = generate_batch(
        count=12,
        seed=451,
        prf_hz=1300.0,
        n_samples=130,
        snr_db=10.0,
    )
    config = {
        "prf_hz": 1300.0,
        "n_samples": 130,
        "models": {
            "m1": {"fft_factor": 16},
            "m2": {
                "fft_factor": 4,
                "soft_teacher_candidates": 65,
                "correction_half_width_bins": 0.5,
            },
        },
        "teacher": {"soft_profile_temperature": 2.0},
    }

    teacher = make_teacher_targets(batch, config)

    np.testing.assert_array_equal(
        teacher["truth_frequency_hz"], batch.frequency_hz
    )
    assert teacher["teacher_hard_frequency_hz"].shape == (12,)
    assert teacher["teacher_soft_response"].shape == (12, 2080)
    assert teacher["m2_teacher_soft_correction_bins"].shape == (12,)
    np.testing.assert_allclose(teacher["teacher_soft_response"].sum(axis=-1), 1.0)
    assert np.max(np.abs(teacher["m2_teacher_soft_correction_bins"])) <= 0.5
    assert teacher["truth_frequency_hz"] is not teacher["teacher_hard_frequency_hz"]


def test_e007_distillation_losses_do_not_use_truth_for_e2_and_e3_validation_uses_truth():
    outputs = torch.tensor([[0.8, -0.2, 0.1], [-0.4, 0.7, 0.2]])
    targets = {
        "truth_correction": torch.tensor([0.1, -0.2]),
        "teacher_hard_correction": torch.tensor([-0.3, 0.4]),
        "teacher_soft_correction": torch.tensor([-0.1, 0.2]),
        "truth_distribution": torch.tensor(
            [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]
        ),
        "teacher_hard_distribution": torch.tensor(
            [[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]
        ),
        "teacher_soft_response": torch.tensor(
            [[0.2, 0.7, 0.1], [0.7, 0.2, 0.1]]
        ),
    }
    changed_truth = dict(targets)
    changed_truth["truth_correction"] = torch.tensor([100.0, 100.0])
    changed_truth["truth_distribution"] = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )

    for model_name in ("M1", "M1_power", "M2", "M3"):
        model_output = (
            torch.tensor([0.8, -0.4])
            if model_name == "M2"
            else outputs
        )
        hard_loss = e007_loss(
            model_output, targets, model_name, "E2-H", teacher_weight=0.5
        )
        hard_changed = e007_loss(
            model_output, changed_truth, model_name, "E2-H", teacher_weight=0.5
        )
        soft_loss = e007_loss(
            model_output, targets, model_name, "E2-S", teacher_weight=0.5
        )
        soft_changed = e007_loss(
            model_output, changed_truth, model_name, "E2-S", teacher_weight=0.5
        )
        joint_validation = e007_loss(
            model_output, targets, model_name, "E3-S", teacher_weight=0.5, validation=True
        )
        truth_validation = e007_loss(
            model_output, targets, model_name, "E1", teacher_weight=0.5, validation=True
        )

        torch.testing.assert_close(hard_loss, hard_changed)
        torch.testing.assert_close(soft_loss, soft_changed)
        torch.testing.assert_close(joint_validation, truth_validation)


def test_screen_selection_applies_frozen_convergence_accuracy_and_adjacent_point_rule():
    scores = {
        "M1": {
            "finite": True,
            "validation_first_loss": 1.0,
            "validation_best_loss": 0.7,
            "development_rmse_20db_hz": 4.9,
            "fft_win_by_snr": [False, False, False, False, False],
            "structural_transform_ok": True,
        },
        "M2": {
            "finite": True,
            "validation_first_loss": 1.0,
            "validation_best_loss": 0.75,
            "development_rmse_20db_hz": 6.0,
            "fft_win_by_snr": [False, True, True, False, False],
            "structural_transform_ok": True,
        },
        "M3": {
            "finite": True,
            "validation_first_loss": 1.0,
            "validation_best_loss": 0.7,
            "development_rmse_20db_hz": 12.0,
            "fft_win_by_snr": [True, False, True, False, False],
            "structural_transform_ok": True,
        },
    }

    selection = screen_selection(scores, max_rmse_20db_hz=5.0, min_adjacent_fft_points=2)

    assert selection["selected"] == ["M1", "M2"]
    assert "development" in selection["rejected"]["M3"].lower()


def test_e007_training_preflight_refuses_cpu_fallback():
    with pytest.raises(RuntimeError, match="CUDA"):
        require_cuda(
            training_device="cuda",
            cuda_available=False,
            device_count=0,
        )


def test_e007_training_preflight_rejects_cpu_configuration():
    with pytest.raises(ValueError, match="cuda"):
        require_cuda(
            training_device="cpu",
            cuda_available=True,
            device_count=1,
        )


def test_e007_training_preflight_returns_cuda_device_when_available():
    device = require_cuda(
        training_device="cuda",
        cuda_available=True,
        device_count=1,
    )

    assert str(device) == "cuda:0"


def test_oversampled_fft_estimators_refine_off_grid_and_wrap_at_nyquist():
    prf_hz = 1300.0
    frequencies = np.asarray([123.41, prf_hz / 2.0 - 0.08])
    signals = np.stack(
        [
            generate_single_tone(
                float(frequency), 0.37, 0.8, np.inf, prf_hz, 130, 100 + index
            )
            for index, frequency in enumerate(frequencies)
        ]
    )

    peak = estimate_fft_nfft_peak(signals, prf_hz, n_fft=520)
    refined = estimate_fft_nfft_interpolated(signals, prf_hz, n_fft=520)

    assert abs(float(wrapped_error(peak[0], frequencies[0], prf_hz))) <= 1.25
    assert abs(float(wrapped_error(refined[0], frequencies[0], prf_hz))) < 0.1
    assert abs(float(wrapped_error(refined[1], frequencies[1], prf_hz))) < 0.1


def test_paired_bootstrap_rmse_difference_is_hierarchical_and_reproducible():
    truth = np.zeros(32)
    baseline = np.stack([np.ones(32), np.ones(32) * 2.0])
    candidate = np.stack([np.ones(32) * 0.5, np.ones(32)])

    result = paired_bootstrap_rmse_difference(
        candidate,
        baseline,
        truth,
        prf_hz=1300.0,
        bootstrap_samples=300,
        seed=441,
    )

    assert result["mean_rmse_difference_hz"] == pytest.approx(-0.75)
    assert result["ci95_low_hz"] < result["ci95_high_hz"] < 0
    assert result == paired_bootstrap_rmse_difference(
        candidate,
        baseline,
        truth,
        prf_hz=1300.0,
        bootstrap_samples=300,
        seed=441,
    )


def test_paired_bootstrap_broadcasts_one_fixed_baseline_across_model_seeds():
    truth = np.zeros(32)
    candidate = np.stack([np.ones(32) * 0.5, np.ones(32)])
    baseline = np.ones((1, 32)) * 2.0

    result = paired_bootstrap_rmse_difference(
        candidate,
        baseline,
        truth,
        prf_hz=1300.0,
        bootstrap_samples=300,
        seed=442,
    )

    assert result["baseline_mean_rmse_hz"] == pytest.approx(2.0)
    assert result["mean_rmse_difference_hz"] == pytest.approx(-1.25)
    assert result["ci95_high_hz"] < 0


def test_grouped_metrics_report_snr_frequency_nyquist_phase_and_amplitude():
    count = 48
    truth = np.linspace(-650.0, 649.0, count)
    batch = type("Batch", (), {})()
    batch.frequency_hz = truth
    batch.phase_rad = np.linspace(-np.pi, np.pi, count, endpoint=False)
    batch.amplitude = np.linspace(0.5, 1.5, count)
    batch.snr_db = np.repeat([0.0, 10.0], count // 2)
    prediction = truth + 1.0

    rows = grouped_frequency_metrics(
        prediction,
        batch,
        prf_hz=1300.0,
        n_samples=130,
        outlier_threshold_hz=5.0,
    )

    groups = {(row["group_type"], row["group"]): row for row in rows}
    assert ("frequency_bin", "bin_0") in groups
    assert ("nyquist", "edge_band") in groups
    assert ("phase_sector", "sector_0") in groups
    assert ("amplitude_quartile", "q1") in groups
    assert groups[("overall", "all")]["rmse_hz"] == pytest.approx(1.0)
    assert groups[("overall", "all")]["outlier_rate_gt_threshold"] == 0.0


def test_completed_screen_checkpoint_can_recover_its_selection_record(tmp_path):
    checkpoint_path = tmp_path / "M1_E1_seed17117.pt"
    history_path = tmp_path / "training_history.csv"
    torch.save(
        {
            "model": "M1",
            "strategy": "E1",
            "train_seed": 17117,
            "torch_seed": 80101,
            "selected_epoch": 1,
            "validation_selection_loss": 0.4,
            "state_dict": {},
            "batch_size": 64,
        },
        checkpoint_path,
    )
    history_path.write_text(
        "phase,model,strategy,train_seed,torch_seed,epoch,train_loss,validation_loss,validation_selection_target,device,batch_size\n"
        "screen,M1,E1,17117,80101,1,0.5,0.4,truth,cuda:0,64\n",
        encoding="utf-8",
    )

    record = _recover_completed_training_record(
        "M1", "E1", 17117, checkpoint_path, history_path
    )

    assert record["finite"] is True
    assert record["epochs_completed"] == 1
    assert record["validation_first_loss"] == pytest.approx(0.4)
    assert record["validation_best_loss"] == pytest.approx(0.4)
    assert record["recovered_from_checkpoint"] is True


def test_final_and_noiseless_datasets_have_registered_seeds_and_nyquist_coverage():
    config = {
        "prf_hz": 1300.0,
        "n_samples": 130,
        "data": {
            "final_test_seed": 9001,
            "test_snr_levels_db": [0, 10],
            "test_per_snr": 4,
            "noiseless_seed": 9002,
            "noiseless_count": 32,
            "amplitude_range": [0.5, 1.5],
        },
    }

    blind, seeds = _make_final_blind_batch(config)
    clean = _make_noiseless_batch(config)

    assert len(blind.iq) == 8
    assert seeds == {"0.0": 9001, "10.0": 10001}
    assert np.unique(blind.snr_db).tolist() == [0.0, 10.0]
    assert len(clean.iq) == 32
    assert np.min(clean.frequency_hz) == pytest.approx(-650.0)
    edge = np.abs(clean.frequency_hz) >= 650.0 - 10.0
    assert edge.sum() >= 16
    assert np.max(clean.frequency_hz) < 650.0


def test_adjacent_snr_gate_requires_a_consecutive_run():
    comparisons = {
        str(float(snr)): {"ci95_high_hz": upper}
        for snr, upper in zip([0, 5, 10, 15, 20], [-0.1, -0.2, 0.1, -0.1, -0.1])
    }

    gate = _adjacent_gate(comparisons, [0, 5, 10, 15, 20], 2)

    assert gate["pass"] is True
    assert gate["maximum_adjacent_passes"] == 2


def test_cached_predictions_recover_classical_method_names():
    predictions = {
        "baseline__B1_fft_peak": np.zeros((1, 4)),
        "baseline__B9_4N_local_parabolic": np.zeros((1, 4)),
        "model__M1__E1": np.zeros((3, 4)),
    }

    assert _baseline_method_names(predictions) == [
        "B1_fft_peak",
        "B9_4N_local_parabolic",
    ]
