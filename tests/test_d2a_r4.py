from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_d2a_r4 as runner


def test_scale_is_constructed_in_complex128_before_cuda_input_cast():
    iq = np.asarray([1.0 + 2.0j, -0.2 + 0.7j], dtype=np.complex64)
    factor = 0.6 + 1.2j

    scaled = runner.scale_complex128(iq, factor)

    expected = iq.astype(np.complex128) * np.complex128(factor)
    assert scaled.dtype == np.complex128
    np.testing.assert_array_equal(scaled, expected)


def test_numeric_comparison_reports_absolute_relative_peak_and_threshold_changes():
    reference_score = np.asarray([1.0, 2.0])
    candidate_score = np.asarray([1.0 + 1e-6, 2.0 - 2e-6])
    reference_map = np.asarray([[[0.0, 2.0]], [[3.0, 0.0]]])
    candidate_map = np.asarray([[[0.0, 2.0 + 1e-6]], [[3.0, 0.0]]])

    result = runner.compare_numeric_outputs(
        reference_score,
        candidate_score,
        reference_map,
        candidate_map,
        threshold=1.5,
    )

    assert result["max_abs_score_difference"] == pytest.approx(2e-6)
    assert result["max_abs_map_difference"] == pytest.approx(1e-6)
    assert result["max_relative_score_difference"] > 0.0
    assert result["changed_peak_count"] == 0
    assert result["changed_threshold_decision_count"] == 0


def test_s1_pilot_requires_new_r4_engineering_gate_and_replicated_phase_issue():
    assert runner.s1_pilot_trigger(
        engineering_gate_passed=True,
        public_phase_agreement=0.98,
        doppler_rotation_p95_hz=2.0,
    )
    assert runner.s1_pilot_trigger(
        engineering_gate_passed=True,
        public_phase_agreement=1.0,
        doppler_rotation_p95_hz=10.1,
    )
    assert not runner.s1_pilot_trigger(
        engineering_gate_passed=False,
        public_phase_agreement=0.5,
        doppler_rotation_p95_hz=100.0,
    )
    assert not runner.s1_pilot_trigger(
        engineering_gate_passed=True,
        public_phase_agreement=0.99,
        doppler_rotation_p95_hz=10.0,
    )


def test_t001_frozen_gate_is_not_overwritten_by_r4_diagnostic_tolerance():
    result = runner.record_gate_lineage(
        t001_gate_passed=False,
        t001_tolerance=1e-6,
        r4_tolerance=1e-5,
        measured_difference=4.58e-5,
    )

    assert result["t001_gate_passed"] is False
    assert result["t001_tolerance"] == 1e-6
    assert result["historical_value_within_r4_tolerance_counterfactual_only"] is False
    assert result["r4_new_data_gate_result"] == "reported_separately"
    assert result["r4_tolerance"] == 1e-5
    assert result["exploratory_only"] is True


def test_shared_phase_augmentation_preserves_h0_h1_pair_relation():
    batch = {
        "iq_ri": np.zeros((2, 6, 130, 32), dtype=np.float32),
        "truth_presence": np.asarray([False, True]),
        "truth_range_cell": np.asarray([0.0, 7.25], dtype=np.float32),
        "truth_doppler_hz": np.asarray([0.0, 12.0], dtype=np.float32),
    }
    batch["iq_ri"][0, 0, 3, 5] = 1.0
    batch["iq_ri"][1, 0, 3, 5] = 2.0
    labels = {key: value.copy() for key, value in batch.items() if key != "iq_ri"}

    runner._augment_pair_phase(batch, np.random.default_rng(15))

    restored = batch["iq_ri"][:, :3] + 1j * batch["iq_ri"][:, 3:]
    difference = restored[1, 0, 3, 5] - restored[0, 0, 3, 5]
    assert abs(abs(difference) - 1.0) < 1e-6
    for key, value in labels.items():
        np.testing.assert_array_equal(batch[key], value)


def test_h0_background_is_independent_of_target_scenario_label():
    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    root = Path(__file__).resolve().parents[1]
    physics = yaml.safe_load((root / "experiments/D2-A-E001/config.yaml").read_text())

    result = runner.check_h0_scenario_independence(physics, seed=44_500_000)

    assert result["h0_exactly_independent_of_target_scenario_label"] is True
    assert all(
        row["iq_h0_exactly_equal_to_reference"] and row["noise_power_equal"]
        for row in result["per_scenario"].values()
    )


def test_cuda_b3_t1_preserve_frozen_cpu_formula_on_small_batch():
    import torch
    import yaml

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the R4 accelerated teacher implementation")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from radardistill.d2a_baselines import run_b3
    from radardistill.d2a_r3_baselines import run_t1_global_normalized
    from radardistill.simulation.d2a import generate_d2a_paired_batch

    root = Path(__file__).resolve().parents[1]
    physics = yaml.safe_load((root / "experiments/D2-A-E001/config.yaml").read_text())
    options = yaml.safe_load((root / "experiments/D2-A-R3/config.yaml").read_text())[
        "detection"
    ]["t1_global_robust_background"]
    paired = generate_d2a_paired_batch(
        4, 49_000_000, 10.0, 10.0, physics, scenario_id="separated"
    )
    iq = paired.iq_h0
    accelerated = runner._b3_t1_cuda(iq, physics, options, torch.device("cuda:0"))
    frozen_b3 = run_b3(iq, 1300.0, physics)
    frozen_t1 = run_t1_global_normalized(iq, 1300.0, physics, options)
    np.testing.assert_allclose(accelerated["B3"]["score"], frozen_b3.score, rtol=2e-5, atol=1e-3)
    np.testing.assert_allclose(accelerated["B3"]["score_map"], frozen_b3.rd_power, rtol=2e-5, atol=1e-3)
    np.testing.assert_allclose(accelerated["T1"]["score"], frozen_t1.score, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(accelerated["T1"]["score_map"], frozen_t1.score_map, rtol=2e-5, atol=2e-5)
