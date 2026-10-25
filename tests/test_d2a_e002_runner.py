from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_d2a_e002 as runner

from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.task_student import TaskStudent


def test_g5_gate_summary_preserves_numeric_course_comparisons():
    course_signal = {
        "B_minus_A": {
            "joint_success_gain": 0.06,
            "joint_success_gain_ci95": [0.01, 0.10],
        },
        "C_minus_B": {
            "joint_success_gain": 0.01,
            "joint_success_gain_ci95": [-0.01, 0.03],
        },
        "convergence": {
            "B": {"update_saving_fraction_vs_A_best_update": None},
            "C": {"update_saving_fraction_vs_A_best_update": 0.25},
        },
    }

    gates = runner._g5_gate_summary(course_signal)

    assert gates["B_minus_A"]["pilot_signal_present"] is True
    assert gates["C_minus_B"]["pilot_signal_present"] is True
    assert course_signal["B_minus_A"]["joint_success_gain"] == 0.06
    assert course_signal["B_minus_A"]["joint_success_gain_ci95"] == [0.01, 0.10]
    assert course_signal["C_minus_B"]["joint_success_gain"] == 0.01
    assert course_signal["C_minus_B"]["joint_success_gain_ci95"] == [-0.01, 0.03]


def test_e002_physical_diagnostic_runs_on_fixed_paired_subset():
    with Path("experiments/D2-A-E001/config.yaml").open(encoding="utf-8") as stream:
        physics = yaml.safe_load(stream)
    with Path("experiments/D2-A-E002/config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["student"]["width"] = 8
    config["student"]["temporal_dilations"] = [1]
    config["statistics"]["physical_diagnostic_pairs"] = 4
    batch = generate_d2a_paired_batch(
        4,
        99201,
        10.0,
        10.0,
        physics,
        scenario_id="separated",
    )
    arrays = {
        "iq_h0": batch.iq_h0,
        "iq_h1": batch.iq_h1,
        "course_level": np.asarray(["C0", "C1", "C2", "C3"]),
    }
    model = TaskStudent(width=8, temporal_dilations=[1], include_spectrum_head=False)
    result = runner._physical_diagnostics(
        model,
        arrays,
        threshold=0.0,
        config=config,
        device=torch.device("cpu"),
    )
    assert result["n_paired_scenes"] == 4
    assert result["public_common_phase_rotations"] == 32
    assert 0.0 <= result["phase_decision_agreement_rate"] <= 1.0
    assert set(result["target_frequency_shift_hz"]) == {"-25.0", "-10.0", "10.0", "25.0"}
    assert result["used_original_calibration_threshold"] is True
