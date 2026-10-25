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
