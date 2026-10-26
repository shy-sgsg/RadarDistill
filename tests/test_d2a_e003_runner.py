from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_d2a_e003 as runner


def test_e003_teacher_diagnostic_uses_cuda_baseline_and_not_cpu(monkeypatch):
    calls = []

    def cuda_baselines(iq, physics, options, device):
        calls.append((len(iq), str(device)))
        scores = np.arange(len(iq), dtype=np.float64) + (1.0 if len(calls) % 2 else 0.0)
        return {
            "B3": {"score": scores, "range_cell": scores, "doppler_hz": scores},
            "T1": {"score": scores, "range_cell": scores, "doppler_hz": scores},
        }

    def forbidden_cpu(*args, **kwargs):
        raise AssertionError("E003 R2 teacher diagnostics must run on CUDA")

    monkeypatch.setattr(runner, "_b3_t1_cuda", cuda_baselines)
    monkeypatch.setattr(runner, "run_b3", forbidden_cpu, raising=False)
    monkeypatch.setattr(runner, "run_t1_global_normalized", forbidden_cpu, raising=False)
    batch = SimpleNamespace(
        iq_h0=np.zeros((3, 3, 130, 32), dtype=np.complex64),
        iq_h1=np.zeros((3, 3, 130, 32), dtype=np.complex64),
        truth_target_range_cell=np.asarray([0.0, 1.0, 2.0]),
        truth_target_doppler_hz=np.asarray([0.0, 1.0, 2.0]),
    )

    result = runner._teacher_diagnostic(
        batch, {}, {}, device="cuda:0", batch_size=2
    )

    assert calls == [(2, "cuda:0"), (1, "cuda:0"), (2, "cuda:0"), (1, "cuda:0")]
    assert set(result) == {"B3", "T1"}
    assert result["B3"]["teacher_status"] == "research_reference_only; not an approved teacher"


def test_frequency_shift_summary_wraps_nyquist_crossing():
    baseline = {
        "score": np.asarray([2.0]),
        "range_cell": np.asarray([4.0]),
        "doppler_hz": np.asarray([645.0]),
    }
    shifted = {
        "score": np.asarray([2.1]),
        "range_cell": np.asarray([4.0]),
        "doppler_hz": np.asarray([-645.0]),
    }

    result = runner._frequency_shift_summary(
        baseline,
        shifted,
        truth_range=np.asarray([4.0]),
        truth_doppler=np.asarray([645.0]),
        threshold=1.0,
        shift_hz=10.0,
    )

    assert result["estimated_doppler_shift_hz_mean"] == 10.0
    assert result["estimated_doppler_shift_error_abs_p95_hz"] == 0.0
    assert result["baseline_doppler_rmse_hz"] == 0.0
    assert result["intervention_doppler_rmse_hz"] == 0.0
    assert result["intervention_joint_success"] == 1.0
