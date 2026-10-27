from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_d2a_e003 as runner


def test_cuda_timing_runs_inference_callable_without_autograd(monkeypatch):
    class FakeEvent:
        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, other):
            return 1.0

    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner.torch.cuda, "Event", lambda **kwargs: FakeEvent())

    projection = torch.nn.Conv2d(1, 1, kernel_size=1)
    with torch.inference_mode():
        projected = torch.ones((1, 1, 2, 2))

    result = runner._cuda_timing(
        lambda: projection(projected), warmup=1, repeats=1
    )

    assert result["repeats"] == 1


def test_post_training_preflight_revalidation_preserves_initial_record(tmp_path, monkeypatch):
    initial = {
        "git_sha": "original-preflight-sha",
        "passed": True,
        "training_started": False,
        "disk_free_bytes_before_generation": 123456,
    }
    (tmp_path / "preflight.json").write_text(json.dumps(initial), encoding="utf-8")
    (tmp_path / "training.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(runner, "OUT", tmp_path)
    monkeypatch.setattr(runner, "_require_frozen_main", lambda: "evaluation-repair-sha")
    monkeypatch.setattr(runner, "_require_cuda", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(runner, "_environment_record", lambda commit: {"git_sha": commit, "gpu_name": "RTX 3050"})
    monkeypatch.setattr(runner, "_estimated_cache_bytes", lambda config: 1)
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=10**12))

    result = runner.run_preflight()

    archived = json.loads((tmp_path / "preflight_initial.json").read_text(encoding="utf-8"))
    assert archived == initial
    assert result["preflight_type"] == "post_training_evaluation_revalidation"
    assert result["training_started"] is True
    assert result["initial_preflight_git_sha"] == "original-preflight-sha"
    assert result["disk_free_bytes_before_generation"] == 123456
    assert result["disk_free_bytes_at_check"] == 10**12


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
