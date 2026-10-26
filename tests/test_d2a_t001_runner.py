from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_d2a_t001 as runner

from radardistill.teachers.d2a_t001 import CovarianceBank


def read_yaml(path: str):
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_course_seed_subblocks_are_disjoint_for_full_development_size():
    seeds = [
        runner._seed(34_000_000, chunk, 1009, course, course_stride=250_000)
        for course in range(4)
        for chunk in range(188)
    ]
    assert len(seeds) == len(set(seeds))
    assert min(seeds) == 34_000_000
    assert max(seeds) < 35_000_000


def test_g3_is_reported_as_not_evaluated_when_g2_fails():
    result = runner._adjudicate_g3(
        g2_passed=False,
        soft_summaries={},
        gates={},
    )

    assert result == {
        "evaluated": False,
        "passed": False,
        "reason": "G2 failed; the preregistered protocol stops before G3 adjudication",
    }


def test_g3_applies_frozen_limits_after_g2_passes():
    config = read_yaml("experiments/D2-A-T001/config.yaml")
    soft_summaries = {
        "ACE_AMF": {
            "hard_condition": {
                "target_window_mass_lift_ci95": [0.01, 0.03],
                "teacher_top_cell_localization_rate": 0.25,
            },
            "pooled_C3": {"target_window_mass_lift_ci95": [0.01, 0.03]},
        }
    }

    result = runner._adjudicate_g3(
        g2_passed=True,
        soft_summaries=soft_summaries,
        gates=config["metrics"]["teacher_gates"],
    )

    assert result["evaluated"] is True
    assert result["passed"] is True
    assert result["hard_target_mass_lift_ci_lower_positive"] is True
    assert result["hard_top_cell_localization_minimum"] is True
    assert result["pooled_C3_mass_lift_ci_lower_positive"] is True


def test_h0_runner_streams_scores_and_preserves_scene_ids():
    config = read_yaml("experiments/D2-A-T001/config.yaml")
    physics = read_yaml("experiments/D2-A-E001/config.yaml")
    r3 = read_yaml("experiments/D2-A-R3/config.yaml")
    bins = 130 * 4
    identity = np.broadcast_to(np.eye(3, dtype=np.complex128), (bins, 3, 3)).copy()
    covariance = CovarianceBank(
        covariance=identity,
        inverse=identity,
        eigenvalues=np.ones((bins, 3)),
        condition_number=np.ones(bins),
        eigenvalue_floor_fraction=np.zeros(bins),
        scene_count=0,
        pulse_count=130,
        channel_count=3,
        oversampling_factor=4,
    )
    split = {
        "scenes": 4,
        "seed_base": 9900100,
        "mixture_seed": 901,
    }
    rows, ids, timing_iq = runner._score_h0_split(
        "test",
        split,
        config,
        physics,
        covariance,
        r3["detection"]["t1_global_robust_background"],
    )
    assert len(rows) == 4
    assert len(ids) == 4
    assert timing_iq.shape == (4, 3, 130, 32)
    assert all(np.isfinite(row[f"score_{method}"]) for row in rows for method in runner.METHODS)
    assert all("iq" not in row for row in rows)
