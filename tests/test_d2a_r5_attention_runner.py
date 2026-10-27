import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_d2a_r5_attention import (  # noqa: E402
    _attention_learning_rate,
    _microbatch_candidates,
    _validate_attention_pair_plan,
)


def test_attention_schedule_matches_both_frozen_cosine_segments():
    assert _attention_learning_rate(1) == pytest.approx(1e-3)
    assert _attention_learning_rate(6400) == pytest.approx(1e-4)
    assert _attention_learning_rate(6401) == pytest.approx(1e-4)
    assert _attention_learning_rate(12800) == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        _attention_learning_rate(0)
    with pytest.raises(ValueError):
        _attention_learning_rate(12801)


def test_attention_microbatch_fallbacks_preserve_effective_batch_divisibility():
    config = {"effective_batch_size": 64, "initial_microbatch_size": 16, "oom_microbatch_fallbacks": [8, 4]}
    assert _microbatch_candidates(config) == (16, 8, 4)
    with pytest.raises(ValueError, match="even divisor"):
        _microbatch_candidates({**config, "initial_microbatch_size": 12})


def test_attention_pair_plan_checks_complete_frozen_dataset_passes():
    plan = np.array([[0, 1], [2, 3], [0, 1], [2, 3]], dtype=np.int64)
    _validate_attention_pair_plan(plan, scene_count=4, pairs_per_update=2, expected_updates=4)
    invalid = plan.copy()
    invalid[1, 1] = 99
    with pytest.raises(ValueError, match="outside"):
        _validate_attention_pair_plan(invalid, scene_count=4, pairs_per_update=2, expected_updates=4)
    with pytest.raises(ValueError, match="complete pass"):
        _validate_attention_pair_plan(plan[:3], scene_count=4, pairs_per_update=2, expected_updates=3)
