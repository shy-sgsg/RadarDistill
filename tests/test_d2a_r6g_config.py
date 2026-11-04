from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from radardistill.training import d2a_r6g


CONFIG_PATH = Path(__file__).resolve().parents[1] / "experiments/D2-A-R6/R6-G/config.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_accepts_frozen_r6g_protocol_config() -> None:
    d2a_r6g.validate_r6g_config(_config())


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("arms", None, ["T0", "T1", "T2", "T3"]),
        ("training", "checkpoint_selection", "minimum_arm_total_loss"),
        ("training", "effective_batch_size_cpi", 128),
        ("teacher", "device", "cpu"),
        ("experiment", "final_blind_authorized", True),
    ],
)
def test_rejects_protocol_drift(section: str, key: str | None, value: object) -> None:
    config = deepcopy(_config())
    if key is None:
        config[section] = value
    else:
        config[section][key] = value
    with pytest.raises(ValueError):
        d2a_r6g.validate_r6g_config(config)
