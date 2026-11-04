from __future__ import annotations

from pathlib import Path

import yaml

from radardistill.training.d2a_r6e import _all_split_plans, _scene_counts
from radardistill.training.d2a_r6d_joint import audit_r6d_scene_plans


ROOT = Path(__file__).resolve().parents[1]
R6A_CONFIG = ROOT / "experiments/D2-A-R6/R6-A/config.yaml"


def _load_config(stage: str) -> dict:
    path = ROOT / f"experiments/D2-A-R6/R6-{stage}/config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _scene_ids(plans: dict) -> set[str]:
    return {
        f"d2a-scene-{int(row['seed'])}-{index}"
        for rows in plans.values()
        for row in rows
        for index in range(int(row["count"]))
    }


def test_r6h_draft_is_not_executable_or_blind_authorized() -> None:
    config = _load_config("H")

    assert config["experiment"]["status"] == "draft"
    assert config["experiment"]["execution_authorized"] is False
    assert config["experiment"]["final_blind_authorized"] is False
    assert config["training"]["history_written_every_update"] is False


def test_r6h_registered_split_plans_are_balanced_and_disjoint_from_e_and_g() -> None:
    r6a = yaml.safe_load(R6A_CONFIG.read_text(encoding="utf-8"))
    planned = {
        stage: _all_split_plans(_load_config(stage), r6a)
        for stage in ("E", "G", "H")
    }

    h_counts = _scene_counts(planned["H"])
    audit = audit_r6d_scene_plans(
        planned["H"], expected_counts=h_counts, prior_scene_ids=set()
    )
    assert audit["overlap_count"] == 0
    assert h_counts == {
        "train": 51_200,
        "validation": 5_000,
        "h0_calibration": 120_000,
        "h0_verification": 120_000,
        "h1_development": 6_144,
        "g1_test": 16,
    }
    h_ids = _scene_ids(planned["H"])
    for stage in ("E", "G"):
        assert h_ids.isdisjoint(_scene_ids(planned[stage]))
