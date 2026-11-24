from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/D2-A-R6/R6-J4/config.yaml"
PROTOCOL = ROOT / "experiments/D2-A-R6/R6-J4/protocol.md"
R6A_CONFIG = ROOT / "experiments/D2-A-R6/R6-A/config.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_protocol_and_config_are_registered_before_data_generation() -> None:
    assert CONFIG.is_file()
    assert PROTOCOL.is_file()
    text = PROTOCOL.read_text(encoding="utf-8")
    config = _config()

    assert config["experiment"]["id"] == "D2-A-R6-J4"
    assert config["experiment"]["status"] == "preregistered"
    assert config["experiment"]["final_blind_authorized"] is False
    assert "final blind" in text.lower()
    assert "预注册" in text
    assert "不得" in text


def test_radar_grid_and_registered_model_comparison_match_r6a() -> None:
    config = _config()
    radar = config["radar"]
    r6a_radar = yaml.safe_load(R6A_CONFIG.read_text(encoding="utf-8"))["radar"]

    assert radar["channel_count"] == 3
    assert radar["pulse_count"] == 130
    assert radar["range_bin_count"] == 32
    assert radar["prf_hz"] == 1300.0
    assert radar["conditions"] == r6a_radar["conditions"]
    assert len(radar["conditions"]) == 12
    assert config["models"]["arms"] == ["j1_joint", "j4_joint"]
    assert config["models"]["fixed_baselines"] == [
        "b3_130_fft",
        "fixed_dpca_130_fft_peak",
        "fixed_dpca_520_fft_peak",
    ]
    assert config["models"]["j4_frontend"]["channel_pair"] == [0, 1]
    assert config["models"]["j4_frontend"]["doppler_fft_bins"] == 520


def test_matched_training_budget_and_split_counts_are_frozen() -> None:
    config = _config()
    training = config["training"]
    splits = config["data"]["splits"]

    assert training["training_seed"] == 909
    assert training["fixed_updates"] == 25_600
    assert training["effective_batch_size_cpi"] == 64
    assert training["history_interval_updates"] == 400
    assert training["checkpoint_interval_updates"] == 800
    assert training["arms_share_scene_order"] is True
    assert training["arms_share_update_budget"] is True
    assert splits["train"]["paired_scenes"] == 51_200
    assert splits["validation"]["paired_scenes"] == 5_000
    assert splits["h0_calibration"]["scene_count"] == 120_000
    assert splits["h0_verification"]["scene_count"] == 120_000
    assert splits["h1_development"]["scene_count"] == 6_144
    assert config["evaluation"]["final_blind_authorized"] is False


def test_scene_seed_bases_are_new_and_splits_do_not_alias_each_other() -> None:
    splits = _config()["data"]["splits"]
    all_bases: list[int] = []
    for split in splits.values():
        bases = split["seed_bases_by_course"] if "seed_bases_by_course" in split else [split["seed_base"]]
        assert bases
        assert min(bases) > 3_400_000_000
        all_bases.extend(int(seed) for seed in bases)
    assert len(all_bases) == len(set(all_bases))


def test_resource_contract_requires_aisar_cuda_without_cpu_fallback() -> None:
    resources = _config()["resources"]
    assert resources["required_environment"] == "AISAR"
    assert resources["required_device"] == "cuda:0"
    assert resources["required_gpu_name"] == "NVIDIA GeForce RTX 3050 Laptop GPU"
    assert resources["cpu_fallback"] == "forbidden"
    assert resources["max_project_size_gib"] == 20
