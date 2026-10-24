from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "experiments" / "D1-E002" / "config.yaml"


def _config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _stream(base: int, count: int, stride: int) -> set[int]:
    return {base + index * stride for index in range(count)}


def _e002_data_seeds(config: dict) -> set[int]:
    stride = config["seeds"]["stream_stride"]
    splits = config["splits"]
    seeds: set[int] = set()
    for split in [splits["pilot"], *splits["formal"]]:
        chunks = split["train_examples_per_class"] // split["train_examples_per_chunk"]
        seeds.update(_stream(split["train_seed"], chunks, stride))
        seeds.update(_stream(split["validation_seed"], len(config["data"]["snr_db"]), stride))
    calibration = splits["calibration"]
    snr_count = len(config["data"]["snr_db"])
    chunk_size = config["data"]["evaluation_generation_chunk_size"]
    for snr_index in range(snr_count):
        chunks = (calibration["h0_examples_per_snr"] + chunk_size - 1) // chunk_size
        seeds.update(
            calibration["h0_seed_base"]
            + snr_index * stride
            + chunk * stride * snr_count
            for chunk in range(chunks)
        )
    blind = splits["blind"]
    for seed_key, count_key in (("h0_seed_base", "h0_examples_per_snr"), ("h1_seed_base", "h1_examples_per_snr")):
        for snr_index in range(snr_count):
            chunks = (blind[count_key] + chunk_size - 1) // chunk_size
            seeds.update(
                blind[seed_key] + snr_index * stride + chunk * stride * snr_count
                for chunk in range(chunks)
            )
    seeds.add(splits["physical_consistency"]["seed"])
    return seeds


def _e002_all_seeds(config: dict) -> set[int]:
    seeds = _e002_data_seeds(config)
    seeds.add(config["seeds"]["bootstrap"])
    for split in [config["splits"]["pilot"], *config["splits"]["formal"]]:
        seeds.add(split["initialization_seed"])
        seeds.add(split["batch_order_seed"])
    return seeds


def _collect_named_seed_values(value: object, key: str = "") -> set[int]:
    if isinstance(value, dict):
        values: set[int] = set()
        for child_key, child_value in value.items():
            values.update(_collect_named_seed_values(child_value, str(child_key)))
        return values
    if isinstance(value, list):
        values: set[int] = set()
        for child_value in value:
            values.update(_collect_named_seed_values(child_value, key))
        return values
    if "seed" in key.lower() and isinstance(value, int) and not isinstance(value, bool):
        return {value}
    return set()


def _prior_registered_seeds() -> set[int]:
    values: set[int] = set()
    for path in (ROOT / "experiments").rglob("config.yaml"):
        if path.parent.name == "D1-E002":
            continue
        values.update(_collect_named_seed_values(yaml.safe_load(path.read_text())))

    d0 = yaml.safe_load((ROOT / "experiments" / "D0-E001" / "config.yaml").read_text())
    d1 = yaml.safe_load((ROOT / "experiments" / "D1-E001" / "config.yaml").read_text())
    seeds = values
    seeds.update(d0["seeds"].values())
    stride = d1["data"]["training_chunk_seed_stride_by_index"]
    snr_count = len(d1["data"]["evaluation_snr_db"])
    splits = d1["splits"]
    for split in [splits["pilot"], *splits["formal"]["training_seeds"]]:
        train_base = split["train_seed"] if "train_seed" in split else split["train"]
        validation_base = (
            split["validation_seed"] if "validation_seed" in split else split["validation"]
        )
        train_count = split.get(
            "train_examples_per_class", splits["formal"]["train_examples_per_class"]
        )
        chunk_size = 64  # D1-E001 protocol: 64 paired examples per chunk.
        seeds.update(_stream(train_base, train_count // chunk_size, stride))
        seeds.update(_stream(validation_base, snr_count, stride))
        seeds.add(split.get("initialization_seed", split.get("initialization")))
    calibration = splits["detection_calibration"]
    seeds.update(_stream(calibration["seed_base"], snr_count, stride))
    blind = splits["final_blind"]
    seeds.update(_stream(blind["seed_base_h0"], snr_count, stride))
    seeds.update(_stream(blind["seed_base_h1"], snr_count, stride))
    seeds.add(splits["physical_consistency_validation"]["seed"])
    return seeds


def _expected_e002_data_seed_count(config: dict) -> int:
    splits = config["splits"]
    stream_count = len(config["data"]["snr_db"])
    train_chunks = sum(
        split["train_examples_per_class"] // split["train_examples_per_chunk"]
        for split in [splits["pilot"], *splits["formal"]]
    )
    validation_streams = 1 + len(splits["formal"])
    chunk_size = config["data"]["evaluation_generation_chunk_size"]
    calibration = config["splits"]["calibration"]
    blind = config["splits"]["blind"]
    calibration_chunks = (calibration["h0_examples_per_snr"] + chunk_size - 1) // chunk_size
    blind_h0_chunks = (blind["h0_examples_per_snr"] + chunk_size - 1) // chunk_size
    blind_h1_chunks = (blind["h1_examples_per_snr"] + chunk_size - 1) // chunk_size
    return (
        train_chunks
        + validation_streams * stream_count
        + stream_count * (calibration_chunks + blind_h0_chunks + blind_h1_chunks)
        + 1
    )


def test_e002_preserves_the_frozen_single_target_signal_model() -> None:
    config = _config()
    assert config["experiment_id"] == "D1-E002"
    assert config["status"] == "preregistered_before_candidate_training"
    assert {key: config["radar"][key] for key in (
        "prf_hz", "pulse_count", "channel_count", "range_bin_count"
    )} == {
        "prf_hz": 1300,
        "pulse_count": 130,
        "channel_count": 3,
        "range_bin_count": 32,
    }
    assert config["data"]["target_count"] == "0_or_1"
    assert config["data"]["noise"] == "circular_complex_awgn"
    assert config["blind_generation"]["generated"] is False


def test_e002_seeds_are_disjoint_from_prior_registered_data_and_each_other() -> None:
    config = _config()
    seeds = _e002_data_seeds(config)
    assert len(seeds) == _expected_e002_data_seed_count(config)
    assert seeds.isdisjoint(_prior_registered_seeds())
    all_seeds = _e002_all_seeds(config)
    assert len(all_seeds) == len(seeds) + 1 + 2 * (1 + len(config["splits"]["formal"]))
    assert all_seeds.isdisjoint(_prior_registered_seeds())


def test_e002_gate_and_selection_rules_are_complete_and_frozen() -> None:
    config = _config()
    gates = config["acceptance"]
    assert gates["primary_endpoint"]["metric"] == "mean_snr_doppler_rmse_hz"
    assert gates["primary_endpoint"]["snr_db"] == [0, 5, 10, 20]
    assert gates["primary_endpoint"]["minimum_relative_gain_vs_s0"] == 0.10
    assert gates["detection_guardrail"]["pfa"] == 0.01
    assert gates["detection_guardrail"]["low_snr_db"] == [-20, -15, -10, -5]
    assert gates["detection_guardrail"]["noninferiority_margin_pd"] == 0.05
    assert gates["range_guardrail"]["snr_db"] == [0, 5, 10, 20]
    assert gates["range_guardrail"]["noninferiority_margin_cells"] == 0.05
    assert gates["uncertainty"]["bootstrap_replicates"] == 2000
    assert gates["pilot_screen"]["uses_only_pilot_validation"] is True
    assert gates["pilot_screen"]["s1_s2_comparator"] == "S0"
    assert gates["pilot_screen"]["s3_comparator"] == "matched_nonlearning_three_point_fft"
    assert config["blind_generation"]["blind_used_for_model_selection"] is False
    assert gates["hybrid_endpoint"]["applies_only_to"] == "S3"
    assert gates["hybrid_endpoint"]["minimum_relative_gain"] == 0.10
    assert gates["pilot_screen"]["no_candidate_pass_means_no_formal_training"] is True
    assert gates["pilot_screen"]["no_candidate_pass_means_no_new_blind_generation"] is True
    assert gates["stage_decision"]["no_pure_IQ_candidate_pass_means_stop_D1_E002_structure_search"] is True


def test_candidate_training_budget_and_cuda_policy_are_matched() -> None:
    config = _config()
    training = config["training"]
    candidates = [config["models"][name] for name in ("S1", "S2", "S3")]
    assert {model["optimizer_updates"] for model in candidates} == {1600}
    assert {model["effective_batch_size"] for model in candidates} == {64}
    assert training["device"] == "cuda:0"
    assert training["cpu_fallback"] is False
    assert training["precision"] == "float32"
    assert training["microbatch_fallbacks"] == [32, 16]
    assert training["oom_preserves_effective_batch_and_updates"] is True
    assert training["objective"] == "E1_physics_simulation_truth_supervision_only"
    assert training["teacher_labels_used"] is False
    assert training["teacher_soft_responses_used"] is False


def test_candidate_parameter_budgets_and_hybrid_attribution_are_frozen() -> None:
    models = _config()["models"]
    assert models["S0"]["parameter_count"] == 9093
    assert models["S1"]["parameter_count"] == 9285
    assert models["S1"]["added_parameters_vs_S0"] == 192
    assert models["S2"]["parameter_count"] == 9105
    assert models["S2"]["added_parameters_vs_S0"] == 12
    assert models["S3"]["parameter_count"] == 9412
    assert models["S3"]["added_parameters_vs_S0"] == 319
    assert models["S3"]["fft"]["nfft"] == 520
    assert models["S3"]["attribution"] == "hybrid_only_never_pure_IQ_success"


def test_physical_consistency_diagnostics_are_specified_without_posthoc_gate_changes() -> None:
    checks = _config()["splits"]["physical_consistency"]
    assert checks["examples"] == 256
    assert checks["snr_db"] == [0, 10]
    assert checks["common_phase_rotation_radians"] == [0.37, 1.41, 2.83]
    assert checks["frequency_shift_equivariance_hz"] == [0.625, 5.0]
    assert checks["purpose"] == "diagnostic_only_not_a_candidate_selection_gate"
