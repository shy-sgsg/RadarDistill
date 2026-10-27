from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts.run_d2a_r5c_teacher_audit import (
    _build_batch_plan,
    _local_response_metrics,
    _map_response_metrics,
    _manifest_seed_conflicts,
    _reset_cuda_peak_memory_stats,
    _validate_config,
    _validate_pair_batch,
)
from radardistill.simulation.d2a import generate_d2a_paired_batch


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict:
    value = yaml.safe_load((ROOT / "experiments/D2-A-R5/R5-C/config.yaml").read_text())
    value["_physics_config"] = yaml.safe_load(
        (ROOT / "experiments/D2-A-E001/config.yaml").read_text()
    )
    return value


def test_cuda_memory_tracking_initializes_the_explicit_device_first() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("R5-C GPU initialization test requires CUDA")

    _reset_cuda_peak_memory_stats("cuda:0")

    assert torch.cuda.current_device() == 0


def test_r5c_configuration_freezes_exact_sample_allocation(config: dict) -> None:
    _validate_config(config)
    plan = _build_batch_plan(config)

    assert len(plan) == 272
    assert len({batch["seed"] for batch in plan}) == len(plan)
    assert sum(batch["scene_count"] for batch in plan) == 4352
    assert {batch["scene_count"] for batch in plan} == {16}


def test_r5c_configuration_rejects_changed_scene_budget(config: dict) -> None:
    invalid = yaml.safe_load(yaml.safe_dump(config))
    invalid["data"]["conditions"][0]["scenes"] += 16

    with pytest.raises(ValueError, match="4352"):
        _validate_config(invalid)


def test_batch_plan_seed_ids_stay_inside_disjoint_condition_blocks(config: dict) -> None:
    plan = _build_batch_plan(config)
    block_start = config["data"]["seed_block_start"]
    block_stride = config["data"]["seed_block_stride"]
    for batch in plan:
        lower = block_start + batch["condition_index"] * block_stride
        assert lower <= batch["seed"] < lower + block_stride
        assert (batch["seed"] - lower) % config["data"]["chunk_seed_stride"] == 0


def test_generated_h0_h1_pair_has_shared_scene_ids_and_target_identity(config: dict) -> None:
    physics = config["_physics_config"]
    batch = generate_d2a_paired_batch(
        2,
        740000001,
        10.0,
        10.0,
        physics,
        scenario_id="partial_overlap",
    )

    ids = _validate_pair_batch(batch, expected_count=2)

    assert len(ids) == 2
    assert len(set(ids)) == 2
    assert np.max(np.abs(batch.iq_h1 - batch.iq_h0 - batch.target_signal_iq)) < 1e-6


def test_pair_batch_seed_matches_the_preregistered_seed(config: dict) -> None:
    physics = config["_physics_config"]
    batch = generate_d2a_paired_batch(
        2, 740000001, 10.0, 10.0, physics, scenario_id="partial_overlap"
    )

    with pytest.raises(ValueError, match="seed"):
        _validate_pair_batch(batch, expected_count=2, expected_seed=740000002)


def test_seed_manifest_scanner_detects_reserved_seed_and_scene_id() -> None:
    manifest = {
        "streams": {"train_C0": [725999999, 726000000, 726000001]},
        "selected": {"base_scene_id": "d2a-scene-727100000-4"},
    }

    conflicts = _manifest_seed_conflicts(
        manifest, reserved_low=726000000, reserved_high=727200000
    )

    assert [conflict[1] for conflict in conflicts] == [726000000, 726000001, 727100000]


@pytest.mark.parametrize("factor", [1, 4])
def test_map_metrics_measure_target_support_and_peak_errors(factor: int) -> None:
    prf_hz = 1300.0
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(130 * factor, d=1.0 / prf_hz))
    truth_range = np.asarray([10.0])
    target_doppler = float(frequency_axis[17 * factor])
    h0 = np.full((1, 130 * factor, 32), 0.0, dtype=np.float32)
    h0[0, 30 * factor, 20] = 1.0
    h1 = np.full_like(h0, 0.0)
    h1[0, 17 * factor, 10] = 0.7
    h1[0, 30 * factor, 20] = 0.3

    rows = _map_response_metrics(
        candidate="K1" if factor == 1 else "K3",
        course="C3",
        scenario_id="partial_overlap",
        snr_db=0.0,
        cnr_db=20.0,
        base_scene_ids=np.asarray(["d2a-scene-test-0"]),
        h0_response=h0,
        h1_response=h1,
        truth_range_cell=truth_range,
        truth_doppler_hz=np.asarray([target_doppler]),
        prf_hz=prf_hz,
        fft_factor=factor,
    )

    assert len(rows) == 1
    assert rows[0]["h0_window_mass"] == pytest.approx(0.0)
    assert rows[0]["h1_window_mass"] == pytest.approx(0.7)
    assert rows[0]["target_window_mass_lift"] == pytest.approx(0.7)
    assert rows[0]["range_abs_error_cells"] == pytest.approx(0.0)
    assert rows[0]["doppler_wrapped_error_hz"] == pytest.approx(0.0)
    assert rows[0]["top_cell_hit"] is True
    assert rows[0]["response_confidence"] == pytest.approx(0.7)


def test_local_response_uses_teacher_selected_range_without_truth_input() -> None:
    prf_hz = 1300.0
    factor = 4
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(130 * factor, d=1.0 / prf_hz))
    target_doppler = float(frequency_axis[41])
    h0 = np.zeros((1, 520, 32), dtype=np.float32)
    h1 = np.zeros_like(h0)
    h0[0, 200, 4] = 1.0
    h1[0, 41, 10] = 0.8
    h1[0, 200, 4] = 0.2

    rows = _local_response_metrics(
        course="C3",
        scenario_id="partial_overlap",
        snr_db=0.0,
        cnr_db=20.0,
        base_scene_ids=np.asarray(["d2a-scene-local-0"]),
        h0_response=h0,
        h1_response=h1,
        truth_range_cell=np.asarray([10.0]),
        truth_doppler_hz=np.asarray([target_doppler]),
        prf_hz=prf_hz,
    )

    assert rows[0]["predicted_range_cell"] == 10
    assert rows[0]["h1_window_mass"] == pytest.approx(1.0)
    assert rows[0]["top_cell_hit"] is True
