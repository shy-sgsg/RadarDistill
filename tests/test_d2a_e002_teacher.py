from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from radardistill.teachers.d2a_k2 import k2_response
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.training.d2a_e002_data import (
    materialize_h0_mixture,
    materialize_paired_split,
    validate_disjoint_scene_ids,
)


@pytest.fixture
def d2a_config():
    with open("experiments/D2-A-E001/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def test_k2_is_finite_nonnegative_normalized_and_uses_only_iq(d2a_config):
    batch = generate_d2a_paired_batch(
        3, 93001, 10.0, 10.0, d2a_config, scenario_id="partial_overlap"
    )
    response = k2_response(batch.iq_h1, d2a_config)
    assert response.shape == (3, 130, 32)
    assert np.all(np.isfinite(response))
    assert np.all(response >= 0.0)
    np.testing.assert_allclose(response.sum(axis=(1, 2)), 1.0, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("scale", [0.5, 2.0, 0.6 + 1.2j])
def test_k2_normalization_is_invariant_to_common_complex_scale(d2a_config, scale):
    batch = generate_d2a_paired_batch(
        2, 93002, 20.0, -5.0, d2a_config, scenario_id="separated"
    )
    original = k2_response(batch.iq_h1, d2a_config)
    scaled = k2_response(batch.iq_h1.astype(np.complex128) * scale, d2a_config)
    np.testing.assert_allclose(scaled, original, rtol=1e-6, atol=1e-6)


def test_k2_api_has_no_truth_or_metadata_arguments():
    import inspect

    assert tuple(inspect.signature(k2_response).parameters) == ("iq", "config")


def test_k2_does_not_mutate_the_observed_iq(d2a_config):
    batch = generate_d2a_paired_batch(
        1, 93003, 10.0, 10.0, d2a_config, scenario_id="near_clutter_edge"
    )
    iq = batch.iq_h1.copy()
    before = copy.deepcopy(iq)
    k2_response(iq, d2a_config)
    np.testing.assert_array_equal(iq, before)


def test_paired_split_cache_preserves_counterfactual_pairs_and_cell_metadata(tmp_path, d2a_config):
    with open("experiments/D2-A-E002/config.yaml", encoding="utf-8") as stream:
        e002 = yaml.safe_load(stream)
    cell_counts = {
        "C0": {"separated": 1},
        "C1": {"near_clutter_edge": 1},
        "C2": {"partial_overlap": 1},
        "C3": {"separated_0db": 1},
    }
    arrays = materialize_paired_split(
        tmp_path,
        "tiny_pairs",
        d2a_config,
        e002["data"]["course_levels"],
        cell_counts,
        [9400000, 9500000, 9600000, 9700000],
        assignment_seed=31,
        chunk_size=1,
        include_k2=True,
    )
    assert arrays["iq_h0"].shape == (4, 3, 130, 32)
    assert arrays["iq_h1"].shape == (4, 3, 130, 32)
    assert arrays["teacher_response"].shape == (4, 130, 32)
    assert set(arrays["course_level"].astype(str)) == {"C0", "C1", "C2", "C3"}
    assert set(arrays["scenario_id"].astype(str)) == {
        "separated", "near_clutter_edge", "partial_overlap"
    }
    assert np.all(np.isfinite(arrays["iq_h0"]))
    assert np.all(np.isfinite(arrays["iq_h1"]))
    assert np.all(np.any(arrays["iq_h1"] != arrays["iq_h0"], axis=(1, 2, 3)))
    assert np.all(np.isfinite(arrays["teacher_response"]))
    np.testing.assert_allclose(arrays["teacher_response"].sum(axis=(1, 2)), 1.0, atol=1e-6)
    reopened = materialize_paired_split(
        tmp_path,
        "tiny_pairs",
        d2a_config,
        e002["data"]["course_levels"],
        cell_counts,
        [9400000, 9500000, 9600000, 9700000],
        assignment_seed=31,
        chunk_size=1,
        include_k2=True,
    )
    np.testing.assert_array_equal(arrays["iq_h0"], reopened["iq_h0"])
    validate_disjoint_scene_ids({"tiny_pairs": arrays})


def test_h0_mixture_cache_keeps_amplitude_as_diagnostic_only(tmp_path, d2a_config):
    with open("experiments/D2-A-E002/config.yaml", encoding="utf-8") as stream:
        e002 = yaml.safe_load(stream)
    arrays = materialize_h0_mixture(
        tmp_path,
        "tiny_h0",
        8,
        d2a_config,
        e002["data"]["course_levels"],
        seed_base=9800000,
        mixture_assignment_seed=41,
        chunk_size=4,
    )
    assert arrays["iq_h0"].shape == (8, 3, 130, 32)
    assert set(arrays["course_level"].astype(str)).issubset({"C0", "C1", "C2", "C3"})
    expected_noise_power = (
        arrays["truth_amplitude_diagnostic"].astype(np.float64) ** 2
        * 10.0 ** (-arrays["snr_db"].astype(np.float64) / 10.0)
    )
    np.testing.assert_allclose(
        arrays["noise_power_diagnostic"], expected_noise_power, rtol=2e-6, atol=1e-7
    )
    validate_disjoint_scene_ids({"tiny_h0": arrays})


def test_split_disjointness_rejects_shared_base_scene_id():
    first = {"base_scene_id": np.asarray(["scene-a", "scene-b"])}
    second = {"base_scene_id": np.asarray(["scene-b", "scene-c"])}
    with pytest.raises(ValueError, match="overlaps"):
        validate_disjoint_scene_ids({"first": first, "second": second})
