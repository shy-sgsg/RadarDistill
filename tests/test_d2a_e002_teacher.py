from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from radardistill.teachers.d2a_k2 import k2_response
from radardistill.simulation.d2a import generate_d2a_paired_batch


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
