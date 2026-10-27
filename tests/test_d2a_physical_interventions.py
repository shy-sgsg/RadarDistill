from __future__ import annotations

import numpy as np
import pytest
import yaml

from radardistill.simulation.d2a import (
    compute_scene_power_metrics,
    generate_d2a_paired_batch,
)
from radardistill.evaluation.d2a_physical_interventions import (
    scale_d2a_paired_components,
)


@pytest.fixture(scope="module")
def d2a_config():
    with open("experiments/D2-A-E001/config.yaml", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture()
def paired_batch(d2a_config):
    return generate_d2a_paired_batch(3, 817301, 10.0, 5.0, d2a_config)


def test_zero_component_shift_preserves_original_paired_scenes(paired_batch):
    intervention = scale_d2a_paired_components(paired_batch)

    np.testing.assert_array_equal(intervention.iq_h0, paired_batch.iq_h0)
    np.testing.assert_array_equal(intervention.iq_h1, paired_batch.iq_h1)
    np.testing.assert_array_equal(intervention.clutter_iq, paired_batch.clutter_iq)
    np.testing.assert_array_equal(intervention.noise_iq, paired_batch.noise_iq)
    np.testing.assert_array_equal(intervention.target_signal_iq, paired_batch.target_signal_iq)
    np.testing.assert_array_equal(intervention.base_scene_id, paired_batch.base_scene_id)
    np.testing.assert_array_equal(
        intervention.truth_target_amplitude, paired_batch.truth_target_amplitude
    )


def test_target_power_intervention_changes_only_h1_target_component(paired_batch):
    intervention = scale_d2a_paired_components(
        paired_batch, target_power_shift_db=20.0 * np.log10(2.0)
    )

    np.testing.assert_array_equal(intervention.iq_h0, paired_batch.iq_h0)
    np.testing.assert_array_equal(intervention.clutter_iq, paired_batch.clutter_iq)
    np.testing.assert_array_equal(intervention.noise_iq, paired_batch.noise_iq)
    np.testing.assert_allclose(
        intervention.target_signal_iq,
        2.0 * paired_batch.target_signal_iq,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        intervention.iq_h1,
        intervention.iq_h0 + intervention.target_signal_iq,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        intervention.truth_target_amplitude,
        2.0 * paired_batch.truth_target_amplitude,
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        intervention.truth_target_range_cell, paired_batch.truth_target_range_cell
    )
    np.testing.assert_array_equal(
        intervention.truth_target_doppler_hz, paired_batch.truth_target_doppler_hz
    )
    np.testing.assert_allclose(
        intervention.component_metrics["target_power"],
        4.0
        * compute_scene_power_metrics(
            paired_batch.target_signal_iq, paired_batch.clutter_iq, paired_batch.noise_iq
        )["target_power"],
        rtol=2e-6,
        atol=1e-8,
    )


def test_clutter_power_intervention_preserves_target_and_noise(paired_batch):
    intervention = scale_d2a_paired_components(
        paired_batch, clutter_power_shift_db=20.0 * np.log10(2.0)
    )

    np.testing.assert_allclose(
        intervention.clutter_iq,
        2.0 * paired_batch.clutter_iq,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(intervention.noise_iq, paired_batch.noise_iq)
    np.testing.assert_array_equal(
        intervention.target_signal_iq, paired_batch.target_signal_iq
    )
    np.testing.assert_allclose(
        intervention.iq_h0,
        intervention.clutter_iq + intervention.noise_iq,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        intervention.iq_h1,
        intervention.iq_h0 + intervention.target_signal_iq,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        intervention.component_metrics["cnr_db"],
        compute_scene_power_metrics(
            paired_batch.target_signal_iq, paired_batch.clutter_iq, paired_batch.noise_iq
        )["cnr_db"]
        + 20.0 * np.log10(2.0),
        rtol=0.0,
        atol=2e-5,
    )


@pytest.mark.parametrize("field", ["target_power_shift_db", "clutter_power_shift_db"])
def test_nonfinite_power_shift_is_rejected(paired_batch, field):
    with pytest.raises(ValueError, match="finite"):
        scale_d2a_paired_components(paired_batch, **{field: np.inf})


def test_unrepresentable_component_scale_is_rejected(paired_batch):
    with pytest.raises(ValueError, match="not representable"):
        scale_d2a_paired_components(paired_batch, target_power_shift_db=2000.0)


def test_inconsistent_source_components_are_rejected(paired_batch):
    corrupted = type(paired_batch)(
        **{
            name: (
                np.array(getattr(paired_batch, name), copy=True)
                if isinstance(getattr(paired_batch, name), np.ndarray)
                else getattr(paired_batch, name)
            )
            for name in paired_batch.__dataclass_fields__
        }
    )
    corrupted.clutter_iq[0, 0, 0, 0] += np.complex64(1.0)

    with pytest.raises(ValueError, match="iq_h0.*clutter.*noise"):
        scale_d2a_paired_components(corrupted)
