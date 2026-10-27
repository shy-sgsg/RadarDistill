from __future__ import annotations

import numpy as np
import pytest
import yaml

from radardistill.simulation.d2a import (
    compute_scene_power_metrics,
    generate_d2a_paired_batch,
)
from radardistill.evaluation.d2a_physical_interventions import (
    apply_d2a_channel_phase_offsets,
    mask_d2a_channels,
    scale_d2a_paired_components,
    shift_d2a_target_doppler,
)
from radardistill.evaluation.physical_consistency import wrap_periodic_difference


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


def test_target_doppler_shift_modulates_target_only_and_wraps_truth(paired_batch):
    prf_hz = 1300.0
    delta_hz = 17.5
    shifted = shift_d2a_target_doppler(
        paired_batch, doppler_shift_hz=delta_hz, prf_hz=prf_hz
    )
    pulse_phase = np.exp(2j * np.pi * delta_hz * np.arange(130) / prf_hz)

    np.testing.assert_array_equal(shifted.iq_h0, paired_batch.iq_h0)
    np.testing.assert_array_equal(shifted.clutter_iq, paired_batch.clutter_iq)
    np.testing.assert_array_equal(shifted.noise_iq, paired_batch.noise_iq)
    np.testing.assert_allclose(
        shifted.target_signal_iq,
        paired_batch.target_signal_iq * pulse_phase[None, None, :, None],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        shifted.truth_target_doppler_hz,
        wrap_periodic_difference(
            paired_batch.truth_target_doppler_hz + delta_hz, period=prf_hz
        ),
        rtol=0.0,
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        shifted.truth_target_range_cell, paired_batch.truth_target_range_cell
    )
    np.testing.assert_allclose(
        shifted.iq_h1,
        shifted.iq_h0 + shifted.target_signal_iq,
        rtol=1e-6,
        atol=1e-6,
    )


def test_prf_doppler_shift_is_an_aliased_identity(paired_batch):
    shifted = shift_d2a_target_doppler(
        paired_batch, doppler_shift_hz=1300.0, prf_hz=1300.0
    )

    np.testing.assert_allclose(shifted.target_signal_iq, paired_batch.target_signal_iq, atol=1e-6)
    np.testing.assert_allclose(
        shifted.truth_target_doppler_hz, paired_batch.truth_target_doppler_hz, atol=1e-10
    )


def test_channel_phase_perturbation_is_reversible(paired_batch):
    offsets = np.asarray([0.4, -0.7, 1.1], dtype=np.float64)
    perturbed = apply_d2a_channel_phase_offsets(paired_batch, offsets)
    restored = apply_d2a_channel_phase_offsets(perturbed, -offsets)

    channel_phase = np.exp(1j * offsets)[None, :, None, None]
    np.testing.assert_allclose(
        perturbed.iq_h1, paired_batch.iq_h1 * channel_phase, rtol=1e-6, atol=1e-6
    )
    base_cross_channel = np.sum(
        paired_batch.iq_h1[:, 0] * np.conj(paired_batch.iq_h1[:, 1]), axis=(1, 2)
    )
    perturbed_cross_channel = np.sum(
        perturbed.iq_h1[:, 0] * np.conj(perturbed.iq_h1[:, 1]), axis=(1, 2)
    )
    np.testing.assert_allclose(
        perturbed_cross_channel,
        base_cross_channel * np.exp(1j * (offsets[0] - offsets[1])),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(restored.iq_h0, paired_batch.iq_h0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(restored.iq_h1, paired_batch.iq_h1, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        restored.channel_phase_offsets_rad, np.zeros(3), rtol=0.0, atol=1e-12
    )


def test_channel_mask_changes_all_components_and_can_be_recovered_from_source(paired_batch):
    keep = np.asarray([True, False, True])
    masked = mask_d2a_channels(paired_batch, keep)
    restored = mask_d2a_channels(paired_batch, np.ones(3, dtype=bool))

    for component_name in ("iq_h0", "iq_h1", "clutter_iq", "noise_iq", "target_signal_iq"):
        component = getattr(masked, component_name)
        assert np.all(component[:, 1] == 0)
        np.testing.assert_array_equal(component[:, 0], getattr(paired_batch, component_name)[:, 0])
        np.testing.assert_array_equal(component[:, 2], getattr(paired_batch, component_name)[:, 2])
        np.testing.assert_array_equal(
            getattr(restored, component_name), getattr(paired_batch, component_name)
        )
    np.testing.assert_array_equal(masked.channel_keep_mask, keep)
    np.testing.assert_array_equal(restored.channel_keep_mask, np.ones(3, dtype=bool))


def test_invalid_channel_interventions_are_rejected(paired_batch):
    with pytest.raises(ValueError, match="one phase offset per channel"):
        apply_d2a_channel_phase_offsets(paired_batch, [0.1, 0.2])
    with pytest.raises(ValueError, match="at least one channel"):
        mask_d2a_channels(paired_batch, [False, False, False])


def test_cumulative_channel_mask_cannot_remove_every_channel(paired_batch):
    first_mask = mask_d2a_channels(paired_batch, [True, False, False])

    with pytest.raises(ValueError, match="at least one cumulative channel"):
        mask_d2a_channels(first_mask, [False, True, True])
