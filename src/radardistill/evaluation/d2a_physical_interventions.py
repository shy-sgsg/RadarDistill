"""Controlled component-strength stimuli for D2-A physical diagnostics.

These transforms scale already-synthesized target or clutter IQ components while
holding the other components and each scene's random realization fixed. They
are counterfactual diagnostic inputs, not a replacement for physical scene
simulation or evidence about hardware behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from radardistill.simulation.d2a import D2APairedBatch, compute_scene_power_metrics


@dataclass(frozen=True)
class D2APhysicalInterventionBatch:
    """Paired IQ and truth for a component-scaled copy of source scenes."""

    iq_h0: np.ndarray
    iq_h1: np.ndarray
    clutter_iq: np.ndarray
    noise_iq: np.ndarray
    target_signal_iq: np.ndarray
    truth_presence_h0: np.ndarray
    truth_presence_h1: np.ndarray
    truth_target_range_cell: np.ndarray
    truth_target_doppler_hz: np.ndarray
    truth_target_amplitude: np.ndarray
    base_scene_id: np.ndarray
    target_power_shift_db: float
    clutter_power_shift_db: float
    component_metrics: dict[str, np.ndarray]


def _power_shift_to_amplitude_scale(power_shift_db: float, *, name: str) -> float:
    value = float(power_shift_db)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = float(np.power(10.0, value / 20.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} is outside the representable amplitude-scale range")
    return scale


def _validated_component(batch: D2APairedBatch, name: str) -> np.ndarray:
    values = np.asarray(getattr(batch, name))
    if values.ndim != 4 or not np.iscomplexobj(values):
        raise ValueError(f"{name} must be complex IQ with shape (scenes, channels, pulses, range)")
    if values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError(f"{name} must be nonempty and finite")
    return values


def _scaled_iq(values: np.ndarray, amplitude_scale: float) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        scaled = np.asarray(values.astype(np.complex128) * amplitude_scale, dtype=values.dtype)
    if np.any(~np.isfinite(scaled)):
        raise ValueError("scaled IQ component is not representable in its source dtype")
    return scaled


def scale_d2a_paired_components(
    batch: D2APairedBatch,
    *,
    target_power_shift_db: float = 0.0,
    clutter_power_shift_db: float = 0.0,
) -> D2APhysicalInterventionBatch:
    """Scale target and/or clutter power in paired H0/H1 scenes.

    Shifts are power changes in dB and become IQ-amplitude multipliers of
    ``10 ** (shift_db / 20)``. Noise is unchanged. Target scaling changes H1
    only and updates target-amplitude truth; clutter scaling changes both H0
    and H1 while preserving the target component. Range, Doppler, phases,
    scene IDs, and component realizations remain fixed.

    The returned data are controlled component-level counterfactuals. They do
    not resynthesize scatterers, represent a new independent scene, or assert
    that real sensors follow this exact scaling model.
    """
    target_scale = _power_shift_to_amplitude_scale(
        target_power_shift_db, name="target_power_shift_db"
    )
    clutter_scale = _power_shift_to_amplitude_scale(
        clutter_power_shift_db, name="clutter_power_shift_db"
    )

    iq_h0_source = _validated_component(batch, "iq_h0")
    iq_h1_source = _validated_component(batch, "iq_h1")
    clutter_source = _validated_component(batch, "clutter_iq")
    noise_source = _validated_component(batch, "noise_iq")
    target_source = _validated_component(batch, "target_signal_iq")
    component_shape = clutter_source.shape
    if any(
        values.shape != component_shape
        for values in (iq_h0_source, iq_h1_source, noise_source, target_source)
    ):
        raise ValueError("all paired IQ components must have identical shapes")

    source_h0 = np.asarray(clutter_source + noise_source, dtype=iq_h0_source.dtype)
    if not np.allclose(iq_h0_source, source_h0, rtol=1e-6, atol=1e-6):
        raise ValueError("iq_h0 must equal clutter_iq + noise_iq")
    source_h1 = np.asarray(iq_h0_source + target_source, dtype=iq_h1_source.dtype)
    if not np.allclose(iq_h1_source, source_h1, rtol=1e-6, atol=1e-6):
        raise ValueError("iq_h1 must equal iq_h0 + target_signal_iq")

    scene_count = component_shape[0]
    scene_ids = np.asarray(batch.base_scene_id).reshape(-1).astype(str)
    if (
        scene_ids.size != scene_count
        or np.any(scene_ids == "")
        or np.unique(scene_ids).size != scene_count
    ):
        raise ValueError("base_scene_id must contain one unique nonempty ID per scene")

    presence_h0 = np.asarray(batch.truth_presence_h0, dtype=bool).reshape(-1)
    presence_h1 = np.asarray(batch.truth_presence_h1, dtype=bool).reshape(-1)
    if (
        presence_h0.size != scene_count
        or presence_h1.size != scene_count
        or np.any(presence_h0)
        or np.any(~presence_h1)
    ):
        raise ValueError("paired batch must label every H0 absent and every H1 present")

    truth_range = np.asarray(batch.truth_target_range_cell, dtype=np.float64).reshape(-1)
    truth_doppler = np.asarray(batch.truth_target_doppler_hz, dtype=np.float64).reshape(-1)
    truth_amplitude = np.asarray(batch.truth_target_amplitude, dtype=np.float64).reshape(-1)
    if any(values.size != scene_count for values in (truth_range, truth_doppler, truth_amplitude)):
        raise ValueError("target truth fields must contain one value per scene")
    if any(
        np.any(~np.isfinite(values))
        for values in (truth_range, truth_doppler, truth_amplitude)
    ):
        raise ValueError("target truth fields must be finite")
    changed_truth_amplitude = truth_amplitude * target_scale
    if np.any(~np.isfinite(changed_truth_amplitude)):
        raise ValueError("scaled target-amplitude truth is not representable")

    clutter_iq = _scaled_iq(clutter_source, clutter_scale)
    noise_iq = noise_source.copy()
    target_iq = _scaled_iq(target_source, target_scale)
    if clutter_scale == 1.0:
        iq_h0 = iq_h0_source.copy()
    else:
        iq_h0 = np.asarray(clutter_iq + noise_iq, dtype=iq_h0_source.dtype)
    if target_scale == 1.0 and clutter_scale == 1.0:
        iq_h1 = iq_h1_source.copy()
    else:
        iq_h1 = np.asarray(iq_h0 + target_iq, dtype=iq_h1_source.dtype)

    metrics = compute_scene_power_metrics(target_iq, clutter_iq, noise_iq)
    return D2APhysicalInterventionBatch(
        iq_h0=iq_h0,
        iq_h1=iq_h1,
        clutter_iq=clutter_iq,
        noise_iq=noise_iq,
        target_signal_iq=target_iq,
        truth_presence_h0=presence_h0.copy(),
        truth_presence_h1=presence_h1.copy(),
        truth_target_range_cell=truth_range.copy(),
        truth_target_doppler_hz=truth_doppler.copy(),
        truth_target_amplitude=changed_truth_amplitude,
        base_scene_id=scene_ids.copy(),
        target_power_shift_db=float(target_power_shift_db),
        clutter_power_shift_db=float(clutter_power_shift_db),
        component_metrics={
            name: np.asarray(value, dtype=np.float64).copy()
            for name, value in metrics.items()
        },
    )
