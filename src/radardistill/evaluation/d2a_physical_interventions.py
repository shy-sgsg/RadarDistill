"""Controlled component-strength stimuli for D2-A physical diagnostics.

These transforms scale already-synthesized target or clutter IQ components while
holding the other components and each scene's random realization fixed. They
are counterfactual diagnostic inputs, not a replacement for physical scene
simulation or evidence about hardware behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from radardistill.evaluation.physical_consistency import wrap_periodic_difference
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
    target_doppler_shift_hz: float = 0.0
    channel_phase_offsets_rad: np.ndarray | None = None
    channel_keep_mask: np.ndarray | None = None


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


def _component_metrics(
    target_iq: np.ndarray, clutter_iq: np.ndarray, noise_iq: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value, dtype=np.float64).copy()
        for name, value in compute_scene_power_metrics(target_iq, clutter_iq, noise_iq).items()
    }


def _rotate_channels(values: np.ndarray, phase_offsets_rad: np.ndarray) -> np.ndarray:
    phase = np.exp(1j * phase_offsets_rad)[None, :, None, None]
    with np.errstate(over="ignore", invalid="ignore"):
        rotated = np.asarray(values.astype(np.complex128) * phase, dtype=values.dtype)
    if np.any(~np.isfinite(rotated)):
        raise ValueError("channel phase transform produced non-finite IQ")
    return rotated


def _metadata(batch: D2APairedBatch | D2APhysicalInterventionBatch, name: str, default):
    if isinstance(batch, D2APhysicalInterventionBatch):
        return getattr(batch, name)
    return default


def scale_d2a_paired_components(
    batch: D2APairedBatch | D2APhysicalInterventionBatch,
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

    prior_channel_phase = _metadata(batch, "channel_phase_offsets_rad", None)
    prior_channel_mask = _metadata(batch, "channel_keep_mask", None)
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
        target_power_shift_db=(
            _metadata(batch, "target_power_shift_db", 0.0) + float(target_power_shift_db)
        ),
        clutter_power_shift_db=(
            _metadata(batch, "clutter_power_shift_db", 0.0) + float(clutter_power_shift_db)
        ),
        component_metrics=_component_metrics(target_iq, clutter_iq, noise_iq),
        target_doppler_shift_hz=_metadata(batch, "target_doppler_shift_hz", 0.0),
        channel_phase_offsets_rad=(
            None if prior_channel_phase is None else np.asarray(prior_channel_phase).copy()
        ),
        channel_keep_mask=(
            None if prior_channel_mask is None else np.asarray(prior_channel_mask).copy()
        ),
    )


def shift_d2a_target_doppler(
    batch: D2APairedBatch | D2APhysicalInterventionBatch,
    *,
    doppler_shift_hz: float,
    prf_hz: float,
) -> D2APhysicalInterventionBatch:
    """Apply ``exp(j 2π Δf n / PRF)`` to target IQ and wrap Doppler truth.

    Clutter, noise, and H0 are fixed. This exact slow-time modulation is a
    controlled target-frequency counterfactual; it does not resynthesize the
    target trajectory or update a physical velocity label.
    """
    shift = float(doppler_shift_hz)
    prf = float(prf_hz)
    if not np.isfinite(shift):
        raise ValueError("doppler_shift_hz must be finite")
    if not np.isfinite(prf) or prf <= 0.0:
        raise ValueError("prf_hz must be positive and finite")

    baseline = scale_d2a_paired_components(batch)
    pulse_count = baseline.target_signal_iq.shape[2]
    pulse_times = np.arange(pulse_count, dtype=np.float64) / prf
    wrapped_shift = float(wrap_periodic_difference(shift, period=prf))
    phase = np.exp(2j * np.pi * wrapped_shift * pulse_times)[None, None, :, None]
    with np.errstate(over="ignore", invalid="ignore"):
        shifted_target = np.asarray(
            baseline.target_signal_iq.astype(np.complex128) * phase,
            dtype=baseline.target_signal_iq.dtype,
        )
    if np.any(~np.isfinite(shifted_target)):
        raise ValueError("Doppler shift produced non-finite target IQ")
    shifted_h1 = np.asarray(baseline.iq_h0 + shifted_target, dtype=baseline.iq_h1.dtype)
    shifted_truth = wrap_periodic_difference(
        baseline.truth_target_doppler_hz + wrapped_shift, period=prf
    )
    return replace(
        baseline,
        iq_h1=shifted_h1,
        target_signal_iq=shifted_target,
        truth_target_doppler_hz=shifted_truth,
        target_doppler_shift_hz=baseline.target_doppler_shift_hz + shift,
        component_metrics=_component_metrics(
            shifted_target, baseline.clutter_iq, baseline.noise_iq
        ),
    )


def apply_d2a_channel_phase_offsets(
    batch: D2APairedBatch | D2APhysicalInterventionBatch,
    phase_offsets_rad: np.ndarray,
) -> D2APhysicalInterventionBatch:
    """Apply fixed receive-channel phase offsets to every IQ component.

    A common phase is invariant across channels; unequal offsets perturb the
    relative array phase. Applying the negative offsets to this returned batch
    restores the original coherent IQ to numerical precision.
    """
    baseline = scale_d2a_paired_components(batch)
    offsets = np.asarray(phase_offsets_rad, dtype=np.float64).reshape(-1)
    channel_count = baseline.iq_h0.shape[1]
    if offsets.size != channel_count:
        raise ValueError("provide one phase offset per channel")
    if np.any(~np.isfinite(offsets)):
        raise ValueError("phase offsets must be finite")

    clutter_iq = _rotate_channels(baseline.clutter_iq, offsets)
    noise_iq = _rotate_channels(baseline.noise_iq, offsets)
    target_iq = _rotate_channels(baseline.target_signal_iq, offsets)
    iq_h0 = np.asarray(clutter_iq + noise_iq, dtype=baseline.iq_h0.dtype)
    iq_h1 = np.asarray(iq_h0 + target_iq, dtype=baseline.iq_h1.dtype)
    prior = baseline.channel_phase_offsets_rad
    cumulative = np.zeros(channel_count, dtype=np.float64) if prior is None else prior
    cumulative = wrap_periodic_difference(cumulative + offsets, period=2.0 * np.pi)
    keep_mask = baseline.channel_keep_mask
    return replace(
        baseline,
        iq_h0=iq_h0,
        iq_h1=iq_h1,
        clutter_iq=clutter_iq,
        noise_iq=noise_iq,
        target_signal_iq=target_iq,
        channel_phase_offsets_rad=cumulative,
        channel_keep_mask=None if keep_mask is None else keep_mask.copy(),
        component_metrics=_component_metrics(target_iq, clutter_iq, noise_iq),
    )


def mask_d2a_channels(
    batch: D2APairedBatch | D2APhysicalInterventionBatch,
    keep_channels: np.ndarray,
) -> D2APhysicalInterventionBatch:
    """Zero omitted receive channels in H0/H1 and all component IQ arrays.

    The keep mask is cumulative. To restore channels, start from the original
    paired batch; a zeroed channel cannot be recovered from a masked result.
    """
    baseline = scale_d2a_paired_components(batch)
    keep = np.asarray(keep_channels, dtype=bool).reshape(-1)
    channel_count = baseline.iq_h0.shape[1]
    if keep.size != channel_count:
        raise ValueError("provide one keep flag per channel")
    if not np.any(keep):
        raise ValueError("at least one channel must be kept")
    prior = baseline.channel_keep_mask
    cumulative = keep if prior is None else (prior & keep)
    if not np.any(cumulative):
        raise ValueError("at least one cumulative channel must be kept")

    def mask(values: np.ndarray) -> np.ndarray:
        return np.asarray(values * cumulative[None, :, None, None], dtype=values.dtype)

    clutter_iq = mask(baseline.clutter_iq)
    noise_iq = mask(baseline.noise_iq)
    target_iq = mask(baseline.target_signal_iq)
    iq_h0 = np.asarray(clutter_iq + noise_iq, dtype=baseline.iq_h0.dtype)
    iq_h1 = np.asarray(iq_h0 + target_iq, dtype=baseline.iq_h1.dtype)
    phase_offsets = baseline.channel_phase_offsets_rad
    return replace(
        baseline,
        iq_h0=iq_h0,
        iq_h1=iq_h1,
        clutter_iq=clutter_iq,
        noise_iq=noise_iq,
        target_signal_iq=target_iq,
        channel_phase_offsets_rad=(
            None if phase_offsets is None else phase_offsets.copy()
        ),
        channel_keep_mask=cumulative.copy(),
        component_metrics=_component_metrics(target_iq, clutter_iq, noise_iq),
    )
