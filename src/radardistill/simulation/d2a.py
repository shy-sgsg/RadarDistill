"""Paired, geometry-driven three-channel stationary-clutter simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from radardistill.simulation.d0 import D0Batch, generate_complex_awgn, generate_d0_batch


@dataclass(frozen=True)
class ScattererEchoResponse:
    """Range-compressed response and geometry diagnostics for point scatterers."""

    iq: np.ndarray
    path_length_m: np.ndarray
    effective_range_m: np.ndarray
    range_cell: np.ndarray
    beam_gain: np.ndarray
    scatterer_positions_xyz_m: np.ndarray
    reflectivity: np.ndarray


@dataclass(frozen=True)
class D2APairedBatch:
    """Counterfactual H0/H1 scenes sharing clutter and complex noise."""

    iq_h0: np.ndarray
    iq_h1: np.ndarray
    clutter_iq: np.ndarray
    noise_iq: np.ndarray
    target_signal_iq: np.ndarray
    truth_presence_h0: np.ndarray
    truth_presence_h1: np.ndarray
    counterfactual_target_range_cell: np.ndarray
    truth_target_range_cell: np.ndarray
    truth_target_doppler_hz: np.ndarray
    truth_target_radial_velocity_mps: np.ndarray
    truth_target_azimuth_deg: np.ndarray
    truth_target_phase_rad: np.ndarray
    truth_target_amplitude: np.ndarray
    noise_power: np.ndarray
    snr_db: np.ndarray
    requested_cnr_db: np.ndarray
    base_scene_id: np.ndarray
    background_id: np.ndarray
    target_id: np.ndarray
    realization_id: np.ndarray
    scatterer_positions_xyz_m: np.ndarray
    scatterer_reflectivity: np.ndarray


def _required(config: dict[str, Any], section: str, key: str) -> Any:
    try:
        return config[section][key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing D2-A configuration value {section}.{key}") from exc


def _validate_scatterer_inputs(
    positions: np.ndarray, reflectivity: np.ndarray, pulse_times: np.ndarray
) -> None:
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("scatterer_positions_xyz_m must have shape (scatterers, 3)")
    if reflectivity.shape != (positions.shape[0],):
        raise ValueError("reflectivity must have one complex value per scatterer")
    if pulse_times.ndim != 1 or pulse_times.size < 2 or np.any(~np.isfinite(pulse_times)):
        raise ValueError("pulse_times_s must be a finite vector with at least two samples")
    if np.any(~np.isfinite(positions)) or np.any(~np.isfinite(reflectivity)):
        raise ValueError("scatterer positions and reflectivity must be finite")


def synthesize_scatterer_echoes(
    scatterer_positions_xyz_m: np.ndarray,
    reflectivity: np.ndarray,
    pulse_times_s: np.ndarray,
    config: dict[str, Any],
    *,
    platform_position_xyz_m: np.ndarray | None = None,
    platform_velocity_xyz_mps: np.ndarray | None = None,
    scatterer_velocity_xyz_mps: np.ndarray | None = None,
    beam_center_azimuth_deg: float | None = None,
) -> ScattererEchoResponse:
    """Synthesize shared point scatterers with exact per-pulse Tx/Rx path phase.

    Coordinates use the configured local frame. Positive azimuth runs from +x
    toward +y. The transmitter phase center is an explicitly configured offset
    from the platform reference; receive centers are channel-specific. The
    returned slow-time phase follows ``exp(-j 2 pi (Rtx + Rrx) / wavelength)``.
    """
    positions = np.asarray(scatterer_positions_xyz_m, dtype=np.float64)
    coefficients = np.asarray(reflectivity, dtype=np.complex128)
    pulse_times = np.asarray(pulse_times_s, dtype=np.float64)
    _validate_scatterer_inputs(positions, coefficients, pulse_times)

    channel_positions = np.asarray(_required(config, "channels", "positions_xyz_m"), dtype=np.float64)
    if channel_positions.ndim != 2 or channel_positions.shape[1] != 3:
        raise ValueError("channels.positions_xyz_m must have shape (channels, 3)")
    expected_channels = int(_required(config, "radar", "channel_count"))
    if channel_positions.shape[0] != expected_channels:
        raise ValueError("channel position count must match radar.channel_count")

    platform_origin = np.asarray(
        platform_position_xyz_m
        if platform_position_xyz_m is not None
        else _required(config, "platform", "initial_position_xyz_m"),
        dtype=np.float64,
    )
    platform_velocity = np.asarray(
        platform_velocity_xyz_mps
        if platform_velocity_xyz_mps is not None
        else _required(config, "platform", "velocity_xyz_mps"),
        dtype=np.float64,
    )
    if platform_origin.shape != (3,) or platform_velocity.shape != (3,):
        raise ValueError("platform position and velocity must be three-vectors")

    if scatterer_velocity_xyz_mps is None:
        scatterer_velocity = np.zeros_like(positions)
    else:
        scatterer_velocity = np.asarray(scatterer_velocity_xyz_mps, dtype=np.float64)
        if scatterer_velocity.shape != positions.shape or np.any(~np.isfinite(scatterer_velocity)):
            raise ValueError("scatterer_velocity_xyz_mps must match scatterer position shape")

    tx_offset = np.asarray(
        _required(config, "transmitter", "phase_center_offset_xyz_m"), dtype=np.float64
    )
    if tx_offset.shape != (3,):
        raise ValueError("transmitter.phase_center_offset_xyz_m must be a three-vector")

    radar = config["radar"]
    waveform = config["waveform"]
    range_grid = config["range_grid"]
    beam = config["beam"]
    pulse_count = int(radar["pulse_count"])
    range_count = int(range_grid["count"])
    if pulse_times.shape != (pulse_count,):
        raise ValueError(f"pulse_times_s must have shape ({pulse_count},)")
    if range_count != int(radar["range_bin_count"]):
        raise ValueError("range_grid.count must match radar.range_bin_count")
    carrier_hz = float(waveform["carrier_hz"])
    propagation_speed_mps = float(waveform["propagation_speed_mps"])
    range_cell_m = float(waveform["range_cell_m"])
    range_start_m = float(range_grid["axis_start_m"])
    beam_width_deg = float(beam["azimuth_beamwidth_deg"])
    beam_center = float(
        beam["center_azimuth_deg"]
        if beam_center_azimuth_deg is None
        else beam_center_azimuth_deg
    )
    if min(carrier_hz, propagation_speed_mps, range_cell_m, beam_width_deg) <= 0.0:
        raise ValueError("carrier, propagation speed, range cell, and beamwidth must be positive")

    platform_by_time = platform_origin[None, :] + pulse_times[:, None] * platform_velocity[None, :]
    scatterers_by_time = positions[None, :, :] + pulse_times[:, None, None] * scatterer_velocity[None, :, :]
    tx_by_time = platform_by_time + tx_offset[None, :]
    rx_by_time = platform_by_time[None, :, :] + channel_positions[:, None, :]

    tx_vectors = scatterers_by_time - tx_by_time[:, None, :]
    tx_ranges = np.linalg.norm(tx_vectors, axis=-1)
    rx_vectors = scatterers_by_time[None, :, :, :] - rx_by_time[:, :, None, :]
    rx_ranges = np.linalg.norm(rx_vectors, axis=-1)
    path_length = tx_ranges[None, :, :] + rx_ranges
    effective_range = path_length / 2.0
    range_cell = (effective_range - range_start_m) / range_cell_m

    scatterer_azimuth = np.rad2deg(np.arctan2(tx_vectors[..., 1], tx_vectors[..., 0]))
    azimuth_offset = (scatterer_azimuth - beam_center + 180.0) % 360.0 - 180.0
    beam_gain = np.exp(
        -2.0 * np.log(2.0) * (azimuth_offset / beam_width_deg) ** 2
    )

    wavelength = propagation_speed_mps / carrier_hz
    phase = np.exp(-2j * np.pi * path_length / wavelength)
    range_axis = np.arange(range_count, dtype=np.float64)[None, None, None, :]
    range_response = np.sinc(range_cell[..., None] - range_axis)
    field_by_scatterer = (
        coefficients[None, None, :, None]
        * beam_gain[None, :, :, None]
        * phase[..., None]
        * range_response
    )
    iq = np.sum(field_by_scatterer, axis=2).astype(np.complex64)

    return ScattererEchoResponse(
        iq=iq,
        path_length_m=path_length,
        effective_range_m=effective_range,
        range_cell=range_cell,
        beam_gain=beam_gain,
        scatterer_positions_xyz_m=positions.copy(),
        reflectivity=coefficients.copy(),
    )


def _sample_parameter(
    value: float | tuple[float, float] | np.ndarray,
    count: int,
    rng: np.random.Generator,
    name: str,
) -> np.ndarray:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} interval must contain two endpoints")
        low, high = (float(item) for item in value)
        if not np.isfinite(low) or not np.isfinite(high) or low > high:
            raise ValueError(f"{name} interval endpoints must be finite and ordered")
        sampled = rng.uniform(low, high, size=count)
        return np.minimum(sampled, np.nextafter(high, low))
    if np.isscalar(value):
        scalar = float(value)
        if not np.isfinite(scalar):
            raise ValueError(f"{name} must be finite")
        return np.full(count, scalar, dtype=np.float64)
    sampled = np.asarray(value, dtype=np.float64)
    if sampled.shape != (count,) or np.any(~np.isfinite(sampled)):
        raise ValueError(f"{name} must be finite and have shape ({count},)")
    return sampled


def _optional_parameter(
    value: float | np.ndarray | None,
    fallback: np.ndarray,
    count: int,
    name: str,
) -> np.ndarray:
    if value is None:
        return fallback.copy()
    if np.isscalar(value):
        result = np.full(count, float(value), dtype=np.float64)
    else:
        result = np.asarray(value, dtype=np.float64)
    if result.shape != (count,) or np.any(~np.isfinite(result)):
        raise ValueError(f"{name} must be finite and have shape ({count},)")
    return result


def _ground_position_from_slant_range(
    slant_range_m: float,
    azimuth_deg: float,
    platform_position_xyz_m: np.ndarray,
) -> np.ndarray:
    altitude_m = float(platform_position_xyz_m[2])
    if slant_range_m <= abs(altitude_m):
        raise ValueError("slant range must exceed absolute platform altitude")
    ground_range_m = np.sqrt(slant_range_m**2 - altitude_m**2)
    azimuth_rad = np.deg2rad(azimuth_deg)
    return np.asarray(
        [
            platform_position_xyz_m[0] + ground_range_m * np.cos(azimuth_rad),
            platform_position_xyz_m[1] + ground_range_m * np.sin(azimuth_rad),
            0.0,
        ],
        dtype=np.float64,
    )


def _azimuth_from_position(position: np.ndarray, platform: np.ndarray) -> float:
    delta = position - platform
    return float(np.rad2deg(np.arctan2(delta[1], delta[0])))


def compute_scene_power_metrics(
    target_iq: np.ndarray,
    clutter_iq: np.ndarray,
    noise_iq: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Return component-power SNR/CNR/SCNR over the full range-compressed CPI.

    Inputs may be one ``(channel, pulse, range)`` CPI or a batch with a leading
    scene axis. Component powers are means over every IQ sample in the same
    domain, so only two independent ratios exist:
    ``SNR=Ps/Pn``, ``CNR=Pc/Pn``, and ``SCNR=Ps/(Pc+Pn)``.
    """
    target = np.asarray(target_iq)
    clutter = np.asarray(clutter_iq)
    noise = np.asarray(noise_iq)
    if target.shape != clutter.shape or target.shape != noise.shape:
        raise ValueError("target, clutter, and noise arrays must have identical shapes")
    if target.ndim not in (3, 4) or any(size == 0 for size in target.shape):
        raise ValueError("components must have shape (C,N,R) or (B,C,N,R)")
    if any(np.any(~np.isfinite(values)) for values in (target, clutter, noise)):
        raise ValueError("component IQ arrays must be finite")
    axes = (1, 2, 3) if target.ndim == 4 else (0, 1, 2)
    ps = np.mean(np.abs(target.astype(np.complex128)) ** 2, axis=axes)
    pc = np.mean(np.abs(clutter.astype(np.complex128)) ** 2, axis=axes)
    pn = np.mean(np.abs(noise.astype(np.complex128)) ** 2, axis=axes)
    snr = np.divide(ps, pn, out=np.full_like(ps, np.inf, dtype=np.float64), where=pn > 0.0)
    cnr = np.divide(pc, pn, out=np.full_like(pc, np.inf, dtype=np.float64), where=pn > 0.0)
    scnr = np.divide(
        ps,
        pc + pn,
        out=np.zeros_like(ps, dtype=np.float64),
        where=(pc + pn) > 0.0,
    )

    def to_db(value: np.ndarray | np.float64) -> np.ndarray | float:
        value_array = np.asarray(value, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            result = 10.0 * np.log10(value_array)
        return float(result) if result.ndim == 0 else result

    scalar = target.ndim == 3
    values: dict[str, np.ndarray | float] = {
        "target_power": ps,
        "clutter_power": pc,
        "noise_power": pn,
        "snr_linear": snr,
        "cnr_linear": cnr,
        "scnr_linear": scnr,
        "snr_db": to_db(snr),
        "cnr_db": to_db(cnr),
        "scnr_db": to_db(scnr),
    }
    if scalar:
        return {name: float(np.asarray(value)) for name, value in values.items()}
    return values


def generate_d2a_paired_batch(
    count: int,
    seed: int,
    snr_db: float | tuple[float, float] | np.ndarray,
    cnr_db: float | tuple[float, float] | np.ndarray,
    config: dict[str, Any],
    *,
    scenario_id: str = "separated",
    target_range_cell: float | np.ndarray | None = None,
    target_azimuth_deg: float | np.ndarray | None = None,
    target_radial_velocity_mps: float | np.ndarray | None = None,
) -> D2APairedBatch:
    """Generate matched H0/H1 counterfactual scenes with common clutter/noise."""
    if count <= 0:
        raise ValueError("count must be positive")
    scenario_map = config.get("identifiability_scenarios", {})
    valid_scenarios = {entry["scenario_id"] for entry in scenario_map.values()}
    if scenario_id not in valid_scenarios:
        raise ValueError(f"unknown D2-A scenario_id: {scenario_id}")

    seed_sequence = np.random.SeedSequence(int(seed))
    background_seed, noise_seed, target_seed = seed_sequence.spawn(3)
    background_rng = np.random.default_rng(background_seed)
    noise_rng = np.random.default_rng(noise_seed)
    target_rng = np.random.default_rng(target_seed)

    radar = config["radar"]
    waveform = config["waveform"]
    range_grid = config["range_grid"]
    platform_config = config["platform"]
    clutter_config = config["clutter"]
    target_config = config["target"]
    beam_config = config["beam"]
    pulse_times = np.arange(int(radar["pulse_count"]), dtype=np.float64) / float(radar["prf_hz"])
    channel_count = int(radar["channel_count"])
    range_count = int(radar["range_bin_count"])
    scatterer_count = int(clutter_config["scatterer_count"])
    if scatterer_count <= 0:
        raise ValueError("clutter.scatterer_count must be positive")

    snr_samples = _sample_parameter(snr_db, count, target_rng, "snr_db")
    cnr_samples = _sample_parameter(cnr_db, count, target_rng, "cnr_db")
    range_low, range_high = (float(v) for v in target_config["range_cell_interval"])
    azimuth_low, azimuth_high = (float(v) for v in target_config["azimuth_interval_deg"])
    target_range_defaults = target_rng.uniform(range_low, range_high, size=count)
    target_azimuth_defaults = target_rng.uniform(azimuth_low, azimuth_high, size=count)
    target_range_samples = _optional_parameter(
        target_range_cell, target_range_defaults, count, "target_range_cell"
    )
    target_azimuth_samples = _optional_parameter(
        target_azimuth_deg, target_azimuth_defaults, count, "target_azimuth_deg"
    )
    target_velocity_db = target_config["scenario_radial_velocity_mps"][scenario_id]
    target_velocity_defaults = np.full(count, float(target_velocity_db), dtype=np.float64)
    target_velocity_samples = _optional_parameter(
        target_radial_velocity_mps,
        target_velocity_defaults,
        count,
        "target_radial_velocity_mps",
    )
    truth_target_velocity = target_velocity_samples.copy()
    phase_low, phase_high = (float(v) for v in target_config["initial_phase_interval_rad"])
    phase_samples = target_rng.uniform(phase_low, phase_high, size=count)
    amplitude_low, amplitude_high = (
        float(v) for v in target_config["reference_amplitude_db_interval"]
    )
    amplitude_db = target_rng.uniform(amplitude_low, amplitude_high, size=count)
    amplitude_samples = np.power(10.0, amplitude_db / 20.0)

    platform = np.asarray(platform_config["initial_position_xyz_m"], dtype=np.float64)
    platform_velocity = np.asarray(platform_config["velocity_xyz_mps"], dtype=np.float64)
    beam_center = float(beam_config["center_azimuth_deg"])
    scatterer_az_low, scatterer_az_high = (
        float(v) for v in clutter_config["azimuth_interval_deg"]
    )
    scatterer_range_low, scatterer_range_high = (
        float(v) for v in clutter_config["range_cell_interval"]
    )
    range_start = float(range_grid["axis_start_m"])
    range_cell_m = float(waveform["range_cell_m"])
    truth_target_azimuth = target_azimuth_samples.copy()

    shape = (count, channel_count, int(radar["pulse_count"]), range_count)
    clutter_iq = np.empty(shape, dtype=np.complex64)
    noise_iq = np.empty(shape, dtype=np.complex64)
    target_signal_iq = np.empty(shape, dtype=np.complex64)
    scatterer_positions = np.empty((count, scatterer_count, 3), dtype=np.float64)
    scatterer_reflectivity = np.empty((count, scatterer_count), dtype=np.complex128)
    truth_target_range = target_range_samples.copy()
    truth_target_doppler = np.empty(count, dtype=np.float64)

    for index in range(count):
        scatterer_cells = background_rng.uniform(
            scatterer_range_low, scatterer_range_high, size=scatterer_count
        )
        scatterer_azimuths = background_rng.uniform(
            scatterer_az_low, scatterer_az_high, size=scatterer_count
        )
        slant_ranges = range_start + scatterer_cells * range_cell_m
        ground_ranges = np.sqrt(np.maximum(slant_ranges**2 - platform[2] ** 2, 0.0))
        azimuth_radians = np.deg2rad(scatterer_azimuths)
        positions = np.column_stack(
            (
                platform[0] + ground_ranges * np.cos(azimuth_radians),
                platform[1] + ground_ranges * np.sin(azimuth_radians),
                np.zeros(scatterer_count),
            )
        )
        reflectivity = (
            background_rng.standard_normal(scatterer_count)
            + 1j * background_rng.standard_normal(scatterer_count)
        ) / np.sqrt(2.0)
        clutter_response = synthesize_scatterer_echoes(
            positions,
            reflectivity,
            pulse_times,
            config,
            platform_position_xyz_m=platform,
            platform_velocity_xyz_mps=platform_velocity,
            beam_center_azimuth_deg=beam_center,
        )

        noise_power = amplitude_samples[index] ** 2 * 10.0 ** (-snr_samples[index] / 10.0)
        desired_clutter_power = noise_power * 10.0 ** (cnr_samples[index] / 10.0)
        raw_clutter_power = float(np.mean(np.abs(clutter_response.iq.astype(np.complex128)) ** 2))
        if not np.isfinite(raw_clutter_power) or raw_clutter_power <= 0.0:
            raise RuntimeError("random clutter scene has zero or invalid power")
        clutter_iq[index] = clutter_response.iq * np.sqrt(desired_clutter_power / raw_clutter_power)
        noise_iq[index] = generate_complex_awgn(
            noise_rng,
            (channel_count, int(radar["pulse_count"]), range_count),
            noise_power,
        )

        if scenario_id == "exact_overlap":
            target_position = positions[0].copy()
            target_azimuth = _azimuth_from_position(target_position, platform)
            target_velocity = np.zeros(3, dtype=np.float64)
            truth_target_azimuth[index] = target_azimuth
            truth_target_velocity[index] = 0.0
        else:
            target_azimuth = float(target_azimuth_samples[index])
            desired_slant_range = range_start + float(target_range_samples[index]) * range_cell_m
            target_position = _ground_position_from_slant_range(
                desired_slant_range, target_azimuth, platform
            )
            horizontal_unit = np.asarray(
                [np.cos(np.deg2rad(target_azimuth)), np.sin(np.deg2rad(target_azimuth)), 0.0]
            )
            horizontal_range = np.linalg.norm(target_position[:2] - platform[:2])
            cos_elevation = horizontal_range / desired_slant_range
            target_velocity = (
                float(target_velocity_samples[index]) / cos_elevation * horizontal_unit
            )

        target_response = synthesize_scatterer_echoes(
            target_position[None, :],
            np.asarray(
                [amplitude_samples[index] * np.exp(1j * phase_samples[index])],
                dtype=np.complex128,
            ),
            pulse_times,
            config,
            platform_position_xyz_m=platform,
            platform_velocity_xyz_mps=platform_velocity,
            scatterer_velocity_xyz_mps=target_velocity[None, :],
            beam_center_azimuth_deg=beam_center,
        )
        target_signal_iq[index] = target_response.iq
        truth_target_range[index] = float(np.mean(target_response.range_cell[:, 0, 0]))
        truth_target_azimuth[index] = _azimuth_from_position(target_position, platform)
        mean_path = np.mean(target_response.path_length_m[:, :, 0], axis=0)
        instantaneous_doppler = -np.diff(mean_path) * float(radar["prf_hz"]) / (
            float(waveform["propagation_speed_mps"]) / float(waveform["carrier_hz"])
        )
        raw_doppler = float(np.mean(instantaneous_doppler))
        prf_hz = float(radar["prf_hz"])
        truth_target_doppler[index] = (raw_doppler + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
        scatterer_positions[index] = positions
        scatterer_reflectivity[index] = reflectivity

    iq_h0 = np.asarray(clutter_iq + noise_iq, dtype=np.complex64)
    iq_h1 = np.asarray(iq_h0 + target_signal_iq, dtype=np.complex64)
    scene_indices = np.arange(count, dtype=np.int64)
    base_ids = np.asarray([f"d2a-scene-{seed}-{i}" for i in scene_indices])
    background_ids = np.asarray([f"d2a-background-{seed}-{i}" for i in scene_indices])
    target_ids = np.asarray([f"d2a-target-{seed}-{i}" for i in scene_indices])
    realization_ids = np.asarray([f"d2a-realization-{seed}-{i}" for i in scene_indices])

    return D2APairedBatch(
        iq_h0=iq_h0,
        iq_h1=iq_h1,
        clutter_iq=clutter_iq,
        noise_iq=noise_iq,
        target_signal_iq=target_signal_iq,
        truth_presence_h0=np.zeros(count, dtype=np.bool_),
        truth_presence_h1=np.ones(count, dtype=np.bool_),
        counterfactual_target_range_cell=truth_target_range.copy(),
        truth_target_range_cell=truth_target_range.copy(),
        truth_target_doppler_hz=truth_target_doppler.astype(np.float64),
        truth_target_radial_velocity_mps=truth_target_velocity.astype(np.float64),
        truth_target_azimuth_deg=truth_target_azimuth.astype(np.float64),
        truth_target_phase_rad=phase_samples.astype(np.float64),
        truth_target_amplitude=amplitude_samples.astype(np.float64),
        noise_power=(amplitude_samples**2 * 10.0 ** (-snr_samples / 10.0)).astype(np.float64),
        snr_db=snr_samples.astype(np.float64),
        requested_cnr_db=cnr_samples.astype(np.float64),
        base_scene_id=base_ids,
        background_id=background_ids,
        target_id=target_ids,
        realization_id=realization_ids,
        scatterer_positions_xyz_m=scatterer_positions,
        scatterer_reflectivity=scatterer_reflectivity,
    )


def generate_d0_compatible_no_clutter(
    count: int,
    seed: int,
    snr_db: float | tuple[float, float] | np.ndarray,
    presence_ratio: float,
    d0_config: dict[str, Any],
) -> D0Batch:
    """Preserve the frozen D0 zero-clutter path without modifying its generator."""
    return generate_d0_batch(count, seed, snr_db, presence_ratio, d0_config)
