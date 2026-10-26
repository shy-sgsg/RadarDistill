from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.d2a_baselines import steering_vector
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.teachers.d2a_t001 import (
    CovarianceBank,
    fit_post_doppler_covariance,
    post_doppler_ace_amf,
)


@pytest.fixture
def physics_config():
    with Path("experiments/D2-A-E001/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


@pytest.fixture
def teacher_config():
    with Path("experiments/D2-A-T001/config.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)["teacher"]


def identity_bank(pulse_count: int = 130, oversampling_factor: int = 4) -> CovarianceBank:
    bins = pulse_count * oversampling_factor
    matrices = np.broadcast_to(np.eye(3, dtype=np.complex128), (bins, 3, 3)).copy()
    return CovarianceBank(
        covariance=matrices,
        inverse=matrices.copy(),
        eigenvalues=np.ones((bins, 3), dtype=np.float64),
        condition_number=np.ones(bins, dtype=np.float64),
        eigenvalue_floor_fraction=np.zeros(bins, dtype=np.float64),
        scene_count=0,
        pulse_count=pulse_count,
        channel_count=3,
        oversampling_factor=oversampling_factor,
    )


def test_covariance_fit_is_hermitian_positive_definite_and_records_each_bin():
    rng = np.random.default_rng(22001)
    batches = [
        (rng.normal(size=(2, 3, 130, 32)) + 1j * rng.normal(size=(2, 3, 130, 32))).astype(
            np.complex64
        )
        for _ in range(2)
    ]
    bank = fit_post_doppler_covariance(
        batches,
        oversampling_factor=4,
        eigenvalue_floor_relative_trace=1e-8,
    )
    assert bank.covariance.shape == bank.inverse.shape == (520, 3, 3)
    assert bank.eigenvalues.shape == (520, 3)
    assert bank.condition_number.shape == bank.eigenvalue_floor_fraction.shape == (520,)
    assert bank.scene_count == 4
    assert np.all(np.isfinite(bank.covariance))
    assert np.all(np.isfinite(bank.inverse))
    np.testing.assert_allclose(bank.covariance, bank.covariance.conj().transpose(0, 2, 1), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(bank.covariance) > 0.0)
    np.testing.assert_allclose(
        np.einsum("kij,kjl->kil", bank.covariance, bank.inverse),
        np.broadcast_to(np.eye(3), (520, 3, 3)),
        rtol=1e-7,
        atol=1e-7,
    )


def test_known_complex_tone_maps_to_its_doppler_range_and_steering(physics_config, teacher_config):
    prf_hz = float(physics_config["radar"]["prf_hz"])
    pulse_count = int(physics_config["radar"]["pulse_count"])
    target_range = 12
    target_doppler = 123.4
    target_angle = 90.1425
    steering = steering_vector(
        target_angle,
        float(physics_config["beam"]["center_elevation_deg"]),
        physics_config,
    )
    time = np.arange(pulse_count, dtype=np.float64) / prf_hz
    iq = np.zeros((3, pulse_count, 32), dtype=np.complex128)
    iq[:, :, target_range] = 3.0 * steering[:, None] * np.exp(
        2j * np.pi * target_doppler * time
    )[None, :]
    output = post_doppler_ace_amf(
        iq,
        identity_bank(),
        physics_config,
        teacher_config,
    )
    assert output.range_cell == pytest.approx(target_range, abs=0.02)
    assert output.doppler_hz == pytest.approx(target_doppler, abs=1.5)
    assert output.angle_deg == pytest.approx(target_angle, abs=1e-9)
    assert np.max(output.spatial_coherence_map[:, target_range]) > 0.999999
    assert output.score_map.shape == (130 * 4, 32)
    assert output.soft_response.shape == (130, 32)
    assert output.soft_response.sum() == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("scale", [0.5 + 0j, 2.0 + 0j, 0.6 + 1.2j])
def test_detection_score_and_soft_map_are_invariant_to_common_complex_scale(
    physics_config, teacher_config, scale
):
    paired = generate_d2a_paired_batch(
        1, 22002, 10.0, 10.0, physics_config, scenario_id="partial_overlap"
    )
    iq = paired.iq_h1[0].astype(np.complex128)
    bank = identity_bank()
    original = post_doppler_ace_amf(iq, bank, physics_config, teacher_config)
    scaled = post_doppler_ace_amf(iq * scale, bank, physics_config, teacher_config)
    assert scaled.score == pytest.approx(original.score, rel=1e-10, abs=1e-12)
    np.testing.assert_allclose(scaled.score_map, original.score_map, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        scaled.spatial_coherence_map,
        original.spatial_coherence_map,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(scaled.soft_response, original.soft_response, rtol=1e-10, atol=1e-12)


def test_zero_iq_returns_finite_empty_maps(physics_config, teacher_config):
    output = post_doppler_ace_amf(
        np.zeros((3, 130, 32), dtype=np.complex64),
        identity_bank(),
        physics_config,
        teacher_config,
    )
    assert output.score == 0.0
    assert np.isfinite(output.doppler_hz)
    assert np.isfinite(output.range_cell)
    assert np.all(np.isfinite(output.score_map))
    assert np.all(np.isfinite(output.soft_response))
    assert np.count_nonzero(output.soft_response) == 0


def test_batch_and_scalar_output_shapes_agree(physics_config, teacher_config):
    paired = generate_d2a_paired_batch(2, 22003, 15.0, 0.0, physics_config)
    bank = identity_bank()
    batch = post_doppler_ace_amf(paired.iq_h1, bank, physics_config, teacher_config)
    scalar = post_doppler_ace_amf(paired.iq_h1[0], bank, physics_config, teacher_config)
    assert np.shape(batch.score) == (2,)
    assert batch.score_map.shape == (2, 520, 32)
    assert batch.soft_response.shape == (2, 130, 32)
    assert np.shape(scalar.score) == ()
    assert scalar.score_map.shape == (520, 32)
    assert scalar.soft_response.shape == (130, 32)
    assert batch.score[0] == pytest.approx(scalar.score)


def test_teacher_functions_do_not_accept_truth_or_scene_metadata():
    fit_signature = inspect.signature(fit_post_doppler_covariance)
    assert tuple(fit_signature.parameters) == (
        "h0_iq_batches",
        "oversampling_factor",
        "eigenvalue_floor_relative_trace",
    )
    assert fit_signature.parameters["oversampling_factor"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(inspect.signature(post_doppler_ace_amf).parameters) == (
        "iq",
        "covariance_bank",
        "physics_config",
        "teacher_config",
    )
