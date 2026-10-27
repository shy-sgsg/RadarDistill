from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from radardistill.teachers.d2a_k2 import k2_response
from radardistill.teachers.d2a_k2_cuda import k2_response_cuda


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def physics_config() -> dict:
    return yaml.safe_load((ROOT / "experiments/D2-A-E001/config.yaml").read_text())


@pytest.fixture(scope="module")
def cuda_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA parity tests require the registered GPU environment")
    return "cuda:0"


def _random_iq(seed: int = 12, batch_size: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    real = rng.normal(size=(batch_size, 3, 130, 32)).astype(np.float32)
    imag = rng.normal(size=(batch_size, 3, 130, 32)).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def test_cuda_teacher_rejects_cpu_device(physics_config: dict) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        k2_response_cuda(_random_iq(batch_size=1), physics_config, device="cpu")


def test_cuda_teacher_rejects_malformed_and_nonfinite_iq(
    physics_config: dict, cuda_device: str
) -> None:
    with pytest.raises(ValueError, match="shape"):
        k2_response_cuda(np.zeros((1, 3, 130, 31), dtype=np.complex64), physics_config, device=cuda_device)

    invalid = _random_iq(batch_size=1)
    invalid[0, 0, 0, 0] = np.nan + 0j
    with pytest.raises(ValueError, match="finite"):
        k2_response_cuda(invalid, physics_config, device=cuda_device)


def test_cuda_native_response_matches_frozen_numpy_k2(
    physics_config: dict, cuda_device: str
) -> None:
    iq = _random_iq()
    expected = k2_response(iq, physics_config)
    actual = k2_response_cuda(iq, physics_config, device=cuda_device, oversampling_factor=1)

    assert actual.shape == (2, 130, 32)
    assert np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))) <= 1e-6


def test_cuda_response_is_finite_nonnegative_normalized_and_deterministic(
    physics_config: dict, cuda_device: str
) -> None:
    iq = _random_iq(seed=18, batch_size=1)
    first = k2_response_cuda(iq, physics_config, device=cuda_device, oversampling_factor=4)
    second = k2_response_cuda(iq, physics_config, device=cuda_device, oversampling_factor=4)

    assert first.shape == (1, 520, 32)
    assert np.all(np.isfinite(first))
    assert np.all(first >= 0.0)
    assert np.allclose(first.sum(axis=(1, 2)), 1.0, atol=1e-6, rtol=0.0)
    assert np.array_equal(first, second)


def test_cuda_normalized_response_is_invariant_to_global_complex_scale(
    physics_config: dict, cuda_device: str
) -> None:
    iq = _random_iq(seed=29, batch_size=1)
    reference = k2_response_cuda(iq, physics_config, device=cuda_device, oversampling_factor=4)

    for factor in (0.5 + 0j, 2.0 + 0j, 0.6 + 1.2j):
        scaled = k2_response_cuda(iq.astype(np.complex128) * factor, physics_config, device=cuda_device, oversampling_factor=4)
        assert np.max(np.abs(scaled.astype(np.float64) - reference.astype(np.float64))) <= 1e-6


def test_fourfold_zero_padding_resolves_off_bin_tone_within_one_grid_bin(
    physics_config: dict, cuda_device: str
) -> None:
    frequency_hz = 237.3
    pulse = np.arange(130, dtype=np.float64)
    iq = np.zeros((1, 3, 130, 32), dtype=np.complex128)
    iq[0, 0, :, 11] = np.exp(2j * np.pi * frequency_hz * pulse / 1300.0)

    response = k2_response_cuda(iq, physics_config, device=cuda_device, oversampling_factor=4)[0]
    frequency_axis = np.fft.fftshift(np.fft.fftfreq(520, d=1.0 / 1300.0))
    peak_doppler, peak_range = np.unravel_index(int(np.argmax(response)), response.shape)

    assert peak_range == 11
    assert abs(float(frequency_axis[peak_doppler]) - frequency_hz) <= 2.5
