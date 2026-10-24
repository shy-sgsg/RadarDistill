import numpy as np
import pytest
import yaml

from radardistill.simulation.d0 import generate_d0_batch
from radardistill.teachers.apx15_doppler import APX15DopplerTeacher


@pytest.fixture(scope="module")
def d0_config():
    with open("experiments/D0-E001/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def teacher(d0_config, tmp_path_factory):
    return APX15DopplerTeacher(
        d0_config,
        build_dir=tmp_path_factory.mktemp("apx15-doppler-build"),
    )


def test_cpp_adapter_matches_fftw_centered_numpy_transform(teacher, d0_config):
    rng = np.random.default_rng(14001)
    iq = (
        rng.standard_normal((3, 3, 130, 32))
        + 1j * rng.standard_normal((3, 3, 130, 32))
    ).astype(np.complex64)

    actual = teacher.transform(iq, nfft=130)
    expected = np.fft.fftshift(
        np.fft.fft(iq.astype(np.complex128), n=130, axis=2), axes=2
    )

    assert actual.shape == (3, 3, 130, 32)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-9)


def test_apx15_native_frequency_axis_has_correct_sign_and_centering(teacher, d0_config):
    prf = d0_config["radar"]["prf_hz"]
    n = d0_config["radar"]["pulse_count"]
    frequencies = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / prf))
    times = np.arange(n, dtype=np.float64) / prf
    iq = np.exp(2j * np.pi * frequencies[:, None] * times[None, :])
    iq = np.broadcast_to(iq[:, None, :, None], (n, 3, n, 32)).copy().astype(np.complex64)

    spectrum = teacher.transform(iq, nfft=n)
    peak_bins = np.argmax(np.abs(spectrum[:, 0, :, 0]) ** 2, axis=1)

    np.testing.assert_array_equal(peak_bins, np.arange(n))


def test_teacher_response_is_full_map_normalized_and_has_separate_fields(teacher, d0_config):
    batch = generate_d0_batch(12, 14001, 0.0, 0.5, d0_config)
    prediction = teacher.predict(batch.iq)

    assert prediction.teacher_response.shape == (12, 130, 32)
    assert prediction.teacher_detection_score.shape == (12,)
    assert prediction.teacher_range_cell.shape == (12,)
    assert prediction.teacher_doppler_hz.shape == (12,)
    assert prediction.teacher_confidence.shape == (12,)
    assert np.all(np.isfinite(prediction.teacher_response))
    assert np.all(prediction.teacher_response >= 0.0)
    np.testing.assert_allclose(prediction.teacher_response.sum(axis=(1, 2)), 1.0, atol=1e-6)
    assert np.all(np.isfinite(prediction.teacher_detection_score))
    assert np.all(np.isfinite(prediction.teacher_range_cell))
    assert np.all(np.isfinite(prediction.teacher_doppler_hz))
    assert np.all(np.isfinite(prediction.teacher_confidence))
    assert prediction.teacher_response.dtype == np.float32


def test_noiseless_teacher_recovers_continuous_range_and_wrapped_doppler(teacher, d0_config):
    batch = generate_d0_batch(512, 14002, np.inf, 1.0, d0_config)
    prediction = teacher.predict(batch.iq)
    range_error = prediction.teacher_range_cell - batch.truth_range_cell
    doppler_error = (
        prediction.teacher_doppler_hz - batch.truth_doppler_hz + 650.0
    ) % 1300.0 - 650.0

    assert np.sqrt(np.mean(range_error**2)) <= 1e-6
    assert np.max(np.abs(range_error)) <= 1e-5
    assert np.sqrt(np.mean(doppler_error**2)) <= 0.035
    assert np.max(np.abs(doppler_error)) <= 0.06


def test_h0_teacher_estimates_are_retained_as_false_alarm_candidates(teacher, d0_config):
    h0 = generate_d0_batch(24, 14003, -5.0, 0.0, d0_config)
    prediction = teacher.predict(h0.iq)

    assert np.all(np.isfinite(prediction.teacher_detection_score))
    assert np.all(np.isfinite(prediction.teacher_range_cell))
    assert np.all(np.isfinite(prediction.teacher_doppler_hz))
    assert np.all(np.isfinite(prediction.teacher_confidence))
    assert np.all(np.isnan(h0.truth_range_cell))
    assert np.all(np.isnan(h0.truth_doppler_hz))


def test_doppler_estimator_handles_both_nyquist_sides(teacher, d0_config):
    frequencies = np.asarray([-650.0, -649.99, 0.0, 649.99])
    times = np.arange(130, dtype=np.float64) / 1300.0
    iq = np.exp(2j * np.pi * frequencies[:, None] * times[None, :])
    iq = np.broadcast_to(iq[:, None, :, None], (4, 3, 130, 32)).copy().astype(np.complex64)

    # A direct tone is enough to exercise the fine-FFT and circular interpolation path.
    estimate = teacher.predict(iq).teacher_doppler_hz
    error = (estimate - frequencies + 650.0) % 1300.0 - 650.0
    assert np.max(np.abs(error)) <= 0.06
