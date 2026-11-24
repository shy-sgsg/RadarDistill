from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from radardistill.d2a_baselines import dpca_time_shift_s
from radardistill.distillation.task_losses import encode_doppler_hz
from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.students.r6_j4 import (
    R6J4Student,
    r6j4_dpca_fft,
    r6j4_loss,
    wrap_r6j4_doppler,
)


ROOT = Path(__file__).resolve().parents[1]
PRF = 1300.0
NFFT = 520
PHYSICS = yaml.safe_load((ROOT / "experiments/D2-A-E001/config.yaml").read_text())
DELAY_S = dpca_time_shift_s(PHYSICS)


def _random_iq(batch: int, *, seed: int = 7, device: str = "cpu") -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(batch, 6, 130, 32, generator=generator, dtype=torch.float32).to(device)


def _numpy_reference(iq_ri: torch.Tensor, *, delay_s: float) -> np.ndarray:
    iq = iq_ri.detach().cpu().numpy().astype(np.float64)
    complex_iq = iq[:, :3] + 1j * iq[:, 3:]
    shifted_second = np.fft.ifft(
        np.fft.fft(complex_iq[:, 1], axis=1)
        * np.exp(2j * np.pi * np.fft.fftfreq(130)[None, :, None] * (delay_s * PRF)),
        axis=1,
    )
    difference = complex_iq[:, 0] - shifted_second
    return np.fft.fftshift(np.fft.fft(difference, n=NFFT, axis=1), axes=1)


def test_fixed_frontend_matches_complex128_reference_on_cpu_and_cuda() -> None:
    iq = _random_iq(2)
    reference = _numpy_reference(iq, delay_s=DELAY_S)
    for device in ["cpu", *(["cuda:0"] if torch.cuda.is_available() else [])]:
        response = r6j4_dpca_fft(iq.to(device), dpca_delay_s=DELAY_S, prf_hz=PRF, nfft=NFFT)
        assert response.shape == (2, NFFT, 32)
        assert response.is_complex()
        np.testing.assert_allclose(
            response.detach().cpu().numpy(), reference, rtol=1e-4, atol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="R6-J4 G1 requires the registered CUDA device")
def test_registered_g1_cuda_scene_matches_complex128_reference() -> None:
    generated = generate_d2a_paired_batch(
        16,
        3_620_000_000,
        20.0,
        -5.0,
        PHYSICS,
        scenario_id="separated",
    )
    iq = np.asarray(generated.iq_h1[:4], dtype=np.complex64)
    iq_ri = np.concatenate((iq.real, iq.imag), axis=1).astype(np.float32)
    reference = _numpy_reference(torch.from_numpy(iq_ri), delay_s=DELAY_S)
    response = r6j4_dpca_fft(
        torch.from_numpy(iq_ri).cuda(), dpca_delay_s=DELAY_S, prf_hz=PRF, nfft=NFFT
    )
    np.testing.assert_allclose(response.cpu().numpy(), reference, rtol=1e-4, atol=1e-6)


def test_frontend_doppler_peak_follows_tone_and_wraps_circularly() -> None:
    pulse = torch.arange(130, dtype=torch.float32)
    iq = torch.zeros(1, 6, 130, 32)
    base_bin = 19
    base_hz = (base_bin - NFFT // 2) * (PRF / NFFT)
    iq[0, 0] = torch.cos(2 * torch.pi * base_hz * pulse / PRF)[:, None]
    iq[0, 3] = torch.sin(2 * torch.pi * base_hz * pulse / PRF)[:, None]
    first = r6j4_dpca_fft(iq, dpca_delay_s=0.0, prf_hz=PRF, nfft=NFFT)
    first_bin = int(first.abs().square()[0, :, 5].argmax())
    assert first_bin == base_bin

    delta_bins = 7
    delta_hz = delta_bins * PRF / NFFT
    rotation = torch.exp(2j * torch.pi * delta_hz * pulse / PRF)
    complex_iq = torch.complex(iq[:, :3], iq[:, 3:])
    shifted = complex_iq * rotation[None, None, :, None]
    shifted_ri = torch.cat((shifted.real, shifted.imag), dim=1)
    second = r6j4_dpca_fft(shifted_ri, dpca_delay_s=0.0, prf_hz=PRF, nfft=NFFT)
    second_bin = int(second.abs().square()[0, :, 5].argmax())
    assert (second_bin - first_bin) % NFFT == delta_bins
    assert wrap_r6j4_doppler(torch.tensor([PRF / 2]), PRF).item() == pytest.approx(-PRF / 2)


def test_student_forward_accepts_only_observable_iq_and_outputs_normalized_candidates() -> None:
    assert list(inspect.signature(R6J4Student.forward).parameters) == ["self", "iq_ri"]
    model = R6J4Student(dpca_delay_s=DELAY_S, width=8).eval()
    outputs = model(_random_iq(2))
    assert outputs["presence_logit"].shape == (2,)
    assert outputs["candidate_probability"].shape == (2, 1 + NFFT * 32)
    assert outputs["candidate_logits"].shape == (2, NFFT * 32)
    assert outputs["candidate_range"].shape == (2, NFFT, 32)
    assert outputs["candidate_doppler_hz"].shape == (2, NFFT, 32)
    assert outputs["range_cell_estimate"].shape == (2,)
    assert outputs["doppler_vector_pooled"].shape == (2, 2)
    assert torch.isfinite(outputs["candidate_probability"]).all()
    assert torch.all(outputs["candidate_probability"] >= 0)
    torch.testing.assert_close(
        outputs["candidate_probability"].sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        outputs["presence_logit"],
        torch.log(outputs["candidate_probability"][:, 1:].sum(-1))
        - torch.log(outputs["candidate_probability"][:, 0]),
        atol=1e-5,
        rtol=1e-5,
    )


def test_power_readout_is_common_phase_invariant_and_batch_permutation_equivariant() -> None:
    torch.manual_seed(31)
    model = R6J4Student(dpca_delay_s=DELAY_S, width=8).eval()
    iq = _random_iq(3, seed=32)
    phase = 1.17
    complex_iq = torch.complex(iq[:, :3], iq[:, 3:]) * torch.exp(torch.tensor(1j * phase))
    rotated = torch.cat((complex_iq.real, complex_iq.imag), dim=1)
    original = model(iq)
    transformed = model(rotated)
    for key in ("presence_logit", "candidate_probability", "range_cell_estimate", "doppler_vector_pooled"):
        torch.testing.assert_close(transformed[key], original[key], atol=2e-5, rtol=2e-5)
    permutation = torch.tensor([2, 0, 1])
    permuted = model(iq[permutation])
    for key in original:
        if torch.is_tensor(original[key]) and original[key].shape[:1] == (3,):
            torch.testing.assert_close(permuted[key], original[key][permutation], atol=1e-5, rtol=1e-5)


def _truth(presence: list[float]) -> dict[str, torch.Tensor]:
    return {
        "truth_presence": torch.tensor(presence, dtype=torch.float32),
        "truth_range_cell": torch.full((len(presence),), 11.25),
        "truth_doppler_hz": torch.full((len(presence),), 649.9),
    }


def test_r6j4_loss_masks_h0_parameters_and_keeps_finite_h1_gradients() -> None:
    torch.manual_seed(33)
    model = R6J4Student(dpca_delay_s=DELAY_S, width=8)
    outputs = model(_random_iq(4, seed=34))
    terms = r6j4_loss(outputs, _truth([0, 0, 1, 1]), prf_hz=PRF)
    assert terms["presence"].item() > 0
    assert terms["candidate"].item() > 0
    assert terms["range"].item() > 0
    assert terms["doppler"].item() >= 0
    assert all(torch.isfinite(value).all() for value in terms.values())
    terms["total"].backward()
    for module in (model.candidate_head, model.parameter_head, model.feedback_output):
        assert module.weight.grad is not None
        assert torch.isfinite(module.weight.grad).all()
        assert torch.linalg.vector_norm(module.weight.grad).item() > 0


def test_detection_and_parameter_losses_keep_the_registered_feedback_paths() -> None:
    torch.manual_seed(35)
    detection_model = R6J4Student(dpca_delay_s=DELAY_S, width=8)
    outputs = detection_model(_random_iq(2, seed=36))
    detection = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["presence_logit"], torch.ones(2)
    )
    detection.backward()
    assert detection_model.parameter_head.weight.grad is not None
    assert torch.linalg.vector_norm(detection_model.parameter_head.weight.grad).item() > 0

    parameter_model = R6J4Student(dpca_delay_s=DELAY_S, width=8)
    outputs = parameter_model(_random_iq(2, seed=37))
    parameter = torch.nn.functional.smooth_l1_loss(
        outputs["range_cell_estimate"], torch.full((2,), 11.25)
    )
    parameter.backward()
    assert parameter_model.candidate_head.weight.grad is not None
    assert torch.linalg.vector_norm(parameter_model.candidate_head.weight.grad).item() > 0
