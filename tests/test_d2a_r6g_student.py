from __future__ import annotations

import math

import pytest
import torch

from radardistill.students import r6g_student
from radardistill.students.r6_architectures import R6ECEStudent


def _student():
    student_type = getattr(r6g_student, "R6GCEStudent", None)
    assert student_type is not None, "R6GCEStudent must preserve CE-TCN task outputs and add a training response head"
    return student_type(width=64)


def _phase_rotate(iq_ri: torch.Tensor, phase: float) -> torch.Tensor:
    iq = torch.complex(iq_ri[:, :3], iq_ri[:, 3:])
    rotated = iq * torch.exp(torch.tensor(1j * phase, device=iq.device))
    return torch.cat((rotated.real, rotated.imag), dim=1)


def test_task_outputs_match_r6e_ce_tcn_when_shared_weights_are_loaded() -> None:
    torch.manual_seed(5)
    reference = R6ECEStudent(width=64)
    student = _student()
    student.load_state_dict(reference.state_dict(), strict=False)
    student.eval()
    reference.eval()
    iq = torch.randn(2, 6, 130, 32)
    with torch.inference_mode():
        expected = reference(iq)
        actual = student(iq, compute_spectrum=False)
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0.0, atol=0.0)


def test_response_head_has_finite_output_and_nonzero_gradient() -> None:
    student = _student()
    iq = torch.randn(2, 6, 130, 32)
    output = student(iq)
    assert output["spectrum_logits"].shape == (2, 520, 32)
    assert torch.isfinite(output["spectrum_logits"]).all()
    output["spectrum_logits"].square().mean().backward()
    gradient = student.training_response_head.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.linalg.vector_norm(gradient).item() > 0.0


def test_task_outputs_are_public_phase_invariant() -> None:
    torch.manual_seed(8)
    student = _student().eval()
    iq = torch.randn(2, 6, 130, 32)
    with torch.inference_mode():
        original = student(iq)
        rotated = student(_phase_rotate(iq, 0.73))
    for key in ("presence_logit", "range_logits", "range_offset", "doppler_vector", "spectrum_logits"):
        torch.testing.assert_close(original[key], rotated[key], rtol=1e-5, atol=1e-6)


def test_export_omits_training_only_response_head() -> None:
    student = _student()
    export = getattr(r6g_student, "export_r6g_inference_model", None)
    assert callable(export), "export_r6g_inference_model must remove the training-only head"
    inference_model = export(student)
    assert isinstance(inference_model, R6ECEStudent)
    assert not hasattr(inference_model, "training_response_head")
    with torch.inference_mode():
        output = inference_model(torch.randn(1, 6, 130, 32))
    assert set(output) == {"presence_logit", "range_logits", "range_offset", "doppler_vector"}


@pytest.mark.parametrize("frequency_bins", [0, -1, 1.5, True])
def test_rejects_invalid_response_frequency_bin_count(frequency_bins: object) -> None:
    student_type = getattr(r6g_student, "R6GCEStudent", None)
    assert student_type is not None
    with pytest.raises(ValueError):
        student_type(width=64, spectrum_frequency_bins=frequency_bins)


@pytest.mark.parametrize("phase", [-math.pi, -1.1, 0.0, 2.7, math.pi])
def test_exported_task_head_is_stable_at_phase_boundaries(phase: float) -> None:
    student = _student()
    export = getattr(r6g_student, "export_r6g_inference_model", None)
    assert callable(export), "export_r6g_inference_model must remove the training-only head"
    inference_model = export(student).eval()
    iq = torch.randn(1, 6, 130, 32)
    with torch.inference_mode():
        first = inference_model(iq)
        second = inference_model(_phase_rotate(iq, phase))
    torch.testing.assert_close(first["doppler_vector"], second["doppler_vector"], rtol=1e-5, atol=1e-6)
