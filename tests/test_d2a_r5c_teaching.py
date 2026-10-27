import inspect

import pytest
import torch

from radardistill.distillation.r5c_teaching import (
    k2_local_targets,
    response_kl,
    truth_dirichlet_targets,
)
from radardistill.students.task_student import TaskStudent, export_inference_model


def test_default_task_student_keeps_legacy_130_bin_spectrum_shape():
    model = TaskStudent(width=32).eval()

    outputs = model(torch.zeros(1, 6, 130, 32))

    assert outputs["spectrum_logits"].shape == (1, 130, 32)


def test_task_student_can_emit_520_bin_training_response():
    model = TaskStudent(width=32, spectrum_frequency_bins=520).eval()

    outputs = model(torch.zeros(1, 6, 130, 32))

    assert outputs["spectrum_logits"].shape == (1, 520, 32)


def test_task_student_rejects_nonpositive_spectrum_frequency_bins():
    with pytest.raises(ValueError, match="spectrum_frequency_bins"):
        TaskStudent(width=32, spectrum_frequency_bins=0)


def test_520_bin_response_head_is_removed_from_inference_export():
    training_model = TaskStudent(width=32, spectrum_frequency_bins=520).eval()
    inference_model = export_inference_model(training_model).eval()
    inputs = torch.zeros(1, 6, 130, 32)

    outputs = inference_model(inputs)

    assert inference_model.spectrum_head is None
    assert "spectrum_logits" not in outputs
    torch.testing.assert_close(
        outputs["presence_logit"], training_model(inputs)["presence_logit"],
        rtol=0,
        atol=0,
    )


def test_identical_full_response_distribution_has_zero_kl():
    teacher = torch.rand(2, 520, 32)
    teacher = teacher / teacher.sum(dim=(1, 2), keepdim=True)
    logits = teacher.log()

    loss = response_kl(logits, "T1", teacher_map=teacher)

    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_full_map_kl_uses_one_normalized_distribution_per_cpi():
    teacher = torch.zeros(1, 520, 32)
    teacher[0, 4, 9] = 3.0
    teacher[0, 5, 9] = 1.0
    logits = torch.zeros_like(teacher)

    loss = response_kl(logits, "T1", teacher_map=teacher)
    expected = (
        0.75 * torch.log(torch.tensor(0.75))
        + 0.25 * torch.log(torch.tensor(0.25))
        + torch.log(torch.tensor(520.0 * 32.0))
    )

    assert loss.item() == pytest.approx(float(expected), abs=1e-6)


def test_k2_local_target_selects_range_from_iq_only_map_peak():
    teacher = torch.zeros(2, 520, 32)
    teacher[0, 42, 7] = 2.0
    teacher[0, 43, 7] = 1.0
    teacher[1, 200, 19] = 4.0
    teacher[1, 201, 19] = 2.0

    selected_range, local = k2_local_targets(teacher)

    assert selected_range.tolist() == [7, 19]
    torch.testing.assert_close(local.sum(dim=1), torch.ones(2))
    assert local[0, 42].item() == pytest.approx(2.0 / 3.0)
    assert local[0, 43].item() == pytest.approx(1.0 / 3.0)
    assert "truth_range_cell" not in inspect.signature(k2_local_targets).parameters
    assert "truth_doppler_hz" not in inspect.signature(k2_local_targets).parameters


def test_t2_local_kl_uses_teacher_selected_range():
    teacher = torch.zeros(1, 520, 32)
    teacher[0, 13, 6] = 1.0
    logits = torch.zeros_like(teacher)
    logits[0, 13, 6] = 16.0

    loss = response_kl(logits, "T2", teacher_map=teacher)

    assert 0.0 <= loss.item() < 0.1


def test_truth_dirichlet_target_is_normalized_and_periodic_at_nyquist():
    range_cells = torch.tensor([2.7, 31.2])
    doppler = torch.tensor([-649.5, 649.75])
    indices, target = truth_dirichlet_targets(range_cells, doppler)
    shifted_indices, shifted = truth_dirichlet_targets(
        range_cells, doppler + torch.tensor([1300.0, -1300.0])
    )

    assert indices.tolist() == [2, 31]
    assert shifted_indices.tolist() == [2, 31]
    torch.testing.assert_close(target.sum(dim=1), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(target, shifted, atol=1e-6, rtol=0)
    assert target.shape == (2, 520)


def test_t3_local_kl_uses_truth_range_and_normalized_dirichlet_response():
    range_cells = torch.tensor([11.6])
    doppler = torch.tensor([123.0])
    selected_range, target = truth_dirichlet_targets(range_cells, doppler)
    logits = torch.zeros(1, 520, 32)
    logits[0, :, 11] = target[0].clamp_min(1e-12).log()

    loss = response_kl(
        logits,
        "T3",
        truth_range_cell=range_cells,
        truth_doppler_hz=doppler,
    )

    assert selected_range.item() == 11
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_t0_is_zero_and_rejects_auxiliary_supervision_inputs():
    logits = torch.randn(2, 520, 32, requires_grad=True)
    loss = response_kl(logits, "T0")

    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() == 0
    with pytest.raises(ValueError, match="does not accept"):
        response_kl(logits.detach(), "T0", teacher_map=torch.ones_like(logits))


@pytest.mark.parametrize(
    "teacher,match",
    [
        (torch.ones(1, 520, 31), "shape"),
        (torch.full((1, 520, 32), -1.0), "nonnegative"),
        (torch.zeros(1, 520, 32), "positive total"),
    ],
)
def test_k2_local_targets_reject_invalid_maps(teacher, match):
    with pytest.raises(ValueError, match=match):
        k2_local_targets(teacher)
