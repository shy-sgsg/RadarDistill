import math

import torch

from radardistill.distillation.task_losses import (
    decode_task_outputs,
    doppler_circular_loss,
    encode_doppler_hz,
    task_loss,
)
from radardistill.students.task_student import TaskStudent, export_inference_model


def test_task_student_outputs_cover_every_task_and_full_slow_time_context():
    torch.manual_seed(17)
    model = TaskStudent()
    inputs = torch.randn(3, 6, 130, 32)

    outputs = model(inputs)

    assert outputs["presence_logit"].shape == (3,)
    assert outputs["range_logits"].shape == (3, 32)
    assert outputs["range_offset"].shape == (3, 32)
    assert outputs["doppler_vector"].shape == (3, 32, 2)
    assert outputs["spectrum_logits"].shape == (3, 130, 32)
    assert model.temporal_receptive_field == 255
    assert model.temporal_receptive_field >= inputs.shape[2]
    assert sum(parameter.numel() for parameter in model.parameters()) < 150_000


def test_h0_masks_range_and_doppler_task_gradients():
    model = TaskStudent()
    outputs = model(torch.randn(2, 6, 130, 32))
    outputs["range_logits"].retain_grad()
    outputs["range_offset"].retain_grad()
    outputs["doppler_vector"].retain_grad()
    truth = {
        "truth_presence": torch.zeros(2, dtype=torch.bool),
        "truth_range_cell": torch.full((2,), float("nan")),
        "truth_doppler_hz": torch.full((2,), float("nan")),
        "teacher_response": torch.full((2, 130, 32), 1.0 / (130 * 32)),
    }
    losses = task_loss(outputs, truth, {"radar": {"prf_hz": 1300.0}})

    losses["total"].backward()

    assert torch.isfinite(losses["total"])
    assert torch.count_nonzero(outputs["range_logits"].grad) == 0
    assert torch.count_nonzero(outputs["range_offset"].grad) == 0
    assert torch.count_nonzero(outputs["doppler_vector"].grad) == 0


def test_doppler_loss_and_decoder_wrap_at_nyquist_boundary():
    truth_hz = torch.tensor([649.0, -649.0])
    prediction_hz = torch.tensor([-649.0, 649.0])
    target_vector = encode_doppler_hz(truth_hz, 1300.0)
    prediction_vector = encode_doppler_hz(prediction_hz, 1300.0)

    loss = doppler_circular_loss(prediction_vector, target_vector)
    decoded = decode_task_outputs(
        {
            "presence_logit": torch.tensor([0.0, 0.0]),
            "range_logits": torch.tensor([[0.0] * 4 + [1.0] + [0.0] * 27] * 2),
            "range_offset": torch.zeros(2, 32),
            "doppler_vector": prediction_vector[:, None, :].expand(2, 32, 2),
        },
        prf_hz=1300.0,
    )

    assert torch.all(loss < 0.001)
    assert torch.allclose(decoded["doppler_hz"], prediction_hz, atol=1e-4)
    circular_error = (decoded["doppler_hz"] - truth_hz + 650.0) % 1300.0 - 650.0
    assert torch.allclose(circular_error.abs(), torch.full((2,), 2.0), atol=1e-4)
    assert math.isfinite(float(loss.mean()))


def test_mixed_h0_h1_losses_have_finite_gradients_and_use_soft_teacher_map():
    torch.manual_seed(29)
    model = TaskStudent()
    outputs = model(torch.randn(2, 6, 130, 32))
    truth = {
        "truth_presence": torch.tensor([False, True]),
        "truth_range_cell": torch.tensor([float("nan"), 12.375]),
        "truth_doppler_hz": torch.tensor([float("nan"), 649.0]),
        "teacher_response": torch.softmax(torch.randn(2, 130 * 32), dim=-1).reshape(2, 130, 32),
    }

    losses = task_loss(outputs, truth, {"radar": {"prf_hz": 1300.0}})
    losses["total"].backward()

    assert set(losses) >= {"presence", "range_cell", "range_offset", "doppler", "teacher_kl", "total"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_inference_export_removes_training_only_spectrum_head():
    torch.manual_seed(31)
    training_model = TaskStudent().eval()
    inference_model = export_inference_model(training_model)
    inputs = torch.randn(2, 6, 130, 32)
    outputs = inference_model(inputs)
    training_outputs = training_model(inputs)

    assert inference_model.spectrum_head is None
    assert "spectrum_logits" not in outputs
    assert set(outputs) == {"presence_logit", "range_logits", "range_offset", "doppler_vector"}
    torch.testing.assert_close(
        outputs["presence_logit"], training_outputs["presence_logit"],
        rtol=0,
        atol=0,
    )
