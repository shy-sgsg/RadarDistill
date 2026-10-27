import pytest
import torch

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
