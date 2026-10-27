import pytest
import torch

from radardistill.distillation.task_losses import decode_task_outputs, task_loss
from radardistill.students.temporal_frequency_attention import (
    TemporalFrequencyAttentionStudent,
)


def test_tfas_preserves_raw_ri_task_interface_and_preregistered_size():
    torch.manual_seed(23)
    model = TemporalFrequencyAttentionStudent().eval()
    inputs = torch.randn(2, 6, 130, 32)

    outputs = model(inputs, compute_spectrum=False)

    assert set(outputs) == {
        "presence_logit", "range_logits", "range_offset", "doppler_vector"
    }
    assert outputs["presence_logit"].shape == (2,)
    assert outputs["range_logits"].shape == (2, 32)
    assert outputs["range_offset"].shape == (2, 32)
    assert outputs["doppler_vector"].shape == (2, 32, 2)
    assert sum(parameter.numel() for parameter in model.parameters()) == 79_638
    assert torch.isfinite(torch.cat([
        outputs["presence_logit"],
        outputs["range_logits"].flatten(),
        outputs["range_offset"].flatten(),
        outputs["doppler_vector"].flatten(),
    ])).all()
    decoded = decode_task_outputs(outputs, prf_hz=1300.0)
    assert torch.all(decoded["doppler_hz"] >= -650.0)
    assert torch.all(decoded["doppler_hz"] < 650.0)


def test_tfas_truth_losses_backpropagate_into_every_task_head():
    torch.manual_seed(31)
    model = TemporalFrequencyAttentionStudent()
    outputs = model(torch.randn(2, 6, 130, 32), compute_spectrum=False)
    truth = {
        "truth_presence": torch.tensor([False, True]),
        "truth_range_cell": torch.tensor([float("nan"), 12.375]),
        "truth_doppler_hz": torch.tensor([float("nan"), 649.0]),
    }

    losses = task_loss(outputs, truth, {"radar": {"prf_hz": 1300.0}}, teacher_weight=0.0)
    losses["task_total"].backward()

    assert all(torch.isfinite(value) for value in losses.values())
    for name in ("presence_head", "range_head", "range_offset_head", "doppler_head"):
        module = getattr(model, name)
        gradients = [parameter.grad for parameter in module.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(int(torch.count_nonzero(gradient)) for gradient in gradients) > 0


@pytest.mark.parametrize(
    "shape",
    [(2, 3, 130, 32), (2, 6, 129, 32), (2, 6, 130, 31)],
)
def test_tfas_rejects_input_that_is_not_frozen_raw_ri_layout(shape):
    model = TemporalFrequencyAttentionStudent()
    with pytest.raises(ValueError, match=r"shape \[B,6,130,32\]"):
        model(torch.zeros(shape), compute_spectrum=False)
