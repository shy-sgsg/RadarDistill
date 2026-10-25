from __future__ import annotations

import torch

from radardistill.distillation.d2a_e002_losses import masked_k2_kl, d2a_e002_loss


def test_masked_k2_kl_backpropagates_only_for_h1_rows():
    logits = torch.randn(3, 130, 32, requires_grad=True)
    response = torch.rand(3, 130, 32)
    h1 = torch.tensor([False, True, True])
    loss = masked_k2_kl(logits, response, h1)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[0]) == 0
    assert torch.count_nonzero(logits.grad[1:]) > 0


def test_masked_k2_kl_is_unchanged_when_h0_teacher_response_changes():
    logits = torch.randn(2, 130, 32, requires_grad=True)
    response = torch.rand(2, 130, 32)
    h1 = torch.tensor([False, True])
    before = masked_k2_kl(logits, response, h1)
    response[0] = torch.rand_like(response[0]) * 1000.0
    after = masked_k2_kl(logits, response, h1)
    torch.testing.assert_close(before, after)


def test_task_truth_loss_remains_active_when_k2_weight_is_zero():
    outputs = {
        "presence_logit": torch.tensor([0.0, 0.0], requires_grad=True),
        "range_logits": torch.zeros(2, 32, requires_grad=True),
        "range_offset": torch.full((2, 32), 0.5, requires_grad=True),
        "doppler_vector": torch.nn.functional.normalize(torch.ones(2, 32, 2), dim=-1).requires_grad_(),
        "spectrum_logits": torch.zeros(2, 130, 32, requires_grad=True),
    }
    truth = {
        "truth_presence": torch.tensor([False, True]),
        "truth_range_cell": torch.tensor([0.0, 4.5]),
        "truth_doppler_hz": torch.tensor([0.0, 10.0]),
    }
    config = {"radar": {"prf_hz": 1300.0}, "training": {"loss": {}}}
    result = d2a_e002_loss(outputs, truth, config, teacher_response=None, teacher_weight=0.0)
    assert result["task_total"].item() > 0.0
    assert result["total"].item() == result["task_total"].item()
