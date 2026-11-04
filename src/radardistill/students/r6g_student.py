"""Training-time response-head student for D2-A-R6-G."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .r6_architectures import R6ECEStudent


class R6GCEStudent(R6ECEStudent):
    """R6-E CE-TCN task model with a training-only RD response head."""

    def __init__(self, *, width: int = 64, spectrum_frequency_bins: int = 520) -> None:
        super().__init__(width=width)
        if (
            isinstance(spectrum_frequency_bins, bool)
            or not isinstance(spectrum_frequency_bins, int)
            or spectrum_frequency_bins <= 0
        ):
            raise ValueError("spectrum_frequency_bins must be a positive integer")
        self.spectrum_frequency_bins = int(spectrum_frequency_bins)
        self.training_response_head = nn.Conv1d(
            self.width, self.spectrum_frequency_bins, kernel_size=1
        )

    def forward(
        self, iq_ri: torch.Tensor, *, compute_spectrum: bool = True
    ) -> dict[str, torch.Tensor]:
        invariant = self.forward_features(iq_ri).abs().mean(dim=2)
        presence_by_range = self.presence_per_range(invariant).squeeze(1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - math.log(
            invariant.shape[-1]
        )
        range_logits = self.range_head(invariant).squeeze(1)
        range_offset = torch.sigmoid(self.range_offset_head(invariant).squeeze(1))
        doppler_vector = self.doppler_head(invariant).transpose(1, 2)
        doppler_vector = F.normalize(doppler_vector, p=2.0, dim=-1, eps=1e-6)
        outputs = {
            "presence_logit": presence_logit,
            "range_logits": range_logits,
            "range_offset": range_offset,
            "doppler_vector": doppler_vector,
        }
        if compute_spectrum:
            outputs["spectrum_logits"] = self.training_response_head(invariant)
        return outputs


def export_r6g_inference_model(training_model: R6GCEStudent) -> R6ECEStudent:
    """Export only task parameters; omit the response head used for training."""
    if not isinstance(training_model, R6GCEStudent):
        raise TypeError("training_model must be an R6GCEStudent")
    model = R6ECEStudent(width=training_model.width)
    task_state = {
        key: value.detach()
        for key, value in training_model.state_dict().items()
        if not key.startswith("training_response_head.")
    }
    model.load_state_dict(task_state, strict=True)
    return model.eval()
