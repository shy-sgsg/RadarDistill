"""Lightweight temporal-convolution student for D1 task outputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


class _TemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            width,
            width,
            kernel_size=(3, 1),
            padding=(dilation, 0),
            dilation=(dilation, 1),
            groups=width,
            bias=False,
        )
        self.pointwise = nn.Conv2d(width, width, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(8, width)
        self.activation = nn.SiLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.pointwise(self.depthwise(inputs))
        return inputs + self.activation(self.norm(residual))


class TaskStudent(nn.Module):
    """Consumes real/imag IQ planes and predicts presence, range, and Doppler.

    Input layout is [B,6,130,32]: real channels followed by imaginary channels.
    The auxiliary range-Doppler response is omitted from inference exports.
    """

    def __init__(
        self,
        *,
        width: int = 32,
        temporal_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        include_spectrum_head: bool = True,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 8:
            raise ValueError("width must be a positive multiple of 8 for GroupNorm(8)")
        dilations = tuple(int(value) for value in temporal_dilations)
        if not dilations or any(value <= 0 for value in dilations):
            raise ValueError("temporal dilations must be positive")
        self.width = int(width)
        self.temporal_dilations = dilations
        self.temporal_receptive_field = 1 + 2 * sum(dilations)
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.temporal_blocks = nn.Sequential(
            *(_TemporalResidualBlock(width, dilation) for dilation in dilations)
        )
        self.presence_per_range = nn.Conv1d(width, 1, kernel_size=1)
        self.range_head = nn.Conv1d(width, 1, kernel_size=1)
        self.range_offset_head = nn.Conv1d(width, 1, kernel_size=1)
        self.doppler_head = nn.Conv1d(width, 2, kernel_size=1)
        self.spectrum_head: nn.Module | None = (
            nn.Conv1d(width, 130, kernel_size=1) if include_spectrum_head else None
        )

    def forward(
        self, iq_ri: torch.Tensor, *, compute_spectrum: bool = True
    ) -> dict[str, torch.Tensor]:
        if iq_ri.ndim != 4 or iq_ri.shape[1] != 6 or iq_ri.shape[2] != 130 or iq_ri.shape[3] != 32:
            raise ValueError("iq_ri must have shape [B,6,130,32]")
        if not torch.is_floating_point(iq_ri) or not torch.isfinite(iq_ri).all():
            raise ValueError("iq_ri must contain finite floating-point real/imag planes")

        features = self.temporal_blocks(self.stem(iq_ri)).mean(dim=2)
        presence_by_range = self.presence_per_range(features).squeeze(1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - torch.log(
            torch.as_tensor(features.shape[-1], dtype=features.dtype, device=features.device)
        )
        range_logits = self.range_head(features).squeeze(1)
        range_offset = torch.sigmoid(self.range_offset_head(features).squeeze(1))
        doppler_vector = self.doppler_head(features).transpose(1, 2)
        doppler_vector = F.normalize(doppler_vector, p=2.0, dim=-1, eps=1e-6)

        outputs = {
            "presence_logit": presence_logit,
            "range_logits": range_logits,
            "range_offset": range_offset,
            "doppler_vector": doppler_vector,
        }
        if compute_spectrum and self.spectrum_head is not None:
            outputs["spectrum_logits"] = self.spectrum_head(features)
        return outputs


def export_inference_model(training_model: TaskStudent) -> TaskStudent:
    """Clone a trained model while omitting its training-only spectrum head."""
    if not isinstance(training_model, TaskStudent):
        raise TypeError("training_model must be a TaskStudent")
    inference_model = TaskStudent(
        width=training_model.width,
        temporal_dilations=training_model.temporal_dilations,
        include_spectrum_head=False,
    )
    state = {
        key: value.detach().cpu()
        for key, value in training_model.state_dict().items()
        if not key.startswith("spectrum_head.")
    }
    incompatibility = inference_model.load_state_dict(state, strict=False)
    if incompatibility.unexpected_keys or incompatibility.missing_keys:
        raise RuntimeError(f"inference export state mismatch: {incompatibility}")
    return inference_model
