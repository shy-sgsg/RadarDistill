"""Factorized slow-time/range attention student for raw complex IQ planes."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def _sinusoidal_positions(length: int, width: int) -> torch.Tensor:
    if length <= 0 or width <= 0:
        raise ValueError("position dimensions must be positive")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros(length, width, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies)
    return encoding.unsqueeze(0)


def _encoder_layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=56,
        nhead=4,
        dim_feedforward=112,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class TemporalFrequencyAttentionStudent(nn.Module):
    """Use raw I/Q, slow-time self-attention and light range-axis interaction.

    Input is the TaskStudent real/imag layout ``[B,6,130,32]``. Frequency
    information remains in the coherent slow-time I/Q phase progression; the
    model does not form an FFT or use a fixed coherence-feature projection.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_embedding = nn.Linear(6, 56)
        self.register_buffer(
            "temporal_position_encoding",
            _sinusoidal_positions(130, 56),
            persistent=False,
        )
        self.temporal_layers = nn.ModuleList([_encoder_layer() for _ in range(2)])
        self.temporal_pool_score = nn.Linear(56, 1)
        self.range_position_embedding = nn.Parameter(torch.empty(1, 32, 56))
        nn.init.normal_(self.range_position_embedding, mean=0.0, std=0.02)
        self.range_layer = _encoder_layer()

        self.presence_head = nn.Linear(56, 1)
        self.range_head = nn.Linear(56, 1)
        self.range_offset_head = nn.Linear(56, 1)
        self.doppler_head = nn.Linear(56, 2)

    def forward(
        self,
        iq_ri: torch.Tensor,
        *,
        compute_spectrum: bool = False,
    ) -> dict[str, torch.Tensor]:
        del compute_spectrum  # The frozen E1 truth-only loss has no spectrum head.
        if (
            iq_ri.ndim != 4
            or iq_ri.shape[1] != 6
            or iq_ri.shape[2] != 130
            or iq_ri.shape[3] != 32
        ):
            raise ValueError("iq_ri must have shape [B,6,130,32]")
        if not torch.is_floating_point(iq_ri):
            raise ValueError("iq_ri must contain real/imag floating-point planes")

        batch_size = iq_ri.shape[0]
        # [B,6,T,R] -> [B*R,T,6]; real and imaginary planes stay explicit.
        tokens = iq_ri.permute(0, 3, 2, 1).reshape(batch_size * 32, 130, 6)
        tokens = self.input_embedding(tokens)
        tokens = tokens + self.temporal_position_encoding.to(dtype=tokens.dtype)
        for layer in self.temporal_layers:
            tokens = layer(tokens)

        temporal_weights = torch.softmax(self.temporal_pool_score(tokens), dim=1)
        range_tokens = (temporal_weights * tokens).sum(dim=1).reshape(batch_size, 32, 56)
        range_tokens = range_tokens + self.range_position_embedding
        range_features = self.range_layer(range_tokens)

        presence_by_range = self.presence_head(range_features).squeeze(-1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - math.log(32.0)
        doppler_vector = F.normalize(
            self.doppler_head(range_features), p=2.0, dim=-1, eps=1e-6
        )
        return {
            "presence_logit": presence_logit,
            "range_logits": self.range_head(range_features).squeeze(-1),
            "range_offset": torch.sigmoid(
                self.range_offset_head(range_features).squeeze(-1)
            ),
            "doppler_vector": doppler_vector,
        }
