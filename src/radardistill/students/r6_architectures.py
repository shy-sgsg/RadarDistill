"""Mechanism-motivated raw-IQ student architectures for D2-A-R6-E."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from .r6_joint import R6J1Student


R6E_ARMS = ("j1_joint", "ce_tcn", "rax_transformer")
R6E_INPUT_SHAPE = (6, 130, 32)
R6E_WIDTH = 64
R6E_RAX_DIMENSION = 48
R6E_RAX_HEADS = 4
R6E_RAX_BLOCKS = 2


def _validate_iq_ri(iq_ri: torch.Tensor) -> None:
    if iq_ri.ndim != 4 or tuple(iq_ri.shape[1:]) != R6E_INPUT_SHAPE:
        raise ValueError("iq_ri must have floating point shape [B,6,130,32]")
    if not torch.is_floating_point(iq_ri):
        raise ValueError("iq_ri must have floating point shape [B,6,130,32]")


class _ComplexConv2d(nn.Module):
    """Bias-free complex convolution with real parameters for each component."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        *,
        dilation: tuple[int, int] = (1, 1),
        groups: int = 1,
    ) -> None:
        super().__init__()
        if in_channels % groups or out_channels % groups:
            raise ValueError("complex convolution groups must divide input and output channels")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = tuple(int(value) for value in kernel_size)
        self.dilation = tuple(int(value) for value in dilation)
        self.groups = int(groups)
        self.padding = tuple(
            (self.kernel_size[index] // 2) * self.dilation[index] for index in range(2)
        )
        shape = (
            self.out_channels,
            self.in_channels // self.groups,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        self.weight_real = nn.Parameter(torch.empty(shape))
        self.weight_imag = nn.Parameter(torch.empty(shape))
        fan_in = (self.in_channels // self.groups) * math.prod(self.kernel_size)
        component_std = math.sqrt(0.5 / fan_in)
        nn.init.normal_(self.weight_real, mean=0.0, std=component_std)
        nn.init.normal_(self.weight_imag, mean=0.0, std=component_std)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(values):
            raise TypeError("complex convolution expects a complex tensor")
        real, imag = values.real, values.imag
        real_part = F.conv2d(
            real, self.weight_real, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
        ) - F.conv2d(
            imag, self.weight_imag, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
        )
        imag_part = F.conv2d(
            real, self.weight_imag, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
        ) + F.conv2d(
            imag, self.weight_real, padding=self.padding, dilation=self.dilation,
            groups=self.groups,
        )
        return torch.complex(real_part, imag_part)


class _ComplexGroupRMSNorm(nn.Module):
    """RMS normalization based only on complex magnitudes; preserves phase."""

    def __init__(self, channels: int, groups: int = 8, eps: float = 1e-8) -> None:
        super().__init__()
        if channels <= 0 or groups <= 0 or channels % groups:
            raise ValueError("complex RMS normalization groups must divide channels")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")
        self.channels = int(channels)
        self.groups = int(groups)
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(1, self.channels, 1, 1))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(values) or values.shape[1] != self.channels:
            raise ValueError("complex RMS normalization received an incompatible tensor")
        batch, _, time, ranges = values.shape
        grouped = values.abs().square().reshape(
            batch, self.groups, self.channels // self.groups, time, ranges
        )
        rms = grouped.mean(dim=(2, 3, 4), keepdim=True).add(self.eps).sqrt()
        scale = self.scale.to(dtype=values.real.dtype)
        real = values.real.reshape(batch, self.groups, self.channels // self.groups, time, ranges)
        imag = values.imag.reshape(batch, self.groups, self.channels // self.groups, time, ranges)
        real = real / rms * scale.reshape(1, self.groups, self.channels // self.groups, 1, 1)
        imag = imag / rms * scale.reshape(1, self.groups, self.channels // self.groups, 1, 1)
        return torch.complex(real.reshape_as(values.real), imag.reshape_as(values.imag))


class _RadialModReLU(nn.Module):
    """Complex activation that applies a learned real threshold to magnitude."""

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = float(eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        magnitude = values.abs()
        scale = F.relu(magnitude + self.threshold) / magnitude.clamp_min(self.eps)
        return torch.complex(values.real * scale, values.imag * scale)


class _ComplexTemporalResidualBlock(nn.Module):
    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = _ComplexConv2d(
            width, width, (3, 1), dilation=(int(dilation), 1), groups=width
        )
        self.depthwise_norm = _ComplexGroupRMSNorm(width)
        self.depthwise_activation = _RadialModReLU(width)
        self.pointwise = _ComplexConv2d(width, width, (1, 1))
        self.pointwise_norm = _ComplexGroupRMSNorm(width)
        self.pointwise_activation = _RadialModReLU(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise_activation(self.depthwise_norm(self.depthwise(values)))
        residual = self.pointwise_activation(self.pointwise_norm(self.pointwise(residual)))
        return values + residual


class R6ECEStudent(nn.Module):
    """Complex TCN with exact global-phase equivariance and invariant task readout."""

    def __init__(self, *, width: int = R6E_WIDTH) -> None:
        super().__init__()
        if width <= 0 or width % 8:
            raise ValueError("CE-TCN width must be a positive multiple of 8")
        self.width = int(width)
        self.stem = _ComplexConv2d(3, self.width, (1, 3))
        self.stem_norm = _ComplexGroupRMSNorm(self.width)
        self.stem_activation = _RadialModReLU(self.width)
        self.temporal_dilations = (1, 2, 4, 8, 16, 32, 64)
        self.blocks = nn.Sequential(
            *(_ComplexTemporalResidualBlock(self.width, dilation) for dilation in self.temporal_dilations)
        )
        self.presence_per_range = nn.Conv1d(self.width, 1, kernel_size=1)
        self.range_head = nn.Conv1d(self.width, 1, kernel_size=1)
        self.range_offset_head = nn.Conv1d(self.width, 1, kernel_size=1)
        self.doppler_head = nn.Conv1d(self.width, 2, kernel_size=1)

    def forward_features(self, iq_ri: torch.Tensor) -> torch.Tensor:
        _validate_iq_ri(iq_ri)
        complex_iq = torch.complex(iq_ri[:, :3], iq_ri[:, 3:])
        values = self.stem_activation(self.stem_norm(self.stem(complex_iq)))
        return self.blocks(values)

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        invariant = self.forward_features(iq_ri).abs().mean(dim=2)
        presence_by_range = self.presence_per_range(invariant).squeeze(1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - math.log(invariant.shape[-1])
        range_logits = self.range_head(invariant).squeeze(1)
        range_offset = torch.sigmoid(self.range_offset_head(invariant).squeeze(1))
        doppler_vector = self.doppler_head(invariant).transpose(1, 2)
        doppler_vector = F.normalize(doppler_vector, p=2.0, dim=-1, eps=1e-6)
        return {
            "presence_logit": presence_logit,
            "range_logits": range_logits,
            "range_offset": range_offset,
            "doppler_vector": doppler_vector,
        }


class _RelativeAxialAttention(nn.Module):
    def __init__(
        self, dimension: int, heads: int, sequence_length: int, axis: Literal["time", "range"]
    ) -> None:
        super().__init__()
        if dimension % heads:
            raise ValueError("attention dimension must divide evenly across heads")
        if axis not in {"time", "range"}:
            raise ValueError("axis must be 'time' or 'range'")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.head_dimension = self.dimension // self.heads
        self.sequence_length = int(sequence_length)
        self.axis = axis
        self.qkv = nn.Linear(self.dimension, 3 * self.dimension)
        self.output = nn.Linear(self.dimension, self.dimension)
        self.relative_bias = nn.Parameter(torch.zeros(self.heads, 2 * self.sequence_length - 1))
        positions = torch.arange(self.sequence_length)
        self.register_buffer(
            "relative_index",
            positions[:, None] - positions[None, :] + self.sequence_length - 1,
            persistent=False,
        )

    def _attend(self, sequence: torch.Tensor) -> torch.Tensor:
        batch, length, dimension = sequence.shape
        qkv = self.qkv(sequence).reshape(
            batch, length, 3, self.heads, self.head_dimension
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.head_dimension**-0.5)
        relative = self.relative_bias[:, self.relative_index].unsqueeze(0)
        weights = torch.softmax(scores + relative, dim=-1)
        attended = torch.matmul(weights, value).transpose(1, 2).reshape(batch, length, dimension)
        return self.output(attended)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        batch, time, ranges, dimension = grid.shape
        if dimension != self.dimension:
            raise ValueError("axial attention received an unexpected feature dimension")
        if self.axis == "time":
            if time != self.sequence_length:
                raise ValueError("time-axis attention received an unexpected sequence length")
            sequence = grid.permute(0, 2, 1, 3).reshape(batch * ranges, time, dimension)
            attended = self._attend(sequence)
            return attended.reshape(batch, ranges, time, dimension).permute(0, 2, 1, 3)
        if ranges != self.sequence_length:
            raise ValueError("range-axis attention received an unexpected sequence length")
        sequence = grid.reshape(batch * time, ranges, dimension)
        attended = self._attend(sequence)
        return attended.reshape(batch, time, ranges, dimension)


class _R6EAxialBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, time: int, ranges: int) -> None:
        super().__init__()
        self.time_norm = nn.LayerNorm(dimension)
        self.time_attention = _RelativeAxialAttention(dimension, heads, time, "time")
        self.range_norm = nn.LayerNorm(dimension)
        self.range_attention = _RelativeAxialAttention(dimension, heads, ranges, "range")
        self.feedforward_norm = nn.LayerNorm(dimension)
        self.feedforward = nn.Sequential(
            nn.Linear(dimension, 2 * dimension), nn.GELU(), nn.Linear(2 * dimension, dimension)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid + self.time_attention(self.time_norm(grid))
        grid = grid + self.range_attention(self.range_norm(grid))
        return grid + self.feedforward(self.feedforward_norm(grid))


class R6ERAXTransformer(nn.Module):
    """Small raw-IQ Transformer that alternates attention over full time/range axes."""

    def __init__(
        self,
        *,
        dimension: int = R6E_RAX_DIMENSION,
        heads: int = R6E_RAX_HEADS,
        blocks: int = R6E_RAX_BLOCKS,
    ) -> None:
        super().__init__()
        if dimension <= 0 or heads <= 0 or dimension % heads:
            raise ValueError("RAX dimension must be positive and divisible by heads")
        if blocks <= 0:
            raise ValueError("RAX must have at least one axial block")
        self.dimension = int(dimension)
        self.heads = int(heads)
        self.embedding = nn.Conv2d(6, self.dimension, kernel_size=1)
        self.blocks = nn.ModuleList(
            _R6EAxialBlock(self.dimension, self.heads, 130, 32) for _ in range(blocks)
        )
        self.presence_per_range = nn.Linear(self.dimension, 1)
        self.range_head = nn.Linear(self.dimension, 1)
        self.range_offset_head = nn.Linear(self.dimension, 1)
        self.doppler_head = nn.Linear(self.dimension, 2)

    def encode_grid(self, iq_ri: torch.Tensor) -> torch.Tensor:
        _validate_iq_ri(iq_ri)
        grid = self.embedding(iq_ri).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            grid = block(grid)
        return grid

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode_grid(iq_ri).mean(dim=1)
        presence_by_range = self.presence_per_range(features).squeeze(-1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - math.log(features.shape[1])
        range_logits = self.range_head(features).squeeze(-1)
        range_offset = torch.sigmoid(self.range_offset_head(features).squeeze(-1))
        doppler_vector = F.normalize(self.doppler_head(features), p=2.0, dim=-1, eps=1e-6)
        return {
            "presence_logit": presence_logit,
            "range_logits": range_logits,
            "range_offset": range_offset,
            "doppler_vector": doppler_vector,
        }


def build_r6e_student(arm: str, *, seed: int | None = None) -> nn.Module:
    if arm not in R6E_ARMS:
        raise ValueError(f"unknown R6-E arm {arm!r}")
    if seed is None:
        if arm == "j1_joint":
            return R6J1Student(width=96)
        return R6ECEStudent() if arm == "ce_tcn" else R6ERAXTransformer()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        if arm == "j1_joint":
            return R6J1Student(width=96)
        return R6ECEStudent() if arm == "ce_tcn" else R6ERAXTransformer()


def count_r6e_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}
