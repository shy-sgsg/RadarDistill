"""Complex diagonal SSM students for the bounded D2-A-R6-H screen.

These models use a time-invariant, learnable complex state kernel evaluated as
zero-padded linear convolution. They are S4D-like diagonal SSMs, not selective
Mamba implementations.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .r6_architectures import (
    _ComplexConv2d,
    _ComplexGroupRMSNorm,
    _ComplexTemporalResidualBlock,
    _RadialModReLU,
    _validate_iq_ri,
)


R6H_WIDTH = 56
R6H_MODES = 8
R6H_BLOCKS = 3


def _complex_linear_conv_fft(
    values: torch.Tensor,
    kernel: torch.Tensor,
    *,
    time_dim: int = 2,
) -> torch.Tensor:
    """Apply a channel-specific causal linear convolution along one axis.

    ``values`` must have complex shape ``[B, D, ..., T, ...]`` and ``kernel``
    must have complex shape ``[D, K]``. The output is cropped to the original
    sequence length, so samples at the end of a CPI cannot wrap to its start.
    """

    if not torch.is_complex(values) or not torch.is_complex(kernel):
        raise TypeError("linear complex convolution requires complex inputs and kernels")
    if values.ndim < 3 or kernel.ndim != 2:
        raise ValueError("values must have at least 3 dimensions and kernel must be [D,K]")
    if values.dtype != kernel.dtype:
        raise TypeError("values and kernel must have the same complex dtype")
    if kernel.shape[0] != values.shape[1] or kernel.shape[1] <= 0:
        raise ValueError("kernel must have shape [values.shape[1], K] with K > 0")
    time_dim = time_dim % values.ndim
    if time_dim == 1:
        raise ValueError("time_dim cannot be the channel dimension")
    if values.shape[time_dim] <= 0:
        raise ValueError("time dimension must be non-empty")

    time_length = values.shape[time_dim]
    kernel_length = kernel.shape[1]
    fft_length = 1 << (time_length + kernel_length - 2).bit_length()
    moved = values.movedim(time_dim, -1)
    value_spectrum = torch.fft.fft(moved, n=fft_length, dim=-1)
    kernel_spectrum = torch.fft.fft(kernel, n=fft_length, dim=-1)
    kernel_shape = [1] * moved.ndim
    kernel_shape[1] = kernel.shape[0]
    kernel_shape[-1] = fft_length
    result = torch.fft.ifft(
        value_spectrum * kernel_spectrum.reshape(kernel_shape), n=fft_length, dim=-1
    )[..., :time_length]
    return result.movedim(-1, time_dim)


class R6ComplexDiagonalSSMBlock(nn.Module):
    """Bidirectional complex diagonal SSM over pulses, independently by range."""

    def __init__(
        self,
        width: int,
        modes: int,
        sequence_length: int = 130,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 8:
            raise ValueError("SSM width must be a positive multiple of 8")
        if modes <= 0 or sequence_length <= 0:
            raise ValueError("modes and sequence_length must be positive")
        self.width = int(width)
        self.modes = int(modes)
        self.sequence_length = int(sequence_length)

        for direction in ("forward", "reverse"):
            setattr(self, f"{direction}_log_decay", nn.Parameter(torch.full(
                (self.width, self.modes), math.log(0.9 / 0.1)
            )))
            phase = nn.Parameter(torch.empty(self.width, self.modes))
            nn.init.uniform_(phase, -math.pi, math.pi)
            setattr(self, f"{direction}_phase", phase)
            for name in ("b", "c"):
                real = nn.Parameter(torch.empty(self.width, self.modes))
                imag = nn.Parameter(torch.empty(self.width, self.modes))
                nn.init.normal_(real, mean=0.0, std=math.sqrt(0.5 / self.modes))
                nn.init.normal_(imag, mean=0.0, std=math.sqrt(0.5 / self.modes))
                setattr(self, f"{direction}_{name}_real", real)
                setattr(self, f"{direction}_{name}_imag", imag)

        self.forward_mixer = _ComplexConv2d(self.width, self.width, (1, 1))
        self.reverse_mixer = _ComplexConv2d(self.width, self.width, (1, 1))
        self.magnitude_gate = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.norm = _ComplexGroupRMSNorm(self.width)
        self.activation = _RadialModReLU(self.width)

    def _state_kernel(self, direction: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
        log_decay = getattr(self, f"{direction}_log_decay").to(device=device, dtype=real_dtype)
        phase = getattr(self, f"{direction}_phase").to(device=device, dtype=real_dtype)
        magnitude = torch.sigmoid(log_decay).clamp(max=1.0 - 1e-6)
        steps = torch.arange(self.sequence_length, device=device, dtype=real_dtype)
        amplitudes = magnitude.unsqueeze(-1).pow(steps)
        angles = phase.unsqueeze(-1) * steps
        state_powers = torch.polar(amplitudes, angles)
        b = torch.complex(
            getattr(self, f"{direction}_b_real").to(device=device, dtype=real_dtype),
            getattr(self, f"{direction}_b_imag").to(device=device, dtype=real_dtype),
        )
        c = torch.complex(
            getattr(self, f"{direction}_c_real").to(device=device, dtype=real_dtype),
            getattr(self, f"{direction}_c_imag").to(device=device, dtype=real_dtype),
        )
        return (state_powers * (b * c).unsqueeze(-1)).sum(dim=1).to(dtype=dtype)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(values) or values.ndim != 4 or values.shape[1] != self.width:
            raise ValueError("SSM block expects complex values with shape [B,width,time,range]")
        if values.shape[2] != self.sequence_length:
            raise ValueError("SSM block received an unexpected slow-time length")

        forward = _complex_linear_conv_fft(
            values,
            self._state_kernel("forward", device=values.device, dtype=values.dtype),
            time_dim=2,
        )
        reverse_input = values.flip(dims=(2,))
        reverse = _complex_linear_conv_fft(
            reverse_input,
            self._state_kernel("reverse", device=values.device, dtype=values.dtype),
            time_dim=2,
        ).flip(dims=(2,))
        update = self.forward_mixer(forward) + self.reverse_mixer(reverse)
        update = self.activation(self.norm(update))
        gate = torch.sigmoid(self.magnitude_gate(values.abs()))
        return values + update * gate


class _R6ComplexSSMBase(nn.Module):
    def __init__(
        self,
        *,
        width: int,
        modes: int,
        blocks: int,
        local_temporal_block: bool,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 8:
            raise ValueError("student width must be a positive multiple of 8")
        if modes <= 0 or blocks <= 0:
            raise ValueError("modes and blocks must be positive")
        self.width = int(width)
        self.modes = int(modes)
        self.stem = _ComplexConv2d(3, self.width, (1, 3))
        self.stem_norm = _ComplexGroupRMSNorm(self.width)
        self.stem_activation = _RadialModReLU(self.width)
        self.local_temporal = (
            _ComplexTemporalResidualBlock(self.width, dilation=1)
            if local_temporal_block else None
        )
        self.blocks = nn.ModuleList(
            R6ComplexDiagonalSSMBlock(self.width, self.modes, sequence_length=130)
            for _ in range(int(blocks))
        )
        self.presence_per_range = nn.Conv1d(self.width, 1, kernel_size=1)
        self.range_head = nn.Conv1d(self.width, 1, kernel_size=1)
        self.range_offset_head = nn.Conv1d(self.width, 1, kernel_size=1)
        self.doppler_head = nn.Conv1d(self.width, 2, kernel_size=1)

    def forward_features(self, iq_ri: torch.Tensor) -> torch.Tensor:
        _validate_iq_ri(iq_ri)
        complex_iq = torch.complex(iq_ri[:, :3], iq_ri[:, 3:])
        values = self.stem_activation(self.stem_norm(self.stem(complex_iq)))
        if self.local_temporal is not None:
            values = self.local_temporal(values)
        for block in self.blocks:
            values = block(values)
        return values

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        invariant = self.forward_features(iq_ri).abs().mean(dim=2)
        presence_by_range = self.presence_per_range(invariant).squeeze(1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - math.log(invariant.shape[-1])
        range_logits = self.range_head(invariant).squeeze(1)
        range_offset = torch.sigmoid(self.range_offset_head(invariant).squeeze(1))
        doppler_vector = self.doppler_head(invariant).transpose(1, 2)
        doppler_vector = nn.functional.normalize(doppler_vector, p=2.0, dim=-1, eps=1e-6)
        return {
            "presence_logit": presence_logit,
            "range_logits": range_logits,
            "range_offset": range_offset,
            "doppler_vector": doppler_vector,
        }


class R6ComplexSSMStudent(_R6ComplexSSMBase):
    """Raw-IQ complex SSM with no learned FFT or fixed radar spectral layer."""

    def __init__(
        self,
        *,
        width: int = R6H_WIDTH,
        modes: int = R6H_MODES,
        blocks: int = R6H_BLOCKS,
    ) -> None:
        super().__init__(width=width, modes=modes, blocks=blocks, local_temporal_block=False)


class R6ComplexCNNSSMStudent(_R6ComplexSSMBase):
    """Raw-IQ complex local-convolution front end followed by complex SSMs."""

    def __init__(
        self,
        *,
        width: int = R6H_WIDTH,
        modes: int = R6H_MODES,
        blocks: int = R6H_BLOCKS,
    ) -> None:
        super().__init__(width=width, modes=modes, blocks=blocks, local_temporal_block=True)


def count_r6h_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}
