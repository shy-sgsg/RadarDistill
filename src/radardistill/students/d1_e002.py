"""Frozen-scope Doppler readout candidates for D1-E002."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .task_student import TaskStudent


def wrap_frequency_hz(frequency_hz: torch.Tensor, prf_hz: float) -> torch.Tensor:
    """Wrap frequencies to the half-open Nyquist interval [-PRF/2, PRF/2)."""
    if not math.isfinite(prf_hz) or prf_hz <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    return torch.remainder(frequency_hz + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def lag_one_complex_coherence(
    iq: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Return per-channel/per-range normalized lag-one complex coherence.

    Args:
        iq: Finite complex samples shaped ``[B,3,130,32]``.
        epsilon: Stabilizer added after the square-root energy product.

    Returns:
        Real tensor shaped ``[B,3,32,2]`` with real/imaginary components.
    """
    if iq.ndim != 4 or tuple(iq.shape[1:]) != (3, 130, 32):
        raise ValueError("iq must have shape [B,3,130,32]")
    if not torch.is_complex(iq) or not torch.isfinite(iq).all():
        raise ValueError("iq must contain finite complex samples")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be positive and finite")

    left = iq[:, :, :-1, :]
    right = iq[:, :, 1:, :]
    numerator = torch.sum(right * torch.conj(left), dim=2)
    left_energy = torch.sum(torch.abs(left).square(), dim=2)
    right_energy = torch.sum(torch.abs(right).square(), dim=2)
    denominator = torch.sqrt(left_energy * right_energy) + epsilon
    coherence = numerator / denominator
    return torch.stack((coherence.real, coherence.imag), dim=-1)


class D1E002Student(nn.Module):
    """S1/S2 pure-IQ students and the explicitly hybrid S3 student.

    All variants accept real/imaginary planes shaped ``[B,6,130,32]``. The
    forward signature has no truth or teacher inputs. S1/S2 use the D1 TCN;
    S3 uses that TCN for presence/range and a fixed 520-point FFT plus a small
    learned local residual for Doppler.
    """

    def __init__(
        self,
        *,
        variant: str,
        width: int = 32,
        temporal_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        prf_hz: float = 1300.0,
        fft_nfft: int = 520,
        coherence_epsilon: float = 1.0e-8,
        fft_epsilon: float = 1.0e-12,
    ) -> None:
        super().__init__()
        variant = str(variant).upper()
        if variant not in {"S1", "S2", "S3"}:
            raise ValueError("variant must be S1, S2, or S3")
        if not math.isfinite(prf_hz) or prf_hz <= 0.0:
            raise ValueError("prf_hz must be positive and finite")
        if fft_nfft != 520:
            raise ValueError("D1-E002 S3 freezes fft_nfft=520")
        if coherence_epsilon <= 0.0 or fft_epsilon <= 0.0:
            raise ValueError("feature epsilons must be positive")

        self.variant = variant
        self.width = int(width)
        self.prf_hz = float(prf_hz)
        self.fft_nfft = int(fft_nfft)
        self.coherence_epsilon = float(coherence_epsilon)
        self.fft_epsilon = float(fft_epsilon)
        self.method_kind = "fft_assisted_hybrid" if variant == "S3" else "pure_iq"
        self.is_hybrid = variant == "S3"
        self.backbone = TaskStudent(
            width=width,
            temporal_dilations=temporal_dilations,
            include_spectrum_head=False,
        )

        if variant == "S1":
            self.backbone.doppler_head = nn.Conv1d(width * 4, 2, kernel_size=1)
        elif variant == "S2":
            self.backbone.doppler_head = nn.Conv1d(width + 6, 2, kernel_size=1)
        else:
            self.backbone.doppler_head = None
            self.residual_mlp = nn.Sequential(
                nn.Linear(5, 16),
                nn.SiLU(),
                nn.Linear(16, 16),
                nn.SiLU(),
                nn.Linear(16, 1),
            )

    def _check_input(self, iq_ri: torch.Tensor) -> None:
        if iq_ri.ndim != 4 or tuple(iq_ri.shape[1:]) != (6, 130, 32):
            raise ValueError("iq_ri must have shape [B,6,130,32]")
        if not torch.is_floating_point(iq_ri) or not torch.isfinite(iq_ri).all():
            raise ValueError("iq_ri must contain finite real/imaginary floating-point planes")

    @staticmethod
    def _task_outputs(features: torch.Tensor, backbone: TaskStudent) -> dict[str, torch.Tensor]:
        presence_by_range = backbone.presence_per_range(features).squeeze(1)
        presence_logit = torch.logsumexp(presence_by_range, dim=-1) - torch.log(
            torch.as_tensor(features.shape[-1], dtype=features.dtype, device=features.device)
        )
        range_logits = backbone.range_head(features).squeeze(1)
        range_offset = torch.sigmoid(backbone.range_offset_head(features).squeeze(1))
        return {
            "presence_logit": presence_logit,
            "range_logits": range_logits,
            "range_offset": range_offset,
        }

    @staticmethod
    def _circular_vector(frequency_hz: torch.Tensor, prf_hz: float) -> torch.Tensor:
        phase = frequency_hz * (2.0 * math.pi / prf_hz)
        return F.normalize(torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1), dim=-1, eps=1e-6)

    def _s3_frequency(self, iq_ri: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        real = iq_ri[:, :3]
        imag = iq_ri[:, 3:]
        complex_iq = torch.complex(real, imag)
        spectrum = torch.fft.fftshift(torch.fft.fft(complex_iq, n=self.fft_nfft, dim=2), dim=2)
        power = spectrum.abs().square().sum(dim=1)  # [B,520,32]
        coarse_bin = power.argmax(dim=1)  # [B,32]
        offsets = torch.arange(-2, 3, device=iq_ri.device)
        local_indices = (coarse_bin.unsqueeze(-1) + offsets) % self.fft_nfft
        local_power = torch.gather(power.transpose(1, 2), dim=2, index=local_indices)
        log_patch = torch.log(local_power + self.fft_epsilon)
        patch = log_patch - log_patch.mean(dim=-1, keepdim=True)
        delta_bins = 0.5 * torch.tanh(self.residual_mlp(patch).squeeze(-1))
        coarse_hz = (coarse_bin.to(iq_ri.dtype) - self.fft_nfft // 2) * (
            self.prf_hz / self.fft_nfft
        )
        frequency_hz = wrap_frequency_hz(
            coarse_hz + delta_bins * (self.prf_hz / self.fft_nfft), self.prf_hz
        )
        return coarse_hz, delta_bins, frequency_hz

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        self._check_input(iq_ri)
        temporal = self.backbone.temporal_blocks(self.backbone.stem(iq_ri))
        pooled = temporal.mean(dim=2)
        outputs = self._task_outputs(pooled, self.backbone)

        if self.variant == "S1":
            segments = (
                temporal[:, :, 0:32].mean(dim=2),
                temporal[:, :, 32:64].mean(dim=2),
                temporal[:, :, 64:96].mean(dim=2),
                temporal[:, :, 96:130].mean(dim=2),
            )
            per_range = torch.cat(segments, dim=1)
            vector = self.backbone.doppler_head(per_range).transpose(1, 2)
            outputs["doppler_vector"] = F.normalize(vector, p=2.0, dim=-1, eps=1e-6)
        elif self.variant == "S2":
            complex_iq = torch.complex(iq_ri[:, :3], iq_ri[:, 3:])
            coherence = lag_one_complex_coherence(
                complex_iq, epsilon=self.coherence_epsilon
            )  # [B,3,32,2]
            coherence_features = coherence.permute(0, 1, 3, 2).reshape(
                iq_ri.shape[0], 6, 32
            )
            per_range = torch.cat((pooled, coherence_features), dim=1)
            vector = self.backbone.doppler_head(per_range).transpose(1, 2)
            outputs["doppler_vector"] = F.normalize(vector, p=2.0, dim=-1, eps=1e-6)
        else:
            coarse_hz, delta_bins, frequency_hz = self._s3_frequency(iq_ri)
            outputs["doppler_vector"] = self._circular_vector(frequency_hz, self.prf_hz)
            outputs["doppler_coarse_hz"] = coarse_hz
            outputs["doppler_delta_bins"] = delta_bins
        return outputs
