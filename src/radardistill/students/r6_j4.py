"""Fixed-geometry DPCA/FFT front end and joint student for D2-A-R6-J4."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from radardistill.distillation.task_losses import encode_doppler_hz


R6J4_DOPPLER_BINS = 520
R6J4_RANGE_BINS = 32
R6J4_PULSE_COUNT = 130
R6J4_CHANNEL_COUNT = 3


def wrap_r6j4_doppler(frequency_hz: torch.Tensor, prf_hz: float) -> torch.Tensor:
    """Wrap Doppler estimates into the half-open interval [-PRF/2, PRF/2)."""
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    return torch.remainder(frequency_hz + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def _validate_iq_ri(iq_ri: torch.Tensor) -> None:
    if (
        iq_ri.ndim != 4
        or iq_ri.shape[1:] != (2 * R6J4_CHANNEL_COUNT, R6J4_PULSE_COUNT, R6J4_RANGE_BINS)
        or iq_ri.dtype != torch.float32
    ):
        raise ValueError("iq_ri must be float32 with shape [B,6,130,32]")
    if not torch.isfinite(iq_ri).all():
        raise ValueError("iq_ri must contain only finite values")


def r6j4_dpca_fft(
    iq_ri: torch.Tensor,
    *,
    dpca_delay_s: float,
    prf_hz: float = 1300.0,
    nfft: int = R6J4_DOPPLER_BINS,
) -> torch.Tensor:
    """Apply the fixed channel-pair geometric DPCA and a zero-padded FFT.

    The input planes are ordered as ``[real(ch0..2), imag(ch0..2)]``. The
    complex output is ``[B, nfft, 32]`` with the Doppler axis fftshifted.
    This function is device preserving and performs no truth-dependent work.
    """
    _validate_iq_ri(iq_ri)
    if not math.isfinite(dpca_delay_s):
        raise ValueError("dpca_delay_s must be finite")
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if isinstance(nfft, bool) or not isinstance(nfft, int) or nfft < R6J4_PULSE_COUNT:
        raise ValueError("nfft must be an integer no smaller than the pulse count")

    # The fractional geometry shift subtracts two nearly coherent channels.
    # Accumulate this fixed phase shift in complex128 so the registered CUDA
    # implementation remains within the complex128 reference tolerance; cast
    # its output back to complex64 before the FFT and FP32 learned readout.
    shifted_pair = torch.complex(iq_ri[:, :2], iq_ri[:, R6J4_CHANNEL_COUNT:R6J4_CHANNEL_COUNT + 2])
    shifted_pair = shifted_pair.to(torch.complex128)
    frequencies_hz = torch.fft.fftfreq(
        R6J4_PULSE_COUNT,
        d=1.0 / prf_hz,
        device=iq_ri.device,
        dtype=torch.float64,
    )
    phase = torch.polar(torch.ones_like(frequencies_hz), 2.0 * math.pi * frequencies_hz * dpca_delay_s)
    shifted_second = torch.fft.ifft(
        torch.fft.fft(shifted_pair[:, 1], dim=1) * phase[None, :, None], dim=1
    )
    difference = (shifted_pair[:, 0] - shifted_second).to(torch.complex64)
    return torch.fft.fftshift(torch.fft.fft(difference, n=nfft, dim=1), dim=1)


class _J4ResidualBlock(nn.Module):
    """Compact depthwise-separable residual block without amplitude normalization."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.pointwise = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(F.silu(value))
        residual = self.pointwise(F.silu(residual))
        return value + residual


def _pool_j4_parameters(
    candidate_probability: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_doppler_vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    batch = candidate_probability.shape[0]
    expected = 1 + R6J4_DOPPLER_BINS * R6J4_RANGE_BINS
    if candidate_probability.shape != (batch, expected):
        raise ValueError("candidate_probability has an unexpected null/grid size")
    grid_shape = (batch, R6J4_DOPPLER_BINS, R6J4_RANGE_BINS)
    if candidate_range.shape != grid_shape or candidate_doppler_vector.shape != grid_shape + (2,):
        raise ValueError("candidate parameter grids have unexpected shapes")

    grid_probability = candidate_probability[:, 1:].reshape(grid_shape)
    nonnull_mass = grid_probability.sum(dim=(1, 2))
    conditional = grid_probability / nonnull_mass[:, None, None].clamp_min(1e-12)
    range_cell = (conditional * candidate_range).sum(dim=(1, 2))
    doppler_vector = (conditional.unsqueeze(-1) * candidate_doppler_vector).sum(dim=(1, 2))
    doppler_vector = F.normalize(doppler_vector, p=2.0, dim=-1, eps=1e-6)

    range_mass = grid_probability.sum(dim=1)
    conditional_by_range = grid_probability / range_mass[:, None, :].clamp_min(1e-12)
    range_offset = (conditional_by_range * (candidate_range - torch.arange(
        R6J4_RANGE_BINS, dtype=candidate_range.dtype, device=candidate_range.device
    ).view(1, 1, -1))).sum(dim=1)
    doppler_by_range = (conditional_by_range.unsqueeze(-1) * candidate_doppler_vector).sum(dim=1)
    doppler_by_range = F.normalize(doppler_by_range, p=2.0, dim=-1, eps=1e-6)
    return {
        "range_cell_estimate": range_cell,
        "doppler_vector_pooled": doppler_vector,
        "conditional_candidate_probability": conditional,
        "nonnull_probability": nonnull_mass,
        "range_mass": range_mass,
        "range_offset": range_offset,
        "doppler_vector": doppler_by_range,
    }


class R6J4Student(nn.Module):
    """A fixed DPCA/520-FFT front end with a learned null/candidate task head."""

    def __init__(
        self,
        *,
        dpca_delay_s: float,
        width: int = 48,
        prf_hz: float = 1300.0,
        nfft: int = R6J4_DOPPLER_BINS,
    ) -> None:
        super().__init__()
        if not math.isfinite(dpca_delay_s):
            raise ValueError("dpca_delay_s must be finite")
        if not math.isfinite(prf_hz) or prf_hz <= 0:
            raise ValueError("prf_hz must be positive and finite")
        if isinstance(nfft, bool) or not isinstance(nfft, int) or nfft < R6J4_PULSE_COUNT:
            raise ValueError("nfft must be an integer no smaller than the pulse count")
        if width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of 8")
        self.dpca_delay_s = float(dpca_delay_s)
        self.prf_hz = float(prf_hz)
        self.nfft = int(nfft)
        self.width = int(width)

        self.stem = nn.Conv2d(1, self.width, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*(_J4ResidualBlock(self.width) for _ in range(3)))
        self.candidate_head = nn.Conv2d(self.width, 1, kernel_size=1)
        self.null_head = nn.Linear(self.width, 1)
        self.parameter_head = nn.Conv2d(self.width, 3, kernel_size=1)
        feedback_width = max(8, self.width // 2)
        self.feedback_hidden = nn.Conv2d(self.width + 3, feedback_width, kernel_size=1)
        self.feedback_output = nn.Conv2d(feedback_width, 1, kernel_size=1)
        nn.init.normal_(self.feedback_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.feedback_output.bias)

        bin_width = self.prf_hz / self.nfft
        frequency_centers = torch.fft.fftshift(
            torch.fft.fftfreq(self.nfft, d=1.0 / self.prf_hz)
        )
        self.register_buffer("doppler_centers_hz", frequency_centers, persistent=True)
        self.bin_width_hz = bin_width

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        response = r6j4_dpca_fft(
            iq_ri,
            dpca_delay_s=self.dpca_delay_s,
            prf_hz=self.prf_hz,
            nfft=self.nfft,
        )
        log_power = torch.log1p(response.real.square() + response.imag.square()).unsqueeze(1)
        features = self.blocks(F.silu(self.stem(log_power)))

        parameters = self.parameter_head(features)
        range_offset_candidate = torch.sigmoid(parameters[:, 0])
        doppler_residual_hz = torch.tanh(parameters[:, 1]) * (self.bin_width_hz / 2.0)
        feedback_confidence = torch.tanh(parameters[:, 2])
        phase = doppler_residual_hz * (2.0 * math.pi / self.prf_hz)
        feedback_input = torch.cat(
            (
                features,
                range_offset_candidate.unsqueeze(1),
                torch.sin(phase).unsqueeze(1),
                feedback_confidence.unsqueeze(1),
            ),
            dim=1,
        )
        feedback = torch.tanh(
            self.feedback_output(F.silu(self.feedback_hidden(feedback_input)))
        ).squeeze(1)

        base_logits = self.candidate_head(features).squeeze(1)
        candidate_logits_grid = base_logits + feedback
        candidate_logits = candidate_logits_grid.flatten(start_dim=1)
        null_logit = self.null_head(features.mean(dim=(2, 3))).squeeze(-1)
        presence_logit = torch.logsumexp(candidate_logits, dim=-1) - null_logit
        candidate_probability = torch.softmax(
            torch.cat((null_logit[:, None], candidate_logits), dim=-1), dim=-1
        )

        range_indices = torch.arange(
            R6J4_RANGE_BINS, dtype=features.dtype, device=features.device
        ).view(1, 1, R6J4_RANGE_BINS)
        candidate_range = (range_indices + range_offset_candidate).expand(
            features.shape[0], self.nfft, R6J4_RANGE_BINS
        )
        base_doppler = self.doppler_centers_hz.to(dtype=features.dtype).view(1, self.nfft, 1)
        candidate_doppler_hz = wrap_r6j4_doppler(
            base_doppler + doppler_residual_hz, self.prf_hz
        )
        candidate_phase = candidate_doppler_hz * (2.0 * math.pi / self.prf_hz)
        candidate_doppler_vector = torch.stack(
            (torch.cos(candidate_phase), torch.sin(candidate_phase)), dim=-1
        ).expand(features.shape[0], self.nfft, R6J4_RANGE_BINS, 2)
        pooled = _pool_j4_parameters(
            candidate_probability, candidate_range, candidate_doppler_vector
        )
        return {
            "presence_logit": presence_logit,
            "candidate_probability": candidate_probability,
            "candidate_logits": candidate_logits,
            "candidate_logits_base": base_logits.flatten(start_dim=1),
            "parameter_feedback": feedback.flatten(start_dim=1),
            "null_logit": null_logit,
            "candidate_range": candidate_range,
            "candidate_doppler_hz": candidate_doppler_hz,
            "candidate_doppler_vector": candidate_doppler_vector,
            "range_cell_estimate": pooled["range_cell_estimate"],
            "range_logits": torch.log(pooled["range_mass"].clamp_min(1e-12)),
            "range_offset": pooled["range_offset"],
            "doppler_vector": pooled["doppler_vector"],
            "doppler_vector_pooled": pooled["doppler_vector_pooled"],
            "nonnull_probability": pooled["nonnull_probability"],
        }


def r6j4_loss(
    outputs: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    *,
    prf_hz: float = 1300.0,
    range_beta: float = 0.1,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    """Truth-only joint loss; H0 never receives candidate or parameter labels."""
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if not math.isfinite(range_beta) or range_beta <= 0:
        raise ValueError("range_beta must be positive and finite")
    default_weights = {"presence": 1.0, "candidate": 1.0, "range": 1.0, "doppler": 1.0}
    if weights is not None:
        default_weights.update({key: float(value) for key, value in weights.items()})
    if any(not math.isfinite(value) or value < 0 for value in default_weights.values()):
        raise ValueError("loss weights must be finite and nonnegative")

    presence_logit = outputs["presence_logit"]
    presence_truth = truth["truth_presence"].to(
        device=presence_logit.device, dtype=presence_logit.dtype
    )
    if presence_truth.shape != presence_logit.shape:
        raise ValueError("truth_presence must have one label per sample")
    h1 = presence_truth > 0.5
    presence = F.binary_cross_entropy_with_logits(presence_logit, presence_truth)
    zero = presence_logit.sum() * 0.0
    candidate = zero
    range_loss = zero
    doppler = zero
    if torch.any(h1):
        range_truth = truth["truth_range_cell"].to(
            device=presence_logit.device, dtype=presence_logit.dtype
        )
        doppler_truth = truth["truth_doppler_hz"].to(
            device=presence_logit.device, dtype=presence_logit.dtype
        )
        if range_truth.shape != presence_truth.shape or doppler_truth.shape != presence_truth.shape:
            raise ValueError("range and Doppler truths must have one value per sample")
        range_index = torch.floor(range_truth[h1]).long().clamp(0, R6J4_RANGE_BINS - 1)
        wrapped = wrap_r6j4_doppler(doppler_truth[h1], prf_hz)
        doppler_index = torch.floor(
            (wrapped + prf_hz / 2.0) / (prf_hz / R6J4_DOPPLER_BINS)
        ).long().clamp(0, R6J4_DOPPLER_BINS - 1)
        target_index = doppler_index * R6J4_RANGE_BINS + range_index
        candidate = F.cross_entropy(outputs["candidate_logits"][h1], target_index)
        range_loss = F.smooth_l1_loss(
            outputs["range_cell_estimate"][h1], range_truth[h1], beta=range_beta
        )
        target_vector = encode_doppler_hz(doppler_truth[h1], prf_hz)
        predicted_vector = F.normalize(
            outputs["doppler_vector_pooled"][h1], p=2.0, dim=-1, eps=1e-6
        )
        doppler = (1.0 - (predicted_vector * target_vector).sum(dim=-1)).mean()

    total = (
        default_weights["presence"] * presence
        + default_weights["candidate"] * candidate
        + default_weights["range"] * range_loss
        + default_weights["doppler"] * doppler
    )
    return {
        "presence": presence,
        "candidate": candidate,
        "range": range_loss,
        "doppler": doppler,
        "total": total,
    }
