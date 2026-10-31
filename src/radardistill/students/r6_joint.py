"""Matched shared-task and soft-candidate students for D2-A-R6-D."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .task_student import TaskStudent, _TemporalResidualBlock


R6D_ARMS = ("j1_detection", "j1_parameter", "j1_joint", "j2_joint")
R6D_DEFAULT_WIDTH = 96
R6D_DOPPLER_BINS = 520
R6D_RANGE_BINS = 32
R6D_TEMPORAL_DILATIONS = (1, 2, 4, 8, 16, 32, 64)


def wrap_r6d_doppler(frequency_hz: torch.Tensor, prf_hz: float) -> torch.Tensor:
    """Wrap frequencies to the half-open interval [-PRF/2, PRF/2)."""
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    return torch.remainder(frequency_hz + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def presence_logit_from_candidate_logits(
    candidate_logits: torch.Tensor, null_logit: torch.Tensor
) -> torch.Tensor:
    """Return log odds for any candidate versus the explicit null class."""
    if candidate_logits.ndim != 2 or null_logit.shape != candidate_logits.shape[:1]:
        raise ValueError("candidate logits must be [B,N] and null logits [B]")
    if candidate_logits.shape[1] == 0:
        raise ValueError("candidate logits must contain at least one candidate")
    return torch.logsumexp(candidate_logits, dim=-1) - null_logit


class R6J1Student(nn.Module):
    """TaskStudent shared trunk with selectable truth-only objective at training."""

    def __init__(
        self,
        *,
        width: int = R6D_DEFAULT_WIDTH,
        temporal_dilations: Sequence[int] = R6D_TEMPORAL_DILATIONS,
    ) -> None:
        super().__init__()
        self.width = int(width)
        self.temporal_dilations = tuple(int(value) for value in temporal_dilations)
        self.task_student = TaskStudent(
            width=self.width,
            temporal_dilations=self.temporal_dilations,
            include_spectrum_head=False,
        )

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.task_student(iq_ri, compute_spectrum=False)


class _R6TemporalTrunk(nn.Module):
    """The TaskStudent stem and residual TCN, preserving its parameterization."""

    def __init__(
        self,
        width: int,
        temporal_dilations: Sequence[int] = R6D_TEMPORAL_DILATIONS,
    ) -> None:
        super().__init__()
        if width <= 0 or width % 8:
            raise ValueError("width must be a positive multiple of 8")
        dilations = tuple(int(value) for value in temporal_dilations)
        if not dilations or any(value <= 0 for value in dilations):
            raise ValueError("temporal dilations must be positive")
        self.stem = nn.Sequential(
            nn.Conv2d(6, width, kernel_size=(1, 3), padding=(0, 1), bias=False),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        self.temporal_blocks = nn.Sequential(
            *(_TemporalResidualBlock(width, dilation) for dilation in dilations)
        )

    def forward(self, iq_ri: torch.Tensor) -> torch.Tensor:
        if (
            iq_ri.ndim != 4
            or iq_ri.shape[1:] != (6, 130, 32)
            or not torch.is_floating_point(iq_ri)
        ):
            raise ValueError("iq_ri must be floating point with shape [B,6,130,32]")
        if not torch.isfinite(iq_ri).all():
            raise ValueError("iq_ri must contain only finite values")
        return self.temporal_blocks(self.stem(iq_ri))


def pool_r6d_j2_parameters(
    candidate_probability: torch.Tensor,
    candidate_range: torch.Tensor,
    candidate_doppler_vector: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Pool continuous parameters over the conditional non-null posterior."""
    if candidate_probability.ndim != 2:
        raise ValueError("candidate_probability must be [B,1+520*32]")
    batch = candidate_probability.shape[0]
    if candidate_probability.shape[1] != 1 + R6D_DOPPLER_BINS * R6D_RANGE_BINS:
        raise ValueError("candidate_probability has an unexpected null/grid size")
    expected_grid = (batch, R6D_DOPPLER_BINS, R6D_RANGE_BINS)
    if candidate_range.shape != expected_grid:
        raise ValueError("candidate_range must be [B,520,32]")
    if candidate_doppler_vector.shape != expected_grid + (2,):
        raise ValueError("candidate_doppler_vector must be [B,520,32,2]")

    probability = candidate_probability[:, 1:].reshape(expected_grid)
    nonnull_mass = probability.sum(dim=(1, 2), keepdim=True)
    conditional = probability / nonnull_mass.clamp_min(1e-12)
    range_cell = (conditional * candidate_range).sum(dim=(1, 2))
    doppler_vector = (conditional.unsqueeze(-1) * candidate_doppler_vector).sum(dim=(1, 2))
    doppler_vector = F.normalize(doppler_vector, p=2.0, dim=-1, eps=1e-6)
    return {
        "range_cell": range_cell,
        "doppler_vector": doppler_vector,
        "conditional_candidate_probability": conditional,
        "nonnull_probability": nonnull_mass.reshape(batch),
    }


class R6J2Student(nn.Module):
    """One-round soft candidate interaction student with a learned 520-bin grid."""

    def __init__(
        self,
        *,
        width: int = R6D_DEFAULT_WIDTH,
        prf_hz: float = 1300.0,
        temporal_dilations: Sequence[int] = R6D_TEMPORAL_DILATIONS,
    ) -> None:
        super().__init__()
        if not math.isfinite(prf_hz) or prf_hz <= 0:
            raise ValueError("prf_hz must be positive and finite")
        if width < 8 or width % 8:
            raise ValueError("width must be a positive multiple of 8")
        self.width = int(width)
        self.prf_hz = float(prf_hz)
        self.trunk = _R6TemporalTrunk(self.width, temporal_dilations)
        # A learned temporal projection, shared over feature channels and range.
        self.temporal_candidate_projection = nn.Linear(130, R6D_DOPPLER_BINS)
        self.candidate_score = nn.Conv2d(self.width, 1, kernel_size=1)
        self.null_score = nn.Linear(self.width, 1)
        self.parameter_head = nn.Conv2d(self.width, 3, kernel_size=1)
        self.feedback_hidden = nn.Conv2d(self.width + 4, max(4, self.width // 2), kernel_size=1)
        self.feedback_output = nn.Conv2d(max(4, self.width // 2), 1, kernel_size=1)
        nn.init.normal_(self.feedback_output.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.feedback_output.bias)
        bin_width = self.prf_hz / R6D_DOPPLER_BINS
        centers = -self.prf_hz / 2.0 + (torch.arange(R6D_DOPPLER_BINS) + 0.5) * bin_width
        self.register_buffer("doppler_centers_hz", centers, persistent=True)

    def forward(self, iq_ri: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal = self.trunk(iq_ri)
        projected = self.temporal_candidate_projection(temporal.permute(0, 1, 3, 2))
        candidate_features = projected.permute(0, 1, 3, 2).contiguous()

        local = self.parameter_head(candidate_features)
        range_offset_candidate = torch.sigmoid(local[:, 0])
        half_bin_hz = self.prf_hz / (2.0 * R6D_DOPPLER_BINS)
        doppler_residual_hz = torch.tanh(local[:, 1]) * half_bin_hz
        uncertainty = local[:, 2].clamp(-6.0, 6.0)
        phase = doppler_residual_hz * (2.0 * math.pi / self.prf_hz)
        feedback_inputs = torch.cat(
            (
                candidate_features,
                range_offset_candidate.unsqueeze(1),
                torch.sin(phase).unsqueeze(1),
                torch.cos(phase).unsqueeze(1),
                uncertainty.unsqueeze(1),
            ),
            dim=1,
        )
        parameter_feedback = torch.tanh(
            self.feedback_output(F.silu(self.feedback_hidden(feedback_inputs)))
        ).squeeze(1)
        candidate_logits_base = self.candidate_score(candidate_features).squeeze(1)
        candidate_logits = candidate_logits_base + parameter_feedback
        pooled = temporal.mean(dim=(2, 3))
        null_logit = self.null_score(pooled).squeeze(-1)
        flat_candidates = candidate_logits.flatten(start_dim=1)
        presence_logit = presence_logit_from_candidate_logits(flat_candidates, null_logit)
        all_logits = torch.cat((null_logit.unsqueeze(1), flat_candidates), dim=1)
        candidate_probability = torch.softmax(all_logits, dim=-1)

        range_indices = torch.arange(
            R6D_RANGE_BINS, dtype=temporal.dtype, device=temporal.device
        ).view(1, 1, R6D_RANGE_BINS)
        candidate_range = range_indices + range_offset_candidate
        base_doppler = self.doppler_centers_hz.to(dtype=temporal.dtype).view(1, R6D_DOPPLER_BINS, 1)
        candidate_doppler_hz = wrap_r6d_doppler(base_doppler + doppler_residual_hz, self.prf_hz)
        candidate_phase = candidate_doppler_hz * (2.0 * math.pi / self.prf_hz)
        candidate_doppler_vector = torch.stack((torch.cos(candidate_phase), torch.sin(candidate_phase)), dim=-1)
        candidate_range = candidate_range.expand(temporal.shape[0], -1, -1)
        candidate_doppler_vector = candidate_doppler_vector.expand(temporal.shape[0], -1, -1, -1)

        grid_probability = candidate_probability[:, 1:].reshape(
            -1, R6D_DOPPLER_BINS, R6D_RANGE_BINS
        )
        range_mass = grid_probability.sum(dim=1)
        per_range = grid_probability / range_mass.unsqueeze(1).clamp_min(1e-12)
        range_offset = (per_range * range_offset_candidate).sum(dim=1)
        doppler_vector_by_range = (per_range.unsqueeze(-1) * candidate_doppler_vector).sum(dim=1)
        doppler_vector_by_range = F.normalize(doppler_vector_by_range, p=2.0, dim=-1, eps=1e-6)
        pooled_parameters = pool_r6d_j2_parameters(
            candidate_probability, candidate_range, candidate_doppler_vector
        )
        return {
            "presence_logit": presence_logit,
            "range_logits": torch.log(range_mass.clamp_min(1e-12)),
            "range_offset": range_offset,
            "doppler_vector": doppler_vector_by_range,
            "candidate_probability": candidate_probability,
            "candidate_logits": candidate_logits,
            "candidate_logits_base": candidate_logits_base.flatten(start_dim=1),
            "parameter_feedback": parameter_feedback.flatten(start_dim=1),
            "null_logit": null_logit,
            "candidate_range": candidate_range,
            "candidate_doppler_vector": candidate_doppler_vector,
            "candidate_doppler_hz": candidate_doppler_hz,
            "range_cell_estimate": pooled_parameters["range_cell"],
            "doppler_vector_pooled": pooled_parameters["doppler_vector"],
            "nonnull_probability": pooled_parameters["nonnull_probability"],
        }


def build_r6d_student(
    arm: str, *, width: int = R6D_DEFAULT_WIDTH, seed: int | None = None, prf_hz: float = 1300.0
) -> nn.Module:
    if arm not in R6D_ARMS:
        raise ValueError(f"unknown R6-D arm {arm!r}")
    if seed is None:
        return R6J2Student(width=width, prf_hz=prf_hz) if arm == "j2_joint" else R6J1Student(width=width)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return R6J2Student(width=width, prf_hz=prf_hz) if arm == "j2_joint" else R6J1Student(width=width)


def count_r6d_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"parameters": int(total), "trainable_parameters": int(trainable)}
