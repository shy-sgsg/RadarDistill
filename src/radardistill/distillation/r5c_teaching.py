"""Response targets and losses for the D2-A-R5-C teaching pilot."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


_FREQUENCY_BINS = 520
_RANGE_CELLS = 32


def _validate_response_map(teacher_map: torch.Tensor) -> torch.Tensor:
    if not isinstance(teacher_map, torch.Tensor):
        raise ValueError("teacher_map must be a torch.Tensor")
    if teacher_map.ndim != 3 or teacher_map.shape[0] <= 0 or teacher_map.shape[1:] != (
        _FREQUENCY_BINS,
        _RANGE_CELLS,
    ):
        raise ValueError("teacher_map must have shape [B,520,32]")
    if not torch.is_floating_point(teacher_map):
        raise ValueError("teacher_map must have a floating-point dtype")
    if not bool(torch.isfinite(teacher_map).all()):
        raise ValueError("teacher_map must contain only finite values")
    if bool(torch.any(teacher_map < 0.0)):
        raise ValueError("teacher_map must be nonnegative")
    totals = teacher_map.sum(dim=(1, 2))
    if not bool(torch.isfinite(totals).all()) or bool(torch.any(totals <= 0.0)):
        raise ValueError("each teacher map must have positive total mass")
    return teacher_map.detach()


def k2_local_targets(teacher_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the IQ-only global peak's range and normalize its 520-bin slice.

    ``teacher_map`` is the observable-IQ K3 response with shape ``[B,520,32]``.
    Range selection uses the maximum cell of that map and receives no truth
    inputs.  The returned conditional response has shape ``[B,520]``.
    """
    response = _validate_response_map(teacher_map)
    batch = response.shape[0]
    peak = response.reshape(batch, -1).argmax(dim=1)
    range_index = torch.remainder(peak, _RANGE_CELLS)
    rows = torch.arange(batch, device=response.device)
    local = response[rows, :, range_index]
    totals = local.sum(dim=1, keepdim=True)
    if bool(torch.any(totals <= 0.0)):
        raise ValueError("each teacher-selected local response must have positive total mass")
    return range_index, local / totals


def truth_dirichlet_targets(
    range_cell: torch.Tensor,
    doppler_hz: torch.Tensor,
    *,
    prf_hz: float = 1300.0,
    pulse_count: int = 130,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return truth range cells and normalized rectangular CPI power responses.

    The response samples the exact finite coherent sum on the centered 520-bin
    axis (2.5 Hz spacing for the frozen 1300 Hz PRF). Frequencies are wrapped
    modulo PRF before evaluating the sum, preserving the Nyquist boundary.
    """
    if not isinstance(range_cell, torch.Tensor) or not isinstance(doppler_hz, torch.Tensor):
        raise ValueError("range_cell and doppler_hz must be torch.Tensors")
    if range_cell.ndim != 1 or range_cell.shape[0] <= 0 or doppler_hz.shape != range_cell.shape:
        raise ValueError("range_cell and doppler_hz must have the same nonempty [B] shape")
    if not torch.is_floating_point(range_cell) or not torch.is_floating_point(doppler_hz):
        raise ValueError("range_cell and doppler_hz must have floating-point dtypes")
    if not math.isfinite(float(prf_hz)) or float(prf_hz) <= 0.0:
        raise ValueError("prf_hz must be positive and finite")
    if isinstance(pulse_count, bool) or not isinstance(pulse_count, int) or pulse_count <= 0:
        raise ValueError("pulse_count must be a positive integer")
    if not bool(torch.isfinite(range_cell).all()) or not bool(torch.isfinite(doppler_hz).all()):
        raise ValueError("range_cell and doppler_hz must be finite")
    if bool(torch.any(range_cell < 0.0)) or bool(torch.any(range_cell >= _RANGE_CELLS)):
        raise ValueError("range_cell must be in [0,32)")

    # Compute labels in float64/complex128 even when student training is FP32.
    # The finite sum is short (130 pulses) and avoids a removable-singularity
    # problem at integer-PRF offsets in a closed-form sinc ratio.
    device = doppler_hz.device
    dtype = doppler_hz.dtype
    prf = float(prf_hz)
    with torch.no_grad():
        doppler = doppler_hz.to(device=device, dtype=torch.float64)
        wrapped = torch.remainder(doppler + prf / 2.0, prf) - prf / 2.0
        bins = (torch.arange(_FREQUENCY_BINS, device=device, dtype=torch.float64) - 260.0)
        bins = bins * (prf / _FREQUENCY_BINS)
        delta = wrapped[:, None] - bins[None, :]
        pulses = torch.arange(pulse_count, device=device, dtype=torch.float64)
        phase = (2.0 * math.pi / prf) * delta[:, :, None] * pulses[None, None, :]
        coherent_sum = torch.exp((1j * phase)).sum(dim=-1)
        power = coherent_sum.abs().square()
        totals = power.sum(dim=1, keepdim=True)
        if not bool(torch.isfinite(power).all()) or bool(torch.any(totals <= 0.0)):
            raise FloatingPointError("truth Dirichlet response is invalid")
        target = power / totals
        indices = torch.floor(range_cell).to(torch.long)
    return indices, target.to(dtype=dtype)


def _categorical_kl(student_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.to(device=student_logits.device, dtype=student_logits.dtype)
    target = target / target.sum(dim=-1, keepdim=True)
    target = target.clamp_min(1e-12)
    target = target / target.sum(dim=-1, keepdim=True)
    log_target = torch.log(target)
    log_student = F.log_softmax(student_logits, dim=-1)
    return torch.sum(target * (log_target - log_student), dim=-1).mean()


def response_kl(
    student_logits: torch.Tensor,
    arm: str,
    *,
    teacher_map: torch.Tensor | None = None,
    truth_range_cell: torch.Tensor | None = None,
    truth_doppler_hz: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the arm-specific response KL; callers pass H1 rows only.

    T1 matches the complete normalized RD support map. T2 embeds its
    teacher-selected conditional frequency response at the selected range
    gate; T3 embeds the truth-selected analytic rectangular-DFT response at
    the truth range gate. T2/T3 then use the same flattened joint-map KL as
    T1. T0 returns a differentiable zero. Truth values are rejected on
    teacher arms so they cannot affect teacher target selection.
    """
    if not isinstance(student_logits, torch.Tensor):
        raise ValueError("student_logits must be a torch.Tensor")
    if student_logits.ndim != 3 or student_logits.shape[0] <= 0 or student_logits.shape[1:] != (
        _FREQUENCY_BINS,
        _RANGE_CELLS,
    ):
        raise ValueError("student_logits must have shape [B,520,32]")
    if not torch.is_floating_point(student_logits) or not bool(torch.isfinite(student_logits).all()):
        raise ValueError("student_logits must be finite floating-point values")
    if not isinstance(arm, str) or arm not in {"T0", "T1", "T2", "T3"}:
        raise ValueError("arm must be one of T0, T1, T2, or T3")

    truth_given = truth_range_cell is not None or truth_doppler_hz is not None
    if arm == "T0":
        if teacher_map is not None or truth_given:
            raise ValueError("T0 does not accept auxiliary supervision inputs")
        return student_logits.sum() * 0.0

    if arm in {"T1", "T2"}:
        if teacher_map is None:
            raise ValueError(f"{arm} requires teacher_map")
        if truth_given:
            raise ValueError(f"{arm} does not accept truth inputs")
        if not isinstance(teacher_map, torch.Tensor):
            raise ValueError("teacher_map must be a torch.Tensor")
        if teacher_map.shape[0] != student_logits.shape[0]:
            raise ValueError("teacher_map batch size must match student_logits")
        response = _validate_response_map(teacher_map.to(device=student_logits.device))
        if arm == "T1":
            target = response.reshape(response.shape[0], -1)
            return _categorical_kl(student_logits.reshape(student_logits.shape[0], -1), target)
        range_index, target = k2_local_targets(response)
    else:
        if teacher_map is not None:
            raise ValueError("T3 does not accept teacher_map")
        if truth_range_cell is None or truth_doppler_hz is None:
            raise ValueError("T3 requires truth_range_cell and truth_doppler_hz")
        if not isinstance(truth_range_cell, torch.Tensor) or not isinstance(
            truth_doppler_hz, torch.Tensor
        ):
            raise ValueError("T3 truth inputs must be torch.Tensors")
        if truth_range_cell.shape[0] != student_logits.shape[0]:
            raise ValueError("truth batch size must match student_logits")
        range_index, target = truth_dirichlet_targets(truth_range_cell, truth_doppler_hz)

    range_index = range_index.to(device=student_logits.device)
    target = target.to(device=student_logits.device)
    rows = torch.arange(student_logits.shape[0], device=student_logits.device)
    joint_target = torch.zeros_like(student_logits)
    joint_target[rows, :, range_index] = target
    return _categorical_kl(
        student_logits.reshape(student_logits.shape[0], -1),
        joint_target.reshape(joint_target.shape[0], -1),
    )
