"""D2-A-E002 truth-task and H1-only K2 response losses."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from .task_losses import task_loss


def masked_k2_kl(
    student_logits: torch.Tensor,
    teacher_response: torch.Tensor,
    truth_presence: torch.Tensor,
) -> torch.Tensor:
    """Mean ``KL(q_teacher || p_student)`` over H1 rows only."""
    if student_logits.ndim != 3 or student_logits.shape[1:] != (130, 32):
        raise ValueError("student_logits must have shape [B,130,32]")
    if teacher_response.shape != student_logits.shape:
        raise ValueError("teacher_response must match student logits [B,130,32]")
    mask = torch.as_tensor(truth_presence, device=student_logits.device).reshape(-1).bool()
    if mask.shape != (student_logits.shape[0],):
        raise ValueError("truth_presence must have one value per sample")
    if not torch.is_floating_point(teacher_response) or not torch.isfinite(teacher_response).all():
        raise ValueError("teacher_response must contain finite floating-point values")
    if torch.any(teacher_response < 0.0):
        raise ValueError("teacher_response must be nonnegative")
    if not torch.isfinite(student_logits).all():
        raise ValueError("student_logits must be finite")
    if not torch.any(mask):
        return student_logits.sum() * 0.0

    teacher = teacher_response.flatten(start_dim=1)
    normalizers = teacher.sum(dim=-1, keepdim=True)
    if torch.any(normalizers[mask] <= 0.0):
        raise ValueError("each H1 teacher map must have positive total mass")
    teacher = teacher / normalizers.clamp_min(torch.finfo(teacher.dtype).tiny)
    teacher = teacher.clamp_min(1e-12)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True)
    log_teacher = torch.log(teacher)
    log_student = F.log_softmax(student_logits.flatten(start_dim=1), dim=-1)
    per_sample = torch.sum(teacher * (log_teacher - log_student), dim=-1)
    return per_sample[mask].mean()


def d2a_e002_loss(
    outputs: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    teacher_response: torch.Tensor | None,
    teacher_weight: float,
) -> dict[str, torch.Tensor]:
    """Apply frozen physical task loss plus an optional H1-only K2 loss."""
    if teacher_weight < 0.0 or not torch.isfinite(torch.tensor(float(teacher_weight))):
        raise ValueError("teacher_weight must be finite and nonnegative")
    terms = task_loss(outputs, truth, config, teacher_weight=0.0)
    if teacher_response is None:
        kl = outputs["presence_logit"].sum() * 0.0
    else:
        if "spectrum_logits" not in outputs:
            raise ValueError("K2 loss requires the training-only student spectrum head")
        kl = masked_k2_kl(
            outputs["spectrum_logits"],
            teacher_response,
            truth["truth_presence"],
        )
    total = terms["task_total"] + float(teacher_weight) * kl
    return {**terms, "teacher_kl": kl, "total": total}
