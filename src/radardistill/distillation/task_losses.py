"""D1 detection, range, circular Doppler, and teacher response losses."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F


def encode_doppler_hz(frequency_hz: torch.Tensor, prf_hz: float) -> torch.Tensor:
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    phase = frequency_hz * (2.0 * math.pi / prf_hz)
    return torch.stack((torch.cos(phase), torch.sin(phase)), dim=-1)


def circular_doppler_error(
    estimate_hz: torch.Tensor, truth_hz: torch.Tensor, prf_hz: float
) -> torch.Tensor:
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    return torch.remainder(estimate_hz - truth_hz + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


def doppler_circular_loss(
    prediction_vector: torch.Tensor, target_vector: torch.Tensor
) -> torch.Tensor:
    if prediction_vector.shape != target_vector.shape or prediction_vector.shape[-1] != 2:
        raise ValueError("Doppler vectors must have equal shape ending in cos/sin dimension 2")
    predicted = F.normalize(prediction_vector, p=2.0, dim=-1, eps=1e-6)
    target = F.normalize(target_vector, p=2.0, dim=-1, eps=1e-6)
    dot = torch.sum(predicted * target, dim=-1).clamp(-1.0, 1.0)
    return 1.0 - dot


def decode_task_outputs(outputs: dict[str, torch.Tensor], prf_hz: float) -> dict[str, torch.Tensor]:
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    cell = outputs["range_logits"].argmax(dim=-1)
    row = torch.arange(cell.shape[0], device=cell.device)
    offset = outputs["range_offset"][row, cell]
    vector = outputs["doppler_vector"][row, cell]
    angle = torch.atan2(vector[:, 1], vector[:, 0])
    doppler_hz = torch.remainder(angle * (prf_hz / (2.0 * math.pi)) + prf_hz / 2.0, prf_hz) - prf_hz / 2.0
    return {
        "presence_probability": torch.sigmoid(outputs["presence_logit"]),
        "range_cell": cell.to(offset.dtype) + offset,
        "doppler_hz": doppler_hz,
    }


def task_loss(
    outputs: dict[str, torch.Tensor],
    truth: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    task_weight: float = 1.0,
    teacher_weight: float | None = None,
) -> dict[str, torch.Tensor]:
    """Return separate truth-task and whole-map teacher losses.

    H0 records contribute to detection and, when present, the full teacher map
    KL only. Parameter losses are masked to H1 records.
    """
    radar = config.get("radar", {})
    loss_config = config.get("training", {}).get("loss", config.get("loss", {}))
    prf_hz = float(radar.get("prf_hz", 1300.0))
    if not math.isfinite(task_weight) or task_weight < 0:
        raise ValueError("task_weight must be finite and non-negative")
    if teacher_weight is None:
        teacher_weight = float(loss_config.get("teacher_response_kl_weight", 0.05))
    if not math.isfinite(teacher_weight) or teacher_weight < 0:
        raise ValueError("teacher_weight must be finite and non-negative")

    presence_truth = truth["truth_presence"].to(
        device=outputs["presence_logit"].device, dtype=outputs["presence_logit"].dtype
    )
    if presence_truth.shape != outputs["presence_logit"].shape:
        raise ValueError("truth_presence must have one label per sample")
    presence = F.binary_cross_entropy_with_logits(outputs["presence_logit"], presence_truth)
    h1 = presence_truth > 0.5
    range_truth = truth["truth_range_cell"].to(outputs["range_logits"].device)
    doppler_truth = truth["truth_doppler_hz"].to(outputs["range_logits"].device)
    if range_truth.shape != presence_truth.shape or doppler_truth.shape != presence_truth.shape:
        raise ValueError("range and Doppler truths must have one value per sample")

    if torch.any(h1):
        range_indices = torch.floor(range_truth[h1]).to(torch.long).clamp(0, outputs["range_logits"].shape[1] - 1)
        range_targets = range_truth[h1] - range_indices.to(range_truth.dtype)
        range_cell = F.cross_entropy(outputs["range_logits"][h1], range_indices)
        rows = torch.nonzero(h1, as_tuple=False).squeeze(-1)
        predicted_offset = outputs["range_offset"][rows, range_indices]
        beta = float(loss_config.get("range_offset_smooth_l1_beta", 0.1))
        range_offset = F.smooth_l1_loss(predicted_offset, range_targets, beta=beta)
        target_vector = encode_doppler_hz(doppler_truth[h1], prf_hz)
        predicted_vector = outputs["doppler_vector"][rows, range_indices]
        doppler = doppler_circular_loss(predicted_vector, target_vector).mean()
    else:
        zero = outputs["range_logits"].sum() * 0.0
        range_cell = zero
        range_offset = outputs["range_offset"].sum() * 0.0
        doppler = outputs["doppler_vector"].sum() * 0.0

    teacher_response = truth.get("teacher_response")
    if teacher_response is not None and "spectrum_logits" in outputs:
        response = teacher_response.to(
            device=outputs["spectrum_logits"].device,
            dtype=outputs["spectrum_logits"].dtype,
        )
        if response.shape != outputs["spectrum_logits"].shape:
            raise ValueError("teacher_response must match the student [B,130,32] spectrum head")
        probabilities = response.flatten(start_dim=1).clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        log_student = F.log_softmax(outputs["spectrum_logits"].flatten(start_dim=1), dim=-1)
        teacher_kl = torch.sum(
            probabilities * (torch.log(probabilities) - log_student), dim=-1
        ).mean()
    else:
        reference = outputs.get("spectrum_logits", outputs["presence_logit"])
        teacher_kl = reference.sum() * 0.0

    task_total = (
        float(loss_config.get("presence_bce_weight", 1.0)) * presence
        + float(loss_config.get("range_cell_cross_entropy_weight", 1.0)) * range_cell
        + float(loss_config.get("range_offset_smooth_l1_weight", 1.0)) * range_offset
        + float(loss_config.get("doppler_circular_cosine_weight", 1.0)) * doppler
    )
    total = task_weight * task_total + teacher_weight * teacher_kl
    return {
        "presence": presence,
        "range_cell": range_cell,
        "range_offset": range_offset,
        "doppler": doppler,
        "teacher_kl": teacher_kl,
        "task_total": task_total,
        "total": total,
    }
