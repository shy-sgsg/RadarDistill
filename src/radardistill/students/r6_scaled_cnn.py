"""Width-scaled versions of the frozen R6-C TaskStudent topology."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from radardistill.students.task_student import TaskStudent


R6C_MODEL_SPECS: dict[str, dict[str, int]] = {
    "Medium": {"width": 96, "parameters": 70_277},
    "Large": {"width": 200, "parameters": 292_005},
    "XLarge": {"width": 376, "parameters": 1_012_197},
}
R6C_TEMPORAL_DILATIONS = (1, 2, 4, 8, 16, 32, 64)


def build_r6c_student(name: str, *, seed: int | None = None) -> TaskStudent:
    """Create one width candidate without the unused training-only spectrum head."""
    if name not in R6C_MODEL_SPECS:
        raise ValueError(f"unknown R6-C model candidate: {name}")
    if seed is None:
        return TaskStudent(
            width=R6C_MODEL_SPECS[name]["width"],
            temporal_dilations=R6C_TEMPORAL_DILATIONS,
            include_spectrum_head=False,
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TaskStudent(
            width=R6C_MODEL_SPECS[name]["width"],
            temporal_dilations=R6C_TEMPORAL_DILATIONS,
            include_spectrum_head=False,
        )


def count_r6c_parameters(model: nn.Module) -> dict[str, int]:
    """Report trainable and inference parameter counts as separate quantities."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"trainable_parameters": int(trainable), "inference_parameters": int(total)}


def validate_r6c_student(model: TaskStudent, name: str) -> Mapping[str, int]:
    """Fail if a candidate silently differs from its preregistered topology/count."""
    if name not in R6C_MODEL_SPECS:
        raise ValueError(f"unknown R6-C model candidate: {name}")
    expected = R6C_MODEL_SPECS[name]
    if model.width != expected["width"] or model.temporal_dilations != R6C_TEMPORAL_DILATIONS:
        raise ValueError(f"R6-C {name} does not use the registered TaskStudent topology")
    if model.spectrum_head is not None:
        raise ValueError("R6-C candidates must omit the unused spectrum head")
    counts = count_r6c_parameters(model)
    if counts["trainable_parameters"] != expected["parameters"] or counts["inference_parameters"] != expected["parameters"]:
        raise ValueError(f"R6-C {name} parameter count differs from the registered count")
    return counts
