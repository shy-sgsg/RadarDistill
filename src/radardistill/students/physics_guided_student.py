"""E003 task student with a fixed coherence front end and small projection."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from radardistill.students.physical_coherence import FixedCoherenceFeatures
from radardistill.students.task_student import TaskStudent


class PhysicsGuidedTaskStudent(nn.Module):
    """Six-plane IQ input → fixed 28 features → 1×1 28→6 → TaskStudent."""

    def __init__(
        self,
        *,
        width: int = 32,
        temporal_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        include_spectrum_head: bool = True,
    ) -> None:
        super().__init__()
        self.frontend = FixedCoherenceFeatures()
        self.projection = nn.Conv2d(28, 6, kernel_size=1, bias=True)
        self.student = TaskStudent(
            width=width,
            temporal_dilations=temporal_dilations,
            include_spectrum_head=include_spectrum_head,
        )

    @staticmethod
    def make_raw_student(
        *,
        width: int = 32,
        temporal_dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        include_spectrum_head: bool = True,
    ) -> TaskStudent:
        return TaskStudent(
            width=width,
            temporal_dilations=temporal_dilations,
            include_spectrum_head=include_spectrum_head,
        )

    @property
    def shared_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.student.parameters())

    @property
    def projection_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.projection.parameters())

    @property
    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, iq_ri: torch.Tensor, *, compute_spectrum: bool = True
    ) -> dict[str, torch.Tensor]:
        if (
            iq_ri.ndim != 4
            or tuple(iq_ri.shape[1:]) != (6, 130, 32)
            or not torch.is_floating_point(iq_ri)
            or not torch.isfinite(iq_ri).all()
        ):
            raise ValueError("iq_ri must be finite real/imag planes with shape [B,6,130,32]")
        iq = torch.complex(iq_ri[:, :3], iq_ri[:, 3:])
        features = self.frontend(iq)
        projected = self.projection(features)
        return self.student(projected, compute_spectrum=compute_spectrum)

