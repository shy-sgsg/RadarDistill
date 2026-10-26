"""Fixed multi-delay and cross-channel coherence features for D2-A-E003."""

from __future__ import annotations

import math

import torch
from torch import nn


class FixedCoherenceFeatures(nn.Module):
    """Map complex three-channel IQ to the frozen 28-channel feature set.

    Channel order is real parts of nine same-channel correlations followed by
    three cross-channel correlations; the corresponding imaginary parts; three
    normalized channel powers; and broadcast log scene power.
    """

    lags = (1, 4, 16)
    cross_pairs = ((0, 1), (0, 2), (1, 2))
    pulse_count = 130
    range_count = 32
    feature_count = 28

    def __init__(self, *, epsilon_power: float = 1e-12) -> None:
        super().__init__()
        if not math.isfinite(epsilon_power) or epsilon_power <= 0.0:
            raise ValueError("epsilon_power must be positive and finite")
        self.epsilon_power = float(epsilon_power)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        if (
            iq.ndim != 4
            or iq.shape[0] == 0
            or tuple(iq.shape[1:]) != (3, self.pulse_count, self.range_count)
        ):
            raise ValueError("iq must have shape [B,3,130,32] with a nonempty batch")
        if not iq.is_complex() or iq.dtype not in (torch.complex64, torch.complex128):
            raise ValueError("iq must use complex64 or complex128")
        if not torch.isfinite(iq).all():
            raise ValueError("iq must contain only finite complex values")

        real_dtype = iq.real.dtype
        scale = iq.abs().amax(dim=(1, 2, 3), keepdim=True)
        nonzero = scale > 0
        safe_scale = torch.where(nonzero, scale, torch.ones_like(scale))
        normalized_iq = iq / safe_scale
        normalized_power = normalized_iq.real.square() + normalized_iq.imag.square()
        mean_normalized_power = normalized_power.mean(dim=(1, 2, 3), keepdim=True)

        log_scale = torch.log(scale.to(torch.float64).clamp_min(torch.finfo(real_dtype).tiny))
        log_mean_power = torch.log(mean_normalized_power.to(torch.float64).clamp_min(torch.finfo(real_dtype).tiny))
        log_actual_power = 2.0 * log_scale + log_mean_power
        negative_infinity = torch.full_like(log_actual_power, -torch.inf)
        log_actual_power = torch.where(nonzero, log_actual_power, negative_infinity)
        log_epsilon = math.log(self.epsilon_power)
        use_epsilon_floor = log_actual_power < log_epsilon
        log_scene_power = torch.maximum(log_actual_power, torch.full_like(log_actual_power, log_epsilon))

        # Scaling before products avoids overflow. The epsilon branch rescales
        # the normalized products by |scale|^2/epsilon without forming |X|^2.
        epsilon_log_factor = torch.where(
            use_epsilon_floor,
            2.0 * log_scale - log_epsilon,
            torch.zeros_like(log_scale),
        )
        epsilon_factor = torch.exp(epsilon_log_factor).to(real_dtype)
        denominator = mean_normalized_power.clamp_min(torch.finfo(real_dtype).tiny)
        denominator_corr = denominator[:, 0]
        epsilon_factor_corr = epsilon_factor[:, 0]
        use_floor_corr = use_epsilon_floor[:, 0]

        def normalize_product(product: torch.Tensor) -> torch.Tensor:
            regular = product / denominator_corr
            floored = product * epsilon_factor_corr
            return torch.where(use_floor_corr, floored, regular)

        correlations: list[torch.Tensor] = []
        for channel in range(3):
            for lag in self.lags:
                shifted = torch.zeros_like(normalized_iq[:, channel])
                shifted[:, :-lag, :] = normalized_iq[:, channel, lag:, :]
                product = shifted * normalized_iq[:, channel].conj()
                correlations.append(normalize_product(product))
        for first, second in self.cross_pairs:
            product = normalized_iq[:, first] * normalized_iq[:, second].conj()
            correlations.append(normalize_product(product))

        correlation_stack = torch.stack(correlations, dim=1)
        powers = torch.where(
            use_epsilon_floor,
            normalized_power * epsilon_factor,
            normalized_power / denominator,
        )
        scene_scale = log_scene_power.to(real_dtype).expand(-1, 1, self.pulse_count, self.range_count)
        features = torch.cat(
            (
                correlation_stack.real,
                correlation_stack.imag,
                powers,
                scene_scale,
            ),
            dim=1,
        )
        if not torch.isfinite(features).all():
            raise FloatingPointError("coherence features became non-finite")
        return features

