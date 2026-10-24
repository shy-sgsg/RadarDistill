"""PyTorch models and physical feature maps for bounded P0 experiments."""

from __future__ import annotations

import math

import torch
from torch import nn


def _complex_batch(iq: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(iq)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2 or not torch.is_complex(values):
        raise ValueError("iq must be a complex tensor with shape (N,) or (batch, N)")
    if values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("iq must contain a nonempty batch of signals with at least two samples")
    return values.to(torch.complex64)


def correlation_features(iq: torch.Tensor, lags: tuple[int, ...] = (1, 2, 4, 8)) -> torch.Tensor:
    """Normalized complex autocorrelations at fixed lags, as real/imag channels.

    For each lag, the mean product is divided by mean input power. Therefore a
    common phase rotation and a nonzero global complex scale cancel exactly,
    while lag phase continues to encode Doppler.
    """
    values = _complex_batch(iq)
    if not lags or len(set(lags)) != len(lags) or any(lag <= 0 for lag in lags):
        raise ValueError("lags must be distinct positive integers")
    if any(lag >= values.shape[-1] for lag in lags):
        raise ValueError("each lag must be smaller than the number of samples")
    power = values.abs().square().mean(dim=-1).clamp_min(torch.finfo(torch.float32).tiny)
    correlations = []
    for lag in lags:
        products = values[:, lag:] * values[:, :-lag].conj()
        correlations.append(products.mean(dim=-1) / power)
    stacked = torch.stack(correlations, dim=-1)
    return torch.stack((stacked.real, stacked.imag), dim=-1).flatten(start_dim=1).float()


def frequency_response_features(iq: torch.Tensor, n_fft: int = 520) -> torch.Tensor:
    """Peak-normalized log-power frequency response from a fixed zero-padded FFT."""
    values = _complex_batch(iq)
    if n_fft < values.shape[-1] or n_fft % 2:
        raise ValueError("n_fft must be even and at least the signal length")
    spectrum = torch.fft.fftshift(torch.fft.fft(values, n=n_fft, dim=-1), dim=-1)
    log_power = torch.log(spectrum.abs().square().clamp_min(torch.finfo(torch.float32).tiny))
    return (log_power - log_power.amax(dim=-1, keepdim=True)).float()


def spectral_coherence_features(iq: torch.Tensor, n_fft: int = 2080) -> torch.Tensor:
    """Fixed complex FFT features with phase-invariant adjacent-bin coherence.

    The output channels are peak-normalized log power and the real/imaginary
    parts of X[k+1] conj(X[k]) normalized by the two bin magnitudes. A common
    phase rotation cancels from the coherence term; a modulation by one FFT
    frequency step circularly rolls all three channels by one bin.
    """
    values = _complex_batch(iq)
    if n_fft < values.shape[-1] or n_fft % 2:
        raise ValueError("n_fft must be even and at least the signal length")
    spectrum = torch.fft.fftshift(torch.fft.fft(values, n=n_fft, dim=-1), dim=-1)
    power = spectrum.abs().square()
    log_power = torch.log(power.clamp_min(torch.finfo(torch.float32).tiny))
    log_power = log_power - log_power.amax(dim=-1, keepdim=True)
    following = torch.roll(spectrum, shifts=-1, dims=-1)
    denominator = (spectrum.abs() * following.abs()).clamp_min(torch.finfo(torch.float32).tiny)
    coherence = following * spectrum.conj() / denominator
    return torch.stack((log_power, coherence.real, coherence.imag), dim=1).float()


def local_spectrum_features(
    iq: torch.Tensor, n_fft: int = 520, window_bins: int = 9
) -> torch.Tensor:
    """Build a phase-invariant local spectrum vector centered on its FFT peak.

    For an odd window of W bins, the output concatenates W normalized log-power
    values, W relative-phase real values, and W relative-phase imaginary values.
    The representation is invariant to common complex scaling and to a circular
    shift of the whole spectrum because the window is peak-centered.
    """
    values = _complex_batch(iq)
    if n_fft < values.shape[-1] or n_fft % 2:
        raise ValueError("n_fft must be even and at least the signal length")
    if window_bins < 3 or window_bins % 2 == 0 or window_bins > n_fft:
        raise ValueError("window_bins must be an odd integer between 3 and n_fft")
    spectrum = torch.fft.fftshift(torch.fft.fft(values, n=n_fft, dim=-1), dim=-1)
    power = spectrum.abs().square()
    peak = power.argmax(dim=-1, keepdim=True)
    radius = window_bins // 2
    offsets = torch.arange(-radius, radius + 1, device=values.device)
    indices = torch.remainder(peak + offsets.unsqueeze(0), n_fft)
    local = torch.gather(spectrum, dim=-1, index=indices)
    local_power = local.abs().square()
    local_log_power = torch.log(local_power.clamp_min(torch.finfo(torch.float32).tiny))
    local_log_power = local_log_power - local_log_power.amax(dim=-1, keepdim=True)
    center = local[:, radius : radius + 1]
    denominator = (local.abs() * center.abs()).clamp_min(torch.finfo(torch.float32).tiny)
    relative = local * center.conj() / denominator
    return torch.cat((local_log_power, relative.real, relative.imag), dim=-1).float()


def circular_frequency_targets(
    frequency_hz: torch.Tensor | float,
    prf_hz: float,
    n_bins: int,
    sigma_bins: float = 0.5,
) -> torch.Tensor:
    """Return normalized wrapped-Gaussian frequency targets on a circular grid."""
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if n_bins < 3 or not math.isfinite(sigma_bins) or sigma_bins <= 0:
        raise ValueError("n_bins must be at least three and sigma_bins positive")
    frequencies = torch.as_tensor(frequency_hz, dtype=torch.float32)
    if frequencies.ndim == 0:
        frequencies = frequencies.unsqueeze(0)
    if frequencies.ndim != 1 or not torch.isfinite(frequencies).all():
        raise ValueError("frequency_hz must be a finite scalar or vector")
    wrapped = torch.remainder(frequencies + prf_hz / 2.0, prf_hz) - prf_hz / 2.0
    spacing = prf_hz / n_bins
    centers = -prf_hz / 2.0 + torch.arange(
        n_bins, dtype=torch.float32, device=frequencies.device
    ) * spacing
    distance = torch.remainder(
        wrapped[:, None] - centers[None, :] + prf_hz / 2.0, prf_hz
    ) - prf_hz / 2.0
    sigma_hz = sigma_bins * spacing
    targets = torch.exp(-0.5 * (distance / sigma_hz).square())
    return targets / targets.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)


def decode_circular_logits(logits: torch.Tensor, prf_hz: float) -> torch.Tensor:
    """Decode circular-grid logits using the circular probability mean."""
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if logits.ndim < 1 or logits.shape[-1] < 3:
        raise ValueError("logits must have at least three circular bins")
    probabilities = torch.softmax(logits.float(), dim=-1)
    n_bins = logits.shape[-1]
    centers = -prf_hz / 2.0 + torch.arange(
        n_bins, dtype=probabilities.dtype, device=probabilities.device
    ) * (prf_hz / n_bins)
    angles = centers * (2.0 * math.pi / prf_hz)
    cosine = torch.sum(probabilities * torch.cos(angles), dim=-1)
    sine = torch.sum(probabilities * torch.sin(angles), dim=-1)
    frequency = torch.atan2(sine, cosine) * (prf_hz / (2.0 * math.pi))
    return torch.remainder(frequency + prf_hz / 2.0, prf_hz) - prf_hz / 2.0


class CircularSpectralNet(nn.Module):
    """Small circular-convolution readout over fixed physical spectral features."""

    def __init__(
        self,
        input_channels: int = 3,
        hidden_channels: tuple[int, ...] = (16, 16, 16),
        kernel_size: int = 7,
    ):
        super().__init__()
        if input_channels <= 0 or not hidden_channels or any(width <= 0 for width in hidden_channels):
            raise ValueError("input_channels and every hidden channel width must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        layers: list[nn.Module] = []
        in_channels = input_channels
        for out_channels in hidden_channels:
            layers.append(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    padding_mode="circular",
                )
            )
            layers.append(nn.GELU())
            in_channels = out_channels
        layers.append(nn.Conv1d(in_channels, 1, kernel_size=1))
        self.input_channels = int(input_channels)
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != self.input_channels:
            raise ValueError(
                f"features must have shape (batch, {self.input_channels}, frequency_bins)"
            )
        if features.shape[0] == 0 or features.shape[-1] < 3:
            raise ValueError("features must have a nonempty batch and at least three frequency bins")
        return self.network(features.float()).squeeze(1)


class FFTSubbinCorrector(nn.Module):
    """MLP that predicts a bounded residual around a fixed FFT coarse peak."""

    def __init__(
        self,
        input_features: int = 27,
        hidden_units: tuple[int, int] = (64, 32),
        half_bin: float = 0.5,
    ):
        super().__init__()
        if input_features <= 0 or len(hidden_units) != 2 or any(width <= 0 for width in hidden_units):
            raise ValueError("input_features and both hidden widths must be positive")
        if not math.isfinite(half_bin) or half_bin <= 0:
            raise ValueError("half_bin must be positive and finite")
        self.trunk = nn.Sequential(
            nn.Linear(input_features, hidden_units[0]),
            nn.GELU(),
            nn.Linear(hidden_units[0], hidden_units[1]),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden_units[1], 1)
        self.half_bin = float(half_bin)
        self.input_features = int(input_features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[-1] != self.input_features:
            raise ValueError(
                f"features must have shape (batch, {self.input_features})"
            )
        return torch.tanh(self.head(self.trunk(features.float())).squeeze(-1)) * self.half_bin


def _sinusoidal_position_encoding(length: int, d_model: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros((length, d_model), dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class IQTransformer(nn.Module):
    """Compact encoder that maps RMS-normalized I/Q slow-time tokens to a circular grid."""

    def __init__(
        self,
        n_samples: int = 130,
        output_bins: int = 2080,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        feedforward_dim: int = 64,
        dropout: float = 0.05,
    ):
        super().__init__()
        if (
            n_samples < 2
            or output_bins < 3
            or d_model <= 0
            or n_heads <= 0
            or d_model % n_heads != 0
            or n_layers <= 0
            or feedforward_dim <= 0
            or not 0.0 <= dropout < 1.0
        ):
            raise ValueError("invalid IQ Transformer dimensions or dropout")
        self.n_samples = int(n_samples)
        self.input_projection = nn.Linear(2, d_model)
        self.pool_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.pool_token, mean=0.0, std=0.02)
        self.register_buffer(
            "position_encoding",
            _sinusoidal_position_encoding(n_samples + 1, d_model),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.frequency_head = nn.Linear(d_model, output_bins)

    def forward(self, iq_tokens: torch.Tensor) -> torch.Tensor:
        if iq_tokens.ndim != 3 or iq_tokens.shape[1:] != (self.n_samples, 2):
            raise ValueError(
                f"iq_tokens must have shape (batch, {self.n_samples}, 2)"
            )
        values = iq_tokens.float()
        if values.shape[0] == 0:
            raise ValueError("iq_tokens must contain a nonempty batch")
        rms = values.square().sum(dim=-1).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        values = values / rms.unsqueeze(-1)
        encoded = self.input_projection(values)
        pool = self.pool_token.expand(values.shape[0], -1, -1)
        sequence = torch.cat((pool, encoded), dim=1)
        sequence = sequence + self.position_encoding.to(dtype=sequence.dtype)
        pooled = self.encoder(sequence)[:, 0]
        return self.frequency_head(self.output_norm(pooled))


class CorrelationMLP(nn.Module):
    """Small MLP over four complex lag-correlation features."""

    def __init__(self, hidden_units: int = 32, response_bins: int = 65):
        super().__init__()
        if hidden_units <= 0 or response_bins < 3:
            raise ValueError("hidden_units must be positive and response_bins at least three")
        self.lags = (1, 2, 4, 8)
        self.response_bins = int(response_bins)
        self.trunk = nn.Sequential(
            nn.Linear(2 * len(self.lags), hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.GELU(),
        )
        self.frequency_head = nn.Linear(hidden_units, 2)
        self.response_head = nn.Linear(hidden_units, response_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(features.float())
        return torch.cat((self.frequency_head(hidden), self.response_head(hidden)), dim=-1)


class FrequencyResponseMLP(nn.Module):
    """Lightweight frequency-response-guided hybrid student."""

    def __init__(self, n_fft: int = 520, hidden_units: int = 64, response_bins: int = 65):
        super().__init__()
        if n_fft < 2 or n_fft % 2 or hidden_units <= 0 or response_bins < 3:
            raise ValueError("invalid FFT length, hidden width, or response output size")
        self.n_fft = int(n_fft)
        self.response_bins = int(response_bins)
        self.trunk = nn.Sequential(
            nn.Linear(n_fft, hidden_units),
            nn.GELU(),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.GELU(),
        )
        head_units = hidden_units // 2
        self.frequency_head = nn.Linear(head_units, 2)
        self.response_head = nn.Linear(head_units, response_bins)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(features.float())
        return torch.cat((self.frequency_head(hidden), self.response_head(hidden)), dim=-1)


class FrozenE005CNN(nn.Module):
    """PyTorch inference port of the frozen NumPy P0-E005 mean-pooling CNN."""

    def __init__(self, channels: int = 6, kernel_size: int = 9):
        super().__init__()
        self.conv = nn.Conv1d(2, channels, kernel_size=kernel_size, bias=True)
        self.head = nn.Linear(channels, 2, bias=True)

    def load_numpy_checkpoint(self, path: str, device: torch.device | str = "cpu") -> None:
        import numpy as np

        with np.load(path) as weights, torch.no_grad():
            self.conv.weight.copy_(torch.as_tensor(weights["conv_weight"], dtype=torch.float32, device=device))
            self.conv.bias.copy_(torch.as_tensor(weights["conv_bias"], dtype=torch.float32, device=device))
            self.head.weight.copy_(torch.as_tensor(weights["head_weight"], dtype=torch.float32, device=device))
            self.head.bias.copy_(torch.as_tensor(weights["head_bias"], dtype=torch.float32, device=device))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        activations = torch.relu(self.conv(features.float()))
        pooled = activations.mean(dim=-1)
        return self.head(pooled)


def output_to_frequency(output: torch.Tensor, prf_hz: float) -> torch.Tensor:
    """Decode the first two circular outputs to the half-open Doppler interval."""
    if not math.isfinite(prf_hz) or prf_hz <= 0 or output.shape[-1] < 2:
        raise ValueError("positive finite PRF and two circular output values are required")
    angle = torch.atan2(output[..., 1], output[..., 0])
    frequency = angle * (prf_hz / (2.0 * math.pi))
    return torch.remainder(frequency + prf_hz / 2.0, prf_hz) - prf_hz / 2.0
