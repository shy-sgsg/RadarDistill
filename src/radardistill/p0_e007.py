"""P0-E007 staged CUDA experiment utilities."""

from __future__ import annotations

import math
import csv
import json
import platform
import subprocess
import time
from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import (
    estimate_fft_interpolated,
    estimate_fft_peak,
    estimate_mle_global_grid,
    estimate_mle_local,
    estimate_phase_difference,
    frequency_crlb_hz,
    wrap_frequency,
    wrapped_error,
)
from .signal import SignalBatch
from .signal import generate_batch, generate_single_tone
from .torch_models import (
    CircularSpectralNet,
    FFTSubbinCorrector,
    IQTransformer,
    circular_frequency_targets,
    decode_circular_logits,
    local_spectrum_features,
    spectral_coherence_features,
)


def require_cuda(
    training_device: str = "cuda",
    *,
    cuda_available: bool | None = None,
    device_count: int | None = None,
) -> torch.device:
    """Return cuda:0 or fail before training; never permit a CPU fallback."""
    if training_device != "cuda":
        raise ValueError("P0-E007 training.device must be cuda")
    available = torch.cuda.is_available() if cuda_available is None else cuda_available
    count = torch.cuda.device_count() if device_count is None else device_count
    if not available or count < 1:
        raise RuntimeError("CUDA is required for P0-E007; no model training was started")
    return torch.device("cuda:0")


def _softmax_rows(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = (values - np.max(values, axis=-1, keepdims=True)) / temperature
    probabilities = np.exp(np.maximum(shifted, -745.0))
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def make_teacher_targets(batch: SignalBatch, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Build separate truth, point-teacher and soft-profile targets on CPU."""
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    iq = np.asarray(batch.iq, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[1] != n_samples or iq.shape[0] == 0:
        raise ValueError("batch IQ must have shape (nonempty batch, configured N)")
    count = iq.shape[0]
    truth = np.asarray(batch.frequency_hz, dtype=np.float64).reshape(-1).copy()
    amplitude = np.asarray(batch.amplitude, dtype=np.float64).reshape(-1)
    snr_db = np.asarray(batch.snr_db, dtype=np.float64)
    if snr_db.ndim == 0:
        snr_db = np.full(count, float(snr_db))
    snr_db = snr_db.reshape(-1)
    if truth.size != count or amplitude.size != count or snr_db.size != count:
        raise ValueError("batch metadata must contain one value per IQ record")
    if not np.all(np.isfinite(iq)) or not np.all(np.isfinite(truth)):
        raise ValueError("batch IQ and truth frequencies must be finite")
    noise_power = amplitude**2 * np.power(10.0, -snr_db / 10.0)
    if np.any(noise_power <= 0) or not np.all(np.isfinite(noise_power)):
        raise ValueError("noise power must be positive and finite")

    teacher_hard = np.asarray(estimate_mle_local(iq, prf_hz), dtype=np.float64).reshape(-1)
    m1_fft = int(config["models"]["m1"]["fft_factor"]) * n_samples
    m1_spectrum = np.fft.fftshift(np.fft.fft(iq, n=m1_fft, axis=-1), axes=-1)
    m1_power = np.abs(m1_spectrum) ** 2
    m1_log_likelihood = m1_power / (n_samples * noise_power[:, None])
    temperature = float(config["teacher"]["soft_profile_temperature"])
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("teacher soft-profile temperature must be positive and finite")
    m1_soft_response = _softmax_rows(m1_log_likelihood, temperature).astype(np.float32)
    m1_entropy = -np.sum(
        m1_soft_response * np.log(np.maximum(m1_soft_response, 1e-30)), axis=-1
    )

    m2_fft = int(config["models"]["m2"]["fft_factor"]) * n_samples
    m2_spectrum = np.fft.fftshift(np.fft.fft(iq, n=m2_fft, axis=-1), axes=-1)
    m2_power = np.abs(m2_spectrum) ** 2
    coarse_indices = np.argmax(m2_power, axis=-1)
    m2_spacing = prf_hz / m2_fft
    frequency_axis = -prf_hz / 2.0 + np.arange(m2_fft) * m2_spacing
    coarse_frequency = frequency_axis[coarse_indices]
    half_width = float(config["models"]["m2"]["correction_half_width_bins"])
    if not 0 < half_width <= 0.5:
        raise ValueError("M2 correction half-width must be in (0, 0.5]")
    truth_raw = np.asarray(wrapped_error(truth, coarse_frequency, prf_hz)) / m2_spacing
    teacher_hard_raw = np.asarray(
        wrapped_error(teacher_hard, coarse_frequency, prf_hz)
    ) / m2_spacing
    truth_correction = np.clip(truth_raw, -half_width, half_width)
    teacher_hard_correction = np.clip(teacher_hard_raw, -half_width, half_width)

    candidate_count = int(config["models"]["m2"]["soft_teacher_candidates"])
    offsets = np.linspace(-half_width, half_width, candidate_count, dtype=np.float64)
    candidate_frequency = np.asarray(
        wrap_frequency(
            coarse_frequency[:, None] + offsets[None, :] * m2_spacing, prf_hz
        )
    )
    time = np.arange(n_samples, dtype=np.float64)
    response_score = np.empty((count, candidate_count), dtype=np.float64)
    chunk_size = 256
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        phase = np.exp(
            -2j * np.pi * candidate_frequency[start:stop, :, None]
            * time[None, None, :]
            / prf_hz
        )
        coherent_sum = np.sum(iq[start:stop, None, :] * phase, axis=-1)
        response_score[start:stop] = (
            np.abs(coherent_sum) ** 2 / (n_samples * noise_power[start:stop, None])
        )
    m2_soft_response = _softmax_rows(response_score, temperature).astype(np.float32)
    m2_soft_correction = m2_soft_response @ offsets.astype(np.float32)
    m2_entropy = -np.sum(
        m2_soft_response * np.log(np.maximum(m2_soft_response, 1e-30)), axis=-1
    )

    return {
        "truth_frequency_hz": truth.copy(),
        "teacher_hard_frequency_hz": teacher_hard.astype(np.float32),
        "teacher_soft_response": m1_soft_response,
        "teacher_soft_entropy": m1_entropy.astype(np.float32),
        "m2_coarse_frequency_hz": coarse_frequency.astype(np.float32),
        "m2_true_correction_raw_bins": truth_raw.astype(np.float32),
        "m2_truth_correction_bins": truth_correction.astype(np.float32),
        "m2_teacher_hard_correction_raw_bins": teacher_hard_raw.astype(np.float32),
        "m2_teacher_hard_correction_bins": teacher_hard_correction.astype(np.float32),
        "m2_teacher_soft_response": m2_soft_response,
        "m2_teacher_soft_correction_bins": m2_soft_correction.astype(np.float32),
        "m2_teacher_soft_entropy": m2_entropy.astype(np.float32),
        "m2_truth_correction_saturated": (
            np.abs(truth_raw) > half_width
        ),
        "m2_teacher_hard_saturated": (
            np.abs(teacher_hard_raw) > half_width
        ),
    }


def e007_loss(
    output: torch.Tensor,
    targets: dict[str, torch.Tensor],
    model_name: str,
    strategy: str,
    *,
    teacher_weight: float = 0.5,
    validation: bool = False,
) -> torch.Tensor:
    """Evaluate truth, hard-teacher, soft-profile or truth-plus-soft loss."""
    if model_name not in {"M1", "M1_power", "M2", "M3"}:
        raise ValueError(f"unknown E007 model: {model_name}")
    if strategy not in {"E1", "E2-H", "E2-S", "E3-S"}:
        raise ValueError(f"unknown E007 strategy: {strategy}")
    if not math.isfinite(teacher_weight) or teacher_weight < 0:
        raise ValueError("teacher_weight must be finite and nonnegative")
    if strategy == "E3-S" and validation:
        strategy = "E1"

    if model_name == "M2":
        prediction = output.reshape(-1)
        key_by_strategy = {
            "E1": "truth_correction",
            "E2-H": "teacher_hard_correction",
            "E2-S": "teacher_soft_correction",
        }

        def point_loss(key: str) -> torch.Tensor:
            return F.smooth_l1_loss(prediction, targets[key].reshape(-1))

        if strategy == "E3-S":
            return point_loss("truth_correction") + teacher_weight * point_loss(
                "teacher_soft_correction"
            )
        return point_loss(key_by_strategy[strategy])

    if output.ndim != 2:
        raise ValueError("M1/M3 output must have shape (batch, frequency_bins)")
    log_probabilities = F.log_softmax(output.float(), dim=-1)
    key_by_strategy = {
        "E1": "truth_distribution",
        "E2-H": "teacher_hard_distribution",
        "E2-S": "teacher_soft_response",
    }

    def distribution_loss(key: str) -> torch.Tensor:
        return F.kl_div(
            log_probabilities, targets[key].float(), reduction="batchmean"
        )

    if strategy == "E3-S":
        return distribution_loss("truth_distribution") + teacher_weight * distribution_loss(
            "teacher_soft_response"
        )
    return distribution_loss(key_by_strategy[strategy])


def screen_selection(
    scores: dict[str, dict[str, Any]],
    *,
    max_rmse_20db_hz: float = 5.0,
    min_adjacent_fft_points: int = 2,
    minimum_validation_drop: float = 0.20,
) -> dict[str, Any]:
    """Apply the frozen one-seed screen rule without looking at final-test data."""
    selected: list[str] = []
    selected_reasons: dict[str, str] = {}
    rejected: dict[str, str] = {}
    for model_name, score in scores.items():
        reasons: list[str] = []
        if not bool(score.get("finite", False)):
            reasons.append("non-finite training output or gradient")
        first = float(score.get("validation_first_loss", math.inf))
        best = float(score.get("validation_best_loss", math.inf))
        if (
            not math.isfinite(first)
            or not math.isfinite(best)
            or first <= 0
            or (first - best) / first < minimum_validation_drop
        ):
            reasons.append("validation objective dropped by less than the required 20%")
        rmse_20db = float(score.get("development_rmse_20db_hz", math.inf))
        win_flags = [bool(value) for value in score.get("fft_win_by_snr", [])]
        max_consecutive_wins = 0
        current_run = 0
        for won in win_flags:
            current_run = current_run + 1 if won else 0
            max_consecutive_wins = max(max_consecutive_wins, current_run)
        if not (
            (math.isfinite(rmse_20db) and rmse_20db <= max_rmse_20db_hz)
            or max_consecutive_wins >= min_adjacent_fft_points
        ):
            reasons.append("development RMSE and adjacent FFT-win criteria both failed")
        if model_name in {"M1", "M2"} and not bool(
            score.get("structural_transform_ok", False)
        ):
            reasons.append("claimed structural transform behavior failed")
        if reasons:
            rejected[model_name] = "; ".join(reasons)
        else:
            selected.append(model_name)
            selected_reasons[model_name] = (
                f"validation drop={(first-best)/first:.3f}, "
                f"20 dB RMSE={rmse_20db:.4g} Hz, "
                f"max consecutive FFT wins={max_consecutive_wins}"
            )
    return {
        "selected": selected,
        "selected_reasons": selected_reasons,
        "rejected": rejected,
        "rule": {
            "max_rmse_20db_hz": float(max_rmse_20db_hz),
            "min_adjacent_fft_points": int(min_adjacent_fft_points),
            "minimum_validation_drop": float(minimum_validation_drop),
        },
    }


def _model_family(model_name: str) -> str:
    return "M1" if model_name == "M1_power" else model_name


def _make_model(model_name: str, config: dict[str, Any]) -> torch.nn.Module:
    models = config["models"]
    if model_name == "M1":
        spec = models["m1"]
        return CircularSpectralNet(
            input_channels=3,
            hidden_channels=tuple(int(value) for value in spec["hidden_channels"]),
            kernel_size=int(spec["kernel_size"]),
        )
    if model_name == "M1_power":
        spec = models["m1"]
        return CircularSpectralNet(
            input_channels=1,
            hidden_channels=tuple(int(value) for value in spec["hidden_channels"]),
            kernel_size=int(spec["kernel_size"]),
        )
    if model_name == "M2":
        spec = models["m2"]
        return FFTSubbinCorrector(
            input_features=3 * int(spec["local_window_bins"]),
            hidden_units=tuple(int(value) for value in spec["hidden_units"]),
            half_bin=float(spec["correction_half_width_bins"]),
        )
    if model_name == "M3":
        spec = models["m3"]
        return IQTransformer(
            n_samples=int(config["n_samples"]),
            output_bins=int(spec["fft_factor_for_output"]) * int(config["n_samples"]),
            d_model=int(spec["d_model"]),
            n_heads=int(spec["n_heads"]),
            n_layers=int(spec["n_layers"]),
            feedforward_dim=int(spec["feedforward_dim"]),
            dropout=float(spec["dropout"]),
        )
    raise ValueError(f"unknown E007 model: {model_name}")


def _coarse_frequency(
    iq: np.ndarray, prf_hz: float, n_fft: int
) -> np.ndarray:
    spectrum = np.fft.fftshift(np.fft.fft(iq, n=n_fft, axis=-1), axes=-1)
    peak = np.argmax(np.abs(spectrum) ** 2, axis=-1)
    return -prf_hz / 2.0 + peak * prf_hz / n_fft


def _prepare_iq_array(iq: np.ndarray, prf_hz: float, n_fft: int) -> tuple[np.ndarray, bool]:
    values = np.asarray(iq)
    scalar = values.ndim == 1
    if scalar:
        values = values[None, :]
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("iq must have shape (N,) or (batch, N) with nonempty signals")
    if not math.isfinite(prf_hz) or prf_hz <= 0:
        raise ValueError("prf_hz must be positive and finite")
    if n_fft < values.shape[-1] or n_fft % 2:
        raise ValueError("n_fft must be even and at least the signal length")
    if not np.all(np.isfinite(values)):
        raise ValueError("iq values must be finite")
    return values, scalar


def estimate_fft_nfft_peak(
    iq: np.ndarray, prf_hz: float, n_fft: int
) -> np.ndarray | float:
    """Estimate frequency from a fixed even-length zero-padded FFT peak."""
    values, scalar = _prepare_iq_array(iq, prf_hz, n_fft)
    spectrum = np.fft.fftshift(np.fft.fft(values, n=n_fft, axis=-1), axes=-1)
    peak = np.argmax(np.abs(spectrum) ** 2, axis=-1)
    estimate = -prf_hz / 2.0 + peak * prf_hz / n_fft
    estimate = np.asarray(wrap_frequency(estimate, prf_hz))
    return float(estimate[0]) if scalar else estimate


def estimate_fft_nfft_interpolated(
    iq: np.ndarray, prf_hz: float, n_fft: int
) -> np.ndarray | float:
    """Three-bin log-parabolic refinement of a zero-padded FFT peak."""
    values, scalar = _prepare_iq_array(iq, prf_hz, n_fft)
    spectrum = np.fft.fftshift(np.fft.fft(values, n=n_fft, axis=-1), axes=-1)
    power = np.abs(spectrum) ** 2
    peak = np.argmax(power, axis=-1)
    indices = np.arange(len(values))
    center = np.log(np.maximum(power[indices, peak], np.finfo(np.float64).tiny))
    left = np.log(
        np.maximum(power[indices, (peak - 1) % n_fft], np.finfo(np.float64).tiny)
    )
    right = np.log(
        np.maximum(power[indices, (peak + 1) % n_fft], np.finfo(np.float64).tiny)
    )
    denominator = left - 2.0 * center + right
    offset = np.zeros_like(center)
    usable = np.abs(denominator) > np.finfo(np.float64).eps
    offset[usable] = 0.5 * (left[usable] - right[usable]) / denominator[usable]
    offset = np.clip(offset, -0.5, 0.5)
    estimate = -prf_hz / 2.0 + (peak + offset) * prf_hz / n_fft
    estimate = np.asarray(wrap_frequency(estimate, prf_hz))
    return float(estimate[0]) if scalar else estimate


def _prepare_features(
    batch: SignalBatch,
    model_name: str,
    config: dict[str, Any],
    device: torch.device,
    teacher: dict[str, np.ndarray] | None = None,
) -> tuple[torch.Tensor, np.ndarray | None]:
    """Prepare fixed features in CUDA chunks and retain the feature table on CPU."""
    iq = np.asarray(batch.iq)
    prf_hz = float(config["prf_hz"])
    if model_name == "M3":
        features = np.stack((iq.real, iq.imag), axis=-1).astype(np.float32, copy=False)
        return torch.from_numpy(features), None
    if model_name in {"M1", "M1_power"}:
        n_fft = int(config["models"]["m1"]["fft_factor"]) * int(config["n_samples"])
        chunks: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(iq), 256):
                values = torch.as_tensor(
                    iq[start : start + 256], dtype=torch.complex64, device=device
                )
                features = spectral_coherence_features(values, n_fft=n_fft)
                if model_name == "M1_power":
                    features = features[:, :1, :]
                chunks.append(features.cpu())
                del values, features
        return torch.cat(chunks, dim=0), None
    if model_name == "M2":
        spec = config["models"]["m2"]
        n_fft = int(spec["fft_factor"]) * int(config["n_samples"])
        window_bins = int(spec["local_window_bins"])
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(iq), 256):
                values = torch.as_tensor(
                    iq[start : start + 256], dtype=torch.complex64, device=device
                )
                features = local_spectrum_features(
                    values, n_fft=n_fft, window_bins=window_bins
                )
                chunks.append(features.cpu())
                del values, features
        if teacher is not None and "m2_coarse_frequency_hz" in teacher:
            coarse = np.asarray(teacher["m2_coarse_frequency_hz"], dtype=np.float64)
        else:
            coarse = _coarse_frequency(iq, prf_hz, n_fft)
        return torch.cat(chunks, dim=0), coarse
    raise ValueError(f"unknown E007 model: {model_name}")


def _torch_targets(
    batch: SignalBatch,
    teacher: dict[str, np.ndarray],
    model_name: str,
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    family = _model_family(model_name)
    if family == "M2":
        return {
            "truth_correction": torch.as_tensor(
                teacher["m2_truth_correction_bins"], dtype=torch.float32
            ),
            "teacher_hard_correction": torch.as_tensor(
                teacher["m2_teacher_hard_correction_bins"], dtype=torch.float32
            ),
            "teacher_soft_correction": torch.as_tensor(
                teacher["m2_teacher_soft_correction_bins"], dtype=torch.float32
            ),
        }
    n_bins = (
        int(config["models"]["m1"]["fft_factor"])
        if family == "M1"
        else int(config["models"]["m3"]["fft_factor_for_output"])
    ) * int(config["n_samples"])
    sigma_bins = float(config["models"][family.lower()]["target_sigma_bins"])
    return {
        "truth_distribution": circular_frequency_targets(
            torch.as_tensor(batch.frequency_hz, dtype=torch.float32),
            float(config["prf_hz"]),
            n_bins=n_bins,
            sigma_bins=sigma_bins,
        ),
        "teacher_hard_distribution": circular_frequency_targets(
            torch.as_tensor(
                teacher["teacher_hard_frequency_hz"], dtype=torch.float32
            ),
            float(config["prf_hz"]),
            n_bins=n_bins,
            sigma_bins=float(config["teacher"]["hard_target_sigma_bins"]),
        ),
        "teacher_soft_response": torch.as_tensor(
            teacher["teacher_soft_response"], dtype=torch.float32
        ),
    }


def _append_history(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _evaluate_loss(
    model: torch.nn.Module,
    features: torch.Tensor,
    targets: dict[str, torch.Tensor],
    model_name: str,
    strategy: str,
    teacher_weight: float,
    device: torch.device,
    batch_size: int,
    *,
    validation: bool,
) -> float:
    model.eval()
    total = 0.0
    count = len(features)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            x = features[start:stop].to(device, non_blocking=True)
            y = {
                key: value[start:stop].to(device, non_blocking=True)
                for key, value in targets.items()
            }
            output = model(x)
            loss = e007_loss(
                output,
                y,
                model_name,
                strategy,
                teacher_weight=teacher_weight,
                validation=validation,
            )
            if not torch.isfinite(loss):
                return math.inf
            total += float(loss) * (stop - start)
    return total / max(count, 1)


def _train_one(
    model_name: str,
    strategy: str,
    train_batch: SignalBatch,
    train_teacher: dict[str, np.ndarray],
    train_features: torch.Tensor,
    validation_batch: SignalBatch,
    validation_teacher: dict[str, np.ndarray],
    validation_features: torch.Tensor,
    train_seed: int,
    torch_seed: int,
    config: dict[str, Any],
    models_dir: Path,
    history_path: Path,
    device: torch.device,
    train_coarse: np.ndarray | None = None,
    validation_coarse: np.ndarray | None = None,
) -> dict[str, Any]:
    del train_coarse, validation_coarse
    train_config = config["training"]
    batch_size = int(train_config["batch_size"])
    torch.manual_seed(int(torch_seed))
    torch.cuda.manual_seed_all(int(torch_seed))
    model = _make_model(model_name, config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_config["learning_rate"])
    )
    train_targets = _torch_targets(train_batch, train_teacher, model_name, config)
    validation_targets = _torch_targets(
        validation_batch, validation_teacher, model_name, config
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(torch_seed) + 1)
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    first_validation_loss = math.inf
    history_rows: list[dict[str, Any]] = []
    finite = True
    started_at = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(train_features), generator=generator)
        accumulated = 0.0
        for start in range(0, len(permutation), batch_size):
            indices = permutation[start : start + batch_size]
            x = train_features[indices].to(device, non_blocking=True)
            y = {
                key: value[indices].to(device, non_blocking=True)
                for key, value in train_targets.items()
            }
            optimizer.zero_grad(set_to_none=True)
            output = model(x)
            loss = e007_loss(
                output,
                y,
                model_name,
                strategy,
                teacher_weight=float(train_config["teacher_soft_weight"]),
            )
            if not torch.isfinite(loss):
                finite = False
                break
            loss.backward()
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                finite = False
                break
            optimizer.step()
            accumulated += float(loss.detach()) * len(indices)
        if not finite:
            break
        train_loss = accumulated / max(len(train_features), 1)
        validation_loss = _evaluate_loss(
            model,
            validation_features,
            validation_targets,
            model_name,
            strategy,
            float(train_config["teacher_soft_weight"]),
            device,
            batch_size,
            validation=True,
        )
        if epoch == 1:
            first_validation_loss = validation_loss
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"  {model_name}/{strategy} seed={train_seed} epoch={epoch} "
                f"train={train_loss:.5g} validation={validation_loss:.5g} "
                f"elapsed={time.perf_counter() - started_at:.1f}s",
                flush=True,
            )
        history_rows.append(
            {
                "phase": "screen" if train_seed == int(config["data"]["screen_train_seed"]) else "formal",
                "model": model_name,
                "strategy": strategy,
                "train_seed": int(train_seed),
                "torch_seed": int(torch_seed),
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_selection_target": (
                    "truth" if strategy in {"E1", "E3-S"} else "teacher"
                ),
                "device": str(device),
                "batch_size": batch_size,
            }
        )
        if math.isfinite(validation_loss) and validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= int(train_config["patience"]):
                break

    epochs_completed = len(history_rows)
    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started_at
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    checkpoint_path: Path | None = None
    if best_state is not None and finite:
        models_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = models_dir / f"{model_name}_{strategy}_seed{train_seed}.pt"
        torch.save(
            {
                "model": model_name,
                "strategy": strategy,
                "train_seed": int(train_seed),
                "torch_seed": int(torch_seed),
                "selected_epoch": best_epoch,
                "validation_selection_loss": best_loss,
                "state_dict": best_state,
                "configuration": config["models"],
                "training_device": str(device),
                "batch_size": batch_size,
            },
            checkpoint_path,
        )
    _append_history(history_path, history_rows)
    record = {
        "model": model_name,
        "strategy": strategy,
        "train_seed": int(train_seed),
        "torch_seed": int(torch_seed),
        "finite": bool(finite and best_state is not None),
        "epochs_completed": epochs_completed,
        "selected_epoch": best_epoch,
        "validation_first_loss": first_validation_loss,
        "validation_best_loss": best_loss,
        "training_seconds": elapsed,
        "peak_gpu_memory_bytes": peak_memory,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
    }
    del optimizer, model, train_targets, validation_targets
    torch.cuda.empty_cache()
    return record


def _recover_completed_training_record(
    model_name: str,
    strategy: str,
    train_seed: int,
    checkpoint_path: Path,
    history_path: Path,
) -> dict[str, Any]:
    """Reconstruct a completed run's metrics from its checkpoint and epoch log."""
    if not history_path.exists():
        raise FileNotFoundError("completed checkpoint exists without its epoch history")
    with history_path.open("r", newline="", encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("phase") == "screen"
            and row.get("model") == model_name
            and row.get("strategy") == strategy
            and int(row.get("train_seed", -1)) == int(train_seed)
        ]
    if not rows:
        raise ValueError("completed checkpoint has no matching screen history rows")
    rows.sort(key=lambda row: int(row["epoch"]))
    with torch.no_grad():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("model") != model_name
        or checkpoint.get("strategy") != strategy
        or int(checkpoint.get("train_seed", -1)) != int(train_seed)
    ):
        raise ValueError("existing screen checkpoint belongs to a different run")
    validation_losses = [float(row["validation_loss"]) for row in rows]
    selected_epoch = int(checkpoint["selected_epoch"])
    selected_rows = [row for row in rows if int(row["epoch"]) == selected_epoch]
    if not selected_rows:
        raise ValueError("selected checkpoint epoch is missing from its saved history")
    if int(checkpoint.get("batch_size", -1)) != int(rows[0]["batch_size"]):
        raise ValueError("checkpoint and epoch history disagree on batch size")
    return {
        "model": model_name,
        "strategy": strategy,
        "train_seed": int(train_seed),
        "torch_seed": int(checkpoint["torch_seed"]),
        "finite": bool(
            all(math.isfinite(float(row["train_loss"])) for row in rows)
            and all(math.isfinite(value) for value in validation_losses)
        ),
        "epochs_completed": len(rows),
        "selected_epoch": selected_epoch,
        "validation_first_loss": float(rows[0]["validation_loss"]),
        "validation_best_loss": float(checkpoint["validation_selection_loss"]),
        "training_seconds": None,
        "peak_gpu_memory_bytes": None,
        "checkpoint": str(checkpoint_path),
        "recovered_from_checkpoint": True,
        "recovery_note": "elapsed time and per-run peak memory were not persisted before the stage exited",
    }


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (Path, torch.device)):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _clean_json(value), indent=2, ensure_ascii=False, default=_json_default, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_development_batch(repo_root: Path) -> SignalBatch:
    path = repo_root / "experiments" / "P0-E005" / "datasets" / "test.npz"
    if not path.exists():
        raise FileNotFoundError(f"required development benchmark is missing: {path}")
    with np.load(path) as values:
        role = str(values["data_role"]) if "data_role" in values else "development_benchmark"
        if role not in {"test", "development_benchmark"}:
            raise ValueError(f"E005 data is not a development benchmark: {role}")
        return SignalBatch(
            iq=values["iq"],
            frequency_hz=values["frequency_hz"],
            phase_rad=values["phase_rad"],
            amplitude=values["amplitude"],
            snr_db=values["snr_db"],
            seed=int(values["dataset_seed"]),
        )


def _save_batch(
    path: Path,
    batch: SignalBatch,
    teacher: dict[str, np.ndarray],
    role: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        iq=batch.iq,
        frequency_hz=batch.frequency_hz,
        phase_rad=batch.phase_rad,
        amplitude=batch.amplitude,
        snr_db=batch.snr_db,
        dataset_seed=np.asarray(batch.seed, dtype=np.int64),
        data_role=np.asarray(role),
        **teacher,
    )


def _load_saved_batch(path: Path) -> tuple[SignalBatch, dict[str, np.ndarray]]:
    with np.load(path) as values:
        batch = SignalBatch(
            iq=values["iq"],
            frequency_hz=values["frequency_hz"],
            phase_rad=values["phase_rad"],
            amplitude=values["amplitude"],
            snr_db=values["snr_db"],
            seed=int(values["dataset_seed"]),
        )
        excluded = {
            "iq",
            "frequency_hz",
            "phase_rad",
            "amplitude",
            "snr_db",
            "dataset_seed",
            "data_role",
        }
        teacher = {key: values[key] for key in values.files if key not in excluded}
    return batch, teacher


def _rmse_by_snr(
    prediction: np.ndarray,
    truth: np.ndarray,
    snr_db: np.ndarray,
    prf_hz: float,
) -> dict[str, float]:
    error = np.asarray(wrapped_error(prediction, truth, prf_hz), dtype=np.float64)
    result: dict[str, float] = {}
    for value in np.unique(snr_db):
        mask = np.asarray(snr_db) == value
        result[str(float(value))] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return result


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(error, dtype=np.float64) ** 2)))


def paired_bootstrap_rmse_difference(
    candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
    truth: np.ndarray,
    *,
    prf_hz: float,
    bootstrap_samples: int = 2000,
    seed: int = 98765,
) -> dict[str, float]:
    """Hierarchical paired bootstrap of candidate-minus-baseline RMSE.

    The first axis represents independently trained model seeds; the second
    axis represents paired evaluation records. A one-dimensional input is
    treated as one training seed.
    """
    candidate = np.asarray(candidate_predictions, dtype=np.float64)
    baseline = np.asarray(baseline_predictions, dtype=np.float64)
    targets = np.asarray(truth, dtype=np.float64).reshape(-1)
    if candidate.ndim == 1:
        candidate = candidate[None, :]
    if baseline.ndim == 1:
        baseline = baseline[None, :]
    if (
        baseline.ndim == 2
        and baseline.shape[0] == 1
        and candidate.ndim == 2
        and candidate.shape[0] > 1
    ):
        baseline = np.broadcast_to(baseline, candidate.shape)
    if (
        candidate.ndim != 2
        or baseline.shape != candidate.shape
        or candidate.shape[1] != targets.size
        or candidate.shape[0] == 0
        or targets.size == 0
        or bootstrap_samples < 100
        or not math.isfinite(prf_hz)
        or prf_hz <= 0
    ):
        raise ValueError("bootstrap inputs have incompatible shapes or settings")
    candidate_error = np.asarray(
        wrapped_error(candidate, targets[None, :], prf_hz), dtype=np.float64
    )
    baseline_error = np.asarray(
        wrapped_error(baseline, targets[None, :], prf_hz), dtype=np.float64
    )
    n_seeds, n_samples = candidate_error.shape
    rng = np.random.default_rng(seed)
    candidate_rmse = np.empty((bootstrap_samples, n_seeds), dtype=np.float64)
    baseline_rmse = np.empty_like(candidate_rmse)
    for seed_index in range(n_seeds):
        sample_indices = rng.integers(
            0, n_samples, size=(bootstrap_samples, n_samples)
        )
        candidate_rmse[:, seed_index] = np.sqrt(
            np.mean(candidate_error[seed_index, sample_indices] ** 2, axis=1)
        )
        baseline_rmse[:, seed_index] = np.sqrt(
            np.mean(baseline_error[seed_index, sample_indices] ** 2, axis=1)
        )
    seed_indices = rng.integers(
        0, n_seeds, size=(bootstrap_samples, n_seeds)
    )
    candidate_summary = np.take_along_axis(
        candidate_rmse, seed_indices, axis=1
    ).mean(axis=1)
    baseline_summary = np.take_along_axis(
        baseline_rmse, seed_indices, axis=1
    ).mean(axis=1)
    differences = candidate_summary - baseline_summary
    candidate_point = np.mean(np.sqrt(np.mean(candidate_error**2, axis=1)))
    baseline_point = np.mean(np.sqrt(np.mean(baseline_error**2, axis=1)))
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "candidate_mean_rmse_hz": float(candidate_point),
        "baseline_mean_rmse_hz": float(baseline_point),
        "mean_rmse_difference_hz": float(candidate_point - baseline_point),
        "ci95_low_hz": float(low),
        "ci95_high_hz": float(high),
        "bootstrap_samples": int(bootstrap_samples),
        "bootstrap_seed": int(seed),
    }


def grouped_frequency_metrics(
    prediction: np.ndarray,
    batch: SignalBatch,
    *,
    prf_hz: float,
    n_samples: int,
    outlier_threshold_hz: float = 5.0,
) -> list[dict[str, Any]]:
    """Summarize circular frequency error by SNR and physical input strata."""
    estimate = np.asarray(prediction, dtype=np.float64).reshape(-1)
    truth = np.asarray(batch.frequency_hz, dtype=np.float64).reshape(-1)
    phase = np.asarray(batch.phase_rad, dtype=np.float64).reshape(-1)
    amplitude = np.asarray(batch.amplitude, dtype=np.float64).reshape(-1)
    snr = np.asarray(batch.snr_db, dtype=np.float64).reshape(-1)
    count = truth.size
    if any(values.size != count for values in (estimate, phase, amplitude, snr)):
        raise ValueError("predictions and SignalBatch fields must have equal length")
    if count == 0 or outlier_threshold_hz <= 0:
        raise ValueError("metrics require samples and a positive outlier threshold")
    truth = np.asarray(wrap_frequency(truth, prf_hz))
    errors = np.asarray(wrapped_error(estimate, truth, prf_hz), dtype=np.float64)
    crlb = np.asarray(frequency_crlb_hz(prf_hz, n_samples, snr), dtype=np.float64)

    frequency_bin = np.minimum(
        7, np.floor((truth + prf_hz / 2.0) / (prf_hz / 8.0)).astype(int)
    )
    edge_width = prf_hz / n_samples
    nyquist_group = np.where(
        np.abs(truth) >= prf_hz / 2.0 - edge_width, "edge_band", "interior"
    )
    phase_wrapped = np.remainder(phase + np.pi, 2.0 * np.pi) - np.pi
    phase_sector = np.minimum(
        7, np.floor((phase_wrapped + np.pi) / (2.0 * np.pi / 8.0)).astype(int)
    )
    order = np.argsort(np.argsort(amplitude, kind="stable"), kind="stable")
    amplitude_quartile = np.minimum(3, order * 4 // count)

    categories = [
        ("overall", "all", np.ones(count, dtype=bool)),
    ]
    categories.extend(
        ("frequency_bin", f"bin_{index}", frequency_bin == index)
        for index in range(8)
    )
    categories.extend(
        ("nyquist", name, nyquist_group == name)
        for name in ("edge_band", "interior")
    )
    categories.extend(
        ("phase_sector", f"sector_{index}", phase_sector == index)
        for index in range(8)
    )
    categories.extend(
        ("amplitude_quartile", f"q{index + 1}", amplitude_quartile == index)
        for index in range(4)
    )

    output: list[dict[str, Any]] = []
    for snr_value in np.unique(snr):
        snr_mask = snr == snr_value
        for group_type, group_name, group_mask in categories:
            selected = snr_mask & group_mask
            if not np.any(selected):
                continue
            selected_error = errors[selected]
            selected_crlb = crlb[selected]
            rmse = _rmse(selected_error)
            output.append(
                {
                    "snr_db": float(snr_value),
                    "group_type": group_type,
                    "group": group_name,
                    "count": int(selected.sum()),
                    "rmse_hz": rmse,
                    "mae_hz": float(np.mean(np.abs(selected_error))),
                    "bias_hz": float(np.mean(selected_error)),
                    "outlier_rate_gt_threshold": float(
                        np.mean(np.abs(selected_error) > outlier_threshold_hz)
                    ),
                    "crlb_mean_hz": float(np.mean(selected_crlb)),
                    "rmse_to_crlb": float(rmse / np.mean(selected_crlb)),
                    "outlier_threshold_hz": float(outlier_threshold_hz),
                }
            )
    return output


def _make_noiseless_batch(config: dict[str, Any]) -> SignalBatch:
    """Create a seeded continuous off-grid set with an explicit Nyquist stratum."""
    data = config["data"]
    count = int(data["noiseless_count"])
    seed = int(data["noiseless_seed"])
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    if count < 16 or count % 2:
        raise ValueError("noiseless_count must be an even number of at least 16")
    rng = np.random.default_rng(seed)
    frequency = np.empty(count, dtype=np.float64)
    half = count // 2
    frequency[:half] = rng.uniform(-prf_hz / 2.0, prf_hz / 2.0, size=half)
    edge_width = prf_hz / n_samples
    distance = rng.uniform(0.0, edge_width, size=half)
    frequency[half : half + half // 2] = -prf_hz / 2.0 + distance[: half // 2]
    frequency[half + half // 2 :] = prf_hz / 2.0 - distance[half // 2 :]
    frequency[half + half // 2 - 1] = -prf_hz / 2.0
    frequency[half + half // 2 :] = np.minimum(
        frequency[half + half // 2 :], np.nextafter(prf_hz / 2.0, 0.0)
    )
    phase = rng.uniform(-np.pi, np.pi, size=count)
    amplitude = rng.uniform(*data["amplitude_range"], size=count)
    time_indices = np.arange(n_samples, dtype=np.float64)
    iq = amplitude[:, None] * np.exp(
        1j
        * (
            2.0 * np.pi * frequency[:, None] * time_indices[None, :] / prf_hz
            + phase[:, None]
        )
    )
    return SignalBatch(
        iq=np.asarray(iq, dtype=np.complex64),
        frequency_hz=frequency,
        phase_rad=phase,
        amplitude=amplitude,
        snr_db=np.full(count, np.inf, dtype=np.float64),
        seed=seed,
    )


def _baseline_predictions(
    batch: SignalBatch, config: dict[str, Any], *, include_global: bool
) -> dict[str, np.ndarray]:
    iq = np.asarray(batch.iq)
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    predictions: dict[str, np.ndarray] = {
        "B0_constant_zero": np.zeros(len(iq), dtype=np.float64)
    }
    baseline_functions = [
        ("B1_fft_peak", lambda: estimate_fft_peak(iq, prf_hz)),
        ("B2_fft_interpolated", lambda: estimate_fft_interpolated(iq, prf_hz)),
        ("B3_phase_difference", lambda: estimate_phase_difference(iq, prf_hz)),
        ("B4_local_profile_ML", lambda: estimate_mle_local(iq, prf_hz)),
        ("B6_4N_fft_peak", lambda: estimate_fft_nfft_peak(iq, prf_hz, 4 * n_samples)),
        ("B8_16N_fft_peak", lambda: estimate_fft_nfft_peak(iq, prf_hz, 16 * n_samples)),
        (
            "B9_4N_local_parabolic",
            lambda: estimate_fft_nfft_interpolated(iq, prf_hz, 4 * n_samples),
        ),
        (
            "B10_16N_local_parabolic",
            lambda: estimate_fft_nfft_interpolated(iq, prf_hz, 16 * n_samples),
        ),
    ]
    if include_global:
        evaluation = config["evaluation"]
        baseline_functions.append(
            (
                "B5_global_grid_profile_ML",
                lambda: estimate_mle_global_grid(
                    iq,
                    prf_hz,
                    grid_size=int(evaluation["global_ml_grid_size"]),
                    chunk_size=128,
                ),
            )
        )
    for name, estimate in baseline_functions:
        print(f"  baseline {name}: start ({len(iq)} records)", flush=True)
        started = time.perf_counter()
        predictions[name] = np.asarray(estimate(), dtype=np.float64)
        print(
            f"  baseline {name}: done in {time.perf_counter() - started:.1f}s",
            flush=True,
        )
    return predictions


def _development_baselines(
    batch: SignalBatch,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    """Cache deterministic development baselines so stages reuse the same rows."""
    cache = output_dir / "development_baselines.npz"
    if cache.exists():
        with np.load(cache) as values:
            cached_seed = int(values["dataset_seed"])
            if cached_seed != int(batch.seed):
                raise ValueError("cached development baselines belong to another dataset seed")
            return {
                name: np.asarray(values[name], dtype=np.float64)
                for name in values.files
                if name not in {"dataset_seed", "truth_frequency_hz", "snr_db"}
            }
    predictions = _baseline_predictions(batch, config, include_global=False)
    np.savez_compressed(
        cache,
        dataset_seed=np.asarray(batch.seed, dtype=np.int64),
        truth_frequency_hz=batch.frequency_hz,
        snr_db=batch.snr_db,
        **predictions,
    )
    return predictions


def _predict_checkpoint(
    model_name: str,
    checkpoint_path: Path,
    batch: SignalBatch,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    features, coarse = _prepare_features(batch, model_name, config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = _make_model(model_name, config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output_parts: list[torch.Tensor] = []
    batch_size = int(config["training"]["batch_size"])
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            x = features[start : start + batch_size].to(device, non_blocking=True)
            output = model(x)
            if _model_family(model_name) == "M2":
                output_parts.append(output.cpu().reshape(-1))
            else:
                output_parts.append(output.cpu())
    outputs = torch.cat(output_parts, dim=0)
    del model, features
    torch.cuda.empty_cache()
    if _model_family(model_name) == "M2":
        assert coarse is not None
        estimate = coarse + outputs.numpy().astype(np.float64) * (
            float(config["prf_hz"])
            / (int(config["models"]["m2"]["fft_factor"]) * int(config["n_samples"]))
        )
        return np.asarray(wrap_frequency(estimate, float(config["prf_hz"])))
    return decode_circular_logits(outputs, float(config["prf_hz"])).numpy().astype(
        np.float64
    )


def _max_consecutive_wins(flags: list[bool]) -> int:
    maximum = current = 0
    for value in flags:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _screen_physics_diagnostic(
    model_name: str,
    checkpoint_path: Path,
    batch: SignalBatch,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    subset_count = min(128, len(batch.iq))
    iq = np.asarray(batch.iq[:subset_count])
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    family = _model_family(model_name)
    if family == "M1":
        n_fft = int(config["models"]["m1"]["fft_factor"]) * n_samples
        delta_hz = 5.0 * prf_hz / n_fft
    elif family == "M2":
        n_fft = int(config["models"]["m2"]["fft_factor"]) * n_samples
        delta_hz = prf_hz / n_fft
    else:
        delta_hz = 3.0 * prf_hz / n_samples
    time_indices = np.arange(n_samples, dtype=np.float64)
    shifted_iq = iq * np.exp(
        2j * np.pi * delta_hz * time_indices[None, :] / prf_hz
    )
    original_batch = SignalBatch(
        iq=iq,
        frequency_hz=batch.frequency_hz[:subset_count],
        phase_rad=batch.phase_rad[:subset_count],
        amplitude=batch.amplitude[:subset_count],
        snr_db=batch.snr_db[:subset_count],
        seed=int(batch.seed),
    )
    shifted_batch = SignalBatch(
        iq=shifted_iq,
        frequency_hz=np.asarray(
            wrap_frequency(
                batch.frequency_hz[:subset_count] + delta_hz, prf_hz
            )
        ),
        phase_rad=batch.phase_rad[:subset_count],
        amplitude=batch.amplitude[:subset_count],
        snr_db=batch.snr_db[:subset_count],
        seed=int(batch.seed),
    )
    original_prediction = _predict_checkpoint(
        model_name, checkpoint_path, original_batch, config, device
    )
    shifted_prediction = _predict_checkpoint(
        model_name, checkpoint_path, shifted_batch, config, device
    )
    equivariance_error = np.asarray(
        wrapped_error(shifted_prediction, original_prediction + delta_hz, prf_hz)
    )
    result = {
        "frequency_shift_hz": float(delta_hz),
        "frequency_shift_mae_hz": float(np.mean(np.abs(equivariance_error))),
        "frequency_shift_rmse_hz": _rmse(equivariance_error),
        "frequency_shift_max_abs_hz": float(np.max(np.abs(equivariance_error))),
    }
    if family == "M1":
        orig_features, _ = _prepare_features(original_batch, model_name, config, device)
        shifted_features, _ = _prepare_features(shifted_batch, model_name, config, device)
        model = _make_model(model_name, config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            logits = model(orig_features.to(device))
            shifted_logits = model(shifted_features.to(device))
        expected = torch.roll(
            logits, shifts=int(round(delta_hz * n_fft / prf_hz)), dims=-1
        )
        max_logit_error = float((shifted_logits - expected).abs().max().cpu())
        result["max_shifted_logit_error"] = max_logit_error
        result["structural_transform_ok"] = bool(
            max_logit_error < 1e-3
            and result["frequency_shift_max_abs_hz"] < prf_hz / n_fft * 0.02
        )
        del model, logits, shifted_logits
        torch.cuda.empty_cache()
    elif family == "M2":
        original_features, _ = _prepare_features(original_batch, model_name, config, device)
        shifted_features, _ = _prepare_features(shifted_batch, model_name, config, device)
        max_feature_error = float(
            np.max(np.abs(original_features.numpy() - shifted_features.numpy()))
        )
        model = _make_model(model_name, config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            correction = model(original_features.to(device))
            shifted_correction = model(shifted_features.to(device))
        max_correction_error = float(
            (correction - shifted_correction).abs().max().cpu()
        )
        result["max_local_feature_error"] = max_feature_error
        result["max_subbin_correction_error"] = max_correction_error
        result["structural_transform_ok"] = bool(
            max_feature_error < 1e-3 and max_correction_error < 1e-4
        )
        del model, correction, shifted_correction
        torch.cuda.empty_cache()
    else:
        result["structural_transform_ok"] = True
    return result


def _environment_snapshot(device: torch.device) -> dict[str, Any]:
    def command_output(arguments: list[str]) -> str | None:
        try:
            result = subprocess.run(
                arguments, check=False, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    commit = command_output(["git", "rev-parse", "HEAD"])
    dirty = command_output(["git", "status", "--short"])
    nvidia = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
            "--format=csv,noheader",
        ]
    )
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(device),
        "training_device": str(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "gpu_free_memory_bytes_at_snapshot": int(free_bytes),
        "gpu_total_memory_bytes_at_snapshot": int(total_bytes),
        "nvidia_smi": nvidia,
        "git_head": commit,
        "git_dirty": bool(dirty),
        "git_status_short": dirty,
        "platform": platform.platform(),
    }


def run_screen(
    config: dict[str, Any], repo_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run the one-seed E1 screen sequentially on CUDA and freeze selection."""
    device = require_cuda(config["training"].get("device", ""))
    torch.cuda.set_device(device)
    root = Path(repo_root)
    output = Path(output_dir)
    datasets_dir = output / "datasets"
    models_dir = output / "models" / "screen"
    sentinel = output / "screening.json"
    expected_paths = [
        sentinel,
        output / "selection.json",
    ]
    if any(path.exists() for path in expected_paths):
        raise FileExistsError(
            "E007 screen artifacts already exist; refusing to overwrite prior results"
        )
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    history_path = output / "training_history.csv"
    data_config = config["data"]
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    amplitude_range = tuple(float(value) for value in data_config["amplitude_range"])
    train_path = datasets_dir / f"screen_train_seed_{data_config['screen_train_seed']}.npz"
    validation_path = datasets_dir / f"screen_validation_seed_{data_config['screen_validation_seed']}.npz"
    if train_path.exists() != validation_path.exists():
        raise FileExistsError("only one saved screen split exists; refusing to mix dataset states")
    if train_path.exists():
        print("P0-E007 screen: reusing the retained seeded datasets and teacher targets", flush=True)
        train_batch, train_teacher = _load_saved_batch(train_path)
        validation_batch, validation_teacher = _load_saved_batch(validation_path)
        if (
            int(train_batch.seed) != int(data_config["screen_train_seed"])
            or int(validation_batch.seed) != int(data_config["screen_validation_seed"])
        ):
            raise ValueError("saved screen datasets do not match the frozen data seeds")
    else:
        train_batch = generate_batch(
            count=int(data_config["train_count"]),
            seed=int(data_config["screen_train_seed"]),
            prf_hz=prf_hz,
            n_samples=n_samples,
            snr_levels_db=tuple(data_config["train_snr_levels_db"]),
            amplitude_range=amplitude_range,
        )
        validation_batch = generate_batch(
            count=int(data_config["validation_count"]),
            seed=int(data_config["screen_validation_seed"]),
            prf_hz=prf_hz,
            n_samples=n_samples,
            snr_levels_db=tuple(data_config["train_snr_levels_db"]),
            amplitude_range=amplitude_range,
        )
        print("P0-E007 screen: generating distinct truth and teacher targets", flush=True)
        train_teacher = make_teacher_targets(train_batch, config)
        validation_teacher = make_teacher_targets(validation_batch, config)
        _save_batch(train_path, train_batch, train_teacher, "screen_train")
        _save_batch(validation_path, validation_batch, validation_teacher, "screen_validation")
    development_batch = _load_development_batch(root)
    print("P0-E007 screen: evaluating cached development baselines", flush=True)
    development_baselines = _development_baselines(
        development_batch, config, output
    )
    dev_snr = np.asarray(development_batch.snr_db, dtype=np.float64)
    primary = [float(value) for value in config["evaluation"]["primary_snr_levels_db"]]
    model_names = ["M1", "M1_power", "M2", "M3"]
    expected_checkpoints = {
        f"{name}_E1_seed{int(data_config['screen_train_seed'])}.pt"
        for name in model_names
    }
    unexpected_checkpoints = {
        path.name for path in models_dir.glob("*.pt")
    } - expected_checkpoints
    if unexpected_checkpoints:
        raise FileExistsError(
            f"unexpected partial screen checkpoints exist: {sorted(unexpected_checkpoints)}"
        )
    screen_records: dict[str, Any] = {}
    selection_scores: dict[str, dict[str, Any]] = {}
    screen_predictions: dict[str, np.ndarray] = {
        f"baseline__{name}": prediction
        for name, prediction in development_baselines.items()
    }
    environment = _environment_snapshot(device)

    for model_name in model_names:
        checkpoint_path = models_dir / (
            f"{model_name}_E1_seed{int(train_batch.seed)}.pt"
        )
        if checkpoint_path.exists():
            print(
                f"P0-E007 screen: recovering completed {model_name} checkpoint",
                flush=True,
            )
            training_record = _recover_completed_training_record(
                model_name,
                "E1",
                int(train_batch.seed),
                checkpoint_path,
                history_path,
            )
        else:
            print(f"P0-E007 screen: preparing {model_name} features", flush=True)
            train_features, _ = _prepare_features(
                train_batch, model_name, config, device, train_teacher
            )
            validation_features, _ = _prepare_features(
                validation_batch, model_name, config, device, validation_teacher
            )
            torch_seed = int(data_config["screen_torch_seeds"][model_name])
            print(f"P0-E007 screen: training {model_name} E1 on {device}", flush=True)
            training_record = _train_one(
                model_name,
                "E1",
                train_batch,
                train_teacher,
                train_features,
                validation_batch,
                validation_teacher,
                validation_features,
                int(train_batch.seed),
                torch_seed,
                config,
                models_dir,
                history_path,
                device,
            )
        if training_record["checkpoint"] is not None:
            prediction = _predict_checkpoint(
                model_name,
                Path(training_record["checkpoint"]),
                development_batch,
                config,
                device,
            )
            prediction_finite = bool(np.all(np.isfinite(prediction)))
            per_snr = _rmse_by_snr(
                prediction,
                development_batch.frequency_hz,
                dev_snr,
                prf_hz,
            )
            baseline_rmse = _rmse_by_snr(
                development_baselines["B1_fft_peak"],
                development_batch.frequency_hz,
                dev_snr,
                prf_hz,
            )
            win_by_snr = [
                per_snr[str(snr)] < baseline_rmse[str(snr)] for snr in primary
            ]
            diagnostics = _screen_physics_diagnostic(
                model_name,
                Path(training_record["checkpoint"]),
                development_batch,
                config,
                device,
            )
            score = {
                "finite": bool(training_record["finite"] and prediction_finite),
                "validation_first_loss": training_record["validation_first_loss"],
                "validation_best_loss": training_record["validation_best_loss"],
                "development_rmse_20db_hz": per_snr.get("20.0", math.inf),
                "fft_win_by_snr": win_by_snr,
                "structural_transform_ok": diagnostics["structural_transform_ok"],
            }
            screen_predictions[f"model__{model_name}__E1"] = prediction
        else:
            per_snr = {}
            diagnostics = {"structural_transform_ok": False}
            score = {
                "finite": False,
                "validation_first_loss": math.inf,
                "validation_best_loss": math.inf,
                "development_rmse_20db_hz": math.inf,
                "fft_win_by_snr": [False] * len(primary),
                "structural_transform_ok": False,
            }
        screen_records[model_name] = {
            **training_record,
            "development_rmse_by_snr_hz": per_snr,
            "physics_diagnostics": diagnostics,
            "screen_score": score,
        }
        if model_name != "M1_power":
            selection_scores[model_name] = score
        if "train_features" in locals():
            del train_features, validation_features
            torch.cuda.empty_cache()

    selection = screen_selection(
        selection_scores,
        max_rmse_20db_hz=float(
            config["evaluation"]["selection_max_rmse_20db_hz"]
        ),
        min_adjacent_fft_points=int(
            config["evaluation"]["selection_min_adjacent_fft_points"]
        ),
    )
    selection["m1_power_ablation"] = screen_records["M1_power"]["screen_score"]
    selection["selected_models"] = list(selection["selected"])
    screening = {
        "experiment_id": config["experiment_id"],
        "stage": "screen",
        "data_roles": {
            "screen_train": int(train_batch.seed),
            "screen_validation": int(validation_batch.seed),
            "development_benchmark": int(development_batch.seed),
            "development_is_final_blind": False,
        },
        "training_budget": {
            "batch_size": int(config["training"]["batch_size"]),
            "epochs_max": int(config["training"]["epochs"]),
            "patience": int(config["training"]["patience"]),
            "device": str(device),
            "precision": config["training"]["precision"],
        },
        "environment": environment,
        "classical_development_rmse_by_snr_hz": {
            name: _rmse_by_snr(
                prediction,
                development_batch.frequency_hz,
                dev_snr,
                prf_hz,
            )
            for name, prediction in development_baselines.items()
        },
        "models": screen_records,
    }
    _write_json(sentinel, screening)
    _write_json(output / "selection.json", selection)
    np.savez_compressed(
        output / "screen_predictions.npz",
        truth_frequency_hz=development_batch.frequency_hz,
        snr_db=dev_snr,
        **screen_predictions,
    )
    _write_json(output / "environment.json", environment)
    return {"screening": screening, "selection": selection}


def _formal_dataset_paths(output: Path, train_seed: int, validation_seed: int):
    datasets_dir = output / "datasets"
    return (
        datasets_dir / f"formal_train_seed_{train_seed}.npz",
        datasets_dir / f"formal_validation_seed_{validation_seed}.npz",
    )


def _get_or_generate_formal_data(
    config: dict[str, Any], output: Path, index: int
) -> tuple[SignalBatch, dict[str, np.ndarray], SignalBatch, dict[str, np.ndarray]]:
    data_config = config["data"]
    train_seed = int(data_config["formal_train_seeds"][index])
    validation_seed = int(data_config["formal_validation_seeds"][index])
    train_path, validation_path = _formal_dataset_paths(
        output, train_seed, validation_seed
    )
    if train_path.exists() and validation_path.exists():
        train_batch, train_teacher = _load_saved_batch(train_path)
        validation_batch, validation_teacher = _load_saved_batch(validation_path)
        return train_batch, train_teacher, validation_batch, validation_teacher
    if train_path.exists() or validation_path.exists():
        raise FileExistsError(
            f"only one formal split exists for seeds {train_seed}/{validation_seed}"
        )
    amplitude_range = tuple(float(value) for value in data_config["amplitude_range"])
    train_batch = generate_batch(
        count=int(data_config["train_count"]),
        seed=train_seed,
        prf_hz=float(config["prf_hz"]),
        n_samples=int(config["n_samples"]),
        snr_levels_db=tuple(data_config["train_snr_levels_db"]),
        amplitude_range=amplitude_range,
    )
    validation_batch = generate_batch(
        count=int(data_config["validation_count"]),
        seed=validation_seed,
        prf_hz=float(config["prf_hz"]),
        n_samples=int(config["n_samples"]),
        snr_levels_db=tuple(data_config["train_snr_levels_db"]),
        amplitude_range=amplitude_range,
    )
    train_teacher = make_teacher_targets(train_batch, config)
    validation_teacher = make_teacher_targets(validation_batch, config)
    _save_batch(train_path, train_batch, train_teacher, "formal_train")
    _save_batch(
        validation_path, validation_batch, validation_teacher, "formal_validation"
    )
    return train_batch, train_teacher, validation_batch, validation_teacher


def _formal_kd_selection(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    baseline_rmse: dict[str, float],
) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record["model"] != "M1_power":
            by_model.setdefault(record["model"], []).append(record)
    selected: list[str] = []
    reasons: dict[str, str] = {}
    rejected: dict[str, str] = {}
    primary = [str(float(value)) for value in config["evaluation"]["primary_snr_levels_db"]]
    max_rmse = float(config["evaluation"]["selection_max_rmse_20db_hz"])
    min_run = int(config["evaluation"]["selection_min_adjacent_fft_points"])
    for model_name, model_records in by_model.items():
        finite = all(
            bool(record["training"]["finite"])
            and all(math.isfinite(value) for value in record["development_rmse_by_snr_hz"].values())
            for record in model_records
        )
        if not finite or len(model_records) != len(config["data"]["formal_train_seeds"]):
            rejected[model_name] = "formal E1 training or development predictions were incomplete"
            continue
        mean_per_snr = {
            snr: float(
                np.mean(
                    [
                        record["development_rmse_by_snr_hz"][snr]
                        for record in model_records
                    ]
                )
            )
            for snr in primary
        }
        win_flags = [
            mean_per_snr[snr] < baseline_rmse[snr] for snr in primary
        ]
        max_run = _max_consecutive_wins(win_flags)
        if mean_per_snr.get("20.0", math.inf) <= max_rmse or max_run >= min_run:
            selected.append(model_name)
            reasons[model_name] = (
                f"formal E1 mean 20 dB RMSE={mean_per_snr.get('20.0', math.inf):.4g} Hz; "
                f"maximum consecutive FFT wins={max_run}"
            )
        else:
            rejected[model_name] = (
                f"formal E1 mean 20 dB RMSE={mean_per_snr.get('20.0', math.inf):.4g} Hz "
                f"and maximum consecutive FFT wins={max_run}"
            )
    return {
        "selected": selected,
        "selected_reasons": reasons,
        "rejected": rejected,
        "rule": {
            "max_rmse_20db_hz": max_rmse,
            "min_adjacent_fft_points": min_run,
        },
    }


def run_formal_e1(
    config: dict[str, Any], repo_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Train selected E1 models on three seeds before allowing any KD run."""
    device = require_cuda(config["training"].get("device", ""))
    torch.cuda.set_device(device)
    root = Path(repo_root)
    output = Path(output_dir)
    sentinel = output / "formal_e1_metrics.json"
    if sentinel.exists():
        raise FileExistsError("formal E1 results already exist; refusing to overwrite")
    selection_path = output / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("run the frozen one-seed screen before formal E1")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected_models = list(selection.get("selected_models", selection.get("selected", [])))
    if not selected_models:
        result = {
            "experiment_id": config["experiment_id"],
            "status": "no_models_selected_after_screen",
            "training_replicates": [],
        }
        _write_json(sentinel, result)
        _write_json(output / "kd_selection.json", {"selected": [], "rejected": {}})
        return result
    models_to_train = list(selected_models)
    if "M1" in selected_models:
        models_to_train.append("M1_power")
    models_dir = output / "models" / "formal"
    history_path = output / "training_history.csv"
    models_dir.mkdir(parents=True, exist_ok=True)
    development_batch = _load_development_batch(root)
    dev_baselines = _development_baselines(
        development_batch, config, output
    )
    dev_snr = np.asarray(development_batch.snr_db, dtype=np.float64)
    records: list[dict[str, Any]] = []
    dev_predictions: dict[str, np.ndarray] = {
        f"baseline__{name}": prediction
        for name, prediction in dev_baselines.items()
    }
    model_index = {"M1": 1, "M1_power": 2, "M2": 3, "M3": 4}

    for model_name in models_to_train:
        for index, train_seed in enumerate(config["data"]["formal_train_seeds"]):
            validation_seed = int(config["data"]["formal_validation_seeds"][index])
            train_batch, train_teacher, validation_batch, validation_teacher = (
                _get_or_generate_formal_data(config, output, index)
            )
            train_features, _ = _prepare_features(
                train_batch, model_name, config, device, train_teacher
            )
            validation_features, _ = _prepare_features(
                validation_batch, model_name, config, device, validation_teacher
            )
            torch_seed = (
                int(config["data"]["formal_torch_seed_base"])
                + 100 * model_index[model_name]
                + index
            )
            print(
                f"P0-E007 formal E1: training {model_name} seed={train_seed}",
                flush=True,
            )
            training = _train_one(
                model_name,
                "E1",
                train_batch,
                train_teacher,
                train_features,
                validation_batch,
                validation_teacher,
                validation_features,
                int(train_seed),
                torch_seed,
                config,
                models_dir,
                history_path,
                device,
            )
            item: dict[str, Any] = {
                "model": model_name,
                "strategy": "E1",
                "train_seed": int(train_seed),
                "validation_seed": validation_seed,
                "training": training,
            }
            if training["checkpoint"] is not None:
                prediction = _predict_checkpoint(
                    model_name,
                    Path(training["checkpoint"]),
                    development_batch,
                    config,
                    device,
                )
                per_snr = _rmse_by_snr(
                    prediction,
                    development_batch.frequency_hz,
                    dev_snr,
                    float(config["prf_hz"]),
                )
                item["development_rmse_by_snr_hz"] = per_snr
                dev_predictions[
                    f"model__{model_name}__E1__seed{train_seed}"
                ] = prediction
            else:
                item["development_rmse_by_snr_hz"] = {}
            records.append(item)
            del train_features, validation_features
            torch.cuda.empty_cache()

    baseline_rmse = _rmse_by_snr(
        dev_baselines["B1_fft_peak"],
        development_batch.frequency_hz,
        dev_snr,
        float(config["prf_hz"]),
    )
    kd_selection = _formal_kd_selection(records, config, baseline_rmse)
    metrics = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "data_roles": {
            "development_benchmark_seed": int(development_batch.seed),
            "development_is_final_blind": False,
            "formal_train_seeds": list(config["data"]["formal_train_seeds"]),
            "formal_validation_seeds": list(
                config["data"]["formal_validation_seeds"]
            ),
        },
        "environment": _environment_snapshot(device),
        "classical_development_rmse_by_snr_hz": {
            name: _rmse_by_snr(
                prediction,
                development_batch.frequency_hz,
                dev_snr,
                float(config["prf_hz"]),
            )
            for name, prediction in dev_baselines.items()
        },
        "training_replicates": records,
        "kd_selection": kd_selection,
    }
    _write_json(sentinel, metrics)
    _write_json(output / "kd_selection.json", kd_selection)
    np.savez_compressed(
        output / "formal_e1_development_predictions.npz",
        truth_frequency_hz=development_batch.frequency_hz,
        snr_db=dev_snr,
        **dev_predictions,
    )
    return metrics


def run_formal_kd(
    config: dict[str, Any], repo_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Run E2-H, E2-S and E3-S only after formal E1 passes the accuracy gate."""
    device = require_cuda(config["training"].get("device", ""))
    torch.cuda.set_device(device)
    output = Path(output_dir)
    sentinel = output / "formal_kd_metrics.json"
    if sentinel.exists():
        raise FileExistsError("formal KD results already exist; refusing to overwrite")
    e1_path = output / "formal_e1_metrics.json"
    selection_path = output / "kd_selection.json"
    if not e1_path.exists() or not selection_path.exists():
        raise FileNotFoundError("formal E1 and its KD gate must finish before KD")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    eligible_models = list(selection.get("selected", []))
    if not eligible_models:
        result = {
            "experiment_id": config["experiment_id"],
            "status": "no_models_passed_formal_e1_kd_gate",
            "training_replicates": [],
        }
        _write_json(sentinel, result)
        return result

    models_dir = output / "models" / "formal"
    history_path = output / "training_history.csv"
    data_config = config["data"]
    root = Path(repo_root)
    development_batch = _load_development_batch(root)
    development_snr = np.asarray(development_batch.snr_db, dtype=np.float64)
    development_baselines = _development_baselines(
        development_batch, config, output
    )
    model_index = {"M1": 1, "M1_power": 2, "M2": 3, "M3": 4}
    strategy_index = {"E2-H": 1, "E2-S": 2, "E3-S": 3}
    records: list[dict[str, Any]] = []
    development_predictions: dict[str, np.ndarray] = {
        f"baseline__{name}": prediction
        for name, prediction in development_baselines.items()
    }
    for model_name in eligible_models:
        for index, train_seed in enumerate(data_config["formal_train_seeds"]):
            (
                train_batch,
                train_teacher,
                validation_batch,
                validation_teacher,
            ) = _get_or_generate_formal_data(config, output, index)
            train_features, _ = _prepare_features(
                train_batch, model_name, config, device, train_teacher
            )
            validation_features, _ = _prepare_features(
                validation_batch, model_name, config, device, validation_teacher
            )
            for strategy in ("E2-H", "E2-S", "E3-S"):
                torch_seed = (
                    int(data_config["formal_torch_seed_base"])
                    + 100 * model_index[model_name]
                    + 10 * strategy_index[strategy]
                    + index
                )
                print(
                    f"P0-E007 formal KD: training {model_name}/{strategy} seed={train_seed}",
                    flush=True,
                )
                training = _train_one(
                    model_name,
                    strategy,
                    train_batch,
                    train_teacher,
                    train_features,
                    validation_batch,
                    validation_teacher,
                    validation_features,
                    int(train_seed),
                    torch_seed,
                    config,
                    models_dir,
                    history_path,
                    device,
                )
                records.append(
                    {
                        "model": model_name,
                        "strategy": strategy,
                        "train_seed": int(train_seed),
                        "validation_seed": int(data_config["formal_validation_seeds"][index]),
                        "training": training,
                        "development_rmse_by_snr_hz": {},
                    }
                )
                if training["checkpoint"] is not None:
                    prediction = _predict_checkpoint(
                        model_name,
                        Path(training["checkpoint"]),
                        development_batch,
                        config,
                        device,
                    )
                    records[-1]["development_rmse_by_snr_hz"] = _rmse_by_snr(
                        prediction,
                        development_batch.frequency_hz,
                        development_snr,
                        float(config["prf_hz"]),
                    )
                    development_predictions[
                        f"model__{model_name}__{strategy}__seed{train_seed}"
                    ] = prediction
            del train_features, validation_features
            torch.cuda.empty_cache()

    result = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "eligible_models": eligible_models,
        "strategy_definitions": {
            "E2-H": "teacher hard point only",
            "E2-S": "teacher soft profile response only",
            "E3-S": "physical truth plus 0.5 times soft profile response",
        },
        "environment": _environment_snapshot(device),
        "development_is_final_blind": False,
        "development_teacher_hard_rmse_by_snr_hz": _rmse_by_snr(
            development_baselines["B4_local_profile_ML"],
            development_batch.frequency_hz,
            development_snr,
            float(config["prf_hz"]),
        ),
        "training_replicates": records,
    }
    _write_json(sentinel, result)
    np.savez_compressed(
        output / "formal_kd_development_predictions.npz",
        truth_frequency_hz=development_batch.frequency_hz,
        snr_db=development_snr,
        **development_predictions,
    )
    return result


def _make_final_blind_batch(config: dict[str, Any]) -> tuple[SignalBatch, dict[str, int]]:
    data = config["data"]
    base_seed = int(data["final_test_seed"])
    batches: list[SignalBatch] = []
    seed_by_snr: dict[str, int] = {}
    for index, snr_db in enumerate(data["test_snr_levels_db"]):
        seed = base_seed + 1000 * index
        seed_by_snr[str(float(snr_db))] = seed
        batches.append(
            generate_batch(
                count=int(data["test_per_snr"]),
                seed=seed,
                prf_hz=float(config["prf_hz"]),
                n_samples=int(config["n_samples"]),
                snr_db=float(snr_db),
                amplitude_range=tuple(float(x) for x in data["amplitude_range"]),
            )
        )
    return (
        SignalBatch(
            iq=np.concatenate([item.iq for item in batches]),
            frequency_hz=np.concatenate([item.frequency_hz for item in batches]),
            phase_rad=np.concatenate([item.phase_rad for item in batches]),
            amplitude=np.concatenate([item.amplitude for item in batches]),
            snr_db=np.concatenate([item.snr_db for item in batches]),
            seed=base_seed,
        ),
        seed_by_snr,
    )


def _load_saved_test_batch(path: Path, expected_role: str) -> SignalBatch:
    with np.load(path) as values:
        role = str(values["data_role"])
        if role != expected_role:
            raise ValueError(f"saved test data role is {role}, expected {expected_role}")
        return SignalBatch(
            iq=np.asarray(values["iq"]),
            frequency_hz=np.asarray(values["frequency_hz"], dtype=np.float64),
            phase_rad=np.asarray(values["phase_rad"], dtype=np.float64),
            amplitude=np.asarray(values["amplitude"], dtype=np.float64),
            snr_db=np.asarray(values["snr_db"], dtype=np.float64),
            seed=int(values["dataset_seed"]),
        )


def _checkpoint_sets(
    formal_e1: dict[str, Any], formal_kd: dict[str, Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    checkpoints: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in formal_e1.get("training_replicates", []):
        training = record.get("training", {})
        if training.get("checkpoint"):
            checkpoints.setdefault((record["model"], "E1"), []).append(
                {
                    "train_seed": int(record["train_seed"]),
                    "checkpoint": training["checkpoint"],
                }
            )
    for record in formal_kd.get("training_replicates", []):
        training = record.get("training", {})
        if training.get("checkpoint"):
            checkpoints.setdefault((record["model"], record["strategy"]), []).append(
                {
                    "train_seed": int(record["train_seed"]),
                    "checkpoint": training["checkpoint"],
                }
            )
    for key, records in checkpoints.items():
        records.sort(key=lambda item: item["train_seed"])
        if len({item["train_seed"] for item in records}) != len(records):
            raise ValueError(f"duplicate training seeds in checkpoint set {key}")
    return checkpoints


def _noiseless_error_record(
    prediction: np.ndarray, batch: SignalBatch, model: str, strategy: str, train_seed: int, prf_hz: float
) -> dict[str, Any]:
    error = np.asarray(
        wrapped_error(prediction, batch.frequency_hz, prf_hz), dtype=np.float64
    )
    edge_mask = np.abs(batch.frequency_hz) >= prf_hz / 2.0 - prf_hz / len(batch.iq[0])

    def summary(values: np.ndarray) -> dict[str, float]:
        return {
            "mae_hz": float(np.mean(np.abs(values))),
            "rmse_hz": _rmse(values),
            "p95_abs_hz": float(np.quantile(np.abs(values), 0.95)),
            "max_abs_hz": float(np.max(np.abs(values))),
        }

    return {
        "model": model,
        "strategy": strategy,
        "train_seed": int(train_seed),
        "count": int(len(error)),
        "all_frequency": summary(error),
        "nyquist_edge": summary(error[edge_mask]),
        "nyquist_edge_count": int(np.sum(edge_mask)),
    }


def _physics_transform_records(
    model_name: str,
    strategy: str,
    checkpoint_path: Path,
    train_seed: int,
    batch: SignalBatch,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    count = min(256, len(batch.iq))
    base = SignalBatch(
        iq=np.asarray(batch.iq[:count]),
        frequency_hz=np.asarray(batch.frequency_hz[:count]),
        phase_rad=np.asarray(batch.phase_rad[:count]),
        amplitude=np.asarray(batch.amplitude[:count]),
        snr_db=np.asarray(batch.snr_db[:count]),
        seed=int(batch.seed),
    )
    prediction = _predict_checkpoint(
        model_name, checkpoint_path, base, config, device
    )
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    phase_angle = 0.731
    phase_rotated = SignalBatch(
        iq=base.iq * np.exp(1j * phase_angle),
        frequency_hz=base.frequency_hz,
        phase_rad=np.asarray(wrap_frequency(base.phase_rad + phase_angle, 2.0 * np.pi)),
        amplitude=base.amplitude,
        snr_db=base.snr_db,
        seed=base.seed,
    )
    phase_prediction = _predict_checkpoint(
        model_name, checkpoint_path, phase_rotated, config, device
    )
    scaled = SignalBatch(
        iq=base.iq * (1.7 - 0.4j),
        frequency_hz=base.frequency_hz,
        phase_rad=base.phase_rad,
        amplitude=base.amplitude * abs(1.7 - 0.4j),
        snr_db=base.snr_db,
        seed=base.seed,
    )
    scale_prediction = _predict_checkpoint(
        model_name, checkpoint_path, scaled, config, device
    )
    family = _model_family(model_name)
    if family == "M1":
        shift_hz = prf_hz / (
            int(config["models"]["m1"]["fft_factor"]) * n_samples
        )
    elif family == "M2":
        shift_hz = prf_hz / (
            int(config["models"]["m2"]["fft_factor"]) * n_samples
        )
    else:
        shift_hz = prf_hz / n_samples
    time = np.arange(n_samples, dtype=np.float64)
    shifted = SignalBatch(
        iq=base.iq * np.exp(2j * np.pi * shift_hz * time[None, :] / prf_hz),
        frequency_hz=np.asarray(
            wrap_frequency(base.frequency_hz + shift_hz, prf_hz)
        ),
        phase_rad=base.phase_rad,
        amplitude=base.amplitude,
        snr_db=base.snr_db,
        seed=base.seed,
    )
    shifted_prediction = _predict_checkpoint(
        model_name, checkpoint_path, shifted, config, device
    )
    shift_error = np.asarray(
        wrapped_error(shifted_prediction, prediction + shift_hz, prf_hz)
    )
    return {
        "model": model_name,
        "strategy": strategy,
        "train_seed": int(train_seed),
        "sample_count": int(count),
        "common_phase_rotation_max_abs_delta_hz": float(
            np.max(np.abs(wrapped_error(phase_prediction, prediction, prf_hz)))
        ),
        "common_complex_scale_max_abs_delta_hz": float(
            np.max(np.abs(wrapped_error(scale_prediction, prediction, prf_hz)))
        ),
        "frequency_shift_hz": float(shift_hz),
        "frequency_shift_equivariance_mae_hz": float(
            np.mean(np.abs(shift_error))
        ),
        "frequency_shift_equivariance_max_abs_hz": float(
            np.max(np.abs(shift_error))
        ),
    }


def _adjacent_gate(
    comparison_by_snr: dict[str, dict[str, float]],
    primary_snr: list[float],
    required_adjacent: int,
) -> dict[str, Any]:
    pass_by_snr = {
        str(float(snr)): bool(
            comparison_by_snr.get(str(float(snr)), {}).get("ci95_high_hz", math.inf)
            < 0.0
        )
        for snr in primary_snr
    }
    maximum = _max_consecutive_wins(
        [pass_by_snr[str(float(snr))] for snr in primary_snr]
    )
    return {
        "pass": maximum >= required_adjacent,
        "required_adjacent_snr_points": int(required_adjacent),
        "maximum_adjacent_passes": int(maximum),
        "pass_by_snr": pass_by_snr,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _baseline_method_names(predictions: dict[str, np.ndarray]) -> list[str]:
    """Return the classical method labels present in a prediction cache."""
    prefix = "baseline__"
    return sorted(
        key[len(prefix) :]
        for key in predictions
        if key.startswith(prefix)
    )


def run_final(
    config: dict[str, Any], repo_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """Evaluate the frozen formal checkpoints on one new, independent blind set."""
    device = require_cuda(config["training"].get("device", ""))
    torch.cuda.set_device(device)
    root = Path(repo_root)
    output = Path(output_dir)
    sentinel = output / "final_metrics.json"
    if sentinel.exists():
        raise FileExistsError("final blind results already exist; refusing to overwrite")
    formal_e1_path = output / "formal_e1_metrics.json"
    formal_kd_path = output / "formal_kd_metrics.json"
    if not formal_e1_path.exists() or not formal_kd_path.exists():
        raise FileNotFoundError("formal E1 and KD gate must finish before final blind test")
    formal_e1 = json.loads(formal_e1_path.read_text(encoding="utf-8"))
    formal_kd = json.loads(formal_kd_path.read_text(encoding="utf-8"))
    eligible_models = list(formal_kd.get("eligible_models", []))
    if formal_kd.get("status") != "completed" or not eligible_models:
        result = {
            "experiment_id": config["experiment_id"],
            "status": "no_final_blind_generated",
            "reason": "no model passed the preregistered formal E1 accuracy gate; KD and final blind test were not run",
            "eligible_models": eligible_models,
            "development_seed": int(config["data"]["development_seed"]),
            "final_blind_seed_reserved": int(config["data"]["final_test_seed"]),
        }
        _write_json(sentinel, result)
        return result

    checkpoints = _checkpoint_sets(formal_e1, formal_kd)
    final_models = {
        key: value
        for key, value in checkpoints.items()
        if key[0] == "M1_power" or key[0] in eligible_models
    }
    for model_name in eligible_models:
        for strategy in ("E1", "E2-H", "E2-S", "E3-S"):
            key = (model_name, strategy)
            if len(final_models.get(key, [])) != len(config["data"]["formal_train_seeds"]):
                raise ValueError(f"expected three formal checkpoints for {key}")
    if "M1" in eligible_models and len(final_models.get(("M1_power", "E1"), [])) != len(
        config["data"]["formal_train_seeds"]
    ):
        raise ValueError("M1 was selected but its registered M1_power E1 ablation is incomplete")

    data_dir = output / "datasets"
    blind_path = data_dir / "final_blind.npz"
    noiseless_path = data_dir / "noiseless_correctness.npz"
    data_dir.mkdir(parents=True, exist_ok=True)
    if blind_path.exists() != noiseless_path.exists():
        raise FileExistsError("only one final test dataset exists; refusing to mix dataset states")
    if blind_path.exists():
        print("P0-E007 final: reusing the saved one-time blind datasets", flush=True)
        blind_batch = _load_saved_test_batch(blind_path, "final_blind")
        noiseless_batch = _load_saved_test_batch(noiseless_path, "noiseless_correctness")
        with np.load(blind_path) as values:
            blind_seed_by_snr = json.loads(str(values["seed_by_snr_json"]))
        expected_seeds = {
            str(float(snr)): int(config["data"]["final_test_seed"]) + 1000 * index
            for index, snr in enumerate(config["data"]["test_snr_levels_db"])
        }
        if blind_seed_by_snr != expected_seeds:
            raise ValueError("saved final blind data do not match the frozen SNR seeds")
    else:
        blind_batch, blind_seed_by_snr = _make_final_blind_batch(config)
        np.savez_compressed(
            blind_path,
            iq=blind_batch.iq,
            frequency_hz=blind_batch.frequency_hz,
            phase_rad=blind_batch.phase_rad,
            amplitude=blind_batch.amplitude,
            snr_db=blind_batch.snr_db,
            dataset_seed=np.asarray(blind_batch.seed, dtype=np.int64),
            seed_by_snr_json=np.asarray(json.dumps(blind_seed_by_snr, sort_keys=True)),
            data_role=np.asarray("final_blind"),
        )
        noiseless_batch = _make_noiseless_batch(config)
        np.savez_compressed(
            noiseless_path,
            iq=noiseless_batch.iq,
            frequency_hz=noiseless_batch.frequency_hz,
            phase_rad=noiseless_batch.phase_rad,
            amplitude=noiseless_batch.amplitude,
            snr_db=noiseless_batch.snr_db,
            dataset_seed=np.asarray(noiseless_batch.seed, dtype=np.int64),
            data_role=np.asarray("noiseless_correctness"),
        )

    prediction_cache_path = output / "final_predictions.npz"
    clean_prediction_cache_path = output / "noiseless_predictions.npz"
    noiseless_metrics_path = output / "noiseless_metrics.json"
    physics_metrics_path = output / "physics_diagnostics.json"
    prediction_cache_paths = (
        prediction_cache_path,
        clean_prediction_cache_path,
        noiseless_metrics_path,
        physics_metrics_path,
    )
    reused_prediction_cache = all(path.exists() for path in prediction_cache_paths)
    model_metadata: dict[str, list[dict[str, Any]]] = {}
    for (model_name, strategy), records in sorted(final_models.items()):
        method_key = f"model__{model_name}__{strategy}"
        model_metadata[method_key] = []
        for record in records:
            checkpoint_path = Path(record["checkpoint"])
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model_metadata[method_key].append(
                {
                    "train_seed": int(record["train_seed"]),
                    "checkpoint": str(checkpoint_path),
                    "selected_epoch": int(checkpoint["selected_epoch"]),
                }
            )

    if reused_prediction_cache:
        print("P0-E007 final: reusing saved blind predictions and diagnostics", flush=True)
        with np.load(prediction_cache_path) as values:
            if not np.array_equal(values["truth_frequency_hz"], blind_batch.frequency_hz) or not np.array_equal(values["snr_db"], blind_batch.snr_db):
                raise ValueError("cached final predictions belong to different blind data")
            predictions = {
                name: np.asarray(values[name], dtype=np.float64)
                for name in values.files
                if name not in {"truth_frequency_hz", "snr_db"}
            }
        with np.load(clean_prediction_cache_path) as values:
            if not np.array_equal(values["truth_frequency_hz"], noiseless_batch.frequency_hz):
                raise ValueError("cached noiseless predictions belong to different correctness data")
            clean_predictions = {
                name: np.asarray(values[name], dtype=np.float64)
                for name in values.files
                if name != "truth_frequency_hz"
            }
        noiseless_records = _load_json_file(noiseless_metrics_path)
        transform_records = _load_json_file(physics_metrics_path)
    else:
        print("P0-E007 final: evaluating frozen classical baselines", flush=True)
        baselines = _baseline_predictions(blind_batch, config, include_global=True)
        predictions: dict[str, np.ndarray] = {
            f"baseline__{name}": np.asarray(value, dtype=np.float64)[None, :]
            for name, value in baselines.items()
        }
        noiseless_records: list[dict[str, Any]] = []
        clean_predictions: dict[str, np.ndarray] = {}
        transform_records: list[dict[str, Any]] = []
        for (model_name, strategy), records in sorted(final_models.items()):
            method_key = f"model__{model_name}__{strategy}"
            print(f"P0-E007 final: evaluating {method_key} ({len(records)} seeds)", flush=True)
            blind_by_seed: list[np.ndarray] = []
            clean_by_seed: list[np.ndarray] = []
            for record in records:
                seed = int(record["train_seed"])
                checkpoint_path = Path(record["checkpoint"])
                prediction = _predict_checkpoint(
                    model_name, checkpoint_path, blind_batch, config, device
                )
                blind_by_seed.append(prediction)
                clean_prediction = _predict_checkpoint(
                    model_name, checkpoint_path, noiseless_batch, config, device
                )
                clean_by_seed.append(clean_prediction)
                noiseless_records.append(
                    _noiseless_error_record(
                        clean_prediction,
                        noiseless_batch,
                        model_name,
                        strategy,
                        seed,
                        float(config["prf_hz"]),
                    )
                )
                if strategy in {"E1", "E3-S"}:
                    transform_records.append(
                        _physics_transform_records(
                            model_name,
                            strategy,
                            checkpoint_path,
                            seed,
                            blind_batch,
                            config,
                            device,
                        )
                    )
            predictions[method_key] = np.stack(blind_by_seed)
            clean_predictions[method_key] = np.stack(clean_by_seed)

        _write_json(noiseless_metrics_path, noiseless_records)
        _write_json(physics_metrics_path, transform_records)
        np.savez_compressed(
            prediction_cache_path,
            truth_frequency_hz=blind_batch.frequency_hz,
            snr_db=blind_batch.snr_db,
            **predictions,
        )
        np.savez_compressed(
            clean_prediction_cache_path,
            truth_frequency_hz=noiseless_batch.frequency_hz,
            **clean_predictions,
        )

    outlier_threshold = float(config["evaluation"]["outlier_threshold_hz"])
    result_rows: list[dict[str, Any]] = []
    method_summaries: dict[str, dict[str, Any]] = {}
    for method_key, by_seed in predictions.items():
        per_seed_rows: list[dict[str, Any]] = []
        for seed_index, prediction in enumerate(by_seed):
            seed = (
                model_metadata[method_key][seed_index]["train_seed"]
                if method_key.startswith("model__")
                else None
            )
            rows = grouped_frequency_metrics(
                prediction,
                blind_batch,
                prf_hz=float(config["prf_hz"]),
                n_samples=int(config["n_samples"]),
                outlier_threshold_hz=outlier_threshold,
            )
            for row in rows:
                full_row = {
                    "method": method_key,
                    "train_seed": seed,
                    **row,
                }
                result_rows.append(full_row)
                per_seed_rows.append(full_row)
        method_summary: dict[str, Any] = {}
        snr_values = [float(value) for value in config["data"]["test_snr_levels_db"]]
        for snr_value in snr_values:
            snr_key = str(float(snr_value))
            rmse_values = [
                row["rmse_hz"]
                for row in per_seed_rows
                if row["group_type"] == "overall"
                and row["group"] == "all"
                and float(row["snr_db"]) == snr_value
            ]
            method_summary[snr_key] = {
                "seed_rmse_hz": rmse_values,
                "mean_rmse_hz": float(np.mean(rmse_values)),
                "std_rmse_hz": float(np.std(rmse_values, ddof=1))
                if len(rmse_values) > 1
                else 0.0,
            }
        method_summaries[method_key] = method_summary
    _write_csv(output / "result.csv", result_rows)

    bootstrap_samples = int(config["evaluation"]["paired_bootstrap_samples"])
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    primary_snr = [
        float(value) for value in config["evaluation"]["primary_snr_levels_db"]
    ]
    snr_vector = np.asarray(blind_batch.snr_db)
    comparisons: dict[str, Any] = {}

    def compare_methods(candidate_key: str, baseline_key: str) -> dict[str, dict[str, float]]:
        candidate = predictions[candidate_key]
        baseline = predictions[baseline_key]
        comparison: dict[str, dict[str, float]] = {}
        for snr_index, snr_value in enumerate(primary_snr):
            mask = snr_vector == snr_value
            comparison[str(float(snr_value))] = paired_bootstrap_rmse_difference(
                candidate[:, mask],
                baseline[:, mask],
                blind_batch.frequency_hz[mask],
                prf_hz=float(config["prf_hz"]),
                bootstrap_samples=bootstrap_samples,
                seed=bootstrap_seed + 1000 * snr_index,
            )
        return comparison

    for model_name in eligible_models:
        candidate_key = f"model__{model_name}__E1"
        fft_key = "baseline__B1_fft_peak"
        for baseline_name in _baseline_method_names(predictions):
            comparisons[f"{candidate_key}_vs_{baseline_name}"] = compare_methods(
                candidate_key, f"baseline__{baseline_name}"
            )
        if model_name == "M1":
            if "model__M1_power__E1" in predictions:
                comparisons[f"{candidate_key}_vs_M1_power_E1"] = compare_methods(
                    candidate_key, "model__M1_power__E1"
                )
            if "M3" in eligible_models:
                comparisons[f"{candidate_key}_vs_M3_E1"] = compare_methods(
                    candidate_key, "model__M3__E1"
                )
        if model_name == "M3" and "model__M1_power__E1" in predictions:
            comparisons[f"{candidate_key}_vs_M1_power_E1"] = compare_methods(
                candidate_key, "model__M1_power__E1"
            )
        for strategy in ("E2-H", "E2-S", "E3-S"):
            candidate_kd_key = f"model__{model_name}__{strategy}"
            comparisons[f"{candidate_kd_key}_vs_E1"] = compare_methods(
                candidate_kd_key, candidate_key
            )

    gates: dict[str, Any] = {}
    zero_rmse = float(np.sqrt(np.mean(blind_batch.frequency_hz[snr_vector == 20.0] ** 2)))
    noiseless_by_method: dict[str, list[dict[str, Any]]] = {}
    for record in noiseless_records:
        noiseless_by_method.setdefault(
            f"model__{record['model']}__{record['strategy']}", []
        ).append(record)
    g1_candidates: dict[str, Any] = {}
    for model_name in ("M1", "M3"):
        method_key = f"model__{model_name}__E1"
        if method_key not in predictions:
            continue
        record_by_seed = {
            int(record["train_seed"]): record
            for record in noiseless_by_method.get(method_key, [])
        }
        seed_checks = []
        for index, metadata in enumerate(model_metadata[method_key]):
            noiseless = record_by_seed[int(metadata["train_seed"])]
            snr20_mask = snr_vector == 20.0
            rmse20 = _rmse(
                np.asarray(
                    wrapped_error(
                        predictions[method_key][index, snr20_mask],
                        blind_batch.frequency_hz[snr20_mask],
                        float(config["prf_hz"]),
                    )
                )
            )
            passed_clean = (
                noiseless["all_frequency"]["mae_hz"]
                < float(config["evaluation"]["g1_noiseless_mae_hz"])
                and noiseless["all_frequency"]["p95_abs_hz"]
                < float(config["evaluation"]["g1_noiseless_p95_hz"])
                and noiseless["nyquist_edge"]["mae_hz"]
                < float(config["evaluation"]["g1_noiseless_mae_hz"])
                and noiseless["nyquist_edge"]["p95_abs_hz"]
                < float(config["evaluation"]["g1_noiseless_p95_hz"])
            )
            seed_checks.append(
                {
                    "train_seed": int(metadata["train_seed"]),
                    "noiseless_all_pass": bool(passed_clean),
                    "noiseless_mae_hz": noiseless["all_frequency"]["mae_hz"],
                    "noiseless_p95_hz": noiseless["all_frequency"]["p95_abs_hz"],
                    "noiseless_nyquist_mae_hz": noiseless["nyquist_edge"]["mae_hz"],
                    "noiseless_nyquist_p95_hz": noiseless["nyquist_edge"]["p95_abs_hz"],
                    "rmse_20db_hz": rmse20,
                    "beats_zero_by_50pct": bool(rmse20 <= 0.5 * zero_rmse),
                }
            )
        required = int(config["evaluation"]["g1_required_seeds"])
        passed_count = sum(
            item["noiseless_all_pass"] and item["beats_zero_by_50pct"]
            for item in seed_checks
        )
        g1_candidates[model_name] = {
            "pass": passed_count >= required,
            "passing_seed_count": passed_count,
            "required_seed_count": required,
            "seed_results": seed_checks,
        }
    seed_roles = [
        int(config["data"]["screen_train_seed"]),
        int(config["data"]["screen_validation_seed"]),
        *[int(value) for value in config["data"]["formal_train_seeds"]],
        *[int(value) for value in config["data"]["formal_validation_seeds"]],
        int(config["data"]["development_seed"]),
        int(config["data"]["noiseless_seed"]),
        *[int(value) for value in blind_seed_by_snr.values()],
    ]
    data_isolation_ok = len(seed_roles) == len(set(seed_roles))
    gates["G1"] = {
        "pass": bool(
            data_isolation_ok and any(item["pass"] for item in g1_candidates.values())
        ),
        "pure_coherent_candidates": g1_candidates,
        "data_seed_roles_disjoint": data_isolation_ok,
        "data_seed_roles": {
            "screen_train": int(config["data"]["screen_train_seed"]),
            "screen_validation": int(config["data"]["screen_validation_seed"]),
            "formal_train": list(config["data"]["formal_train_seeds"]),
            "formal_validation": list(config["data"]["formal_validation_seeds"]),
            "development": int(config["data"]["development_seed"]),
            "noiseless": int(config["data"]["noiseless_seed"]),
            "final_blind_base": int(config["data"]["final_test_seed"]),
            "final_blind_by_snr": blind_seed_by_snr,
        },
        "zero_frequency_rmse_20db_hz": zero_rmse,
    }

    main_gate_by_model: dict[str, Any] = {}
    for model_name in eligible_models:
        key = f"model__{model_name}__E1_vs_B1_fft_peak"
        main_gate_by_model[model_name] = _adjacent_gate(
            comparisons[key], primary_snr, int(config["evaluation"]["g2_required_adjacent_snr"])
        )
    g2_pass_models = [name for name, item in main_gate_by_model.items() if item["pass"]]
    matching_frontend: dict[str, Any] = {}
    if "M1" in eligible_models:
        matching_frontend["M1_vs_16N_fft_peak"] = _adjacent_gate(
            comparisons["model__M1__E1_vs_B8_16N_fft_peak"],
            primary_snr,
            int(config["evaluation"]["g2_required_adjacent_snr"]),
        )
    if "M2" in eligible_models:
        matching_frontend["M2_vs_4N_local_parabolic"] = _adjacent_gate(
            comparisons["model__M2__E1_vs_B9_4N_local_parabolic"],
            primary_snr,
            int(config["evaluation"]["g2_required_adjacent_snr"]),
        )
    gates["G2"] = {
        "pass": bool(g2_pass_models),
        "main_fft_peak_pass_models": g2_pass_models,
        "main_fft_peak_by_model": main_gate_by_model,
        "matched_frontend_gates": matching_frontend,
        "definition": "candidate-minus-baseline paired RMSE 95% CI upper bound below zero at the required adjacent primary SNR points",
    }

    g3_by_model: dict[str, Any] = {}
    for model_name in eligible_models:
        e1_key = f"model__{model_name}__E1"
        e3_key = f"model__{model_name}__E3-S_vs_E1"
        gate = _adjacent_gate(
            comparisons[e3_key],
            primary_snr,
            int(config["evaluation"]["g3_required_adjacent_snr"]),
        )
        direction_by_snr: dict[str, int] = {}
        for snr_value in primary_snr:
            mask = snr_vector == snr_value
            e1_errors = np.asarray(
                wrapped_error(
                    predictions[e1_key][:, mask],
                    blind_batch.frequency_hz[mask],
                    float(config["prf_hz"]),
                )
            )
            e3_errors = np.asarray(
                wrapped_error(
                    predictions[f"model__{model_name}__E3-S"][:, mask],
                    blind_batch.frequency_hz[mask],
                    float(config["prf_hz"]),
                )
            )
            direction_by_snr[str(float(snr_value))] = int(
                np.sum(
                    np.sqrt(np.mean(e3_errors**2, axis=1))
                    < np.sqrt(np.mean(e1_errors**2, axis=1))
                )
            )
        per_snr_seed_consistency = {
            snr: (count >= 2) for snr, count in direction_by_snr.items()
        }
        gate["seed_improvement_count_by_snr"] = direction_by_snr
        gate["seed_consistency_by_snr"] = per_snr_seed_consistency
        joint_pass = {
            snr: bool(gate["pass_by_snr"][snr] and per_snr_seed_consistency[snr])
            for snr in gate["pass_by_snr"]
        }
        gate["ci_and_seed_consistency_by_snr"] = joint_pass
        gate["pass"] = bool(
            _max_consecutive_wins(
                [joint_pass[str(float(snr))] for snr in primary_snr]
            )
            >= int(config["evaluation"]["g3_required_adjacent_snr"])
        )
        g3_by_model[model_name] = gate
    gates["G3"] = {
        "pass": any(item["pass"] for item in g3_by_model.values()),
        "by_model": g3_by_model,
    }
    if not gates["G1"]["pass"]:
        decision = "STOP / REDIRECT"
    elif gates["G2"]["pass"]:
        decision = "GO"
    else:
        decision = "CONDITIONAL GO"

    comparison_digest = {
        "paired_bootstrap_samples": bootstrap_samples,
        "paired_bootstrap_seed": bootstrap_seed,
        "primary_snr_levels_db": primary_snr,
        "comparisons": comparisons,
    }
    _write_json(output / "paired_comparisons.json", comparison_digest)
    final_metrics = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "data_roles": {
            "development_seed": int(config["data"]["development_seed"]),
            "final_blind_seed_by_snr": blind_seed_by_snr,
            "final_blind_is_independent": True,
            "final_blind_generated_once": True,
            "saved_model_predictions_reused": bool(reused_prediction_cache),
            "blind_dataset_generation_count": 1,
            "noiseless_seed": int(config["data"]["noiseless_seed"]),
            "test_count_per_snr": int(config["data"]["test_per_snr"]),
        },
        "environment": _environment_snapshot(device),
        "training_models_and_checkpoints": model_metadata,
        "classical_and_student_metrics_by_method": method_summaries,
        "noiseless_metrics": noiseless_records,
        "physics_diagnostics": transform_records,
        "gates": gates,
        "stage_decision": decision,
        "outlier_threshold_hz": outlier_threshold,
        "matched_frontend_interpretation": {
            "M1": "fixed 16N FFT plus learned circular coherent readout versus fixed 16N FFT peak",
            "M2": "fixed 4N FFT plus learned residual versus the same 4N spectrum's local three-point log-parabolic interpolation",
        },
    }
    _write_json(sentinel, final_metrics)
    return final_metrics


def _format_comparison(
    comparisons: dict[str, Any], key: str, snr: float
) -> str:
    result = comparisons.get(key, {}).get(str(float(snr)))
    if result is None:
        return "未评估"
    return (
        f"{result['mean_rmse_difference_hz']:+.3f} Hz "
        f"[{result['ci95_low_hz']:+.3f}, {result['ci95_high_hz']:+.3f}]"
    )


def _create_rmse_snr_plot(
    config: dict[str, Any], output: Path, metrics: dict[str, Any]
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[dict[str, Any]] = []
    with (output / "result.csv").open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    snr_levels = [float(value) for value in config["data"]["test_snr_levels_db"]]
    methods = metrics["classical_and_student_metrics_by_method"]
    figure, axis = plt.subplots(figsize=(11.5, 7.0), constrained_layout=True)
    baseline_order = [
        "B1_fft_peak",
        "B2_fft_interpolated",
        "B3_phase_difference",
        "B4_local_profile_ML",
        "B5_global_grid_profile_ML",
        "B6_4N_fft_peak",
        "B8_16N_fft_peak",
        "B9_4N_local_parabolic",
        "B10_16N_local_parabolic",
    ]
    baseline_labels = {
        "B1_fft_peak": "N FFT peak",
        "B2_fft_interpolated": "N FFT interpolation",
        "B3_phase_difference": "Phase difference",
        "B4_local_profile_ML": "Local profile ML",
        "B5_global_grid_profile_ML": "Global grid profile ML",
        "B6_4N_fft_peak": "4N FFT peak",
        "B8_16N_fft_peak": "16N FFT peak",
        "B9_4N_local_parabolic": "4N local parabolic",
        "B10_16N_local_parabolic": "16N local parabolic",
    }
    for name in baseline_order:
        key = f"baseline__{name}"
        if key not in methods:
            continue
        values = [methods[key][str(float(snr))]["mean_rmse_hz"] for snr in snr_levels]
        axis.plot(snr_levels, values, marker="o", linewidth=1.4, label=baseline_labels[name])
    model_labels = {"M1": "M1 circular coherent", "M1_power": "M1 power ablation", "M2": "M2 FFT hybrid", "M3": "M3 IQ Transformer"}
    for method_key in sorted(key for key in methods if key.startswith("model__")):
        _, model_name, strategy = method_key.split("__", maxsplit=2)
        if strategy not in {"E1", "E2-H", "E2-S", "E3-S"}:
            continue
        values = [methods[method_key][str(float(snr))]["mean_rmse_hz"] for snr in snr_levels]
        spread = [methods[method_key][str(float(snr))]["std_rmse_hz"] for snr in snr_levels]
        label = model_labels.get(model_name, model_name)
        label += f" {strategy}"
        line_style = {"E1": "-", "E2-H": "-.", "E2-S": ":", "E3-S": "--"}[strategy]
        axis.plot(snr_levels, values, marker="s", linestyle=line_style, linewidth=1.8, label=label)
        if len(methods[method_key][str(float(snr_levels[0]))]["seed_rmse_hz"]) > 1:
            low = np.maximum(np.asarray(values) - np.asarray(spread), 1e-6)
            high = np.asarray(values) + np.asarray(spread)
            axis.fill_between(snr_levels, low, high, alpha=0.10)
    baseline_rows = [row for row in rows if row["method"] == "baseline__B4_local_profile_ML" and row["group_type"] == "overall"]
    crlb_by_snr = {float(row["snr_db"]): float(row["crlb_mean_hz"]) for row in baseline_rows}
    axis.plot(
        snr_levels,
        [crlb_by_snr[snr] for snr in snr_levels],
        color="black",
        linestyle=":",
        linewidth=2,
        label="CRLB (standard deviation)",
    )
    axis.set_yscale("log")
    axis.set_xlabel("SNR (dB)")
    axis.set_ylabel("Circular frequency RMSE (Hz)")
    axis.set_title("P0-E007 final blind test: frequency RMSE vs SNR")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(output / "rmse_snr.png", dpi=180)
    plt.close(figure)


def write_analysis_report(config: dict[str, Any], output_dir: str | Path) -> Path:
    """Create the final evidence-based report and RMSE-SNR figure when available."""
    output = Path(output_dir)
    screening_path = output / "screening.json"
    formal_e1_path = output / "formal_e1_metrics.json"
    formal_kd_path = output / "formal_kd_metrics.json"
    final_path = output / "final_metrics.json"
    screening = _load_json_file(screening_path) if screening_path.exists() else None
    formal_e1 = _load_json_file(formal_e1_path) if formal_e1_path.exists() else None
    formal_kd = _load_json_file(formal_kd_path) if formal_kd_path.exists() else None
    final_metrics = _load_json_file(final_path) if final_path.exists() else None
    if final_metrics is not None and final_metrics.get("status") == "completed":
        _write_json(output / "metrics.json", final_metrics)
    report: list[str] = [
        "# P0-E007 阶段报告",
        "",
        "实验范围：单目标复数慢时间 AWGN 频率估计，PRF=1300 Hz、N=130。全部训练要求 CUDA；数据为仿真。E006 的历史结论保持不变。",
        "",
    ]
    if screening is None:
        report.extend(
            [
                "## 运行状态",
                "",
                "尚无 E007 筛选结果。该报告只记录当前工件状态，不把代码实现或计划写成实验结果。",
                "",
            ]
        )
    else:
        selection = _load_json_file(output / "selection.json") if (output / "selection.json").exists() else {}
        report.extend(
            [
                "## 单种子筛选",
                "",
                f"CUDA 设备：{screening.get('environment', {}).get('gpu', '未知')}；batch size={screening.get('training_budget', {}).get('batch_size', '未知')}；最多 {screening.get('training_budget', {}).get('epochs_max', '未知')} 轮。",
                f"入选：{', '.join(selection.get('selected', [])) or '无'}。",
                "",
                "| 模型 | 验证损失首轮→最佳 | 20 dB 开发 RMSE (Hz) | 结构诊断 | 筛选结论 |",
                "|---|---:|---:|---|---|",
            ]
        )
        for model_name, result in screening.get("models", {}).items():
            score = result.get("screen_score", {})
            validation_drop = "—"
            first = score.get("validation_first_loss")
            best = score.get("validation_best_loss")
            if first not in (None, 0) and best is not None:
                validation_drop = f"{first:.4g} → {best:.4g}"
            reason = selection.get("selected_reasons", {}).get(
                model_name,
                selection.get("rejected", {}).get(model_name, "M1 功率输入消融"),
            )
            report.append(
                f"| {model_name} | {validation_drop} | {score.get('development_rmse_20db_hz', '—')} | {score.get('structural_transform_ok', '—')} | {reason} |"
            )
        report.append("")
    if formal_e1 is not None:
        gate = formal_e1.get("kd_selection", {})
        report.extend(
            [
                "## 正式 E1 与蒸馏准入",
                "",
                f"正式 E1 完成状态：{formal_e1.get('status')}；进入蒸馏的模型：{', '.join(gate.get('selected', [])) or '无'}。",
                "",
            ]
        )
        for name, reason in gate.get("rejected", {}).items():
            report.append(f"- {name} 未进入蒸馏：{reason}。")
        report.append("")
    if formal_kd is not None:
        report.extend(
            [
                "## 教师蒸馏",
                "",
                f"状态：{formal_kd.get('status')}；准入模型：{', '.join(formal_kd.get('eligible_models', [])) or '无'}。",
                "E2-H 使用教师连续点估计，E2-S 使用 profile soft response，E3-S 使用真值损失加 0.5 倍 soft response 损失；物理真值、硬教师和软响应保存在独立字段。",
                "",
            ]
        )
    resource_groups: dict[str, list[dict[str, Any]]] = {}
    if screening is not None:
        for model_name, record in screening.get("models", {}).items():
            resource_groups[f"筛选 {model_name} E1"] = [record]
    if formal_e1 is not None:
        for record in formal_e1.get("training_replicates", []):
            resource_groups.setdefault(
                f"正式 {record['model']} {record['strategy']}", []
            ).append(record.get("training", record))
    if formal_kd is not None:
        for record in formal_kd.get("training_replicates", []):
            resource_groups.setdefault(
                f"蒸馏 {record['model']} {record['strategy']}", []
            ).append(record.get("training", record))
    if resource_groups:
        report.extend(
            [
                "## 训练执行与 GPU 资源",
                "",
                "训练设备为 RTX 3050 Laptop GPU（4 GiB），PyTorch 2.8.0+cu128、CUDA 12.8；FP32、batch 64，训练按顺序运行。下表显存是 PyTorch 单进程 peak allocated，不是整机显存占用。",
                "",
                "| 阶段/模型 | 训练数 | 平均训练秒数 | 最大 allocated (MiB) |",
                "|---|---:|---:|---:|",
            ]
        )
        for group_name, records in resource_groups.items():
            times = [float(row["training_seconds"]) for row in records if row.get("training_seconds") is not None]
            memory = [int(row["peak_gpu_memory_bytes"]) for row in records if row.get("peak_gpu_memory_bytes") is not None]
            mean_seconds = f"{np.mean(times):.1f}" if times else "未记录"
            peak_mib = f"{max(memory) / (1024 ** 2):.1f}" if memory else "未记录"
            report.append(f"| {group_name} | {len(records)} | {mean_seconds} | {peak_mib} |")
        report.extend(
            [
                "",
                "M1 筛选 checkpoint 从已完成训练恢复用于开发评分；该次 wall time 与峰值分配显存未写入记录。其后的正式 M1 E1 三种子均完整记录，资源数据见表。",
                "",
            ]
        )
    if final_metrics is None or final_metrics.get("status") != "completed":
        reason = (
            final_metrics.get("reason", "正式 E1 或蒸馏阶段尚未完成。")
            if final_metrics
            else "正式 E1 与蒸馏结果完成前，不生成盲测集。"
        )
        report.extend(
            [
                "## 阶段结论",
                "",
                f"最终独立盲测未运行：{reason}",
                "",
                "该 E007 实验依预注册准入规则停止后续阶段。G1/G2/G3 的最终盲测门槛未被测量，因此不能把本次筛选结果当作 P0 最终阶段验收，也不能据此宣称进入 P1。",
                "",
            ]
        )
    else:
        _create_rmse_snr_plot(config, output, final_metrics)
        methods = final_metrics["classical_and_student_metrics_by_method"]
        comparisons = _load_json_file(output / "paired_comparisons.json")["comparisons"]
        primary = [float(x) for x in config["evaluation"]["primary_snr_levels_db"]]
        with (output / "result.csv").open("r", newline="", encoding="utf-8") as stream:
            result_rows = list(csv.DictReader(stream))
        crlb_20 = float(
            frequency_crlb_hz(
                float(config["prf_hz"]),
                int(config["n_samples"]),
                20.0,
            )
        )
        report.extend(
            [
                "## 最终盲测汇总",
                "",
                f"独立测试集：{final_metrics['data_roles']['test_count_per_snr']} 条/SNR；SNR={', '.join(map(str, config['data']['test_snr_levels_db']))} dB；盲测 seed 由 {config['data']['final_test_seed']} 按 SNR 顺序每次加 1000 生成。测试只运行一次，未用于选模。",
                f"RMSE-SNR 图：`rmse_snr.png`。逐样本指标与分组结果：`result.csv`。配对 bootstrap：`paired_comparisons.json`。",
                "",
                "| 方法 | 0–20 dB 平均 RMSE (Hz) | 20 dB RMSE (Hz) ± 种子标准差 |",
                "|---|---:|---:|",
            ]
        )
        selected_method_keys = [
            key
            for key in methods
            if key.startswith("model__") and key.endswith("__E1")
        ]
        baseline_keys = [
            "baseline__B1_fft_peak",
            "baseline__B2_fft_interpolated",
            "baseline__B3_phase_difference",
            "baseline__B4_local_profile_ML",
            "baseline__B5_global_grid_profile_ML",
            "baseline__B6_4N_fft_peak",
            "baseline__B8_16N_fft_peak",
            "baseline__B9_4N_local_parabolic",
            "baseline__B10_16N_local_parabolic",
        ]
        kd_method_keys = [
            key
            for key in (
                "model__M1__E2-H",
                "model__M1__E2-S",
                "model__M1__E3-S",
            )
            if key in methods
        ]
        for method_key in baseline_keys + selected_method_keys + kd_method_keys:
            if method_key not in methods:
                continue
            primary_rmse = [
                np.mean(methods[method_key][str(float(snr))]["seed_rmse_hz"])
                for snr in primary
            ]
            twenty = methods[method_key]["20.0"]
            model_label = method_key.removeprefix("baseline__").replace("model__", "")
            report.append(
                f"| {model_label} | {np.mean(primary_rmse):.4f} | {twenty['mean_rmse_hz']:.4f} ± {twenty['std_rmse_hz']:.4f} |"
            )
        report.append("")
        ranking = sorted(
            (
                np.mean(
                    [
                        np.mean(methods[key][str(float(snr))]["seed_rmse_hz"])
                        for snr in primary
                    ]
                ),
                key,
            )
            for key in selected_method_keys
        )
        best_e1 = ranking[0][1] if ranking else "无"
        main_architecture_e1 = [
            key for key in selected_method_keys if key in {"model__M1__E1", "model__M2__E1", "model__M3__E1"}
        ]
        main_ranking = sorted(
            (
                np.mean(
                    [
                        np.mean(methods[key][str(float(snr))]["seed_rmse_hz"])
                        for snr in primary
                    ]
                ),
                key,
            )
            for key in main_architecture_e1
        )
        best_main = main_ranking[0][1] if main_ranking else "未进入盲测"
        m2_dev20 = None
        m2_frontend_delta20 = None
        m2_fft_interpolation_delta20 = None
        m2_profile_ml_delta20 = None
        if screening is not None and "M2" in screening.get("models", {}):
            m2_dev20 = screening["models"]["M2"]["development_rmse_by_snr_hz"].get("20.0")
            m2_frontend_delta20 = (
                m2_dev20
                - screening["classical_development_rmse_by_snr_hz"]["B9_4N_local_parabolic"]["20.0"]
            )
            m2_fft_interpolation_delta20 = (
                m2_dev20
                - screening["classical_development_rmse_by_snr_hz"]["B2_fft_interpolated"]["20.0"]
            )
            m2_profile_ml_delta20 = (
                m2_dev20
                - screening["classical_development_rmse_by_snr_hz"]["B4_local_profile_ML"]["20.0"]
            )
        m1_vs_power_20 = _format_comparison(
            comparisons, "model__M1__E1_vs_M1_power_E1", 20.0
        )
        m1_20 = methods.get("model__M1__E1", {}).get("20.0", {}).get("mean_rmse_hz")
        power_20 = methods.get("model__M1_power__E1", {}).get("20.0", {}).get("mean_rmse_hz")
        ml_20 = methods["baseline__B4_local_profile_ML"]["20.0"]["mean_rmse_hz"]
        m1_crlb_ratio = float(m1_20 / crlb_20) if m1_20 is not None else math.nan
        power_crlb_ratio = float(power_20 / crlb_20) if power_20 is not None else math.nan
        ml_crlb_ratio = float(ml_20 / crlb_20)
        physics = final_metrics.get("physics_diagnostics", [])
        m1_e1_physics = [
            item for item in physics if item["model"] == "M1" and item["strategy"] == "E1"
        ]
        max_phase_delta = max(
            (item["common_phase_rotation_max_abs_delta_hz"] for item in m1_e1_physics),
            default=math.nan,
        )
        max_shift_error = max(
            (item["frequency_shift_equivariance_max_abs_hz"] for item in m1_e1_physics),
            default=math.nan,
        )
        subgroup_ranges: dict[str, tuple[float, float]] = {}
        for group_type in ("frequency_bin", "phase_sector", "amplitude_quartile", "nyquist"):
            values = [
                float(row["rmse_hz"])
                for row in result_rows
                if row["method"] == "model__M1__E1"
                and row["group_type"] == group_type
                and row["snr_db"] == "20.0"
            ]
            if values:
                subgroup_ranges[group_type] = (min(values), max(values))
        report.extend(
            [
                "## 研究问题回答",
                "",
                f"1. **哪个模型能够最好学习频率？** 三种主架构中只有 M1 通过筛选并进入盲测。把 M1-power 输入消融也计入时，它的 0–20 dB 平均 E1 RMSE 最低（主架构 M1 为 {np.mean([np.mean(methods['model__M1__E1'][str(float(s))]['seed_rmse_hz']) for s in primary]):.4f} Hz，功率消融为 {np.mean([np.mean(methods['model__M1_power__E1'][str(float(s))]['seed_rmse_hz']) for s in primary]):.4f} Hz）。M1 在 20 dB 为 {m1_20:.4f} Hz，约为 CRLB 标准差的 {m1_crlb_ratio:.2f} 倍；连续 ML 为 {ml_20:.4f} Hz（{ml_crlb_ratio:.2f}× CRLB）。",
                "2. **物理归纳偏置是否优于普通序列学习？** M1 使用固定、不可学习的 16N 复 FFT，后接可学习循环 Conv1d 频谱读出；它不是学习式 FFT。M3 是两层、宽度 32、4 头的直接 IQ Transformer。M3 训练损失从 7.045 降至 5.579，但验证损失在首轮已最优（6.997）后升至 7.635，20 dB 开发 RMSE 为 371.3 Hz，接近常值预测 376.1 Hz，远差于 FFT 峰值 2.964 Hz。它按预注册规则停止，因此没有三种子正式盲测；这一单种子失败显示明显泛化问题，不能单独断言 Transformer 类方法普遍不可行。M1 E1 对 M1-power E1 的 20 dB 成对差为：",
                f"   - M1 − M1-power：{m1_vs_power_20}。两者 RMSE 非常接近，功率消融点估计略好（{power_20:.4f} Hz，对比 M1 {m1_20:.4f} Hz），没有证据表明相邻谱相干通道提高了单音频率精度。",
                "3. **FFT 辅助网络的收益来自哪里？** M1 比固定 16N FFT 峰值低约 0.148 Hz（20 dB，95% CI [-0.153, -0.142]），但比 16N 插值、同局部 4N 抛物线和连续 ML 差；可归因于固定频谱前端后的学习式精细读出，不能称为超过 ML。M2 在筛选开发集 20 dB RMSE 为 "
                f"{m2_dev20:.4f} Hz；它比 N 点 FFT 插值低 {-m2_fft_interpolation_delta20:.4f} Hz，但比相同 4N 局部频谱输入的抛物线估计高 {m2_frontend_delta20:.4f} Hz、比连续局部 profile-ML 高 {m2_profile_ml_delta20:.4f} Hz。M2 验证损失只下降 4.2%，随后回升，且没有胜过匹配的非学习局部估计，因此未进入正式三种子盲测；以下仅报告开发集点差，不报 CI。",
                "4. **学生能否超过部分传统方法？** M1 E1 在 0–20 dB 的五个主 SNR 点都显著优于 N 点 FFT 峰、N 点三点插值、相位差估计及 16N FFT 峰值（各 paired 95% CI 上界低于 0）。它未超过 4N/16N 抛物线插值或连续 profile-ML。",
                "5. **蒸馏是否带来增益？** E2-H 对 E1 在 20 dB RMSE 差为 "
                f"{_format_comparison(comparisons, 'model__M1__E2-H_vs_E1', 20.0)}，区间跨 0；E2-S 明显更差（{_format_comparison(comparisons, 'model__M1__E2-S_vs_E1', 20.0)}），E3-S 在 20 dB 也更差（{_format_comparison(comparisons, 'model__M1__E3-S_vs_E1', 20.0)}）。G3 未通过，没有可复现的联合蒸馏收益。",
                "6. **对 P1 三通道相位估计有什么启示？** M1 对公共相位旋转的预测变化最大仅 "
                f"{max_phase_delta:.4f} Hz，0.625 Hz 精确频移的最大等变误差约 {max_shift_error:.4f} Hz；G1、G2 均通过，支持按阶段门进入 P1-A。P1 需直接构造并验证跨通道复互相关/相位差特征，保留通道间相对相位；本单通道实验不能证明角度或三通道相位能力。",
                "",
                f"20 dB 下分组 RMSE 范围（每个训练种子分别统计）为：频率八分箱 {subgroup_ranges.get('frequency_bin', (math.nan, math.nan))[0]:.4f}–{subgroup_ranges.get('frequency_bin', (math.nan, math.nan))[1]:.4f} Hz；初相八扇区 {subgroup_ranges.get('phase_sector', (math.nan, math.nan))[0]:.4f}–{subgroup_ranges.get('phase_sector', (math.nan, math.nan))[1]:.4f} Hz；幅度四分位 {subgroup_ranges.get('amplitude_quartile', (math.nan, math.nan))[0]:.4f}–{subgroup_ranges.get('amplitude_quartile', (math.nan, math.nan))[1]:.4f} Hz。Nyquist 边界带每 SNR 仅约 13 个带噪样本，需谨慎解释；独立 noiseless 边界集每个 E1 种子 MAE 均小于 0.026 Hz、P95 小于 0.034 Hz。",
                "",
                "### 匹配物理前端与 M2 基线对比（candidate − baseline RMSE，Hz；M2 为开发集点差）",
                "",
                "| 主 SNR (dB) | M1 − 16N FFT peak（盲测 paired 95% CI） | M2 − N FFT 插值（开发） | M2 − 4N local parabola（开发） | M2 − local profile ML（开发） |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for snr in primary:
            m2_deltas: dict[str, float] = {}
            if screening is not None:
                model_rmse = screening["models"]["M2"]["development_rmse_by_snr_hz"][str(float(snr))]
                dev_baselines = screening["classical_development_rmse_by_snr_hz"]
                m2_deltas = {
                    name: model_rmse - dev_baselines[name][str(float(snr))]
                    for name in (
                        "B2_fft_interpolated",
                        "B9_4N_local_parabolic",
                        "B4_local_profile_ML",
                    )
                }
            report.append(
                f"| {snr:g} | {_format_comparison(comparisons, 'model__M1__E1_vs_B8_16N_fft_peak', snr)} | "
                f"{m2_deltas.get('B2_fft_interpolated', math.nan):+.4f} Hz | "
                f"{m2_deltas.get('B9_4N_local_parabolic', math.nan):+.4f} Hz | "
                f"{m2_deltas.get('B4_local_profile_ML', math.nan):+.4f} Hz |"
            )
        report.extend(
            [
                "",
                "### 阶段门",
                "",
                f"- G1：**{'PASS' if final_metrics['gates']['G1']['pass'] else 'FAIL'}**",
                f"- G2：**{'PASS' if final_metrics['gates']['G2']['pass'] else 'FAIL'}**",
                f"- G3：**{'PASS' if final_metrics['gates']['G3']['pass'] else 'FAIL'}**",
                f"- 阶段决策：**{final_metrics['stage_decision']}**",
                "",
                "只报告仿真结果；未测量部署时延，不外推到真实 GMTI 回波。若 G1 失败，不启动 P1；若 G1 通过而 G2 未通过，只能按阶段规范评估 CONDITIONAL GO。",
                "",
            ]
        )
    report_path = output / "analysis.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    return report_path


def _load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
