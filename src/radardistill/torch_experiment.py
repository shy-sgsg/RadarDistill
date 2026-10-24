"""GPU-only runner for the bounded P0-E006 PyTorch study."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .baselines import (
    estimate_fft_interpolated,
    estimate_fft_peak,
    estimate_mle_global_grid,
    estimate_mle_local,
    estimate_multilag_correlation,
    estimate_phase_difference,
    frequency_crlb_hz,
    local_teacher_profile_likelihood,
    wrapped_error,
)
from .cnn import TinyCircularCNN, circle_to_frequency, frequency_to_circle
from .experiment import _student_input, _test_subsets
from .signal import SignalBatch, generate_batch, generate_single_tone
from .torch_models import (
    CorrelationMLP,
    FrequencyResponseMLP,
    FrozenE005CNN,
    correlation_features,
    frequency_response_features,
    output_to_frequency,
)


@dataclass(frozen=True)
class Strategy:
    name: str
    train_target: str
    teacher_mode: str | None
    validation_mode: str
    combined: bool = False


def strategies_for_student(student: str) -> list[Strategy]:
    base = [
        Strategy("E1_truth", "truth", None, "truth"),
        Strategy("E2A_T1_hard", "teacher_hard", "T1", "teacher_hard"),
        Strategy("E2A_T2_local_soft", "teacher_soft", "T2", "teacher_soft"),
        Strategy("E3_T2_truth_plus_soft", "truth", "T2", "truth", combined=True),
    ]
    if student == "S1":
        return base + [
            Strategy("E2A_T3_full_response", "teacher_hard", "T3", "teacher_full"),
            Strategy("E2B_T2_truth_validation", "teacher_soft", "T2", "truth"),
            Strategy("E3_T3_truth_plus_response", "truth", "T3", "truth", combined=True),
        ]
    if student == "S2":
        return base
    raise ValueError(f"unknown student architecture: {student}")


def _make_student(name: str, config: dict[str, Any]) -> torch.nn.Module:
    model = config["model"]
    if name == "S1":
        return CorrelationMLP(
            hidden_units=int(model["correlation_hidden_units"]),
            response_bins=int(model["local_teacher_candidates"]),
        )
    if name == "S2":
        n_fft = int(config["n_samples"]) * int(model["spectrum_fft_factor"])
        return FrequencyResponseMLP(
            n_fft=n_fft,
            hidden_units=int(model["spectrum_hidden_units"]),
            response_bins=int(model["local_teacher_candidates"]),
        )
    raise ValueError(f"unknown student architecture: {name}")


def _features(iq: torch.Tensor, student: str, config: dict[str, Any]) -> torch.Tensor:
    if student == "S1":
        return correlation_features(iq, tuple(int(value) for value in config["model"]["lag_features"]))
    if student == "S2":
        n_fft = int(config["n_samples"]) * int(config["model"]["spectrum_fft_factor"])
        return frequency_response_features(iq, n_fft=n_fft)
    raise ValueError(f"unknown student architecture: {student}")


def _circle(frequency: np.ndarray, prf_hz: float, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(frequency_to_circle(frequency, prf_hz), dtype=torch.float32, device=device)


def _teacher_data(batch: SignalBatch, config: dict[str, Any]) -> dict[str, np.ndarray]:
    prf_hz = float(config["prf_hz"])
    model_config = config["model"]
    noise_power = batch.amplitude**2 * np.power(10.0, -batch.snr_db / 10.0)
    teacher_hz = np.asarray(estimate_mle_local(batch.iq, prf_hz), dtype=np.float64)
    candidates, response = local_teacher_profile_likelihood(
        batch.iq,
        teacher_hz,
        noise_power,
        prf_hz,
        candidate_count=int(model_config["local_teacher_candidates"]),
        half_width_bins=float(model_config["local_teacher_half_width_native_bins"]),
        temperature=float(model_config["local_teacher_temperature"]),
    )
    angles = 2.0 * np.pi * candidates / prf_hz
    soft_target = np.stack(
        (
            np.sum(response * np.cos(angles), axis=-1),
            np.sum(response * np.sin(angles), axis=-1),
        ),
        axis=-1,
    )
    return {
        "teacher_frequency_hz": teacher_hz,
        "teacher_candidates_hz": candidates,
        "teacher_local_response": response,
        "teacher_soft_target": soft_target,
        "noise_power": noise_power,
    }


def _save_batch(path: Path, batch: SignalBatch, role: str, teacher: dict[str, np.ndarray]) -> None:
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


def _combine(batches: list[SignalBatch], seed: int) -> SignalBatch:
    return SignalBatch(
        iq=np.concatenate([batch.iq for batch in batches]),
        frequency_hz=np.concatenate([batch.frequency_hz for batch in batches]),
        phase_rad=np.concatenate([batch.phase_rad for batch in batches]),
        amplitude=np.concatenate([batch.amplitude for batch in batches]),
        snr_db=np.concatenate([batch.snr_db for batch in batches]),
        seed=int(seed),
    )


def _training_objective(
    output: torch.Tensor,
    target: dict[str, torch.Tensor],
    strategy: Strategy,
    config: dict[str, Any],
    *,
    validation: bool = False,
) -> torch.Tensor:
    prediction = output[:, :2]
    if validation and strategy.validation_mode == "truth":
        return F.mse_loss(prediction, target["truth_circle"])
    if validation and strategy.validation_mode == "teacher_hard":
        return F.mse_loss(prediction, target["teacher_circle"])
    if validation and strategy.validation_mode == "teacher_soft":
        return F.mse_loss(prediction, target["soft_circle"])
    if validation and strategy.validation_mode == "teacher_full":
        point_loss = F.mse_loss(prediction, target["teacher_circle"])
        response_loss = F.kl_div(
            F.log_softmax(output[:, 2:], dim=-1), target["teacher_response"], reduction="batchmean"
        )
        return point_loss + float(config["training"]["response_weight"]) * response_loss

    if strategy.train_target == "truth":
        supervised = F.mse_loss(prediction, target["truth_circle"])
    elif strategy.train_target == "teacher_hard":
        supervised = F.mse_loss(prediction, target["teacher_circle"])
    elif strategy.train_target == "teacher_soft":
        supervised = F.mse_loss(prediction, target["soft_circle"])
    else:
        raise ValueError(f"unknown training target: {strategy.train_target}")

    if not strategy.combined:
        if strategy.teacher_mode == "T3":
            response_loss = F.kl_div(
                F.log_softmax(output[:, 2:], dim=-1),
                target["teacher_response"],
                reduction="batchmean",
            )
            return supervised + float(config["training"]["response_weight"]) * response_loss
        return supervised

    teacher_weight = float(config["training"]["teacher_weight"])
    if strategy.teacher_mode == "T2":
        return supervised + teacher_weight * F.mse_loss(prediction, target["soft_circle"])
    if strategy.teacher_mode == "T3":
        point_loss = F.mse_loss(prediction, target["teacher_circle"])
        response_loss = F.kl_div(
            F.log_softmax(output[:, 2:], dim=-1), target["teacher_response"], reduction="batchmean"
        )
        return supervised + teacher_weight * (
            point_loss + float(config["training"]["response_weight"]) * response_loss
        )
    raise ValueError(f"unknown combined teacher mode: {strategy.teacher_mode}")


def _tensor_targets(
    batch: SignalBatch,
    teacher: dict[str, np.ndarray],
    prf_hz: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "truth_circle": _circle(batch.frequency_hz, prf_hz, device),
        "teacher_circle": _circle(teacher["teacher_frequency_hz"], prf_hz, device),
        "soft_circle": torch.as_tensor(teacher["teacher_soft_target"], dtype=torch.float32, device=device),
        "teacher_response": torch.as_tensor(teacher["teacher_local_response"], dtype=torch.float32, device=device),
    }


def _train_one(
    student_name: str,
    strategy: Strategy,
    train_batch: SignalBatch,
    train_teacher: dict[str, np.ndarray],
    validation_batch: SignalBatch,
    validation_teacher: dict[str, np.ndarray],
    train_seed: int,
    replicate_index: int,
    config: dict[str, Any],
    models_dir: Path,
    history_rows: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    train_config = config["training"]
    torch_seed = int(train_config["torch_seed_base"]) + replicate_index * 1000 + (0 if student_name == "S1" else 100)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    model = _make_student(student_name, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_config["learning_rate"]))
    train_x = _features(torch.as_tensor(train_batch.iq, device=device), student_name, config)
    val_x = _features(torch.as_tensor(validation_batch.iq, device=device), student_name, config)
    train_y = _tensor_targets(train_batch, train_teacher, float(config["prf_hz"]), device)
    val_y = _tensor_targets(validation_batch, validation_teacher, float(config["prf_hz"]), device)
    generator = torch.Generator(device=device)
    generator.manual_seed(torch_seed + 1)
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    final_train_loss = float("nan")

    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator, device=device)
        accumulated = torch.zeros((), dtype=torch.float64, device=device)
        for start in range(0, len(order), int(train_config["batch_size"])):
            indices = order[start : start + int(train_config["batch_size"])]
            output = model(train_x[indices])
            loss = _training_objective(output, {key: value[indices] for key, value in train_y.items()}, strategy, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            accumulated += loss.detach().to(torch.float64) * len(indices)
        final_train_loss = float(accumulated.item() / len(train_x))

        model.eval()
        with torch.no_grad():
            validation_output = model(val_x)
            validation_loss = float(
                _training_objective(validation_output, val_y, strategy, config, validation=True).detach()
            )
        history_rows.append(
            {
                "student": student_name,
                "strategy": strategy.name,
                "train_seed": int(train_seed),
                "torch_seed": torch_seed,
                "epoch": epoch,
                "training_loss": final_train_loss,
                "validation_selection_loss": validation_loss,
                "validation_selection_target": strategy.validation_mode,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= int(train_config["patience"]):
                break

    if best_state is None:
        raise RuntimeError(f"training produced no checkpoint for {student_name}/{strategy.name}")
    model.load_state_dict(best_state)
    checkpoint = models_dir / f"{student_name}_seed{train_seed}_{strategy.name}.pt"
    torch.save(
        {
            "student": student_name,
            "strategy": asdict(strategy),
            "train_seed": int(train_seed),
            "torch_seed": torch_seed,
            "selected_epoch": best_epoch,
            "best_validation_selection_loss": best_loss,
            "state_dict": best_state,
            "model_config": config["model"],
        },
        checkpoint,
    )
    model.eval()
    return {
        "student": student_name,
        "strategy": strategy.name,
        "train_seed": int(train_seed),
        "torch_seed": torch_seed,
        "epochs_completed": epoch,
        "selected_epoch": best_epoch,
        "best_validation_selection_loss": best_loss,
        "final_training_loss": final_train_loss,
        "validation_selection_target": strategy.validation_mode,
        "checkpoint": str(checkpoint),
        "model": model,
    }


def _generate_train_and_validation(config: dict[str, Any], datasets_dir: Path):
    data_config = config["data"]
    train_records: list[dict[str, Any]] = []
    train_batches: dict[int, SignalBatch] = {}
    validation_batches: dict[int, SignalBatch] = {}
    train_teachers: dict[int, dict[str, np.ndarray]] = {}
    validation_teachers: dict[int, dict[str, np.ndarray]] = {}
    for index, train_seed in enumerate(data_config["train_seeds"]):
        train = generate_batch(
            count=int(data_config["train_count"]),
            seed=int(train_seed),
            prf_hz=float(config["prf_hz"]),
            n_samples=int(config["n_samples"]),
            snr_levels_db=tuple(float(value) for value in data_config["train_snr_levels_db"]),
            amplitude_range=tuple(float(value) for value in data_config["amplitude_range"]),
        )
        validation_seed = int(data_config["validation_seeds"][index])
        validation = generate_batch(
            count=int(data_config["validation_count"]),
            seed=validation_seed,
            prf_hz=float(config["prf_hz"]),
            n_samples=int(config["n_samples"]),
            snr_levels_db=tuple(float(value) for value in data_config["train_snr_levels_db"]),
            amplitude_range=tuple(float(value) for value in data_config["amplitude_range"]),
        )
        train_teacher = _teacher_data(train, config)
        validation_teacher = _teacher_data(validation, config)
        _save_batch(datasets_dir / f"train_seed_{train_seed}.npz", train, "train", train_teacher)
        _save_batch(
            datasets_dir / f"validation_seed_{validation_seed}.npz",
            validation,
            "validation",
            validation_teacher,
        )
        train_batches[int(train_seed)] = train
        validation_batches[int(train_seed)] = validation
        train_teachers[int(train_seed)] = train_teacher
        validation_teachers[int(train_seed)] = validation_teacher
        train_records.append(
            {
                "train_seed": int(train_seed),
                "validation_seed": validation_seed,
                "train_count": len(train.iq),
                "validation_count": len(validation.iq),
            }
        )
    return train_batches, validation_batches, train_teachers, validation_teachers, train_records


def _load_development_batch(repo_root: Path, config: dict[str, Any]) -> SignalBatch:
    path = repo_root / "experiments" / "P0-E005" / "datasets" / "test.npz"
    with np.load(path) as values:
        return SignalBatch(
            iq=values["iq"],
            frequency_hz=values["frequency_hz"],
            phase_rad=values["phase_rad"],
            amplitude=values["amplitude"],
            snr_db=values["snr_db"],
            seed=int(values["dataset_seed"]),
        )


def _generate_final_batch(config: dict[str, Any]) -> SignalBatch:
    data_config = config["data"]
    batches = [
        generate_batch(
            count=int(data_config["test_per_snr"]),
            seed=int(data_config["final_test_seed"]) + 1000 * index,
            prf_hz=float(config["prf_hz"]),
            n_samples=int(config["n_samples"]),
            snr_db=float(snr),
            amplitude_range=tuple(float(value) for value in data_config["amplitude_range"]),
        )
        for index, snr in enumerate(data_config["test_snr_levels_db"])
    ]
    return _combine(batches, int(data_config["final_test_seed"]))


def _predict_torch_models(
    model_records: list[dict[str, Any]], batch: SignalBatch, config: dict[str, Any], device: torch.device
) -> dict[str, list[np.ndarray]]:
    predictions: dict[str, list[np.ndarray]] = {}
    iq = torch.as_tensor(batch.iq, device=device)
    with torch.no_grad():
        for record in model_records:
            student = record["student"]
            model = record["model"]
            model.eval()
            features = _features(iq, student, config)
            output = model(features)
            predicted = output_to_frequency(output, float(config["prf_hz"])).cpu().numpy()
            key = f"{student}/{record['strategy']}"
            predictions.setdefault(key, []).append(np.asarray(predicted, dtype=np.float64))
    return predictions


def _predict_s0(batch: SignalBatch, repo_root: Path, prf_hz: float) -> dict[str, list[np.ndarray]]:
    model_dir = repo_root / "experiments" / "P0-E005" / "models"
    values = _student_input(batch.iq, "unit_phase_increment")
    predictions: dict[str, list[np.ndarray]] = {f"S0/{name}": [] for name in (
        "E1_supervised", "E2_teacher_KD", "E3_supervised_plus_KD"
    )}
    for seed in (101, 202, 303):
        for strategy in predictions:
            strategy_name = strategy.split("/", 1)[1]
            model = TinyCircularCNN(n_samples=values.shape[-1], seed=0, channels=6, kernel_size=9)
            model.load(str(model_dir / f"seed_{seed}_{strategy_name}.npz"))
            output = model.predict(values)
            predictions[strategy].append(np.asarray(circle_to_frequency(output, prf_hz), dtype=np.float64))
    return predictions


def _classical_predictions(
    batch: SignalBatch,
    config: dict[str, Any],
    local_ml: np.ndarray | None = None,
    progress_label: str = "",
) -> dict[str, np.ndarray]:
    prf_hz = float(config["prf_hz"])
    grid_size = int(config["evaluation"]["global_ml_grid_size"])
    correlation_grid = int(config["evaluation"]["correlation_grid_size"])
    n_fft = int(config["n_samples"]) * int(config["model"]["spectrum_fft_factor"])
    methods = [
        ("B0_constant_zero", lambda: np.zeros(len(batch.iq), dtype=np.float64)),
        ("B1_fft_peak", lambda: estimate_fft_peak(batch.iq, prf_hz)),
        ("B2_fft_interpolated", lambda: estimate_fft_interpolated(batch.iq, prf_hz)),
        ("B3_phase_difference", lambda: estimate_phase_difference(batch.iq, prf_hz)),
        ("B4_local_profile_ML", lambda: estimate_mle_local(batch.iq, prf_hz) if local_ml is None else local_ml),
        ("B5_global_grid_profile_ML", lambda: estimate_mle_global_grid(batch.iq, prf_hz, grid_size=grid_size)),
        ("B6_4N_fft_peak", lambda: _fft_peak_n(batch.iq, prf_hz, n_fft)),
        (
            "B7_multilag_correlation",
            lambda: estimate_multilag_correlation(
                batch.iq,
                prf_hz,
                lags=tuple(int(value) for value in config["model"]["lag_features"]),
                grid_size=correlation_grid,
            ),
        ),
    ]
    output = {}
    for name, estimator in methods:
        started = time.perf_counter()
        output[name] = np.asarray(estimator(), dtype=np.float64)
        print(
            f"{progress_label} {name}: {time.perf_counter() - started:.2f}s",
            flush=True,
        )
    return output


def _fft_peak_n(iq: np.ndarray, prf_hz: float, n_fft: int) -> np.ndarray | float:
    x = np.asarray(iq)
    scalar = x.ndim == 1
    if scalar:
        x = x[None, :]
    spectrum = np.fft.fftshift(np.fft.fft(x, n=n_fft, axis=-1), axes=-1)
    peak = np.argmax(np.abs(spectrum), axis=-1)
    frequencies = (np.arange(n_fft) - n_fft // 2) * prf_hz / n_fft
    result = frequencies[peak]
    return float(result[0]) if scalar else result


def _basic_metrics(errors: np.ndarray, outlier_threshold_hz: float) -> dict[str, float | int | str]:
    """Fast point estimates for per-seed and grouped rows; CIs are paired separately."""
    values = np.asarray(errors, dtype=np.float64)
    if len(values) == 0:
        return {
            "n": 0,
            "mae_hz": float("nan"),
            "rmse_hz": float("nan"),
            "bias_hz": float("nan"),
            "outlier_rate": float("nan"),
            "rmse_ci95_low_hz": "",
            "rmse_ci95_high_hz": "",
        }
    return {
        "n": int(len(values)),
        "mae_hz": float(np.mean(np.abs(values))),
        "rmse_hz": float(np.sqrt(np.mean(values**2))),
        "bias_hz": float(np.mean(values)),
        "outlier_rate": float(np.mean(np.abs(values) > outlier_threshold_hz)),
        "rmse_ci95_low_hz": "",
        "rmse_ci95_high_hz": "",
    }


def _prediction_replicates(prediction_values, expected_shape: tuple[int, ...], method: str) -> list[np.ndarray]:
    """Normalize one batch prediction or a seed stack to a list of full vectors."""
    if isinstance(prediction_values, np.ndarray):
        if prediction_values.ndim == 1:
            replicates = [prediction_values]
        elif prediction_values.ndim == 2:
            replicates = [prediction_values[index] for index in range(len(prediction_values))]
        else:
            raise ValueError(f"{method} predictions must be one vector or a stack of vectors")
    else:
        replicates = list(prediction_values)
    if not replicates or any(np.asarray(value).shape != expected_shape for value in replicates):
        raise ValueError(f"{method} must provide one frequency prediction per test example")
    return [np.asarray(value, dtype=np.float64) for value in replicates]


def _evaluate_predictions(
    role: str,
    batch: SignalBatch,
    predictions: dict[str, list[np.ndarray] | np.ndarray],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    threshold = prf_hz / (2.0 * n_samples)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    error_payload: dict[str, np.ndarray] = {}
    subsets = _test_subsets(batch, prf_hz)
    for method, prediction_values in predictions.items():
        replicate_predictions = _prediction_replicates(prediction_values, batch.frequency_hz.shape, method)
        summary[method] = {}
        replicate_errors = np.stack(
            [wrapped_error(prediction, batch.frequency_hz, prf_hz) for prediction in replicate_predictions]
        )
        error_payload[f"{role}__{method}"] = replicate_errors
        for snr in config["data"]["test_snr_levels_db"]:
            mask = np.isclose(batch.snr_db, float(snr))
            per_seed = []
            for seed_index, errors in enumerate(replicate_errors):
                values = _basic_metrics(errors[mask], threshold)
                crlb = float(frequency_crlb_hz(prf_hz, n_samples, float(snr)))
                seed_label = seed_index if len(replicate_errors) > 1 else "reference"
                rows.append(
                    {
                        "data_role": role,
                        "method": method,
                        "seed_index": seed_label,
                        "snr_db": float(snr),
                        "subset": "all",
                        **values,
                        "crlb_sd_hz": crlb,
                    }
                )
                per_seed.append(float(values["rmse_hz"]))
            summary[method][str(float(snr))] = {
                "rmse_mean_seed_hz": float(np.mean(per_seed)),
                "rmse_sd_seed_hz": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                "crlb_sd_hz": float(frequency_crlb_hz(prf_hz, n_samples, float(snr))),
                "n_per_seed": int(np.sum(mask)),
            }
        for subset, subset_mask in subsets.items():
            if subset == "all":
                continue
            for seed_index, errors in enumerate(replicate_errors):
                selected = errors[subset_mask]
                values = _basic_metrics(selected, threshold)
                rows.append(
                    {
                        "data_role": role,
                        "method": method,
                        "seed_index": seed_index if len(replicate_errors) > 1 else "reference",
                        "snr_db": "all",
                        "subset": subset,
                        **values,
                        "crlb_sd_hz": "",
                    }
                )
    return rows, summary, error_payload


def _paired_rmse_delta_ci(
    candidate_errors: np.ndarray,
    reference_errors: np.ndarray,
    samples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    seed_count, sample_count = candidate_errors.shape
    deltas = np.empty(samples, dtype=np.float64)
    candidate_rmse = np.empty(samples, dtype=np.float64)
    reference_rmse = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        seeds = rng.integers(0, seed_count, size=seed_count)
        selected = rng.integers(0, sample_count, size=sample_count)
        candidate = candidate_errors[np.ix_(seeds, selected)]
        if reference_errors.ndim == 1:
            reference = np.broadcast_to(reference_errors[selected], candidate.shape)
        elif reference_errors.shape[0] == 1:
            reference = np.broadcast_to(reference_errors[0, selected], candidate.shape)
        elif reference_errors.shape[0] == seed_count:
            reference = reference_errors[np.ix_(seeds, selected)]
        else:
            raise ValueError("paired reference errors must be sample-only or match the candidate seed count")
        candidate_rmse[index] = np.sqrt(np.mean(candidate**2))
        reference_rmse[index] = np.sqrt(np.mean(reference**2))
        deltas[index] = candidate_rmse[index] - reference_rmse[index]
    estimate = np.sqrt(np.mean(candidate_errors**2)) - np.sqrt(np.mean(reference_errors**2))
    low, high = np.quantile(deltas, [0.025, 0.975])
    candidate_low, candidate_high = np.quantile(candidate_rmse, [0.025, 0.975])
    reference_low, reference_high = np.quantile(reference_rmse, [0.025, 0.975])
    return {
        "delta_rmse_hz": float(estimate),
        "ci95_low_hz": float(low),
        "ci95_high_hz": float(high),
        "candidate_rmse_ci95_low_hz": float(candidate_low),
        "candidate_rmse_ci95_high_hz": float(candidate_high),
        "reference_rmse_ci95_low_hz": float(reference_low),
        "reference_rmse_ci95_high_hz": float(reference_high),
    }


def _noiseless_batch(config: dict[str, Any]) -> SignalBatch:
    data_config = config["data"]
    rng = np.random.default_rng(int(data_config["noiseless_seed"]))
    count = int(data_config["noiseless_count"])
    frequencies = rng.uniform(-float(config["prf_hz"]) / 2, float(config["prf_hz"]) / 2, count)
    phase = rng.uniform(-np.pi, np.pi, count)
    amplitudes = rng.uniform(*[float(x) for x in data_config["amplitude_range"]], size=count)
    signals = np.stack(
        [
            generate_single_tone(
                float(frequencies[index]),
                float(phase[index]),
                float(amplitudes[index]),
                np.inf,
                float(config["prf_hz"]),
                int(config["n_samples"]),
                int(data_config["noiseless_seed"]) + index,
            )
            for index in range(count)
        ]
    )
    return SignalBatch(
        iq=signals,
        frequency_hz=frequencies,
        phase_rad=phase,
        amplitude=amplitudes,
        snr_db=np.full(count, np.inf),
        seed=int(data_config["noiseless_seed"]),
    )


def _noiseless_predictions(
    model_records: list[dict[str, Any]], batch: SignalBatch, config: dict[str, Any], device: torch.device
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    iq = torch.as_tensor(batch.iq, device=device)
    with torch.no_grad():
        for record in model_records:
            features = _features(iq, record["student"], config)
            pred = output_to_frequency(record["model"](features), float(config["prf_hz"])).cpu().numpy()
            key = f"{record['student']}/{record['strategy']}/seed_{record['train_seed']}"
            output[key] = np.asarray(pred, dtype=np.float64)
    return output


def _student_physics_diagnostics(
    model_records: list[dict[str, Any]], batch: SignalBatch, config: dict[str, Any], device: torch.device
) -> list[dict[str, Any]]:
    prf_hz = float(config["prf_hz"])
    subset = batch.iq[: min(128, len(batch.iq))]
    x = torch.as_tensor(subset, device=device)
    phase = torch.exp(torch.tensor(0.77j, dtype=torch.complex64, device=device))
    scale = torch.tensor(1.7 - 0.3j, dtype=torch.complex64, device=device)
    delta_hz = prf_hz / 17.0
    n = torch.arange(x.shape[-1], dtype=torch.float32, device=device)
    shifted_x = x * torch.exp(2j * torch.pi * delta_hz * n / prf_hz)
    conjugate_x = x.conj()
    index = np.arange(x.shape[-1], dtype=np.float64)
    clean = batch.amplitude[: len(x), None] * np.exp(
        1j * (
            2.0 * np.pi * batch.frequency_hz[: len(x), None] * index[None, :] / prf_hz
            + batch.phase_rad[: len(x), None]
        )
    )
    noise = batch.iq[: len(x)] - clean
    signal_scale = 1.7
    signal_only_scaled = torch.as_tensor(clean * signal_scale + noise, device=device)
    results = []
    with torch.no_grad():
        for record in model_records:
            student, model = record["student"], record["model"]
            base = output_to_frequency(model(_features(x, student, config)), prf_hz)
            rotated = output_to_frequency(model(_features(x * phase, student, config)), prf_hz)
            scaled = output_to_frequency(model(_features(x * scale, student, config)), prf_hz)
            shifted = output_to_frequency(model(_features(shifted_x, student, config)), prf_hz)
            conjugated = output_to_frequency(model(_features(conjugate_x, student, config)), prf_hz)
            signal_scaled = output_to_frequency(
                model(_features(signal_only_scaled, student, config)), prf_hz
            )
            base_np = base.cpu().numpy()
            for transform, observed, expected in (
                ("common_phase_rotation", rotated.cpu().numpy(), base_np),
                ("whole_observation_scale", scaled.cpu().numpy(), base_np),
                ("frequency_shift", shifted.cpu().numpy(), (base_np + delta_hz + prf_hz / 2) % prf_hz - prf_hz / 2),
                ("complex_conjugation", conjugated.cpu().numpy(), (-base_np + prf_hz / 2) % prf_hz - prf_hz / 2),
            ):
                errors = np.asarray(wrapped_error(observed, expected, prf_hz))
                results.append(
                    {
                        "student": student,
                        "strategy": record["strategy"],
                        "train_seed": record["train_seed"],
                        "transform": transform,
                        "mae_deviation_hz": float(np.mean(np.abs(errors))),
                        "rmse_deviation_hz": float(np.sqrt(np.mean(errors**2))),
                    }
                )
            scaled_error = np.asarray(
                wrapped_error(signal_scaled.cpu().numpy(), batch.frequency_hz[: len(x)], prf_hz)
            )
            base_error = np.asarray(wrapped_error(base_np, batch.frequency_hz[: len(x)], prf_hz))
            results.append(
                {
                    "student": student,
                    "strategy": record["strategy"],
                    "train_seed": record["train_seed"],
                    "transform": "signal_only_amplitude_scale_changes_snr",
                    "signal_amplitude_scale": signal_scale,
                    "snr_change_db": float(20.0 * np.log10(signal_scale)),
                    "base_frequency_rmse_hz": float(np.sqrt(np.mean(base_error**2))),
                    "scaled_signal_frequency_rmse_hz": float(np.sqrt(np.mean(scaled_error**2))),
                }
            )
    return results


def _s0_predict(record: dict[str, Any], iq: np.ndarray, prf_hz: float, device: torch.device) -> np.ndarray:
    increments = _student_input(iq, "unit_phase_increment")
    features = torch.stack(
        (
            torch.as_tensor(increments.real, dtype=torch.float32, device=device),
            torch.as_tensor(increments.imag, dtype=torch.float32, device=device),
        ),
        dim=1,
    )
    with torch.no_grad():
        prediction = output_to_frequency(record["model"](features), prf_hz)
    return prediction.cpu().numpy()


def _s0_physics_diagnostics(
    records: list[dict[str, Any]], batch: SignalBatch, prf_hz: float, device: torch.device
) -> list[dict[str, Any]]:
    count = min(128, len(batch.iq))
    x = batch.iq[:count]
    index = np.arange(x.shape[-1], dtype=np.float64)
    clean = batch.amplitude[:count, None] * np.exp(
        1j * (2.0 * np.pi * batch.frequency_hz[:count, None] * index[None, :] / prf_hz + batch.phase_rad[:count, None])
    )
    noise = x - clean
    signal_scale = 1.7
    transforms = {
        "common_phase_rotation": x * np.exp(0.77j),
        "whole_observation_scale": x * (1.7 - 0.3j),
        "frequency_shift": x * np.exp(2j * np.pi * (prf_hz / 17.0) * index[None, :] / prf_hz),
        "complex_conjugation": np.conj(x),
        "signal_only_amplitude_scale_changes_snr": clean * signal_scale + noise,
    }
    results = []
    for record in records:
        base = _s0_predict(record, x, prf_hz, device)
        for name, transformed_x in transforms.items():
            prediction = _s0_predict(record, transformed_x, prf_hz, device)
            if name == "frequency_shift":
                expected = (base + prf_hz / 17.0 + prf_hz / 2) % prf_hz - prf_hz / 2
            elif name == "complex_conjugation":
                expected = (-base + prf_hz / 2) % prf_hz - prf_hz / 2
            else:
                expected = base
            delta = np.asarray(wrapped_error(prediction, expected, prf_hz))
            row = {
                "student": "S0",
                "strategy": record["strategy"],
                "train_seed": record["train_seed"],
                "transform": name,
                "mae_deviation_hz": float(np.mean(np.abs(delta))),
                "rmse_deviation_hz": float(np.sqrt(np.mean(delta**2))),
            }
            if name == "signal_only_amplitude_scale_changes_snr":
                truth = batch.frequency_hz[:count]
                row["signal_amplitude_scale"] = signal_scale
                row["snr_change_db"] = float(20.0 * np.log10(signal_scale))
                row["base_frequency_rmse_hz"] = float(np.sqrt(np.mean(wrapped_error(base, truth, prf_hz) ** 2)))
                row["scaled_signal_frequency_rmse_hz"] = float(
                    np.sqrt(np.mean(wrapped_error(prediction, truth, prf_hz) ** 2))
                )
            results.append(row)
    return results


def _load_frozen_s0_models(repo_root: Path, device: torch.device) -> list[dict[str, Any]]:
    model_dir = repo_root / "experiments" / "P0-E005" / "models"
    records = []
    for seed in (101, 202, 303):
        for strategy in ("E1_supervised", "E2_teacher_KD", "E3_supervised_plus_KD"):
            model = FrozenE005CNN(channels=6, kernel_size=9).to(device)
            path = model_dir / f"seed_{seed}_{strategy}.npz"
            model.load_numpy_checkpoint(str(path), device=device)
            records.append({"student": "S0", "strategy": strategy, "train_seed": seed, "model": model})
    return records


def _predict_s0_models(
    records: list[dict[str, Any]], batch: SignalBatch, prf_hz: float, device: torch.device
) -> dict[str, list[np.ndarray]]:
    predictions: dict[str, list[np.ndarray]] = {}
    for record in records:
        result = _s0_predict(record, batch.iq, prf_hz, device)
        predictions.setdefault(f"S0/{record['strategy']}", []).append(np.asarray(result, dtype=np.float64))
    return predictions


def _load_s0_development_predictions(repo_root: Path) -> dict[str, list[np.ndarray]]:
    path = repo_root / "experiments" / "P0-E005" / "predictions.npz"
    predictions: dict[str, list[np.ndarray]] = {}
    with np.load(path) as values:
        for strategy in ("E1_supervised", "E2_teacher_KD", "E3_supervised_plus_KD"):
            predictions[f"S0/{strategy}"] = [
                np.asarray(values[f"{strategy}__seed_{seed}"], dtype=np.float64)
                for seed in (101, 202, 303)
            ]
    return predictions


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "data_role", "method", "seed_index", "snr_db", "subset", "n", "mae_hz", "rmse_hz",
        "bias_hz", "outlier_rate", "rmse_ci95_low_hz", "rmse_ci95_high_hz", "crlb_sd_hz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "student", "strategy", "train_seed", "torch_seed", "epoch", "training_loss",
        "validation_selection_loss", "validation_selection_target",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_manifest(paths: list[Path], root: Path) -> list[dict[str, Any]]:
    manifest = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        manifest.append(
            {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        )
    return manifest


def _plots(
    output: Path,
    metrics_by_role: dict[str, dict[str, Any]],
    error_payload: dict[str, np.ndarray],
    history_rows: list[dict[str, Any]],
    config: dict[str, Any],
    final_batch: SignalBatch,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    final_metrics = metrics_by_role["final_blind"]
    snrs = [float(value) for value in config["data"]["test_snr_levels_db"]]
    palette = {
        "B1_fft_peak": "#4c78a8",
        "B2_fft_interpolated": "#72b7b2",
        "B3_phase_difference": "#f58518",
        "B4_local_profile_ML": "#e45756",
        "B5_global_grid_profile_ML": "#b279a2",
        "B6_4N_fft_peak": "#54a24b",
        "B7_multilag_correlation": "#ff9da6",
        "S0/E1_supervised": "#9d755d",
        "S1/E1_truth": "#bab0ac",
        "S2/E1_truth": "#79706e",
    }

    def save(fig, name):
        path = figures / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return str(path.relative_to(output))

    methods = list(final_metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        values = final_metrics[method]
        rmse = [values[str(float(snr))]["rmse_mean_seed_hz"] for snr in snrs]
        ax.plot(snrs, rmse, marker="o", label=method, color=palette.get(method))
    ax.set(xlabel="SNR (dB)", ylabel="RMSE (Hz)", yscale="log", title="P0-E006 final blind RMSE")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    rmse_plot = save(fig, "rmse_snr.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for method in methods:
        values = final_metrics[method]
        ratio = [
            values[str(float(snr))]["rmse_mean_seed_hz"] / values[str(float(snr))]["crlb_sd_hz"]
            for snr in snrs
        ]
        ax.plot(snrs, ratio, marker="o", label=method, color=palette.get(method))
    ax.set(xlabel="SNR (dB)", ylabel="RMSE / CRLB SD", yscale="log", title="Distance from the CRLB")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    crlb_plot = save(fig, "rmse_crlb_ratio.png")

    history_methods = ["S1/E1_truth", "S2/E1_truth", "S1/E2A_T2_local_soft", "S2/E2A_T2_local_soft"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in history_methods:
        student, strategy = method.split("/", 1)
        selected = [row for row in history_rows if row["student"] == student and row["strategy"] == strategy]
        for seed in config["data"]["train_seeds"]:
            track = [row for row in selected if row["train_seed"] == int(seed)]
            if track:
                ax.plot([row["epoch"] for row in track], [row["training_loss"] for row in track], alpha=0.45)
    ax.set(xlabel="Epoch", ylabel="Training objective", yscale="log", title="Training loss (each line is one seed)")
    ax.grid(True, which="both", alpha=0.25)
    loss_plot = save(fig, "training_loss.png")

    snr_mask = np.isclose(final_batch.snr_db, 20.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("final_blind__B1_fft_peak", "final_blind__S0/E1_supervised", "final_blind__S1/E1_truth", "final_blind__S2/E1_truth"):
        if key in error_payload:
            values = error_payload[key][:, snr_mask].ravel()
            ax.hist(values, bins=60, density=True, alpha=0.45, label=key.split("__", 1)[1])
    ax.set(xlabel="Wrapped frequency error (Hz)", ylabel="Density", title="Error distributions at 20 dB")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    error_plot = save(fig, "error_distribution_20db.png")

    frequency_bins = np.linspace(-float(config["prf_hz"]) / 2, float(config["prf_hz"]) / 2, 6)
    fig, ax = plt.subplots(figsize=(10, 5))
    for key in ("final_blind__B1_fft_peak", "final_blind__S0/E1_supervised", "final_blind__S1/E1_truth", "final_blind__S2/E1_truth"):
        if key not in error_payload:
            continue
        selected_errors = error_payload[key][:, snr_mask]
        errs = selected_errors.reshape(-1)
        selected_truth = final_batch.frequency_hz[snr_mask]
        ids = np.tile(np.clip(np.digitize(selected_truth, frequency_bins[1:-1]), 0, 4), selected_errors.shape[0])
        rmses = [
            np.sqrt(np.mean(errs[ids == i] ** 2)) if np.any(ids == i) else np.nan
            for i in range(5)
        ]
        ax.plot(range(5), rmses, marker="o", label=key.split("__", 1)[1])
    ax.set(xlabel="Frequency group (low to high)", ylabel="RMSE (Hz)", title="Frequency-group error at 20 dB")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    frequency_plot = save(fig, "frequency_groups_20db.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    for student, strategies in (("S1", ["E1_truth", "E2A_T1_hard", "E2A_T2_local_soft", "E2A_T3_full_response", "E3_T2_truth_plus_soft", "E3_T3_truth_plus_response"]),
                                ("S2", ["E1_truth", "E2A_T1_hard", "E2A_T2_local_soft", "E3_T2_truth_plus_soft"])):
        labels, rmses = [], []
        for strategy in strategies:
            key = f"{student}/{strategy}"
            if key in final_metrics:
                labels.append(f"{student}:{strategy.replace('E', '')}")
                rmses.append(final_metrics[key]["20.0"]["rmse_mean_seed_hz"])
        if labels:
            ax.plot(labels, rmses, marker="o", label=student)
    ax.set(ylabel="RMSE (Hz)", title="Supervision and teacher ablation at 20 dB")
    ax.tick_params(axis="x", rotation=65, labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend()
    distillation_plot = save(fig, "distillation_ablation_20db.png")
    return [rmse_plot, crlb_plot, loss_plot, error_plot, frequency_plot, distillation_plot]


def run_p0_e006(config: dict[str, Any], output_dir: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """Run the frozen experiment. Refuses to train unless CUDA is available."""
    training_config = config["training"]
    if training_config.get("device") != "cuda":
        raise ValueError("P0-E006 preregistration requires training.device: cuda")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required; no model training was started")
    device = torch.device("cuda:0")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(int(training_config["torch_seed_base"]))
    torch.cuda.manual_seed_all(int(training_config["torch_seed_base"]))

    output = Path(output_dir)
    repo_root = Path(repo_root)
    output.mkdir(parents=True, exist_ok=True)
    datasets_dir, models_dir = output / "datasets", output / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    protocol_source = repo_root / "experiments" / "P0-E006" / "protocol.md"
    protocol_output = output / "protocol.md"
    if protocol_source.resolve() != protocol_output.resolve():
        shutil.copyfile(protocol_source, protocol_output)

    print("P0-E006: generating training and validation targets", flush=True)
    train_batches, validation_batches, train_teachers, validation_teachers, split_records = _generate_train_and_validation(
        config, datasets_dir
    )
    history_rows: list[dict[str, Any]] = []
    model_records: list[dict[str, Any]] = []
    training_records: list[dict[str, Any]] = []
    students = ("S1", "S2")
    for replicate_index, train_seed in enumerate(config["data"]["train_seeds"]):
        for student in students:
            for strategy in strategies_for_student(student):
                record = _train_one(
                    student,
                    strategy,
                    train_batches[int(train_seed)],
                    train_teachers[int(train_seed)],
                    validation_batches[int(train_seed)],
                    validation_teachers[int(train_seed)],
                    int(train_seed),
                    replicate_index,
                    config,
                    models_dir,
                    history_rows,
                    device,
                )
                model_records.append(record)
                training_records.append({key: value for key, value in record.items() if key != "model"})
                print(
                    f"trained {student}/{strategy.name} seed={train_seed}: "
                    f"epochs={record['epochs_completed']} selected={record['selected_epoch']} "
                    f"validation={record['best_validation_selection_loss']:.6g}",
                    flush=True,
                )
    _write_history(output / "training_history.csv", history_rows)

    print("P0-E006: evaluating development benchmark", flush=True)
    dev_batch = _load_development_batch(repo_root, config)
    s0_records = _load_frozen_s0_models(repo_root, device)
    dev_data_path = repo_root / "experiments" / "P0-E005" / "datasets" / "test.npz"
    with np.load(dev_data_path) as dev_data:
        dev_local_ml = np.asarray(dev_data["teacher_frequency_hz"], dtype=np.float64)
    dev_predictions = _classical_predictions(dev_batch, config, local_ml=dev_local_ml, progress_label="development")
    dev_predictions.update(_load_s0_development_predictions(repo_root))
    dev_predictions.update(_predict_torch_models(model_records, dev_batch, config, device))
    bootstrap_rng = np.random.default_rng(int(config["evaluation"]["bootstrap_seed"]))
    print("P0-E006: computing development metrics", flush=True)
    dev_rows, dev_metrics, dev_errors = _evaluate_predictions(
        "development_benchmark", dev_batch, dev_predictions, config
    )

    print("P0-E006: running noiseless correctness check", flush=True)
    noiseless = _noiseless_batch(config)
    noiseless_teacher = {
        "teacher_frequency_hz": noiseless.frequency_hz,
        "teacher_candidates_hz": np.empty((len(noiseless.iq), 0)),
        "teacher_local_response": np.empty((len(noiseless.iq), 0)),
        "teacher_soft_target": frequency_to_circle(noiseless.frequency_hz, float(config["prf_hz"])),
        "noise_power": np.zeros(len(noiseless.iq)),
    }
    _save_batch(datasets_dir / "noiseless_correctness.npz", noiseless, "noiseless_correctness", noiseless_teacher)
    noiseless_predictions = _noiseless_predictions(model_records, noiseless, config, device)
    noiseless_metrics = {}
    for name, pred in noiseless_predictions.items():
        error = np.asarray(wrapped_error(pred, noiseless.frequency_hz, float(config["prf_hz"])))
        noiseless_metrics[name] = {
            "n": len(error),
            "mae_hz": float(np.mean(np.abs(error))),
            "rmse_hz": float(np.sqrt(np.mean(error**2))),
            "p95_abs_error_hz": float(np.quantile(np.abs(error), 0.95)),
        }

    # The final blind test is generated only after training and checkpoint selection finish.
    print("P0-E006: generating and evaluating final blind test", flush=True)
    final_batch = _generate_final_batch(config)
    final_teacher = _teacher_data(final_batch, config)
    _save_batch(datasets_dir / "final_blind_test.npz", final_batch, "final_blind", final_teacher)
    final_predictions = _classical_predictions(
        final_batch,
        config,
        local_ml=final_teacher["teacher_frequency_hz"],
        progress_label="final",
    )
    final_predictions.update(_predict_s0_models(s0_records, final_batch, float(config["prf_hz"]), device))
    final_predictions.update(_predict_torch_models(model_records, final_batch, config, device))
    print("P0-E006: computing final blind metrics", flush=True)
    final_rows, final_metrics, final_errors = _evaluate_predictions(
        "final_blind", final_batch, final_predictions, config
    )
    errors = {**dev_errors, **final_errors}
    physics_diagnostics = _student_physics_diagnostics(model_records, dev_batch, config, device)
    physics_diagnostics.extend(
        _s0_physics_diagnostics(s0_records, dev_batch, float(config["prf_hz"]), device)
    )
    combined_rows = dev_rows + final_rows
    _write_rows(output / "result.csv", combined_rows)
    prediction_payload = {}
    for role, source, evaluated_batch in (
        ("development_benchmark", dev_predictions, dev_batch),
        ("final_blind", final_predictions, final_batch),
    ):
        for method, values in source.items():
            replicates = _prediction_replicates(values, evaluated_batch.frequency_hz.shape, method)
            for index, prediction in enumerate(replicates):
                prediction_payload[f"{role}__{method}__seed_{index}"] = prediction
    np.savez_compressed(output / "predictions.npz", **prediction_payload)
    (output / "physics_diagnostics.json").write_text(
        json.dumps(physics_diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "noiseless_metrics.json").write_text(
        json.dumps(noiseless_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    paired_e1_fft: dict[str, dict[str, Any]] = {}
    paired_e1_frontend: dict[str, dict[str, Any]] = {}
    paired_e2: dict[str, dict[str, Any]] = {}
    paired_teacher_ablation: dict[str, dict[str, Any]] = {}
    paired_distillation: dict[str, dict[str, Any]] = {}
    primary_snrs = [float(value) for value in config["evaluation"]["primary_snr_levels_db"]]
    e1_methods = ("S1/E1_truth", "S2/E1_truth")
    for method in e1_methods:
        candidate = final_errors[f"final_blind__{method}"]
        reference = final_errors["final_blind__B1_fft_peak"][0]
        paired_e1_fft[method] = {}
        for snr in primary_snrs:
            mask = np.isclose(final_batch.snr_db, snr)
            paired_e1_fft[method][str(snr)] = _paired_rmse_delta_ci(
                candidate[:, mask], reference[mask],
                int(config["evaluation"]["bootstrap_samples"]), bootstrap_rng,
            )
    for student, baseline in (("S1", "B7_multilag_correlation"), ("S2", "B6_4N_fft_peak")):
        method = f"{student}/E1_truth"
        candidate = final_errors[f"final_blind__{method}"]
        reference = final_errors[f"final_blind__{baseline}"][0]
        paired_e1_frontend[method] = {}
        for snr in primary_snrs:
            mask = np.isclose(final_batch.snr_db, snr)
            paired_e1_frontend[method][str(snr)] = _paired_rmse_delta_ci(
                candidate[:, mask], reference[mask],
                int(config["evaluation"]["bootstrap_samples"]), bootstrap_rng,
            )
    for student in ("S1", "S2"):
        e1_errors = final_errors[f"final_blind__{student}/E1_truth"]
        e2_methods = [
            method for method in final_predictions
            if method.startswith(f"{student}/E2")
        ]
        for method in e2_methods:
            paired_e2[method] = {}
            candidate = final_errors[f"final_blind__{method}"]
            for snr in primary_snrs:
                mask = np.isclose(final_batch.snr_db, snr)
                paired_e2[method][str(snr)] = _paired_rmse_delta_ci(
                    candidate[:, mask], e1_errors[:, mask],
                    int(config["evaluation"]["bootstrap_samples"]), bootstrap_rng,
                )
    for left, right, label in (
        ("S1/E2A_T2_local_soft", "S1/E2B_T2_truth_validation", "S1/E2A_soft_minus_E2B_soft"),
        ("S1/E2A_T3_full_response", "S1/E2A_T1_hard", "S1/T3_full_response_minus_T1_point"),
        ("S1/E2A_T2_local_soft", "S1/E2A_T1_hard", "S1/T2_local_soft_minus_T1_point"),
    ):
        paired_teacher_ablation[label] = {}
        left_errors = final_errors[f"final_blind__{left}"]
        right_errors = final_errors[f"final_blind__{right}"]
        for snr in primary_snrs:
            mask = np.isclose(final_batch.snr_db, snr)
            paired_teacher_ablation[label][str(snr)] = _paired_rmse_delta_ci(
                left_errors[:, mask], right_errors[:, mask],
                int(config["evaluation"]["bootstrap_samples"]), bootstrap_rng,
            )
    for student in ("S1", "S2"):
        e1_key = f"final_blind__{student}/E1_truth"
        candidates = [f"{student}/E3_T2_truth_plus_soft"]
        if student == "S1":
            candidates.append("S1/E3_T3_truth_plus_response")
        for candidate_name in candidates:
            key = f"final_blind__{candidate_name}"
            paired_distillation[candidate_name] = {}
            for snr in primary_snrs:
                mask = np.isclose(final_batch.snr_db, snr)
                paired_distillation[candidate_name][str(snr)] = _paired_rmse_delta_ci(
                    final_errors[key][:, mask], final_errors[e1_key][:, mask],
                    int(config["evaluation"]["bootstrap_samples"]), bootstrap_rng,
                )

    gate = _gate_decisions(
        final_metrics,
        paired_e1_fft,
        paired_e1_frontend,
        paired_distillation,
        noiseless_metrics,
        final_errors,
        final_batch.snr_db,
        config,
    )
    plot_files = _plots(output, {"final_blind": final_metrics}, errors, history_rows, config, final_batch)
    (output / "analysis.md").write_text(
        _analysis(
            config, final_metrics, paired_e1_fft, paired_e2, paired_teacher_ablation,
            paired_distillation, gate, noiseless_metrics, plot_files,
        ),
        encoding="utf-8",
    )

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    free_memory, total_memory = torch.cuda.mem_get_info(device)
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "data_type": "simulated single-target complex slow-time data",
        "data_roles": {"E001-E005": "development_benchmark", "E006_test": "final_blind"},
        "configuration": config,
        "code_version": {"commit": commit, "dirty_at_start": dirty},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(device),
            "free_gpu_bytes_after_run": int(free_memory),
            "total_gpu_bytes": int(total_memory),
            "platform": platform.platform(),
            "training_device": str(device),
        },
        "splits": split_records,
        "training_replicates": training_records,
        "noiseless_correctness": noiseless_metrics,
        "per_snr": {"development_benchmark": dev_metrics, "final_blind": final_metrics},
        "paired_E1_minus_fft_peak_rmse": paired_e1_fft,
        "paired_E1_minus_same_frontend_rmse": paired_e1_frontend,
        "paired_E2_minus_E1_rmse": paired_e2,
        "paired_teacher_ablation_rmse": paired_teacher_ablation,
        "paired_E3_minus_E1_rmse": paired_distillation,
        "gates": gate,
        "physics_diagnostics_file": "physics_diagnostics.json",
        "plots": plot_files,
        "artifact_files": [
            "config.yaml", "protocol.md", "metrics.json", "result.csv", "analysis.md", "predictions.npz",
            "training_history.csv", "physics_diagnostics.json", "noiseless_metrics.json", "datasets/", "models/", "figures/",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    archive_root = output / "attempts"
    artifact_paths = [
        path for path in output.rglob("*")
        if path.is_file()
        and path.name != "data_manifest.json"
        and not path.is_relative_to(archive_root)
    ]
    source_paths = [repo_root / "experiments/P0-E005/datasets/test.npz"]
    (output / "data_manifest.json").write_text(
        json.dumps(
            {
                "experiment_artifacts": _sha256_manifest(artifact_paths, output),
                "referenced_development_artifacts": _sha256_manifest(source_paths, repo_root),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return summary


def _g3_gate_windows(paired_distillation, final_errors, snr_db, primary_snrs, required):
    """Require paired intervals and seed-level direction agreement for G3."""
    snr_db = np.asarray(snr_db, dtype=np.float64)
    windows_by_method = {}
    seed_direction_by_method = {}
    for method, paired_by_snr in paired_distillation.items():
        student = method.split("/", 1)[0]
        candidate = final_errors[f"final_blind__{method}"]
        reference = final_errors[f"final_blind__{student}/E1_truth"]
        seed_direction_by_method[method] = {}
        eligible = set()
        for snr in primary_snrs:
            mask = np.isclose(snr_db, float(snr))
            candidate_rmse = np.sqrt(np.mean(candidate[:, mask] ** 2, axis=1))
            reference_rmse = np.sqrt(np.mean(reference[:, mask] ** 2, axis=1))
            seed_deltas = candidate_rmse - reference_rmse
            seeds_improved = int(np.sum(seed_deltas < 0.0))
            key = str(float(snr))
            seed_direction_by_method[method][key] = {
                "seeds_improved": seeds_improved,
                "seed_count": int(candidate.shape[0]),
                "per_seed_delta_rmse_hz": [float(value) for value in seed_deltas],
            }
            if paired_by_snr[key]["ci95_high_hz"] < 0.0 and seeds_improved >= 2:
                eligible.add(float(snr))
        windows_by_method[method] = [
            [float(value) for value in primary_snrs[index : index + required]]
            for index in range(len(primary_snrs) - required + 1)
            if all(float(snr) in eligible for snr in primary_snrs[index : index + required])
        ]
    return windows_by_method, seed_direction_by_method


def _gate_decisions(
    final_metrics,
    paired_e1_fft,
    paired_e1_frontend,
    paired_distillation,
    noiseless_metrics,
    final_errors,
    final_snr_db,
    config,
):
    primary = [str(float(value)) for value in config["evaluation"]["primary_snr_levels_db"]]
    required = int(config["evaluation"]["required_adjacent_primary_snr_points"])
    fft_pass_windows = {}
    for method, values in paired_e1_fft.items():
        passed = {snr for snr, delta in values.items() if delta["ci95_high_hz"] < 0.0}
        windows = [
            primary[index : index + required]
            for index in range(len(primary) - required + 1)
            if all(snr in passed for snr in primary[index : index + required])
        ]
        fft_pass_windows[method] = windows
    primary_pass = any(fft_pass_windows.values())

    zero_20 = final_metrics["B0_constant_zero"]["20.0"]["rmse_mean_seed_hz"]
    g1_candidates = []
    for student in ("S1", "S2"):
        key = f"{student}/E1_truth"
        metrics = final_metrics[key]["20.0"]["rmse_mean_seed_hz"]
        seeds_ok = sum(
            value["p95_abs_error_hz"] < 5.0 and value["mae_hz"] < 1.0
            for name, value in noiseless_metrics.items()
            if name.startswith(f"{student}/E1_truth/seed_")
        )
        g1_candidates.append(
            {"student": student, "noiseless_seeds_passed": seeds_ok, "rmse_20db_hz": metrics, "zero_rmse_20db_hz": zero_20}
        )
    g1 = any(item["noiseless_seeds_passed"] >= 2 and item["rmse_20db_hz"] < 0.5 * zero_20 for item in g1_candidates)

    same_frontend = {"S1": "B7_multilag_correlation", "S2": "B6_4N_fft_peak"}
    frontend_pass_windows = {}
    for student in same_frontend:
        model = f"{student}/E1_truth"
        values = paired_e1_frontend[model]
        passed = {snr for snr, delta in values.items() if delta["ci95_high_hz"] < 0.0}
        frontend_pass_windows[student] = [
            primary[index : index + required]
            for index in range(len(primary) - required + 1)
            if all(snr in passed for snr in primary[index : index + required])
        ]

    g3_pass_windows, g3_seed_direction = _g3_gate_windows(
        paired_distillation, final_errors, final_snr_db, [float(value) for value in primary], required
    )
    return {
        "G1_learning_feasibility": {"passed": bool(g1), "criteria": "noiseless MAE<1 Hz and P95<5 Hz for >=2/3 seeds; 20 dB RMSE at least 50% below zero predictor", "students": g1_candidates},
        "G2_classical_competitiveness": {"passed": bool(primary_pass and any(frontend_pass_windows.values())), "fft_primary_gate_passed": bool(primary_pass), "fft_windows_by_student": fft_pass_windows, "same_frontend_windows_by_student": frontend_pass_windows},
        "G3_distillation_value": {
            "passed": bool(any(g3_pass_windows.values())),
            "criteria": "paired 95% CI wholly below zero at two adjacent primary SNR points and at least 2/3 seed-level RMSE deltas below zero at each point",
            "passing_adjacent_windows_by_method": g3_pass_windows,
            "seed_direction_by_snr": g3_seed_direction,
        },
    }


def _analysis(config, metrics, paired_fft, paired_e2, paired_teacher_ablation, paired_distillation, gates, noiseless_metrics, plot_files):
    methods = ["B1_fft_peak", "B2_fft_interpolated", "B3_phase_difference", "B4_local_profile_ML", "B5_global_grid_profile_ML", "B6_4N_fft_peak", "B7_multilag_correlation", "S0/E1_supervised", "S1/E1_truth", "S2/E1_truth"]
    at_20 = [(name, metrics[name]["20.0"]["rmse_mean_seed_hz"]) for name in methods if name in metrics]
    pair_lines = [
        f"| {student} | {snr} | {row['delta_rmse_hz']:.4g} | [{row['ci95_low_hz']:.4g}, {row['ci95_high_hz']:.4g}] |"
        for student, values in paired_fft.items() for snr, row in values.items()
    ]
    distill_lines = [
        f"| {method} | {snr} | {row['delta_rmse_hz']:.4g} | [{row['ci95_low_hz']:.4g}, {row['ci95_high_hz']:.4g}] |"
        for method, values in paired_distillation.items() for snr, row in values.items()
    ]
    e2_lines = [
        f"| {method} | {snr} | {row['delta_rmse_hz']:.4g} | [{row['ci95_low_hz']:.4g}, {row['ci95_high_hz']:.4g}] |"
        for method, values in paired_e2.items() for snr, row in values.items()
    ]
    teacher_lines = [
        f"| {method} | {snr} | {row['delta_rmse_hz']:.4g} | [{row['ci95_low_hz']:.4g}, {row['ci95_high_hz']:.4g}] |"
        for method, values in paired_teacher_ablation.items() for snr, row in values.items()
    ]
    plots = "\n".join(f"![{Path(path).stem}]({path})" for path in plot_files)
    return f"""# P0-E006 分析

## 条件与盲测

AISAR 中 PyTorch CUDA 训练；PRF={config['prf_hz']} Hz，N={config['n_samples']}，独立训练种子 {config['data']['train_seeds']}。E001–E005 的 seed 7070 数据仅作为开发基准；本报告最终结果使用协议冻结后生成的 final seed {config['data']['final_test_seed']}，每个 SNR {config['data']['test_per_snr']} 例。全为复单音 AWGN 仿真。

S0 是冻结的 E005 CNN。S1 是 lags={config['model']['lag_features']} 的归一化复相关 MLP，并对比同样复相关的非学习相干评分。S2 输入固定 {int(config['n_samples']) * int(config['model']['spectrum_fft_factor'])} 点 FFT 对数功率，是频率响应引导混合模型。完整 local profile 目标只在预注册指定的 S1 对照中使用。

## 20 dB 代表性 RMSE

| Method | RMSE (Hz) |
|---|---:|
{chr(10).join(f'| {name} | {value:.6g} |' for name, value in at_20)}

## E1 对 FFT 峰值的配对比较

负差值支持学生；区间跨零表示证据不充分。

| Student | SNR (dB) | E1 − FFT RMSE (Hz) | paired 95% CI (Hz) |
|---|---:|---:|---:|
{chr(10).join(pair_lines)}

## 蒸馏比较

| E3 method | SNR (dB) | E3 − E1 RMSE (Hz) | paired 95% CI (Hz) |
|---|---:|---:|---:|
{chr(10).join(distill_lines)}

## E2 教师目标及验证协议对照

| E2 method (minus matched reference) | SNR (dB) | ΔRMSE (Hz) | paired 95% CI (Hz) |
|---|---:|---:|---:|
{chr(10).join(e2_lines)}

| Teacher/validation contrast | SNR (dB) | ΔRMSE (Hz) | paired 95% CI (Hz) |
|---|---:|---:|---:|
{chr(10).join(teacher_lines)}

## 独立阶段门

- **G1 learning feasibility:** **{'通过' if gates['G1_learning_feasibility']['passed'] else '未通过'}**；无噪声与 20 dB 标准见 `metrics.json`。
- **G2 classical competitiveness:** **{'通过' if gates['G2_classical_competitiveness']['passed'] else '未通过'}**；须同时超过 FFT 主门和同一物理前端基线。
- **G3 distillation value:** **{'通过' if gates['G3_distillation_value']['passed'] else '未通过'}**；要求配对 95% 区间低于零、相邻主 SNR 两点通过，且每点至少 2/3 个训练种子的 RMSE 方向改善。

## 物理变换与分组

`physics_diagnostics.json` 报告公共相位、整体复幅度缩放、频率平移和复共轭下学生输出的实测偏差。对特征的解析不变性不自动等于网络严格等变。`result.csv` 含 SNR、频率、初相、幅度和 FFT 栅格偏移分组。

## 图表

{plots}

## 限制

这些结果只适用于单目标复正弦 + 圆对称复 AWGN 仿真。S2 显式使用 FFT 物理前端；其性能不能解释为纯原始 IQ 端到端优势。局部教师似然只覆盖教师峰周围 ±2 个原生 FFT bin，不是全局后验。任何结果均不证明杂波检测、多目标能力、真实数据泛化、目标角度/速度估计或部署加速。
"""
