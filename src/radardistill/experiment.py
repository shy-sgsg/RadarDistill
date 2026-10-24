"""Run and report the P0 simulation, baseline, and distillation comparison."""

from __future__ import annotations

import csv
import json
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

from .baselines import (
    estimate_fft_interpolated,
    estimate_fft_peak,
    estimate_mle,
    estimate_phase_difference,
    frequency_crlb_hz,
    local_teacher_soft_target,
    wrapped_error,
)
from .cnn import TinyCircularCNN, circle_to_frequency, frequency_to_circle
from .signal import SignalBatch, generate_batch


def _save_batch(path: Path, batch: SignalBatch, extra: dict[str, np.ndarray] | None = None) -> None:
    values = {
        "iq": batch.iq,
        "frequency_hz": batch.frequency_hz,
        "phase_rad": batch.phase_rad,
        "amplitude": batch.amplitude,
        "snr_db": batch.snr_db,
        "dataset_seed": np.asarray(batch.seed, dtype=np.int64),
    }
    if extra:
        values.update(extra)
    np.savez_compressed(path, **values)


def _teacher_labels(batch: SignalBatch, prf_hz: float) -> tuple[np.ndarray, np.ndarray]:
    teacher_frequency = estimate_mle(batch.iq, prf_hz)
    noise_power = batch.amplitude**2 * np.power(10.0, -batch.snr_db / 10.0)
    soft_target = local_teacher_soft_target(
        batch.iq,
        teacher_frequency,
        noise_power,
        prf_hz,
    )
    return np.asarray(teacher_frequency), soft_target


def _combine_batches(batches: list[SignalBatch], seed: int) -> SignalBatch:
    return SignalBatch(
        iq=np.concatenate([batch.iq for batch in batches], axis=0),
        frequency_hz=np.concatenate([batch.frequency_hz for batch in batches]),
        phase_rad=np.concatenate([batch.phase_rad for batch in batches]),
        amplitude=np.concatenate([batch.amplitude for batch in batches]),
        snr_db=np.concatenate([batch.snr_db for batch in batches]),
        seed=seed,
    )


def _student_input(iq: np.ndarray, representation: str) -> np.ndarray:
    """Prepare model inputs while preserving raw IQ for data records and baselines."""
    values = np.asarray(iq)
    if representation == "raw_iq":
        return values
    if representation == "unit_phase_increment":
        products = values[:, 1:] * np.conjugate(values[:, :-1])
        magnitudes = np.abs(products)
        return np.divide(
            products,
            magnitudes,
            out=np.zeros_like(products),
            where=magnitudes > 0.0,
        )
    raise ValueError(f"unknown student input representation: {representation}")


def _select_validation_target(
    protocol: str,
    validation_truth: np.ndarray,
    validation_teacher: np.ndarray,
) -> np.ndarray:
    """Choose E2 checkpoint-selection labels through an explicit protocol."""
    if protocol == "teacher":
        return validation_teacher
    if protocol == "truth":
        return validation_truth
    raise ValueError("E2 validation protocol must be 'teacher' or 'truth'")


def _bootstrap_rmse_ci(errors: np.ndarray, rng: np.random.Generator, samples: int) -> tuple[float, float]:
    errors = np.asarray(errors, dtype=np.float64)
    if len(errors) == 0:
        return float("nan"), float("nan")
    draws = rng.integers(0, len(errors), size=(samples, len(errors)))
    rmse = np.sqrt(np.mean(errors[draws] ** 2, axis=1))
    low, high = np.quantile(rmse, [0.025, 0.975])
    return float(low), float(high)


def _metrics(
    errors: np.ndarray,
    rng: np.random.Generator,
    bootstrap_samples: int,
    outlier_threshold_hz: float,
) -> dict[str, float | int]:
    errors = np.asarray(errors, dtype=np.float64)
    ci_low, ci_high = _bootstrap_rmse_ci(errors, rng, bootstrap_samples)
    return {
        "n": int(len(errors)),
        "mae_hz": float(np.mean(np.abs(errors))),
        "rmse_hz": float(np.sqrt(np.mean(errors**2))),
        "bias_hz": float(np.mean(errors)),
        "outlier_rate": float(np.mean(np.abs(errors) > outlier_threshold_hz)),
        "rmse_ci95_low_hz": ci_low,
        "rmse_ci95_high_hz": ci_high,
    }


def _test_subsets(batch: SignalBatch, prf_hz: float) -> dict[str, np.ndarray]:
    n = len(batch.frequency_hz)
    subsets: dict[str, np.ndarray] = {"all": np.ones(n, dtype=bool)}
    frequency_bins = np.linspace(-prf_hz / 2.0, prf_hz / 2.0, 6)
    frequency_ids = np.clip(np.digitize(batch.frequency_hz, frequency_bins[1:-1]), 0, 4)
    for index in range(5):
        subsets[f"frequency_bin_{index}"] = frequency_ids == index

    phase_ids = np.floor((batch.phase_rad + np.pi) / (np.pi / 2.0)).astype(int) % 4
    for index in range(4):
        subsets[f"initial_phase_sector_{index}"] = phase_ids == index

    amp_edges = np.quantile(batch.amplitude, [0.0, 0.25, 0.5, 0.75, 1.0])
    amp_ids = np.clip(np.digitize(batch.amplitude, amp_edges[1:-1]), 0, 3)
    for index in range(4):
        subsets[f"amplitude_quartile_{index}"] = amp_ids == index

    n_samples = batch.iq.shape[-1]
    bin_width = prf_hz / n_samples
    offset = (batch.frequency_hz / bin_width + 0.5) % 1.0 - 0.5
    offset_ids = np.clip(np.floor((offset + 0.5) * 4.0).astype(int), 0, 3)
    for index in range(4):
        subsets[f"fft_bin_offset_quartile_{index}"] = offset_ids == index
    return subsets


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "experiment_id",
        "method",
        "train_seed",
        "snr_db",
        "subset",
        "n",
        "mae_hz",
        "rmse_hz",
        "bias_hz",
        "outlier_rate",
        "crlb_sd_hz",
        "rmse_ci95_low_hz",
        "rmse_ci95_high_hz",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_rmse_difference_ci(
    supervised_errors: np.ndarray,
    joint_errors: np.ndarray,
    rng: np.random.Generator,
    samples: int,
) -> dict[str, float]:
    """Two-level paired bootstrap over trained seeds and shared test samples."""
    seed_count, test_count = supervised_errors.shape
    delta = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        seed_indices = rng.integers(0, seed_count, size=seed_count)
        sample_indices = rng.integers(0, test_count, size=test_count)
        supervised = supervised_errors[np.ix_(seed_indices, sample_indices)]
        joint = joint_errors[np.ix_(seed_indices, sample_indices)]
        delta[draw] = np.sqrt(np.mean(joint**2)) - np.sqrt(np.mean(supervised**2))
    low, high = np.quantile(delta, [0.025, 0.975])
    estimate = np.sqrt(np.mean(joint_errors**2)) - np.sqrt(np.mean(supervised_errors**2))
    return {"delta_rmse_hz": float(estimate), "ci95_low_hz": float(low), "ci95_high_hz": float(high)}


def _analysis_text(
    config: dict,
    metrics: dict,
    paired: dict,
    paired_e1_fft: dict,
) -> str:
    snr_values = config["test_snr_levels_db"]
    rows = metrics["per_snr"]
    zero_method = "constant_zero"
    e1_method = "E1_supervised"
    e2_method = "E2_teacher_KD"
    e3_method = "E3_supervised_plus_KD"
    pooling = str(config.get("pooling", "mean"))
    if pooling == "flatten":
        student_description = (
            f"NumPy Conv1D with {config['channels']} filters of length {config['kernel_size']}, "
            f"ReLU, flatten, Dense({config.get('hidden_units', 32)}), ReLU, and a two-value circular output"
        )
    else:
        student_description = (
            f"NumPy Conv1D with {config['channels']} filters of length {config['kernel_size']}, "
            "ReLU, global average pooling, and a two-value circular output"
        )
    clear_snr = []
    for snr in snr_values:
        key = str(float(snr))
        if rows[e1_method][key]["rmse_mean_hz"] < 0.8 * rows[zero_method][key]["rmse_mean_hz"]:
            clear_snr.append(float(snr))

    strongest_snr = max(snr_values)
    strongest_key = str(float(strongest_snr))
    highest_rows = {
        name: rows[name][strongest_key]["rmse_mean_hz"]
        for name in ("B0_fft_peak", "B1_fft_interpolated", "B2_phase_difference", "B3_continuous_ML", e1_method, e2_method, e3_method)
    }
    delta_lines = []
    for snr in snr_values:
        key = str(float(snr))
        result = paired[key]
        delta_lines.append(
            f"| {float(snr):g} | {result['delta_rmse_hz']:.6g} | "
            f"[{result['ci95_low_hz']:.6g}, {result['ci95_high_hz']:.6g}] |"
        )
    fft_delta_lines = []
    for snr in snr_values:
        key = str(float(snr))
        result = paired_e1_fft[key]
        fft_delta_lines.append(
            f"| {float(snr):g} | {result['delta_rmse_hz']:.6g} | "
            f"[{result['ci95_low_hz']:.6g}, {result['ci95_high_hz']:.6g}] |"
        )
    best_comparison = min(highest_rows.items(), key=lambda item: item[1])
    e1_beats_fft = [
        float(snr)
        for snr in snr_values
        if rows[e1_method][str(float(snr))]["rmse_mean_hz"]
        < rows["B0_fft_peak"][str(float(snr))]["rmse_mean_hz"]
    ]
    learning_assessment = (
        f"E1 is at least 20% better than the zero-frequency predictor at {len(clear_snr)}/{len(snr_values)} "
        f"test SNR points ({', '.join(f'{value:g} dB' for value in clear_snr) or 'none'}). This is evidence of "
        "basic frequency learning, but it is not sufficient for the P0 continuation gate. "
        f"E1 beats FFT peak at: {', '.join(f'{value:g} dB' for value in e1_beats_fft) or 'none of the tested SNR points'}."
    )
    primary_snrs = [float(value) for value in config.get("primary_snr_levels_db", [])]
    required_adjacent_points = int(config.get("required_adjacent_primary_snr_points", 0))
    passing_snrs = {
        snr for snr in primary_snrs
        if paired_e1_fft[str(float(snr))]["ci95_high_hz"] < 0.0
    }
    passing_windows = [
        primary_snrs[index : index + required_adjacent_points]
        for index in range(max(0, len(primary_snrs) - required_adjacent_points + 1))
        if required_adjacent_points > 0
        and all(snr in passing_snrs for snr in primary_snrs[index : index + required_adjacent_points])
    ]
    gate_passed = bool(passing_windows)
    passing_window_text = "; ".join(
        f"{window[0]:g}–{window[-1]:g} dB" for window in passing_windows
    ) or "无"
    fft_comparison_title = (
        "Preregistered E1 vs. FFT peak comparison" if primary_snrs else "E1 vs. FFT peak comparison"
    )
    primary_gate_section = ""
    primary_gate_assessment = ""
    if primary_snrs:
        primary_gate_section = (
            f"\n\nPrimary gate: **{'passed' if gate_passed else 'not passed'}**. "
            f"It requires wholly negative intervals at {required_adjacent_points} consecutive primary SNR points "
            f"({', '.join(f'{value:g}' for value in primary_snrs)} dB). Passing windows: {passing_window_text}."
        )
        primary_gate_assessment = (
            f"\n- **P0-E005 preregistered gate:** **{'passed' if gate_passed else 'not passed'}**. "
            f"Passing windows: {passing_window_text}."
        )
    input_representation = str(config.get("input_representation", "raw_iq"))
    if input_representation == "unit_phase_increment":
        input_description = (
            "单位相位增量 `u[n] = x[n+1]·conj(x[n]) / |x[n+1]·conj(x[n])|`；"
            f"输入长度为 {int(config['n_samples']) - 1}，零幅度增量设为 0。"
        )
    else:
        input_description = "原始复 IQ。"
    e2_validation_target = str(config.get("e2_validation_target", "truth_legacy"))
    e2_validation_description = {
        "teacher": "teacher-validation selection (E2-A)",
        "truth": "truth-validation selection (E2-B)",
        "truth_legacy": "truth-validation selection inherited from the original protocol (E2-B)",
    }.get(e2_validation_target, f"invalid E2 validation mode: {e2_validation_target}")
    return f"""# P0 experiment analysis

**Experiment:** `{config['experiment_id']}`  
**Scope:** single-target complex slow-time Doppler estimation under the stated AWGN model. All data are simulated.

## Conditions

- PRF: {config['prf_hz']} Hz; samples per dwell: {config['n_samples']}.
- Training SNR levels: {', '.join(str(value) for value in config['train_snr_levels_db'])} dB.
- Test SNR levels: {', '.join(str(value) for value in snr_values)} dB.
- Independent training replicates: {len(config['train_seeds'])}; held-out samples per SNR: {config['test_per_snr']}.
- Student input: {input_description}
- Student: {student_description}.
- E1 trains on physical truth; E2 trains only on local teacher likelihood soft targets and uses {e2_validation_description}; E3 combines physical truth loss with teacher loss of weight {config['teacher_weight']}.
- Errors wrap modulo PRF. An outlier is an absolute wrapped error above half one native FFT bin ({config['prf_hz'] / (2 * config['n_samples']):.6g} Hz).

## Highest-SNR RMSE comparison

At {strongest_snr:g} dB, mean student RMSE across training seeds and reference-method RMSE are:

| Method | RMSE (Hz) |
|---|---:|
{chr(10).join(f'| {name} | {value:.6g} |' for name, value in highest_rows.items())}

Best measured method at this SNR: **{best_comparison[0]}** ({best_comparison[1]:.6g} Hz). This comparison is specific to the simulated signal model and test distribution.

## Did the student learn Doppler?

{learning_assessment}

Read `result.csv` for all SNRs and grouped diagnostics by frequency, initial phase, amplitude, and FFT-bin offset. The grouped estimates use the same independent test set.

## Distillation comparison

The paired bootstrap compares E3 minus E1 RMSE on shared held-out samples and resamples training replicates when multiple are available. A negative delta favors joint training; an interval spanning zero is inconclusive. Single-seed diagnostics do not estimate between-training-seed variability.

| SNR (dB) | E3 − E1 RMSE (Hz) | 95% paired bootstrap interval (Hz) |
|---:|---:|---:|
{chr(10).join(delta_lines)}

## {fft_comparison_title}

Each delta is E1 minus B0 FFT-peak RMSE; negative values favor E1. The paired bootstrap resamples the three training seeds and shared held-out samples.

| SNR (dB) | E1 − FFT peak RMSE (Hz) | 95% paired bootstrap interval (Hz) |
|---:|---:|---:|
{chr(10).join(fft_delta_lines)}
{primary_gate_section}

## Bound and interpretation

`crlb_sd_hz` in `result.csv` is the standard-deviation Cramér–Rao bound for an unbiased estimator with unknown complex amplitude/phase and known per-sample SNR. It is a theoretical reference, not a guaranteed finite-sample RMSE. The continuous ML estimator is the profile-likelihood reference for this correctly specified single-tone model.

## P0 stage assessment

- **Implementation and pipeline:** the experiment completed with independent train/validation/test seeds and saved inputs, labels, weights, and predictions; repository test results are recorded separately.
- **Basic signal learning:** {learning_assessment}
{primary_gate_assessment}
- **P1:** this diagnostic alone does not start P1. Record the result, then follow the P0 error-source analysis and route decision.

Do not interpret these single-target AWGN results as GMTI detection, clutter rejection, real-data generalization, or deployment evidence. Keep the held-out test data fixed.
"""


def run_p0(config: dict, output_dir: str | Path) -> dict:
    """Execute one fully recorded P0 experiment and write its artifacts."""
    required = {
        "experiment_id", "prf_hz", "n_samples", "train_count", "validation_count", "test_per_snr",
        "train_snr_levels_db", "test_snr_levels_db", "train_seeds", "validation_seed", "test_seed",
        "epochs", "batch_size", "learning_rate", "patience", "teacher_weight", "channels", "kernel_size",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing required experiment config keys: {', '.join(missing)}")
    output = Path(output_dir)
    datasets_dir = output / "datasets"
    models_dir = output / "models"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    prf_hz = float(config["prf_hz"])
    n_samples = int(config["n_samples"])
    input_representation = str(config.get("input_representation", "raw_iq"))
    e2_validation_mode = str(config.get("e2_validation_target", "truth_legacy"))
    if e2_validation_mode not in {"teacher", "truth", "truth_legacy"}:
        raise ValueError("e2_validation_target must be 'teacher', 'truth', or 'truth_legacy'")
    rng = np.random.default_rng(int(config.get("bootstrap_seed", 9182)))

    test_batches = [
        generate_batch(
            count=int(config["test_per_snr"]),
            seed=int(config["test_seed"]) + 1000 * index,
            prf_hz=prf_hz,
            n_samples=n_samples,
            snr_db=float(snr),
        )
        for index, snr in enumerate(config["test_snr_levels_db"])
    ]
    test = _combine_batches(test_batches, int(config["test_seed"]))
    student_test_iq = _student_input(test.iq, input_representation)
    test_teacher, test_teacher_soft = _teacher_labels(test, prf_hz)
    _save_batch(
        datasets_dir / "test.npz",
        test,
        {"teacher_frequency_hz": test_teacher, "teacher_soft_target": test_teacher_soft},
    )

    baseline_predictions: dict[str, np.ndarray] = {
        "constant_zero": np.zeros(len(test.iq), dtype=np.float64),
        "random_uniform": np.random.default_rng(int(config["test_seed"]) + 17).uniform(
            -prf_hz / 2.0, prf_hz / 2.0, size=len(test.iq)
        ),
        "B0_fft_peak": np.asarray(estimate_fft_peak(test.iq, prf_hz)),
        "B1_fft_interpolated": np.asarray(estimate_fft_interpolated(test.iq, prf_hz)),
        "B2_phase_difference": np.asarray(estimate_phase_difference(test.iq, prf_hz)),
        "B3_continuous_ML": np.asarray(test_teacher),
    }

    train_records = []
    model_predictions: dict[str, list[np.ndarray]] = {
        "E1_supervised": [],
        "E2_teacher_KD": [],
        "E3_supervised_plus_KD": [],
    }
    model_names = {
        "E1_supervised": "E1_supervised",
        "E2_teacher_KD": "E2_teacher_KD",
        "E3_supervised_plus_KD": "E3_supervised_plus_KD",
    }

    for replicate_index, train_seed_value in enumerate(config["train_seeds"]):
        train_seed = int(train_seed_value)
        train = generate_batch(
            count=int(config["train_count"]),
            seed=train_seed,
            prf_hz=prf_hz,
            n_samples=n_samples,
            snr_levels_db=tuple(float(value) for value in config["train_snr_levels_db"]),
        )
        validation_seed = int(config["validation_seed"]) + replicate_index
        validation = generate_batch(
            count=int(config["validation_count"]),
            seed=validation_seed,
            prf_hz=prf_hz,
            n_samples=n_samples,
            snr_levels_db=tuple(float(value) for value in config["train_snr_levels_db"]),
        )
        train_teacher, train_soft = _teacher_labels(train, prf_hz)
        validation_teacher, validation_soft = _teacher_labels(validation, prf_hz)
        student_train_iq = _student_input(train.iq, input_representation)
        student_validation_iq = _student_input(validation.iq, input_representation)
        _save_batch(
            datasets_dir / f"train_seed_{train_seed}.npz",
            train,
            {"teacher_frequency_hz": train_teacher, "teacher_soft_target": train_soft},
        )
        _save_batch(
            datasets_dir / f"validation_seed_{validation_seed}.npz",
            validation,
            {"teacher_frequency_hz": validation_teacher, "teacher_soft_target": validation_soft},
        )

        truth_targets = frequency_to_circle(train.frequency_hz, prf_hz).astype(np.float32)
        validation_truth = frequency_to_circle(validation.frequency_hz, prf_hz).astype(np.float32)
        validation_selection_mode = "truth" if e2_validation_mode != "teacher" else "teacher"
        validation_selection_target = _select_validation_target(
            validation_selection_mode,
            validation_truth,
            validation_soft.astype(np.float32),
        )
        model_seed = train_seed + 771
        train_specs = [
            ("E1_supervised", truth_targets, None, 0.0, validation_truth, "truth"),
            (
                "E2_teacher_KD",
                train_soft.astype(np.float32),
                None,
                0.0,
                validation_selection_target,
                e2_validation_mode,
            ),
            (
                "E3_supervised_plus_KD",
                truth_targets,
                train_soft.astype(np.float32),
                float(config["teacher_weight"]),
                validation_truth,
                "truth",
            ),
        ]
        for strategy, targets, teacher_targets, teacher_weight, selection_targets, selection_name in train_specs:
            model = TinyCircularCNN(
                n_samples=student_train_iq.shape[1],
                seed=model_seed,
                channels=int(config["channels"]),
                kernel_size=int(config["kernel_size"]),
                pooling=str(config.get("pooling", "mean")),
                hidden_units=int(config.get("hidden_units", 32)),
            )
            history = model.fit(
                student_train_iq,
                targets,
                epochs=int(config["epochs"]),
                batch_size=int(config["batch_size"]),
                learning_rate=float(config["learning_rate"]),
                seed=model_seed,
                teacher_targets=teacher_targets,
                distillation_weight=teacher_weight,
                validation_data=(student_validation_iq, selection_targets),
                patience=int(config["patience"]),
            )
            prediction = circle_to_frequency(model.predict(student_test_iq), prf_hz)
            model_predictions[model_names[strategy]].append(np.asarray(prediction))
            model_path = models_dir / f"seed_{train_seed}_{strategy}.npz"
            model.save(str(model_path))
            train_records.append(
                {
                    "method": strategy,
                    "train_seed": train_seed,
                    "model_seed": model_seed,
                    "epochs_completed": len(history),
                    "final_training_loss": float(history[-1]),
                    "validation_selection_target": selection_name,
                    "best_validation_selection_mse": float(
                        np.mean((model.predict(student_validation_iq) - selection_targets) ** 2)
                    ),
                    "best_validation_truth_mse": float(
                        np.mean((model.predict(student_validation_iq) - validation_truth) ** 2)
                    ),
                    "weights_file": str(model_path.relative_to(output)),
                }
            )

    all_predictions = dict(baseline_predictions)
    all_predictions.update({key: np.mean(np.stack(values), axis=0) for key, values in model_predictions.items()})
    result_rows: list[dict[str, object]] = []
    per_snr: dict[str, dict[str, dict[str, float]]] = {}
    test_subsets = _test_subsets(test, prf_hz)
    snr_list = [float(value) for value in config["test_snr_levels_db"]]
    seed_labels = {name: ["reference"] for name in baseline_predictions}
    seed_labels.update({name: [int(value) for value in config["train_seeds"]] for name in model_predictions})

    for method, predictions in all_predictions.items():
        if method in model_predictions:
            replicate_predictions = model_predictions[method]
        else:
            replicate_predictions = [predictions]
        per_snr[method] = {}
        for replicate_index, prediction in enumerate(replicate_predictions):
            train_seed_label = seed_labels[method][replicate_index]
            errors = np.asarray(wrapped_error(prediction, test.frequency_hz, prf_hz), dtype=np.float64)
            for snr in snr_list:
                mask = np.isclose(test.snr_db, snr)
                values = _metrics(
                    errors[mask], rng, int(config.get("bootstrap_samples", 500)), prf_hz / (2.0 * n_samples)
                )
                crlb = float(frequency_crlb_hz(prf_hz, n_samples, snr))
                result_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "method": method,
                        "train_seed": train_seed_label,
                        "snr_db": snr,
                        "subset": "all",
                        **values,
                        "crlb_sd_hz": crlb,
                    }
                )
                if method not in per_snr:
                    per_snr[method] = {}
                if str(snr) not in per_snr[method]:
                    per_snr[method][str(snr)] = {"seed_rmse_hz": []}
                per_snr[method][str(snr)]["seed_rmse_hz"].append(float(values["rmse_hz"]))

            for subset, subset_mask in test_subsets.items():
                if subset == "all":
                    continue
                values = _metrics(
                    errors[subset_mask], rng, int(config.get("bootstrap_samples", 500)), prf_hz / (2.0 * n_samples)
                )
                result_rows.append(
                    {
                        "experiment_id": config["experiment_id"],
                        "method": method,
                        "train_seed": train_seed_label,
                        "snr_db": "all",
                        "subset": subset,
                        **values,
                        "crlb_sd_hz": "",
                    }
                )

    for method, snr_metrics in per_snr.items():
        for key, value in snr_metrics.items():
            rmse_values = np.asarray(value.pop("seed_rmse_hz"), dtype=np.float64)
            value["rmse_mean_hz"] = float(np.mean(rmse_values))
            value["rmse_std_hz"] = float(np.std(rmse_values, ddof=1)) if len(rmse_values) > 1 else 0.0
            value["crlb_sd_hz"] = float(frequency_crlb_hz(prf_hz, n_samples, float(key)))

    paired_by_snr = {}
    if "E1_supervised" in model_predictions and "E3_supervised_plus_KD" in model_predictions:
        e1_errors = np.stack([
            wrapped_error(prediction, test.frequency_hz, prf_hz)
            for prediction in model_predictions["E1_supervised"]
        ])
        e3_errors = np.stack([
            wrapped_error(prediction, test.frequency_hz, prf_hz)
            for prediction in model_predictions["E3_supervised_plus_KD"]
        ])
        for snr in snr_list:
            mask = np.isclose(test.snr_db, snr)
            paired_by_snr[str(snr)] = _paired_rmse_difference_ci(
                e1_errors[:, mask], e3_errors[:, mask], rng, int(config.get("bootstrap_samples", 500))
            )

    paired_e1_fft_by_snr = {}
    if "E1_supervised" in model_predictions and "B0_fft_peak" in baseline_predictions:
        e1_errors = np.stack([
            wrapped_error(prediction, test.frequency_hz, prf_hz)
            for prediction in model_predictions["E1_supervised"]
        ])
        fft_peak_errors = wrapped_error(baseline_predictions["B0_fft_peak"], test.frequency_hz, prf_hz)
        fft_peak_errors = np.broadcast_to(fft_peak_errors, e1_errors.shape)
        for snr in snr_list:
            mask = np.isclose(test.snr_db, snr)
            paired_e1_fft_by_snr[str(snr)] = _paired_rmse_difference_ci(
                fft_peak_errors[:, mask], e1_errors[:, mask], rng, int(config.get("bootstrap_samples", 500))
            )

    _write_rows(output / "result.csv", result_rows)
    prediction_payload = {
        f"{method}__seed_{seed}": np.asarray(prediction)
        for method, predictions in model_predictions.items()
        for seed, prediction in zip(config["train_seeds"], predictions)
    }
    prediction_payload.update({f"{method}__reference": values for method, values in baseline_predictions.items()})
    np.savez_compressed(output / "predictions.npz", **prediction_payload)

    summary = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "data_type": "simulated single-target complex slow-time data",
        "configuration": config,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "device": "CPU; legacy student uses the NumPy implementation",
        },
        "student": {
            "architecture": (
                f"Conv1D(2, {int(config['channels'])}, kernel={int(config['kernel_size'])}), ReLU, flatten, "
                f"Dense({int(config.get('hidden_units', 32))}), ReLU, linear 2-vector"
                if config.get("pooling", "mean") == "flatten"
                else f"Conv1D(2, {int(config['channels'])}, kernel={int(config['kernel_size'])}), ReLU, "
                "global average pooling, linear 2-vector"
            ),
            "pooling": str(config.get("pooling", "mean")),
            "input_representation": input_representation,
            "input_samples": int(student_test_iq.shape[1]),
            "hidden_units": int(config.get("hidden_units", 32)) if config.get("pooling", "mean") == "flatten" else None,
            "training_replicates": train_records,
            "output_parameterization": "[cos(2*pi*f/PRF), sin(2*pi*f/PRF)]",
        },
        "teacher": {
            "method": "continuous profile maximum likelihood from the periodogram",
            "soft_target": "local profile-likelihood mean of circular frequency vectors in +/- 2 FFT bins around the ML mode",
            "note": "Teacher estimates and soft targets are stored separately from physical frequency truth.",
        },
        "validation_protocols": {
            "E1": "physical truth",
            "E2": e2_validation_mode,
            "E3": "physical truth (truth is part of the training objective)",
        },
        "metrics_definition": {
            "frequency_error": "wrapped to [-PRF/2, PRF/2)",
            "outlier_threshold_hz": prf_hz / (2.0 * n_samples),
            "crlb": "standard-deviation CRLB for a complex single tone with unknown complex amplitude/phase and known per-sample SNR",
            "confidence_interval": "95% percentile bootstrap interval over held-out test samples",
        },
        "per_snr": per_snr,
        "paired_E3_minus_E1_rmse": paired_by_snr,
        "paired_E1_minus_B0_fft_peak_rmse": paired_e1_fft_by_snr,
        "artifact_files": [
            "config.yaml", "metrics.json", "result.csv", "analysis.md", "predictions.npz",
            "datasets/train_seed_*.npz", "datasets/validation_seed_*.npz", "datasets/test.npz", "models/*.npz",
        ],
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "analysis.md").write_text(
        _analysis_text(config, summary, paired_by_snr, paired_e1_fft_by_snr), encoding="utf-8"
    )
    return summary
