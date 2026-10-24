#!/usr/bin/env python3
"""Measure D1 student and APX15 FFT-subchain inference costs at the IQ boundary."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d1_evaluation import (  # noqa: E402
    _load_student_checkpoint,
    _predict_student,
)
from radardistill.evaluation.inference_benchmark import (  # noqa: E402
    summarize_latency_samples,
)
from radardistill.simulation.d0 import generate_d0_batch  # noqa: E402
from radardistill.teachers.apx15_doppler import APX15DopplerTeacher  # noqa: E402


def _measure(
    call: Callable[[], Any], *, warmup: int, repeats: int, synchronize: Callable[[], None]
) -> list[float]:
    for _ in range(warmup):
        call()
    synchronize()
    values: list[float] = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        call()
        synchronize()
        values.append(time.perf_counter() - start)
    return values


def _environment(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device),
        "cuda_device_total_memory_bytes": int(properties.total_memory),
        "cpu_threads": torch.get_num_threads(),
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def _benchmark_student(
    iq: np.ndarray,
    *,
    checkpoint: Path,
    config: dict[str, Any],
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
    device: torch.device,
) -> dict[str, Any]:
    model = _load_student_checkpoint(checkpoint, config, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    synchronize = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    results: dict[str, Any] = {
        "device": str(device),
        "parameter_count": int(parameter_count),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "batch_sizes": {},
    }
    for batch_size in batch_sizes:
        samples = iq[:batch_size]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
        elapsed = _measure(
            lambda: _predict_student(model, samples, device, float(config["radar"]["prf_hz"])),
            warmup=warmup,
            repeats=repeats,
            synchronize=synchronize,
        )
        summary = summarize_latency_samples(elapsed, batch_size=batch_size)
        if device.type == "cuda":
            summary["allocated_before_measurement_bytes"] = int(allocated_before)
            summary["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            summary["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        results["batch_sizes"][str(batch_size)] = summary
    return results


def _benchmark_teacher(
    iq: np.ndarray,
    *,
    d0_config: dict[str, Any],
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    teacher = APX15DopplerTeacher(d0_config)
    results: dict[str, Any] = {
        "device": "cpu",
        "implementation_scope": "APX15 ColFFTTranspose slow-time FFT subchain and D0 hard estimator",
        "batch_sizes": {},
    }
    try:
        for batch_size in batch_sizes:
            samples = iq[:batch_size]
            elapsed = _measure(
                lambda: teacher.predict(samples),
                warmup=warmup,
                repeats=repeats,
                synchronize=lambda: None,
            )
            results["batch_sizes"][str(batch_size)] = summarize_latency_samples(
                elapsed, batch_size=batch_size
            )
    finally:
        teacher.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d1-config", type=Path, default=ROOT / "experiments/D1-E001/config.yaml")
    parser.add_argument("--d0-config", type=Path, default=ROOT / "experiments/D0-E001/config.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "experiments/D1-E001/formal/seed_1/D_progressive_distillation/best_inference.pt",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiments/D1-E001/benchmark/inference_cost.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32, 128])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats <= 0:
        parser.error("--warmup must be non-negative and --repeats must be positive")
    if not arguments.batch_sizes or any(size <= 0 for size in arguments.batch_sizes):
        parser.error("--batch-sizes must contain positive integers")

    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("the D1 cost benchmark requires an available CUDA device")
    if device.index is not None and device.index >= torch.cuda.device_count():
        parser.error(f"CUDA device {device} is unavailable")
    torch.cuda.set_device(device)
    torch.set_num_threads(1)

    d1_config = yaml.safe_load(arguments.d1_config.resolve().read_text(encoding="utf-8"))
    d0_config = yaml.safe_load(arguments.d0_config.resolve().read_text(encoding="utf-8"))
    max_batch = max(arguments.batch_sizes)
    synthetic = generate_d0_batch(
        max_batch,
        seed=2026102410,
        snr_db=10.0,
        presence_ratio=1.0,
        config=d0_config,
    )
    iq = np.asarray(synthetic.iq, dtype=np.complex64)
    synchronize = lambda: torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    student_cpu = _benchmark_student(
        iq,
        checkpoint=arguments.checkpoint.resolve(),
        config=d1_config,
        batch_sizes=arguments.batch_sizes,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        device=torch.device("cpu"),
    )
    student_cuda = _benchmark_student(
        iq,
        checkpoint=arguments.checkpoint.resolve(),
        config=d1_config,
        batch_sizes=arguments.batch_sizes,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        device=device,
    )
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    teacher_cpu = _benchmark_teacher(
        iq,
        d0_config=d0_config,
        batch_sizes=arguments.batch_sizes,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )

    result = {
        "experiment_id": "D1-E001",
        "measurement_boundary": "already distance-compressed [B,3,130,32] complex IQ",
        "dataset": {
            "synthetic": True,
            "snr_db": 10.0,
            "target_presence_ratio": 1.0,
            "seed": 2026102410,
        },
        "protocol": {
            "warmup_calls_per_batch": arguments.warmup,
            "timed_calls_per_batch": arguments.repeats,
            "timing": "synchronous end-to-end wall time including student host/device transfer, forward, decode, and result transfer",
            "cpu_threads": 1,
            "note": "Student GPU and CPU/APX15 CPU figures use the same IQ boundary but are not a deployment speed comparison; APX15 timing covers only its FFT subchain adapter and D0 estimator.",
        },
        "environment": _environment(device),
        "methods": {
            "D_progressive_distillation_student": student_cpu,
            "D_progressive_distillation_student_cuda": student_cuda,
            "APX15_fft_subchain_cpu": teacher_cpu,
        },
    }
    output = arguments.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "gpu": result["environment"]["cuda_device"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
