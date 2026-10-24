"""APX15 slow-time FFT teacher adapter and D0 teacher outputs."""

from __future__ import annotations

import ctypes
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeacherBatch:
    teacher_response: np.ndarray
    teacher_detection_score: np.ndarray
    teacher_range_cell: np.ndarray
    teacher_doppler_hz: np.ndarray
    teacher_confidence: np.ndarray


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _compile_adapter(build_dir: Path) -> ctypes.CDLL:
    root = _repository_root()
    package = root / "GMTI_APX15_CoreX_Int16_20261014"
    source = package / "tools" / "radardistill_doppler_teacher.cpp"
    header = package / "include" / "transposeFFT.hpp"
    if not source.is_file() or not header.is_file():
        raise FileNotFoundError(
            "APX15 FFT adapter requires its committed wrapper and transposeFFT.hpp"
        )
    build_dir.mkdir(parents=True, exist_ok=True)
    library_path = build_dir / "libradardistill_d0_teacher.so"
    try:
        fftw_flags = subprocess.run(
            ["pkg-config", "--cflags", "--libs", "fftw3"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("D0 APX15 adapter needs g++ and FFTW3 development files") from exc

    command = [
        "g++",
        "-std=c++11",
        "-O3",
        "-fPIC",
        "-shared",
        "-I",
        str(package / "include"),
        str(source),
        "-o",
        str(library_path),
        *shlex.split(fftw_flags),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("D0 APX15 adapter needs g++ on PATH") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"failed to build APX15 FFT adapter: {detail}") from exc

    library = ctypes.CDLL(str(library_path))
    library.radardistill_d0_fft_create.argtypes = (ctypes.c_int, ctypes.c_int)
    library.radardistill_d0_fft_create.restype = ctypes.c_void_p
    library.radardistill_d0_fft_destroy.argtypes = (ctypes.c_void_p,)
    library.radardistill_d0_fft_destroy.restype = None
    library.radardistill_d0_fft_execute_many.argtypes = (
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    )
    library.radardistill_d0_fft_execute_many.restype = ctypes.c_int
    return library


class APX15DopplerTeacher:
    """Reuses APX15's centered slow-time FFT only; the teacher receives IQ alone."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        build_dir: str | Path | None = None,
    ) -> None:
        self.config = config
        self.prf_hz = float(config["radar"]["prf_hz"])
        self.pulse_count = int(config["radar"]["pulse_count"])
        self.channel_count = int(config["radar"]["channel_count"])
        self.range_count = int(config["radar"]["range_bin_count"])
        self.batch_size = int(config["execution"]["batch_size_cpu_teacher"])
        if self.channel_count != 3 or self.range_count != 32:
            raise ValueError("the frozen APX15 D0 teacher expects 3 channels and 32 range bins")
        if self.batch_size <= 0 or self.pulse_count % 2:
            raise ValueError("teacher batch size must be positive and the pulse count must be even")
        if build_dir is None:
            build_dir = Path(tempfile.gettempdir()) / "radardistill-d0-apx15-build"
        self._library = _compile_adapter(Path(build_dir))
        self._contexts: dict[int, int] = {}
        self._width = self.channel_count * self.range_count

    def close(self) -> None:
        for context in self._contexts.values():
            self._library.radardistill_d0_fft_destroy(ctypes.c_void_p(context))
        self._contexts.clear()

    def __del__(self) -> None:  # pragma: no cover - defensive native resource cleanup
        try:
            self.close()
        except Exception:
            pass

    def _context(self, nfft: int) -> int:
        if nfft in self._contexts:
            return self._contexts[nfft]
        context = self._library.radardistill_d0_fft_create(int(nfft), self._width)
        if not context:
            raise RuntimeError(f"APX15 ColFFTTranspose initialization failed for NFFT={nfft}")
        self._contexts[nfft] = int(context)
        return int(context)

    def transform(self, iq: np.ndarray, *, nfft: int) -> np.ndarray:
        """Apply the APX15-centered forward slow-time FFT to [B,C,N,R] IQ."""
        values = np.asarray(iq)
        expected_tail = (self.channel_count, self.pulse_count, self.range_count)
        if values.ndim != 4 or values.shape[1:] != expected_tail:
            raise ValueError(f"IQ must have shape [B,{expected_tail[0]},{expected_tail[1]},{expected_tail[2]}]")
        if values.shape[0] <= 0 or not np.iscomplexobj(values):
            raise ValueError("IQ must be a non-empty complex array")
        if not np.all(np.isfinite(values)):
            raise ValueError("IQ must contain only finite values")
        if nfft < self.pulse_count or nfft % 2:
            raise ValueError("NFFT must be even and at least the pulse count")

        transformed: list[np.ndarray] = []
        context = self._context(int(nfft))
        for start in range(0, values.shape[0], self.batch_size):
            chunk = np.asarray(values[start : start + self.batch_size], dtype=np.complex128)
            time_major = chunk.transpose(0, 2, 1, 3)
            padded = np.zeros(
                (chunk.shape[0], nfft, self.channel_count, self.range_count),
                dtype=np.complex128,
            )
            padded[:, : self.pulse_count] = time_major
            packed = np.ascontiguousarray(padded.reshape(chunk.shape[0], nfft, self._width))
            output = np.empty_like(packed)
            status = self._library.radardistill_d0_fft_execute_many(
                ctypes.c_void_p(context),
                int(chunk.shape[0]),
                ctypes.c_void_p(packed.ctypes.data),
                ctypes.c_void_p(output.ctypes.data),
            )
            if status != 1:
                raise RuntimeError("APX15 ColFFTTranspose execution failed")
            shaped = output.reshape(chunk.shape[0], nfft, self.channel_count, self.range_count)
            transformed.append(shaped.transpose(0, 2, 1, 3).copy())
        return np.concatenate(transformed, axis=0)

    @staticmethod
    def _parabolic_offset(left: float, center: float, right: float) -> float:
        denominator = left - 2.0 * center + right
        if denominator >= 0.0 or abs(denominator) < np.finfo(np.float64).tiny:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))

    def _predict_chunk(self, iq: np.ndarray) -> TeacherBatch:
        native_nfft = int(self.config["teacher"]["response"]["nfft"])
        fine_nfft = int(self.config["teacher"]["continuous_estimation"]["nfft"])
        native_fft = self.transform(iq, nfft=native_nfft)
        fine_fft = self.transform(iq, nfft=fine_nfft)
        native_power = np.sum(np.abs(native_fft) ** 2, axis=1, dtype=np.float64)
        fine_power = np.sum(np.abs(fine_fft) ** 2, axis=1, dtype=np.float64)

        native_total = native_power.sum(axis=(1, 2), keepdims=True)
        teacher_response = np.divide(
            native_power,
            native_total,
            out=np.zeros_like(native_power),
            where=native_total > 0.0,
        ).astype(np.float32)

        fine_total = fine_power.sum(axis=(1, 2))
        fine_peak = fine_power.max(axis=(1, 2))
        fine_median = np.median(fine_power, axis=(1, 2))
        detection_score = np.divide(
            fine_peak,
            fine_median,
            out=np.zeros_like(fine_peak),
            where=fine_median > 0.0,
        )
        confidence = np.divide(
            fine_peak,
            fine_total,
            out=np.zeros_like(fine_peak),
            where=fine_total > 0.0,
        )

        range_power = fine_power.sum(axis=1)
        coarse_range = np.argmax(range_power, axis=1)
        range_estimate = coarse_range.astype(np.float64)
        doppler_estimate = np.empty(iq.shape[0], dtype=np.float64)
        bin_hz = self.prf_hz / fine_nfft
        for sample, range_bin in enumerate(coarse_range):
            if 0 < range_bin < self.range_count - 1:
                left = float(np.sqrt(max(range_power[sample, range_bin - 1], 0.0)))
                right = float(np.sqrt(max(range_power[sample, range_bin + 1], 0.0)))
                denominator = left + right
                if denominator > 0.0:
                    delta_range = float(np.clip((right - left) / denominator, -0.5, 0.5))
                    range_estimate[sample] += delta_range

            spectrum = fine_power[sample, :, range_bin]
            peak_bin = int(np.argmax(spectrum))
            left_power = float(spectrum[(peak_bin - 1) % fine_nfft])
            center_power = float(spectrum[peak_bin])
            right_power = float(spectrum[(peak_bin + 1) % fine_nfft])
            delta_bin = self._parabolic_offset(left_power, center_power, right_power)
            doppler = (peak_bin - fine_nfft // 2 + delta_bin) * bin_hz
            doppler_estimate[sample] = (doppler + self.prf_hz / 2.0) % self.prf_hz - self.prf_hz / 2.0

        return TeacherBatch(
            teacher_response=teacher_response,
            teacher_detection_score=detection_score,
            teacher_range_cell=range_estimate,
            teacher_doppler_hz=doppler_estimate,
            teacher_confidence=confidence,
        )

    def predict(self, iq: np.ndarray) -> TeacherBatch:
        """Return teacher outputs from IQ only; truth/noise metadata are not accepted."""
        values = np.asarray(iq)
        if values.ndim != 4 or values.shape[1:] != (
            self.channel_count,
            self.pulse_count,
            self.range_count,
        ):
            raise ValueError("IQ shape does not match the frozen D0 channel/pulse/range dimensions")
        chunks = [
            self._predict_chunk(values[start : start + self.batch_size])
            for start in range(0, values.shape[0], self.batch_size)
        ]
        if not chunks:
            raise ValueError("teacher input batch must be non-empty")
        return TeacherBatch(
            teacher_response=np.concatenate([x.teacher_response for x in chunks], axis=0),
            teacher_detection_score=np.concatenate(
                [x.teacher_detection_score for x in chunks], axis=0
            ),
            teacher_range_cell=np.concatenate([x.teacher_range_cell for x in chunks], axis=0),
            teacher_doppler_hz=np.concatenate([x.teacher_doppler_hz for x in chunks], axis=0),
            teacher_confidence=np.concatenate([x.teacher_confidence for x in chunks], axis=0),
        )
