"""CPU-only, paired-scene caches for D2-A-E002."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from radardistill.simulation.d2a import compute_scene_power_metrics, generate_d2a_paired_batch
from radardistill.teachers.d2a_k2 import k2_response


PAIRED_ARRAYS = (
    "iq_h0",
    "iq_h1",
    "base_scene_id",
    "course_level",
    "scenario_id",
    "truth_range_cell",
    "truth_doppler_hz",
    "truth_radial_velocity_mps",
    "truth_azimuth_deg",
    "truth_phase_rad",
    "truth_amplitude",
    "snr_db",
    "requested_cnr_db",
    "noise_power",
    "measured_snr_db",
    "measured_cnr_db",
    "measured_scnr_db",
)
H0_ARRAYS = (
    "iq_h0",
    "base_scene_id",
    "course_level",
    "snr_db",
    "requested_cnr_db",
    "truth_amplitude_diagnostic",
    "noise_power_diagnostic",
)


def estimate_cache_bytes(e002_config: dict[str, Any]) -> dict[str, int]:
    """Estimate the local cache size from the exact preregistered array shapes."""
    data = e002_config["data"]
    sizes: dict[str, int] = {}

    def paired_size(count: int, include_k2: bool, include_k2_h0: bool = False) -> int:
        arrays = _paired_shapes(count, include_k2, include_k2_h0)
        return sum(int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize for shape, dtype in arrays.values())

    sizes["train"] = paired_size(
        int(data["train"]["paired_base_scenes"]),
        True,
    )
    sizes["validation"] = paired_size(
        int(data["validation"]["paired_base_scenes"]),
        False,
    )
    sizes["teacher_audit"] = paired_size(
        int(data["teacher_audit"]["paired_base_scenes"]),
        True,
        True,
    )
    sizes["development"] = paired_size(
        int(data["development"]["paired_base_scenes"]),
        False,
    )
    h0_per_scene = sum(
        int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
        for shape, dtype in {
        "iq_h0": ((3, 130, 32), np.complex64),
        "base_scene_id": ((), "U96"),
        "course_level": ((), "U2"),
        "snr_db": ((), np.float32),
        "requested_cnr_db": ((), np.float32),
        "truth_amplitude_diagnostic": ((), np.float32),
        "noise_power_diagnostic": ((), np.float32),
        }.values()
    )
    sizes["h0_calibration"] = h0_per_scene * int(data["h0_calibration"]["scenes"])
    sizes["h0_verification"] = h0_per_scene * int(data["h0_verification"]["scenes"])
    boundary_count = int(data["identifiability_boundary"]["paired_base_scenes"])
    sizes["identifiability_boundary"] = boundary_count * (
        2 * 3 * 130 * 32 * np.dtype(np.complex64).itemsize
        + np.dtype("U96").itemsize
        + 2 * np.dtype(np.float32).itemsize
    )
    sizes["total"] = sum(sizes.values())
    return sizes


def check_cache_disk_space(
    cache_root: str | Path,
    e002_config: dict[str, Any],
    *,
    reserve_gib: float = 3.0,
) -> dict[str, int | float]:
    """Fail before generation unless all caches fit with a working-space reserve."""
    if reserve_gib < 0.0:
        raise ValueError("reserve_gib must be nonnegative")
    probe = Path(cache_root).resolve()
    while not probe.exists():
        if probe.parent == probe:
            raise FileNotFoundError("no existing parent directory for E002 cache root")
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    required = estimate_cache_bytes(e002_config)["total"]
    data = e002_config["data"]
    expected_files: dict[str, tuple[str, ...]] = {
        "train": tuple(_paired_shapes(data["train"]["paired_base_scenes"], True, False)),
        "validation": tuple(_paired_shapes(data["validation"]["paired_base_scenes"], False, False)),
        "teacher_audit": tuple(
            _paired_shapes(data["teacher_audit"]["paired_base_scenes"], True, True)
        ),
        "development": tuple(_paired_shapes(data["development"]["paired_base_scenes"], False, False)),
        "h0_calibration": H0_ARRAYS,
        "h0_verification": H0_ARRAYS,
        "identifiability_boundary": (
            "iq_h0", "iq_h1", "base_scene_id", "truth_range_cell", "truth_doppler_hz"
        ),
    }
    cache_complete = all(
        (Path(cache_root) / split / "manifest.json").is_file()
        and all((Path(cache_root) / split / f"{name}.npy").is_file() for name in names)
        for split, names in expected_files.items()
    )
    remaining = 0 if cache_complete else required
    effective_reserve_gib = min(reserve_gib, 1.0) if cache_complete else reserve_gib
    reserve_bytes = int(effective_reserve_gib * (1024**3))
    if usage.free < remaining + reserve_bytes:
        required_gib = (remaining + reserve_bytes) / 1024**3
        free_gib = usage.free / 1024**3
        raise RuntimeError(
            f"insufficient local disk for remaining E002 caches: need {required_gib:.2f} GiB "
            f"including reserve, have {free_gib:.2f} GiB"
        )
    return {
        "free_bytes_before_generation": int(usage.free),
        "cache_bytes_estimated": int(required),
        "cache_bytes_remaining": int(remaining),
        "cache_is_complete": bool(cache_complete),
        "reserve_gib": float(effective_reserve_gib),
        "cache_bytes_including_reserve": int(remaining + reserve_bytes),
    }


def _parse_cell(
    key: str,
    level_spec: Mapping[str, Any],
    physics_config: dict[str, Any],
) -> tuple[str, float, float]:
    """Resolve a preregistered cell key to scenario, SNR, and CNR."""
    if "scenario_snr_counts" in level_spec or key.endswith(("_0db", "_10db")):
        matching_snr: float | None = None
        scenario = key
        for snr in level_spec.get("snr_db", (0.0, 10.0)):
            suffix = f"_{int(float(snr))}db"
            if key.endswith(suffix):
                scenario = key[: -len(suffix)]
                matching_snr = float(snr)
                break
        if matching_snr is None:
            raise ValueError(f"cell {key} does not match a preregistered SNR in its course level")
        snr_db = matching_snr
    else:
        scenario = key
        snr_values = level_spec.get("snr_db")
        if isinstance(snr_values, Sequence) and not isinstance(snr_values, (str, bytes)):
            if len(snr_values) != 1:
                raise ValueError(f"single-SNR cell {key} has an ambiguous SNR list")
            snr_db = float(snr_values[0])
        else:
            snr_db = float(snr_values)
    if scenario not in physics_config["target"]["scenario_radial_velocity_mps"]:
        raise ValueError(f"unknown E001 target scenario in E002 cell: {scenario}")
    return scenario, snr_db, float(level_spec["cnr_db"])


def _cell_assignments(
    course_level: str,
    counts: Mapping[str, int],
    level_spec: Mapping[str, Any],
    physics_config: dict[str, Any],
    *,
    assignment_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[str, float, float]] = []
    for key, raw_count in counts.items():
        count = int(raw_count)
        if count < 0:
            raise ValueError("cell counts must be nonnegative")
        if count == 0:
            continue
        scenario, snr_db, cnr_db = _parse_cell(key, level_spec, physics_config)
        rows.extend([(scenario, snr_db, cnr_db)] * count)
    if not rows:
        raise ValueError(f"course level {course_level} has no assigned cells")
    rng = np.random.default_rng(int(assignment_seed))
    order = rng.permutation(len(rows))
    selected = [rows[int(index)] for index in order]
    scenarios = np.asarray([row[0] for row in selected], dtype="U32")
    snr = np.asarray([row[1] for row in selected], dtype=np.float64)
    cnr = np.asarray([row[2] for row in selected], dtype=np.float64)
    velocity_map = physics_config["target"]["scenario_radial_velocity_mps"]
    velocity = np.asarray([velocity_map[str(value)] for value in scenarios], dtype=np.float64)
    return scenarios, snr, cnr, velocity


def _paired_shapes(
    count: int,
    include_k2: bool,
    include_k2_h0: bool = False,
) -> dict[str, tuple[tuple[int, ...], Any]]:
    shapes: dict[str, tuple[tuple[int, ...], Any]] = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "iq_h1": ((count, 3, 130, 32), np.complex64),
        "base_scene_id": ((count,), "U96"),
        "course_level": ((count,), "U2"),
        "scenario_id": ((count,), "U32"),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
        "truth_radial_velocity_mps": ((count,), np.float32),
        "truth_azimuth_deg": ((count,), np.float32),
        "truth_phase_rad": ((count,), np.float32),
        "truth_amplitude": ((count,), np.float32),
        "snr_db": ((count,), np.float32),
        "requested_cnr_db": ((count,), np.float32),
        "noise_power": ((count,), np.float32),
        "measured_snr_db": ((count,), np.float32),
        "measured_cnr_db": ((count,), np.float32),
        "measured_scnr_db": ((count,), np.float32),
    }
    if include_k2:
        shapes["teacher_response"] = ((count, 130, 32), np.float32)
    if include_k2_h0:
        shapes["teacher_response_h0"] = ((count, 130, 32), np.float32)
    return shapes


def _load_complete_cache(
    cache_dir: Path,
    expected_manifest: dict[str, Any],
    names: Sequence[str],
) -> dict[str, np.ndarray] | None:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != expected_manifest:
        raise ValueError(f"existing E002 cache manifest differs from the frozen plan: {cache_dir}")
    missing = [name for name in names if not (cache_dir / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"existing E002 cache is incomplete; preserving it: {missing}")
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in names}


def materialize_paired_split(
    cache_root: str | Path,
    split_name: str,
    physics_config: dict[str, Any],
    course_levels: Mapping[str, Mapping[str, Any]],
    cell_counts: Mapping[str, Mapping[str, int]],
    seed_bases_by_course: Sequence[int],
    *,
    assignment_seed: int,
    chunk_size: int = 16,
    chunk_seed_stride: int = 1009,
    include_k2: bool = False,
    include_k2_h0: bool = False,
) -> dict[str, np.ndarray]:
    """Stream one paired H0/H1 split into CPU memmaps in course order."""
    if tuple(cell_counts) != ("C0", "C1", "C2", "C3"):
        raise ValueError("cell_counts must be ordered C0, C1, C2, C3")
    if tuple(course_levels) != ("C0", "C1", "C2", "C3"):
        raise ValueError("course_levels must be ordered C0, C1, C2, C3")
    if len(seed_bases_by_course) != 4 or len(set(int(x) for x in seed_bases_by_course)) != 4:
        raise ValueError("one unique seed base is required for each of C0-C3")
    if chunk_size <= 0 or chunk_seed_stride <= 0 or assignment_seed < 0:
        raise ValueError("chunk and assignment seed settings must be valid")
    counts_by_course = {
        level: sum(int(value) for value in counts.values())
        for level, counts in cell_counts.items()
    }
    if any(count <= 0 for count in counts_by_course.values()):
        raise ValueError("each E002 course level must contain scenes")
    total = sum(counts_by_course.values())
    cache_dir = Path(cache_root) / split_name
    array_names = (
        *PAIRED_ARRAYS,
        *(('teacher_response',) if include_k2 else ()),
        *(('teacher_response_h0',) if include_k2_h0 else ()),
    )
    expected_manifest = {
        "split": split_name,
        "total_paired_scenes": total,
        "cell_counts": {level: {str(k): int(v) for k, v in counts.items()} for level, counts in cell_counts.items()},
        "course_conditions": {
            level: {
                "snr_db": [float(value) for value in course_levels[level]["snr_db"]],
                "cnr_db": float(course_levels[level]["cnr_db"]),
            }
            for level in course_levels
        },
        "seed_bases_by_course": [int(value) for value in seed_bases_by_course],
        "assignment_seed": int(assignment_seed),
        "chunk_size": int(chunk_size),
        "chunk_seed_stride": int(chunk_seed_stride),
        "k2_cached": bool(include_k2),
        "k2_h0_cached": bool(include_k2_h0),
        "iq_shape": [total, 3, 130, 32],
        "iq_dtype": "complex64",
        "physics_config": "experiments/D2-A-E001/config.yaml",
    }
    if cache_dir.exists():
        cached = _load_complete_cache(cache_dir, expected_manifest, array_names)
        if cached is not None:
            return cached
        if any(cache_dir.iterdir()):
            raise FileExistsError(f"partial E002 cache exists; preserving it: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    shapes = _paired_shapes(total, include_k2, include_k2_h0)
    maps = {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in shapes.items()
    }
    cursor = 0
    for course_index, level in enumerate(("C0", "C1", "C2", "C3")):
        scenarios, snr, cnr, velocity = _cell_assignments(
            level,
            cell_counts[level],
            course_levels[level],
            physics_config,
            assignment_seed=assignment_seed + course_index,
        )
        for chunk_index, start in enumerate(range(0, scenarios.size, chunk_size)):
            stop = min(start + chunk_size, scenarios.size)
            size = stop - start
            seed = int(seed_bases_by_course[course_index]) + chunk_index * chunk_seed_stride
            batch = generate_d2a_paired_batch(
                size,
                seed,
                snr[start:stop],
                cnr[start:stop],
                physics_config,
                scenario_id="separated",
                target_radial_velocity_mps=velocity[start:stop],
            )
            target = batch.target_signal_iq
            powers = compute_scene_power_metrics(target, batch.clutter_iq, batch.noise_iq)
            chunk_slice = slice(cursor, cursor + size)
            values: dict[str, np.ndarray] = {
                "iq_h0": batch.iq_h0,
                "iq_h1": batch.iq_h1,
                "base_scene_id": batch.base_scene_id.astype("U96"),
                "course_level": np.full(size, level, dtype="U2"),
                "scenario_id": scenarios[start:stop],
                "truth_range_cell": batch.truth_target_range_cell.astype(np.float32),
                "truth_doppler_hz": batch.truth_target_doppler_hz.astype(np.float32),
                "truth_radial_velocity_mps": batch.truth_target_radial_velocity_mps.astype(np.float32),
                "truth_azimuth_deg": batch.truth_target_azimuth_deg.astype(np.float32),
                "truth_phase_rad": batch.truth_target_phase_rad.astype(np.float32),
                "truth_amplitude": batch.truth_target_amplitude.astype(np.float32),
                "snr_db": batch.snr_db.astype(np.float32),
                "requested_cnr_db": batch.requested_cnr_db.astype(np.float32),
                "noise_power": batch.noise_power.astype(np.float32),
                "measured_snr_db": np.asarray(powers["snr_db"], dtype=np.float32),
                "measured_cnr_db": np.asarray(powers["cnr_db"], dtype=np.float32),
                "measured_scnr_db": np.asarray(powers["scnr_db"], dtype=np.float32),
            }
            if include_k2:
                values["teacher_response"] = k2_response(batch.iq_h1, physics_config)
            if include_k2_h0:
                values["teacher_response_h0"] = k2_response(batch.iq_h0, physics_config)
            for name, value in values.items():
                maps[name][chunk_slice] = value
            cursor += size
        if cursor != sum(counts_by_course[name] for name in ("C0", "C1", "C2", "C3")[: course_index + 1]):
            raise RuntimeError("E002 cache writer produced an unexpected course offset")

    if cursor != total:
        raise RuntimeError(f"E002 cache wrote {cursor} scenes but expected {total}")
    for array in maps.values():
        array.flush()
    (cache_dir / "manifest.json").write_text(
        json.dumps(expected_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in array_names}


def materialize_h0_mixture(
    cache_root: str | Path,
    split_name: str,
    count: int,
    physics_config: dict[str, Any],
    course_levels: Mapping[str, Mapping[str, Any]],
    *,
    seed_base: int,
    mixture_assignment_seed: int,
    chunk_size: int = 16,
    chunk_seed_stride: int = 1009,
) -> dict[str, np.ndarray]:
    """Generate independent H0 scenes from the frozen C0-C3 IID mixture."""
    if count <= 0 or seed_base < 0 or mixture_assignment_seed < 0:
        raise ValueError("H0 mixture count and seeds must be valid")
    if tuple(course_levels) != ("C0", "C1", "C2", "C3"):
        raise ValueError("H0 mixture course levels must be ordered C0-C3")
    if chunk_size <= 0 or chunk_seed_stride <= 0:
        raise ValueError("H0 chunk settings must be positive")
    cache_dir = Path(cache_root) / split_name
    expected = {
        "split": split_name,
        "scenes": int(count),
        "iid_equal_course_level_mixture": True,
        "c3_snr_equal_mixture": [0.0, 10.0],
        "seed_base": int(seed_base),
        "mixture_assignment_seed": int(mixture_assignment_seed),
        "chunk_size": int(chunk_size),
        "chunk_seed_stride": int(chunk_seed_stride),
        "iq_shape": [count, 3, 130, 32],
        "iq_dtype": "complex64",
    }
    if cache_dir.exists():
        cached = _load_complete_cache(cache_dir, expected, H0_ARRAYS)
        if cached is not None:
            return cached
        if any(cache_dir.iterdir()):
            raise FileExistsError(f"partial E002 H0 cache exists; preserving it: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "base_scene_id": ((count,), "U96"),
        "course_level": ((count,), "U2"),
        "snr_db": ((count,), np.float32),
        "requested_cnr_db": ((count,), np.float32),
        "truth_amplitude_diagnostic": ((count,), np.float32),
        "noise_power_diagnostic": ((count,), np.float32),
    }
    maps = {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in shapes.items()
    }
    rng = np.random.default_rng(mixture_assignment_seed)
    selected_levels = rng.choice(np.asarray(("C0", "C1", "C2", "C3")), size=count)
    snr = np.empty(count, dtype=np.float64)
    cnr = np.empty(count, dtype=np.float64)
    for level in ("C0", "C1", "C2", "C3"):
        indices = np.flatnonzero(selected_levels == level)
        level_spec = course_levels[level]
        level_snrs = [float(item) for item in level_spec["snr_db"]]
        if level == "C3":
            snr[indices] = rng.choice(np.asarray(level_snrs), size=indices.size)
        else:
            snr[indices] = level_snrs[0]
        cnr[indices] = float(level_spec["cnr_db"])
    cursor = 0
    for chunk_index, start in enumerate(range(0, count, chunk_size)):
        stop = min(start + chunk_size, count)
        size = stop - start
        seed = int(seed_base) + chunk_index * chunk_seed_stride
        batch = generate_d2a_paired_batch(
            size,
            seed,
            snr[start:stop],
            cnr[start:stop],
            physics_config,
            scenario_id="separated",
        )
        selected = slice(cursor, cursor + size)
        maps["iq_h0"][selected] = batch.iq_h0
        maps["base_scene_id"][selected] = batch.base_scene_id.astype("U96")
        maps["course_level"][selected] = selected_levels[start:stop]
        maps["snr_db"][selected] = batch.snr_db.astype(np.float32)
        maps["requested_cnr_db"][selected] = batch.requested_cnr_db.astype(np.float32)
        maps["truth_amplitude_diagnostic"][selected] = batch.truth_target_amplitude.astype(np.float32)
        maps["noise_power_diagnostic"][selected] = batch.noise_power.astype(np.float32)
        cursor += size
    for array in maps.values():
        array.flush()
    (cache_dir / "manifest.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in H0_ARRAYS}


def materialize_identifiability_boundary(
    cache_root: str | Path,
    split_name: str,
    count: int,
    physics_config: dict[str, Any],
    *,
    snr_db: float,
    cnr_db: float,
    seed_base: int,
    chunk_size: int = 16,
    chunk_seed_stride: int = 1009,
) -> dict[str, np.ndarray]:
    """Cache the exact-overlap D condition for descriptive boundary analysis."""
    if count <= 0 or seed_base < 0 or chunk_size <= 0 or chunk_seed_stride <= 0:
        raise ValueError("boundary count, seeds, and chunk settings must be valid")
    cache_dir = Path(cache_root) / split_name
    expected = {
        "split": split_name,
        "scenes": int(count),
        "scenario": "exact_overlap",
        "snr_db": float(snr_db),
        "cnr_db": float(cnr_db),
        "seed_base": int(seed_base),
        "chunk_size": int(chunk_size),
        "chunk_seed_stride": int(chunk_seed_stride),
    }
    names = ("iq_h0", "iq_h1", "base_scene_id", "truth_range_cell", "truth_doppler_hz")
    if cache_dir.exists():
        cached = _load_complete_cache(cache_dir, expected, names)
        if cached is not None:
            return cached
        if any(cache_dir.iterdir()):
            raise FileExistsError(f"partial E002 boundary cache exists; preserving it: {cache_dir}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "iq_h1": ((count, 3, 130, 32), np.complex64),
        "base_scene_id": ((count,), "U96"),
        "truth_range_cell": ((count,), np.float32),
        "truth_doppler_hz": ((count,), np.float32),
    }
    maps = {
        name: np.lib.format.open_memmap(
            cache_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
        )
        for name, (shape, dtype) in shapes.items()
    }
    for chunk_index, start in enumerate(range(0, count, chunk_size)):
        stop = min(start + chunk_size, count)
        size = stop - start
        batch = generate_d2a_paired_batch(
            size,
            seed_base + chunk_index * chunk_seed_stride,
            float(snr_db),
            float(cnr_db),
            physics_config,
            scenario_id="exact_overlap",
        )
        selected = slice(start, stop)
        maps["iq_h0"][selected] = batch.iq_h0
        maps["iq_h1"][selected] = batch.iq_h1
        maps["base_scene_id"][selected] = batch.base_scene_id.astype("U96")
        maps["truth_range_cell"][selected] = batch.truth_target_range_cell.astype(np.float32)
        maps["truth_doppler_hz"][selected] = batch.truth_target_doppler_hz.astype(np.float32)
    for array in maps.values():
        array.flush()
    (cache_dir / "manifest.json").write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in names}


def materialize_e002_splits(
    cache_root: str | Path,
    e002_config: dict[str, Any],
    physics_config: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Build every preregistered split using CPU memmaps and disjoint seeds."""
    from .d2a_e002 import validate_e002_config

    validate_e002_config(e002_config)
    check_cache_disk_space(cache_root, e002_config)
    data = e002_config["data"]
    levels = data["course_levels"]
    stride = int(data["chunk_seed_stride"])
    paired: dict[str, dict[str, np.ndarray]] = {}

    train_counts = {
        level: section.get("scenario_counts", section.get("scenario_snr_counts"))
        for level, section in levels.items()
    }
    train = data["train"]
    paired["train"] = materialize_paired_split(
        cache_root,
        "train",
        physics_config,
        levels,
        train_counts,
        train["seed_bases_by_course"],
        assignment_seed=int(train["assignment_seed"]),
        chunk_size=int(train["generation_chunk_size"]),
        chunk_seed_stride=stride,
        include_k2=True,
    )
    for name in ("validation", "teacher_audit"):
        section = data[name]
        paired[name] = materialize_paired_split(
            cache_root,
            name,
            physics_config,
            levels,
            section["cell_counts"],
            section["seed_bases_by_course"],
            assignment_seed=int(section["assignment_seed"]),
            chunk_size=int(section["generation_chunk_size"]),
            chunk_seed_stride=stride,
            include_k2=(name == "teacher_audit"),
            include_k2_h0=(name == "teacher_audit"),
        )
    for name in ("h0_calibration", "h0_verification"):
        section = data[name]
        paired[name] = materialize_h0_mixture(
            cache_root,
            name,
            int(section["scenes"]),
            physics_config,
            levels,
            seed_base=int(section["seed_base"]),
            mixture_assignment_seed=int(section["mixture_assignment_seed"]),
            chunk_size=int(section["generation_chunk_size"]),
            chunk_seed_stride=int(section["chunk_seed_stride"]),
        )
    development = data["development"]
    paired["development"] = materialize_paired_split(
        cache_root,
        "development",
        physics_config,
        levels,
        data["validation"]["cell_counts"],
        development["seed_bases_by_course"],
        assignment_seed=int(development["assignment_seed"]),
        chunk_size=int(development["generation_chunk_size"]),
        chunk_seed_stride=stride,
        include_k2=False,
    )
    boundary = data["identifiability_boundary"]
    paired["identifiability_boundary"] = materialize_identifiability_boundary(
        cache_root,
        "identifiability_boundary",
        int(boundary["paired_base_scenes"]),
        physics_config,
        snr_db=float(boundary["snr_db"]),
        cnr_db=float(boundary["cnr_db"]),
        seed_base=int(boundary["seed_base"]),
        chunk_size=int(boundary["generation_chunk_size"]),
        chunk_seed_stride=int(boundary["chunk_seed_stride"]),
    )
    validate_disjoint_scene_ids(paired)
    return paired


def validate_disjoint_scene_ids(splits: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    """Ensure no base scene is shared across any cached experimental split."""
    seen: dict[str, str] = {}
    for split_name, arrays in splits.items():
        ids = np.asarray(arrays["base_scene_id"]).astype(str)
        if np.unique(ids).size != ids.size:
            raise ValueError(f"base_scene_id repeats within split {split_name}")
        for scene_id in ids:
            previous = seen.get(str(scene_id))
            if previous is not None:
                raise ValueError(f"base_scene_id overlaps between {previous} and {split_name}")
            seen[str(scene_id)] = split_name
