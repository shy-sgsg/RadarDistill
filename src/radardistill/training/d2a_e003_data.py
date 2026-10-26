"""CPU-memmap scene materialization for the frozen D2-A-E003 splits."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from radardistill.simulation.d2a import generate_d2a_paired_batch
from radardistill.training.d2a_e002_data import (
    PAIRED_ARRAYS,
    materialize_paired_split,
)


H0_ARRAYS = (
    "iq_h0",
    "base_scene_id",
    "course_level",
    "scenario_id",
    "snr_db",
    "requested_cnr_db",
    "potential_target_amplitude",
    "noise_power",
)


def h0_mixture_assignments(
    *,
    count: int,
    course_levels: Mapping[str, Mapping[str, Any]],
    assignment_seed: int,
    dedicated_c3: bool = False,
) -> dict[str, np.ndarray]:
    """Return exact, shuffled E003 H0 course/nuisance composition metadata."""
    if count <= 0 or assignment_seed < 0:
        raise ValueError("H0 count must be positive and assignment seed nonnegative")
    if tuple(course_levels) != ("C0", "C1", "C2", "C3"):
        raise ValueError("course levels must be ordered C0-C3")
    rng = np.random.default_rng(assignment_seed)
    rows: list[tuple[str, str, float, float]] = []
    if dedicated_c3:
        conditions = [
            ("C3", scenario, snr, float(course_levels["C3"]["cnr_db"]))
            for snr in course_levels["C3"]["snr_db"]
            for scenario in ("separated", "near_clutter_edge", "partial_overlap")
        ]
        for index, row in enumerate(conditions):
            rows.extend([row] * (count // len(conditions) + (1 if index < count % len(conditions) else 0)))
    else:
        if count % 4:
            raise ValueError("mixed H0 count must allocate equally to C0-C3")
        per_course = count // 4
        rows.extend([
            ("C0", "separated", float(course_levels["C0"]["snr_db"][0]), float(course_levels["C0"]["cnr_db"]))
        ] * per_course)
        for scenario in ("separated", "near_clutter_edge"):
            rows.extend([("C1", scenario, float(course_levels["C1"]["snr_db"][0]), float(course_levels["C1"]["cnr_db"]))] * (per_course // 2))
        for scenario, amount in zip(
            ("separated", "near_clutter_edge", "partial_overlap"),
            (per_course // 3 + (1 if index < per_course % 3 else 0) for index in range(3)),
            strict=True,
        ):
            rows.extend([("C2", scenario, float(course_levels["C2"]["snr_db"][0]), float(course_levels["C2"]["cnr_db"]))] * amount)
        c3_snrs = [float(value) for value in course_levels["C3"]["snr_db"]]
        for snr_index, snr in enumerate(c3_snrs):
            per_snr = per_course // len(c3_snrs) + (
                1 if snr_index < per_course % len(c3_snrs) else 0
            )
            for scenario_index, scenario in enumerate(
                ("separated", "near_clutter_edge", "partial_overlap")
            ):
                amount = per_snr // 3 + (1 if scenario_index < per_snr % 3 else 0)
                rows.extend([("C3", scenario, snr, float(course_levels["C3"]["cnr_db"]))] * amount)
    if len(rows) != count:
        raise RuntimeError(f"H0 assignment produced {len(rows)} rows, expected {count}")
    permutation = rng.permutation(count)
    shuffled = [rows[int(index)] for index in permutation]
    return {
        "course_level": np.asarray([row[0] for row in shuffled], dtype="U2"),
        "scenario_id": np.asarray([row[1] for row in shuffled], dtype="U32"),
        "snr_db": np.asarray([row[2] for row in shuffled], dtype=np.float32),
        "requested_cnr_db": np.asarray([row[3] for row in shuffled], dtype=np.float32),
    }


def _load_h0_cache(cache_dir: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray] | None:
    manifest_path = cache_dir / "e003_manifest.json"
    if not manifest_path.exists():
        if cache_dir.exists() and any(cache_dir.iterdir()):
            raise FileExistsError(f"partial cache exists and will be preserved: {cache_dir}")
        return None
    cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if cached_manifest != manifest:
        raise ValueError(f"existing E003 H0 cache differs from the frozen plan: {cache_dir}")
    missing = [name for name in H0_ARRAYS if not (cache_dir / f"{name}.npy").is_file()]
    if missing:
        raise FileNotFoundError(f"E003 H0 cache is incomplete and will be preserved: {missing}")
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in H0_ARRAYS}


def materialize_e003_h0_split(
    cache_root: str | Path,
    split_name: str,
    *,
    count: int,
    course_levels: Mapping[str, Mapping[str, Any]],
    seed_base: int,
    assignment_seed: int,
    physics_config: dict[str, Any],
    chunk_size: int = 16,
    seed_stride: int = 1009,
    dedicated_c3: bool = False,
) -> dict[str, np.ndarray]:
    """Generate one independent H0 split, writing only to the E003 cache."""
    assignments = h0_mixture_assignments(
        count=count,
        course_levels=course_levels,
        assignment_seed=assignment_seed,
        dedicated_c3=dedicated_c3,
    )
    manifest = {
        "experiment": "D2-A-E003",
        "split": split_name,
        "scenes": int(count),
        "seed_base": int(seed_base),
        "seed_stride": int(seed_stride),
        "chunk_size": int(chunk_size),
        "assignment_seed": int(assignment_seed),
        "dedicated_c3": bool(dedicated_c3),
        "course_counts": {
            level: int(np.sum(assignments["course_level"] == level))
            for level in ("C0", "C1", "C2", "C3")
        },
        "scenario_snr_counts": {
            f"{scenario}_{snr:g}dB": int(np.sum(
                (assignments["scenario_id"] == scenario) & (assignments["snr_db"] == snr)
            ))
            for scenario in ("separated", "near_clutter_edge", "partial_overlap")
            for snr in np.unique(assignments["snr_db"])
            if np.any((assignments["scenario_id"] == scenario) & (assignments["snr_db"] == snr))
        },
        "iq_shape": [int(count), 3, 130, 32],
        "iq_dtype": "complex64",
    }
    cache_dir = Path(cache_root) / split_name
    cached = _load_h0_cache(cache_dir, manifest)
    if cached is not None:
        return cached
    cache_dir.mkdir(parents=True, exist_ok=True)
    shapes: dict[str, tuple[tuple[int, ...], Any]] = {
        "iq_h0": ((count, 3, 130, 32), np.complex64),
        "base_scene_id": ((count,), "U96"),
        "course_level": ((count,), "U2"),
        "scenario_id": ((count,), "U32"),
        "snr_db": ((count,), np.float32),
        "requested_cnr_db": ((count,), np.float32),
        "potential_target_amplitude": ((count,), np.float32),
        "noise_power": ((count,), np.float32),
    }
    maps = {
        name: np.lib.format.open_memmap(cache_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in shapes.items()
    }
    for chunk_index, start in enumerate(range(0, count, chunk_size)):
        stop = min(start + chunk_size, count)
        local = slice(start, stop)
        batch = generate_d2a_paired_batch(
            stop - start,
            int(seed_base) + chunk_index * int(seed_stride),
            assignments["snr_db"][local],
            assignments["requested_cnr_db"][local],
            # H0 is independent of the virtual target's geometry; scenario labels
            # remain audit strata and do not enter the detector.
            physics_config,
            scenario_id="separated",
        )
        maps["iq_h0"][local] = batch.iq_h0
        maps["base_scene_id"][local] = batch.base_scene_id.astype("U96")
        maps["course_level"][local] = assignments["course_level"][local]
        maps["scenario_id"][local] = assignments["scenario_id"][local]
        maps["snr_db"][local] = batch.snr_db.astype(np.float32)
        maps["requested_cnr_db"][local] = batch.requested_cnr_db.astype(np.float32)
        maps["potential_target_amplitude"][local] = batch.truth_target_amplitude.astype(np.float32)
        maps["noise_power"][local] = batch.noise_power.astype(np.float32)
    for array in maps.values():
        array.flush()
    (cache_dir / "e003_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in H0_ARRAYS}


def materialize_e003_paired_split(
    cache_root: str | Path,
    split_name: str,
    config: Mapping[str, Any],
    physics_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Reuse the audited E002 memmap writer with E003's frozen, disjoint plan."""
    data = config["data"]
    section = data[split_name]
    levels = config["scenarios"]
    if split_name == "train":
        per_course = int(section["pairs_per_course_level"])
        counts = section["cell_counts"]
        bases = section["seed_bases_by_course"]
    elif split_name == "validation":
        per_course = int(section["pairs_per_course_level"])
        counts = section["cell_counts"]
        bases = section["seed_bases_by_course"]
    elif split_name == "development":
        per_course = int(section["paired_base_scenes"]) // 4
        counts = section["cell_counts"]
        bases = section["seed_bases_by_course"]
    else:
        raise ValueError("paired split must be train, validation or development")
    expected_manifest = {
        "experiment": "D2-A-E003",
        "split": split_name,
        "per_course_count": per_course,
        "cell_counts": counts,
        "seed_bases_by_course": [int(seed) for seed in bases],
        "assignment_seed": int(section["assignment_seed"]),
        "chunk_size": int(section["generation_chunk_size"]),
        "seed_stride": int(data["scene_seed_stride"]),
        "physics_config": "experiments/D2-A-E001/config.yaml",
    }
    cache_dir = Path(cache_root) / split_name
    e003_manifest_path = cache_dir / "e003_manifest.json"
    if e003_manifest_path.exists():
        actual = json.loads(e003_manifest_path.read_text(encoding="utf-8"))
        if actual != expected_manifest:
            raise ValueError(f"existing E003 paired split differs from frozen protocol: {split_name}")
        missing = [name for name in PAIRED_ARRAYS if not (cache_dir / f"{name}.npy").is_file()]
        if missing:
            raise FileNotFoundError(f"E003 paired cache is incomplete and will be preserved: {missing}")
        return {name: np.load(cache_dir / f"{name}.npy", mmap_mode="r") for name in PAIRED_ARRAYS}
    arrays = materialize_paired_split(
        cache_root,
        split_name,
        physics_config,
        levels,
        counts,
        bases,
        assignment_seed=int(section["assignment_seed"]),
        chunk_size=int(section["generation_chunk_size"]),
        chunk_seed_stride=int(data["scene_seed_stride"]),
        include_k2=False,
    )
    (cache_dir / "e003_manifest.json").write_text(
        json.dumps(expected_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return arrays


def validate_e003_split_disjointness(splits: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, int]:
    """Require globally unique base scene IDs across all materialized splits."""
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}
    for name, arrays in splits.items():
        ids = np.asarray(arrays["base_scene_id"]).astype(str)
        if len(np.unique(ids)) != len(ids):
            raise ValueError(f"duplicate base_scene_id within E003 split {name}")
        for scene_id in ids:
            previous = seen.get(str(scene_id))
            if previous is not None:
                raise ValueError(f"E003 scene ID overlap between {previous} and {name}: {scene_id}")
            seen[str(scene_id)] = name
        counts[name] = len(ids)
    return counts

