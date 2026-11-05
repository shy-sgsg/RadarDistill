"""Owner-scoped cleanup for the completed R6-G training caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PAIRED_CACHE_FILES = {
    "owner.json",
    "progress.json",
    "iq_h0.npy",
    "iq_h1.npy",
    "truth_range_cell.npy",
    "truth_doppler_hz.npy",
    "condition_index.npy",
    "scene_id.npy",
}
TARGET_CACHE_FILES = {"owner.json", "progress.json", "k3.npy", "t3.npy"}


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"R6-G cleanup requires a regular {label}: {path}")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    _require_regular_file(path, label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"R6-G {label} must contain a JSON object: {path}")
    return value


def _validated_directory(directory: Path, expected_files: set[str], label: str) -> tuple[int, dict[str, Any]]:
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"R6-G cleanup requires the registered {label} cache directory: {directory}")
    children = list(directory.iterdir())
    if any(child.is_symlink() or not child.is_file() for child in children):
        raise RuntimeError(f"R6-G {label} cache contains a symlink or nested directory: {directory}")
    names = {child.name for child in children}
    if names != expected_files:
        missing = sorted(expected_files - names)
        unexpected = sorted(names - expected_files)
        raise RuntimeError(
            f"R6-G {label} cache contents differ from its owner manifest "
            f"(missing={missing}, unexpected={unexpected}): {directory}"
        )
    owner = _read_json(directory / "owner.json", f"{label} cache owner")
    return sum(child.stat(follow_symlinks=False).st_size for child in children), owner


def _close_mmaps(arrays: Any) -> None:
    if not isinstance(arrays, dict):
        return
    for value in arrays.values():
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()


def _require_completed_evaluation(stage: Path, root: Path) -> dict[str, Any]:
    metrics = _read_json(stage / "metrics.json", "development metrics")
    if (
        metrics.get("status") != "development_evaluation_complete"
        or metrics.get("development_only") is not True
        or metrics.get("final_blind_generated") is not False
    ):
        raise RuntimeError("R6-G cleanup requires a completed development evaluation with no final blind")
    analysis_path = stage / "analysis.md"
    review_path = root / "docs/stage_reviews/D2-A-R6G_review.md"
    _require_regular_file(analysis_path, "analysis report")
    _require_regular_file(review_path, "internal stage report")
    if not analysis_path.stat().st_size or not review_path.stat().st_size:
        raise RuntimeError("R6-G cleanup requires nonempty analysis and internal stage reports")
    return metrics


def _require_completed_training(training: Any, config: dict[str, Any], git_sha: str) -> list[dict[str, Any]]:
    updates = int(config["training"]["fixed_updates_per_arm"])
    records: list[dict[str, Any]] = []
    for arm in config["training"]["arm_order"]:
        record_path = training.RUNS / str(arm) / "training_record.json"
        record = _read_json(record_path, f"{arm} training record")
        if record.get("status") != "complete" or int(record.get("updates", -1)) != updates:
            raise RuntimeError(f"R6-G {arm} training is incomplete or did not reach its frozen update budget")
        if record.get("arm") != arm:
            raise RuntimeError(f"R6-G {arm} training record names a different arm")
        expected_owner = training._r6g_training_owner(
            str(arm), config, git_sha, record.get("sampler_audit", {})
        )
        training._assert_cache_owner(expected_owner, record.get("owner", {}), f"R6-G {arm} training record")
        records.append(record)
    return records


def _remove_validated_directory(directory: Path, expected_files: set[str]) -> None:
    for name in sorted(expected_files):
        (directory / name).unlink()
    directory.rmdir()


def cleanup_owned_r6g_cache(*, training: Any | None = None) -> dict[str, Any]:
    """Delete only complete R6-G IQ/soft-map caches after evaluation and reporting.

    This operation never removes checkpoints, predictions, scene manifests, reports,
    or user files. It must be run from a clean committed ``main`` worktree after the
    registered T0/T1/T3 runs and development evaluation are complete.
    """
    if training is None:
        from radardistill.training import d2a_r6g as training_module

        training = training_module

    g0, git_sha = training._require_g0()
    training._require_matching_g1(g0)
    config, _, _ = training._configs()
    stage = Path(training.STAGE)
    root = Path(training.ROOT)
    cache = Path(training.CACHE)
    audit_path = stage / "cleanup.json"
    audit_temp = stage / "cleanup.json.tmp"
    if audit_path.exists() or audit_path.is_symlink():
        raise RuntimeError(f"R6-G cleanup audit already exists: {audit_path}")
    if audit_temp.exists() or audit_temp.is_symlink():
        raise RuntimeError(f"R6-G cleanup audit temporary file already exists: {audit_temp}")

    metrics = _require_completed_evaluation(stage, root)
    for directory in (cache.parent, cache):
        if directory.is_symlink() or not directory.is_dir():
            raise RuntimeError(f"R6-G cleanup refuses a missing or symlinked cache path: {directory}")
    expected_counts = {
        "train": PAIRED_CACHE_FILES,
        "validation": PAIRED_CACHE_FILES,
        "targets": TARGET_CACHE_FILES,
    }

    # The existing readers verify scene plans, owner records, complete progress,
    # array shapes/dtypes, and target dimensions before any file is removed.
    paired_owners: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation"):
        _, paired_owners[split] = _validated_directory(cache / split, expected_counts[split], split)
        arrays = training._load_pair_cache(split)
        _close_mmaps(arrays)
    _, target_owner = _validated_directory(cache / "targets", expected_counts["targets"], "target")
    target_arrays = training._load_target_cache(config, git_sha)
    _close_mmaps(target_arrays)
    records = _require_completed_training(training, config, git_sha)

    removed: list[dict[str, Any]] = []
    for name, expected_files in expected_counts.items():
        directory = cache / name
        byte_count, owner = _validated_directory(directory, expected_files, name)
        removed.append({"directory": name, "bytes": byte_count, "owner": owner})

    audit = {
        "status": "cleaned",
        "stage": "D2-A-R6-G",
        "g0_git_sha": git_sha,
        "protocol_version": metrics.get("protocol_version"),
        "removed": removed,
        "training_arms": [record["arm"] for record in records],
        "cache_owners": {**paired_owners, "targets": target_owner},
        "preserved": [
            "scene manifests",
            "training records and checkpoints",
            "predictions and metrics",
            "analysis and stage report",
            "all files outside the three registered cache directories",
        ],
        "user_untracked_files_touched": False,
    }

    # Prepare the audit before deletion so serialization or disk errors cannot
    # leave a successful cleanup with no record. The final name appears after all
    # three exact directories have been removed.
    audit_temp.parent.mkdir(parents=True, exist_ok=True)
    with audit_temp.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        for name, expected_files in expected_counts.items():
            _remove_validated_directory(cache / name, expected_files)
        os.replace(audit_temp, audit_path)
    except Exception as exc:
        audit["status"] = "partial_failure"
        audit["error"] = f"{type(exc).__name__}: {exc}"
        with audit_temp.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(audit_temp, audit_path)
        raise

    try:
        cache.rmdir()
    except OSError:
        # Preserve any unrelated sibling content if it exists under cache/.
        pass
    return audit
