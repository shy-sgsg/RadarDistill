from __future__ import annotations

import json
from pathlib import Path

import pytest

from radardistill.training import d2a_r6h


CACHE_FILES = {
    "owner.json",
    "progress.json",
    "iq_h0.npy",
    "iq_h1.npy",
    "truth_range_cell.npy",
    "truth_doppler_hz.npy",
    "condition_index.npy",
    "scene_id.npy",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cleanup_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    root = tmp_path / "project"
    stage = root / "experiments/D2-A-R6/R6-H"
    cache = stage / "cache/training"
    artifacts = stage / "runs/seed_808"
    monkeypatch.setattr(d2a_r6h, "ROOT", root)
    monkeypatch.setattr(d2a_r6h, "STAGE", stage)
    monkeypatch.setattr(d2a_r6h, "CACHE", cache)
    monkeypatch.setattr(d2a_r6h, "ARTIFACTS", artifacts)
    monkeypatch.setattr(
        d2a_r6h,
        "_require_preflight",
        lambda *, require_cuda: ({"git_sha": "test-sha"}, "test-sha"),
    )
    monkeypatch.setattr(d2a_r6h, "r6h_configs", lambda: ({}, {}, {}))
    plans = {"train": [{"scene": "train"}], "validation": [{"scene": "validation"}]}
    monkeypatch.setattr(d2a_r6h, "build_r6h_split_plans", lambda _config, _conditions: plans)
    monkeypatch.setattr(
        d2a_r6h,
        "_cache_owner",
        lambda split, _plan, _git_sha: {"split": split, "scene_count": 1},
    )
    _write_json(
        artifacts / "training_summary.json",
        {
            "status": "complete",
            "arms": {arm: {"status": "complete"} for arm in d2a_r6h.R6H_ARMS},
        },
    )
    _write_json(stage / "metrics.json", {"status": "evaluated", "final_blind_generated": False})
    (stage / "analysis.md").write_text("R6-H analysis", encoding="utf-8")
    report = root / "docs/stage_reviews/D2-A-R6H_review.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("R6-H stage report", encoding="utf-8")
    cache.mkdir(parents=True)
    return {"root": root, "stage": stage, "cache": cache, "report": report}


def _write_complete_cache(directory: Path, split: str) -> dict[str, object]:
    owner = {"split": split, "scene_count": 1}
    _write_json(directory / "owner.json", owner)
    _write_json(directory / "progress.json", {"owner": owner, "rows_written": 1})
    for name in CACHE_FILES - {"owner.json", "progress.json"}:
        (directory / name).write_bytes(b"cache")
    return owner


@pytest.mark.parametrize("missing", ["metrics", "analysis", "report"])
def test_cleanup_requires_evaluation_and_stage_report_before_deleting(
    cleanup_case: dict[str, Path], missing: str
) -> None:
    paths = {
        "metrics": cleanup_case["stage"] / "metrics.json",
        "analysis": cleanup_case["stage"] / "analysis.md",
        "report": cleanup_case["report"],
    }
    paths[missing].unlink()
    train_cache = cleanup_case["cache"] / "train"
    _write_complete_cache(train_cache, "train")

    with pytest.raises(RuntimeError, match="evaluation|metrics|analysis|stage report"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert train_cache.exists()


def test_cleanup_persists_audit_and_preserves_archived_results(
    cleanup_case: dict[str, Path]
) -> None:
    train_cache = cleanup_case["cache"] / "train"
    validation_cache = cleanup_case["cache"] / "validation"
    _write_complete_cache(train_cache, "train")
    _write_complete_cache(validation_cache, "validation")

    result = d2a_r6h.cleanup_owned_r6h_training_cache()

    audit_path = cleanup_case["stage"] / "cleanup.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert result["status"] == audit["status"] == "cleaned"
    assert {entry["split"] for entry in audit["removed"]} == {"train", "validation"}
    assert all(entry["bytes"] > 0 and entry["owner"]["split"] == entry["split"] for entry in audit["removed"])
    assert audit["user_untracked_files_touched"] is False
    assert (cleanup_case["stage"] / "metrics.json").is_file()
    assert (cleanup_case["stage"] / "analysis.md").is_file()
    assert cleanup_case["report"].is_file()
    assert not train_cache.exists()
    assert not validation_cache.exists()


def test_cleanup_refuses_to_overwrite_existing_audit(
    cleanup_case: dict[str, Path]
) -> None:
    train_cache = cleanup_case["cache"] / "train"
    _write_complete_cache(train_cache, "train")
    audit_path = cleanup_case["stage"] / "cleanup.json"
    _write_json(audit_path, {"status": "previous cleanup"})

    with pytest.raises(RuntimeError, match="audit already exists"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert train_cache.exists()


def test_interrupted_cleanup_temp_audit_does_not_claim_success(
    cleanup_case: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    train_cache = cleanup_case["cache"] / "train"
    _write_complete_cache(train_cache, "train")
    original_unlink = Path.unlink

    def interrupt_before_removal(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == train_cache:
            raise KeyboardInterrupt("simulated interruption")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_before_removal)

    with pytest.raises(KeyboardInterrupt, match="simulated interruption"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    temp_audit = cleanup_case["stage"] / "cleanup.json.tmp"
    assert json.loads(temp_audit.read_text(encoding="utf-8"))["status"] == "in_progress"
    assert train_cache.exists()


def test_cleanup_refuses_symlinked_cache_directory_and_preserves_target(
    cleanup_case: dict[str, Path], tmp_path: Path
) -> None:
    external_cache = tmp_path / "user-data-cache"
    _write_complete_cache(external_cache, "train")
    for name in CACHE_FILES:
        assert (external_cache / name).is_file()
    (cleanup_case["cache"] / "train").symlink_to(external_cache, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert all((external_cache / name).is_file() for name in CACHE_FILES)


def test_cleanup_refuses_symlinked_cache_ancestor_and_preserves_target(
    cleanup_case: dict[str, Path], tmp_path: Path
) -> None:
    cache = cleanup_case["cache"]
    external_parent = tmp_path / "external-cache-root"
    external_cache = external_parent / "training/train"
    _write_complete_cache(external_cache, "train")
    cache.rmdir()
    cache.parent.rmdir()
    cache.parent.symlink_to(external_parent, target_is_directory=True)

    with pytest.raises(RuntimeError, match="symlink"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert all((external_cache / name).is_file() for name in CACHE_FILES)


def test_cleanup_refuses_symlinked_owner_file_and_preserves_cache(
    cleanup_case: dict[str, Path], tmp_path: Path
) -> None:
    directory = cleanup_case["cache"] / "train"
    owner = _write_complete_cache(directory, "train")
    outside_owner = tmp_path / "user-owner.json"
    _write_json(outside_owner, owner)
    (directory / "owner.json").unlink()
    (directory / "owner.json").symlink_to(outside_owner)

    with pytest.raises(RuntimeError, match="symlink"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert outside_owner.is_file()
    assert (directory / "iq_h0.npy").is_file()


def test_cleanup_validates_all_caches_before_deleting_any(
    cleanup_case: dict[str, Path], tmp_path: Path
) -> None:
    train_cache = cleanup_case["cache"] / "train"
    _write_complete_cache(train_cache, "train")
    external_validation = tmp_path / "external-validation-cache"
    _write_complete_cache(external_validation, "validation")
    (cleanup_case["cache"] / "validation").symlink_to(
        external_validation, target_is_directory=True
    )

    with pytest.raises(RuntimeError, match="symlink"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert all((train_cache / name).is_file() for name in CACHE_FILES)
    assert all((external_validation / name).is_file() for name in CACHE_FILES)


def test_cleanup_refuses_incomplete_cache_file_set(
    cleanup_case: dict[str, Path]
) -> None:
    directory = cleanup_case["cache"] / "train"
    _write_complete_cache(directory, "train")
    (directory / "iq_h0.npy").unlink()

    with pytest.raises(RuntimeError, match="cache contents"):
        d2a_r6h.cleanup_owned_r6h_training_cache()

    assert (directory / "owner.json").is_file()
