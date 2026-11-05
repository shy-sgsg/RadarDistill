from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from radardistill.maintenance.d2a_r6g_cache_cleanup import cleanup_owned_r6g_cache


PAIRED_FILES = {
    "owner.json",
    "progress.json",
    "iq_h0.npy",
    "iq_h1.npy",
    "truth_range_cell.npy",
    "truth_doppler_hz.npy",
    "condition_index.npy",
    "scene_id.npy",
}
TARGET_FILES = {"owner.json", "progress.json", "k3.npy", "t3.npy"}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def cleanup_case(tmp_path: Path) -> SimpleNamespace:
    root = tmp_path / "project"
    stage = root / "experiments/D2-A-R6/R6-G"
    cache = stage / "cache/training"
    runs = stage / "runs/seed_707_v1.0.1"
    review = root / "docs/stage_reviews/D2-A-R6G_review.md"
    config = {"training": {"arm_order": ["T0", "T1", "T3"], "fixed_updates_per_arm": 25_600}}
    fake = SimpleNamespace(ROOT=root, STAGE=stage, CACHE=cache, RUNS=runs)
    fake._require_g0 = lambda: ({"status": "G0_PASS", "git_sha": "g0-sha"}, "g0-sha")
    fake._require_matching_g1 = lambda _g0: {"status": "G1_PASS"}
    fake._configs = lambda: (config, {}, {})

    def validate_pair(split: str) -> dict[str, object]:
        directory = cache / split
        owner = json.loads((directory / "owner.json").read_text(encoding="utf-8"))
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        if progress.get("owner") != owner or progress.get("rows_written") != 3:
            raise RuntimeError(f"invalid {split} cache owner/progress")
        if {path.name for path in directory.iterdir()} != PAIRED_FILES:
            raise RuntimeError(f"unexpected {split} cache contents")
        return owner

    def validate_targets(_config: object, _git_sha: str) -> dict[str, object]:
        directory = cache / "targets"
        owner = json.loads((directory / "owner.json").read_text(encoding="utf-8"))
        progress = json.loads((directory / "progress.json").read_text(encoding="utf-8"))
        if progress.get("owner") != owner or progress.get("completed_rows") != 3:
            raise RuntimeError("invalid target cache owner/progress")
        if {path.name for path in directory.iterdir()} != TARGET_FILES:
            raise RuntimeError("unexpected target cache contents")
        return owner

    fake._load_pair_cache = validate_pair
    fake._load_target_cache = validate_targets
    fake._r6g_training_owner = lambda arm, _config, _git_sha, _audit: {"arm": arm}
    fake._assert_cache_owner = lambda expected, actual, label: (
        None if dict(expected) == dict(actual) else (_ for _ in ()).throw(RuntimeError(f"{label} owner mismatch"))
    )

    for split in ("train", "validation"):
        directory = cache / split
        owner = {"split": split}
        _write_json(directory / "owner.json", owner)
        _write_json(directory / "progress.json", {"owner": owner, "rows_written": 3})
        for name in PAIRED_FILES - {"owner.json", "progress.json"}:
            (directory / name).write_bytes(b"cache")

    target_dir = cache / "targets"
    target_owner = {"target": "registered"}
    _write_json(target_dir / "owner.json", target_owner)
    _write_json(target_dir / "progress.json", {"owner": target_owner, "completed_rows": 3})
    for name in TARGET_FILES - {"owner.json", "progress.json"}:
        (target_dir / name).write_bytes(b"target")

    for arm in config["training"]["arm_order"]:
        run_dir = runs / arm
        _write_json(run_dir / "training_record.json", {
            "status": "complete",
            "arm": arm,
            "updates": config["training"]["fixed_updates_per_arm"],
            "owner": {"arm": arm},
            "sampler_audit": {"h1_plan_sha256": "h1", "h0_plan_sha256": "h0"},
        })

    _write_json(stage / "metrics.json", {
        "status": "development_evaluation_complete",
        "development_only": True,
        "final_blind_generated": False,
    })
    (stage / "analysis.md").write_text("completed analysis\n", encoding="utf-8")
    review.parent.mkdir(parents=True, exist_ok=True)
    review.write_text("completed stage report\n", encoding="utf-8")
    (stage / "scene_manifest_train.csv").write_text("scene_id\nscene-1\n", encoding="utf-8")
    (stage / "predictions").mkdir()
    (stage / "predictions/preserved.npz").write_bytes(b"prediction")
    user_file = root / "user-data.txt"
    user_file.write_text("keep", encoding="utf-8")
    return SimpleNamespace(fake=fake, root=root, stage=stage, cache=cache, review=review, user_file=user_file)


def test_cleanup_removes_only_complete_owned_caches_and_preserves_results(cleanup_case: SimpleNamespace) -> None:
    result = cleanup_owned_r6g_cache(training=cleanup_case.fake)

    assert result["status"] == "cleaned"
    assert {entry["directory"] for entry in result["removed"]} == {"train", "validation", "targets"}
    assert not (cleanup_case.cache / "train").exists()
    assert not (cleanup_case.cache / "validation").exists()
    assert not (cleanup_case.cache / "targets").exists()
    assert (cleanup_case.stage / "scene_manifest_train.csv").exists()
    assert (cleanup_case.stage / "predictions/preserved.npz").exists()
    assert cleanup_case.user_file.read_text(encoding="utf-8") == "keep"
    assert (cleanup_case.stage / "cleanup.json").is_file()


def test_cleanup_refuses_incomplete_training(cleanup_case: SimpleNamespace) -> None:
    record_path = cleanup_case.fake.RUNS / "T1/training_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["updates"] -= 1
    _write_json(record_path, record)

    with pytest.raises(RuntimeError, match="T1.*budget|T1.*complete"):
        cleanup_owned_r6g_cache(training=cleanup_case.fake)
    assert (cleanup_case.cache / "train").exists()


@pytest.mark.parametrize("missing", ["metrics", "analysis", "review"])
def test_cleanup_requires_completed_evaluation_and_stage_report(
    cleanup_case: SimpleNamespace, missing: str
) -> None:
    path = {
        "metrics": cleanup_case.stage / "metrics.json",
        "analysis": cleanup_case.stage / "analysis.md",
        "review": cleanup_case.review,
    }[missing]
    path.unlink()

    with pytest.raises(RuntimeError, match="metrics|evaluation|analysis|stage report"):
        cleanup_owned_r6g_cache(training=cleanup_case.fake)
    assert (cleanup_case.cache / "train").exists()


def test_cleanup_refuses_owner_or_progress_validation_failure(cleanup_case: SimpleNamespace) -> None:
    _write_json(cleanup_case.cache / "validation/progress.json", {"owner": {"wrong": True}, "rows_written": 3})

    with pytest.raises(RuntimeError, match="owner/progress"):
        cleanup_owned_r6g_cache(training=cleanup_case.fake)
    assert (cleanup_case.cache / "validation/owner.json").exists()


def test_cleanup_refuses_unexpected_file_and_preserves_it(cleanup_case: SimpleNamespace) -> None:
    extra = cleanup_case.cache / "targets/user-note.txt"
    extra.write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected|registered"):
        cleanup_owned_r6g_cache(training=cleanup_case.fake)
    assert extra.read_text(encoding="utf-8") == "keep"
    assert (cleanup_case.cache / "train").exists()


def test_cleanup_refuses_symlinked_cache_file(cleanup_case: SimpleNamespace) -> None:
    link = cleanup_case.cache / "targets/user-note.txt"
    link.symlink_to(cleanup_case.user_file)

    with pytest.raises(RuntimeError, match="symlink|unexpected"):
        cleanup_owned_r6g_cache(training=cleanup_case.fake)
    assert link.is_symlink()
    assert cleanup_case.user_file.read_text(encoding="utf-8") == "keep"


def test_cleanup_refuses_existing_audit_without_deleting_cache(cleanup_case: SimpleNamespace) -> None:
    _write_json(cleanup_case.stage / "cleanup.json", {"status": "previous audit"})

    with pytest.raises(RuntimeError, match="audit already exists"):
        cleanup_owned_r6g_cache(training=cleanup_case.fake)
    assert (cleanup_case.cache / "train").exists()
