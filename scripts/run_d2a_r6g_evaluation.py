#!/usr/bin/env python3
"""Capture preregistered R6-G validation checkpoints and run final development evaluation."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "src"))

from radardistill.evaluation.d2a_r6g import (
    ARMS,
    VALIDATION_POINTS,
    RUNS,
    _write_json,
    capture_validation_checkpoint,
    ensure_initial_validation_checkpoint,
    evaluate_r6g,
)


_EVENT = struct.Struct("iIII")
_IN_CREATE = 0x00000100
_IN_MOVED_TO = 0x00000080
_IN_CLOSE_WRITE = 0x00000008
_WATCH_MASK = _IN_CREATE | _IN_MOVED_TO | _IN_CLOSE_WRITE


def _add_watch(libc: Any, fd: int, path: Path, mask: int) -> int:
    watch = libc.inotify_add_watch(fd, os.fsencode(path), mask)
    if watch < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(path))
    return int(watch)


def existing_validation_updates(arm: str, run_dir: Path) -> set[int]:
    """Return registered snapshots already present in an arm's run directory."""
    if arm not in ARMS:
        raise ValueError(f"unknown R6-G arm: {arm}")
    snapshot_dir = Path(run_dir) / "validation_checkpoints"
    updates: set[int] = set()
    for update in VALIDATION_POINTS:
        path = snapshot_dir / f"update_{update:05d}.pt"
        if not path.is_file():
            continue
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        owner = checkpoint.get("owner", {})
        if (
            int(checkpoint.get("update", -1)) != update
            or owner.get("owner") != "D2-A-R6-G-training-run"
            or owner.get("arm") != arm
        ):
            raise RuntimeError(f"R6-G {arm} existing curve checkpoint owner/update mismatch")
        updates.add(update)
    return updates


def capture_curve_checkpoints() -> dict[str, Any]:
    """Use filesystem events to copy only update 0/12.8k/25.6k/51.2k checkpoints."""
    if not hasattr(ctypes.CDLL(None), "inotify_init1"):
        raise RuntimeError("R6-G checkpoint capture requires Linux inotify support")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    fd = libc.inotify_init1(os.O_CLOEXEC)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    captured: dict[str, set[int]] = {arm: set() for arm in ARMS}
    watches: dict[int, str] = {}
    audit_path = RUNS / "validation_checkpoint_capture.json"
    RUNS.mkdir(parents=True, exist_ok=True)
    root_watch = _add_watch(libc, fd, RUNS, _IN_CREATE | _IN_MOVED_TO)
    watches[root_watch] = "__root__"

    def inspect_arm(arm: str) -> None:
        run_dir = RUNS / arm
        captured[arm].update(existing_validation_updates(arm, run_dir))
        if ensure_initial_validation_checkpoint(arm):
            captured[arm].add(0)
        source = run_dir / "last_checkpoint.pt"
        if not source.is_file():
            return
        update = capture_validation_checkpoint(
            source, arm=arm, target_updates=VALIDATION_POINTS
        )
        if update is not None:
            captured[arm].add(update)
            report = {
                "stage": "D2-A-R6-G",
                "status": "capturing_validation_curve_checkpoints",
                "updates": list(VALIDATION_POINTS),
                "captured": {name: sorted(values) for name, values in captured.items()},
                "capture_dir": str((run_dir / "validation_checkpoints").relative_to(ROOT)),
                "update_00000_source": "shared_initial_state.pt before optimizer updates",
            }
            _write_json(audit_path, report)
            print(json.dumps({"arm": arm, "captured_update": update}, ensure_ascii=False), flush=True)

    def watch_arm(arm: str) -> None:
        if arm not in ARMS or arm in watches.values():
            return
        run_dir = RUNS / arm
        if not run_dir.is_dir():
            return
        wd = _add_watch(libc, fd, run_dir, _WATCH_MASK)
        watches[wd] = arm
        inspect_arm(arm)

    try:
        for arm in ARMS:
            watch_arm(arm)
        required = set(VALIDATION_POINTS)
        while any(captured[arm] != required for arm in ARMS):
            payload = os.read(fd, 65_536)
            offset = 0
            while offset + _EVENT.size <= len(payload):
                wd, _mask, _cookie, name_length = _EVENT.unpack_from(payload, offset)
                offset += _EVENT.size
                raw_name = payload[offset : offset + name_length].split(b"\0", 1)[0]
                offset += name_length
                name = os.fsdecode(raw_name)
                owner = watches.get(wd)
                if owner == "__root__" and name in ARMS:
                    watch_arm(name)
                elif owner in ARMS and name == "last_checkpoint.pt":
                    inspect_arm(owner)
        report = {
            "stage": "D2-A-R6-G",
            "status": "validation_curve_checkpoints_complete",
            "updates": list(VALIDATION_POINTS),
            "captured": {name: sorted(values) for name, values in captured.items()},
        }
        _write_json(audit_path, report)
        return report
    finally:
        os.close(fd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("capture-checkpoints", "evaluate"))
    args = parser.parse_args()
    if args.command == "capture-checkpoints":
        result = capture_curve_checkpoints()
    else:
        result = evaluate_r6g()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
