from __future__ import annotations

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_r6h_entrypoint_help_runs_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_d2a_r6h.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "generate-train" in result.stdout
    assert "train-all" in result.stdout
