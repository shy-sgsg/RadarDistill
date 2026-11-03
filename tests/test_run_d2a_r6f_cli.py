from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_r6f_runner_help_lists_only_frozen_stage_commands():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_d2a_r6f.py"), "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0
    for command in ("preflight", "g1", "evaluate", "summarize", "all"):
        assert command in completed.stdout
