#!/usr/bin/env python3
"""Clean completed, owner-validated R6-G training and target caches."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from radardistill.maintenance.d2a_r6g_cache_cleanup import cleanup_owned_r6g_cache


def main() -> None:
    result = cleanup_owned_r6g_cache()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
