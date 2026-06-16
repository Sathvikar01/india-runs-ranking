"""LTR trainer entry point that can be run as a script (alias for src/training/train_ltr.py)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.training.train_ltr import main

if __name__ == "__main__":
    sys.exit(main())
