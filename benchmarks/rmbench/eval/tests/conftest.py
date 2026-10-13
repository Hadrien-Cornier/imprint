from __future__ import annotations

import sys
from pathlib import Path


EVAL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(EVAL_ROOT / "src"))
