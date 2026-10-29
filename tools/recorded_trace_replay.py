#!/usr/bin/env python3
"""Replay the reviewed deterministic trace corpus."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yewo.evaluation.recorded_trace_replay import main


if __name__ == "__main__":
    raise SystemExit(main())
