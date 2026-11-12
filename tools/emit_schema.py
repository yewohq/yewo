#!/usr/bin/env python3
"""Write yewo-schema.json from the models that validate a yewo.yaml.

uv run python tools/emit_schema.py          # write
uv run python tools/emit_schema.py --check  # fail when it is out of date
"""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from yewo.project_config.schema import build_project_schema  # noqa: E402

SCHEMA_PATH = REPOSITORY_ROOT / "yewo-schema.json"


def main() -> int:
    rendered = build_project_schema()
    if "--check" in sys.argv:
        current = SCHEMA_PATH.read_text(encoding="utf-8") if SCHEMA_PATH.is_file() else ""
        if current != rendered:
            print(f"{SCHEMA_PATH.name} is out of date. Run: uv run python tools/emit_schema.py")
            return 1
        print(f"{SCHEMA_PATH.name} is up to date.")
        return 0
    SCHEMA_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {SCHEMA_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
