#!/usr/bin/env python3
"""Reject Python files that exceed Yewo's physical line limit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

MAX_FILE_LINES = 550
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    REPOSITORY_ROOT / "src" / "yewo",
    REPOSITORY_ROOT / "tests",
)


def _python_files(targets: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for target in targets:
        if target.is_dir():
            files.update(
                path for path in target.rglob("*.py") if "__pycache__" not in path.parts
            )
        elif target.is_file() and target.suffix == ".py":
            files.add(target)
    return sorted(files)


def _display_path(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT)
    except ValueError:
        return path.resolve()


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Python files or directories to inspect (default: src/yewo and tests)",
    )
    args = parser.parse_args(argv)

    requested_paths = args.paths or list(DEFAULT_TARGETS)
    targets = [
        path if path.is_absolute() else REPOSITORY_ROOT / path
        for path in requested_paths
    ]
    python_files = _python_files(targets)
    measured_files = [(path, _line_count(path)) for path in python_files]
    oversized_files = [
        (path, line_count)
        for path, line_count in measured_files
        if line_count > MAX_FILE_LINES
    ]

    if oversized_files:
        for path, line_count in oversized_files:
            print(
                f"{_display_path(path)}: {line_count} lines exceeds limit of "
                f"{MAX_FILE_LINES}",
                file=sys.stderr,
            )
        return 1

    print(f"Quality gate passed for {len(python_files)} Python files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
