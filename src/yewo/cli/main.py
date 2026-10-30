"""Run the Yewo command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yewo", description="Test a mobile application.")
    parser.add_argument("--version", action="version", version=version("yewo"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse command-line arguments."""
    _parser().parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
