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
    """Reject execution until a product workflow is available."""
    parser = _parser()
    parser.parse_args(argv)
    parser.error("No product workflows are available.")


if __name__ == "__main__":
    raise SystemExit(main())
