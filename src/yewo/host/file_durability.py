"""Durable file replacement across hosts with different flush guarantees."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time


_REPLACE_RETRY_INTERVAL_SECONDS = 0.05
_REPLACE_TIMEOUT_SECONDS = 2.0


def flush_directory_entry(directory: Path) -> None:
    """Flush one directory entry so a replaced name survives a host crash.

    Windows exposes no directory handle to flush, so the entry stays in the
    host cache. Durability on Windows is weaker than on POSIX by that much.
    """
    if sys.platform == "win32":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def replace_file_durably(temporary_path: Path, destination: Path) -> None:
    """Replace one file atomically and flush the containing directory."""
    _replace_file(temporary_path, destination)
    flush_directory_entry(destination.parent)


if sys.platform == "win32":

    def _replace_file(temporary_path: Path, destination: Path) -> None:
        # Windows refuses to replace a file that another process holds open,
        # which a concurrent report reader can cause. Retry for a bounded time.
        deadline = time.monotonic() + _REPLACE_TIMEOUT_SECONDS
        while True:
            try:
                os.replace(temporary_path, destination)
                return
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(_REPLACE_RETRY_INTERVAL_SECONDS)

else:

    def _replace_file(temporary_path: Path, destination: Path) -> None:
        os.replace(temporary_path, destination)
