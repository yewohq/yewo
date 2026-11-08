"""Exclusive file locks that hold on every supported host."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
import os
from pathlib import Path
import sys
import time


LOCK_FILE_MODE = 0o600
LOCK_FILE_SUFFIX = ".lock"
_LOCKED_BYTE_COUNT = 1
_RETRY_INTERVAL_SECONDS = 0.05
_DEFAULT_TIMEOUT_SECONDS = 30.0


@contextmanager
def exclusive_file_lock(
    lock_path: Path,
    *,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
) -> Generator[None, None, None]:
    """Hold one exclusive lock through a dedicated lock file.

    The caller owns the lock file. No reader opens it, so a mandatory host
    lock cannot block a reader of the data the lock protects.
    """
    if timeout_seconds <= 0:
        raise ValueError("The file-lock timeout must be greater than zero.")
    flags = os.O_RDWR | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(lock_path, flags, LOCK_FILE_MODE)
    try:
        _acquire_descriptor_lock(descriptor, timeout_seconds)
        try:
            yield
        finally:
            _release_descriptor_lock(descriptor)
    finally:
        os.close(descriptor)


if sys.platform == "win32":
    import msvcrt

    def _acquire_descriptor_lock(descriptor: int, timeout_seconds: float) -> None:
        # Windows locks one byte range from the current position, so every
        # operation seeks to zero first. The range must match on release.
        deadline = time.monotonic() + timeout_seconds
        while True:
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, _LOCKED_BYTE_COUNT)
                return
            except OSError:
                if time.monotonic() >= deadline:
                    raise
            time.sleep(_RETRY_INTERVAL_SECONDS)

    def _release_descriptor_lock(descriptor: int) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, _LOCKED_BYTE_COUNT)

else:
    import fcntl

    def _acquire_descriptor_lock(descriptor: int, timeout_seconds: float) -> None:
        # POSIX blocks until the holder releases, so the timeout is unused.
        del timeout_seconds
        fcntl.flock(descriptor, fcntl.LOCK_EX)

    def _release_descriptor_lock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
