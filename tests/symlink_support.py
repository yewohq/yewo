"""Skip a test when the host refuses to create a symbolic link."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest


def _host_creates_symlinks() -> bool:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "target"
        target.write_bytes(b"")
        try:
            (root / "link").symlink_to(target)
        except (OSError, NotImplementedError):
            return False
    return True


requires_symlinks = pytest.mark.skipif(
    not _host_creates_symlinks(),
    reason="This host refuses to create a symbolic link. Windows needs Developer Mode.",
)
