"""Shared boundaries for reading application artifacts."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Protocol

from yewo.applications.errors import ApplicationInspectionError
from yewo.applications.records import ApplicationDescriptor


class ApplicationArtifactInspector(Protocol):
    """Read application identity without installing the artifact."""

    def inspect(self, artifact_path: Path) -> ApplicationDescriptor:
        """Return identity facts observed from one artifact."""
        ...


def artifact_sha256(artifact_path: Path) -> str:
    """Return a stable digest for one file or directory artifact."""
    if not artifact_path.exists():
        raise ApplicationInspectionError(f"Application artifact does not exist: {artifact_path}")
    digest = sha256()
    try:
        if artifact_path.is_dir():
            members = sorted(path for path in artifact_path.rglob("*") if path.is_file())
            for member in members:
                relative_path = member.relative_to(artifact_path)
                digest.update(relative_path.as_posix().encode("utf-8"))
                digest.update(b"\0")
                _update_digest(digest, member)
                digest.update(b"\0")
        else:
            _update_digest(digest, artifact_path)
    except OSError as error:
        raise ApplicationInspectionError(
            f"Could not read application artifact: {artifact_path}"
        ) from error
    return digest.hexdigest()


def _update_digest(digest, path: Path) -> None:
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
