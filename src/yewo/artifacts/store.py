"""Safe atomic writes beneath one artifact root."""

from __future__ import annotations

from contextlib import suppress
import fcntl
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from yewo.artifacts.errors import ArtifactWriteError


class AtomicArtifactWriter:
    """Create one artifact atomically without overwriting existing evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, relative_path: Path, content: bytes) -> None:
        """Write in-memory content as one immutable artifact."""
        self._write(relative_path, lambda stream: stream.write(content))

    def write_file(self, relative_path: Path, source: Path) -> None:
        """Stream one local file into an immutable artifact."""
        if not source.is_file() or source.is_symlink():
            raise ArtifactWriteError("The artifact source must be one regular file.")

        def copy(stream) -> None:
            with source.open("rb") as source_stream:
                shutil.copyfileobj(source_stream, stream, length=1024 * 1024)

        self._write(relative_path, copy)

    def _write(self, relative_path: Path, write_content) -> None:
        destination = self.root / relative_path
        temporary_path: Path | None = None
        try:
            _prepare_directory(self.root, destination.parent)
            if destination.exists():
                raise ArtifactWriteError(f"The artifact already exists: {relative_path.as_posix()}")
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=".artifact-",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                write_content(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, destination)
        except ArtifactWriteError:
            raise
        except OSError as error:
            raise ArtifactWriteError(
                f"Could not persist artifact: {relative_path.as_posix()}"
            ) from error
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)


class AppendOnlyJsonlWriter:
    """Append canonical records to one durable JSONL ledger."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def append(
        self,
        relative_path: Path,
        record: dict[str, Any],
        *,
        identity_field: str,
    ) -> bytes:
        """Append one uniquely identified record and return its encoded line."""
        identity = record.get(identity_field)
        if not isinstance(identity, str) or not identity:
            raise ArtifactWriteError(
                f"A JSONL record requires a nonempty '{identity_field}' identity."
            )
        try:
            content = (
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as error:
            raise ArtifactWriteError("The JSONL record is not serializable.") from error

        destination = self.root / relative_path
        descriptor: int | None = None
        try:
            _prepare_directory(self.root, destination.parent)
            if destination.is_symlink():
                raise ArtifactWriteError("A JSONL ledger cannot be a symbolic link.")
            flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(destination, flags, 0o600)
            with os.fdopen(descriptor, "r+b") as stream:
                descriptor = None
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
                stream.seek(0)
                for line_number, line in enumerate(stream, start=1):
                    if not line.endswith(b"\n"):
                        raise ArtifactWriteError(
                            f"The JSONL ledger has an incomplete line: {line_number}."
                        )
                    try:
                        existing = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError) as error:
                        raise ArtifactWriteError(
                            f"The JSONL ledger has an invalid line: {line_number}."
                        ) from error
                    if isinstance(existing, dict) and existing.get(identity_field) == identity:
                        raise ArtifactWriteError(
                            f"The JSONL record already exists: {identity_field}={identity}"
                        )
                stream.seek(0, os.SEEK_END)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except ArtifactWriteError:
            raise
        except OSError as error:
            raise ArtifactWriteError(
                f"Could not append artifact: {relative_path.as_posix()}"
            ) from error
        finally:
            if descriptor is not None:
                os.close(descriptor)
        return content


def _prepare_directory(root: Path, directory: Path) -> None:
    try:
        root.mkdir(parents=True, exist_ok=True)
        relative_directory = directory.relative_to(root)
    except (OSError, ValueError) as error:
        raise ArtifactWriteError("The artifact directory is outside the artifact root.") from error
    resolved_root = root.resolve()
    current_directory = root
    for part in relative_directory.parts:
        current_directory /= part
        if current_directory.is_symlink():
            raise ArtifactWriteError("The artifact directory resolves outside the artifact root.")
        try:
            current_directory.mkdir(exist_ok=True)
            current_directory.resolve().relative_to(resolved_root)
        except (OSError, ValueError) as error:
            raise ArtifactWriteError(
                "The artifact directory resolves outside the artifact root."
            ) from error
