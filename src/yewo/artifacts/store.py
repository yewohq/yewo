"""Safe atomic writes beneath one artifact root."""

from __future__ import annotations

from contextlib import suppress
import os
from pathlib import Path
import tempfile

from yewo.artifacts.errors import ArtifactWriteError


class AtomicArtifactWriter:
    """Create one artifact atomically without overwriting existing evidence."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def write(self, relative_path: Path, content: bytes) -> None:
        destination = self.root / relative_path
        temporary_path: Path | None = None
        try:
            self._prepare_directory(destination.parent)
            if destination.exists():
                raise ArtifactWriteError(
                    f"The artifact already exists: {relative_path.as_posix()}"
                )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=destination.parent,
                prefix=".artifact-",
            )
            temporary_path = Path(temporary_name)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
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

    def _prepare_directory(self, directory: Path) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            relative_directory = directory.relative_to(self.root)
        except (OSError, ValueError) as error:
            raise ArtifactWriteError(
                "The artifact directory is outside the artifact root."
            ) from error
        resolved_root = self.root.resolve()
        current_directory = self.root
        for part in relative_directory.parts:
            current_directory /= part
            if current_directory.is_symlink():
                raise ArtifactWriteError(
                    "The artifact directory resolves outside the artifact root."
                )
            try:
                current_directory.mkdir(exist_ok=True)
                current_directory.resolve().relative_to(resolved_root)
            except (OSError, ValueError) as error:
                raise ArtifactWriteError(
                    "The artifact directory resolves outside the artifact root."
                ) from error
