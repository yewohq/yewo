"""Collision-safe run directories and append-only run records."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from yewo.artifacts import (
    RUN_RECORDS_DIRECTORY,
    AtomicArtifactWriter,
    ArtifactWriteError,
    new_record_identifier,
)
from yewo.run_narration import RunNarrator
from yewo.runs.errors import RunCoordinationError
from yewo.runs.records import RunRequest, RunStart, RunTerminal


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class RunWorkspace:
    """One isolated directory, its durable start record, and its narrator."""

    root: Path
    start: RunStart
    narrator: RunNarrator


class RunStore:
    """Persist immutable coordinator records beneath one run root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._writer = AtomicArtifactWriter(root)

    def persist_start(self, start: RunStart) -> Path:
        """Persist the invocation start before mutable resources open."""
        path = RUN_RECORDS_DIRECTORY / "start.json"
        self._write(path, start.model_dump_json(indent=2))
        return path

    def persist_terminal(self, start: RunStart, terminal: RunTerminal) -> Path:
        """Persist exactly one terminal for the matching invocation."""
        if start.run_id != terminal.run_id:
            raise RunCoordinationError("A run terminal must match its start record.")
        if terminal.completed_at < start.started_at:
            raise RunCoordinationError("A run cannot complete before it starts.")
        path = RUN_RECORDS_DIRECTORY / "terminal.json"
        self._write(path, terminal.model_dump_json(indent=2))
        return path

    def _write(self, path: Path, content: str) -> None:
        try:
            self._writer.write(path, f"{content}\n".encode())
        except ArtifactWriteError as error:
            raise RunCoordinationError(
                f"Could not persist the run record: {path.as_posix()}"
            ) from error


class RunWorkspaceFactory:
    """Create one unique run root and establish its durable ownership record."""

    def __init__(
        self,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
        time_factory: Callable[[], datetime] = _utc_now,
        store_factory: Callable[[Path], RunStore] = RunStore,
        narrator_factory: Callable[[Path], RunNarrator] = RunNarrator,
    ) -> None:
        self._identifier_factory = identifier_factory
        self._time_factory = time_factory
        self._store_factory = store_factory
        self._narrator_factory = narrator_factory

    def create(self, request: RunRequest) -> RunWorkspace:
        """Create and claim one directory without reusing existing state."""
        try:
            start = RunStart(
                run_id=self._identifier_factory(),
                request=request,
                started_at=self._time_factory(),
            )
        except ValidationError as error:
            raise RunCoordinationError("Could not create a valid run start record.") from error
        output_root = request.output_root
        run_root = output_root / start.run_id
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            if not output_root.is_dir() or output_root.is_symlink():
                raise RunCoordinationError("The run output root must be a real directory.")
            run_root.mkdir()
        except RunCoordinationError:
            raise
        except FileExistsError as error:
            raise RunCoordinationError(f"The run directory already exists: {run_root}") from error
        except OSError as error:
            raise RunCoordinationError(f"Could not create the run directory: {run_root}") from error
        try:
            self._store_factory(run_root).persist_start(start)
        except BaseException:
            _discard_empty_workspace(run_root)
            raise
        return RunWorkspace(root=run_root, start=start, narrator=self._narrator_factory(run_root))


def _discard_empty_workspace(run_root: Path) -> None:
    directory = run_root / RUN_RECORDS_DIRECTORY
    while directory != run_root:
        with suppress(OSError):
            directory.rmdir()
        directory = directory.parent
    with suppress(OSError):
        run_root.rmdir()
