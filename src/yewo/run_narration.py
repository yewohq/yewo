"""Best-effort operator narration for one run."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import os
from pathlib import Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunNarrator:
    """Append non-authoritative operator messages beneath one run root."""

    def __init__(self, root: Path, *, time_factory: Callable[[], datetime] = _utc_now) -> None:
        self._session_path = root / "runtime" / "yewo-session.log"
        self._trace_path = root / "runtime" / "trace.log"
        self._time_factory = time_factory

    def record(self, source: str, message: str) -> None:
        """Append one high-level message without affecting the run."""
        self._append(self._session_path, source, message)

    def trace(self, source: str, message: str) -> None:
        """Append one diagnostic message without affecting the run."""
        self._append(self._trace_path, source, message)

    def _append(self, path: Path, source: str, message: str) -> None:
        line = f"{self._time_factory().isoformat()} {source}: {_single_line(message)}\n"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError:
            return


def _single_line(message: str) -> str:
    return message.replace("\r", "\\r").replace("\n", "\\n")


_current_narrator: ContextVar[RunNarrator | None] = ContextVar("_current_narrator", default=None)


def current_narrator() -> RunNarrator | None:
    """Return the narrator active in the current run context."""
    return _current_narrator.get()


@contextmanager
def narration_scope(narrator: RunNarrator) -> Generator[None, None, None]:
    """Bind a narrator to shared run boundaries for one scope."""
    token = _current_narrator.set(narrator)
    try:
        yield
    finally:
        _current_narrator.reset(token)


def elapsed_milliseconds(started: float, completed: float) -> float:
    """Return a non-negative elapsed span in milliseconds."""
    return max(0.0, completed - started) * 1_000
