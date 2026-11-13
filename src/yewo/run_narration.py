"""Durable operator narration for one run."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Generator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from multiprocessing import get_context
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import sys
from threading import Condition, Lock, Thread
import time
from typing import Literal, TextIO

# Pipe() returns PipeConnection on Windows and Connection on POSIX. They share no
# public base class, so one alias must stand for both.
if sys.platform == "win32":
    from multiprocessing.connection import PipeConnection as PipeEndpoint
else:
    from multiprocessing.connection import Connection as PipeEndpoint

from yewo.artifacts import RUNTIME_DIRECTORY


_WRITER_STOP_TIMEOUT_SECONDS = 10.0
_OUTBOUND_LINE_CAPACITY = 4_096


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class NarrationPersistenceError(RuntimeError):
    """The run-owned narration writer could not establish durability."""


@dataclass(frozen=True, slots=True)
class _WriterMessage:
    kind: Literal["line", "sync", "close"]
    path: str | None = None
    content: str | None = None
    barrier_id: int | None = None
    enqueued_ns: int = 0
    bounded_line: bool = False


@dataclass(frozen=True, slots=True)
class _WriterAcknowledgement:
    barrier_id: int
    queue_delay_ms: float
    sync_duration_ms: float
    failure: str | None = None


class RunNarrator:
    """Send narration to a run-owned process and gate action durability."""

    def __init__(
        self,
        root: Path,
        *,
        time_factory: Callable[[], datetime] = _utc_now,
        stop_timeout_seconds: float = _WRITER_STOP_TIMEOUT_SECONDS,
        outbound_line_capacity: int = _OUTBOUND_LINE_CAPACITY,
    ) -> None:
        if stop_timeout_seconds <= 0:
            raise ValueError("The narration writer stop timeout must be positive.")
        if outbound_line_capacity <= 0:
            raise ValueError("The narration outbound line capacity must be positive.")
        self._session_path = root / RUNTIME_DIRECTORY / "yewo-session.log"
        self._trace_path = root / RUNTIME_DIRECTORY / "trace.log"
        self._time_factory = time_factory
        self._stop_timeout_seconds = stop_timeout_seconds
        self._outbound_line_capacity = outbound_line_capacity
        self._state_lock = Lock()
        self._receive_lock = Lock()
        self._outbound_condition = Condition()
        self._outbound: deque[_WriterMessage] = deque()
        self._queued_bounded_lines = 0
        self._overflowed_lines = 0
        self._connection: PipeEndpoint | None = None
        self._writer: BaseProcess | None = None
        self._feeder: Thread | None = None
        self._closed = False
        self._next_barrier_id = 1
        self._pending_barrier_ids: deque[int] = deque()
        self._acknowledged_barrier_id = 0
        self._write_failure: str | None = None

    def record(self, source: str, message: str) -> None:
        """Send one high-level message to the writer process."""
        self._send_line(self._session_path, source, message)

    def trace(self, source: str, message: str) -> None:
        """Send one diagnostic message to the writer process."""
        self._send_line(self._trace_path, source, message)

    def sync(self) -> int | None:
        """Request a background durability barrier and return its identifier."""
        with self._state_lock:
            if self._closed or self._write_failure is not None:
                return None
            barrier_id = self._next_barrier_id
            self._next_barrier_id += 1
        message = _WriterMessage(
            kind="sync",
            barrier_id=barrier_id,
            enqueued_ns=time.monotonic_ns(),
        )
        if not self._send(message):
            return None
        with self._state_lock:
            self._pending_barrier_ids.append(barrier_id)
        return barrier_id

    def wait_for_pending_sync(self) -> None:
        """Require the latest action-boundary barrier before another dispatch."""
        with self._state_lock:
            barrier_id = self._pending_barrier_ids[-1] if self._pending_barrier_ids else None
            failure = self._write_failure
        if failure is not None:
            raise NarrationPersistenceError(failure)
        if barrier_id is not None:
            self._wait_for_barrier(barrier_id, record_metrics=True)

    def close(self) -> None:
        """Durably finish narration without allowing a failed writer to hang."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            writer = self._writer
            connection = self._connection
            feeder = self._feeder
            barrier_id = self._next_barrier_id
            self._next_barrier_id += 1
        if writer is None or connection is None:
            return
        deadline = time.monotonic() + self._stop_timeout_seconds
        sent = self._send(
            _WriterMessage(
                kind="close",
                barrier_id=barrier_id,
                enqueued_ns=time.monotonic_ns(),
            ),
            allow_closed=True,
        )
        if sent:
            self._wait_for_barrier(
                barrier_id,
                record_metrics=False,
                raise_failure=False,
                deadline=deadline,
            )
        writer.join(timeout=max(0.0, deadline - time.monotonic()))
        if writer.is_alive():
            self._remember_failure("The narration writer did not stop before its timeout.")
            writer.terminate()
        with suppress(Exception):  # noqa: BLE001 - closing must release a blocked feeder
            connection.close()
        writer.join(timeout=1.0)
        if feeder is not None:
            with self._outbound_condition:
                self._outbound_condition.notify_all()
            feeder.join(timeout=1.0)

    @property
    def write_failure(self) -> str | None:
        """Return the first writer or transport failure."""
        with self._state_lock:
            return self._write_failure

    @property
    def writer_is_alive(self) -> bool:
        """Return whether the run-owned writer process is active."""
        with self._state_lock:
            writer = self._writer
        return writer is not None and writer.is_alive()

    def _send_line(self, path: Path, source: str, message: str) -> None:
        content = f"{self._time_factory().isoformat()} {source}: {_single_line(message)}\n"
        self._send(
            _WriterMessage(
                kind="line",
                path=str(path),
                content=content,
                enqueued_ns=time.monotonic_ns(),
                bounded_line=True,
            )
        )

    def _send(self, message: _WriterMessage, *, allow_closed: bool = False) -> bool:
        try:
            if self._ensure_writer(allow_closed=allow_closed) is None:
                return False
            with self._state_lock:
                if (self._closed and not allow_closed) or self._write_failure is not None:
                    return False
                return self._enqueue(message)
        except Exception as error:  # noqa: BLE001 - narration must not interrupt device work
            self._remember_failure(_format_failure(error))
            return False

    def _enqueue(self, message: _WriterMessage) -> bool:
        with self._outbound_condition:
            if message.bounded_line:
                if self._queued_bounded_lines >= self._outbound_line_capacity:
                    self._overflowed_lines += 1
                    return False
                self._queued_bounded_lines += 1
            else:
                self._append_overflow_notice_locked()
            self._outbound.append(message)
            self._outbound_condition.notify()
        return True

    def _append_overflow_notice_locked(self) -> None:
        if self._overflowed_lines == 0:
            return
        content = (
            f"{self._time_factory().isoformat()} narration-writer: "
            f"transport_overflow_dropped_lines={self._overflowed_lines} "
            f"capacity={self._outbound_line_capacity}\n"
        )
        self._outbound.append(
            _WriterMessage(
                kind="line",
                path=str(self._trace_path),
                content=content,
                enqueued_ns=time.monotonic_ns(),
            )
        )
        self._overflowed_lines = 0

    def _ensure_writer(self, *, allow_closed: bool) -> PipeEndpoint | None:
        with self._state_lock:
            if self._closed and not allow_closed:
                return None
            if self._connection is not None:
                return self._connection
            if self._write_failure is not None:
                return None
            context = get_context("spawn")
            parent_connection, child_connection = context.Pipe(duplex=True)
            writer = context.Process(
                target=_write_narration,
                args=(child_connection,),
                name="yewo-run-narrator",
            )
            writer.start()
            child_connection.close()
            feeder = Thread(
                target=self._feed_writer,
                args=(parent_connection,),
                name="yewo-run-narrator-feeder",
                daemon=True,
            )
            self._connection = parent_connection
            self._writer = writer
            self._feeder = feeder
            feeder.start()
            return parent_connection

    def _feed_writer(self, connection: PipeEndpoint) -> None:
        while True:
            with self._outbound_condition:
                while not self._outbound:
                    self._outbound_condition.wait()
                message = self._outbound.popleft()
                if message.bounded_line:
                    self._queued_bounded_lines -= 1
            try:
                connection.send(message)
            except Exception as error:  # noqa: BLE001 - surface all transport failures
                self._remember_failure(_format_failure(error))
                with self._outbound_condition:
                    self._outbound.clear()
                    self._queued_bounded_lines = 0
                    self._outbound_condition.notify_all()
                return
            if message.kind == "close":
                return

    def _wait_for_barrier(
        self,
        barrier_id: int,
        *,
        record_metrics: bool,
        raise_failure: bool = True,
        deadline: float | None = None,
    ) -> None:
        if deadline is None:
            deadline = time.monotonic() + self._stop_timeout_seconds
        while True:
            with self._state_lock:
                failure = self._write_failure
                acknowledged = self._acknowledged_barrier_id >= barrier_id
                connection = self._connection
                writer = self._writer
            if failure is not None:
                if raise_failure:
                    raise NarrationPersistenceError(failure)
                return
            if acknowledged:
                return
            if connection is None or writer is None:
                self._remember_failure("The narration writer was not available.")
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._remember_failure("The narration durability barrier timed out.")
                continue
            if not writer.is_alive() and not connection.poll():
                self._remember_failure(
                    "The narration writer exited before acknowledging a barrier."
                )
                continue
            try:
                with self._receive_lock:
                    if not connection.poll(min(remaining, 0.1)):
                        continue
                    acknowledgement = connection.recv()
            except (EOFError, OSError) as error:
                self._remember_failure(_format_failure(error))
                continue
            if not isinstance(acknowledgement, _WriterAcknowledgement):
                self._remember_failure("The narration writer returned an invalid acknowledgement.")
                continue
            with self._state_lock:
                self._acknowledged_barrier_id = max(
                    self._acknowledged_barrier_id,
                    acknowledgement.barrier_id,
                )
                while (
                    self._pending_barrier_ids
                    and self._pending_barrier_ids[0] <= acknowledgement.barrier_id
                ):
                    self._pending_barrier_ids.popleft()
            if acknowledgement.failure is not None:
                self._remember_failure(acknowledgement.failure)
                continue
            if record_metrics:
                self.trace(
                    "narration-writer",
                    f"barrier={acknowledgement.barrier_id} "
                    f"queue_delay_ms={acknowledgement.queue_delay_ms:.1f} "
                    f"fsync_duration_ms={acknowledgement.sync_duration_ms:.1f}",
                )

    def _remember_failure(self, failure: str) -> None:
        with self._state_lock:
            if self._write_failure is None:
                self._write_failure = failure


def _write_narration(connection: PipeEndpoint) -> None:
    streams: dict[str, TextIO] = {}
    failure: str | None = None
    try:
        should_close = False
        while not should_close:
            try:
                message = connection.recv()
            except EOFError:
                break
            if not isinstance(message, _WriterMessage):
                failure = failure or "The narration writer received an invalid message."
                continue
            if message.kind == "line":
                failure = failure or _write_line(message, streams)
                continue
            should_close = message.kind == "close"
            sync_started = time.monotonic()
            failure = failure or _sync_streams(streams)
            acknowledgement = _WriterAcknowledgement(
                barrier_id=message.barrier_id or 0,
                queue_delay_ms=max(0.0, time.monotonic_ns() - message.enqueued_ns) / 1_000_000,
                sync_duration_ms=max(0.0, time.monotonic() - sync_started) * 1_000,
                failure=failure,
            )
            try:
                connection.send(acknowledgement)
            except (BrokenPipeError, EOFError, OSError):
                break
    except Exception as error:  # noqa: BLE001 - the writer must always release its owner
        failure = failure or _format_failure(error)
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(_WriterAcknowledgement(0, 0.0, 0.0, failure))
    finally:
        _sync_streams(streams)
        for stream in streams.values():
            with suppress(Exception):  # noqa: BLE001 - process exit must not hang its owner
                stream.close()
        connection.close()


def _write_line(message: _WriterMessage, streams: dict[str, TextIO]) -> str | None:
    if message.path is None or message.content is None:
        return "The narration writer received an incomplete line."
    try:
        stream = streams.get(message.path)
        if stream is None:
            path = Path(message.path)
            path.parent.mkdir(parents=True, exist_ok=True)
            stream = path.open("a", encoding="utf-8")
            streams[message.path] = stream
        stream.write(message.content)
    except Exception as error:  # noqa: BLE001 - diagnostics must not kill the writer
        return _format_failure(error)
    return None


def _sync_streams(streams: dict[str, TextIO]) -> str | None:
    for stream in streams.values():
        try:
            stream.flush()
            os.fsync(stream.fileno())
        except Exception as error:  # noqa: BLE001 - report durability failure to the owner
            return _format_failure(error)
    if os.name != "nt":
        for directory in {str(Path(path).parent) for path in streams}:
            descriptor: int | None = None
            try:
                descriptor = os.open(directory, os.O_RDONLY)
                os.fsync(descriptor)
            except Exception as error:  # noqa: BLE001 - directory durability is part of the barrier
                return _format_failure(error)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
    return None


def _format_failure(error: BaseException) -> str:
    return f"{type(error).__name__}: {_single_line(str(error))}"


def _single_line(message: str) -> str:
    sanitized = message.encode("utf-8", "replace").decode("utf-8")
    return sanitized.replace("\r", "\\r").replace("\n", "\\n")


_current_narrator: ContextVar[RunNarrator | None] = ContextVar("_current_narrator", default=None)


def current_narrator() -> RunNarrator | None:
    """Return the narrator active in the current run context."""
    return _current_narrator.get()


@contextmanager
def narration_scope(narrator: RunNarrator) -> Generator[None, None, None]:
    """Bind one narrator and stop its writer on every exit path."""
    token = _current_narrator.set(narrator)
    try:
        yield
    finally:
        _current_narrator.reset(token)
        narrator.close()


def elapsed_milliseconds(started: float, completed: float) -> float:
    """Return a non-negative elapsed span in milliseconds."""
    return max(0.0, completed - started) * 1_000
