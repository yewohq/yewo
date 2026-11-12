"""Tests for durable, process-owned run narration."""

from datetime import datetime, timezone
from multiprocessing.connection import Connection
from pathlib import Path
import re
from threading import Event, Thread

import pytest

from yewo.run_narration import (
    NarrationPersistenceError,
    RunNarrator,
    current_narrator,
    narration_scope,
)


CAPTURED_AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def test_record_and_trace_append_to_separate_files(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)

    narrator.record("run-coordinator", "device ready: Pixel_8")
    narrator.trace("device-command", "adb: 8 arguments omitted")
    narrator.record("run-coordinator", "appium server acquired")
    narrator.close()

    session_log = (tmp_path / "runtime/yewo-session.log").read_text()
    trace_log = (tmp_path / "runtime/trace.log").read_text()
    assert session_log.splitlines() == [
        f"{CAPTURED_AT.isoformat()} run-coordinator: device ready: Pixel_8",
        f"{CAPTURED_AT.isoformat()} run-coordinator: appium server acquired",
    ]
    assert trace_log.splitlines() == [
        f"{CAPTURED_AT.isoformat()} device-command: adb: 8 arguments omitted",
    ]


def test_writer_failure_is_reported_at_a_durability_boundary(tmp_path: Path) -> None:
    blocked_root = tmp_path / "blocked"
    blocked_root.write_text("not a directory")
    narrator = RunNarrator(blocked_root, time_factory=lambda: CAPTURED_AT)

    narrator.record("run-coordinator", "device ready: Pixel_8")
    narrator.sync()

    with pytest.raises(NarrationPersistenceError, match="FileExistsError|NotADirectoryError"):
        narrator.wait_for_pending_sync()

    assert narrator.write_failure is not None
    narrator.close()


def test_narration_scope_sets_resets_and_stops_the_writer(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path)
    assert current_narrator() is None

    with narration_scope(narrator):
        assert current_narrator() is narrator
        narrator.record("run-coordinator", "started")
        assert narrator.writer_is_alive

    assert current_narrator() is None
    assert not narrator.writer_is_alive


def test_cancellation_cannot_leave_the_writer_running(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path)

    with pytest.raises(RuntimeError, match="cancelled"), narration_scope(narrator):
        narrator.record("run-coordinator", "started")
        raise RuntimeError("cancelled")

    assert not narrator.writer_is_alive


def test_narration_escapes_line_breaks_and_invalid_unicode(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)

    narrator.trace("run-coordinator", "RuntimeError\ncut emoji: \udce9")
    narrator.close()

    lines = (tmp_path / "runtime/trace.log").read_text().splitlines()
    assert lines == [f"{CAPTURED_AT.isoformat()} run-coordinator: RuntimeError\\ncut emoji: ?"]
    assert narrator.write_failure is None


def test_action_barrier_records_queue_and_fsync_timings(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)
    narrator.trace("device-agent", "step=1 outcome=observed")

    barrier_id = narrator.sync()
    narrator.wait_for_pending_sync()
    narrator.close()

    assert barrier_id == 1
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "narration-writer: barrier=1 queue_delay_ms=" in trace
    assert "fsync_duration_ms=" in trace


def test_transport_backpressure_does_not_drop_lines(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)

    for index in range(1_000):
        narrator.trace("device-agent", f"burst_line={index}")
    narrator.sync()
    narrator.wait_for_pending_sync()
    narrator.close()

    trace = (tmp_path / "runtime/trace.log").read_text()
    assert sum("burst_line=" in line for line in trace.splitlines()) == 1_000


def test_stalled_transport_never_blocks_trace_and_reports_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    narrator = RunNarrator(
        tmp_path,
        time_factory=lambda: CAPTURED_AT,
        outbound_line_capacity=8,
    )
    narrator._ensure_writer(allow_closed=False)  # noqa: SLF001 - stall the real transport
    send_started = Event()
    release_send = Event()
    original_send = Connection.send

    def stalled_send(connection: Connection, message: object) -> None:
        send_started.set()
        release_send.wait(timeout=5)
        original_send(connection, message)

    monkeypatch.setattr(Connection, "send", stalled_send)
    completed = Event()

    def emit_burst() -> None:
        for index in range(30):
            narrator.trace("device-agent", f"stalled_line={index}")
        narrator.sync()
        completed.set()

    emitter = Thread(target=emit_burst)
    emitter.start()
    assert send_started.wait(timeout=1)
    returned_while_transport_was_stalled = completed.wait(timeout=1)
    release_send.set()
    emitter.join(timeout=2)

    assert returned_while_transport_was_stalled
    narrator.wait_for_pending_sync()
    narrator.close()

    trace = (tmp_path / "runtime/trace.log").read_text()
    overflow = re.search(r"transport_overflow_dropped_lines=(\d+) capacity=8", trace)
    assert overflow is not None
    landed_lines = sum("stalled_line=" in line for line in trace.splitlines())
    assert landed_lines + int(overflow.group(1)) == 30
