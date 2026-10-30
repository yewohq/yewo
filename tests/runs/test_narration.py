"""Tests for best-effort, two-tier run narration."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from yewo.run_narration import RunNarrator, current_narrator, narration_scope


CAPTURED_AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def test_record_and_trace_append_to_separate_files(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)

    narrator.record("run-coordinator", "device ready: Pixel_8")
    narrator.trace("device-command", "adb: 8 arguments omitted")
    narrator.record("run-coordinator", "appium server acquired")

    session_log = (tmp_path / "runtime/yewo-session.log").read_text()
    trace_log = (tmp_path / "runtime/trace.log").read_text()
    assert session_log.splitlines() == [
        f"{CAPTURED_AT.isoformat()} run-coordinator: device ready: Pixel_8",
        f"{CAPTURED_AT.isoformat()} run-coordinator: appium server acquired",
    ]
    assert trace_log.splitlines() == [
        f"{CAPTURED_AT.isoformat()} device-command: adb: 8 arguments omitted",
    ]


def test_a_write_failure_is_swallowed_not_raised(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)

    with patch.object(Path, "mkdir", side_effect=OSError("disk full")):
        narrator.record("run-coordinator", "device ready: Pixel_8")
        narrator.trace("device-command", "adb: 5 arguments omitted")

    assert not (tmp_path / "runtime").exists()


def test_narration_scope_sets_and_resets_the_current_narrator(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path)
    assert current_narrator() is None

    with narration_scope(narrator):
        assert current_narrator() is narrator

    assert current_narrator() is None


def test_narration_scope_resets_even_when_the_body_raises(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path)

    try:
        with narration_scope(narrator):
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    assert current_narrator() is None


def test_narration_escapes_line_breaks(tmp_path: Path) -> None:
    narrator = RunNarrator(tmp_path, time_factory=lambda: CAPTURED_AT)

    narrator.trace("run-coordinator", "RuntimeError\nforged line")

    lines = (tmp_path / "runtime/trace.log").read_text().splitlines()
    assert lines == [f"{CAPTURED_AT.isoformat()} run-coordinator: RuntimeError\\nforged line"]
