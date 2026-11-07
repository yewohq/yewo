"""Two-tier narration for session evidence services."""

from pathlib import Path

from yewo.run_narration import RunNarrator, narration_scope

from automation.test_session import (
    _RecordingDriverStub,
    _application,
    _device,
    _manager,
    _open,
    _request,
    _server,
)


def test_session_narrates_recording_and_device_log_boundaries(tmp_path: Path) -> None:
    events: list[str] = []
    driver = _RecordingDriverStub(events)
    with narration_scope(RunNarrator(tmp_path)):
        session = _open(
            _manager(driver, installed_before=True),
            _device(events),
            _server(events),
            _request(),
            _application(),
            tmp_path,
        )
        session.close()

    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Device-log capture started" in narration
    assert "Screen recording active." in narration
    assert "Screen recording finished with segment state captured." in narration
    assert "Device-log capture finished" in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "device-logs: boundary=" in trace
    assert "checkpoint=session_start" in trace
    assert "checkpoint=session_end" in trace
    assert "screen-recording: recording=" in trace
    assert "artifact=recordings/" in trace
