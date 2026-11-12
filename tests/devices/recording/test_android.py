"""Tests for Appium-independent Android recording recovery."""

from collections import deque
from pathlib import Path
import subprocess
from typing import cast

import pytest

from yewo.applications import ApplicationDescriptor
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.android._file_commands import AdbFileCommands
from yewo.devices.android._recording_commands import AdbRecordingCommands
from yewo.devices.media.android import AdbMediaTransport
from yewo.devices.media.errors import MediaTransferError
from yewo.devices.recording import (
    ScreenRecordingCaptureMethod,
    ScreenRecordingError,
)
from yewo.devices.recording.android import AdbRecoverableScreenRecorder
from yewo.devices.remote_files import RemoteTemporaryFileLease
from support import ClockStub


VIDEO = b"\x00\x00\x00\x18ftypmp42video"


class _AdbStub:
    def __init__(self) -> None:
        self.size: int | None = 0
        self.commands: list[tuple[str, ...]] = []
        self.stopped_paths: list[str] = []
        self.removed_paths: list[str] = []

    def command(
        self,
        connection_id: str,
        remote_path: str,
        maximum_duration_seconds: int,
    ) -> tuple[tuple[str, ...], int]:
        command = (
            "adb",
            connection_id,
            "screenrecord",
            str(min(maximum_duration_seconds, 180)),
            remote_path,
        )
        self.commands.append(command)
        return command, min(maximum_duration_seconds, 180)

    def remote_file_size(self, connection_id: str, remote_path: str) -> int | None:
        del connection_id, remote_path
        return self.size

    def stop(self, connection_id: str, remote_path: str) -> None:
        del connection_id
        self.stopped_paths.append(remote_path)

    def remove_file(self, connection_id: str, remote_path: str) -> None:
        del connection_id
        self.removed_paths.append(remote_path)


class _TransportStub:
    def __init__(self) -> None:
        self.content = VIDEO
        self.error: MediaTransferError | None = None
        self.pulled_paths: list[str] = []

    def push(self, remote_path: str, content: bytes) -> None:
        del remote_path, content

    def pull(self, remote_path: str, maximum_bytes: int) -> bytes:
        assert maximum_bytes > len(self.content)
        self.pulled_paths.append(remote_path)
        if self.error is not None:
            raise self.error
        return self.content

    def pull_to_path(
        self,
        remote_path: str,
        destination: Path,
        maximum_bytes: int,
        *,
        timeout_seconds: float = 120,
    ) -> None:
        del timeout_seconds
        destination.write_bytes(self.pull(remote_path, maximum_bytes))


class _ProcessStub:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.terminated = True
        self.returncode = -9


class _StuckProcessStub(_ProcessStub):
    def __init__(self) -> None:
        super().__init__()
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            raise subprocess.TimeoutExpired("screenrecord", timeout or 0)
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _lease(adb: _AdbStub) -> RemoteTemporaryFileLease:
    identifiers = deque(("namespace_1", "file_1"))

    class _Remover:
        def remove_file(self, remote_path: str) -> None:
            adb.remove_file("emulator-5554", remote_path)

    return RemoteTemporaryFileLease.open(
        _Remover(),
        DeviceDescriptor(
            platform=DevicePlatform.ANDROID,
            target_id="pixel_9",
            connection_id="emulator-5554",
            display_name="Pixel 9",
            kind=DeviceKind.VIRTUAL,
            readiness=DeviceReadiness.READY,
            native_state="device",
        ),
        ApplicationDescriptor(
            platform=DevicePlatform.ANDROID,
            application_id="com.example.app",
        ),
        identifier_factory=identifiers.popleft,
    )


def test_recorder_stops_pulls_and_removes_without_appium() -> None:
    adb = _AdbStub()
    transport = _TransportStub()
    process = _ProcessStub()
    recorder = AdbRecoverableScreenRecorder(
        cast(AdbRecordingCommands, adb),
        cast(AdbFileCommands, adb),
        cast(AdbMediaTransport, transport),
        _lease(adb),
        "emulator-5554",
        process_starter=lambda command: process,
        identifier_factory=lambda: "segment_1",
        clock=ClockStub(),
    )

    recorder.start(1_800)
    capture = recorder.stop()
    video = capture.path.read_bytes()
    assert capture.start.requested_maximum_seconds == 1_800
    assert capture.start.effective_maximum_seconds == 180
    assert adb.removed_paths == []
    capture.accept()
    capture.close()

    remote_path = adb.commands[0][-1]
    assert video == VIDEO
    assert recorder.capture_method is ScreenRecordingCaptureMethod.REMOTE_FILE
    assert adb.stopped_paths == [remote_path]
    assert transport.pulled_paths == [remote_path]
    assert adb.removed_paths == [remote_path]


def test_failed_pull_leaves_the_exact_path_for_session_cleanup() -> None:
    adb = _AdbStub()
    transport = _TransportStub()
    transport.error = MediaTransferError("pull failed")
    lease = _lease(adb)
    recorder = AdbRecoverableScreenRecorder(
        cast(AdbRecordingCommands, adb),
        cast(AdbFileCommands, adb),
        cast(AdbMediaTransport, transport),
        lease,
        "emulator-5554",
        process_starter=lambda command: _ProcessStub(),
        identifier_factory=lambda: "segment_1",
        clock=ClockStub(),
    )
    recorder.start(30)

    with pytest.raises(ScreenRecordingError, match="recover"):
        recorder.stop()

    assert len(transport.pulled_paths) == 3
    assert recorder.can_start_after_failure is False
    assert len(lease.references) == 1
    lease.close()
    assert len(adb.removed_paths) == 1


def test_failed_pull_uses_one_total_recovery_deadline() -> None:
    adb = _AdbStub()
    clock = ClockStub()

    class _SlowFailingTransport(_TransportStub):
        def __init__(self) -> None:
            super().__init__()
            self.timeouts: list[float] = []

        def pull_to_path(
            self,
            remote_path: str,
            destination: Path,
            maximum_bytes: int,
            *,
            timeout_seconds: float = 120,
        ) -> None:
            del remote_path, destination, maximum_bytes
            self.timeouts.append(timeout_seconds)
            clock.now += 6
            raise MediaTransferError("pull failed")

    transport = _SlowFailingTransport()
    recorder = AdbRecoverableScreenRecorder(
        cast(AdbRecordingCommands, adb),
        cast(AdbFileCommands, adb),
        cast(AdbMediaTransport, transport),
        _lease(adb),
        "emulator-5554",
        process_starter=lambda command: _ProcessStub(),
        identifier_factory=lambda: "segment_1",
        clock=clock,
    )
    recorder.start(30, maximum_pull_seconds=10)

    with pytest.raises(ScreenRecordingError, match="recover"):
        recorder.stop()

    assert transport.timeouts == [10, 3.5]


def test_invalid_recovered_video_allows_a_fresh_segment() -> None:
    adb = _AdbStub()
    transport = _TransportStub()
    transport.content = b"not an mp4"
    lease = _lease(adb)
    recorder = AdbRecoverableScreenRecorder(
        cast(AdbRecordingCommands, adb),
        cast(AdbFileCommands, adb),
        cast(AdbMediaTransport, transport),
        lease,
        "emulator-5554",
        process_starter=lambda command: _ProcessStub(),
        identifier_factory=lambda: "segment_1",
        clock=ClockStub(),
    )
    recorder.start(30)

    with pytest.raises(ScreenRecordingError, match="invalid MP4"):
        recorder.stop()

    assert recorder.can_start_after_failure is True
    assert len(lease.references) == 1
    lease.close()
    assert len(adb.removed_paths) == 1


def test_stuck_process_escalates_from_terminate_to_kill() -> None:
    adb = _AdbStub()
    process = _StuckProcessStub()
    recorder = AdbRecoverableScreenRecorder(
        cast(AdbRecordingCommands, adb),
        cast(AdbFileCommands, adb),
        cast(AdbMediaTransport, _TransportStub()),
        _lease(adb),
        "emulator-5554",
        process_starter=lambda command: process,
        identifier_factory=lambda: "segment_1",
        clock=ClockStub(),
    )
    recorder.start(30)

    capture = recorder.stop()
    capture.accept()
    capture.close()

    assert process.terminated is True
    assert process.killed is True


def test_startup_failure_stops_the_process_and_releases_the_path() -> None:
    adb = _AdbStub()
    adb.size = None
    process = _ProcessStub(returncode=1)
    lease = _lease(adb)
    recorder = AdbRecoverableScreenRecorder(
        cast(AdbRecordingCommands, adb),
        cast(AdbFileCommands, adb),
        cast(AdbMediaTransport, _TransportStub()),
        lease,
        "emulator-5554",
        process_starter=lambda command: process,
        identifier_factory=lambda: "segment_1",
        clock=ClockStub(),
    )

    with pytest.raises(ScreenRecordingError, match="stopped during startup"):
        recorder.start(30)

    assert lease.references == ()
    assert len(adb.removed_paths) == 1
