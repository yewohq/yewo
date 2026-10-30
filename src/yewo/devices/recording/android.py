"""Recoverable Android recording through an independent ADB channel."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time
from typing import Protocol

from yewo.artifacts.records import new_record_identifier
from yewo.devices.android._file_commands import AdbFileCommands
from yewo.devices.android._recording_commands import AdbRecordingCommands
from yewo.devices.media.android import AdbMediaTransport
from yewo.devices.media.errors import MediaTransferError
from yewo.devices.recording.capture import (
    ScreenRecordingCapture,
    ScreenRecordingStart,
)
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.records import ScreenRecordingCaptureMethod
from yewo.devices.remote_files import (
    RemoteTemporaryFileError,
    RemoteTemporaryFileLease,
    RemoteTemporaryFileReference,
)


_START_TIMEOUT_SECONDS = 5.0
_MAXIMUM_RECOVERY_BYTES = 512 * 1024 * 1024
_START_POLL_SECONDS = 0.5
_RECOVERY_ATTEMPTS = 3


class _RecordingProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class _SystemClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def _start_process(command: Sequence[str]) -> _RecordingProcess:
    return subprocess.Popen(  # noqa: S603 - command is an owned argument list
        list(command),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


class AdbRecoverableScreenRecorder:
    """Record to a known remote path and pull it without Appium."""

    def __init__(
        self,
        recording_commands: AdbRecordingCommands,
        file_commands: AdbFileCommands,
        transport: AdbMediaTransport,
        remote_files: RemoteTemporaryFileLease,
        connection_id: str,
        *,
        process_starter: Callable[[Sequence[str]], _RecordingProcess] = _start_process,
        identifier_factory: Callable[[], str] = new_record_identifier,
        clock: _Clock | None = None,
    ) -> None:
        self._recording_commands = recording_commands
        self._file_commands = file_commands
        self._transport = transport
        self._remote_files = remote_files
        self._connection_id = connection_id
        self._process_starter = process_starter
        self._identifier_factory = identifier_factory
        self._clock = clock or _SystemClock()
        self._process: _RecordingProcess | None = None
        self._remote_file: RemoteTemporaryFileReference | None = None
        self._start: ScreenRecordingStart | None = None
        self._can_start_after_failure = True

    @property
    def capture_method(self) -> ScreenRecordingCaptureMethod:
        """Return that video bytes come from an independent remote pull."""
        return ScreenRecordingCaptureMethod.REMOTE_FILE

    @property
    def can_start_after_failure(self) -> bool:
        """Return whether no recorder or pending recovery blocks a new segment."""
        return self._can_start_after_failure

    def start(
        self,
        maximum_duration_seconds: int,
        maximum_stop_seconds: float = 10.0,
        maximum_pull_seconds: float = 120.0,
    ) -> ScreenRecordingStart:
        if self._process is not None or self._remote_file is not None:
            raise ScreenRecordingError("A screen-recording segment is already active.")
        try:
            identifier = self._identifier_factory()
            remote_file = self._remote_files.reserve(f"recording_{identifier}.mp4")
        except Exception as error:  # noqa: BLE001 - identifier and ownership boundary
            raise ScreenRecordingError("ADB could not reserve a recording path.") from error
        self._remote_file = remote_file
        try:
            command, effective_limit = self._recording_commands.command(
                self._connection_id,
                remote_file.remote_path,
                maximum_duration_seconds,
            )
            start = ScreenRecordingStart(
                requested_maximum_seconds=maximum_duration_seconds,
                effective_maximum_seconds=effective_limit,
                maximum_stop_seconds=maximum_stop_seconds,
                maximum_pull_seconds=maximum_pull_seconds,
            )
            process = self._process_starter(command)
        except Exception as error:  # noqa: BLE001 - process and ownership boundary
            failure = ScreenRecordingError("ADB could not start screen recording.")
            self._discard_failed_start(failure)
            raise failure from error
        self._process = process
        self._start = start
        self._can_start_after_failure = False
        try:
            self._wait_until_started(process, remote_file.remote_path)
        except ScreenRecordingError as error:
            self._discard_failed_start(error)
            raise
        return start

    def stop(self) -> ScreenRecordingCapture:
        process = self._process
        remote_file = self._remote_file
        start = self._start
        if remote_file is None or start is None:
            raise ScreenRecordingError("No screen-recording segment is active.")
        if process is not None and process.poll() is None:
            with suppress(Exception):
                self._recording_commands.stop(
                    self._connection_id,
                    remote_file.remote_path,
                )
        if process is not None:
            self._stop_process(process, start.maximum_stop_seconds)
            self._process = None
        directory = TemporaryDirectory(prefix="yewo-adb-recording-")
        path = Path(directory.name) / "capture.mp4"
        try:
            self._recover(
                remote_file.remote_path,
                path,
                start.maximum_pull_seconds,
            )
        except ScreenRecordingError:
            directory.cleanup()
            self._can_start_after_failure = False
            raise
        try:
            with path.open("rb") as stream:
                header = stream.read(12)
            if len(header) != 12 or header[4:8] != b"ftyp":
                raise ScreenRecordingError("ADB recovered an invalid MP4 recording.")
        except (OSError, ScreenRecordingError):
            directory.cleanup()
            self._remote_file = None
            self._start = None
            self._can_start_after_failure = True
            raise

        self._remote_file = None
        self._start = None
        self._can_start_after_failure = True

        def acknowledge() -> None:
            self._remote_files.release(remote_file)

        return ScreenRecordingCapture(
            path,
            start,
            acknowledge=acknowledge,
            cleanup=directory.cleanup,
        )

    def _wait_until_started(
        self,
        process: _RecordingProcess,
        remote_path: str,
    ) -> None:
        deadline = self._clock.monotonic() + _START_TIMEOUT_SECONDS
        while True:
            if self._file_commands.remote_file_size(self._connection_id, remote_path) is not None:
                return
            if process.poll() is not None:
                raise ScreenRecordingError("The Android screen recorder stopped during startup.")
            if self._clock.monotonic() >= deadline:
                raise ScreenRecordingError("The Android screen recorder did not create its file.")
            self._clock.sleep(_START_POLL_SECONDS)

    def _recover(
        self,
        remote_path: str,
        destination: Path,
        timeout_seconds: float,
    ) -> None:
        last_error: MediaTransferError | None = None
        for attempt in range(_RECOVERY_ATTEMPTS):
            try:
                self._transport.pull_to_path(
                    remote_path,
                    destination,
                    _MAXIMUM_RECOVERY_BYTES,
                    timeout_seconds=timeout_seconds,
                )
                return
            except MediaTransferError as error:
                last_error = error
                destination.unlink(missing_ok=True)
                if attempt + 1 < _RECOVERY_ATTEMPTS:
                    self._clock.sleep(_START_POLL_SECONDS)
        raise ScreenRecordingError(
            "ADB could not recover the remote screen recording."
        ) from last_error

    @staticmethod
    def _stop_process(process: _RecordingProcess, timeout_seconds: float) -> None:
        try:
            process.wait(timeout=timeout_seconds)
            return
        except (OSError, subprocess.SubprocessError):
            pass
        with suppress(Exception):
            process.terminate()
        try:
            process.wait(timeout=2)
            return
        except (OSError, subprocess.SubprocessError):
            pass
        with suppress(Exception):
            process.kill()
        try:
            process.wait(timeout=2)
        except (OSError, subprocess.SubprocessError) as error:
            raise ScreenRecordingError(
                "The Android screen recorder did not stop cleanly."
            ) from error

    def _discard_failed_start(self, original_error: ScreenRecordingError) -> None:
        process, self._process = self._process, None
        remote_file, self._remote_file = self._remote_file, None
        self._start = None
        self._can_start_after_failure = True
        if process is not None and process.poll() is None:
            if remote_file is not None:
                with suppress(Exception):
                    self._recording_commands.stop(
                        self._connection_id,
                        remote_file.remote_path,
                    )
            with suppress(Exception):
                process.terminate()
            try:
                process.wait(timeout=2)
            except (OSError, subprocess.SubprocessError):
                with suppress(Exception):
                    process.kill()
                with suppress(Exception):
                    process.wait(timeout=2)
        if remote_file is not None:
            try:
                self._remote_files.release(remote_file)
            except RemoteTemporaryFileError as cleanup_error:
                original_error.add_note(
                    f"The failed recording path also could not be removed: {cleanup_error}"
                )
