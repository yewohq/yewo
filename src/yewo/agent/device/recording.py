"""Recording the screen for one session, on either platform.

Everything about the session video lives here: starting it, saving it to the
host, and removing what it left on the device.

A video exists only once the recorder has stopped, so the run is cut into
segments at every scenario boundary. A crash then costs the segment in
flight instead of the whole session.

Android has two recorders, chosen by USE_MEDIA_PROJECTION. screenrecord is a
shell binary that dies with the run; Media Projection names its file and can
report whether it started.
"""

import base64
import logging
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping, Optional, Protocol

from selenium.common.exceptions import WebDriverException

from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.files import MEDIA_PROJECTION_DIR, SCREENRECORD_DIR, DeviceFiles
from yewo.agent.device.platform import Platform
from yewo.agent.device.recording_redaction import RecordingMask, redact_recording
from yewo.utils.secrets import ProtectedRegion

logger = logging.getLogger(__name__)

START_MEDIA_PROJECTION = "mobile: startMediaProjectionRecording"
STOP_MEDIA_PROJECTION = "mobile: stopMediaProjectionRecording"
MEDIA_PROJECTION_RUNNING = "mobile: isMediaProjectionRecordingRunning"

RECORDING_TIME_LIMIT = 1800
"""Seconds a segment may run before the recorder stops on its own.

Appium defaults to 180 and says nothing when it reaches it, so every run over
three minutes was being truncated. 1800 is the longest uiautomator2 takes.
"""

START_OPTIONS: tuple[Mapping[str, Any], ...] = (
    {"timeLimit": RECORDING_TIME_LIMIT},
    {"timeLimit": RECORDING_TIME_LIMIT},
    {},
)
"""What each attempt at starting a recording asks for.

The first two are the same request twice, because Appium allows screenrecord
a fixed 5s to appear and a loaded emulator loses that race. The last asks for
nothing, so a driver that refuses the longer limit still records: a video cut
at three minutes beats no video at all.
"""

SCREENRECORD_FLUSH = 1.0
"""Seconds to let the device recorder close its file after it is interrupted."""

USE_MEDIA_PROJECTION = False
"""Whether Android records with Media Projection instead of screenrecord.

Media Projection names its file and reports whether it started, but Android
shows a screen-sharing indicator while it runs, which puts system UI on the
screen under test and outlives a killed run.
"""


class RecordingTarget(Protocol):
    """What the recorder needs from the session's device control.

    Named here so the recorder can ask for a driver and a serial without
    reaching for attributes it has not declared.
    """

    driver: Any
    serial: Optional[str]
    screen_width: int
    screen_height: int


class DeviceRecording:
    """The session video, for a DeviceSession to mix in.

    Expects the session to provide `platform`, `device_control`,
    `device_name`, `recordings_dir` and `session_id`.
    """

    platform: Platform
    device_control: Optional[RecordingTarget]
    device_name: str
    recordings_dir: Path
    session_id: str

    is_recording: bool = False
    _remote_path: Optional[str] = None
    _media_projection: bool = False
    _recording_started_at: Optional[float] = None
    _recording_masks: list[RecordingMask] = []
    redact_recording: bool = False

    @property
    def _driver(self) -> Any:
        """The Appium driver. Untyped, because the recording calls are vendor APIs."""
        return self.device_control.driver if self.device_control else None

    def _serial(self) -> Optional[str]:
        serial = self.device_control.serial if self.device_control else None
        return serial or AdbControl.resolve_emulator_serial(self.device_name)

    def start_recording(self) -> None:
        """Begins a segment, or says that this run will not have one.

        The video is an artifact, not evidence a scenario rests on, so failing
        to start one must not end a run that could otherwise report results.
        """
        # Per segment, because the recorder that starts this one decides how
        # it is stopped. A fallback inheriting the last segment's answer is
        # stopped through a recorder that is not running it.
        self._media_projection = False
        self._recording_started_at = None
        self._recording_masks = []
        for attempt, options in enumerate(START_OPTIONS, start=1):
            try:
                if self._start_once(options):
                    self.is_recording = True
                    # The device starts somewhere inside the blocking call. Measuring
                    # from its return masks early rather than exposing a late frame.
                    self._recording_started_at = time.monotonic()
                    return
                logger.warning("The device reports a recording already running.")
                return
            except WebDriverException as exc:
                logger.warning(
                    "Screen recording did not start (attempt %d of %d): %s",
                    attempt,
                    len(START_OPTIONS),
                    exc,
                )
        logger.warning("Continuing without a video of this run.")

    def start_recording_mask(
        self,
        region: ProtectedRegion,
        source_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """Cover a typed secret from now until the current segment ends."""
        if not self.redact_recording or not self.is_recording:
            return
        if source_size is None:
            if self.device_control is None:
                return
            source_size = (
                self.device_control.screen_width,
                self.device_control.screen_height,
            )
        started_at = self._recording_started_at
        if started_at is None or source_size[0] <= 0 or source_size[1] <= 0:
            return
        self._recording_masks.append(
            RecordingMask(
                region=region,
                source_size=source_size,
                starts_at_seconds=max(0.0, time.monotonic() - started_at),
            )
        )

    def _start_once(self, options: Mapping[str, Any]) -> bool:
        """Starts one recording, reporting whether it is now running."""
        if self.platform == Platform.ANDROID and USE_MEDIA_PROJECTION:
            return self._start_media_projection(options)
        self._start_screenrecord(options)
        return True

    def _start_screenrecord(self, options: Mapping[str, Any]) -> None:
        """Starts the driver's recorder, then works out what it named the file.

        Appium chooses the name and never reports it, so a run whose driver
        dies has no way to ask for the video it recorded. Appium waits for the
        file to appear before it returns, so listing the directory on either
        side of the call names it.
        """
        before = self._device_recordings()
        self._driver.start_recording_screen(**options)
        self._remote_path = self._added_recording(before)
        logger.info(
            "Recording this run with screenrecord: %s", self._remote_path or "unnamed"
        )

    def _device_recordings(self) -> set[str]:
        """The names of the videos sitting on the device right now."""
        serial = self._serial() if self.platform == Platform.ANDROID else None
        if not serial:
            return set()
        return {
            name
            for name in DeviceFiles.list_directory(serial, SCREENRECORD_DIR)
            if name.endswith(".mp4")
        }

    def _added_recording(self, before: set[str]) -> Optional[str]:
        """The one video that appeared since `before` was taken, if it is the only one.

        Two new files leave no way to tell which is this recording, and a
        wrong answer here either saves somebody else's video or deletes one
        that was never ours. Naming nothing costs only the fallback.
        """
        added = self._device_recordings() - before
        if len(added) != 1:
            return None
        return f"{SCREENRECORD_DIR}/{added.pop()}"

    def _start_media_projection(self, options: Mapping[str, Any]) -> bool:
        """Starts the Media Projection recorder, which can say whether it did."""
        name = f"yewo_{self.session_id}.mp4"
        try:
            started = self._driver.execute_script(
                START_MEDIA_PROJECTION, {"filename": name}
            )
        except WebDriverException as exc:
            logger.info("Media Projection is unavailable, using screenrecord: %s", exc)
            self._start_screenrecord(options)
            return True

        if not started:
            return False
        self._media_projection = True
        self._remote_path = f"{MEDIA_PROJECTION_DIR}/{name}"
        running = bool(self._driver.execute_script(MEDIA_PROJECTION_RUNNING))
        if running:
            logger.info("Recording this run with Media Projection: %s", self._remote_path)
        return running

    def _free_path(self) -> Path:
        """Where to write the next segment without overwriting an earlier one.

        Segments are numbered rather than named once. The run is cut at every
        scenario boundary, and session recovery starts a new object on the
        same session id. Existence decides the number, so a replacement
        session carries on from where the failed one stopped.
        """
        segment = 1
        while True:
            path = self.recordings_dir / f"session_{self.session_id}_{segment:03d}.mp4"
            if not path.exists():
                return path
            segment += 1

    def save_recording(self) -> Optional[Path]:
        """Writes the current segment to the host, then clears it from the device."""
        if not self.is_recording:
            return None
        self.is_recording = False
        path = self._free_path()
        try:
            raw = (
                self._driver.execute_script(STOP_MEDIA_PROJECTION)
                if self._media_projection
                else self._driver.stop_recording_screen()
            )
        except WebDriverException as exc:
            logger.warning("Failed to stop the screen recording: %s", exc)
            return self._pull_recording(path)

        if not raw:
            logger.warning("The screen recording came back empty.")
            return None
        if not self._write_recording(base64.b64decode(raw), path):
            self._discard_remote()
            return None
        logger.info("Session recording saved to %s", path)
        self._discard_remote()
        return path

    def _write_recording(self, raw: bytes, path: Path) -> bool:
        """Write raw video directly, or publish only its successfully masked copy."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.redact_recording:
            path.write_bytes(raw)
            return True
        with TemporaryDirectory(prefix=".redacting-", dir=path.parent) as directory:
            raw_path = Path(directory) / "raw.mp4"
            raw_path.write_bytes(raw)
            return self._publish_redacted_recording(raw_path, path)

    def _publish_redacted_recording(self, raw_path: Path, path: Path) -> bool:
        """Move only a successfully masked temporary video into run artifacts."""
        safe_path = raw_path.with_name("safe.mp4")
        if not redact_recording(raw_path, safe_path, tuple(self._recording_masks)):
            return False
        os.replace(safe_path, path)
        return True

    def checkpoint_recording(self) -> Optional[Path]:
        """Closes the segment in flight and begins the next one.

        Called at a scenario boundary. The video of a run that dies is
        whatever was already stopped, so cutting here is what decides how
        much of a crashed run can still be watched.
        """
        if not self.is_recording:
            return None
        path = self.save_recording()
        self.start_recording()
        return path

    def _pull_recording(self, path: Path) -> Optional[Path]:
        """Copies the video off the device when the driver can no longer be asked.

        A session that stops responding is the one whose video is worth most.
        It is also the one the driver cannot hand over. Both recorders write
        to a path this session knows by now, so adb can fetch it on its own.

        screenrecord is interrupted first. It writes the index that makes an
        MP4 playable only when it stops, so the file of a recorder still
        running will not open.
        """
        serial = self._serial()
        if not self._remote_path or not serial:
            return None
        if not self._media_projection:
            DeviceFiles.stop_screenrecord(serial, self._remote_path)
            time.sleep(SCREENRECORD_FLUSH)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.redact_recording:
            with TemporaryDirectory(prefix=".redacting-", dir=path.parent) as directory:
                raw_path = Path(directory) / "raw.mp4"
                if not DeviceFiles.pull_file(serial, self._remote_path, str(raw_path)):
                    return None
                if not self._publish_redacted_recording(raw_path, path):
                    self._discard_remote(serial)
                    return None
        elif not DeviceFiles.pull_file(serial, self._remote_path, str(path)):
            return None
        logger.info("Recovered the session recording from the device: %s", path)
        self._discard_remote(serial)
        return path

    def _discard_remote(self, serial: Optional[str] = None) -> None:
        """Removes the segment from the device, now that the host has a copy.

        Only ever called once the video is on the host. A recording that
        could not be saved stays where it is, for the next session's sweep to
        remove or for a person to fetch. It is the only copy there is.
        """
        remote, self._remote_path = self._remote_path, None
        if self.platform != Platform.ANDROID or not remote:
            return
        serial = serial or self._serial()
        if serial:
            DeviceFiles.remove_file(serial, remote)

    def discard_stale_recordings(self) -> None:
        """Removes what earlier runs left on the device, before this one records.

        It sweeps on the way in. A sweep at teardown cannot tell this run's
        video from an abandoned one, and the run worth watching is the one
        that crashed before it saved anything.
        """
        if self.platform != Platform.ANDROID:
            return
        serial = self._serial()
        if serial:
            DeviceFiles.remove_stale_recordings(serial)
