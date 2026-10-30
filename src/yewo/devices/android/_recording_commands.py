"""ADB commands that control one Android screen recorder."""

from __future__ import annotations

import re
import shlex
from typing import Protocol

from yewo.devices.android._validation import validate_remote_value
from yewo.devices.errors import DeviceCommandError


ANDROID_SCREEN_RECORD_LIMIT_SECONDS = 180


class _AdbRecordingBoundary(Protocol):
    @property
    def executable(self) -> str: ...

    def shell(
        self,
        connection_id: str,
        *args: str,
        purpose: str = "run an Android shell command",
        timeout_seconds: float = 30,
    ) -> object: ...


class AdbRecordingCommands:
    """Build and stop Android recordings through one ADB client."""

    def __init__(self, client: _AdbRecordingBoundary) -> None:
        self._client = client

    def command(
        self,
        connection_id: str,
        remote_path: str,
        maximum_duration_seconds: int,
    ) -> tuple[tuple[str, ...], int]:
        """Return the command and its platform-effective duration limit."""
        validate_remote_value(remote_path, "remote recording path")
        if maximum_duration_seconds < 1:
            raise DeviceCommandError("The recording time limit must be positive.")
        effective_limit = min(
            maximum_duration_seconds,
            ANDROID_SCREEN_RECORD_LIMIT_SECONDS,
        )
        return (
            (
                self._client.executable,
                "-s",
                connection_id,
                "shell",
                *(
                    shlex.quote(value)
                    for value in (
                        "screenrecord",
                        "--time-limit",
                        str(effective_limit),
                        remote_path,
                    )
                ),
            ),
            effective_limit,
        )

    def stop(self, connection_id: str, remote_path: str) -> None:
        """Interrupt only the recorder that writes the given remote path."""
        validate_remote_value(remote_path, "remote recording path")
        recording_pattern = rf"screenrecord.*{re.escape(remote_path)}"
        self._client.shell(
            connection_id,
            "pkill",
            "-INT",
            "-f",
            recording_pattern,
            purpose="stop a screen recording",
        )
