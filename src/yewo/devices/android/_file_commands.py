"""ADB commands for files stored on an Android device."""

from __future__ import annotations

import subprocess
from typing import Protocol

from yewo.devices.android._validation import validate_remote_value


class _AdbCommandBoundary(Protocol):
    def adb(
        self,
        connection_id: str,
        *args: str,
        purpose: str = "run an ADB command",
        timeout_seconds: float = 30,
    ) -> subprocess.CompletedProcess[str]: ...

    def shell(
        self,
        connection_id: str,
        *args: str,
        purpose: str = "run an Android shell command",
        timeout_seconds: float = 30,
    ) -> subprocess.CompletedProcess[str]: ...

    def ask(
        self,
        connection_id: str,
        *args: str,
        timeout_seconds: float = 20,
    ) -> str | None: ...


class AdbFileCommands:
    """Add file transfer and media indexing to the ADB command boundary."""

    def __init__(self, client: _AdbCommandBoundary) -> None:
        self._client = client

    def push_file(self, connection_id: str, local_path: str, remote_path: str) -> None:
        validate_remote_value(remote_path, "remote file path")
        self._client.adb(
            connection_id,
            "push",
            local_path,
            remote_path,
            purpose="push a file",
        )

    def pull_file(
        self,
        connection_id: str,
        remote_path: str,
        local_path: str,
        *,
        timeout_seconds: float = 120,
    ) -> None:
        """Copy one exact remote file to the host."""
        validate_remote_value(remote_path, "remote file path")
        self._client.adb(
            connection_id,
            "pull",
            remote_path,
            local_path,
            purpose="pull a file",
            timeout_seconds=timeout_seconds,
        )

    def remove_file(self, connection_id: str, remote_path: str) -> None:
        """Remove one exact remote file and tolerate an absent file."""
        validate_remote_value(remote_path, "remote file path")
        self._client.shell(
            connection_id,
            "rm",
            "-f",
            remote_path,
            purpose="remove a remote file",
        )

    def remote_file_size(self, connection_id: str, remote_path: str) -> int | None:
        """Return one remote file size, or no answer when it is unavailable."""
        validate_remote_value(remote_path, "remote file path")
        answer = self._client.ask(connection_id, "stat", "-c", "%s", remote_path)
        first_line = (answer or "").splitlines()[:1]
        if not first_line or not first_line[0].isdigit():
            return None
        return int(first_line[0])

    def scan_media_file(self, connection_id: str, remote_path: str) -> None:
        validate_remote_value(remote_path, "remote media path")
        self._client.shell(
            connection_id,
            "am",
            "broadcast",
            "-a",
            "android.intent.action.MEDIA_SCANNER_SCAN_FILE",
            "-d",
            f"file://{remote_path}",
            purpose="scan a media file",
        )
