"""The devicectl command surface used by physical iOS devices."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import subprocess
import tempfile

from yewo.devices._commands import CommandRunner, run_command


def _installed_application_ids(payload: object) -> set[str] | None:
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    applications = result.get("apps")
    if not isinstance(applications, list):
        return None

    identifiers: set[str] = set()
    for application in applications:
        if not isinstance(application, dict):
            return None
        identifier = application.get("bundleIdentifier")
        if not isinstance(identifier, str) or not identifier.strip():
            return None
        identifiers.add(identifier.strip())
    return identifiers


class DevicectlClient:
    """Read physical iOS device state through structured devicectl output."""

    def __init__(
        self,
        runner: CommandRunner = run_command,
        *,
        xcrun_executable: str = "xcrun",
    ) -> None:
        self._runner = runner
        self._xcrun_executable = xcrun_executable

    def is_application_installed(
        self,
        connection_id: str,
        application_id: str,
    ) -> bool | None:
        """Return application presence, or no answer when evidence is unreliable."""
        with tempfile.TemporaryDirectory(prefix="yewo-devicectl-") as directory:
            output_path = Path(directory) / "applications.json"
            command = self._application_query(
                connection_id,
                application_id,
                output_path,
            )
            try:
                result = self._runner(command, timeout_seconds=30)
            except (OSError, subprocess.SubprocessError):
                return None
            if result.returncode:
                return None
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                return None

        identifiers = _installed_application_ids(payload)
        return None if identifiers is None else application_id in identifiers

    def _application_query(
        self,
        connection_id: str,
        application_id: str,
        output_path: Path,
    ) -> Sequence[str]:
        return (
            self._xcrun_executable,
            "devicectl",
            "device",
            "info",
            "apps",
            "--device",
            connection_id,
            "--bundle-id",
            application_id,
            "--json-output",
            str(output_path),
            "--quiet",
        )
