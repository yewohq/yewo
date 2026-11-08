"""Tests for structured physical iOS application queries."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import subprocess

from yewo.devices.ios import DevicectlClient


class DevicectlRunnerStub:
    """Write configured JSON to the devicectl output path."""

    def __init__(self, payload: object, *, returncode: int = 0) -> None:
        self._payload = payload
        self._returncode = returncode
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        arguments = tuple(command)
        self.calls.append((arguments, timeout_seconds))
        if self._returncode == 0:
            output_path = Path(arguments[arguments.index("--json-output") + 1])
            output_path.write_text(json.dumps(self._payload), encoding="utf-8")
        return subprocess.CompletedProcess(
            args=arguments,
            returncode=self._returncode,
            stdout="",
            stderr="",
        )


def test_devicectl_reports_matching_application() -> None:
    runner = DevicectlRunnerStub({"result": {"apps": [{"bundleIdentifier": "com.example.app"}]}})

    is_installed = DevicectlClient(runner).is_application_installed(
        "PHYSICAL-ID",
        "com.example.app",
    )

    assert is_installed is True
    command, timeout = runner.calls[0]
    assert command[:7] == (
        "xcrun",
        "devicectl",
        "device",
        "info",
        "apps",
        "--device",
        "PHYSICAL-ID",
    )
    assert command[7:9] == ("--bundle-id", "com.example.app")
    assert command[-1] == "--quiet"
    assert timeout == 30


def test_devicectl_reports_absent_application_from_recognized_empty_result() -> None:
    runner = DevicectlRunnerStub({"result": {"apps": []}})

    is_installed = DevicectlClient(runner).is_application_installed(
        "PHYSICAL-ID",
        "com.example.app",
    )

    assert is_installed is False


def test_devicectl_returns_unknown_for_unrecognized_schema() -> None:
    runner = DevicectlRunnerStub({"result": {"applications": []}})

    is_installed = DevicectlClient(runner).is_application_installed(
        "PHYSICAL-ID",
        "com.example.app",
    )

    assert is_installed is None


def test_devicectl_returns_unknown_when_command_fails() -> None:
    runner = DevicectlRunnerStub({}, returncode=1)

    is_installed = DevicectlClient(runner).is_application_installed(
        "PHYSICAL-ID",
        "com.example.app",
    )

    assert is_installed is None
