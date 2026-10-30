"""Tests for Appium media transport."""

import base64
from pathlib import Path
from typing import cast

import pytest

from yewo.devices.android._file_commands import AdbFileCommands
from yewo.devices.media import AppiumMediaTransport, MediaTransferError
from yewo.devices.media.android import AdbMediaTransport


class _DriverStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []
        self.pull_result: object = base64.b64encode(b"media").decode("ascii")
        self.error: Exception | None = None

    def execute_script(self, script: str, arguments: dict[str, str]) -> object:
        self.calls.append((script, arguments))
        if self.error is not None:
            raise self.error
        return self.pull_result


def test_appium_transport_pushes_and_pulls_binary_content() -> None:
    driver = _DriverStub()
    transport = AppiumMediaTransport(driver)

    transport.push("/remote/photo.png", b"media")
    content = transport.pull("/remote/photo.png", 5)

    assert content == b"media"
    assert driver.calls == [
        (
            "mobile: pushFile",
            {
                "remotePath": "/remote/photo.png",
                "payload": base64.b64encode(b"media").decode("ascii"),
            },
        ),
        (
            "mobile: pullFile",
            {"remotePath": "/remote/photo.png"},
        ),
    ]


@pytest.mark.parametrize("payload", (None, "", "not base64", 4))
def test_appium_transport_rejects_invalid_pull_payloads(payload: object) -> None:
    driver = _DriverStub()
    driver.pull_result = payload

    with pytest.raises(MediaTransferError, match="payload|file"):
        AppiumMediaTransport(driver).pull("/remote/photo.png", 100)


def test_appium_transport_hides_driver_failure_details() -> None:
    driver = _DriverStub()
    driver.error = RuntimeError("private driver detail")

    with pytest.raises(MediaTransferError, match="push") as captured:
        AppiumMediaTransport(driver).push("/remote/photo.png", b"media")

    assert "private driver detail" not in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)


def test_appium_transport_rejects_oversized_payload_before_decoding() -> None:
    driver = _DriverStub()
    driver.pull_result = base64.b64encode(b"oversized").decode("ascii")

    with pytest.raises(MediaTransferError, match="byte limit"):
        AppiumMediaTransport(driver).pull("/remote/photo.png", 4)


def test_adb_transport_uses_host_temporary_files_only_during_transfer() -> None:
    class _AdbStub:
        pushed_content = b""

        def push_file(self, connection_id: str, local_path: str, remote_path: str) -> None:
            assert connection_id == "emulator-5554"
            assert remote_path == "/remote/photo.png"
            self.pushed_content = Path(local_path).read_bytes()

        def pull_file(
            self,
            connection_id: str,
            remote_path: str,
            local_path: str,
            *,
            timeout_seconds: float,
        ) -> None:
            assert timeout_seconds == 120
            assert connection_id == "emulator-5554"
            assert remote_path == "/remote/photo.png"
            Path(local_path).write_bytes(self.pushed_content)

    adb = _AdbStub()
    transport = AdbMediaTransport(cast(AdbFileCommands, adb), "emulator-5554")

    transport.push("/remote/photo.png", b"media")

    assert transport.pull("/remote/photo.png", 5) == b"media"
