"""Binary media transport through Appium."""

from __future__ import annotations

import base64
import binascii
from typing import Protocol, cast

from yewo.devices.media.errors import MediaTransferError


class MediaTransport(Protocol):
    """Push and pull bytes at one exact remote path."""

    def push(self, remote_path: str, content: bytes) -> None: ...

    def pull(self, remote_path: str, maximum_bytes: int) -> bytes: ...


class _AppiumMediaDriver(Protocol):
    def execute_script(self, script: str, arguments: dict[str, str]) -> object: ...


class AppiumMediaTransport:
    """Transfer media through the active platform driver."""

    def __init__(self, driver: object) -> None:
        self._driver = cast(_AppiumMediaDriver, driver)

    def push(self, remote_path: str, content: bytes) -> None:
        try:
            self._driver.execute_script(
                "mobile: pushFile",
                {
                    "remotePath": remote_path,
                    "payload": base64.b64encode(content).decode("ascii"),
                },
            )
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise MediaTransferError("Appium could not push the media file.") from error

    def pull(self, remote_path: str, maximum_bytes: int) -> bytes:
        try:
            encoded = self._driver.execute_script(
                "mobile: pullFile",
                {"remotePath": remote_path},
            )
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise MediaTransferError("Appium could not pull the media file.") from error
        if not isinstance(encoded, str) or not encoded.strip():
            raise MediaTransferError("Appium returned an empty media payload.")
        normalized = encoded.strip()
        maximum_encoded_bytes = 4 * ((maximum_bytes + 2) // 3)
        if len(normalized) > maximum_encoded_bytes:
            raise MediaTransferError("Appium media exceeds the configured byte limit.")
        try:
            content = base64.b64decode(normalized, validate=True)
        except (ValueError, binascii.Error) as error:
            raise MediaTransferError("Appium returned an invalid media payload.") from error
        if not content:
            raise MediaTransferError("Appium returned an empty media file.")
        return content
