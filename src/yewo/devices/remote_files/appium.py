"""Appium adapter for exact remote-file removal."""

from __future__ import annotations

from typing import Protocol, cast

from yewo.devices.remote_files.errors import RemoteTemporaryFileError


_MISSING_PATH_MARKERS = (
    "does not exist",
    "doesn't exist",
    "no such file",
    "object_not_found",
)


class RemoteFileRemover(Protocol):
    """Remove one exact remote file and tolerate an absent file."""

    def remove_file(self, remote_path: str) -> None: ...


class _AppiumFileDriver(Protocol):
    def execute_script(self, script: str, arguments: dict[str, str]) -> object: ...


class AppiumRemoteFileRemover:
    """Remove one remote file through the active platform driver."""

    def __init__(self, driver: object) -> None:
        self._driver = cast(_AppiumFileDriver, driver)

    def remove_file(self, remote_path: str) -> None:
        try:
            self._driver.execute_script(
                "mobile: deleteFile",
                {"remotePath": remote_path},
            )
        except Exception as error:  # noqa: BLE001 - external driver boundary
            normalized_error = str(error).casefold()
            if any(marker in normalized_error for marker in _MISSING_PATH_MARKERS):
                return
            raise RemoteTemporaryFileError(
                "Appium could not remove a run-owned remote temporary file."
            ) from error
