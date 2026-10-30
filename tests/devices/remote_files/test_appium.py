"""Tests for Appium remote-file cleanup."""

import pytest

from yewo.devices.remote_files import (
    AppiumRemoteFileRemover,
    RemoteTemporaryFileError,
)


class _DriverStub:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, str]]] = []

    def execute_script(self, script: str, arguments: dict[str, str]) -> object:
        self.calls.append((script, arguments))
        if self.error is not None:
            raise self.error
        return True


def test_remover_dispatches_one_exact_path() -> None:
    driver = _DriverStub()

    AppiumRemoteFileRemover(driver).remove_file("/remote/yewo/file.png")

    assert driver.calls == [
        (
            "mobile: deleteFile",
            {"remotePath": "/remote/yewo/file.png"},
        )
    ]


@pytest.mark.parametrize(
    "message",
    (
        "The remote path does not exist",
        "No such file",
        "OBJECT_NOT_FOUND",
    ),
)
def test_remover_treats_an_absent_reserved_file_as_clean(message: str) -> None:
    AppiumRemoteFileRemover(_DriverStub(RuntimeError(message))).remove_file("/remote/yewo/file.png")


def test_remover_hides_driver_details_and_preserves_the_cause() -> None:
    remover = AppiumRemoteFileRemover(_DriverStub(RuntimeError("private driver detail")))

    with pytest.raises(RemoteTemporaryFileError, match="run-owned") as captured:
        remover.remove_file("/remote/yewo/file.png")

    assert "private driver detail" not in str(captured.value)
    assert isinstance(captured.value.__cause__, RuntimeError)
