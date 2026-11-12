"""Persistence failure and device-log adapters for session tests."""

from pathlib import Path

from yewo.devices.logs import (
    DeviceLogArtifactReference,
    DeviceLogCaptureError,
    DeviceLogDrainReceipt,
    DeviceLogStore,
)
from yewo.devices.logs.appium import AppiumDeviceLogReader


class FailFinalLogRollbackStore(DeviceLogStore):
    """Fail final receipt persistence and its artifact rollback."""

    def __init__(self, root: Path) -> None:
        super().__init__(root, identifier_factory=lambda: "artifact_1")
        self._receipt_count = 0

    def persist_receipt(self, receipt: DeviceLogDrainReceipt) -> Path:
        self._receipt_count += 1
        if self._receipt_count == 2:
            raise DeviceLogCaptureError("receipt failed")
        return super().persist_receipt(receipt)

    def discard(self, reference: DeviceLogArtifactReference) -> None:
        del reference
        raise DeviceLogCaptureError("rollback failed")


def appium_log_reader(driver: object, device: object, application: object) -> AppiumDeviceLogReader:
    """Return the in-process reader used by session orchestration tests."""
    del device, application
    return AppiumDeviceLogReader(driver)
