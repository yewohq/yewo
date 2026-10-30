"""Tests for source-specific device-log routing."""

import pytest

from yewo.devices.logs import (
    DeviceLogBatch,
    DeviceLogCaptureError,
    DeviceLogCheckpointKind,
    DeviceLogTextEntry,
    RoutedDeviceLogReader,
)


class _ReaderStub:
    def __init__(self, supported: frozenset[str]) -> None:
        self.supported = supported
        self.begins: list[str] = []
        self.reads: list[str] = []
        self.acknowledged: list[str] = []

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        self.begins.append(log_type)
        return self._batch(log_type) if log_type in self.supported else None

    def read(self, log_type: str) -> DeviceLogBatch:
        self.reads.append(log_type)
        return self._batch(log_type)

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        del checkpoint_kind
        return log_type in self.supported

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        del batch
        self.acknowledged.append(log_type)

    @staticmethod
    def _batch(log_type: str) -> DeviceLogBatch:
        return DeviceLogBatch((DeviceLogTextEntry(log_type),), 1)


class _FailingReaderStub(_ReaderStub):
    def begin(self, log_type: str) -> DeviceLogBatch | None:
        raise DeviceLogCaptureError(f"failed {log_type}")


def test_router_delegates_each_source_to_its_owner() -> None:
    appium = _ReaderStub(frozenset({"syslog"}))
    core_device = _ReaderStub(frozenset({"crashlog"}))
    reader = RoutedDeviceLogReader({"syslog": appium, "crashlog": core_device})

    syslog_batch = reader.begin("syslog")
    crash_batch = reader.read("crashlog")
    reader.acknowledge("crashlog", crash_batch)

    assert syslog_batch is not None
    assert appium.begins == ["syslog"]
    assert core_device.reads == ["crashlog"]
    assert core_device.acknowledged == ["crashlog"]


def test_one_source_failure_does_not_prevent_another_source_read() -> None:
    reader = RoutedDeviceLogReader(
        {
            "syslog": _FailingReaderStub(frozenset({"syslog"})),
            "crashlog": _ReaderStub(frozenset({"crashlog"})),
        }
    )

    with pytest.raises(DeviceLogCaptureError, match="failed syslog"):
        reader.begin("syslog")

    assert reader.begin("crashlog") is not None


def test_router_rejects_an_unowned_source() -> None:
    router = RoutedDeviceLogReader({"syslog": _ReaderStub(frozenset({"syslog"}))})

    with pytest.raises(DeviceLogCaptureError, match="No reader owns"):
        router.read("crashlog")
