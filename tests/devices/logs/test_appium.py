"""Tests for the Appium device-log reader boundary."""

from typing import Any

import pytest

from yewo.devices.logs import (
    AppiumDeviceLogReader,
    DeviceLogCaptureError,
    DeviceLogCheckpointKind,
    DeviceLogTextEntry,
)


class _DriverStub:
    def __init__(self) -> None:
        self.log_types: Any = ["logcat"]
        self.entries: Any = [{"timestamp": 0, "level": "INFO", "message": "ready"}]
        self.read_count = 0

    def get_log(self, log_type: str) -> Any:
        assert log_type in self.log_types
        self.read_count += 1
        return self.entries


class _FailingDriverStub:
    @property
    def log_types(self) -> list[str]:
        raise RuntimeError("driver secret")

    def get_log(self, log_type: str) -> list[object]:
        raise AssertionError(f"Unexpected read of {log_type}")


def test_reader_normalizes_available_sources_and_entries() -> None:
    reader = AppiumDeviceLogReader(_DriverStub())

    batch = reader.begin("logcat")

    assert batch is not None
    assert batch.items == (
        DeviceLogTextEntry("1970-01-01T00:00:00.000+00:00 INFO ready"),
    )
    assert batch.observed_item_count == 1
    assert not batch.may_have_omitted_earlier_items
    assert reader.begin("safariConsole") is None


def test_reader_marks_a_full_driver_buffer_as_possibly_incomplete() -> None:
    driver = _DriverStub()
    driver.entries = ["first", "second"]
    reader = AppiumDeviceLogReader(driver, maximum_buffer_entries=2)

    batch = reader.read("logcat")

    assert batch.may_have_omitted_earlier_items


def test_reader_replays_unacknowledged_destructive_batch() -> None:
    driver = _DriverStub()
    reader = AppiumDeviceLogReader(driver)
    first = reader.read("logcat")
    driver.entries = ["newer"]

    replayed = reader.read("logcat")

    assert replayed is first
    assert driver.read_count == 1
    reader.acknowledge("logcat", first)
    assert reader.read("logcat").items == (DeviceLogTextEntry("newer"),)
    assert driver.read_count == 2


def test_reader_rejects_acknowledgement_for_an_unknown_batch() -> None:
    reader = AppiumDeviceLogReader(_DriverStub())
    unknown = reader.read("logcat")

    with pytest.raises(DeviceLogCaptureError, match="unknown device-log batch"):
        reader.acknowledge("syslog", unknown)


def test_reader_accepts_vendor_defined_camel_case_sources() -> None:
    driver = _DriverStub()
    driver.log_types = ["safariConsole"]
    driver.entries = []
    reader = AppiumDeviceLogReader(driver)

    assert reader.begin("safariConsole") is not None


def test_appium_sources_capture_every_checkpoint() -> None:
    reader = AppiumDeviceLogReader(_DriverStub())

    assert reader.captures_checkpoint("logcat", DeviceLogCheckpointKind.ACTION)


def test_reader_normalizes_driver_failure_without_leaking_its_message() -> None:
    with pytest.raises(DeviceLogCaptureError, match="did not list") as captured:
        AppiumDeviceLogReader(_FailingDriverStub()).begin("logcat")

    assert "driver secret" not in str(captured.value)


@pytest.mark.parametrize("value", ["logcat", [""], [1]])
def test_reader_rejects_invalid_source_lists(value: object) -> None:
    driver = _DriverStub()
    driver.log_types = value

    with pytest.raises(DeviceLogCaptureError, match="invalid device-log sources"):
        AppiumDeviceLogReader(driver).begin("logcat")


@pytest.mark.parametrize("value", ["one line", {"message": "ready"}])
def test_reader_rejects_invalid_entry_collections(value: object) -> None:
    driver = _DriverStub()
    driver.entries = value

    with pytest.raises(DeviceLogCaptureError, match="invalid device-log entries"):
        AppiumDeviceLogReader(driver).read("logcat")
