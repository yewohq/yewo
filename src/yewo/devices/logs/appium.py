"""Normalize Appium's incremental device-log boundary."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol, cast

from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import DeviceLogBatch, DeviceLogTextEntry
from yewo.devices.logs.records import DeviceLogCheckpointKind


DEFAULT_APPIUM_LOG_BUFFER_ENTRIES = 10_000


class _AppiumLogDriver(Protocol):
    @property
    def log_types(self) -> Sequence[str]: ...

    def get_log(self, log_type: str) -> Sequence[object]: ...


class AppiumDeviceLogReader:
    """Read and normalize the logs consumed by one Appium session."""

    def __init__(
        self,
        driver: object,
        *,
        maximum_buffer_entries: int = DEFAULT_APPIUM_LOG_BUFFER_ENTRIES,
    ) -> None:
        if maximum_buffer_entries < 1:
            raise ValueError("The Appium log-buffer capacity must be positive.")
        self._driver = cast(_AppiumLogDriver, driver)
        self._maximum_buffer_entries = maximum_buffer_entries
        self._available_log_types: frozenset[str] | None = None
        self._pending_batches: dict[str, DeviceLogBatch] = {}

    def available_log_types(self) -> frozenset[str]:
        try:
            values = self._driver.log_types
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise DeviceLogCaptureError("Appium did not list device-log sources.") from error
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise DeviceLogCaptureError("Appium returned invalid device-log sources.")
        if not all(isinstance(value, str) and value for value in values):
            raise DeviceLogCaptureError("Appium returned invalid device-log sources.")
        available_log_types = frozenset(values)
        self._available_log_types = available_log_types
        return available_log_types

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        known_log_types = self._available_log_types
        if known_log_types is None:
            known_log_types = self.available_log_types()
        return self.read(log_type) if log_type in known_log_types else None

    def read(self, log_type: str) -> DeviceLogBatch:
        pending_batch = self._pending_batches.get(log_type)
        if pending_batch is not None:
            return pending_batch
        try:
            entries = self._driver.get_log(log_type)
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise DeviceLogCaptureError("Appium could not read the device log.") from error
        if isinstance(entries, str) or not isinstance(entries, Sequence):
            raise DeviceLogCaptureError("Appium returned invalid device-log entries.")
        normalized_entries = tuple(
            DeviceLogTextEntry(_format_appium_entry(entry)) for entry in entries
        )
        batch = DeviceLogBatch(
            items=normalized_entries,
            observed_item_count=len(normalized_entries),
            may_have_omitted_earlier_items=(
                len(normalized_entries) >= self._maximum_buffer_entries
            ),
        )
        self._pending_batches[log_type] = batch
        return batch

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        del log_type, checkpoint_kind
        return True

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        if self._pending_batches.get(log_type) is not batch:
            raise DeviceLogCaptureError("Appium received an unknown device-log batch.")
        del self._pending_batches[log_type]


def _format_appium_entry(entry: object) -> str:
    if not isinstance(entry, dict):
        return str(entry)
    timestamp = _format_timestamp(entry.get("timestamp"))
    level = entry.get("level")
    message = entry.get("message")
    parts = (
        timestamp,
        level if isinstance(level, str) else "",
        message if isinstance(message, str) else str(message or ""),
    )
    return " ".join(part for part in parts if part)


def _format_timestamp(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return ""
    try:
        return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat(timespec="milliseconds")
    except (OverflowError, OSError, ValueError):
        return ""
