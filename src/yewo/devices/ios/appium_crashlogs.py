"""Normalize Appium iOS crash reports into diagnostic evidence."""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from typing import Protocol, cast

from yewo.devices.ios.diagnostics import (
    ios_report_contains_application,
    parse_ios_diagnostic_report,
)
from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import DeviceLogBatch
from yewo.devices.logs.records import DeviceLogCheckpointKind


_CRASH_LOG_TYPE = "crashlog"


class _AppiumCrashLogDriver(Protocol):
    @property
    def log_types(self) -> Sequence[str]: ...

    def get_log(self, log_type: str) -> Sequence[object]: ...


class IosAppiumCrashLogReader:
    """Own destructive Appium crash-log reads for one iOS application."""

    def __init__(self, driver: object, application_id: str) -> None:
        if not application_id:
            raise ValueError("The iOS diagnostic application ID cannot be empty.")
        self._driver = cast(_AppiumCrashLogDriver, driver)
        self._application_id = application_id
        self._available: bool | None = None
        self._pending_batch: DeviceLogBatch | None = None

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        if log_type != _CRASH_LOG_TYPE:
            return None
        if not self._is_available():
            return None
        baseline_entries = self._read_entries()
        batch = DeviceLogBatch(
            (),
            observed_item_count=len(baseline_entries),
            filtered_item_count=len(baseline_entries),
        )
        self._pending_batch = batch
        return batch

    def read(self, log_type: str) -> DeviceLogBatch:
        if log_type != _CRASH_LOG_TYPE:
            raise DeviceLogCaptureError("Appium does not own this diagnostic source.")
        if self._pending_batch is not None:
            return self._pending_batch
        entries = self._read_entries()
        reports = []
        filtered_item_count = 0
        unparsed_item_count = 0
        for entry in entries:
            try:
                content = _entry_message(entry)
                contains_application = ios_report_contains_application(
                    content,
                    self._application_id,
                )
            except DeviceLogCaptureError:
                unparsed_item_count += 1
                continue
            if not contains_application:
                filtered_item_count += 1
                continue
            try:
                digest = sha256(content.encode()).hexdigest()[:16]
                reports.append(
                    parse_ios_diagnostic_report(
                        f"appium-crashlog-{digest}.ips",
                        content,
                    )
                )
            except DeviceLogCaptureError:
                unparsed_item_count += 1
        batch = DeviceLogBatch(
            tuple(reports),
            observed_item_count=len(entries),
            filtered_item_count=filtered_item_count,
            unparsed_item_count=unparsed_item_count,
        )
        self._pending_batch = batch
        return batch

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        if log_type != _CRASH_LOG_TYPE:
            raise DeviceLogCaptureError("Appium does not own this diagnostic source.")
        return checkpoint_kind in {
            DeviceLogCheckpointKind.APPLICATION_SETUP,
            DeviceLogCheckpointKind.ACTION,
            DeviceLogCheckpointKind.RECOVERY,
            DeviceLogCheckpointKind.SCENARIO,
            DeviceLogCheckpointKind.SESSION_END,
        }

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        if log_type != _CRASH_LOG_TYPE:
            raise DeviceLogCaptureError("Appium does not own this diagnostic source.")
        if self._pending_batch is not batch:
            raise DeviceLogCaptureError("Appium received an unknown diagnostic batch.")
        self._pending_batch = None

    def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            values = self._driver.log_types
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise DeviceLogCaptureError("Appium did not list diagnostic sources.") from error
        if isinstance(values, str) or not isinstance(values, Sequence):
            raise DeviceLogCaptureError("Appium returned invalid diagnostic sources.")
        self._available = _CRASH_LOG_TYPE in values
        return self._available

    def _read_entries(self) -> Sequence[object]:
        try:
            entries = self._driver.get_log(_CRASH_LOG_TYPE)
        except Exception as error:  # noqa: BLE001 - external driver boundary
            raise DeviceLogCaptureError("Appium could not read iOS diagnostics.") from error
        if isinstance(entries, str) or not isinstance(entries, Sequence):
            raise DeviceLogCaptureError("Appium returned invalid iOS diagnostics.")
        return entries


def _entry_message(entry: object) -> str:
    if isinstance(entry, dict):
        message = entry.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(entry, str) and entry:
        return entry
    raise DeviceLogCaptureError("Appium returned an invalid iOS diagnostic entry.")
