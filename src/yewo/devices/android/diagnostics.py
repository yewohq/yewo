"""Read Android application exit history through ADB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, tzinfo
from hashlib import sha256
import re
from typing import Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from yewo.devices.diagnostics import ApplicationFailureKind
from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import ApplicationDiagnosticReport, DeviceLogBatch
from yewo.devices.logs.records import DeviceLogCheckpointKind


_EXIT_INFO_LOG_TYPE = "exitinfo"
_MINIMUM_EXIT_INFO_API_LEVEL = 30
_ENTRY_START = re.compile(r"(?m)^\s*ApplicationExitInfo\s+#\d+:\s*$")
_EXIT_INFO_HEADER = "ACTIVITY MANAGER PROCESS EXIT INFO (dumpsys activity exit-info)"


@dataclass(frozen=True)
class _ParsedExitEntry:
    event_id: str
    report: ApplicationDiagnosticReport | None


class _AdbDiagnosticBoundary(Protocol):
    def ask(
        self,
        connection_id: str,
        *args: str,
        timeout_seconds: float = 20,
    ) -> str | None: ...


class AdbApplicationExitReader:
    """Diff classified process exits without clearing device-wide history."""

    def __init__(
        self,
        client: _AdbDiagnosticBoundary,
        connection_id: str,
        application_id: str,
    ) -> None:
        if not connection_id or not application_id:
            raise ValueError("Android diagnostic identity cannot be empty.")
        self._client = client
        self._connection_id = connection_id
        self._application_id = application_id
        self._known_event_ids: frozenset[str] | None = None
        self._pending_batch: tuple[DeviceLogBatch, frozenset[str]] | None = None

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        if log_type != _EXIT_INFO_LOG_TYPE:
            return None
        self._require_acknowledged()
        if not self._supports_exit_info():
            return None
        entries = self._read_entries()
        event_ids = frozenset(entry.event_id for entry in entries)
        unparsed_item_count = sum(entry.report is None for entry in entries)
        batch = DeviceLogBatch(
            (),
            observed_item_count=len(entries),
            filtered_item_count=len(entries) - unparsed_item_count,
            unparsed_item_count=unparsed_item_count,
        )
        self._pending_batch = (batch, event_ids)
        return batch

    def read(self, log_type: str) -> DeviceLogBatch:
        if log_type != _EXIT_INFO_LOG_TYPE:
            raise DeviceLogCaptureError("ADB does not own this diagnostic source.")
        if self._pending_batch is not None:
            return self._pending_batch[0]
        if self._known_event_ids is None:
            raise DeviceLogCaptureError("The Android diagnostic boundary is not active.")
        entries = self._read_entries()
        new_entries = tuple(
            entry for entry in entries if entry.event_id not in self._known_event_ids
        )
        event_ids = frozenset(entry.event_id for entry in entries)
        batch = DeviceLogBatch(
            tuple(entry.report for entry in new_entries if entry.report is not None),
            observed_item_count=len(new_entries),
            unparsed_item_count=sum(entry.report is None for entry in new_entries),
        )
        self._pending_batch = (batch, event_ids)
        return batch

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        if log_type != _EXIT_INFO_LOG_TYPE:
            raise DeviceLogCaptureError("ADB does not own this diagnostic source.")
        return checkpoint_kind in {
            DeviceLogCheckpointKind.SCENARIO,
            DeviceLogCheckpointKind.SESSION_END,
        }

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        if log_type != _EXIT_INFO_LOG_TYPE:
            raise DeviceLogCaptureError("ADB does not own this diagnostic source.")
        if self._pending_batch is None or self._pending_batch[0] is not batch:
            raise DeviceLogCaptureError("ADB received an unknown diagnostic batch.")
        self._known_event_ids = self._pending_batch[1]
        self._pending_batch = None

    def _supports_exit_info(self) -> bool:
        output = self._client.ask(
            self._connection_id,
            "getprop",
            "ro.build.version.sdk",
            timeout_seconds=10,
        )
        if output is None:
            raise DeviceLogCaptureError("ADB could not read the Android API level.")
        first_line = output.strip().splitlines()[:1]
        if not first_line or not first_line[0].isdigit():
            raise DeviceLogCaptureError("ADB returned an invalid Android API level.")
        return int(first_line[0]) >= _MINIMUM_EXIT_INFO_API_LEVEL

    def _read_entries(self) -> tuple[_ParsedExitEntry, ...]:
        output = self._client.ask(
            self._connection_id,
            "dumpsys",
            "activity",
            "exit-info",
            self._application_id,
            timeout_seconds=30,
        )
        if output is None:
            raise DeviceLogCaptureError("ADB could not read application exit history.")
        return _parse_android_exit_entries(output, self._read_device_timezone())

    def _read_device_timezone(self) -> tzinfo | None:
        output = self._client.ask(
            self._connection_id,
            "getprop",
            "persist.sys.timezone",
            timeout_seconds=10,
        )
        timezone_name = (output or "").strip()
        if not timezone_name:
            return None
        try:
            return ZoneInfo(timezone_name)
        except ZoneInfoNotFoundError:
            return None

    def _require_acknowledged(self) -> None:
        if self._pending_batch is not None:
            raise DeviceLogCaptureError("The previous Android diagnostic read is not durable.")


def parse_android_exit_info(
    output: str,
    device_timezone: tzinfo | None = None,
) -> tuple[ApplicationDiagnosticReport, ...]:
    """Parse supported `dumpsys activity exit-info` entries."""
    return tuple(
        entry.report
        for entry in _parse_android_exit_entries(output, device_timezone)
        if entry.report is not None
    )


def _parse_android_exit_entries(
    output: str,
    device_timezone: tzinfo | None,
) -> tuple[_ParsedExitEntry, ...]:
    if _EXIT_INFO_HEADER not in output:
        raise DeviceLogCaptureError("ADB returned an unknown application exit-history shape.")
    starts = tuple(_ENTRY_START.finditer(output))
    entries: list[_ParsedExitEntry] = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(output)
        entry = output[start.start() : end].strip()
        try:
            report = _parse_android_exit_entry(entry, device_timezone)
        except DeviceLogCaptureError:
            report = None
        entries.append(
            _ParsedExitEntry(
                event_id=(
                    _event_id(report) if report is not None else sha256(entry.encode()).hexdigest()
                ),
                report=report,
            )
        )
    return tuple(entries)


def _parse_android_exit_entry(
    entry: str,
    device_timezone: tzinfo | None,
) -> ApplicationDiagnosticReport:
    timestamp_value = _required_match(
        entry,
        r"\btimestamp=(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+|\d+)",
        "timestamp",
    )
    process_name = _required_match(entry, r"\bprocess=([^\s]+)", "process")
    reason_match = re.search(
        r"\breason=(\d+)(?:\s+\((.+?)\)\s+(?:subreason|status)=)?",
        entry,
        re.DOTALL,
    )
    if reason_match is None:
        raise DeviceLogCaptureError("Android exit history omitted an exit reason.")
    reason_code = int(reason_match.group(1))
    reason_name = reason_match.group(2) or f"reason_{reason_code}"
    pid_match = re.search(r"\bpid=(\d+)", entry)
    status_match = re.search(r"\bstatus=(-?\d+)", entry)
    identity = "-".join(
        (
            sha256(timestamp_value.encode()).hexdigest()[:12],
            pid_match.group(1) if pid_match else "unknown-pid",
            sha256(process_name.encode()).hexdigest()[:12],
        )
    )
    return ApplicationDiagnosticReport(
        source_name=f"exit-info-{identity}.txt",
        text=f"{entry}\n",
        failure_kind=_android_failure_kind(reason_code),
        occurred_at=_timestamp(timestamp_value, device_timezone),
        platform_reason=reason_name,
        platform_code=(
            f"reason={reason_code},status={status_match.group(1)}"
            if status_match
            else f"reason={reason_code}"
        ),
    )


def _required_match(entry: str, pattern: str, field_name: str) -> str:
    match = re.search(pattern, entry)
    if match is None:
        raise DeviceLogCaptureError(f"Android exit history omitted the {field_name}.")
    return match.group(1)


def _timestamp(value: str, device_timezone: tzinfo | None) -> datetime | None:
    if value.isdigit():
        return _epoch_timestamp(int(value))
    if device_timezone is None:
        return None
    try:
        local_time = datetime.strptime(value, "%Y-%m-%d %H:%M:%S.%f").replace(
            tzinfo=device_timezone
        )
    except ValueError as error:
        raise DeviceLogCaptureError("Android exit history has an invalid timestamp.") from error
    return local_time.astimezone(timezone.utc)


def _epoch_timestamp(milliseconds: int) -> datetime:
    try:
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
    except (OSError, OverflowError, ValueError) as error:
        raise DeviceLogCaptureError("Android exit history has an invalid timestamp.") from error


def _android_failure_kind(reason_code: int) -> ApplicationFailureKind:
    if reason_code in {4, 5}:
        return ApplicationFailureKind.CRASH
    if reason_code == 6:
        return ApplicationFailureKind.ANR
    if reason_code == 0:
        return ApplicationFailureKind.UNCLASSIFIED
    return ApplicationFailureKind.OTHER_TERMINATION


def _event_id(report: ApplicationDiagnosticReport) -> str:
    identity = "\x00".join(
        (
            report.source_name,
            report.platform_code or "",
            report.occurred_at.isoformat() if report.occurred_at else "",
        )
    )
    return sha256(identity.encode()).hexdigest()
