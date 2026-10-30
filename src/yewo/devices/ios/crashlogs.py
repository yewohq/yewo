"""Read new physical iOS crash reports through Apple CoreDevice."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

from yewo.devices._commands import CommandRunner, run_command
from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import DeviceCrashReport, DeviceLogBatch
from yewo.devices.logs.records import DeviceLogCheckpointKind


_CRASH_LOG_TYPE = "crashlog"
_CRASH_LOG_DOMAIN = "systemCrashLogs"
_MAXIMUM_REPORT_BYTES = 16 * 1024 * 1024
_APPLICATION_IDENTITY_KEYS = frozenset(
    {
        "bundleid",
        "bundleidentifier",
        "cfbundleidentifier",
    }
)


def _crash_report_names(payload: object) -> frozenset[str]:
    if not isinstance(payload, dict):
        raise DeviceLogCaptureError("CoreDevice returned an invalid crash-log inventory.")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise DeviceLogCaptureError("CoreDevice returned an invalid crash-log inventory.")
    files = result.get("files")
    if not isinstance(files, list):
        raise DeviceLogCaptureError("CoreDevice returned an invalid crash-log inventory.")
    names: set[str] = set()
    for file_entry in files:
        if not isinstance(file_entry, dict):
            raise DeviceLogCaptureError("CoreDevice returned an invalid crash-log entry.")
        name = file_entry.get("name")
        if not isinstance(name, str) or not name:
            raise DeviceLogCaptureError("CoreDevice returned an invalid crash-log entry.")
        _validate_remote_path(name)
        names.add(name)
    return frozenset(names)


def _validate_remote_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or any(ord(character) < 32 for character in name)
    ):
        raise DeviceLogCaptureError("CoreDevice returned an unsafe crash-log path.")
    return path


def _json_values(content: str) -> Iterator[object]:
    decoder = json.JSONDecoder()
    position = 0
    while position < len(content):
        while position < len(content) and content[position].isspace():
            position += 1
        if position == len(content):
            return
        try:
            value, position = decoder.raw_decode(content, position)
        except json.JSONDecodeError as error:
            raise DeviceLogCaptureError("CoreDevice copied an invalid crash report.") from error
        yield value


def _contains_application_id(value: object, application_id: str) -> bool:
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            if (
                isinstance(key, str)
                and key.casefold() in _APPLICATION_IDENTITY_KEYS
                and nested_value == application_id
            ):
                return True
            if _contains_application_id(nested_value, application_id):
                return True
    elif isinstance(value, list):
        return any(_contains_application_id(item, application_id) for item in value)
    return False


def _is_application_report(content: str, application_id: str) -> bool:
    values = tuple(_json_values(content))
    if not values:
        raise DeviceLogCaptureError("CoreDevice copied an empty crash report.")
    return any(_contains_application_id(value, application_id) for value in values)


class DevicectlCrashLogReader:
    """Diff and read physical iOS crash reports without changing the device."""

    def __init__(
        self,
        connection_id: str,
        application_id: str,
        runner: CommandRunner = run_command,
        *,
        xcrun_executable: str = "xcrun",
    ) -> None:
        if not connection_id or not application_id:
            raise ValueError("CoreDevice crash-log identity cannot be empty.")
        self._connection_id = connection_id
        self._application_id = application_id
        self._runner = runner
        self._xcrun_executable = xcrun_executable
        self._known_report_names: frozenset[str] | None = None
        self._pending_batch: tuple[DeviceLogBatch, frozenset[str]] | None = None

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        if log_type != _CRASH_LOG_TYPE:
            return None
        self._require_acknowledged()
        report_names = self._list_report_names()
        batch = DeviceLogBatch(items=(), observed_item_count=len(report_names))
        self._pending_batch = (batch, report_names)
        return batch

    def read(self, log_type: str) -> DeviceLogBatch:
        if log_type != _CRASH_LOG_TYPE:
            raise DeviceLogCaptureError("CoreDevice does not own this device-log source.")
        if self._pending_batch is not None:
            return self._pending_batch[0]
        if self._known_report_names is None:
            raise DeviceLogCaptureError("The CoreDevice crash-log boundary is not active.")
        report_names = self._list_report_names()

        newly_observed_names = report_names - self._known_report_names
        new_report_names = tuple(
            sorted(name for name in newly_observed_names if name.casefold().endswith(".ips"))
        )
        matching_reports = tuple(
            report
            for name in new_report_names
            for report in (self._copy_application_report(name),)
            if report is not None
        )
        batch = DeviceLogBatch(
            items=matching_reports,
            observed_item_count=len(newly_observed_names),
            filtered_item_count=len(newly_observed_names) - len(matching_reports),
        )
        self._pending_batch = (batch, report_names)
        return batch

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        if log_type != _CRASH_LOG_TYPE:
            raise DeviceLogCaptureError("CoreDevice does not own this device-log source.")
        return checkpoint_kind in {
            DeviceLogCheckpointKind.SCENARIO,
            DeviceLogCheckpointKind.SESSION_END,
        }

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        if log_type != _CRASH_LOG_TYPE:
            raise DeviceLogCaptureError("CoreDevice does not own this device-log source.")
        if self._pending_batch is None or self._pending_batch[0] is not batch:
            raise DeviceLogCaptureError("CoreDevice received an unknown crash-log batch.")
        self._known_report_names = self._pending_batch[1]
        self._pending_batch = None

    def _require_acknowledged(self) -> None:
        if self._pending_batch is not None:
            raise DeviceLogCaptureError("The previous CoreDevice crash-log read is not durable.")

    def _list_report_names(self) -> frozenset[str]:
        with tempfile.TemporaryDirectory(prefix="yewo-crashlog-list-") as directory:
            output_path = Path(directory) / "files.json"
            command = (
                self._xcrun_executable,
                "devicectl",
                "device",
                "info",
                "files",
                "--device",
                self._connection_id,
                "--domain-type",
                _CRASH_LOG_DOMAIN,
                "--json-output",
                str(output_path),
                "--quiet",
            )
            self._run(command, timeout_seconds=30, operation="list crash reports")
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise DeviceLogCaptureError(
                    "CoreDevice did not persist a valid crash-log inventory."
                ) from error
        return _crash_report_names(payload)

    def _copy_application_report(self, name: str) -> DeviceCrashReport | None:
        remote_path = _validate_remote_path(name)
        with tempfile.TemporaryDirectory(prefix="yewo-crashlog-copy-") as directory:
            local_path = Path(directory) / remote_path.name
            output_path = Path(directory) / "copy.json"
            command = (
                self._xcrun_executable,
                "devicectl",
                "device",
                "copy",
                "from",
                "--device",
                self._connection_id,
                "--domain-type",
                _CRASH_LOG_DOMAIN,
                "--source",
                name,
                "--destination",
                str(local_path),
                "--json-output",
                str(output_path),
                "--quiet",
            )
            self._run(command, timeout_seconds=60, operation="copy a crash report")
            try:
                report_bytes = local_path.read_bytes()
                if len(report_bytes) > _MAXIMUM_REPORT_BYTES:
                    raise DeviceLogCaptureError(
                        "CoreDevice copied a crash report that exceeds the size bound."
                    )
                content = report_bytes.decode("utf-8")
            except DeviceLogCaptureError:
                raise
            except (OSError, UnicodeError) as error:
                raise DeviceLogCaptureError(
                    "CoreDevice did not copy a readable crash report."
                ) from error
        if not _is_application_report(content, self._application_id):
            return None
        return DeviceCrashReport(source_name=name, text=content)

    def _run(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
        operation: str,
    ) -> None:
        try:
            result = self._runner(command, timeout_seconds=timeout_seconds)
        except (OSError, subprocess.SubprocessError) as error:
            raise DeviceLogCaptureError(f"Could not {operation} with CoreDevice.") from error
        if result.returncode:
            raise DeviceLogCaptureError(f"Could not {operation} with CoreDevice.")
