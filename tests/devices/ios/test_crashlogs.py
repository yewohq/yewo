"""Tests for physical iOS crash-report collection through CoreDevice."""

from __future__ import annotations

from collections.abc import Sequence
import json
from pathlib import Path
import subprocess

import pytest

from yewo.devices.ios import DevicectlCrashLogReader
from yewo.devices.logs import (
    DeviceCrashReport,
    DeviceLogCaptureError,
    DeviceLogCheckpointKind,
)


def _inventory(*names: str) -> object:
    return {"result": {"files": [{"name": name} for name in names]}}


def _report(application_id: str) -> str:
    return (
        json.dumps({"app_name": "Example", "bundleID": application_id})
        + "\n"
        + json.dumps(
            {
                "procName": "Example",
                "bundleInfo": {"CFBundleIdentifier": application_id},
            }
        )
    )


class _DevicectlRunnerStub:
    def __init__(
        self,
        inventories: Sequence[object],
        reports_by_name: dict[str, str] | None = None,
    ) -> None:
        self._inventories = iter(inventories)
        self._reports_by_name = reports_by_name or {}
        self.calls: list[tuple[tuple[str, ...], float]] = []

    def __call__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> subprocess.CompletedProcess[str]:
        arguments = tuple(command)
        self.calls.append((arguments, timeout_seconds))
        output_path = Path(arguments[arguments.index("--json-output") + 1])
        if "info" in arguments:
            output_path.write_text(json.dumps(next(self._inventories)), encoding="utf-8")
        else:
            report_name = arguments[arguments.index("--source") + 1]
            destination = Path(arguments[arguments.index("--destination") + 1])
            destination.write_text(self._reports_by_name[report_name], encoding="utf-8")
            output_path.write_text(json.dumps({"info": {"outcome": "success"}}))
        return subprocess.CompletedProcess(arguments, 0, "", "")


def test_reader_snapshots_existing_reports_without_copying_them() -> None:
    runner = _DevicectlRunnerStub(
        (_inventory("OldApp-1.ips", "DiagnosticLogs"),),
    )
    reader = DevicectlCrashLogReader("PHONE-ID", "com.example.app", runner)

    baseline = reader.begin("crashlog")

    assert baseline is not None
    assert baseline.items == ()
    assert baseline.observed_item_count == 2
    assert len(runner.calls) == 1
    assert runner.calls[0][0][0:5] == (
        "xcrun",
        "devicectl",
        "device",
        "info",
        "files",
    )
    assert "systemCrashLogs" in runner.calls[0][0]


def test_reader_returns_only_new_reports_for_the_application() -> None:
    runner = _DevicectlRunnerStub(
        (
            _inventory("Existing.ips"),
            _inventory("Existing.ips", "Example-2.ips", "Other-2.ips", "Notes.txt"),
            _inventory("Existing.ips", "Example-2.ips", "Other-2.ips", "Notes.txt"),
        ),
        {
            "Example-2.ips": _report("com.example.app"),
            "Other-2.ips": _report("com.other.app"),
        },
    )
    reader = DevicectlCrashLogReader("PHONE-ID", "com.example.app", runner)

    baseline = reader.begin("crashlog")
    assert baseline is not None
    reader.acknowledge("crashlog", baseline)
    reports = reader.read("crashlog")
    unacknowledged_retry = reader.read("crashlog")
    reader.acknowledge("crashlog", reports)
    repeated_reports = reader.read("crashlog")

    assert reports.items == (
        DeviceCrashReport("Example-2.ips", _report("com.example.app")),
    )
    assert reports.observed_item_count == 3
    assert reports.filtered_item_count == 2
    assert unacknowledged_retry is reports
    assert repeated_reports.items == ()
    copied_names = [
        command[command.index("--source") + 1]
        for command, _ in runner.calls
        if "copy" in command
    ]
    assert copied_names == ["Example-2.ips", "Other-2.ips"]


def test_reader_scans_crashes_only_at_coarse_checkpoints() -> None:
    reader = DevicectlCrashLogReader(
        "PHONE-ID",
        "com.example.app",
        _DevicectlRunnerStub((_inventory(),)),
    )

    assert not reader.captures_checkpoint("crashlog", DeviceLogCheckpointKind.ACTION)
    assert reader.captures_checkpoint("crashlog", DeviceLogCheckpointKind.SCENARIO)
    assert reader.captures_checkpoint("crashlog", DeviceLogCheckpointKind.SESSION_END)


@pytest.mark.parametrize(
    "payload",
    (
        {},
        {"result": {"items": []}},
        {"result": {"files": [{"name": "../private.ips"}]}},
        {"result": {"files": [{"name": "invalid\nname.ips"}]}},
    ),
)
def test_reader_rejects_unrecognized_or_unsafe_inventories(payload: object) -> None:
    reader = DevicectlCrashLogReader(
        "PHONE-ID",
        "com.example.app",
        _DevicectlRunnerStub((payload,)),
    )

    with pytest.raises(DeviceLogCaptureError, match="CoreDevice returned"):
        reader.begin("crashlog")


def test_reader_rejects_a_source_it_does_not_own() -> None:
    reader = DevicectlCrashLogReader(
        "PHONE-ID",
        "com.example.app",
        _DevicectlRunnerStub((_inventory(),)),
    )

    with pytest.raises(DeviceLogCaptureError, match="does not own"):
        reader.read("syslog")
