"""Tests for Android process-exit diagnostic extraction."""

from collections import deque
from datetime import timezone

import pytest

from yewo.devices.android.diagnostics import (
    AdbApplicationExitReader,
    parse_android_exit_info,
)
from yewo.devices.diagnostics import ApplicationFailureKind
from yewo.devices.logs import (
    ApplicationDiagnosticReport,
    DeviceLogCaptureError,
    DeviceLogCheckpointKind,
)


def _entry(
    index: int,
    *,
    timestamp: str,
    reason: int,
    reason_name: str,
    process: str = "com.example.app",
) -> str:
    return f"""
      ApplicationExitInfo #{index}:
        timestamp={timestamp}
        pid={1000 + index}
        realUid=10123
        packageUid=10123
        process={process}
        reason={reason} ({reason_name})
        status=0
        importance=100 pss=12000 rss=34000
    """


def _dump(*entries: str) -> str:
    return "\n".join(
        (
            "ACTIVITY MANAGER PROCESS EXIT INFO (dumpsys activity exit-info)",
            "Last Timestamp of Persistence Into Persistent Storage: 2026-10-30 12:00:00.000",
            *entries,
        )
    )


class _AdbStub:
    def __init__(
        self,
        *exit_outputs: str,
        api_level: str = "35",
        timezones: tuple[str, ...] = ("UTC",),
    ) -> None:
        self.api_level = api_level
        self.exit_outputs = deque(exit_outputs)
        self.timezones = deque(timezones)
        self.last_timezone = timezones[-1]

    def ask(
        self,
        connection_id: str,
        *args: str,
        timeout_seconds: float = 20,
    ) -> str | None:
        del connection_id, timeout_seconds
        if args == ("getprop", "ro.build.version.sdk"):
            return self.api_level
        if args == ("getprop", "persist.sys.timezone"):
            if self.timezones:
                self.last_timezone = self.timezones.popleft()
            return self.last_timezone
        assert args[:3] == ("dumpsys", "activity", "exit-info")
        return self.exit_outputs.popleft()


def test_parser_classifies_java_native_crash_and_anr() -> None:
    output = _dump(
        _entry(0, timestamp="2026-10-30 12:00:00.000", reason=4, reason_name="CRASH"),
        _entry(
            1,
            timestamp="2026-10-30 12:00:01.000",
            reason=5,
            reason_name="CRASH_NATIVE",
        ),
        _entry(2, timestamp="2026-10-30 12:00:02.000", reason=6, reason_name="ANR"),
    )

    reports = parse_android_exit_info(output, timezone.utc)

    assert tuple(report.failure_kind for report in reports) == (
        ApplicationFailureKind.CRASH,
        ApplicationFailureKind.CRASH,
        ApplicationFailureKind.ANR,
    )
    assert reports[2].platform_reason == "ANR"
    assert reports[2].occurred_at is not None


def test_reader_baselines_and_replays_until_acknowledged() -> None:
    existing = _entry(
        0,
        timestamp="2026-10-30 12:00:00.000",
        reason=10,
        reason_name="USER_REQUESTED",
    )
    crash = _entry(
        1,
        timestamp="2026-10-30 12:00:01.000",
        reason=4,
        reason_name="CRASH",
    )
    reader = AdbApplicationExitReader(
        _AdbStub(_dump(existing), _dump(existing, crash), _dump(existing, crash)),
        "emulator-5554",
        "com.example.app",
    )

    baseline = reader.begin("exitinfo")
    assert baseline is not None
    reader.acknowledge("exitinfo", baseline)
    batch = reader.read("exitinfo")
    replay = reader.read("exitinfo")
    reader.acknowledge("exitinfo", batch)
    empty = reader.read("exitinfo")

    assert len(batch.items) == 1
    assert isinstance(batch.items[0], ApplicationDiagnosticReport)
    assert batch.items[0].failure_kind is ApplicationFailureKind.CRASH
    assert replay is batch
    assert empty.items == ()
    assert baseline.filtered_item_count == 1
    assert reader.captures_checkpoint("exitinfo", DeviceLogCheckpointKind.APPLICATION_SETUP)
    assert reader.captures_checkpoint("exitinfo", DeviceLogCheckpointKind.ACTION)
    assert reader.captures_checkpoint("exitinfo", DeviceLogCheckpointKind.RECOVERY)
    assert reader.captures_checkpoint("exitinfo", DeviceLogCheckpointKind.SCENARIO)


def test_reader_reports_unsupported_before_android_11() -> None:
    reader = AdbApplicationExitReader(
        _AdbStub(api_level="29"),
        "device-1",
        "com.example.app",
    )

    assert reader.begin("exitinfo") is None


def test_reader_counts_a_malformed_entry_without_losing_valid_siblings() -> None:
    malformed = """
        ApplicationExitInfo #0:
          timestamp=2026-10-30 12:00:00.000
          pid=1000
          process=com.example.app
          status=0
    """
    crash = _entry(
        1,
        timestamp="2026-10-30 12:00:01.000",
        reason=4,
        reason_name="CRASH",
    )
    reader = AdbApplicationExitReader(
        _AdbStub(_dump(), _dump(malformed, crash)),
        "emulator-5554",
        "com.example.app",
    )
    baseline = reader.begin("exitinfo")
    assert baseline is not None
    reader.acknowledge("exitinfo", baseline)

    batch = reader.read("exitinfo")

    assert batch.observed_item_count == 2
    assert batch.unparsed_item_count == 1
    assert len(batch.items) == 1
    report = batch.items[0]
    assert isinstance(report, ApplicationDiagnosticReport)
    assert report.failure_kind is ApplicationFailureKind.CRASH


def test_parser_rejects_an_unknown_response_shape() -> None:
    with pytest.raises(DeviceLogCaptureError, match="unknown"):
        parse_android_exit_info("vendor output")


def test_reader_refreshes_the_device_timezone_for_each_read() -> None:
    crash = _entry(
        0,
        timestamp="2026-10-30 12:00:00.000",
        reason=4,
        reason_name="CRASH",
    )
    reader = AdbApplicationExitReader(
        _AdbStub(
            _dump(),
            _dump(crash),
            timezones=("UTC", "America/Los_Angeles"),
        ),
        "emulator-5554",
        "com.example.app",
    )
    baseline = reader.begin("exitinfo")
    assert baseline is not None
    reader.acknowledge("exitinfo", baseline)

    report = reader.read("exitinfo").items[0]

    assert isinstance(report, ApplicationDiagnosticReport)
    assert report.occurred_at is not None
    assert report.occurred_at.hour == 19


def test_parser_accepts_current_compact_android_output() -> None:
    output = _dump(
        """
        ApplicationExitInfo #0:
          timestamp=2026-10-30 05:57:37.323 pid=1179 realUid=1000 packageUid=1000
          process=com.android.settings reason=4 (APP CRASH(EXCEPTION)) subreason=0 (UNKNOWN) status=0
          importance=100 pss=0.00 rss=0.00 description=crash state=empty trace=null
        """
    )

    report = parse_android_exit_info(output, timezone.utc)[0]

    assert report.failure_kind is ApplicationFailureKind.CRASH
    assert report.platform_reason == "APP CRASH(EXCEPTION)"
