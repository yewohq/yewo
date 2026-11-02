"""Tests for Appium iOS diagnostic ownership."""

from collections import deque
import json

from yewo.devices.diagnostics import ApplicationFailureKind
from yewo.devices.ios import IosAppiumCrashLogReader
from yewo.devices.logs import ApplicationDiagnosticReport, DeviceLogCheckpointKind


def _report(application_id: str, exception_type: str) -> str:
    return json.dumps(
        {
            "bundleID": application_id,
            "exceptionType": exception_type,
        }
    )


class _DriverStub:
    log_types = ("syslog", "crashlog")

    def __init__(self, *batches: list[object]) -> None:
        self.batches = deque(batches)

    def get_log(self, log_type: str) -> list[object]:
        assert log_type == "crashlog"
        return self.batches.popleft()


def test_reader_discards_baseline_and_classifies_new_application_reports() -> None:
    driver = _DriverStub(
        [{"message": _report("com.example.app", "EXC_BAD_ACCESS")}],
        [
            {"message": _report("com.other.app", "EXC_BAD_ACCESS")},
            {"message": _report("com.example.app", "EXC_BAD_ACCESS")},
        ],
    )
    reader = IosAppiumCrashLogReader(driver, "com.example.app")

    baseline = reader.begin("crashlog")
    assert baseline is not None
    reader.acknowledge("crashlog", baseline)
    batch = reader.read("crashlog")

    assert len(batch.items) == 1
    assert isinstance(batch.items[0], ApplicationDiagnosticReport)
    assert batch.items[0].failure_kind is ApplicationFailureKind.CRASH
    assert batch.observed_item_count == 2
    assert batch.filtered_item_count == 1
    assert baseline.filtered_item_count == 1
    assert reader.captures_checkpoint("crashlog", DeviceLogCheckpointKind.ACTION)
    assert reader.captures_checkpoint("crashlog", DeviceLogCheckpointKind.RECOVERY)


def test_reader_counts_an_invalid_entry_without_losing_valid_reports() -> None:
    driver = _DriverStub(
        [],
        [
            {"message": "not-json"},
            {"message": _report("com.other.app", "EXC_BAD_ACCESS")},
            {"message": _report("com.example.app", "EXC_BAD_ACCESS")},
        ],
    )
    reader = IosAppiumCrashLogReader(driver, "com.example.app")
    baseline = reader.begin("crashlog")
    assert baseline is not None
    reader.acknowledge("crashlog", baseline)

    batch = reader.read("crashlog")

    assert len(batch.items) == 1
    assert batch.observed_item_count == 3
    assert batch.filtered_item_count == 1
    assert batch.unparsed_item_count == 1
