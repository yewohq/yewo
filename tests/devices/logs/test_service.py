"""Tests for explicit device-log boundaries and durable drains."""

from datetime import timedelta
import json
from pathlib import Path

import pytest

from yewo.devices import DevicePlatform
from yewo.devices.logs import (
    ApplicationDiagnosticReport,
    DeviceLogBatch,
    DeviceLogBoundaryState,
    DeviceLogCaptureError,
    DeviceLogCapturePolicy,
    DeviceLogCheckpointKind,
    DeviceLogDrainState,
    DeviceLogOmission,
    DeviceLogRollbackError,
    DeviceLogSource,
    DeviceLogStore,
)
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)


from .support import (
    CAPTURE_TIME,
    ReaderStub as _ReaderStub,
    build_service as _service,
    text_batch as _text_batch,
)


def test_startup_entries_are_retained_as_an_ordered_start_receipt(
    tmp_path: Path,
) -> None:
    startup = _text_batch("launch SECRET")
    reader = _ReaderStub({"logcat": startup}, {"logcat": ()})

    started = _service(reader, tmp_path).begin()

    assert started.boundary.sources[0].state is DeviceLogBoundaryState.READY
    assert started.boundary.sources[0].baseline_item_count == 1
    receipt = started.startup_receipt
    assert receipt.drain_index == 0
    assert receipt.checkpoint_kind is DeviceLogCheckpointKind.SESSION_START
    artifact = receipt.sources[0].artifacts[0]
    assert (tmp_path / artifact.path).read_text() == "launch SECRET\n"
    assert reader.acknowledged_batches == [startup]


def test_action_drain_records_sequence_and_completion_time(tmp_path: Path) -> None:
    times = iter(
        (
            CAPTURE_TIME,
            CAPTURE_TIME + timedelta(seconds=1),
            CAPTURE_TIME + timedelta(seconds=2),
            CAPTURE_TIME + timedelta(seconds=3),
        )
    )
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("action"),)},
    )
    service = _service(reader, tmp_path, time_factory=times.__next__)
    started = service.begin()

    receipt = service.drain(
        started.boundary,
        DeviceLogCheckpointKind.ACTION,
        "action_1",
    )

    assert receipt.drain_index == 1
    assert receipt.started_at == CAPTURE_TIME + timedelta(seconds=2)
    assert receipt.completed_at == CAPTURE_TIME + timedelta(seconds=3)
    ledger = tmp_path / "device-logs" / started.boundary.boundary_id / "drains.jsonl"
    persisted = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert persisted[-1]["drain_id"] == receipt.drain_id


def test_action_drain_extracts_only_safe_app_owned_crash_hints(tmp_path: Path) -> None:
    crash = _text_batch(
        "java.lang.NumberFormatException: user supplied text",
        "at com.example.app.Editor.save(Editor.java:50)",
        "java.lang.IllegalStateException: unrelated",
        "at android.app.Activity.performCreate(Activity.java:9000)",
    )
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (crash,)},
    )
    service = _service(reader, tmp_path)
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert receipt.sources[0].diagnostic_hints == (
        "Exception class: NumberFormatException",
        "App frame: com.example.app.Editor.save(Editor.java:50)",
    )
    assert "user supplied text" not in " ".join(receipt.sources[0].diagnostic_hints)


def test_duplicate_checkpoint_is_rejected_before_another_destructive_read(
    tmp_path: Path,
) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("first"), _text_batch("must remain"))},
    )
    service = _service(reader, tmp_path)
    boundary = service.begin().boundary
    service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    with pytest.raises(DeviceLogCaptureError, match="already drained"):
        service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert reader.read_log_types == ["logcat"]


def test_oversized_single_entry_is_retained_with_explicit_tail_omission(
    tmp_path: Path,
) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("0123456789"),)},
    )
    service = _service(
        reader,
        tmp_path,
        policy=DeviceLogCapturePolicy(maximum_drain_bytes=4),
    )
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    source_receipt = receipt.sources[0]
    assert source_receipt.state is DeviceLogDrainState.CAPTURED
    assert source_receipt.omission is DeviceLogOmission.LATER
    assert (tmp_path / source_receipt.artifacts[0].path).read_bytes() == b"012\n"


def test_tiny_bound_does_not_create_synthetic_log_content(tmp_path: Path) -> None:
    batch = _text_batch("é")
    reader = _ReaderStub({"logcat": _text_batch()}, {"logcat": (batch,)})
    service = _service(
        reader,
        tmp_path,
        policy=DeviceLogCapturePolicy(maximum_drain_bytes=1),
    )
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    source_receipt = receipt.sources[0]
    assert source_receipt.state is DeviceLogDrainState.NO_RETAINED_ITEMS
    assert source_receipt.omission is DeviceLogOmission.LATER
    assert source_receipt.artifacts == ()
    assert reader.acknowledged_batches[-1] is batch


def test_reader_completeness_metadata_marks_earlier_omission(tmp_path: Path) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("entry", may_have_omitted_earlier_items=True),)},
    )
    service = _service(reader, tmp_path)
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert receipt.sources[0].omission is DeviceLogOmission.EARLIER


class _FailReceiptStore(DeviceLogStore):
    def __init__(self, root: Path, fail_on_receipt: int) -> None:
        super().__init__(root, identifier_factory=lambda: "artifact_1")
        self._receipt_count = 0
        self._fail_on_receipt = fail_on_receipt

    def persist_receipt(self, receipt) -> Path:
        self._receipt_count += 1
        if self._receipt_count == self._fail_on_receipt:
            raise DeviceLogCaptureError("receipt failed")
        return super().persist_receipt(receipt)


class _FailRollbackStore(_FailReceiptStore):
    def discard(self, reference) -> None:
        del reference
        raise DeviceLogCaptureError("rollback failed")


def test_receipt_failure_rolls_back_new_segments(tmp_path: Path) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("action"),)},
    )
    service = _service(
        reader,
        tmp_path,
        store=_FailReceiptStore(tmp_path, 2),
    )
    boundary = service.begin().boundary

    with pytest.raises(DeviceLogCaptureError, match="receipt failed"):
        service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert tuple(tmp_path.rglob("*.log")) == ()


def test_rollback_failure_raises_distinct_error_and_leaves_evidence(
    tmp_path: Path,
) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("action"),)},
    )
    service = _service(
        reader,
        tmp_path,
        store=_FailRollbackStore(tmp_path, 2),
    )
    boundary = service.begin().boundary

    with pytest.raises(DeviceLogRollbackError, match="Could not roll back"):
        service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert len(tuple(tmp_path.rglob("*.log"))) == 1


def test_startup_receipt_failure_rolls_back_boundary_and_artifact(
    tmp_path: Path,
) -> None:
    reader = _ReaderStub({"logcat": _text_batch("startup")}, {"logcat": ()})
    service = _service(
        reader,
        tmp_path,
        store=_FailReceiptStore(tmp_path, 1),
    )

    with pytest.raises(DeviceLogCaptureError, match="receipt failed"):
        service.begin()

    assert tuple(tmp_path.rglob("*.json")) == ()
    assert tuple(tmp_path.rglob("*.log")) == ()
    assert not (tmp_path / "device-logs/record_1").exists()


class _FailBoundaryStore(DeviceLogStore):
    def persist_boundary(self, boundary) -> Path:
        del boundary
        raise DeviceLogCaptureError("boundary failed")


def test_boundary_persistence_failure_is_fatal(tmp_path: Path) -> None:
    reader = _ReaderStub({"logcat": _text_batch("startup")}, {"logcat": ()})
    service = _service(reader, tmp_path, store=_FailBoundaryStore(tmp_path))

    with pytest.raises(DeviceLogCaptureError, match="boundary failed"):
        service.begin()


def test_coarse_crash_source_skips_actions_and_persists_scenario_reports(
    tmp_path: Path,
) -> None:
    source = DeviceLogSource(
        log_type="crashlog",
        artifact_stem="crashlog",
        is_diagnostic=True,
    )
    scenario_batch = DeviceLogBatch(
        (
            ApplicationDiagnosticReport(
                "Example-1.ips",
                '{"bundleID":"com.example.app"}',
                ApplicationFailureKind.CRASH,
            ),
        ),
        observed_item_count=2,
        filtered_item_count=1,
    )
    reader = _ReaderStub(
        {"crashlog": DeviceLogBatch((), 12)},
        {"crashlog": (scenario_batch,)},
        coarse_sources=frozenset({"crashlog"}),
    )
    service = _service(
        reader,
        tmp_path,
        sources=(source,),
        platform=DevicePlatform.IOS,
    )
    boundary = service.begin().boundary

    action = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")
    scenario = service.drain(
        boundary,
        DeviceLogCheckpointKind.SCENARIO,
        "scenario_1",
    )

    assert action.sources[0].state is DeviceLogDrainState.NOT_SCHEDULED
    assert scenario.sources[0].filtered_item_count == 1
    artifact = scenario.sources[0].artifacts[0]
    assert artifact.source_name == "Example-1.ips"
    assert artifact.path.suffix == ".ips"
    assert artifact.failure_kind is ApplicationFailureKind.CRASH
    assert (
        scenario.sources[0].diagnostic_classification
        is ApplicationDiagnosticClassification.APPLICATION_CRASH
    )
    assert reader.read_log_types == ["crashlog"]


def test_valid_diagnostic_survives_an_unparsed_sibling(tmp_path: Path) -> None:
    source = DeviceLogSource(
        log_type="crashlog",
        artifact_stem="crashlog",
        is_diagnostic=True,
    )
    scenario_batch = DeviceLogBatch(
        (
            ApplicationDiagnosticReport(
                "Example-1.ips",
                '{"bundleID":"com.example.app"}',
                ApplicationFailureKind.CRASH,
            ),
        ),
        observed_item_count=2,
        unparsed_item_count=1,
    )
    reader = _ReaderStub(
        {"crashlog": DeviceLogBatch((), 0)},
        {"crashlog": (scenario_batch,)},
        coarse_sources=frozenset({"crashlog"}),
    )
    service = _service(
        reader,
        tmp_path,
        sources=(source,),
        platform=DevicePlatform.IOS,
    )
    boundary = service.begin().boundary

    receipt = service.drain(
        boundary,
        DeviceLogCheckpointKind.SCENARIO,
        "scenario_1",
    )

    source_receipt = receipt.sources[0]
    assert source_receipt.unparsed_item_count == 1
    assert (
        source_receipt.diagnostic_classification
        is ApplicationDiagnosticClassification.APPLICATION_CRASH
    )


def test_unsupported_diagnostic_source_is_not_reported_as_no_failure(
    tmp_path: Path,
) -> None:
    source = DeviceLogSource(
        log_type="exitinfo",
        artifact_stem="exit-info",
        is_diagnostic=True,
    )
    reader = _ReaderStub({"exitinfo": None}, {"exitinfo": ()})
    service = _service(reader, tmp_path, sources=(source,))

    receipt = service.begin().startup_receipt

    assert (
        receipt.sources[0].diagnostic_classification
        is ApplicationDiagnosticClassification.DIAGNOSTICS_UNAVAILABLE
    )


def test_diagnostic_read_failure_is_yewo_infrastructure_failure(
    tmp_path: Path,
) -> None:
    source = DeviceLogSource(
        log_type="exitinfo",
        artifact_stem="exit-info",
        is_diagnostic=True,
    )
    reader = _ReaderStub(
        {"exitinfo": DeviceLogBatch((), 0)},
        {"exitinfo": (DeviceLogCaptureError("read failed"),)},
        coarse_sources=frozenset({"exitinfo"}),
    )
    service = _service(reader, tmp_path, sources=(source,))
    boundary = service.begin().boundary

    receipt = service.drain(
        boundary,
        DeviceLogCheckpointKind.SCENARIO,
        "scenario_1",
    )

    assert (
        receipt.sources[0].diagnostic_classification
        is ApplicationDiagnosticClassification.YEWO_INFRASTRUCTURE_FAILURE
    )


def test_diagnostic_baseline_failure_is_yewo_infrastructure_failure(
    tmp_path: Path,
) -> None:
    source = DeviceLogSource(
        log_type="exitinfo",
        artifact_stem="exit-info",
        is_diagnostic=True,
    )
    reader = _ReaderStub(
        {"exitinfo": DeviceLogCaptureError("baseline failed")},
        {"exitinfo": ()},
    )
    service = _service(reader, tmp_path, sources=(source,))

    receipt = service.begin().startup_receipt

    assert (
        receipt.sources[0].diagnostic_classification
        is ApplicationDiagnosticClassification.YEWO_INFRASTRUCTURE_FAILURE
    )
