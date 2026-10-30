"""Tests for explicit device-log boundaries and durable drains."""

from collections import deque
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

import pytest

from yewo.applications import ApplicationDescriptor
from yewo.artifacts import ArtifactSensitivity
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.android import ANDROID_LOG_SOURCES
from yewo.devices.logs import (
    DeviceCrashReport,
    DeviceLogBatch,
    DeviceLogBoundaryState,
    DeviceLogCaptureError,
    DeviceLogCapturePolicy,
    DeviceLogCaptureService,
    DeviceLogCheckpointKind,
    DeviceLogDrainState,
    DeviceLogOmission,
    DeviceLogRedactionResult,
    DeviceLogRollbackError,
    DeviceLogSource,
    DeviceLogStore,
    DeviceLogTextScrubber,
    DeviceLogTextEntry,
)


CAPTURE_TIME = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def _text_batch(
    *entries: str,
    observed_item_count: int | None = None,
    filtered_item_count: int = 0,
    may_have_omitted_earlier_items: bool = False,
) -> DeviceLogBatch:
    return DeviceLogBatch(
        tuple(DeviceLogTextEntry(entry) for entry in entries),
        len(entries) if observed_item_count is None else observed_item_count,
        filtered_item_count,
        may_have_omitted_earlier_items,
    )


class _ReaderStub:
    def __init__(
        self,
        baselines: dict[str, DeviceLogBatch | None | Exception],
        reads: dict[str, tuple[DeviceLogBatch | Exception, ...]],
        *,
        coarse_sources: frozenset[str] = frozenset(),
    ) -> None:
        self._baselines = baselines
        self._reads = {name: deque(values) for name, values in reads.items()}
        self._coarse_sources = coarse_sources
        self.read_log_types: list[str] = []
        self.acknowledged_batches: list[DeviceLogBatch] = []

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        value = self._baselines[log_type]
        if isinstance(value, Exception):
            raise value
        return value

    def read(self, log_type: str) -> DeviceLogBatch:
        self.read_log_types.append(log_type)
        value = self._reads[log_type].popleft()
        if isinstance(value, Exception):
            raise value
        return value

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        return not (
            log_type in self._coarse_sources
            and checkpoint_kind is DeviceLogCheckpointKind.ACTION
        )

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        del log_type
        self.acknowledged_batches.append(batch)


def _redact(text: str) -> DeviceLogRedactionResult:
    count_found = text.count("SECRET")
    return DeviceLogRedactionResult(
        text.replace("SECRET", "[redacted]"),
        "test-policy-v1",
        count_found,
    )


def _raise_redaction_error(text: str) -> DeviceLogRedactionResult:
    del text
    raise RuntimeError("secret")


def _device(platform: DevicePlatform = DevicePlatform.ANDROID) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=platform,
        target_id="test-target",
        connection_id="test-connection",
        display_name="test device",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="ready",
    )


def _application(
    platform: DevicePlatform = DevicePlatform.ANDROID,
) -> ApplicationDescriptor:
    return ApplicationDescriptor(platform=platform, application_id="com.example.app")


def _service(
    reader: _ReaderStub,
    root: Path,
    *,
    sources: tuple[DeviceLogSource, ...] = ANDROID_LOG_SOURCES,
    platform: DevicePlatform = DevicePlatform.ANDROID,
    policy: DeviceLogCapturePolicy | None = None,
    store: DeviceLogStore | None = None,
    scrubber: DeviceLogTextScrubber = _redact,
    time_factory: Callable[[], datetime] = lambda: CAPTURE_TIME,
) -> DeviceLogCaptureService:
    identifiers = (f"record_{number}" for number in count(1))
    artifacts = (f"artifact_{number}" for number in count(1))
    return DeviceLogCaptureService(
        reader,
        _device(platform),
        _application(platform),
        sources,
        store or DeviceLogStore(root, identifier_factory=artifacts.__next__),
        scrubber,
        policy=policy,
        identifier_factory=identifiers.__next__,
        time_factory=time_factory,
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
    assert artifact.redaction_policy_id == "test-policy-v1"
    assert artifact.source_redaction_count == 1
    assert (tmp_path / artifact.path).read_text() == "launch [redacted]\n"
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
    assert (tmp_path / "assets/device-logs" / started.boundary.boundary_id / (
        f"000001-{receipt.drain_id}.json"
    )).is_file()


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


def test_truncated_artifact_counts_source_redactions_before_bounding(
    tmp_path: Path,
) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("SECRET trailing"),)},
    )
    service = _service(
        reader,
        tmp_path,
        policy=DeviceLogCapturePolicy(maximum_drain_bytes=5),
    )
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    artifact = receipt.sources[0].artifacts[0]
    assert artifact.source_redaction_count == 1
    assert (tmp_path / artifact.path).read_bytes() == b"[red\n"


def test_reader_completeness_metadata_marks_earlier_omission(tmp_path: Path) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {
            "logcat": (
                _text_batch("entry", may_have_omitted_earlier_items=True),
            )
        },
    )
    service = _service(reader, tmp_path)
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert receipt.sources[0].omission is DeviceLogOmission.EARLIER


def test_redactor_failure_records_consumed_but_unretained_items(tmp_path: Path) -> None:
    action_batch = _text_batch("entry")
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (action_batch,)},
    )
    service = _service(reader, tmp_path, scrubber=_raise_redaction_error)
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    source_receipt = receipt.sources[0]
    assert source_receipt.state is DeviceLogDrainState.FAILED
    assert source_receipt.observed_item_count == 1
    assert source_receipt.retained_item_count == 0
    assert action_batch not in reader.acknowledged_batches


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
    assert not (tmp_path / "assets/device-logs/record_1").exists()


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
    source = DeviceLogSource(log_type="crashlog", artifact_stem="crashlog")
    scenario_batch = DeviceLogBatch(
        (DeviceCrashReport("Example-1.ips", '{"bundleID":"com.example.app"}'),),
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
    assert reader.read_log_types == ["crashlog"]


def test_policy_can_classify_log_artifacts_as_declared_sensitive(
    tmp_path: Path,
) -> None:
    reader = _ReaderStub(
        {"logcat": _text_batch()},
        {"logcat": (_text_batch("entry"),)},
    )
    service = _service(
        reader,
        tmp_path,
        policy=DeviceLogCapturePolicy(
            sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE
        ),
    )
    boundary = service.begin().boundary

    receipt = service.drain(boundary, DeviceLogCheckpointKind.ACTION, "action_1")

    assert receipt.sources[0].artifacts[0].sensitivity is (
        ArtifactSensitivity.DECLARED_SENSITIVE
    )
