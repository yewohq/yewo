"""Tests for checkpointed screen-recording ownership."""

from collections.abc import Callable
from datetime import datetime, timezone
from itertools import count
import json
from pathlib import Path

import pytest

from yewo.applications import ApplicationDescriptor
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.recording import (
    ScreenRecordingBoundaryState,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
    ScreenRecordingError,
    ScreenRecordingPolicy,
    ScreenRecordingRemoteCleanupState,
    ScreenRecordingRollbackError,
    ScreenRecordingSegmentState,
    ScreenRecordingService,
    ScreenRecordingStore,
    ScreenRecordingVideoMetadata,
)
from .support import (
    RecorderStub as _RecorderStub,
    RemoteRecorderStub as _RemoteRecorderStub,
)


CAPTURE_TIME = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
SCENARIO_VIDEO = b"\x00\x00\x00\x18ftypmp42scenario"
FINAL_VIDEO = b"\x00\x00\x00\x18ftypmp42final"
REDACTED_VIDEO = b"\x00\x00\x00\x18ftypmp42redacted"


def _device() -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="emulator-5554",
        connection_id="emulator-5554",
        display_name="Pixel 9",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="device",
    )


def _application() -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=DevicePlatform.ANDROID,
        application_id="com.example.app",
    )


def _service(
    recorder: _RecorderStub,
    root: Path,
    *,
    store: ScreenRecordingStore | None = None,
    policy: ScreenRecordingPolicy | None = None,
    identifier_factory: Callable[[], str] | None = None,
    monotonic_factory: Callable[[], float] = lambda: 0.0,
    metadata_reader: Callable[[Path], ScreenRecordingVideoMetadata] | None = None,
) -> ScreenRecordingService:
    identifiers = (f"record_{value}" for value in count(1))
    artifacts = (f"artifact_{value}" for value in count(1))
    return ScreenRecordingService(
        recorder,
        _device(),
        _application(),
        store or ScreenRecordingStore(root, identifier_factory=artifacts.__next__),
        policy=policy,
        metadata_reader=metadata_reader
        or (lambda path: ScreenRecordingVideoMetadata(16, 9, 30.0, 1.0)),
        identifier_factory=identifier_factory or identifiers.__next__,
        time_factory=lambda: CAPTURE_TIME,
        monotonic_factory=monotonic_factory,
    )


def test_scenario_checkpoint_persists_segment_and_starts_the_next(
    tmp_path: Path,
) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    scenario = service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )
    final = service.finish(boundary)

    assert boundary.state is ScreenRecordingBoundaryState.ACTIVE
    assert boundary.requested_maximum_seconds == 1_800
    assert boundary.effective_maximum_seconds == 1_800
    assert scenario.state is ScreenRecordingSegmentState.CAPTURED
    assert scenario.continuation_started
    assert final.state is ScreenRecordingSegmentState.CAPTURED
    assert not final.continuation_started
    assert recorder.start_limits == [1_800, 1_800]
    assert recorder.stop_count == 2
    assert scenario.artifact is not None
    assert final.artifact is not None
    assert scenario.artifact.capture_method is ScreenRecordingCaptureMethod.DRIVER_RESPONSE
    assert scenario.artifact.duration_seconds == 1.0
    assert scenario.artifact.reached_effective_limit is False
    assert (tmp_path / scenario.artifact.path).read_bytes() == SCENARIO_VIDEO
    assert (tmp_path / final.artifact.path).read_bytes() == FINAL_VIDEO


def test_remote_capture_records_cleanup_release_after_durable_receipt(
    tmp_path: Path,
) -> None:
    recorder = _RemoteRecorderStub((FINAL_VIDEO,))
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    receipt = service.finish(boundary)

    assert receipt.remote_cleanup_state is ScreenRecordingRemoteCleanupState.RELEASED
    assert recorder.acknowledgement_count == 1
    events_path = next(tmp_path.glob("recordings/*/events.jsonl"))
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [event["event_type"] for event in events].count("remote_cleanup") == 1


def test_remote_capture_records_pending_cleanup_when_release_fails(
    tmp_path: Path,
) -> None:
    recorder = _RemoteRecorderStub((FINAL_VIDEO,))
    recorder.acknowledgement_error = RuntimeError("remote cleanup failed")
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    receipt = service.finish(boundary)

    assert receipt.remote_cleanup_state is ScreenRecordingRemoteCleanupState.PENDING
    assert recorder.acknowledgement_count == 1
    events_path = next(tmp_path.glob("recordings/*/events.jsonl"))
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert all(event["event_type"] != "remote_cleanup" for event in events)


def test_unavailable_first_segment_can_recover_at_scenario_boundary(
    tmp_path: Path,
) -> None:
    recorder = _RecorderStub((FINAL_VIDEO,))
    recorder.start_failures = 1
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    scenario = service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )
    final = service.finish(boundary)

    assert boundary.state is ScreenRecordingBoundaryState.UNAVAILABLE
    assert scenario.state is ScreenRecordingSegmentState.NO_RECORDING
    assert scenario.continuation_started
    assert final.state is ScreenRecordingSegmentState.CAPTURED


def test_unexpected_start_error_is_normalized_as_unavailable(tmp_path: Path) -> None:
    recorder = _RecorderStub()
    recorder.start_error = ValueError("invalid adapter state")
    service = _service(recorder, tmp_path)

    boundary = service.begin()

    assert boundary.state is ScreenRecordingBoundaryState.UNAVAILABLE


def test_disabled_recording_never_calls_the_recorder(tmp_path: Path) -> None:
    recorder = _RecorderStub()
    service = _service(
        recorder,
        tmp_path,
        policy=ScreenRecordingPolicy(enabled=False),
    )
    boundary = service.begin()

    receipt = service.finish(boundary)

    assert boundary.state is ScreenRecordingBoundaryState.DISABLED
    assert receipt.state is ScreenRecordingSegmentState.NO_RECORDING
    assert recorder.start_limits == []
    assert recorder.stop_count == 0


def test_processing_failure_restarts_only_after_the_recorder_is_safe(
    tmp_path: Path,
) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))

    def _failing_metadata(path: Path) -> ScreenRecordingVideoMetadata:
        raise ScreenRecordingError("metadata unreadable")

    service = _service(recorder, tmp_path, metadata_reader=_failing_metadata)
    boundary = service.begin()

    receipt = service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )

    assert receipt.state is ScreenRecordingSegmentState.FAILED
    assert receipt.continuation_started is True
    assert recorder.start_limits == [1_800, 1_800]


def test_stop_failure_is_recorded_without_unsafe_restart(tmp_path: Path) -> None:
    recorder = _RecorderStub((ScreenRecordingError("stop failed"),))
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    receipt = service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )

    assert receipt.state is ScreenRecordingSegmentState.FAILED
    assert not receipt.continuation_started
    assert recorder.start_limits == [1_800]


def test_unexpected_stop_error_is_normalized_as_a_failed_segment(tmp_path: Path) -> None:
    recorder = _RecorderStub((ValueError("invalid adapter state"),))
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    receipt = service.finish(boundary)

    assert receipt.state is ScreenRecordingSegmentState.FAILED


def test_final_stop_failure_persists_failed_session_end_receipt(tmp_path: Path) -> None:
    recorder = _RecorderStub((ScreenRecordingError("stop failed"),))
    service = _service(recorder, tmp_path)
    boundary = service.begin()

    receipt = service.finish(boundary)

    assert receipt.state is ScreenRecordingSegmentState.FAILED
    assert not receipt.continuation_started
    assert receipt.continuation_reason is None


def test_duplicate_checkpoint_is_rejected_before_stopping_again(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO,))
    service = _service(recorder, tmp_path)
    boundary = service.begin()
    service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )

    with pytest.raises(ScreenRecordingError, match="already completed"):
        service.checkpoint(
            boundary,
            ScreenRecordingCheckpointKind.SCENARIO,
            "scenario_1",
        )

    assert recorder.stop_count == 1


class _FailReceiptStore(ScreenRecordingStore):
    def persist_receipt(self, receipt) -> Path:
        del receipt
        raise ScreenRecordingError("receipt failed")


class _FailRollbackStore(_FailReceiptStore):
    def discard(self, reference) -> None:
        del reference
        raise ScreenRecordingError("discard failed")


class _FailBoundaryStore(ScreenRecordingStore):
    def persist_boundary(self, boundary) -> Path:
        del boundary
        raise ScreenRecordingError("boundary failed")


def test_boundary_failure_stops_started_recording(tmp_path: Path) -> None:
    recorder = _RecorderStub((FINAL_VIDEO,))
    service = _service(recorder, tmp_path, store=_FailBoundaryStore(tmp_path))

    with pytest.raises(ScreenRecordingError, match="boundary failed"):
        service.begin()

    assert recorder.stop_count == 1


def test_receipt_failure_discards_video_and_stops_continuation(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    service = _service(recorder, tmp_path, store=_FailReceiptStore(tmp_path))
    boundary = service.begin()

    with pytest.raises(ScreenRecordingError, match="receipt failed"):
        service.checkpoint(
            boundary,
            ScreenRecordingCheckpointKind.SCENARIO,
            "scenario_1",
        )

    assert tuple(tmp_path.rglob("*.mp4")) == ()
    assert recorder.stop_count == 2


def test_invalid_receipt_rolls_back_video_and_continuation(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    identifiers = iter(("recording_1", "INVALID RECEIPT"))
    service = _service(
        recorder,
        tmp_path,
        identifier_factory=identifiers.__next__,
    )
    boundary = service.begin()

    with pytest.raises(ScreenRecordingError, match="receipt identifier"):
        service.checkpoint(
            boundary,
            ScreenRecordingCheckpointKind.SCENARIO,
            "scenario_1",
        )

    assert tuple(tmp_path.rglob("*.mp4")) == ()
    assert recorder.stop_count == 2


def test_failed_artifact_rollback_raises_distinct_error(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    service = _service(recorder, tmp_path, store=_FailRollbackStore(tmp_path))
    boundary = service.begin()

    with pytest.raises(ScreenRecordingRollbackError, match="Could not roll back"):
        service.checkpoint(
            boundary,
            ScreenRecordingCheckpointKind.SCENARIO,
            "scenario_1",
        )
