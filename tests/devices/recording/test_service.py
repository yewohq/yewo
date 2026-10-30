"""Tests for checkpointed screen-recording ownership."""

from collections.abc import Callable, Sequence
from datetime import datetime, timezone
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
from yewo.devices.recording import (
    ScreenRecordingBoundaryState,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
    ScreenRecordingError,
    ScreenRecordingMask,
    ScreenRecordingPolicy,
    ScreenRecordingRedactionProvenance,
    ScreenRecordingRedactionResult,
    ScreenRecordingRedactor,
    ScreenRecordingRemoteCleanupState,
    ScreenRecordingRollbackError,
    ScreenRecordingSegmentState,
    ScreenRecordingService,
    ScreenRecordingStore,
    ScreenRecordingVideoMetadata,
    recording_mask_set_digest,
)
from yewo.observations import RedactionRegion
from .support import (
    RecorderStub as _RecorderStub,
    RemoteRecorderStub as _RemoteRecorderStub,
)


CAPTURE_TIME = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)
SCENARIO_VIDEO = b"\x00\x00\x00\x18ftypmp42scenario"
FINAL_VIDEO = b"\x00\x00\x00\x18ftypmp42final"
REDACTED_VIDEO = b"\x00\x00\x00\x18ftypmp42redacted"


class _RedactorStub:
    def __init__(self, error: ScreenRecordingError | None = None) -> None:
        self.error = error
        self.masks: tuple[ScreenRecordingMask, ...] = ()
        self.mask_batches: list[tuple[ScreenRecordingMask, ...]] = []

    def redact(
        self,
        source: Path,
        destination: Path,
        masks: Sequence[ScreenRecordingMask],
    ) -> ScreenRecordingRedactionResult:
        assert source.read_bytes() in {SCENARIO_VIDEO, FINAL_VIDEO}
        self.masks = tuple(masks)
        self.mask_batches.append(self.masks)
        if self.error is not None:
            raise self.error
        destination.write_bytes(REDACTED_VIDEO)
        return ScreenRecordingRedactionResult(
            ScreenRecordingVideoMetadata(16, 9, 30.0, 1.0),
            ScreenRecordingRedactionProvenance(
                policy_id="recording-region-mask",
                mask_count=len(self.masks),
                mask_set_sha256=recording_mask_set_digest(self.masks),
                input_sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
            ),
        )


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
    redactor: ScreenRecordingRedactor | None = None,
    initial_masks: Sequence[ScreenRecordingMask] = (),
    monotonic_factory: Callable[[], float] = lambda: 0.0,
) -> ScreenRecordingService:
    identifiers = (f"record_{value}" for value in count(1))
    artifacts = (f"artifact_{value}" for value in count(1))
    return ScreenRecordingService(
        recorder,
        _device(),
        _application(),
        store or ScreenRecordingStore(root, identifier_factory=artifacts.__next__),
        policy=policy,
        redactor=redactor,
        initial_masks=initial_masks,
        metadata_reader=lambda path: ScreenRecordingVideoMetadata(16, 9, 30.0, 1.0),
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
    cleanup_records = tuple(tmp_path.glob("assets/recordings/*/*-remote-cleanup-*.json"))
    assert len(cleanup_records) == 1


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
    assert tuple(tmp_path.glob("assets/recordings/*/*-remote-cleanup-*.json")) == ()


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


def test_time_bound_redaction_publishes_only_processed_video(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO,))
    redactor = _RedactorStub()
    monotonic_values = iter((10.0, 12.5, 20.0))
    service = _service(
        recorder,
        tmp_path,
        redactor=redactor,
        monotonic_factory=monotonic_values.__next__,
    )
    boundary = service.begin()
    service.add_redaction_region(
        RedactionRegion(left=2, top=3, right=20, bottom=30),
        80,
        40,
    )

    receipt = service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )

    assert redactor.masks[0].starts_at_seconds == 2.5
    assert receipt.artifact is not None
    assert receipt.artifact.redaction is not None
    assert receipt.artifact.redaction.mask_count == 1
    assert receipt.artifact.sensitivity is ArtifactSensitivity.DECLARED_SENSITIVE
    assert (tmp_path / receipt.artifact.path).read_bytes() == REDACTED_VIDEO


def test_initial_mask_covers_the_first_recording_segment(tmp_path: Path) -> None:
    recorder = _RecorderStub((FINAL_VIDEO,))
    redactor = _RedactorStub()
    mask = ScreenRecordingMask(
        region=RedactionRegion(left=1, top=2, right=8, bottom=9),
        source_width_px=16,
        source_height_px=10,
        starts_at_seconds=0,
    )
    service = _service(
        recorder,
        tmp_path,
        redactor=redactor,
        initial_masks=(mask,),
    )

    receipt = service.finish(service.begin())

    assert receipt.artifact is not None
    assert redactor.masks == (mask,)


def test_declared_sensitive_segment_without_masks_fails_closed(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO,))
    service = _service(
        recorder,
        tmp_path,
        policy=ScreenRecordingPolicy(sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE),
    )
    boundary = service.begin()

    receipt = service.finish(boundary)

    assert receipt.state is ScreenRecordingSegmentState.FAILED
    assert receipt.artifact is None
    assert tuple(tmp_path.rglob("*.mp4")) == ()


def test_redaction_failure_never_publishes_raw_video(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO,))
    redactor = _RedactorStub(ScreenRecordingError("redaction failed"))
    service = _service(recorder, tmp_path, redactor=redactor)
    boundary = service.begin()
    service.add_redaction_region(
        RedactionRegion(left=0, top=0, right=10, bottom=10),
        10,
        10,
    )

    receipt = service.finish(boundary)

    assert receipt.state is ScreenRecordingSegmentState.FAILED
    assert receipt.artifact is None
    assert tuple(tmp_path.rglob("*.mp4")) == ()


def test_processing_failure_restarts_only_after_the_recorder_is_safe(
    tmp_path: Path,
) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    redactor = _RedactorStub(ScreenRecordingError("redaction failed"))
    service = _service(recorder, tmp_path, redactor=redactor)
    boundary = service.begin()
    service.add_redaction_region(
        RedactionRegion(left=0, top=0, right=10, bottom=10),
        10,
        10,
    )

    receipt = service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )

    assert receipt.state is ScreenRecordingSegmentState.FAILED
    assert receipt.continuation_started is True
    assert recorder.start_limits == [1_800, 1_800]


def test_masks_continue_at_zero_after_a_checkpoint(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    redactor = _RedactorStub()
    service = _service(recorder, tmp_path, redactor=redactor)
    boundary = service.begin()
    service.add_redaction_region(
        RedactionRegion(left=0, top=0, right=10, bottom=10),
        10,
        10,
    )

    service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )
    service.finish(boundary)

    assert len(redactor.mask_batches) == 2
    assert redactor.mask_batches[1][0].starts_at_seconds == 0.0


def test_caller_can_clear_masks_before_the_next_segment(tmp_path: Path) -> None:
    recorder = _RecorderStub((SCENARIO_VIDEO, FINAL_VIDEO))
    redactor = _RedactorStub()
    service = _service(recorder, tmp_path, redactor=redactor)
    boundary = service.begin()
    service.add_redaction_region(
        RedactionRegion(left=0, top=0, right=10, bottom=10),
        10,
        10,
    )
    service.clear_redaction_regions()
    service.checkpoint(
        boundary,
        ScreenRecordingCheckpointKind.SCENARIO,
        "scenario_1",
    )

    final = service.finish(boundary)

    assert final.artifact is not None
    assert final.artifact.redaction is None
    assert len(redactor.mask_batches) == 1


def test_mask_registration_requires_an_active_segment(tmp_path: Path) -> None:
    service = _service(
        _RecorderStub(),
        tmp_path,
        policy=ScreenRecordingPolicy(enabled=False),
    )
    service.begin()

    with pytest.raises(ScreenRecordingError, match="active"):
        service.add_redaction_region(
            RedactionRegion(left=0, top=0, right=10, bottom=10),
            10,
            10,
        )


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
