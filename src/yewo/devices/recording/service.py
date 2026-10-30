"""Checkpoint one screen-recording interval into durable segments."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from datetime import datetime, timezone
import math
from pathlib import Path
import time

from pydantic import ValidationError

from yewo.applications.records import ApplicationDescriptor
from yewo.artifacts.records import (
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    new_record_identifier,
    validate_record_identifier,
)
from yewo.devices.inventory import DeviceDescriptor
from yewo.devices.recording.capture import (
    ScreenRecordingCapture,
    ScreenRecordingStart,
)
from yewo.devices.recording.errors import (
    ScreenRecordingError,
    ScreenRecordingRollbackError,
)
from yewo.devices.recording.policy import ScreenRecordingPolicy
from yewo.devices.recording.processor import ScreenRecordingArtifactProcessor
from yewo.devices.recording.redaction import (
    FfmpegScreenRecordingRedactor,
    ScreenRecordingRedactor,
    ScreenRecordingVideoMetadata,
    read_screen_recording_metadata,
)
from yewo.devices.recording.records import (
    ScreenRecordingArtifactReference,
    ScreenRecordingBoundary,
    ScreenRecordingBoundaryState,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
    ScreenRecordingCheckpointReceipt,
    ScreenRecordingMask,
    ScreenRecordingRemoteCleanupState,
    ScreenRecordingRemoteCleanupReceipt,
    ScreenRecordingSegmentState,
    validate_recording_checkpoint,
)
from yewo.devices.recording.recorder import ScreenRecorder, ScreenRecorderBoundary
from yewo.devices.recording.store import ScreenRecordingStore
from yewo.observations.records import RedactionRegion


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScreenRecordingService:
    """Start, checkpoint, and finish one screen-recording interval."""

    def __init__(
        self,
        recorder: ScreenRecorder,
        device: DeviceDescriptor,
        application: ApplicationDescriptor,
        store: ScreenRecordingStore,
        *,
        policy: ScreenRecordingPolicy | None = None,
        redactor: ScreenRecordingRedactor | None = None,
        metadata_reader: Callable[[Path], ScreenRecordingVideoMetadata] = (
            read_screen_recording_metadata
        ),
        identifier_factory: Callable[[], str] = new_record_identifier,
        time_factory: Callable[[], datetime] = _utc_now,
        monotonic_factory: Callable[[], float] = time.monotonic,
    ) -> None:
        if device.platform is not application.platform:
            raise ValueError("The recording device and application must share a platform.")
        self._recorder = ScreenRecorderBoundary(recorder)
        self._device = device
        self._application = application
        self._store = store
        self._policy = policy or ScreenRecordingPolicy()
        self._processor = ScreenRecordingArtifactProcessor(
            store,
            self._policy,
            redactor or FfmpegScreenRecordingRedactor(),
            metadata_reader,
        )
        self._identifier_factory = identifier_factory
        self._time_factory = time_factory
        self._monotonic_factory = monotonic_factory
        self._boundary: ScreenRecordingBoundary | None = None
        self._segment_started_at: datetime | None = None
        self._segment_start: ScreenRecordingStart | None = None
        self._active = False
        self._segment_monotonic_started: float | None = None
        self._segment_masks: list[ScreenRecordingMask] = []
        self._persistent_masks: list[ScreenRecordingMask] = []
        self._next_segment_index = 1
        self._completed_checkpoints: set[tuple[ScreenRecordingCheckpointKind, str]] = set()

    def begin(self) -> ScreenRecordingBoundary:
        """Start the first segment and persist its recording boundary."""
        if self._boundary is not None:
            raise ScreenRecordingError("A screen-recording boundary is already active.")
        recording_id = self._new_identifier("recording")
        started_at = self._capture_time("boundary")
        state, reason, start = self._start_first_segment()
        try:
            boundary = self._build_boundary(
                recording_id,
                started_at,
                state,
                reason,
                start,
            )
            self._store.persist_boundary(boundary)
        except ScreenRecordingError as error:
            self._stop_after_failed_start(error)
            raise
        self._boundary = boundary
        return boundary

    def checkpoint(
        self,
        boundary: ScreenRecordingBoundary,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        checkpoint_id: str | None,
    ) -> ScreenRecordingCheckpointReceipt:
        """Close the current segment and optionally start the next one."""
        self._require_active_boundary(boundary)
        try:
            validated_checkpoint_id = validate_recording_checkpoint(
                checkpoint_kind,
                checkpoint_id,
            )
        except (ValidationError, ValueError) as error:
            raise ScreenRecordingError("The recording checkpoint is invalid.") from error
        checkpoint_key = (
            checkpoint_kind,
            validated_checkpoint_id or checkpoint_kind.value,
        )
        if checkpoint_key in self._completed_checkpoints:
            raise ScreenRecordingError("The recording checkpoint was already completed.")
        receipt = self._close_segment(
            boundary,
            checkpoint_kind,
            validated_checkpoint_id,
        )
        self._completed_checkpoints.add(checkpoint_key)
        self._next_segment_index += 1
        return receipt

    def finish(self, boundary: ScreenRecordingBoundary) -> ScreenRecordingCheckpointReceipt:
        """Close the final segment and release the recording boundary."""
        try:
            return self.checkpoint(
                boundary,
                ScreenRecordingCheckpointKind.SESSION_END,
                None,
            )
        finally:
            if self._boundary == boundary:
                self._boundary = None
                self._clear_active_segment()

    def add_redaction_region(
        self,
        region: RedactionRegion,
        source_width_px: int,
        source_height_px: int,
    ) -> None:
        """Start a mask now. Register it before sensitive pixels appear."""
        monotonic_started = self._segment_monotonic_started
        if not self._active or monotonic_started is None:
            raise ScreenRecordingError(
                "A recording mask requires an active screen-recording segment."
            )
        elapsed_seconds = self._monotonic_factory() - monotonic_started
        if not math.isfinite(elapsed_seconds):
            raise ScreenRecordingError("The recording mask time is invalid.")
        try:
            mask = ScreenRecordingMask(
                region=region,
                source_width_px=source_width_px,
                source_height_px=source_height_px,
                starts_at_seconds=max(0.0, elapsed_seconds),
            )
        except ValidationError as error:
            raise ScreenRecordingError("The recording redaction region is invalid.") from error
        key = self._mask_key(mask)
        self._segment_masks = [
            current for current in self._segment_masks if self._mask_key(current) != key
        ]
        self._persistent_masks = [
            current for current in self._persistent_masks if self._mask_key(current) != key
        ]
        self._segment_masks.append(mask)
        self._persistent_masks.append(mask.model_copy(update={"starts_at_seconds": 0.0}))

    def clear_redaction_regions(self) -> None:
        """End active masks now and stop carrying them into later segments."""
        monotonic_started = self._segment_monotonic_started
        if self._active and monotonic_started is not None:
            elapsed_seconds = max(0.0, self._monotonic_factory() - monotonic_started)
            if not math.isfinite(elapsed_seconds):
                raise ScreenRecordingError("The recording mask time is invalid.")
            self._segment_masks = [
                (
                    mask
                    if mask.ends_at_seconds is not None
                    else mask.model_copy(
                        update={
                            "ends_at_seconds": max(
                                mask.starts_at_seconds,
                                elapsed_seconds,
                            )
                        }
                    )
                )
                for mask in self._segment_masks
            ]
        self._persistent_masks = []

    def _start_first_segment(
        self,
    ) -> tuple[
        ScreenRecordingBoundaryState,
        str | None,
        ScreenRecordingStart | None,
    ]:
        if not self._policy.enabled:
            return (
                ScreenRecordingBoundaryState.DISABLED,
                "The recording policy disables screen recording.",
                None,
            )
        try:
            start = self._start_segment()
        except ScreenRecordingError:
            return (
                ScreenRecordingBoundaryState.UNAVAILABLE,
                "The first screen-recording segment did not start.",
                None,
            )
        return ScreenRecordingBoundaryState.ACTIVE, None, start

    def _start_segment(self) -> ScreenRecordingStart:
        start = self._recorder.start(
            self._policy.maximum_segment_seconds,
            self._policy.maximum_stop_seconds,
            self._policy.maximum_pull_seconds,
        )
        self._active = True
        self._segment_start = start
        try:
            self._segment_started_at = self._capture_time("segment start")
            self._segment_monotonic_started = self._monotonic_factory()
            if not math.isfinite(self._segment_monotonic_started):
                raise ScreenRecordingError("The recording segment clock is invalid.")
            self._segment_masks = [
                mask.model_copy(update={"starts_at_seconds": 0.0})
                for mask in self._persistent_masks
            ]
        except ScreenRecordingError as error:
            self._stop_after_failed_start(error)
            raise
        return start

    def _close_segment(
        self,
        boundary: ScreenRecordingBoundary,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        checkpoint_id: str | None,
    ) -> ScreenRecordingCheckpointReceipt:
        segment_started_at = self._segment_started_at or self._capture_time("empty segment start")
        artifact: ScreenRecordingArtifactReference | None = None
        capture: ScreenRecordingCapture | None = None
        state = ScreenRecordingSegmentState.NO_RECORDING
        reason: str | None = "No screen-recording segment was active."
        continuation_started = False
        try:
            if self._active:
                masks = tuple(self._segment_masks)
                try:
                    capture = self._recorder.stop()
                    self._clear_active_segment()
                    artifact = self._processor.persist(
                        capture,
                        masks,
                        boundary,
                        self._next_segment_index,
                        checkpoint_kind,
                        checkpoint_id,
                        self._recorder.capture_method,
                    )
                    state = ScreenRecordingSegmentState.CAPTURED
                    reason = None
                except ScreenRecordingError:
                    self._clear_active_segment()
                    state = ScreenRecordingSegmentState.FAILED
                    reason = (
                        "The screen-recording segment could not be stopped, redacted, and "
                        "persisted."
                    )
            continuation_started, continuation_reason = self._continue_after_checkpoint(
                checkpoint_kind,
                allow_start=(
                    state is not ScreenRecordingSegmentState.FAILED
                    or self._recorder.can_start_after_failure
                ),
            )
            receipt = self._build_receipt(
                boundary,
                checkpoint_kind,
                checkpoint_id,
                segment_started_at,
                state,
                artifact,
                reason,
                continuation_started,
                continuation_reason,
            )
            self._store.persist_receipt(receipt)
            receipt = self._acknowledge_remote_capture(receipt, capture)
        except ScreenRecordingError as error:
            self._rollback_checkpoint(artifact, continuation_started, error)
            raise
        finally:
            if capture is not None:
                with suppress(ScreenRecordingError):
                    capture.close()
        return receipt

    def _continue_after_checkpoint(
        self,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        *,
        allow_start: bool,
    ) -> tuple[bool, str | None]:
        if checkpoint_kind is ScreenRecordingCheckpointKind.SESSION_END:
            return False, None
        if not self._policy.enabled:
            return False, "The recording policy disables screen recording."
        if not allow_start:
            return False, "The prior recorder state does not permit a safe restart."
        try:
            self._start_segment()
        except ScreenRecordingError:
            return False, "The next screen-recording segment did not start."
        return True, None

    def _build_receipt(
        self,
        boundary: ScreenRecordingBoundary,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        checkpoint_id: str | None,
        started_at: datetime,
        state: ScreenRecordingSegmentState,
        artifact: ScreenRecordingArtifactReference | None,
        reason: str | None,
        continuation_started: bool,
        continuation_reason: str | None,
    ) -> ScreenRecordingCheckpointReceipt:
        try:
            return ScreenRecordingCheckpointReceipt(
                receipt_id=self._new_identifier("receipt"),
                recording_id=boundary.recording_id,
                segment_index=self._next_segment_index,
                checkpoint_kind=checkpoint_kind,
                checkpoint_id=checkpoint_id,
                started_at=started_at,
                completed_at=self._capture_time("checkpoint completion"),
                state=state,
                artifact=artifact,
                reason=reason,
                remote_cleanup_state=(
                    ScreenRecordingRemoteCleanupState.PENDING
                    if artifact is not None
                    and artifact.capture_method is ScreenRecordingCaptureMethod.REMOTE_FILE
                    else ScreenRecordingRemoteCleanupState.NOT_APPLICABLE
                ),
                continuation_started=continuation_started,
                continuation_reason=continuation_reason,
            )
        except ValidationError as error:
            raise ScreenRecordingError("The recording checkpoint receipt is invalid.") from error

    def _acknowledge_remote_capture(
        self,
        receipt: ScreenRecordingCheckpointReceipt,
        capture: ScreenRecordingCapture | None,
    ) -> ScreenRecordingCheckpointReceipt:
        if (
            capture is None
            or not capture.requires_acknowledgement
            or receipt.remote_cleanup_state is not ScreenRecordingRemoteCleanupState.PENDING
        ):
            return receipt
        try:
            capture.accept()
            artifact = receipt.artifact
            if artifact is None:
                raise ScreenRecordingError(
                    "Remote recording cleanup requires a persisted artifact."
                )
            cleanup_receipt = ScreenRecordingRemoteCleanupReceipt(
                cleanup_id=self._new_identifier("remote cleanup"),
                checkpoint_receipt_id=receipt.receipt_id,
                recording_id=receipt.recording_id,
                segment_index=receipt.segment_index,
                artifact_id=artifact.artifact_id,
                released_at=self._capture_time("remote cleanup"),
            )
            self._store.persist_remote_cleanup(cleanup_receipt)
            released_receipt = receipt.model_copy(
                update={"remote_cleanup_state": ScreenRecordingRemoteCleanupState.RELEASED}
            )
        except (ScreenRecordingError, ValidationError):
            return receipt
        return released_receipt

    def _rollback_checkpoint(
        self,
        artifact: ScreenRecordingArtifactReference | None,
        continuation_started: bool,
        original_error: ScreenRecordingError,
    ) -> None:
        failures: list[ScreenRecordingError] = []
        if continuation_started:
            try:
                capture = self._recorder.stop()
                capture.close()
            except ScreenRecordingError as error:
                failures.append(error)
            self._active = False
            self._segment_started_at = None
            self._segment_start = None
            self._segment_monotonic_started = None
            self._segment_masks = []
        if artifact is not None:
            try:
                self._store.discard(artifact)
            except ScreenRecordingError as error:
                failures.append(error)
        if failures:
            failure = ScreenRecordingRollbackError(
                f"Could not roll back {len(failures)} incomplete recording resource(s)."
            )
            failure.add_note(". ".join(str(error) for error in failures))
            raise failure from original_error

    def _stop_after_failed_start(self, original_error: ScreenRecordingError) -> None:
        if not self._active:
            return
        try:
            capture = self._recorder.stop()
            capture.close()
        except ScreenRecordingError as stop_error:
            original_error.add_note(f"Recording cleanup also failed: {stop_error}")
        self._active = False
        self._segment_started_at = None
        self._segment_start = None
        self._segment_monotonic_started = None
        self._segment_masks = []

    def _clear_active_segment(self) -> None:
        self._active = False
        self._segment_started_at = None
        self._segment_start = None
        self._segment_monotonic_started = None
        self._segment_masks = []

    def _build_boundary(
        self,
        recording_id: str,
        started_at: datetime,
        state: ScreenRecordingBoundaryState,
        reason: str | None,
        start: ScreenRecordingStart | None,
    ) -> ScreenRecordingBoundary:
        try:
            return ScreenRecordingBoundary(
                recording_id=recording_id,
                device=EvidenceDeviceIdentity(
                    target_id=self._device.target_id,
                    platform=self._device.platform,
                    kind=self._device.kind,
                    operating_system_version=self._device.operating_system_version,
                    model=self._device.device_model,
                    architecture=self._device.architecture,
                ),
                application=EvidenceApplicationIdentity(
                    application_id=self._application.application_id,
                    platform=self._application.platform,
                    version=self._application.version,
                    build=self._application.build,
                    artifact_sha256=self._application.artifact_sha256,
                ),
                started_at=started_at,
                requested_maximum_seconds=self._policy.maximum_segment_seconds,
                effective_maximum_seconds=(
                    start.effective_maximum_seconds if start is not None else None
                ),
                maximum_stop_seconds=self._policy.maximum_stop_seconds,
                maximum_pull_seconds=self._policy.maximum_pull_seconds,
                state=state,
                reason=reason,
            )
        except ValidationError as error:
            raise ScreenRecordingError("The recording boundary is invalid.") from error

    @staticmethod
    def _mask_key(mask: ScreenRecordingMask) -> tuple[object, ...]:
        region = mask.region
        return (
            region.left,
            region.top,
            region.right,
            region.bottom,
            mask.source_width_px,
            mask.source_height_px,
        )

    def _new_identifier(self, record_kind: str) -> str:
        try:
            return validate_record_identifier(self._identifier_factory())
        except ValidationError as error:
            raise ScreenRecordingError(
                f"The recording {record_kind} identifier is invalid."
            ) from error

    def _capture_time(self, record_kind: str) -> datetime:
        captured_at = self._time_factory()
        if captured_at.utcoffset() is None:
            raise ScreenRecordingError(f"The recording {record_kind} time must include a timezone.")
        return captured_at

    def _require_active_boundary(self, boundary: ScreenRecordingBoundary) -> None:
        if self._boundary != boundary:
            raise ScreenRecordingError("The screen-recording boundary is not active.")
