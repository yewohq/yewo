"""Validate, redact, and persist one owned recording capture."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from yewo.artifacts.records import ArtifactSensitivity
from yewo.devices.recording.capture import ScreenRecordingCapture
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.policy import ScreenRecordingPolicy
from yewo.devices.recording.redaction import (
    ScreenRecordingRedactionResult,
    ScreenRecordingRedactor,
    ScreenRecordingVideoMetadata,
    recording_mask_set_digest,
)
from yewo.devices.recording.records import (
    ScreenRecordingArtifactReference,
    ScreenRecordingBoundary,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
    ScreenRecordingMask,
    ScreenRecordingRedactionProvenance,
    ScreenRecordingRedactionState,
)
from yewo.devices.recording.store import ScreenRecordingStore


class ScreenRecordingArtifactProcessor:
    """Own processing between a recorder capture and immutable evidence."""

    def __init__(
        self,
        store: ScreenRecordingStore,
        policy: ScreenRecordingPolicy,
        redactor: ScreenRecordingRedactor,
        metadata_reader: Callable[[Path], ScreenRecordingVideoMetadata],
    ) -> None:
        self._store = store
        self._policy = policy
        self._redactor = redactor
        self._metadata_reader = metadata_reader

    def persist(
        self,
        capture: ScreenRecordingCapture,
        masks: tuple[ScreenRecordingMask, ...],
        boundary: ScreenRecordingBoundary,
        segment_index: int,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        checkpoint_id: str | None,
        capture_method: ScreenRecordingCaptureMethod,
    ) -> ScreenRecordingArtifactReference:
        """Process one local file without loading its video into memory."""
        source = capture.path
        redaction_state = ScreenRecordingRedactionState.RAW
        redaction: ScreenRecordingRedactionProvenance | None = None
        sensitivity = self._policy.sensitivity
        temporary: TemporaryDirectory[str] | None = None
        try:
            if masks:
                temporary = TemporaryDirectory(prefix="yewo-recording-redaction-")
                source = Path(temporary.name) / "redacted.mp4"
                result = self._redact(capture.path, source, masks)
                metadata = result.metadata
                redaction_state = ScreenRecordingRedactionState.REDACTED
                redaction = result.provenance
                sensitivity = ArtifactSensitivity.DECLARED_SENSITIVE
            else:
                if self._policy.sensitivity is ArtifactSensitivity.DECLARED_SENSITIVE:
                    raise ScreenRecordingError(
                        "A declared-sensitive recording has no approved redaction masks."
                    )
                metadata = self._metadata_reader(source)
            tolerance = max(0.5, 2 / metadata.frames_per_second)
            return self._store.persist_artifact(
                source,
                boundary=boundary,
                segment_index=segment_index,
                checkpoint_kind=checkpoint_kind,
                checkpoint_id=checkpoint_id,
                capture_method=capture_method,
                duration_seconds=metadata.duration_seconds,
                requested_maximum_seconds=capture.start.requested_maximum_seconds,
                effective_maximum_seconds=capture.start.effective_maximum_seconds,
                maximum_stop_seconds=capture.start.maximum_stop_seconds,
                maximum_pull_seconds=capture.start.maximum_pull_seconds,
                reached_effective_limit=(
                    metadata.duration_seconds >= capture.start.effective_maximum_seconds - tolerance
                ),
                redaction_state=redaction_state,
                redaction=redaction,
                sensitivity=sensitivity,
                retention=self._policy.retention,
            )
        finally:
            if temporary is not None:
                temporary.cleanup()

    def _redact(
        self,
        source: Path,
        destination: Path,
        masks: tuple[ScreenRecordingMask, ...],
    ) -> ScreenRecordingRedactionResult:
        try:
            result = self._redactor.redact(source, destination, masks)
        except ScreenRecordingError:
            raise
        except Exception as error:  # noqa: BLE001 - injected redactor boundary
            raise ScreenRecordingError("The recording redactor failed.") from error
        if not isinstance(result, ScreenRecordingRedactionResult):
            raise ScreenRecordingError("The recording redactor returned an invalid result.")
        if (
            result.provenance.mask_count != len(masks)
            or result.provenance.mask_set_sha256 != recording_mask_set_digest(masks)
            or result.provenance.input_sensitivity is not ArtifactSensitivity.DECLARED_SENSITIVE
        ):
            raise ScreenRecordingError("The recording redactor returned invalid provenance.")
        return result
