"""Validate and persist one owned recording capture."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from yewo.devices.recording.capture import ScreenRecordingCapture
from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.policy import ScreenRecordingPolicy
from yewo.devices.recording.video import ScreenRecordingVideoMetadata
from yewo.devices.recording.records import (
    ScreenRecordingArtifactReference,
    ScreenRecordingBoundary,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
)
from yewo.devices.recording.store import ScreenRecordingStore


class ScreenRecordingArtifactProcessor:
    """Own processing between a recorder capture and immutable evidence."""

    def __init__(
        self,
        store: ScreenRecordingStore,
        policy: ScreenRecordingPolicy,
        metadata_reader: Callable[[Path], ScreenRecordingVideoMetadata],
    ) -> None:
        self._store = store
        self._policy = policy
        self._metadata_reader = metadata_reader

    def persist(
        self,
        capture: ScreenRecordingCapture,
        boundary: ScreenRecordingBoundary,
        segment_index: int,
        checkpoint_kind: ScreenRecordingCheckpointKind,
        checkpoint_id: str | None,
        capture_method: ScreenRecordingCaptureMethod,
    ) -> ScreenRecordingArtifactReference:
        """Process one local file without loading its video into memory."""
        source = capture.path
        try:
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
                retention=self._policy.retention,
            )
        except ScreenRecordingError:
            raise
