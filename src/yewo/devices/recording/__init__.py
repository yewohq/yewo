"""Checkpointed screen recording and immutable video evidence."""

from yewo.devices.recording.appium import AppiumScreenRecorder
from yewo.devices.recording.capture import (
    ScreenRecordingCapture,
    ScreenRecordingStart,
)
from yewo.devices.recording.errors import (
    ScreenRecordingError,
    ScreenRecordingRollbackError,
)
from yewo.devices.recording.lease import ScreenRecordingLease
from yewo.devices.recording.policy import ScreenRecordingPolicy
from yewo.devices.recording.video import (
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
    ScreenRecordingRemoteCleanupState,
    ScreenRecordingRemoteCleanupReceipt,
    ScreenRecordingSegmentState,
    validate_recording_checkpoint,
)
from yewo.devices.recording.recorder import ScreenRecorder
from yewo.devices.recording.service import ScreenRecordingService
from yewo.devices.recording.store import ScreenRecordingStore

__all__ = [
    "AppiumScreenRecorder",
    "ScreenRecorder",
    "ScreenRecordingArtifactReference",
    "ScreenRecordingBoundary",
    "ScreenRecordingBoundaryState",
    "ScreenRecordingCapture",
    "ScreenRecordingCaptureMethod",
    "ScreenRecordingCheckpointKind",
    "ScreenRecordingCheckpointReceipt",
    "ScreenRecordingError",
    "ScreenRecordingLease",
    "ScreenRecordingPolicy",
    "ScreenRecordingVideoMetadata",
    "ScreenRecordingRollbackError",
    "ScreenRecordingRemoteCleanupState",
    "ScreenRecordingRemoteCleanupReceipt",
    "ScreenRecordingSegmentState",
    "ScreenRecordingStart",
    "ScreenRecordingService",
    "ScreenRecordingStore",
    "validate_recording_checkpoint",
    "read_screen_recording_metadata",
]
