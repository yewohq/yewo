"""Tests for immutable screen-recording records."""

from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError
import pytest

from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
)
from yewo.devices import DeviceKind, DevicePlatform
from yewo.devices.recording import (
    ScreenRecordingArtifactReference,
    ScreenRecordingCaptureMethod,
    ScreenRecordingCheckpointKind,
    ScreenRecordingCheckpointReceipt,
    ScreenRecordingRedactionProvenance,
    ScreenRecordingRedactionState,
    ScreenRecordingRemoteCleanupState,
    ScreenRecordingSegmentState,
)


CAPTURE_TIME = datetime(2026, 10, 30, tzinfo=timezone.utc)
DEVICE = EvidenceDeviceIdentity(
    target_id="emulator-5554",
    platform=DevicePlatform.ANDROID,
    kind=DeviceKind.VIRTUAL,
)
APPLICATION = EvidenceApplicationIdentity(
    platform=DevicePlatform.ANDROID,
    application_id="com.example.app",
)


def _artifact() -> ScreenRecordingArtifactReference:
    return ScreenRecordingArtifactReference(
        artifact_id="artifact_1",
        recording_id="recording_1",
        segment_index=1,
        checkpoint_kind=ScreenRecordingCheckpointKind.SCENARIO,
        checkpoint_id="scenario_1",
        sha256="0" * 64,
        path=Path("assets/recordings/recording_1/segment.mp4"),
        byte_count=1,
        duration_seconds=1.0,
        requested_maximum_seconds=1_800,
        effective_maximum_seconds=180,
        maximum_stop_seconds=10.0,
        maximum_pull_seconds=120.0,
        reached_effective_limit=False,
        capture_method=ScreenRecordingCaptureMethod.DRIVER_RESPONSE,
        redaction_state=ScreenRecordingRedactionState.RAW,
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def test_captured_receipt_requires_matching_artifact_identity() -> None:
    artifact = _artifact().model_copy(update={"segment_index": 2})

    with pytest.raises(ValidationError, match="identify one segment"):
        ScreenRecordingCheckpointReceipt(
            receipt_id="receipt_1",
            recording_id="recording_1",
            segment_index=1,
            checkpoint_kind=ScreenRecordingCheckpointKind.SCENARIO,
            checkpoint_id="scenario_1",
            started_at=CAPTURE_TIME,
            completed_at=CAPTURE_TIME,
            state=ScreenRecordingSegmentState.CAPTURED,
            artifact=artifact,
            remote_cleanup_state=ScreenRecordingRemoteCleanupState.NOT_APPLICABLE,
            continuation_started=True,
        )


def test_session_end_cannot_claim_a_continuation() -> None:
    with pytest.raises(ValidationError, match="cannot continue"):
        ScreenRecordingCheckpointReceipt(
            receipt_id="receipt_1",
            recording_id="recording_1",
            segment_index=1,
            checkpoint_kind=ScreenRecordingCheckpointKind.SESSION_END,
            started_at=CAPTURE_TIME,
            completed_at=CAPTURE_TIME,
            state=ScreenRecordingSegmentState.NO_RECORDING,
            reason="not active",
            remote_cleanup_state=ScreenRecordingRemoteCleanupState.NOT_APPLICABLE,
            continuation_started=True,
        )


def test_redacted_recording_retains_input_sensitivity() -> None:
    values = _artifact().model_dump()
    values.update(
        {
            "redaction_state": ScreenRecordingRedactionState.REDACTED,
            "redaction": ScreenRecordingRedactionProvenance(
                policy_id="region-mask",
                mask_count=1,
                mask_set_sha256="1" * 64,
                input_sensitivity=ArtifactSensitivity.DECLARED_SENSITIVE,
            ),
            "sensitivity": ArtifactSensitivity.DEVICE_CONTENT,
        }
    )
    with pytest.raises(ValidationError, match="retain its input sensitivity"):
        ScreenRecordingArtifactReference.model_validate(values)


def test_declared_sensitive_recording_cannot_remain_raw() -> None:
    values = _artifact().model_dump()
    values["sensitivity"] = ArtifactSensitivity.DECLARED_SENSITIVE

    with pytest.raises(ValidationError, match="cannot remain raw"):
        ScreenRecordingArtifactReference.model_validate(values)
