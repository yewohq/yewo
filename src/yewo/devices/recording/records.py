"""Immutable records for checkpointed screen recordings."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    RecordIdentifier,
    Sha256Digest,
    validate_record_identifier,
)
from yewo.observations.records import RedactionRegion


class ScreenRecordingBoundaryState(StrEnum):
    """Whether the first recording segment started."""

    ACTIVE = "active"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class ScreenRecordingCheckpointKind(StrEnum):
    """The execution boundary that closes one video segment."""

    SCENARIO = "scenario"
    SESSION_END = "session_end"


class ScreenRecordingSegmentState(StrEnum):
    """What one recording checkpoint retained."""

    CAPTURED = "captured"
    NO_RECORDING = "no_recording"
    FAILED = "failed"


class ScreenRecordingRemoteCleanupState(StrEnum):
    """Whether a durable receipt released its remote recording source."""

    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"
    RELEASED = "released"


class ScreenRecordingCaptureMethod(StrEnum):
    """How retained video bytes reached the host."""

    DRIVER_RESPONSE = "driver_response"
    REMOTE_FILE = "remote_file"


class ScreenRecordingRedactionState(StrEnum):
    """Whether retained video pixels were transformed for safety."""

    RAW = "raw"
    REDACTED = "redacted"


class ScreenRecordingMask(BaseModel):
    """One sensitive region that remains covered for the rest of a segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    region: RedactionRegion
    source_width_px: int = Field(gt=0)
    source_height_px: int = Field(gt=0)
    starts_at_seconds: float = Field(ge=0)
    ends_at_seconds: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_region_bounds(self) -> Self:
        if self.region.right > self.source_width_px or self.region.bottom > self.source_height_px:
            raise ValueError("A recording mask region is outside its source frame.")
        if self.ends_at_seconds is not None and self.ends_at_seconds < self.starts_at_seconds:
            raise ValueError("A recording mask cannot end before it starts.")
        return self


class ScreenRecordingRedactionProvenance(BaseModel):
    """Policy evidence for pixels produced by recording redaction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: RecordIdentifier
    mask_count: int = Field(gt=0)
    mask_set_sha256: Sha256Digest
    input_sensitivity: ArtifactSensitivity


def validate_recording_checkpoint(
    checkpoint_kind: ScreenRecordingCheckpointKind,
    checkpoint_id: str | None,
) -> str | None:
    """Validate the identity required by a recording checkpoint."""
    if checkpoint_kind is ScreenRecordingCheckpointKind.SESSION_END:
        if checkpoint_id is not None:
            raise ValueError("A session-end recording checkpoint cannot have an ID.")
        return None
    if checkpoint_id is None:
        raise ValueError("A scenario recording checkpoint requires an ID.")
    return validate_record_identifier(checkpoint_id)


class ScreenRecordingBoundary(BaseModel):
    """Start of one driver-owned screen-recording interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    recording_id: RecordIdentifier
    device: EvidenceDeviceIdentity
    application: EvidenceApplicationIdentity
    started_at: datetime
    requested_maximum_seconds: int = Field(gt=0)
    effective_maximum_seconds: int | None = Field(default=None, gt=0)
    maximum_stop_seconds: float = Field(gt=0)
    maximum_pull_seconds: float = Field(gt=0)
    state: ScreenRecordingBoundaryState
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_state_reason(self) -> Self:
        if self.started_at.utcoffset() is None:
            raise ValueError("The recording boundary time must include a timezone.")
        if self.state is ScreenRecordingBoundaryState.ACTIVE and self.reason is not None:
            raise ValueError("An active recording boundary cannot have a failure reason.")
        if self.state is not ScreenRecordingBoundaryState.ACTIVE and self.reason is None:
            raise ValueError("An inactive recording boundary must have a reason.")
        if self.state is ScreenRecordingBoundaryState.ACTIVE:
            if self.effective_maximum_seconds is None:
                raise ValueError("An active recording boundary requires an effective limit.")
        elif self.effective_maximum_seconds is not None:
            raise ValueError("An inactive recording boundary cannot claim an effective limit.")
        if (
            self.effective_maximum_seconds is not None
            and self.effective_maximum_seconds > self.requested_maximum_seconds
        ):
            raise ValueError("An effective recording limit cannot exceed its request.")
        return self


class ScreenRecordingArtifactReference(BaseModel):
    """Identity, interpretation, and policy for one retained MP4 segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: RecordIdentifier
    recording_id: RecordIdentifier
    segment_index: int = Field(ge=1)
    checkpoint_kind: ScreenRecordingCheckpointKind
    checkpoint_id: RecordIdentifier | None = None
    sha256: Sha256Digest
    path: Path
    byte_count: int = Field(gt=0)
    duration_seconds: float = Field(gt=0)
    requested_maximum_seconds: int = Field(gt=0)
    effective_maximum_seconds: int = Field(gt=0)
    maximum_stop_seconds: float = Field(gt=0)
    maximum_pull_seconds: float = Field(gt=0)
    reached_effective_limit: bool
    capture_method: ScreenRecordingCaptureMethod
    redaction_state: ScreenRecordingRedactionState
    redaction: ScreenRecordingRedactionProvenance | None = None
    sensitivity: ArtifactSensitivity
    retention: ArtifactRetention

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: Path) -> Path:
        if value.is_absolute() or value.suffix.casefold() != ".mp4" or ".." in value.parts:
            raise ValueError("A recording artifact path must be a safe relative MP4 path.")
        return value

    @model_validator(mode="after")
    def _validate_checkpoint(self) -> Self:
        validate_recording_checkpoint(self.checkpoint_kind, self.checkpoint_id)
        if self.effective_maximum_seconds > self.requested_maximum_seconds:
            raise ValueError("An effective recording limit cannot exceed its request.")
        if self.redaction_state is ScreenRecordingRedactionState.RAW:
            if self.redaction is not None:
                raise ValueError("A raw recording cannot have redaction provenance.")
            if self.sensitivity is ArtifactSensitivity.DECLARED_SENSITIVE:
                raise ValueError("Declared-sensitive recording input cannot remain raw.")
        elif self.redaction is None:
            raise ValueError("A redacted recording must have redaction provenance.")
        elif self.sensitivity is not self.redaction.input_sensitivity:
            raise ValueError(
                "A redacted recording must retain its input sensitivity classification."
            )
        return self


class ScreenRecordingCheckpointReceipt(BaseModel):
    """Durable result of closing one recording segment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    receipt_id: RecordIdentifier
    recording_id: RecordIdentifier
    segment_index: int = Field(ge=1)
    checkpoint_kind: ScreenRecordingCheckpointKind
    checkpoint_id: RecordIdentifier | None = None
    started_at: datetime
    completed_at: datetime
    state: ScreenRecordingSegmentState
    artifact: ScreenRecordingArtifactReference | None = None
    reason: str | None = Field(default=None, min_length=1)
    remote_cleanup_state: ScreenRecordingRemoteCleanupState
    continuation_started: bool
    continuation_reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        validate_recording_checkpoint(self.checkpoint_kind, self.checkpoint_id)
        if self.started_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("Recording checkpoint times must include a timezone.")
        if self.completed_at < self.started_at:
            raise ValueError("A recording checkpoint cannot complete before it starts.")
        if self.state is ScreenRecordingSegmentState.CAPTURED:
            if self.artifact is None or self.reason is not None:
                raise ValueError("A captured recording requires one artifact and no reason.")
        elif self.artifact is not None or self.reason is None:
            raise ValueError("A recording without an artifact requires a reason.")
        if self.artifact is not None and (
            self.artifact.recording_id != self.recording_id
            or self.artifact.segment_index != self.segment_index
            or self.artifact.checkpoint_kind is not self.checkpoint_kind
            or self.artifact.checkpoint_id != self.checkpoint_id
        ):
            raise ValueError("A recording receipt and artifact must identify one segment.")
        uses_remote_file = (
            self.artifact is not None
            and self.artifact.capture_method is ScreenRecordingCaptureMethod.REMOTE_FILE
        )
        if uses_remote_file == (
            self.remote_cleanup_state is ScreenRecordingRemoteCleanupState.NOT_APPLICABLE
        ):
            raise ValueError("A recording receipt has an invalid remote cleanup state.")
        if self.checkpoint_kind is ScreenRecordingCheckpointKind.SESSION_END:
            if self.continuation_started or self.continuation_reason is not None:
                raise ValueError("A session-end checkpoint cannot continue recording.")
        elif self.continuation_started == (self.continuation_reason is not None):
            raise ValueError("A recording continuation failure requires exactly one reason.")
        return self


class ScreenRecordingRemoteCleanupReceipt(BaseModel):
    """Append-only evidence that a remote recording source was released."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cleanup_id: RecordIdentifier
    checkpoint_receipt_id: RecordIdentifier
    recording_id: RecordIdentifier
    segment_index: int = Field(ge=1)
    artifact_id: RecordIdentifier
    released_at: datetime

    @field_validator("released_at")
    @classmethod
    def _validate_released_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("The remote cleanup time must include a timezone.")
        return value
