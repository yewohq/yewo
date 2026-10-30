"""Immutable records for bounded device-log evidence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    RecordIdentifier,
    validate_record_identifier,
)


_LOG_TYPE_PATTERN = r"^[^\s\x00-\x1f\x7f/\\]{1,128}$"
_ARTIFACT_STEM_PATTERN = r"^[a-z][a-z0-9_-]{0,63}$"
_POLICY_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,127}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DeviceLogBoundaryState(StrEnum):
    """Whether one source is ready after its initial boundary read."""

    READY = "ready"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class DeviceLogDrainState(StrEnum):
    """What one source produced during a boundary drain."""

    CAPTURED = "captured"
    NO_RETAINED_ITEMS = "no_retained_items"
    NOT_SCHEDULED = "not_scheduled"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class DeviceLogCheckpointKind(StrEnum):
    """The execution boundary that requested one log drain."""

    SESSION_START = "session_start"
    ACTION = "action"
    SCENARIO = "scenario"
    SESSION_END = "session_end"


class DeviceLogOmission(StrEnum):
    """Which part of an observed source batch is absent from its artifacts."""

    NONE = "none"
    EARLIER = "earlier"
    LATER = "later"
    BOTH = "both"


class DeviceLogArtifactKind(StrEnum):
    """The interpretation and file form of one device-log artifact."""

    TEXT_STREAM = "text_stream"
    CRASH_REPORT = "crash_report"


def validate_device_log_checkpoint(
    checkpoint_kind: DeviceLogCheckpointKind,
    checkpoint_id: str | None,
) -> str | None:
    """Validate and return the identity required by one checkpoint kind."""
    if checkpoint_kind in {
        DeviceLogCheckpointKind.SESSION_START,
        DeviceLogCheckpointKind.SESSION_END,
    }:
        if checkpoint_id is not None:
            raise ValueError("A session boundary log drain cannot have a checkpoint ID.")
        return None
    if checkpoint_id is None:
        raise ValueError("An action or scenario log drain requires a checkpoint ID.")
    return validate_record_identifier(checkpoint_id)


class DeviceLogSource(BaseModel):
    """One vendor log source and its filesystem-safe artifact stem."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    log_type: str = Field(pattern=_LOG_TYPE_PATTERN)
    artifact_stem: str = Field(pattern=_ARTIFACT_STEM_PATTERN)


class DeviceLogArtifactReference(BaseModel):
    """Identity, interpretation, and policy for one persisted log artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    artifact_id: RecordIdentifier
    boundary_id: RecordIdentifier
    drain_id: RecordIdentifier
    log_type: str = Field(pattern=_LOG_TYPE_PATTERN)
    kind: DeviceLogArtifactKind
    source_name: str | None = Field(default=None, min_length=1, max_length=1024)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    path: Path
    byte_count: int = Field(gt=0)
    entry_count: int = Field(gt=0)
    redaction_policy_id: str = Field(pattern=_POLICY_ID_PATTERN)
    source_redaction_count: int = Field(
        ge=0,
        description="Redactions applied before byte bounding the retained source items.",
    )
    sensitivity: ArtifactSensitivity
    retention: ArtifactRetention

    @field_validator("source_name")
    @classmethod
    def _validate_source_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        source_path = PurePosixPath(value)
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or not source_path.parts
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("A device-log source name must be relative and safe.")
        return value

    @model_validator(mode="after")
    def _validate_artifact(self) -> Self:
        if self.path.is_absolute() or ".." in self.path.parts:
            raise ValueError("A device-log artifact path must be relative and safe.")
        required_suffix = (
            ".ips" if self.kind is DeviceLogArtifactKind.CRASH_REPORT else ".log"
        )
        if self.path.suffix != required_suffix:
            raise ValueError(f"A {self.kind} artifact path must end in {required_suffix}.")
        if self.kind is DeviceLogArtifactKind.CRASH_REPORT:
            if self.source_name is None or self.entry_count != 1:
                raise ValueError("A crash-report artifact must identify one source report.")
        elif self.source_name is not None:
            raise ValueError("A text-stream artifact cannot have a source report name.")
        return self


class DeviceLogBoundarySource(BaseModel):
    """Outcome of establishing one source baseline."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source: DeviceLogSource
    state: DeviceLogBoundaryState
    baseline_item_count: int = Field(ge=0)
    may_have_omitted_earlier_items: bool
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_reason(self) -> Self:
        if self.state is DeviceLogBoundaryState.READY and self.reason is not None:
            raise ValueError("A ready device-log source cannot have a failure reason.")
        if self.state is not DeviceLogBoundaryState.READY and self.reason is None:
            raise ValueError("An unavailable device-log source must have a reason.")
        if self.state is not DeviceLogBoundaryState.READY and (
            self.baseline_item_count or self.may_have_omitted_earlier_items
        ):
            raise ValueError("An unavailable source cannot claim a completed baseline.")
        return self


class DeviceLogBoundary(BaseModel):
    """Start of an incremental device-log evidence interval."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    boundary_id: RecordIdentifier
    device: EvidenceDeviceIdentity
    application: EvidenceApplicationIdentity
    started_at: datetime
    sources: tuple[DeviceLogBoundarySource, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_boundary(self) -> Self:
        if self.device.platform is not self.application.platform:
            raise ValueError("A log boundary device and application must share a platform.")
        if self.started_at.utcoffset() is None:
            raise ValueError("The device-log boundary time must include a timezone.")
        log_types = [item.source.log_type for item in self.sources]
        if len(log_types) != len(set(log_types)):
            raise ValueError("A device-log boundary cannot repeat a source.")
        return self


class DeviceLogSourceReceipt(BaseModel):
    """Evidence returned for one source during one incremental drain."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    source: DeviceLogSource
    state: DeviceLogDrainState
    observed_item_count: int = Field(default=0, ge=0)
    filtered_item_count: int = Field(default=0, ge=0)
    retained_item_count: int = Field(default=0, ge=0)
    omission: DeviceLogOmission = DeviceLogOmission.NONE
    artifacts: tuple[DeviceLogArtifactReference, ...] = ()
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.filtered_item_count > self.observed_item_count:
            raise ValueError("A source cannot filter more items than it observed.")
        if self.retained_item_count > self.observed_item_count - self.filtered_item_count:
            raise ValueError("A source cannot retain more items than remained after filtering.")
        if self.state is DeviceLogDrainState.CAPTURED:
            if not self.artifacts or not self.retained_item_count or self.reason is not None:
                raise ValueError("A captured device log needs artifacts and no reason.")
            if sum(artifact.entry_count for artifact in self.artifacts) != self.retained_item_count:
                raise ValueError("Device-log artifacts must account for every retained item.")
        elif self.artifacts or self.retained_item_count:
            raise ValueError("A non-captured device log cannot retain artifacts.")
        elif self.state in {DeviceLogDrainState.UNAVAILABLE, DeviceLogDrainState.FAILED}:
            if self.reason is None:
                raise ValueError("An unavailable device-log drain must have a reason.")
        elif self.reason is not None:
            raise ValueError("A non-failed device-log drain cannot have a failure reason.")
        if self.state is DeviceLogDrainState.NOT_SCHEDULED and (
            self.observed_item_count
            or self.filtered_item_count
            or self.omission is not DeviceLogOmission.NONE
        ):
            raise ValueError("A source that was not scheduled cannot report observed items.")
        return self


class DeviceLogDrainReceipt(BaseModel):
    """Result of one ordered incremental drain after a known boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    drain_id: RecordIdentifier
    boundary_id: RecordIdentifier
    drain_index: int = Field(ge=0)
    checkpoint_kind: DeviceLogCheckpointKind
    checkpoint_id: RecordIdentifier | None = None
    started_at: datetime
    completed_at: datetime
    sources: tuple[DeviceLogSourceReceipt, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_receipt(self) -> Self:
        if self.started_at.utcoffset() is None or self.completed_at.utcoffset() is None:
            raise ValueError("Device-log drain times must include a timezone.")
        if self.completed_at < self.started_at:
            raise ValueError("A device-log drain cannot complete before it starts.")
        validate_device_log_checkpoint(self.checkpoint_kind, self.checkpoint_id)
        if (
            self.checkpoint_kind is DeviceLogCheckpointKind.SESSION_START
            and self.drain_index != 0
        ):
            raise ValueError("The session-start drain must have index zero.")
        log_types = [receipt.source.log_type for receipt in self.sources]
        if len(log_types) != len(set(log_types)):
            raise ValueError("A device-log drain cannot repeat a source.")
        for receipt in self.sources:
            for artifact in receipt.artifacts:
                if (
                    artifact.boundary_id != self.boundary_id
                    or artifact.drain_id != self.drain_id
                    or artifact.log_type != receipt.source.log_type
                ):
                    raise ValueError(
                        "A device-log artifact must match its boundary, drain, and source."
                    )
        return self
