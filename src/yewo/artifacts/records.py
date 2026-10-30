"""Shared identity, sensitivity, and retention records for artifacts."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from yewo.devices.inventory import DeviceKind, DevicePlatform


_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
RecordIdentifier = Annotated[str, Field(pattern=_IDENTIFIER_PATTERN)]
_IDENTIFIER_ADAPTER = TypeAdapter(RecordIdentifier)


def validate_record_identifier(value: str) -> str:
    """Return one validated durable-record identifier."""
    return _IDENTIFIER_ADAPTER.validate_python(value)


def new_record_identifier() -> str:
    """Return one random identifier that satisfies the record contract."""
    return uuid4().hex


class ArtifactSensitivity(StrEnum):
    """The sensitivity governing one persisted artifact."""

    DEVICE_CONTENT = "device_content"
    DECLARED_SENSITIVE = "declared_sensitive"


class ArtifactRetention(StrEnum):
    """How long the owning run may retain an artifact."""

    EPHEMERAL = "ephemeral"
    RUN = "run"
    PERSISTENT = "persistent"


class EvidenceDeviceIdentity(BaseModel):
    """Device identity needed to interpret persisted evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    target_id: str = Field(min_length=1)
    platform: DevicePlatform
    kind: DeviceKind
    operating_system_version: str | None = Field(default=None, min_length=1)
    model: str | None = Field(default=None, min_length=1)
    architecture: str | None = Field(default=None, min_length=1)


class EvidenceApplicationIdentity(BaseModel):
    """Application build identity needed to interpret persisted evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    platform: DevicePlatform
    application_id: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    build: str | None = Field(default=None, min_length=1)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
