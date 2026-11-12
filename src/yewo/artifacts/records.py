"""Shared identity and retention records for artifacts."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path, PurePath
from typing import Annotated
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, TypeAdapter

from yewo.devices.inventory import DeviceKind, DevicePlatform


_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
RecordIdentifier = Annotated[str, Field(pattern=_IDENTIFIER_PATTERN)]
Sha256Digest = Annotated[str, Field(pattern=_SHA256_PATTERN)]


def format_relative_artifact_path(value: PurePath) -> str:
    """Render one stored artifact path the same way on every host.

    Reports link to the stored value and bundles travel between hosts, so a
    Windows separator would break both.
    """
    return value.as_posix()


RelativeArtifactPath = Annotated[
    Path,
    PlainSerializer(format_relative_artifact_path, return_type=str, when_used="json"),
]
_IDENTIFIER_ADAPTER = TypeAdapter(RecordIdentifier)


def validate_record_identifier(value: str) -> str:
    """Return one validated durable-record identifier."""
    return _IDENTIFIER_ADAPTER.validate_python(value)


def new_record_identifier() -> str:
    """Return one random identifier that satisfies the record contract."""
    return uuid4().hex


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
    artifact_sha256: Sha256Digest | None = None
