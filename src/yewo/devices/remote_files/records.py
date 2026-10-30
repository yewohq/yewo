"""Immutable identities for run-owned remote temporary files."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from yewo.artifacts.records import (
    RecordIdentifier,
    validate_record_identifier,
)
from yewo.devices.inventory import DevicePlatform
from yewo.resources import ResourceOwnership


_REMOTE_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_APPLICATION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,254}$"
_REMOTE_NAME_ADAPTER = TypeAdapter(Annotated[str, Field(pattern=_REMOTE_NAME_PATTERN)])
_APPLICATION_ID_ADAPTER = TypeAdapter(Annotated[str, Field(pattern=_APPLICATION_ID_PATTERN)])


def remote_temporary_path(
    platform: DevicePlatform,
    application_id: str,
    namespace_id: str,
    name: str,
) -> str:
    """Build a run-scoped path accepted by the platform Appium driver."""
    application_id = _APPLICATION_ID_ADAPTER.validate_python(application_id)
    namespace_id = validate_record_identifier(namespace_id)
    name = _REMOTE_NAME_ADAPTER.validate_python(name)
    if platform is DevicePlatform.ANDROID:
        return f"/sdcard/Download/.yewo_{namespace_id}_{name}"
    return f"@{application_id}:data/tmp/yewo_{namespace_id}_{name}"


class RemoteTemporaryFileReference(BaseModel):
    """One exact remote path reserved for cleanup by the current run."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    file_id: RecordIdentifier
    namespace_id: RecordIdentifier
    platform: DevicePlatform
    target_id: str = Field(min_length=1)
    application_id: str = Field(pattern=_APPLICATION_ID_PATTERN)
    name: str = Field(pattern=_REMOTE_NAME_PATTERN)
    remote_path: str = Field(min_length=1)
    ownership: Literal[ResourceOwnership.RUN] = ResourceOwnership.RUN

    @model_validator(mode="after")
    def validate_owned_path(self) -> RemoteTemporaryFileReference:
        expected = remote_temporary_path(
            self.platform,
            self.application_id,
            self.namespace_id,
            self.name,
        )
        if self.remote_path != expected:
            raise ValueError("A remote temporary-file path must match its run namespace.")
        return self
