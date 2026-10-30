"""Integrity identity for one pushed media file."""

from pydantic import BaseModel, ConfigDict, Field

from yewo.devices.remote_files import RemoteTemporaryFileReference


class RemoteMediaReference(BaseModel):
    """One run-owned remote file and its expected content identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_file: RemoteTemporaryFileReference
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_count: int = Field(gt=0)
