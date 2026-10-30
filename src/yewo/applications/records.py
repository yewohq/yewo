"""Requests and observed identity records for mobile applications."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.devices.inventory import DevicePlatform


class ApplicationInstallMode(StrEnum):
    """How a session may prepare an application installation."""

    INSTALL = "install"
    INSTALL_IF_NEEDED = "install_if_needed"
    REUSE = "reuse"


class ApplicationInstallationChange(StrEnum):
    """Observed or requested installation mutation during session startup."""

    NONE = "none"
    INSTALLED = "installed"
    REINSTALLED = "reinstalled"
    UNVERIFIED = "unverified"


class ApplicationRequest(BaseModel):
    """Caller intent for selecting and preparing one application."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    platform: DevicePlatform
    install_mode: ApplicationInstallMode = ApplicationInstallMode.INSTALL
    artifact_path: Path | None = None
    application_id: str | None = Field(default=None, min_length=1)
    retain_installation: bool = False

    @model_validator(mode="after")
    def _validate_install_source(self) -> Self:
        if (
            self.install_mode
            in {ApplicationInstallMode.INSTALL, ApplicationInstallMode.INSTALL_IF_NEEDED}
            and self.artifact_path is None
        ):
            raise ValueError(f"Install mode '{self.install_mode}' requires an artifact path.")
        if self.install_mode is ApplicationInstallMode.REUSE and self.application_id is None:
            raise ValueError("Reuse mode requires an application ID.")
        return self


class ApplicationDescriptor(BaseModel):
    """Identity facts observed from an artifact or installed application."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    platform: DevicePlatform
    application_id: str = Field(min_length=1)
    display_name: str | None = Field(default=None, min_length=1)
    version: str | None = Field(default=None, min_length=1)
    build: str | None = Field(default=None, min_length=1)
    launch_target: str | None = Field(default=None, min_length=1)
    artifact_path: Path | None = None
    artifact_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
