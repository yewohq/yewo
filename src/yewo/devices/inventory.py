"""Contracts for discovering devices without changing their state."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DevicePlatform(StrEnum):
    """A mobile platform supported by Yewo."""

    ANDROID = "android"
    IOS = "ios"


class DeviceKind(StrEnum):
    """Whether a device is physical or software-defined."""

    PHYSICAL = "physical"
    VIRTUAL = "virtual"


class DeviceReadiness(StrEnum):
    """Whether device lifecycle code can use or prepare a target."""

    READY = "ready"
    STARTABLE = "startable"
    TRANSITIONING = "transitioning"
    NOT_READY = "not_ready"
    UNKNOWN = "unknown"


class DeviceQuery(BaseModel):
    """Optional criteria used to select one discovered device."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    platform: DevicePlatform | None = None
    target_id: str | None = Field(default=None, min_length=1)
    display_name: str | None = Field(default=None, min_length=1)


class DeviceDescriptor(BaseModel):
    """Facts observed about one device during inventory."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    platform: DevicePlatform
    target_id: str = Field(min_length=1)
    connection_id: str | None = Field(default=None, min_length=1)
    display_name: str = Field(min_length=1)
    kind: DeviceKind
    readiness: DeviceReadiness
    native_state: str = Field(min_length=1)
    operating_system_version: str | None = Field(default=None, min_length=1)
    device_model: str | None = Field(default=None, min_length=1)
    architecture: str | None = Field(default=None, min_length=1)
    screen_width_px: int | None = Field(default=None, gt=0)
    screen_height_px: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_screen_dimensions(self) -> Self:
        width_is_known = self.screen_width_px is not None
        height_is_known = self.screen_height_px is not None
        if width_is_known != height_is_known:
            raise ValueError("Screen width and height must be present together.")
        return self


class DeviceInventory(Protocol):
    """Discover devices without starting, stopping, or modifying them."""

    def list_devices(self) -> Sequence[DeviceDescriptor]:
        """Return the devices observed by this inventory."""
        ...
