"""Typed device preconditions declared in a Yewo project file."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CapabilityKind(StrEnum):
    """Device-state families accepted by the public YAML contract."""

    LOCATION = "location"
    NETWORK = "network"
    BATTERY = "battery"
    TIMEZONE = "timezone"
    LOCALE = "locale"
    PERMISSION = "permission"
    INTERRUPT = "interrupt"
    BIOMETRIC = "biometric"
    CLIPBOARD = "clipboard"
    DEEPLINK = "deeplink"
    APP_STATE = "app_state"
    ORIENTATION = "orientation"
    DEVICE_LOCK = "device_lock"
    MEDIA_INJECT = "media_inject"
    ACCESSIBILITY = "accessibility"


class CapabilitySpec(BaseModel):
    """Strict base record for one requested device capability."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class LocationCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.LOCATION] = CapabilityKind.LOCATION
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class NetworkCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.NETWORK] = CapabilityKind.NETWORK
    mode: Literal[
        "offline",
        "online",
        "wifi_only",
        "data_only",
        "airplane",
        "slow_2g",
        "slow_3g",
        "high_latency",
    ]


class BatteryCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.BATTERY] = CapabilityKind.BATTERY
    level: int = Field(ge=0, le=100)
    charging: bool = True


class TimezoneCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.TIMEZONE] = CapabilityKind.TIMEZONE
    timezone: str = Field(min_length=1)


class LocaleCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.LOCALE] = CapabilityKind.LOCALE
    language: str = Field(min_length=1)
    country: str = ""


class PermissionCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.PERMISSION] = CapabilityKind.PERMISSION
    permission: Literal[
        "camera",
        "location",
        "microphone",
        "notifications",
        "photos",
        "contacts",
        "storage",
    ]
    grant: bool


class InterruptCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.INTERRUPT] = CapabilityKind.INTERRUPT
    type: Literal["low_memory", "incoming_call"]
    phone: str = "555-5555"


class BiometricCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.BIOMETRIC] = CapabilityKind.BIOMETRIC
    action: Literal["match", "unmatch", "enroll"]
    type: Literal["faceId", "touchId"] = "faceId"
    finger_id: int = Field(default=1, ge=1)


class ClipboardCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.CLIPBOARD] = CapabilityKind.CLIPBOARD
    text: str


class DeepLinkCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.DEEPLINK] = CapabilityKind.DEEPLINK
    url: str = Field(min_length=1)


class AppStateCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.APP_STATE] = CapabilityKind.APP_STATE
    action: Literal["force_stop", "clear_data", "relaunch", "background_resume"]
    duration_seconds: float | None = Field(default=None, ge=0, le=60)

    @model_validator(mode="before")
    @classmethod
    def _default_background_duration(cls, value: object) -> object:
        if isinstance(value, dict) and value.get("action") == "background_resume":
            return {"duration_seconds": 1.0, **value}
        return value

    @model_validator(mode="after")
    def _validate_duration(self) -> Self:
        if self.action != "background_resume" and self.duration_seconds is not None:
            raise ValueError("duration_seconds requires background_resume.")
        return self


class OrientationCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.ORIENTATION] = CapabilityKind.ORIENTATION
    orientation: Literal["portrait", "landscape"]


class DeviceLockCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.DEVICE_LOCK] = CapabilityKind.DEVICE_LOCK
    duration_seconds: float = Field(default=1.0, ge=0, le=60)


class MediaInjectCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.MEDIA_INJECT] = CapabilityKind.MEDIA_INJECT
    local_path: str = Field(min_length=1)


class AccessibilityCapabilitySpec(CapabilitySpec):
    kind: Literal[CapabilityKind.ACCESSIBILITY] = CapabilityKind.ACCESSIBILITY
    enabled: bool


DeviceCapabilitySpec = Annotated[
    LocationCapabilitySpec
    | NetworkCapabilitySpec
    | BatteryCapabilitySpec
    | TimezoneCapabilitySpec
    | LocaleCapabilitySpec
    | PermissionCapabilitySpec
    | InterruptCapabilitySpec
    | BiometricCapabilitySpec
    | ClipboardCapabilitySpec
    | DeepLinkCapabilitySpec
    | AppStateCapabilitySpec
    | OrientationCapabilitySpec
    | DeviceLockCapabilitySpec
    | MediaInjectCapabilitySpec
    | AccessibilityCapabilitySpec,
    Field(discriminator="kind"),
]


class ScenarioPreconditionsSpec(BaseModel):
    """Application and device state requested before visible execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    app_state: Literal["clear_data", "restart", "preserve"] = "clear_data"
    capabilities: tuple[DeviceCapabilitySpec, ...] = ()
