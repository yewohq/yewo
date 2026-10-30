"""Immutable records for captured device evidence."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    RecordIdentifier,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.screen import SystemRegionKind
from yewo.observations._region_mask import system_region_mask_digest


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class ScreenshotRedactionState(StrEnum):
    """Whether screenshot pixels passed through an approved redactor."""

    RAW = "raw"
    REDACTED = "redacted"


class DeviceOrientation(StrEnum):
    """Normalized device orientation observed from the platform driver."""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    UNKNOWN = "unknown"


class RedactionProvenance(BaseModel):
    """Policy evidence for pixels produced by screenshot redaction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_id: RecordIdentifier
    region_count: int = Field(ge=0)
    region_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    input_sensitivity: ArtifactSensitivity


class CoordinateScale(BaseModel):
    """Scale from screenshot pixels to Appium driver coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(gt=0)
    y: float = Field(gt=0)


class PixelRegion(BaseModel):
    """One half-open rectangle in screenshot pixels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: int = Field(ge=0)
    top: int = Field(ge=0)
    right: int = Field(gt=0)
    bottom: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_positive_area(self) -> Self:
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("A pixel region must have positive width and height.")
        return self


class ScreenRegion(PixelRegion):
    """One system-owned rectangle in screenshot pixels."""

    kind: SystemRegionKind


class RedactionRegion(PixelRegion):
    """One declared-sensitive rectangle in screenshot pixels."""


def validate_screen_regions(
    regions: tuple[ScreenRegion, ...] | None,
    width: int,
    height: int,
) -> None:
    """Reject a region that extends beyond one screenshot."""
    if regions is not None and any(
        region.right > width or region.bottom > height for region in regions
    ):
        raise ValueError("A system region is outside the screenshot.")


class ScreenshotReference(BaseModel):
    """Identity, geometry, policy, and location of one PNG artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: RecordIdentifier
    sha256: str = Field(pattern=_SHA256_PATTERN)
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    path: Path
    redaction_state: ScreenshotRedactionState
    redaction: RedactionProvenance | None = None
    sensitivity: ArtifactSensitivity
    retention: ArtifactRetention

    @field_validator("path")
    @classmethod
    def _validate_relative_path(cls, path: Path) -> Path:
        if path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
            raise ValueError("An artifact path must be a safe relative path.")
        if path.suffix.lower() != ".png":
            raise ValueError("A screenshot artifact path must end in .png.")
        return path

    @model_validator(mode="after")
    def _validate_redaction_provenance(self) -> Self:
        if self.redaction_state is ScreenshotRedactionState.RAW:
            if self.redaction is not None:
                raise ValueError("A raw screenshot cannot have redaction provenance.")
        elif self.redaction is None:
            raise ValueError("A redacted screenshot must have redaction provenance.")
        elif self.sensitivity is not self.redaction.input_sensitivity:
            raise ValueError(
                "A redacted screenshot must retain its input sensitivity classification."
            )
        return self


class ScreenFingerprint(BaseModel):
    """Versioned perceptual fingerprint of application-owned pixels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["application-phash-v1"] = "application-phash-v1"
    value: str = Field(pattern=r"^[0-9a-f]{16}$")
    system_region_mask_sha256: str | None = Field(
        default=None,
        pattern=_SHA256_PATTERN,
    )
    source_artifact_id: RecordIdentifier


class Observation(BaseModel):
    """Evidence captured from one stable device moment."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    schema_version: Literal[1] = 1
    observation_id: RecordIdentifier
    device: EvidenceDeviceIdentity
    application: EvidenceApplicationIdentity
    foreground_application_id: str | None = Field(default=None, min_length=1)
    application_state: ApplicationState
    platform_context: str | None = Field(default=None, min_length=1)
    keyboard_visible: bool | None
    orientation: DeviceOrientation | None
    screenshot_to_driver_scale: CoordinateScale
    screen_fingerprint: ScreenFingerprint
    raw_screenshot: ScreenshotReference | None = None
    redacted_screenshot: ScreenshotReference
    system_regions: tuple[ScreenRegion, ...] | None
    captured_at: datetime
    predecessor_observation_id: RecordIdentifier | None = None

    @model_validator(mode="after")
    def _validate_evidence(self) -> Self:
        if (
            self.raw_screenshot is not None
            and self.raw_screenshot.redaction_state is not ScreenshotRedactionState.RAW
        ):
            raise ValueError("The raw screenshot reference must contain raw pixels.")
        if self.redacted_screenshot.redaction_state is not ScreenshotRedactionState.REDACTED:
            raise ValueError("The redacted screenshot reference must contain redacted pixels.")
        if self.screen_fingerprint.source_artifact_id != self.redacted_screenshot.artifact_id:
            raise ValueError("The screen fingerprint must identify its source screenshot.")
        if self.predecessor_observation_id == self.observation_id:
            raise ValueError("An observation cannot name itself as its predecessor.")
        if self.captured_at.utcoffset() is None:
            raise ValueError("The observation capture time must include a timezone.")
        self._validate_screenshot_geometry()
        expected_mask_digest = system_region_mask_digest(
            self.system_regions,
            self.redacted_screenshot.width_px,
            self.redacted_screenshot.height_px,
        )
        if self.screen_fingerprint.system_region_mask_sha256 != expected_mask_digest:
            raise ValueError(
                "The screen fingerprint does not match the observation system-region mask."
            )
        return self

    def _validate_screenshot_geometry(self) -> None:
        screenshots = tuple(
            screenshot
            for screenshot in (self.raw_screenshot, self.redacted_screenshot)
            if screenshot is not None
        )
        dimensions = {(item.width_px, item.height_px) for item in screenshots}
        if len(dimensions) != 1:
            raise ValueError("Observation screenshots must use the same dimensions.")
        if self.system_regions is None:
            return
        width, height = next(iter(dimensions))
        validate_screen_regions(self.system_regions, width, height)
