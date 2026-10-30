"""Public records for resolving visual targets on one screenshot."""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.models import (
    ModelArtifactKind,
    ModelArtifactReference,
    ModelCallDisposition,
    ModelCallUsage,
)
from yewo.observations import (
    PixelRegion,
    ScreenRegion,
    ScreenshotRedactionState,
    ScreenshotReference,
)
from yewo.observations.records import validate_screen_regions


_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class GroundingTargetOperation(StrEnum):
    """The action role of one requested visual target."""

    TAP = "tap"
    TEXT_FIELD = "text_field"
    DRAG_SOURCE = "drag_source"
    DRAG_DESTINATION = "drag_destination"
    SCROLL_REGION = "scroll_region"


class GroundingOutputShape(StrEnum):
    """The geometry requested from a grounder."""

    POINT = "point"
    REGION = "region"


class GroundingDisposition(StrEnum):
    """The deterministic interpretation of one grounding attempt."""

    ACCEPTED = "accepted"
    REFUSED = "refused"
    MALFORMED = "malformed"
    OUT_OF_BOUNDS = "out_of_bounds"
    AMBIGUOUS = "ambiguous"


class GroundingCoordinateConvention(StrEnum):
    """The coordinate space declared by a grounding profile."""

    SOURCE_PIXELS = "source_pixels"
    NORMALIZED = "normalized"


class NormalizedPoint(BaseModel):
    """One point in the closed canonical zero-to-one coordinate space."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_finite_coordinates(self) -> Self:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("Normalized point coordinates must be finite.")
        return self


class PixelPoint(BaseModel):
    """One point in source-screenshot pixels."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class NormalizedRegion(BaseModel):
    """One half-open region in canonical normalized coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    left: float = Field(ge=0, le=1)
    top: float = Field(ge=0, le=1)
    right: float = Field(ge=0, le=1)
    bottom: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_region(self) -> Self:
        coordinates = (self.left, self.top, self.right, self.bottom)
        if not all(math.isfinite(coordinate) for coordinate in coordinates):
            raise ValueError("Normalized region coordinates must be finite.")
        if self.left >= self.right or self.top >= self.bottom:
            raise ValueError("A normalized region must have positive width and height.")
        return self


class GroundedPoint(BaseModel):
    """One accepted point in canonical and source-pixel coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["point"] = "point"
    normalized: NormalizedPoint
    pixel: PixelPoint


class GroundedRegion(BaseModel):
    """One accepted region in canonical and source-pixel coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["region"] = "region"
    normalized: NormalizedRegion
    pixel: PixelRegion


GroundedLocation = Annotated[
    GroundedPoint | GroundedRegion,
    Field(discriminator="kind"),
]


class GroundingRequest(BaseModel):
    """Resolve one referring expression on one exact redacted screenshot."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    grounding_id: RecordIdentifier
    screenshot: ScreenshotReference
    expression: str = Field(min_length=1)
    operation: GroundingTargetOperation
    output_shape: GroundingOutputShape
    excluded_regions: tuple[ScreenRegion, ...] | None = None
    retry_feedback: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def _validate_request(self) -> Self:
        if self.screenshot.redaction_state is not ScreenshotRedactionState.REDACTED:
            raise ValueError("A grounding request must use a redacted screenshot.")
        validate_screen_regions(
            self.excluded_regions,
            self.screenshot.width_px,
            self.screenshot.height_px,
        )
        if (
            self.operation is not GroundingTargetOperation.SCROLL_REGION
            and self.output_shape is GroundingOutputShape.REGION
        ):
            raise ValueError("Only a scroll-region request can require a region.")
        return self


class GroundingProfile(BaseModel):
    """Select a model profile and its grounding wire protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: RecordIdentifier
    model_profile_id: RecordIdentifier
    prompt_protocol: Literal["point-v1"] = "point-v1"
    response_codec: Literal["coordinate-text-v1"] = "coordinate-text-v1"
    coordinate_convention: GroundingCoordinateConvention
    image_policy: Literal["source-image"] = "source-image"
    supported_operations: frozenset[GroundingTargetOperation] = Field(min_length=1)
    supported_shapes: frozenset[GroundingOutputShape] = Field(min_length=1)


class GroundingResult(BaseModel):
    """A validated grounding decision with its model-call evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    grounding_id: RecordIdentifier
    profile_id: RecordIdentifier
    model_profile_id: RecordIdentifier
    source_screenshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    output_shape: GroundingOutputShape
    disposition: GroundingDisposition
    location: GroundedLocation | None = None
    reason: str | None = Field(default=None, min_length=1)
    requested_provider: str = Field(min_length=1)
    requested_model: str = Field(min_length=1)
    requested_revision: str | None = Field(default=None, min_length=1)
    served_model: str | None = Field(default=None, min_length=1)
    upstream_provider: str | None = Field(default=None, min_length=1)
    transport_disposition: ModelCallDisposition
    raw_response_artifact: ModelArtifactReference
    latency_ms: float | None = Field(default=None, ge=0)
    usage: ModelCallUsage
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if self.raw_response_artifact.kind is not ModelArtifactKind.CALL:
            raise ValueError("A grounding response artifact must contain a model call.")
        if self.disposition is GroundingDisposition.ACCEPTED:
            if self.location is None or self.reason is not None:
                raise ValueError("An accepted grounding must contain one location and no reason.")
            if self.location.kind != self.output_shape.value:
                raise ValueError("The grounded location must match the requested output shape.")
        elif self.location is not None or self.reason is None:
            raise ValueError("A rejected grounding must contain a reason and no location.")
        if self.confidence is not None and self.disposition is not GroundingDisposition.ACCEPTED:
            raise ValueError("Only an accepted grounding can contain confidence.")
        return self


class VisualGrounder(Protocol):
    """Resolve a visual target without receiving planning state."""

    def ground(self, request: GroundingRequest) -> GroundingResult: ...
