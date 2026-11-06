"""Public records for resolving visual targets on one screenshot."""

from __future__ import annotations

from enum import StrEnum
import math
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier, Sha256Digest
from yewo.observations import (
    PixelRegion,
    PresentationKey,
    ScreenRegion,
    ScreenshotRedactionState,
    ScreenshotReference,
)
from yewo.observations.records import validate_screen_regions


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
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    REFUSED = "refused"
    MALFORMED = "malformed"
    OUT_OF_BOUNDS = "out_of_bounds"
    REPEATED_POINT = "repeated_point"
    AMBIGUOUS = "ambiguous"


class GroundingResolutionSource(StrEnum):
    """The evidence source that resolved one grounding request."""

    MODEL = "model"
    SCREEN_ANALYSIS = "screen_analysis"
    MEMORY = "memory"
    PROCEDURE = "procedure"


class ModelGroundingResolution(BaseModel):
    """Model-call identities used to resolve one grounding request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["model"] = "model"
    grounding_profile_id: RecordIdentifier
    model_request_ids: tuple[RecordIdentifier, ...] = Field(min_length=1)


class ScreenAnalysisGroundingResolution(BaseModel):
    """Screen-analysis element evidence used without another model call."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["screen_analysis"] = "screen_analysis"
    analysis_id: RecordIdentifier
    element_ref: RecordIdentifier


class MemoryGroundingResolution(BaseModel):
    """Accepted transition reused from application memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["memory"] = "memory"
    transition_id: RecordIdentifier


class ProcedureGroundingResolution(BaseModel):
    """Accepted source step reused from a compiled procedure."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["procedure"] = "procedure"
    source_step_id: RecordIdentifier


GroundingResolution = Annotated[
    ModelGroundingResolution
    | ScreenAnalysisGroundingResolution
    | MemoryGroundingResolution
    | ProcedureGroundingResolution,
    Field(discriminator="kind"),
]


class GroundingCoordinateConvention(StrEnum):
    """The coordinate space declared by a grounding profile."""

    SOURCE_PIXELS = "source_pixels"
    NORMALIZED = "normalized"
    NORMALIZED_1000 = "normalized_1000"


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


class ScreenAnalysisGroundingCandidate(BaseModel):
    """One high-confidence analyzed region eligible for direct grounding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    analysis_id: RecordIdentifier
    element_ref: RecordIdentifier
    screenshot_sha256: Sha256Digest
    region: NormalizedRegion


class GroundedPoint(BaseModel):
    """One accepted point in canonical and source-pixel coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["point"] = "point"
    normalized: NormalizedPoint
    pixel: PixelPoint


class GroundingCorrection(BaseModel):
    """A previous ineffective point that one correction must not repeat."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    previous_point: PixelPoint
    minimum_distance_px: int = Field(gt=0)


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
    source_presentation: PresentationKey | None = None
    expression: str = Field(min_length=1)
    operation: GroundingTargetOperation
    output_shape: GroundingOutputShape
    excluded_regions: tuple[ScreenRegion, ...] | None = None
    correction: GroundingCorrection | None = None
    screen_analysis_candidate: ScreenAnalysisGroundingCandidate | None = None

    @model_validator(mode="after")
    def _validate_request(self) -> Self:
        if self.screenshot.redaction_state is not ScreenshotRedactionState.REDACTED:
            raise ValueError("A grounding request must use a redacted screenshot.")
        if (
            self.screen_analysis_candidate is not None
            and self.screen_analysis_candidate.screenshot_sha256 != self.screenshot.sha256
        ):
            raise ValueError("A screen-analysis grounding candidate must match the screenshot.")
        if self.screen_analysis_candidate is not None and (
            self.output_shape is not GroundingOutputShape.POINT or self.correction is not None
        ):
            raise ValueError(
                "A screen-analysis grounding candidate requires uncorrected point output."
            )
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
        if self.correction is not None:
            point = self.correction.previous_point
            if self.output_shape is not GroundingOutputShape.POINT:
                raise ValueError("A grounding correction requires point output.")
            if point.x >= self.screenshot.width_px or point.y >= self.screenshot.height_px:
                raise ValueError("The previous grounding point must be within the screenshot.")
        return self


class GroundingCandidate(BaseModel):
    """One grounding model and the coordinate space that model answers in."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_profile_id: RecordIdentifier
    coordinate_convention: GroundingCoordinateConvention


class GroundingProfile(BaseModel):
    """Select a model profile and its grounding wire protocol."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profile_id: RecordIdentifier
    model_profile_id: RecordIdentifier
    coordinate_convention: GroundingCoordinateConvention
    fallbacks: tuple[GroundingCandidate, ...] = ()
    supported_operations: frozenset[GroundingTargetOperation] = Field(min_length=1)
    supported_shapes: frozenset[GroundingOutputShape] = Field(min_length=1)

    @property
    def candidates(self) -> tuple[GroundingCandidate, ...]:
        """Return the primary grounder followed by each declared fallback."""
        primary = GroundingCandidate(
            model_profile_id=self.model_profile_id,
            coordinate_convention=self.coordinate_convention,
        )
        return (primary, *self.fallbacks)

    @model_validator(mode="after")
    def _validate_distinct_fallbacks(self) -> Self:
        identifiers = [candidate.model_profile_id for candidate in self.candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Each grounding fallback must name a different model profile.")
        return self


class GroundingInputError(ValueError):
    """Raised when a grounding request is incompatible with its profile."""


def validate_grounding_request(
    profile: GroundingProfile,
    request: GroundingRequest,
) -> None:
    """Require one grounding request to use a supported operation and shape."""
    if request.operation not in profile.supported_operations:
        raise GroundingInputError(
            f"The grounding profile does not support {request.operation.value}."
        )
    if request.output_shape not in profile.supported_shapes:
        raise GroundingInputError(
            f"The grounding profile does not support {request.output_shape.value} output."
        )


class GroundingResult(BaseModel):
    """A validated grounding decision with its model-call evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    grounding_id: RecordIdentifier
    source_screenshot_sha256: Sha256Digest
    output_shape: GroundingOutputShape
    disposition: GroundingDisposition
    resolution: GroundingResolution
    location: GroundedLocation | None = None
    reason: str | None = Field(default=None, min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @property
    def resolution_source(self) -> GroundingResolutionSource:
        """Return the typed resolution discriminator."""
        return GroundingResolutionSource(self.resolution.kind)

    @property
    def model_request_count(self) -> int:
        """Return logical grounding-model requests consumed by this result."""
        if isinstance(self.resolution, ModelGroundingResolution):
            return len(self.resolution.model_request_ids)
        return 0

    @model_validator(mode="after")
    def _validate_result(self) -> Self:
        if not isinstance(self.resolution, ModelGroundingResolution) and (
            self.disposition is not GroundingDisposition.ACCEPTED
        ):
            raise ValueError("Only model grounding can produce a non-accepted result.")
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


class GroundingEvidence(BaseModel):
    """Bind one grounding request to its validated result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: GroundingRequest
    result: GroundingResult

    @model_validator(mode="after")
    def _validate_request_result_pair(self) -> Self:
        if self.result.grounding_id != self.request.grounding_id:
            raise ValueError("The grounding result does not match the request identifier.")
        if self.result.source_screenshot_sha256 != self.request.screenshot.sha256:
            raise ValueError("The grounding result does not match the request screenshot.")
        if self.result.output_shape is not self.request.output_shape:
            raise ValueError("The grounding result does not match the requested shape.")
        return self


class VisualGrounder(Protocol):
    """Resolve a visual target without receiving planning state."""

    def required_model_requests(self, request: GroundingRequest) -> int: ...

    def ground(self, request: GroundingRequest) -> GroundingResult: ...


def resolve_grounding_request(
    grounder: VisualGrounder,
    request: GroundingRequest | None,
) -> GroundingEvidence | None:
    """Resolve one optional request and bind its evidence."""
    if request is None:
        return None
    return GroundingEvidence(request=request, result=grounder.ground(request))
