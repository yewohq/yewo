"""Model-pluggable visual grounding contracts and service."""

from yewo.grounding.codecs import CoordinateTextCodec, DecodedGrounding
from yewo.grounding.records import (
    GroundedLocation,
    GroundedPoint,
    GroundedRegion,
    GroundingCoordinateConvention,
    GroundingDisposition,
    GroundingOutputShape,
    GroundingProfile,
    GroundingRequest,
    GroundingResult,
    GroundingTargetOperation,
    NormalizedPoint,
    NormalizedRegion,
    PixelPoint,
    VisualGrounder,
)
from yewo.grounding.service import (
    GroundingEvidenceError,
    GroundingInputError,
    GroundingService,
    validate_distinct_drag_points,
)

__all__ = [
    "CoordinateTextCodec",
    "DecodedGrounding",
    "GroundedLocation",
    "GroundedPoint",
    "GroundedRegion",
    "GroundingCoordinateConvention",
    "GroundingDisposition",
    "GroundingEvidenceError",
    "GroundingInputError",
    "GroundingOutputShape",
    "GroundingProfile",
    "GroundingRequest",
    "GroundingResult",
    "GroundingService",
    "GroundingTargetOperation",
    "NormalizedPoint",
    "NormalizedRegion",
    "PixelPoint",
    "VisualGrounder",
    "validate_distinct_drag_points",
]
