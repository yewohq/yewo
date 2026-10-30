"""Strict shared codecs for grounding response content."""

from __future__ import annotations

from dataclasses import dataclass
import re

from yewo.grounding.records import (
    GroundedPoint,
    GroundedRegion,
    GroundingCoordinateConvention,
    GroundingDisposition,
    GroundingOutputShape,
    NormalizedPoint,
    NormalizedRegion,
    PixelPoint,
)
from yewo.observations import PixelRegion


_PIXEL_POINT_PATTERN = re.compile(r"\A\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\Z")
_NORMALIZED_POINT_PATTERN = re.compile(
    r"\A\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\Z"
)
_PIXEL_REGION_PATTERN = re.compile(
    r"\A\(\s*([+-]?\d+)\s*,\s*([+-]?\d+)\s*,\s*"
    r"([+-]?\d+)\s*,\s*([+-]?\d+)\s*\)\Z"
)
_NORMALIZED_REGION_PATTERN = re.compile(
    r"\A\(\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*\)\Z"
)
_REFUSAL_PATTERN = re.compile(r"\Arefused(?:\s*:\s*(.+))?\Z", re.IGNORECASE)
_AMBIGUITY_PATTERN = re.compile(r"\Aambiguous(?:\s*:\s*(.+))?\Z", re.IGNORECASE)


@dataclass(frozen=True)
class DecodedGrounding:
    """One response decoded without applying request-specific policy."""

    disposition: GroundingDisposition
    location: GroundedPoint | GroundedRegion | None = None
    reason: str | None = None


class CoordinateTextCodec:
    """Decode one strict point or an explicit refusal from text."""

    def decode(
        self,
        text: str,
        convention: GroundingCoordinateConvention,
        width_px: int,
        height_px: int,
        output_shape: GroundingOutputShape = GroundingOutputShape.POINT,
    ) -> DecodedGrounding:
        """Decode one response against the declared source geometry."""
        refusal = _REFUSAL_PATTERN.fullmatch(text.strip())
        if refusal is not None:
            return DecodedGrounding(
                GroundingDisposition.REFUSED,
                reason=refusal.group(1) or "The model refused to identify one target.",
            )
        ambiguity = _AMBIGUITY_PATTERN.fullmatch(text.strip())
        if ambiguity is not None:
            return DecodedGrounding(
                GroundingDisposition.AMBIGUOUS,
                reason=ambiguity.group(1) or "The model found more than one possible target.",
            )
        if output_shape is GroundingOutputShape.REGION:
            if convention is GroundingCoordinateConvention.SOURCE_PIXELS:
                return self._decode_source_region(text.strip(), width_px, height_px)
            return self._decode_normalized_region(text.strip(), width_px, height_px)
        if convention is GroundingCoordinateConvention.SOURCE_PIXELS:
            return self._decode_source_pixels(text.strip(), width_px, height_px)
        return self._decode_normalized(text.strip(), width_px, height_px)

    @staticmethod
    def _decode_source_pixels(text: str, width_px: int, height_px: int) -> DecodedGrounding:
        match = _PIXEL_POINT_PATTERN.fullmatch(text)
        if match is None:
            return _malformed_point()
        x_px, y_px = (int(value) for value in match.groups())
        if x_px < 0 or y_px < 0 or x_px >= width_px or y_px >= height_px:
            return DecodedGrounding(
                GroundingDisposition.OUT_OF_BOUNDS,
                reason="The model point is outside the source screenshot.",
            )
        normalized = NormalizedPoint(
            x=_normalize_pixel(x_px, width_px),
            y=_normalize_pixel(y_px, height_px),
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedPoint(normalized=normalized, pixel=PixelPoint(x=x_px, y=y_px)),
        )

    @staticmethod
    def _decode_normalized(text: str, width_px: int, height_px: int) -> DecodedGrounding:
        match = _NORMALIZED_POINT_PATTERN.fullmatch(text)
        if match is None:
            return _malformed_point()
        x_normalized, y_normalized = (float(value) for value in match.groups())
        if not 0 <= x_normalized <= 1 or not 0 <= y_normalized <= 1:
            return DecodedGrounding(
                GroundingDisposition.OUT_OF_BOUNDS,
                reason="The model point is outside the normalized coordinate space.",
            )
        normalized = NormalizedPoint(x=x_normalized, y=y_normalized)
        pixel = PixelPoint(
            x=round(normalized.x * (width_px - 1)),
            y=round(normalized.y * (height_px - 1)),
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedPoint(normalized=normalized, pixel=pixel),
        )

    @staticmethod
    def _decode_source_region(text: str, width_px: int, height_px: int) -> DecodedGrounding:
        match = _PIXEL_REGION_PATTERN.fullmatch(text)
        if match is None:
            return _malformed_region()
        left, top, right, bottom = (int(value) for value in match.groups())
        if left < 0 or top < 0 or right > width_px or bottom > height_px:
            return _out_of_bounds_region()
        if left >= right or top >= bottom:
            return _malformed_region()
        pixel = PixelRegion(left=left, top=top, right=right, bottom=bottom)
        normalized = NormalizedRegion(
            left=left / width_px,
            top=top / height_px,
            right=right / width_px,
            bottom=bottom / height_px,
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedRegion(normalized=normalized, pixel=pixel),
        )

    @staticmethod
    def _decode_normalized_region(
        text: str,
        width_px: int,
        height_px: int,
    ) -> DecodedGrounding:
        match = _NORMALIZED_REGION_PATTERN.fullmatch(text)
        if match is None:
            return _malformed_region()
        left, top, right, bottom = (float(value) for value in match.groups())
        if any(value < 0 or value > 1 for value in (left, top, right, bottom)):
            return _out_of_bounds_region()
        if left >= right or top >= bottom:
            return _malformed_region()
        normalized = NormalizedRegion(left=left, top=top, right=right, bottom=bottom)
        pixel = PixelRegion(
            left=round(left * width_px),
            top=round(top * height_px),
            right=round(right * width_px),
            bottom=round(bottom * height_px),
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedRegion(normalized=normalized, pixel=pixel),
        )


def _normalize_pixel(coordinate: int, dimension: int) -> float:
    return coordinate / (dimension - 1) if dimension > 1 else 0.0


def _malformed_point() -> DecodedGrounding:
    return DecodedGrounding(
        GroundingDisposition.MALFORMED,
        reason="The model response is not one coordinate in the declared format.",
    )


def _malformed_region() -> DecodedGrounding:
    return DecodedGrounding(
        GroundingDisposition.MALFORMED,
        reason="The model response is not one region in the declared format.",
    )


def _out_of_bounds_region() -> DecodedGrounding:
    return DecodedGrounding(
        GroundingDisposition.OUT_OF_BOUNDS,
        reason="The model region is outside the declared coordinate space.",
    )
