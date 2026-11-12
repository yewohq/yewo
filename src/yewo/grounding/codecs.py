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


_INTEGER = r"[+-]?\d+"
_DECIMAL = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"


def _coordinates(number: str, count: int) -> re.Pattern[str]:
    """Match a bare coordinate tuple. Wrappers are removed before matching."""
    return re.compile(r"\A" + r"\s*,\s*".join([f"({number})"] * count) + r"\Z")


_PIXEL_POINT_PATTERN = _coordinates(_INTEGER, 2)
_NORMALIZED_POINT_PATTERN = _coordinates(_DECIMAL, 2)
_PIXEL_REGION_PATTERN = _coordinates(_INTEGER, 4)
_NORMALIZED_REGION_PATTERN = _coordinates(_DECIMAL, 4)
_WRAPPERS = (("(", ")"), ("[", "]"), ("{", "}"))
_REFUSAL_PATTERN = re.compile(r"\Arefused(?:\s*:\s*(.+))?\Z", re.IGNORECASE)
_AMBIGUITY_PATTERN = re.compile(r"\Aambiguous(?:\s*:\s*(.+))?\Z", re.IGNORECASE)


def _unwrap(text: str) -> str:
    """Drop one matched bracket pair. Models wrap coordinates, or do not, at random."""
    stripped = text.strip()
    for opening, closing in _WRAPPERS:
        if stripped.startswith(opening) and stripped.endswith(closing):
            return stripped[1:-1].strip()
    return stripped


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
        text = _unwrap(text)
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
            return self._decode_normalized_region(
                text.strip(),
                width_px,
                height_px,
                _normalized_scale(convention),
                require_integers=convention is GroundingCoordinateConvention.NORMALIZED_1000,
            )
        if convention is GroundingCoordinateConvention.SOURCE_PIXELS:
            return self._decode_source_pixels(text.strip(), width_px, height_px)
        return self._decode_normalized(
            text.strip(),
            width_px,
            height_px,
            _normalized_scale(convention),
            require_integers=convention is GroundingCoordinateConvention.NORMALIZED_1000,
        )

    @staticmethod
    def _decode_source_pixels(text: str, width_px: int, height_px: int) -> DecodedGrounding:
        match = _PIXEL_POINT_PATTERN.fullmatch(text) or _NORMALIZED_POINT_PATTERN.fullmatch(text)
        if match is None:
            return _malformed_point()
        values = [float(value) for value in match.groups()]
        # A fractional pixel is usable, but a pair that could be a normalized answer is not:
        # rounding (0.51,0.42) would silently dispatch a tap at the screen corner. The bounds are
        # inclusive, because (0.0,0.73) and (1.0,1.0) are normalized answers that round to that
        # same corner. This also refuses a literal pixel inside the top-left 2x2, which no
        # referring expression names and a replan recovers from.
        if all(0 <= value <= 1 for value in values):
            return _malformed_point()
        x_px, y_px = (round(value) for value in values)
        if x_px < 0 or y_px < 0 or x_px >= width_px or y_px >= height_px:
            return DecodedGrounding(
                GroundingDisposition.OUT_OF_BOUNDS,
                reason="The model point is outside the source screenshot.",
            )
        normalized_point = NormalizedPoint(
            x=_normalize_pixel(x_px, width_px),
            y=_normalize_pixel(y_px, height_px),
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedPoint(normalized=normalized_point, pixel=PixelPoint(x=x_px, y=y_px)),
        )

    @staticmethod
    def _decode_normalized(
        text: str,
        width_px: int,
        height_px: int,
        scale: float = 1.0,
        *,
        require_integers: bool = False,
    ) -> DecodedGrounding:
        pattern = _PIXEL_POINT_PATTERN if require_integers else _NORMALIZED_POINT_PATTERN
        match = pattern.fullmatch(text)
        if match is None:
            return _malformed_point()
        x_scaled, y_scaled = (float(value) for value in match.groups())
        x_normalized, y_normalized = x_scaled / scale, y_scaled / scale
        if not 0 <= x_normalized <= 1 or not 0 <= y_normalized <= 1:
            return DecodedGrounding(
                GroundingDisposition.OUT_OF_BOUNDS,
                reason="The model point is outside the normalized coordinate space.",
            )
        normalized_point = NormalizedPoint(x=x_normalized, y=y_normalized)
        pixel = PixelPoint(
            x=round(normalized_point.x * (width_px - 1)),
            y=round(normalized_point.y * (height_px - 1)),
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedPoint(normalized=normalized_point, pixel=pixel),
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
        normalized_region = NormalizedRegion(
            left=left / width_px,
            top=top / height_px,
            right=right / width_px,
            bottom=bottom / height_px,
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedRegion(normalized=normalized_region, pixel=pixel),
        )

    @staticmethod
    def _decode_normalized_region(
        text: str,
        width_px: int,
        height_px: int,
        scale: float = 1.0,
        *,
        require_integers: bool = False,
    ) -> DecodedGrounding:
        pattern = _PIXEL_REGION_PATTERN if require_integers else _NORMALIZED_REGION_PATTERN
        match = pattern.fullmatch(text)
        if match is None:
            return _malformed_region()
        left, top, right, bottom = (float(value) / scale for value in match.groups())
        if any(value < 0 or value > 1 for value in (left, top, right, bottom)):
            return _out_of_bounds_region()
        if left >= right or top >= bottom:
            return _malformed_region()
        normalized_region = NormalizedRegion(left=left, top=top, right=right, bottom=bottom)
        pixel_left = round(left * width_px)
        pixel_top = round(top * height_px)
        pixel_right = round(right * width_px)
        pixel_bottom = round(bottom * height_px)
        if pixel_left >= pixel_right or pixel_top >= pixel_bottom:
            return _malformed_region()
        pixel = PixelRegion(
            left=pixel_left,
            top=pixel_top,
            right=pixel_right,
            bottom=pixel_bottom,
        )
        return DecodedGrounding(
            GroundingDisposition.ACCEPTED,
            GroundedRegion(normalized=normalized_region, pixel=pixel),
        )


def _normalized_scale(convention: GroundingCoordinateConvention) -> float:
    if convention is GroundingCoordinateConvention.NORMALIZED_1000:
        return 1000.0
    return 1.0


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
