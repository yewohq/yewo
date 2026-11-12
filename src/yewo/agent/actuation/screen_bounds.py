"""Whether a grounded target lies inside the application-owned screen."""

from __future__ import annotations

from yewo.grounding import PixelPoint
from yewo.observations import Observation, PixelRegion, ScreenRegion


def is_point_in_screenshot(point: PixelPoint, observation: Observation) -> bool:
    """Whether one aim point addresses a pixel the screenshot contains."""
    screenshot = observation.screenshot
    return point.x < screenshot.width_px and point.y < screenshot.height_px


def is_region_in_screenshot(region: PixelRegion, observation: Observation) -> bool:
    """Whether one half-open rectangle fits inside the screenshot."""
    screenshot = observation.screenshot
    return region.right <= screenshot.width_px and region.bottom <= screenshot.height_px


def is_point_in_system_region(
    point: PixelPoint,
    regions: tuple[ScreenRegion, ...] | None,
) -> bool:
    """Whether one aim point lands on platform-owned chrome rather than the application."""
    return any(
        region.left <= point.x < region.right and region.top <= point.y < region.bottom
        for region in regions or ()
    )


def overlaps_system_region(
    target: PixelRegion,
    regions: tuple[ScreenRegion, ...] | None,
) -> bool:
    """Whether one target rectangle reaches into platform-owned chrome."""
    return any(
        target.left < region.right
        and target.right > region.left
        and target.top < region.bottom
        and target.bottom > region.top
        for region in regions or ()
    )


__all__ = [
    "is_point_in_screenshot",
    "is_point_in_system_region",
    "is_region_in_screenshot",
    "overlaps_system_region",
]
