"""Semantic scroll gesture geometry."""

from __future__ import annotations

from enum import StrEnum

from yewo.devices.errors import DeviceActionRejectedError


class ScrollDirection(StrEnum):
    """A semantic direction for moving visible content."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


def scroll_gesture(
    direction: ScrollDirection,
    left: int,
    top: int,
    right: int,
    bottom: int,
    distance_fraction: float,
) -> tuple[int, int, int, int]:
    """Return start and end points for one bounded semantic scroll."""
    if distance_fraction <= 0 or distance_fraction > 0.9:
        raise DeviceActionRejectedError("Scroll distance must be greater than 0 and at most 0.9.")
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    travel_x = max(1, round((right - left) * distance_fraction))
    travel_y = max(1, round((bottom - top) * distance_fraction))
    horizontal_start = center_x - travel_x // 2
    horizontal_end = horizontal_start + travel_x
    vertical_start = center_y - travel_y // 2
    vertical_end = vertical_start + travel_y
    return {
        ScrollDirection.UP: (center_x, vertical_end, center_x, vertical_start),
        ScrollDirection.DOWN: (center_x, vertical_start, center_x, vertical_end),
        ScrollDirection.LEFT: (horizontal_end, center_y, horizontal_start, center_y),
        ScrollDirection.RIGHT: (horizontal_start, center_y, horizontal_end, center_y),
    }[direction]
