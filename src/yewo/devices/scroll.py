"""Semantic scroll gesture geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from yewo.devices.errors import DeviceActionRejectedError


class ScrollDirection(StrEnum):
    """The direction to advance through scrollable content."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


def validate_scroll_distance(distance_fraction: float) -> None:
    """Bound one scroll distance before any platform dispatches the gesture."""
    if distance_fraction <= 0 or distance_fraction > 0.9:
        raise DeviceActionRejectedError("Scroll distance must be greater than 0 and at most 0.9.")


@dataclass(frozen=True)
class ScrollDispatch:
    """Whether a backend dispatched a scroll and whether content remains.

    The explicit dispatch flag distinguishes a declined gesture from a dispatched gesture whose
    backend cannot report whether more content remains. Conflating those states previously ran a
    fallback swipe over a region the backend had already scrolled.
    """

    dispatched: bool
    can_scroll_more: bool | None = None


SCROLL_DECLINED = ScrollDispatch(dispatched=False)


def scroll_gesture(
    direction: ScrollDirection,
    left: int,
    top: int,
    right: int,
    bottom: int,
    distance_fraction: float,
) -> tuple[int, int, int, int]:
    """Return start and end points for one bounded semantic scroll."""
    validate_scroll_distance(distance_fraction)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    travel_x = max(1, round((right - left) * distance_fraction))
    travel_y = max(1, round((bottom - top) * distance_fraction))
    horizontal_start = center_x - travel_x // 2
    horizontal_end = horizontal_start + travel_x
    vertical_start = center_y - travel_y // 2
    vertical_end = vertical_start + travel_y
    # A scroll direction names the content the gesture reveals, so the finger travels the
    # opposite way: scrolling down drags upward to bring lower content into view.
    return {
        ScrollDirection.UP: (center_x, vertical_start, center_x, vertical_end),
        ScrollDirection.DOWN: (center_x, vertical_end, center_x, vertical_start),
        ScrollDirection.LEFT: (horizontal_start, center_y, horizontal_end, center_y),
        ScrollDirection.RIGHT: (horizontal_end, center_y, horizontal_start, center_y),
    }[direction]
