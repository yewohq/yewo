"""Scroll gesture geometry for content-relative directions."""

import pytest

from yewo.devices.errors import DeviceActionRejectedError
from yewo.devices.scroll import ScrollDirection, scroll_gesture


@pytest.mark.parametrize(
    ("direction", "expected"),
    [
        # Revealing lower content drags the finger upward, and the reverse for each axis.
        (ScrollDirection.DOWN, (50, 160, 50, 40)),
        (ScrollDirection.UP, (50, 40, 50, 160)),
        (ScrollDirection.RIGHT, (80, 100, 20, 100)),
        (ScrollDirection.LEFT, (20, 100, 80, 100)),
    ],
)
def test_gesture_travels_opposite_to_the_revealed_content(
    direction: ScrollDirection,
    expected: tuple[int, int, int, int],
) -> None:
    assert scroll_gesture(direction, 0, 0, 100, 200, 0.6) == expected


def test_gesture_rejects_an_unbounded_distance() -> None:
    with pytest.raises(DeviceActionRejectedError, match="at most 0.9"):
        scroll_gesture(ScrollDirection.DOWN, 0, 0, 100, 200, 1.0)
