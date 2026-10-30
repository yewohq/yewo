"""Normalize platform-owned screen regions into screenshot rectangles."""

from collections.abc import Mapping, Sequence


def is_inside_system_chrome(
    bounds: tuple[float, float, float, float],
    chrome: Sequence[tuple[int, int, int, int]],
) -> bool:
    """Whether the platform drew this region, which means the application did not.

    One rule, because three readers ask it. A control inside the status bar is
    not a frontier. A line inside it is not what the screen is called. A clock
    inside it is not part of the question a screen puts to the model.
    """
    left, top, right, bottom = bounds
    return any(
        chrome_left <= left
        and chrome_top <= top
        and right <= chrome_right
        and bottom <= chrome_bottom
        for chrome_left, chrome_top, chrome_right, chrome_bottom in chrome
    )


def android_system_chrome(
    system_bars: Mapping[str, object] | None,
    screen_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    if system_bars is None:
        return ()
    return tuple(
        bounds
        for value in system_bars.values()
        if (bounds := _bounded_rectangle(value, screen_size)) is not None
    )


def ios_system_chrome(
    viewport_rect: Mapping[str, object] | None,
    screen_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], ...]:
    viewport = _bounded_rectangle(viewport_rect, screen_size)
    if viewport is None:
        return ()
    width, height = screen_size
    left, top, right, bottom = viewport
    regions = (
        (0, 0, width, top),
        (0, bottom, width, height),
        (0, top, left, bottom),
        (right, top, width, bottom),
    )
    return tuple(region for region in regions if _rectangle_area(region) > 0)


def _bounded_rectangle(
    value: object,
    screen_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(value, Mapping):
        return None
    raw_coordinates = tuple(
        value.get(key) for key in ("x", "y", "width", "height")
    )
    if not all(isinstance(item, int | float) for item in raw_coordinates):
        return None
    coordinates = tuple(
        item for item in raw_coordinates if isinstance(item, int | float)
    )
    x, y, width, height = (round(item) for item in coordinates)
    screen_width, screen_height = screen_size
    left = max(0, min(screen_width, x))
    top = max(0, min(screen_height, y))
    right = max(left, min(screen_width, x + width))
    bottom = max(top, min(screen_height, y + height))
    return (left, top, right, bottom) if right > left and bottom > top else None


def _rectangle_area(bounds: tuple[int, int, int, int]) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
