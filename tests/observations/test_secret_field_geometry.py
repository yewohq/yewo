"""Tests for placing a protected field in screenshot pixels."""

from yewo.devices.screen import ScreenCalibration
from yewo.observations.service import _secret_field_edges

# A Pixel-class phone: the driver reports density-independent points, the screenshot is pixels.
CALIBRATION = ScreenCalibration(
    screenshot_width_px=1080,
    screenshot_height_px=2400,
    driver_width_px=412,
    driver_height_px=916,
)
DIMENSIONS = (1080, 2400)


def test_driver_points_become_screenshot_pixels() -> None:
    edges = _secret_field_edges((50, 100, 362, 140), CALIBRATION, DIMENSIONS)

    # Masking the driver numbers as if they were pixels would cover the top-left eighth of the
    # screen and leave the field itself readable.
    assert edges is not None
    assert (edges[0], edges[2]) == (131, 949)
    assert (edges[1], edges[3]) == (262, 367)


def test_a_field_reported_beyond_the_screen_is_clamped_to_it() -> None:
    edges = _secret_field_edges((0, 0, 10_000, 10_000), CALIBRATION, DIMENSIONS)

    assert edges is not None
    assert (edges[2], edges[3]) == DIMENSIONS


def test_a_field_entirely_off_the_screen_leaves_nothing_to_cover() -> None:
    # Clamping collapses the rectangle, and a collapsed rectangle is not a region. Building one
    # anyway raises out of observation capture and fails the run.
    assert _secret_field_edges((5_000, 6_000, 5_400, 6_100), CALIBRATION, DIMENSIONS) is None
