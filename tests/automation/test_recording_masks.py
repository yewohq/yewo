"""Tests for pre-recording redaction-mask geometry."""

import pytest

from yewo.automation.errors import AutomationSessionError
from yewo.automation.recording_masks import build_initial_recording_masks
from yewo.observations import RedactionRegion


def _png_header(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n" + (b"\x00" * 8) + width.to_bytes(4, "big") + height.to_bytes(4, "big")
    )


class _DriverStub:
    def __init__(self, screenshot: bytes) -> None:
        self.screenshot = screenshot

    def get_screenshot_as_png(self) -> bytes:
        return self.screenshot


def test_declared_regions_are_bound_to_current_screenshot_geometry() -> None:
    masks = build_initial_recording_masks(
        _DriverStub(_png_header(100, 200)),
        (RedactionRegion(left=10, top=20, right=30, bottom=40),),
    )

    assert len(masks) == 1
    assert masks[0].source_width_px == 100
    assert masks[0].source_height_px == 200
    assert masks[0].starts_at_seconds == 0


def test_invalid_screenshot_fails_before_recording_can_start() -> None:
    with pytest.raises(AutomationSessionError, match="screenshot is invalid"):
        build_initial_recording_masks(
            _DriverStub(b"not a png"),
            (RedactionRegion(left=0, top=0, right=1, bottom=1),),
        )


def test_out_of_bounds_region_fails_before_recording_can_start() -> None:
    with pytest.raises(AutomationSessionError, match="region is invalid"):
        build_initial_recording_masks(
            _DriverStub(_png_header(10, 10)),
            (RedactionRegion(left=0, top=0, right=11, bottom=10),),
        )
