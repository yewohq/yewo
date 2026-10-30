"""Convert declared screenshot regions into initial recording masks."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import ValidationError

from yewo.automation.errors import AutomationSessionError
from yewo.devices.recording import ScreenRecordingMask
from yewo.devices.screen import png_size
from yewo.observations import RedactionRegion


def build_initial_recording_masks(
    driver: object,
    regions: Sequence[RedactionRegion],
) -> tuple[ScreenRecordingMask, ...]:
    """Bind declared pixel regions to geometry captured before recording starts."""
    if not regions:
        return ()
    screenshot = getattr(driver, "get_screenshot_as_png", None)
    try:
        png = screenshot() if callable(screenshot) else None
    except Exception as error:  # noqa: BLE001 - external driver boundary
        raise AutomationSessionError("Could not capture recording-mask source geometry.") from error
    dimensions = png_size(png) if isinstance(png, bytes) else None
    if dimensions is None:
        raise AutomationSessionError("The recording-mask source screenshot is invalid.")
    width, height = dimensions
    try:
        return tuple(
            ScreenRecordingMask(
                region=region,
                source_width_px=width,
                source_height_px=height,
                starts_at_seconds=0,
            )
            for region in regions
        )
    except ValidationError as error:
        raise AutomationSessionError("A recording redaction region is invalid.") from error
