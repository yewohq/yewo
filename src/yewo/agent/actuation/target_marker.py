"""Mark where an action was aimed, for a reader that judges what the action did."""

from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageDraw, UnidentifiedImageError

from yewo.brand import CLAY
from yewo.grounding import PixelPoint
from yewo.observations.errors import ObservationCaptureError


# Proportion of the screenshot's short edge, so the marker reads the same on any density.
_RADIUS_RATIO = 0.028
_MINIMUM_RADIUS_PX = 12
_STROKE_RATIO = 0.22


def draw_target_marker(png: bytes, target: PixelPoint) -> bytes:
    """Return the screenshot with a ring around the point the action was aimed at.

    A ring rather than a filled dot, because the reader has to judge the control underneath it.
    """
    try:
        with Image.open(BytesIO(png)) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ObservationCaptureError("Could not decode the screenshot to mark it.") from error
    radius = max(_MINIMUM_RADIUS_PX, round(min(image.size) * _RADIUS_RATIO))
    drawing = ImageDraw.Draw(image)
    drawing.ellipse(
        (target.x - radius, target.y - radius, target.x + radius, target.y + radius),
        outline=CLAY,
        width=max(2, round(radius * _STROKE_RATIO)),
    )
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


__all__ = ["draw_target_marker"]
