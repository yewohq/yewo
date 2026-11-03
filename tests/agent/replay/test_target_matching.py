"""Deterministic target-neighborhood matching tests."""

from io import BytesIO

from PIL import Image, ImageDraw

from yewo.agent.replay.visual_matching import (
    build_target_visual_descriptor,
    match_target_neighborhood,
)
from yewo.grounding import GroundedPoint, NormalizedPoint, PixelPoint


def _screen(target_x: int, target_y: int) -> bytes:
    image = Image.new("RGB", (320, 320), "white")
    drawing = ImageDraw.Draw(image)
    drawing.rounded_rectangle(
        (target_x - 45, target_y - 25, target_x + 45, target_y + 25),
        radius=8,
        fill="#204060",
    )
    drawing.line(
        (target_x - 30, target_y, target_x + 30, target_y),
        fill="white",
        width=5,
    )
    drawing.ellipse(
        (target_x - 10, target_y - 15, target_x + 10, target_y + 5),
        fill="#f08020",
    )
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _grounded_point(x: int, y: int) -> GroundedPoint:
    return GroundedPoint(
        normalized=NormalizedPoint(x=x / 319, y=y / 319),
        pixel=PixelPoint(x=x, y=y),
    )


def test_target_match_reuses_unchanged_neighborhood() -> None:
    png = _screen(160, 160)
    descriptor = build_target_visual_descriptor(png, PixelPoint(x=160, y=160), 320, 320)
    assert descriptor is not None

    result = match_target_neighborhood(
        png,
        descriptor,
        _grounded_point(160, 160),
        320,
        320,
    )

    assert result is not None
    assert not result.relocated
    assert result.hash_distance == 0


def test_target_match_relocates_one_unique_shift() -> None:
    source = _screen(160, 160)
    descriptor = build_target_visual_descriptor(
        source,
        PixelPoint(x=160, y=160),
        320,
        320,
    )
    assert descriptor is not None
    descriptor = descriptor.model_copy(
        update={"maximum_hash_distance": 4, "minimum_runner_up_margin": 0}
    )

    result = match_target_neighborhood(
        _screen(200, 176),
        descriptor,
        _grounded_point(160, 160),
        320,
        320,
    )

    assert result is not None
    assert result.relocated
    assert abs(result.point.pixel.x - 200) <= descriptor.search_stride_px
    assert abs(result.point.pixel.y - 176) <= descriptor.search_stride_px
