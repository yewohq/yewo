"""Stable-region screen signature tests."""

from io import BytesIO

from PIL import Image, ImageDraw

from yewo.agent.replay.visual_matching import (
    build_state_visual_signature,
    state_visual_match_ratio,
)


def _screen(
    dynamic_pattern: int,
    stable_change: bool = False,
    major_change: bool = False,
) -> bytes:
    image = Image.new("RGB", (300, 500), "white")
    drawing = ImageDraw.Draw(image)
    if dynamic_pattern == 1:
        drawing.line((5, 5, 45, 45), fill="black", width=4)
    elif dynamic_pattern == 2:
        drawing.ellipse((5, 5, 45, 45), outline="black", width=4)
    else:
        drawing.rectangle((5, 5, 45, 45), outline="black", width=4)
    if stable_change:
        drawing.line((55, 55, 95, 95), fill="black", width=5)
    if major_change:
        for top in range(100, 250, 50):
            for left in range(0, 300, 50):
                drawing.line((left + 5, top + 5, left + 45, top + 45), fill="black", width=5)
                drawing.line((left + 45, top + 5, left + 5, top + 45), fill="black", width=5)
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_state_signature_ignores_observed_volatile_tile() -> None:
    signature = build_state_visual_signature((_screen(1), _screen(2)))
    assert signature is not None
    assert len(signature.stable_tiles) < signature.column_count * signature.row_count

    ratio = state_visual_match_ratio(_screen(3), signature)

    assert ratio == 1.0


def test_signature_tolerance_reflects_observation_count() -> None:
    one_sample = build_state_visual_signature((_screen(1),))
    two_samples = build_state_visual_signature((_screen(1), _screen(1)))
    three_samples = build_state_visual_signature((_screen(1), _screen(1), _screen(1)))

    assert one_sample is not None
    assert two_samples is not None
    assert three_samples is not None
    assert one_sample.minimum_match_ratio == 0.8
    assert two_samples.minimum_match_ratio == 0.85
    assert three_samples.minimum_match_ratio == 0.9


def test_state_signature_tolerates_one_changed_stable_tile() -> None:
    signature = build_state_visual_signature((_screen(1), _screen(2)))
    assert signature is not None

    ratio = state_visual_match_ratio(_screen(3, stable_change=True), signature)

    assert ratio is not None
    assert ratio < 1
    assert ratio >= signature.minimum_match_ratio


def test_state_signature_rejects_large_stable_region_change() -> None:
    signature = build_state_visual_signature((_screen(1), _screen(2)))
    assert signature is not None

    ratio = state_visual_match_ratio(_screen(3, major_change=True), signature)

    assert ratio is not None
    assert ratio < signature.minimum_match_ratio
