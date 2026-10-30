"""Tests for versioned application-screen fingerprints."""

from io import BytesIO

from PIL import Image, ImageDraw

from yewo.devices import SystemRegionKind
from yewo.observations import (
    ScreenRegion,
    build_screen_fingerprint,
    fingerprint_distance,
)


def _screen(*, clock_offset: int, alternate_state: bool = False) -> bytes:
    image = Image.new("RGB", (200, 400), color="white")
    drawing = ImageDraw.Draw(image)
    drawing.rectangle((0, 0, 199, 39), fill="gray")
    drawing.rectangle((clock_offset, 10, clock_offset + 20, 20), fill="black")
    if alternate_state:
        drawing.rectangle((40, 120, 160, 300), fill="black")
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_fingerprint_ignores_changes_inside_known_system_regions() -> None:
    status_bar = ScreenRegion(
        kind=SystemRegionKind.STATUS_BAR,
        left=0,
        top=0,
        right=200,
        bottom=40,
    )

    first = build_screen_fingerprint(
        _screen(clock_offset=10),
        (status_bar,),
        source_artifact_id="shot_1",
    )
    second = build_screen_fingerprint(
        _screen(clock_offset=80),
        (status_bar,),
        source_artifact_id="shot_2",
    )

    assert first.value == second.value
    assert fingerprint_distance(first, second) == 0
    assert first.system_region_mask_sha256 is not None


def test_fingerprint_distinguishes_different_application_states() -> None:
    baseline = build_screen_fingerprint(
        _screen(clock_offset=10), (), source_artifact_id="shot_1"
    )
    alternate = build_screen_fingerprint(
        _screen(clock_offset=10, alternate_state=True),
        (),
        source_artifact_id="shot_2",
    )

    assert fingerprint_distance(baseline, alternate) > 0


def test_fingerprint_rejects_unknown_region_comparison() -> None:
    known = build_screen_fingerprint(
        _screen(clock_offset=10), (), source_artifact_id="shot_1"
    )
    unknown = build_screen_fingerprint(
        _screen(clock_offset=10), None, source_artifact_id="shot_2"
    )

    try:
        fingerprint_distance(known, unknown)
    except ValueError as error:
        assert "system-region masks" in str(error)
    else:
        raise AssertionError("Expected incomparable fingerprints to be rejected.")


def test_fingerprint_rejects_different_known_region_masks() -> None:
    top = ScreenRegion(
        kind=SystemRegionKind.STATUS_BAR,
        left=0,
        top=0,
        right=200,
        bottom=40,
    )
    bottom = ScreenRegion(
        kind=SystemRegionKind.NAVIGATION_BAR,
        left=0,
        top=360,
        right=200,
        bottom=400,
    )
    first = build_screen_fingerprint(
        _screen(clock_offset=10), (top,), source_artifact_id="shot_1"
    )
    second = build_screen_fingerprint(
        _screen(clock_offset=10), (bottom,), source_artifact_id="shot_2"
    )

    try:
        fingerprint_distance(first, second)
    except ValueError as error:
        assert "system-region masks" in str(error)
    else:
        raise AssertionError("Expected different masks to be rejected.")


def test_fingerprint_accepts_equivalent_effective_region_masks() -> None:
    whole = ScreenRegion(
        kind=SystemRegionKind.STATUS_BAR,
        left=0,
        top=0,
        right=200,
        bottom=40,
    )
    left = ScreenRegion(
        kind=SystemRegionKind.OTHER,
        left=0,
        top=0,
        right=100,
        bottom=40,
    )
    right = ScreenRegion(
        kind=SystemRegionKind.OTHER,
        left=100,
        top=0,
        right=200,
        bottom=40,
    )
    first = build_screen_fingerprint(
        _screen(clock_offset=10), (whole,), source_artifact_id="shot_1"
    )
    second = build_screen_fingerprint(
        _screen(clock_offset=80), (right, left), source_artifact_id="shot_2"
    )

    assert fingerprint_distance(first, second) == 0
