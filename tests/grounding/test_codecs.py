"""Tests for strict grounding response codecs."""

import pytest

from yewo.grounding import (
    CoordinateTextCodec,
    GroundedPoint,
    GroundedRegion,
    GroundingCoordinateConvention,
    GroundingDisposition,
    GroundingOutputShape,
)


@pytest.mark.parametrize(
    ("text", "disposition"),
    [
        ("point=(10,20)", GroundingDisposition.MALFORMED),
        ("(10,20) or (30,40)", GroundingDisposition.MALFORMED),
        ("(-1,20)", GroundingDisposition.OUT_OF_BOUNDS),
        ("(200,20)", GroundingDisposition.OUT_OF_BOUNDS),
        ("REFUSED: no matching control", GroundingDisposition.REFUSED),
        ("AMBIGUOUS: two save buttons", GroundingDisposition.AMBIGUOUS),
    ],
)
def test_source_pixel_codec_rejects_or_preserves_nonaccepted_results(
    text: str,
    disposition: GroundingDisposition,
) -> None:
    decoded = CoordinateTextCodec().decode(
        text,
        GroundingCoordinateConvention.SOURCE_PIXELS,
        200,
        400,
    )

    assert decoded.disposition is disposition
    assert decoded.location is None
    assert decoded.reason is not None


def test_source_pixel_codec_preserves_pixel_and_normalized_coordinates() -> None:
    decoded = CoordinateTextCodec().decode(
        "(199,399)",
        GroundingCoordinateConvention.SOURCE_PIXELS,
        200,
        400,
    )

    assert decoded.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(decoded.location, GroundedPoint)
    assert decoded.location.pixel.model_dump() == {"x": 199, "y": 399}
    assert decoded.location.normalized.model_dump() == {"x": 1.0, "y": 1.0}


def test_normalized_codec_projects_against_source_pixel_centres() -> None:
    decoded = CoordinateTextCodec().decode(
        "(0.5,1.0)",
        GroundingCoordinateConvention.NORMALIZED,
        201,
        401,
    )

    assert isinstance(decoded.location, GroundedPoint)
    assert decoded.location.pixel.model_dump() == {"x": 100, "y": 400}


def test_normalized_codec_distinguishes_invalid_format_from_out_of_bounds() -> None:
    codec = CoordinateTextCodec()

    assert (
        codec.decode("(1.1,0.5)", GroundingCoordinateConvention.NORMALIZED, 200, 400).disposition
        is GroundingDisposition.OUT_OF_BOUNDS
    )
    assert (
        codec.decode("(nan,0.5)", GroundingCoordinateConvention.NORMALIZED, 200, 400).disposition
        is GroundingDisposition.MALFORMED
    )


@pytest.mark.parametrize(
    ("convention", "text", "expected_pixel"),
    [
        (
            GroundingCoordinateConvention.SOURCE_PIXELS,
            "(20,40,180,360)",
            {"left": 20, "top": 40, "right": 180, "bottom": 360},
        ),
        (
            GroundingCoordinateConvention.NORMALIZED,
            "(0.1,0.1,0.9,0.9)",
            {"left": 20, "top": 40, "right": 180, "bottom": 360},
        ),
    ],
)
def test_region_codec_preserves_half_open_bounds(
    convention: GroundingCoordinateConvention,
    text: str,
    expected_pixel: dict[str, int],
) -> None:
    decoded = CoordinateTextCodec().decode(
        text,
        convention,
        200,
        400,
        GroundingOutputShape.REGION,
    )

    assert decoded.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(decoded.location, GroundedRegion)
    assert decoded.location.pixel.model_dump() == expected_pixel


@pytest.mark.parametrize(
    ("text", "disposition"),
    [
        ("(10,20,10,30)", GroundingDisposition.MALFORMED),
        ("(10,20,201,30)", GroundingDisposition.OUT_OF_BOUNDS),
        ("(10,20)", GroundingDisposition.MALFORMED),
    ],
)
def test_region_codec_rejects_invalid_geometry(
    text: str,
    disposition: GroundingDisposition,
) -> None:
    decoded = CoordinateTextCodec().decode(
        text,
        GroundingCoordinateConvention.SOURCE_PIXELS,
        200,
        400,
        GroundingOutputShape.REGION,
    )

    assert decoded.disposition is disposition
    assert decoded.location is None


@pytest.mark.parametrize(
    "text",
    [
        "(0.501,0.2,0.504,0.8)",
        "(0.2,0.501,0.8,0.504)",
        "(0.001,0.001,0.004,0.004)",
    ],
)
def test_normalized_regions_that_collapse_in_pixels_are_malformed(text: str) -> None:
    decoded = CoordinateTextCodec().decode(
        text,
        GroundingCoordinateConvention.NORMALIZED,
        100,
        100,
        GroundingOutputShape.REGION,
    )

    assert decoded.disposition is GroundingDisposition.MALFORMED
    assert decoded.location is None
