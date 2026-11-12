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
    ("text", "output_shape"),
    [
        ("(0.5,0.7)", GroundingOutputShape.POINT),
        ("(0.5,0.7,500,700)", GroundingOutputShape.REGION),
    ],
)
def test_normalized_1000_requires_integer_coordinates(
    text: str,
    output_shape: GroundingOutputShape,
) -> None:
    decoded = CoordinateTextCodec().decode(
        text,
        GroundingCoordinateConvention.NORMALIZED_1000,
        1080,
        1920,
        output_shape,
    )

    assert decoded.disposition is GroundingDisposition.MALFORMED
    assert decoded.location is None


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


def test_a_fractional_pixel_point_rounds_to_a_usable_coordinate() -> None:
    # .7 rounds up where truncation would not, so this pins rounding rather than int().
    decoded = CoordinateTextCodec().decode(
        "(546.2,1240.7)", GroundingCoordinateConvention.SOURCE_PIXELS, 1080, 2400
    )

    assert decoded.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(decoded.location, GroundedPoint)
    assert decoded.location.pixel.model_dump() == {"x": 546, "y": 1241}


@pytest.mark.parametrize(
    "answer",
    ["(0.51,0.42)", "(0.0, 0.73)", "(0.5, 1.0)", "(1.0, 1.0)", "(0.73, 0.0)", "(0,0)", "(1,1)"],
)
def test_a_normalized_looking_pair_is_not_read_as_pixels(answer: str) -> None:
    # Rounding any of these dispatches a tap at the screen corner instead of refusing. A strict
    # open interval let every pair carrying a 0 or a 1 through to do exactly that.
    decoded = CoordinateTextCodec().decode(
        answer, GroundingCoordinateConvention.SOURCE_PIXELS, 1080, 2400
    )

    assert decoded.disposition is GroundingDisposition.MALFORMED


@pytest.mark.parametrize("answer", ["(0, 1200)", "(540, 1200)", "(540.5, 1200.5)", "(2, 2)"])
def test_a_pixel_pair_outside_the_normalized_range_is_accepted(answer: str) -> None:
    decoded = CoordinateTextCodec().decode(
        answer, GroundingCoordinateConvention.SOURCE_PIXELS, 1080, 2400
    )

    assert decoded.disposition is GroundingDisposition.ACCEPTED


@pytest.mark.parametrize(
    "answer",
    ["(546,934)", "546,934", "[546, 934]", " 546 , 934 ", "{546,934}"],
)
def test_a_point_is_read_however_the_model_wraps_it(answer: str) -> None:
    # Gemini omits the brackets on about half its answers; the coordinates are identical.
    decoded = CoordinateTextCodec().decode(
        answer, GroundingCoordinateConvention.SOURCE_PIXELS, 1080, 2400
    )

    assert decoded.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(decoded.location, GroundedPoint)
    assert decoded.location.pixel.model_dump() == {"x": 546, "y": 934}


def test_an_unwrapped_region_is_read_as_a_region() -> None:
    decoded = CoordinateTextCodec().decode(
        "10,20,90,180",
        GroundingCoordinateConvention.SOURCE_PIXELS,
        1080,
        2400,
        GroundingOutputShape.REGION,
    )

    assert decoded.disposition is GroundingDisposition.ACCEPTED
    assert isinstance(decoded.location, GroundedRegion)
    assert decoded.location.pixel.model_dump() == {
        "left": 10,
        "top": 20,
        "right": 90,
        "bottom": 180,
    }


def test_a_mismatched_wrapper_is_still_rejected() -> None:
    decoded = CoordinateTextCodec().decode(
        "(546,934", GroundingCoordinateConvention.SOURCE_PIXELS, 1080, 2400
    )

    assert decoded.disposition is GroundingDisposition.MALFORMED
