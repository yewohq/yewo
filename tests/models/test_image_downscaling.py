"""Tests for billing-side image downscaling at the provider boundary."""

from io import BytesIO

from PIL import Image, ImageDraw
import pytest

from pathlib import Path

from yewo.artifacts import ArtifactRetention
from yewo.models import (
    ImageModelContent,
    ModelMessage,
    ModelMessageRole,
    TextModelContent,
)
from yewo.models.artifacts import ModelImageReader
from yewo.models.message_encoding import ModelMessageEncoder, _downscaled_png
from yewo.observations import ScreenshotStore

from models.support import build_model_profile


def _png(width: int, height: int) -> bytes:
    """Draw a cheap two-tone image. These tests measure geometry, never content.

    Noise was generated here, which costs the random draw and then defeats PNG compression at
    a few megapixels. One band compresses to almost nothing and resizes to different bytes
    just the same.
    """
    image = Image.new("RGB", (width, height), (240, 240, 240))
    ImageDraw.Draw(image).rectangle((0, 0, max(width // 2, 1) - 1, height - 1), fill=(20, 20, 20))
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _size(png: bytes) -> tuple[int, int]:
    with Image.open(BytesIO(png)) as image:
        return image.size


def _encoder_with_oversized_image(
    tmp_path: Path,
    *,
    max_long_edge_px: int | None = 1_568,
) -> tuple[ModelMessageEncoder, tuple[ModelMessage, ...]]:
    png = _png(2_000, 1_000)
    screenshot = ScreenshotStore(tmp_path, identifier_factory=lambda: "shot-1").persist(
        png,
        retention=ArtifactRetention.RUN,
    )
    profile = build_model_profile()
    profile = profile.model_copy(
        update={
            "images": profile.images.model_copy(
                update={
                    "max_image_bytes": 40_000_000,
                    "max_width_px": 4_096,
                    "max_height_px": 4_096,
                    "max_long_edge_px": max_long_edge_px,
                }
            )
        }
    )
    messages = (
        ModelMessage(
            role=ModelMessageRole.USER,
            content=(
                TextModelContent(text="Read this screen."),
                ImageModelContent(screenshot=screenshot),
            ),
        ),
    )
    return ModelMessageEncoder(ModelImageReader(tmp_path), profile), messages


def test_an_unset_long_edge_leaves_the_image_untouched() -> None:
    png = _png(1080, 2400)

    assert _downscaled_png(png, None) is png


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1080, 2400, (706, 1568)),
        (2400, 1080, (1568, 706)),
        (1600, 1600, (1568, 1568)),
    ],
)
def test_the_long_edge_is_capped_and_the_aspect_ratio_is_preserved(
    width: int,
    height: int,
    expected: tuple[int, int],
) -> None:
    resized = _downscaled_png(_png(width, height), 1568)

    assert _size(resized) == expected
    assert max(_size(resized)) == 1568


def test_a_very_narrow_image_keeps_at_least_one_pixel_on_its_short_edge() -> None:
    resized = _downscaled_png(_png(2000, 1), 1568)

    assert _size(resized) == (1568, 1)


def test_the_ledger_records_the_image_the_provider_was_given(tmp_path: Path) -> None:
    # A reference that names only the stored artifact cannot prove what the provider read, and a
    # downscaled image is a different image.
    encoder, messages = _encoder_with_oversized_image(tmp_path)

    encoder.provider_messages(messages)
    artifacts = encoder.artifact_messages(messages)

    content = artifacts[0]["content"]
    assert isinstance(content, list)
    reference = content[1]
    sent = reference["sent"]
    assert sent["sha256"] != reference["sha256"]
    assert max(sent["width_px"], sent["height_px"]) == 1_568
    assert sent["transform"] == "downscaled-long-edge"


def test_an_untransformed_image_records_no_second_description(tmp_path: Path) -> None:
    encoder, messages = _encoder_with_oversized_image(tmp_path, max_long_edge_px=None)

    encoder.provider_messages(messages)
    artifacts = encoder.artifact_messages(messages)

    content = artifacts[0]["content"]
    assert isinstance(content, list)
    assert "sent" not in content[1]
