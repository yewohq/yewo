"""Tests for fail-closed screenshot redaction."""

from io import BytesIO

from PIL import Image

from yewo.artifacts import ArtifactSensitivity, YEWO_REDACTION_POLICY_ID
from yewo.brand import CLAY, MOSS, SAND
from yewo.observations import (
    RedactionRegion,
    RegionScreenshotRedactor,
)


def test_region_redactor_covers_only_declared_sensitive_pixels() -> None:
    source = Image.new("RGB", (180, 80), color="white")
    stream = BytesIO()
    source.save(stream, format="PNG")
    redactor = RegionScreenshotRedactor(
        (
            RedactionRegion(
                left=10,
                top=10,
                right=170,
                bottom=70,
            ),
        )
    )

    result = redactor.redact(stream.getvalue())
    redacted = Image.open(BytesIO(result.png)).convert("RGB")

    colors = redacted.crop((10, 10, 170, 70)).getcolors(maxcolors=160 * 60)
    assert colors is not None
    masked_colors = {color for _, color in colors}
    assert MOSS in masked_colors
    assert SAND in masked_colors
    assert CLAY in masked_colors
    assert redacted.getpixel((2, 2)) == (255, 255, 255)
    assert result.provenance.policy_id == YEWO_REDACTION_POLICY_ID
    assert result.provenance.region_count == 1
    assert result.provenance.input_sensitivity is ArtifactSensitivity.DECLARED_SENSITIVE
