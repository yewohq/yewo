"""Tests for fail-closed screenshot redaction."""

from io import BytesIO

from PIL import Image

from yewo.observations import (
    ArtifactSensitivity,
    RedactionRegion,
    RegionScreenshotRedactor,
)


def test_region_redactor_covers_only_declared_sensitive_pixels() -> None:
    source = Image.new("RGB", (20, 20), color="white")
    stream = BytesIO()
    source.save(stream, format="PNG")
    redactor = RegionScreenshotRedactor(
        (
            RedactionRegion(
                left=5,
                top=5,
                right=15,
                bottom=15,
            ),
        )
    )

    result = redactor.redact(stream.getvalue())
    redacted = Image.open(BytesIO(result.png)).convert("RGB")

    assert redacted.getpixel((10, 10)) == (0, 0, 0)
    assert redacted.getpixel((2, 2)) == (255, 255, 255)
    assert result.provenance.policy_id == "region-mask"
    assert result.provenance.region_count == 1
    assert result.provenance.input_sensitivity is ArtifactSensitivity.DECLARED_SENSITIVE
