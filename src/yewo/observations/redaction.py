"""Fail-closed screenshot redaction boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
from typing import Protocol

from PIL import Image, ImageDraw, UnidentifiedImageError

from yewo.artifacts.records import ArtifactSensitivity
from yewo.observations.errors import ObservationCaptureError
from yewo.observations.records import (
    RedactionProvenance,
    RedactionRegion,
)


@dataclass(frozen=True)
class RedactionResult:
    """Processed PNG bytes and evidence describing the applied mask."""

    png: bytes
    provenance: RedactionProvenance


class ScreenshotRedactor(Protocol):
    """Return PNG pixels after applying one approved redaction policy."""

    def redact(self, png: bytes) -> RedactionResult: ...


class RegionScreenshotRedactor:
    """Cover declared sensitive rectangles with opaque pixels."""

    def __init__(self, regions: Sequence[RedactionRegion]) -> None:
        self._regions = tuple(regions)

    def redact(self, png: bytes) -> RedactionResult:
        try:
            with Image.open(BytesIO(png)) as source:
                image = source.convert("RGB")
        except (OSError, UnidentifiedImageError) as error:
            raise ObservationCaptureError(
                "Could not decode the screenshot for redaction."
            ) from error
        width, height = image.size
        if any(region.right > width or region.bottom > height for region in self._regions):
            raise ObservationCaptureError(
                "A screenshot redaction region is outside the screenshot."
            )
        drawing = ImageDraw.Draw(image)
        for region in self._regions:
            drawing.rectangle(
                (region.left, region.top, region.right - 1, region.bottom - 1),
                fill="black",
            )
        stream = BytesIO()
        image.save(stream, format="PNG")
        ordered_regions = sorted(
            self._regions,
            key=lambda item: (item.left, item.top, item.right, item.bottom),
        )
        mask = json.dumps(
            [region.model_dump(mode="json") for region in ordered_regions],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return RedactionResult(
            png=stream.getvalue(),
            provenance=RedactionProvenance(
                policy_id="region-mask",
                region_count=len(self._regions),
                region_set_sha256=sha256(mask).hexdigest(),
                input_sensitivity=(
                    ArtifactSensitivity.DECLARED_SENSITIVE
                    if self._regions
                    else ArtifactSensitivity.DEVICE_CONTENT
                ),
            ),
        )
