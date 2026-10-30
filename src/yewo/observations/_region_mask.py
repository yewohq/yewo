"""Canonical pixel-mask identity for comparable screen fingerprints."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

from PIL import Image, ImageDraw


class PixelBounds(Protocol):
    """Half-open integer bounds used to build a pixel mask."""

    left: int
    top: int
    right: int
    bottom: int


def system_region_mask_digest(
    regions: tuple[PixelBounds, ...] | None,
    width: int,
    height: int,
) -> str | None:
    """Return the digest of the effective system-region pixel mask."""
    if regions is None:
        return None
    mask = Image.new("1", (width, height), color=0)
    drawing = ImageDraw.Draw(mask)
    for region in regions:
        drawing.rectangle(
            (region.left, region.top, region.right - 1, region.bottom - 1),
            fill=1,
        )
    identity = b"system-region-mask-v1\0"
    identity += width.to_bytes(8, "big") + height.to_bytes(8, "big")
    identity += mask.tobytes()
    return sha256(identity).hexdigest()
