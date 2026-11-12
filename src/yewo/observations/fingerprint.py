"""Versioned application-screen fingerprinting."""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO

import imagehash
from PIL import Image, ImageDraw, UnidentifiedImageError

from yewo.observations.errors import ObservationCaptureError
from yewo.observations._region_mask import system_region_mask_digest
from yewo.observations.records import ScreenFingerprint, ScreenRegion


def build_screen_fingerprint(
    png: bytes,
    system_regions: tuple[ScreenRegion, ...] | None,
    *,
    source_artifact_id: str | None,
) -> ScreenFingerprint:
    """Fingerprint the screenshot after masking known platform-owned pixels."""
    try:
        with Image.open(BytesIO(png)) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as error:
        raise ObservationCaptureError(
            "Could not decode the screenshot for fingerprinting."
        ) from error
    if system_regions is not None:
        drawing = ImageDraw.Draw(image)
        for region in system_regions:
            drawing.rectangle(
                (region.left, region.top, region.right - 1, region.bottom - 1),
                fill="black",
            )
    masked_pixels_digest = sha256()
    masked_pixels_digest.update(f"RGB:{image.width}x{image.height}\0".encode())
    masked_pixels_digest.update(image.tobytes())
    return ScreenFingerprint(
        value=str(imagehash.phash(image, hash_size=8)),
        masked_pixels_sha256=masked_pixels_digest.hexdigest(),
        system_region_mask_sha256=system_region_mask_digest(
            system_regions,
            image.width,
            image.height,
        ),
        source_artifact_id=source_artifact_id,
    )


def fingerprint_distance(left: ScreenFingerprint, right: ScreenFingerprint) -> int:
    """Return the Hamming distance between compatible fingerprints."""
    if left.algorithm != right.algorithm:
        raise ValueError("Screen fingerprints use different algorithms.")
    if left.system_region_mask_sha256 != right.system_region_mask_sha256:
        raise ValueError("Screen fingerprints used different system-region masks.")
    return (int(left.value, 16) ^ int(right.value, 16)).bit_count()
