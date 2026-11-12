"""Shared observation-record factories for deterministic tests."""

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image

from yewo.artifacts import (
    ArtifactRetention,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.inventory import DeviceKind, DevicePlatform
from yewo.observations import (
    CoordinateScale,
    Observation,
    ScreenRegion,
    ScreenFingerprint,
    ScreenshotReference,
)
from yewo.observations._region_mask import system_region_mask_digest
from yewo.observations.fingerprint import build_screen_fingerprint
from yewo.devices.screen import png_size


def build_screenshot(*, artifact_id: str = "shot_1") -> ScreenshotReference:
    return ScreenshotReference(
        artifact_id=artifact_id,
        sha256="a" * 64,
        width_px=200,
        height_px=400,
        path=Path(f"screenshots/{artifact_id}.png"),
        retention=ArtifactRetention.RUN,
    )


def build_observation(**updates: object) -> Observation:
    values: dict[str, object] = {
        "observation_id": "obs_1",
        "device": EvidenceDeviceIdentity(
            target_id="pixel_8",
            platform=DevicePlatform.ANDROID,
            kind=DeviceKind.VIRTUAL,
            operating_system_version="16",
            model="Pixel 8",
            architecture="arm64-v8a",
        ),
        "application": EvidenceApplicationIdentity(
            platform=DevicePlatform.ANDROID,
            application_id="com.example.app",
            version="2.1.0",
            build="42",
            artifact_sha256="b" * 64,
        ),
        "foreground_application_id": "com.example.app",
        "application_state": ApplicationState.FOREGROUND,
        "platform_context": "com.example.MainActivity",
        "keyboard_visible": False,
        "orientation": "portrait",
        "screenshot_to_driver_scale": CoordinateScale(x=0.5, y=0.5),
        "screenshot": build_screenshot(),
        "system_regions": None,
        "captured_at": datetime(2026, 10, 30, tzinfo=timezone.utc),
        "predecessor_observation_id": None,
    }
    values.update(updates)
    if "screen_fingerprint" not in updates:
        screenshot = values["screenshot"]
        assert isinstance(screenshot, ScreenshotReference)
        values["screen_fingerprint"] = ScreenFingerprint(
            value="0123456789abcdef",
            masked_pixels_sha256=sha256(b"0123456789abcdef").hexdigest(),
            system_region_mask_sha256=system_region_mask_digest(
                cast(tuple[ScreenRegion, ...] | None, values["system_regions"]),
                screenshot.width_px,
                screenshot.height_px,
            ),
            source_artifact_id=screenshot.artifact_id,
        )
    return Observation.model_validate(values)


def render_screen_png(seed: int, *, width_px: int = 200, height_px: int = 400) -> bytes:
    """Render one deterministic screen that a perceptual hash can tell from the others.

    Laid out in coarse bands rather than noise. A perceptual hash describes low-frequency
    structure, so two noise images hash almost alike however different their pixels are.
    """
    generator = np.random.default_rng(seed)
    pixels = np.zeros((height_px, width_px, 3), dtype=np.uint8)
    band_height = max(1, height_px // 8)
    for index in range(0, height_px, band_height):
        pixels[index : index + band_height] = generator.integers(0, 256, 3, dtype=np.uint8)
    for index in range(0, width_px, max(1, width_px // 5)):
        pixels[:, index : index + 4] = generator.integers(0, 256, 3, dtype=np.uint8)
    stream = BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return stream.getvalue()


def render_observation(
    png: bytes,
    *,
    observation_id: str = "obs_1",
    captured_at: datetime | None = None,
    **updates: object,
) -> Observation:
    """Build one observation from real pixels, so its fingerprint reflects them."""
    dimensions = png_size(png)
    assert dimensions is not None
    width_px, height_px = dimensions
    screenshot = ScreenshotReference(
        artifact_id=f"shot-{observation_id}",
        sha256=sha256(png).hexdigest(),
        width_px=width_px,
        height_px=height_px,
        path=Path(f"screenshots/shot-{observation_id}.png"),
        retention=ArtifactRetention.RUN,
    )
    values: dict[str, object] = {
        "observation_id": observation_id,
        "screenshot": screenshot,
        "screen_fingerprint": build_screen_fingerprint(
            png,
            None,
            source_artifact_id=screenshot.artifact_id,
        ),
    }
    if captured_at is not None:
        values["captured_at"] = captured_at
    values.update(updates)
    return build_observation(**values)
