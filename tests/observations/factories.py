"""Shared observation-record factories for deterministic tests."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import cast

from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.inventory import DeviceKind, DevicePlatform
from yewo.observations import (
    CoordinateScale,
    Observation,
    RedactionProvenance,
    ScreenRegion,
    ScreenFingerprint,
    ScreenshotRedactionState,
    ScreenshotReference,
)
from yewo.observations._region_mask import system_region_mask_digest


def build_screenshot(
    *,
    artifact_id: str = "shot_1",
    redaction_state: ScreenshotRedactionState = ScreenshotRedactionState.REDACTED,
) -> ScreenshotReference:
    return ScreenshotReference(
        artifact_id=artifact_id,
        sha256="a" * 64,
        width_px=200,
        height_px=400,
        path=Path(f"assets/screenshots/{artifact_id}.png"),
        redaction_state=redaction_state,
        redaction=(
            RedactionProvenance(
                policy_id="region-mask",
                region_count=0,
                region_set_sha256="c" * 64,
                input_sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            )
            if redaction_state is ScreenshotRedactionState.REDACTED
            else None
        ),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
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
        "raw_screenshot": None,
        "redacted_screenshot": build_screenshot(),
        "system_regions": None,
        "captured_at": datetime(2026, 10, 30, tzinfo=timezone.utc),
        "predecessor_observation_id": None,
    }
    values.update(updates)
    if "screen_fingerprint" not in updates:
        screenshot = values["redacted_screenshot"]
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
