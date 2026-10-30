"""Capture coherent observations from the device interaction boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor
from typing import Protocol

from pydantic import ValidationError

from yewo.applications.records import ApplicationDescriptor
from yewo.artifacts.records import (
    ArtifactRetention,
    ArtifactSensitivity,
    EvidenceApplicationIdentity,
    EvidenceDeviceIdentity,
    new_record_identifier,
    validate_record_identifier,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.inventory import DeviceDescriptor
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenCoordinateSpace,
    ScreenRegionReading,
    png_size,
)
from yewo.observations.errors import ObservationCaptureError
from yewo.observations.fingerprint import build_screen_fingerprint
from yewo.observations.records import (
    CoordinateScale,
    DeviceOrientation,
    Observation,
    ScreenRegion,
    ScreenshotReference,
    validate_screen_regions,
)
from yewo.observations.redaction import ScreenshotRedactor
from yewo.observations.store import ObservationStore, ScreenshotStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ObservationCapturePolicy:
    """Declare which screenshot forms this run may retain."""

    retain_raw: bool = False
    raw_retention: ArtifactRetention = ArtifactRetention.RUN
    redacted_retention: ArtifactRetention = ArtifactRetention.RUN


class ObservationSource(Protocol):
    """Read evidence without exposing the underlying Appium client."""

    def screenshot(self) -> bytes: ...

    def screen_calibration(self) -> ScreenCalibration: ...

    def active_application_id(self) -> str | None: ...

    def application_state(self) -> ApplicationState: ...

    def application_context(self) -> str | None: ...

    def keyboard_visible(self) -> bool | None: ...

    def device_orientation(self) -> str | None: ...

    def system_regions(self) -> tuple[ScreenRegionReading, ...] | None: ...


class ObservationService:
    """Capture one internally consistent observation and persist its evidence."""

    def __init__(
        self,
        source: ObservationSource,
        device: DeviceDescriptor,
        application: ApplicationDescriptor,
        screenshot_store: ScreenshotStore,
        observation_store: ObservationStore,
        redactor: ScreenshotRedactor,
        *,
        policy: ObservationCapturePolicy | None = None,
        identifier_factory: Callable[[], str] = new_record_identifier,
        time_factory: Callable[[], datetime] = _utc_now,
    ) -> None:
        if device.platform is not application.platform:
            raise ValueError("The observation device and application must use the same platform.")
        if screenshot_store.root.resolve() != observation_store.root.resolve():
            raise ValueError("Screenshot and observation stores must use the same artifact root.")
        self._source = source
        self._device = device
        self._application = application
        self._screenshot_store = screenshot_store
        self._observation_store = observation_store
        self._redactor = redactor
        self._policy = policy or ObservationCapturePolicy()
        self._identifier_factory = identifier_factory
        self._time_factory = time_factory

    def capture(
        self,
        *,
        predecessor_observation_id: str | None = None,
    ) -> Observation:
        """Return evidence captured while the foreground identity stayed stable."""
        observation_id, predecessor_id = self._validated_identifiers(predecessor_observation_id)
        foreground_before = self._source.active_application_id()
        png = self._source.screenshot()
        calibration = self._source.screen_calibration()
        application_state = self._source.application_state()
        platform_context = self._source.application_context()
        keyboard_visible = self._source.keyboard_visible()
        orientation = _normalize_orientation(self._source.device_orientation())
        driver_regions = self._source.system_regions()
        foreground_after = self._source.active_application_id()
        if foreground_before != foreground_after:
            raise ObservationCaptureError(
                "The foreground application changed during screenshot capture."
            )
        dimensions = _validated_dimensions(png, calibration)
        try:
            system_regions = _screenshot_regions(driver_regions, calibration)
            validate_screen_regions(system_regions, *dimensions)
        except (ValidationError, ValueError) as error:
            raise ObservationCaptureError(
                "The platform returned invalid system-region geometry."
            ) from error
        captured_at = self._time_factory()
        if captured_at.utcoffset() is None:
            raise ObservationCaptureError("The observation capture time must include a timezone.")
        redaction = self._redactor.redact(png)
        if png_size(redaction.png) != dimensions:
            raise ObservationCaptureError(
                "The redacted screenshot dimensions differ from the raw screenshot."
            )
        references: list[ScreenshotReference] = []
        try:
            sensitivity = redaction.provenance.input_sensitivity
            raw_screenshot = self._persist_raw(png, sensitivity, references)
            redacted_screenshot = self._screenshot_store.persist_redacted(
                redaction.png,
                sensitivity=sensitivity,
                retention=self._policy.redacted_retention,
                redaction=redaction.provenance,
            )
            references.append(redacted_screenshot)
            observation = Observation(
                observation_id=observation_id,
                device=_device_identity(self._device),
                application=_application_identity(self._application),
                foreground_application_id=foreground_after,
                application_state=application_state,
                platform_context=platform_context,
                keyboard_visible=keyboard_visible,
                orientation=orientation,
                screenshot_to_driver_scale=CoordinateScale(
                    x=calibration.scale_x,
                    y=calibration.scale_y,
                ),
                screen_fingerprint=build_screen_fingerprint(
                    redaction.png,
                    system_regions,
                    source_artifact_id=redacted_screenshot.artifact_id,
                ),
                raw_screenshot=raw_screenshot,
                redacted_screenshot=redacted_screenshot,
                system_regions=system_regions,
                captured_at=captured_at,
                predecessor_observation_id=predecessor_id,
            )
            self._observation_store.persist(observation)
        except ValidationError as error:
            capture_error = ObservationCaptureError("Captured observation evidence is invalid.")
            cleanup_errors = self._discard_incomplete_screenshots(references)
            if cleanup_errors:
                capture_error.add_note(
                    "Screenshot rollback also failed: "
                    + "; ".join(str(item) for item in cleanup_errors)
                )
            raise capture_error from error
        except BaseException as error:
            cleanup_errors = self._discard_incomplete_screenshots(references)
            if cleanup_errors:
                error.add_note(
                    "Screenshot rollback also failed: "
                    + "; ".join(str(item) for item in cleanup_errors)
                )
            raise
        return observation

    def _validated_identifiers(
        self,
        predecessor_observation_id: str | None,
    ) -> tuple[str, str | None]:
        try:
            observation_id = validate_record_identifier(self._identifier_factory())
            if predecessor_observation_id is not None:
                predecessor_observation_id = validate_record_identifier(predecessor_observation_id)
        except ValidationError as error:
            raise ObservationCaptureError("The observation identifier is invalid.") from error
        return observation_id, predecessor_observation_id

    def _persist_raw(
        self,
        png: bytes,
        sensitivity: ArtifactSensitivity,
        references: list[ScreenshotReference],
    ) -> ScreenshotReference | None:
        if not self._policy.retain_raw:
            return None
        reference = self._screenshot_store.persist_raw(
            png,
            sensitivity=sensitivity,
            retention=self._policy.raw_retention,
        )
        references.append(reference)
        return reference

    def _discard_incomplete_screenshots(
        self,
        references: list[ScreenshotReference],
    ) -> list[ObservationCaptureError]:
        errors: list[ObservationCaptureError] = []
        for reference in reversed(references):
            try:
                self._screenshot_store.discard(reference)
            except ObservationCaptureError as error:
                errors.append(error)
        return errors


def _validated_dimensions(
    png: bytes,
    calibration: ScreenCalibration,
) -> tuple[int, int]:
    dimensions = png_size(png)
    calibrated_dimensions = (
        calibration.screenshot_width_px,
        calibration.screenshot_height_px,
    )
    if dimensions is None or dimensions != calibrated_dimensions:
        raise ObservationCaptureError(
            "The screenshot dimensions do not match the coordinate calibration."
        )
    return dimensions


def _screenshot_regions(
    regions: tuple[ScreenRegionReading, ...] | None,
    calibration: ScreenCalibration,
) -> tuple[ScreenRegion, ...] | None:
    if regions is None:
        return None
    screenshot_width = calibration.screenshot_width_px
    screenshot_height = calibration.screenshot_height_px
    converted: list[ScreenRegion] = []
    for region in regions:
        if region.coordinate_space is ScreenCoordinateSpace.DRIVER:
            left = floor(region.left / calibration.scale_x)
            top = floor(region.top / calibration.scale_y)
            right = ceil(region.right / calibration.scale_x)
            bottom = ceil(region.bottom / calibration.scale_y)
        else:
            left, top, right, bottom = (
                region.left,
                region.top,
                region.right,
                region.bottom,
            )
        converted.append(
            ScreenRegion(
                kind=region.kind,
                left=max(0, min(screenshot_width, left)),
                top=max(0, min(screenshot_height, top)),
                right=max(0, min(screenshot_width, right)),
                bottom=max(0, min(screenshot_height, bottom)),
            )
        )
    return tuple(converted)


def _device_identity(device: DeviceDescriptor) -> EvidenceDeviceIdentity:
    return EvidenceDeviceIdentity(
        target_id=device.target_id,
        platform=device.platform,
        kind=device.kind,
        operating_system_version=device.operating_system_version,
        model=device.device_model,
        architecture=device.architecture,
    )


def _application_identity(
    application: ApplicationDescriptor,
) -> EvidenceApplicationIdentity:
    return EvidenceApplicationIdentity(
        platform=application.platform,
        application_id=application.application_id,
        version=application.version,
        build=application.build,
        artifact_sha256=application.artifact_sha256,
    )


def _normalize_orientation(value: str | None) -> DeviceOrientation | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized.startswith("portrait"):
        return DeviceOrientation.PORTRAIT
    if normalized.startswith("landscape"):
        return DeviceOrientation.LANDSCAPE
    return DeviceOrientation.UNKNOWN
