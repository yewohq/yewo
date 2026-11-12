"""Capture coherent observations from the device interaction boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil, floor
import time
from typing import Protocol, Self

from pydantic import ValidationError

from yewo.applications.records import ApplicationDescriptor
from yewo.artifacts.records import (
    ArtifactRetention,
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
    ScreenFingerprint,
    ScreenRegion,
    ScreenshotReference,
    validate_screen_regions,
)
from yewo.observations.store import ObservationStore, ScreenshotStore
from yewo.run_narration import current_narrator, elapsed_milliseconds


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ObservationCapturePolicy:
    """How long a captured screenshot is kept, and how it is classified."""

    retention: ArtifactRetention = ArtifactRetention.RUN


@dataclass(frozen=True, slots=True)
class ObservationProbe:
    """Non-persisted visual and platform state used only to detect settlement."""

    device: EvidenceDeviceIdentity
    application: EvidenceApplicationIdentity
    foreground_application_id: str | None
    application_state: ApplicationState
    platform_context: str | None
    keyboard_visible: bool | None
    orientation: DeviceOrientation | None
    screenshot_to_driver_scale: CoordinateScale
    width_px: int
    height_px: int
    fingerprint: ScreenFingerprint

    @classmethod
    def from_observation(cls, observation: Observation) -> Self:
        """Project one persisted observation into the lightweight comparison contract."""
        return cls(
            device=observation.device,
            application=observation.application,
            foreground_application_id=observation.foreground_application_id,
            application_state=observation.application_state,
            platform_context=observation.platform_context,
            keyboard_visible=observation.keyboard_visible,
            orientation=observation.orientation,
            screenshot_to_driver_scale=observation.screenshot_to_driver_scale,
            width_px=observation.screenshot.width_px,
            height_px=observation.screenshot.height_px,
            fingerprint=observation.screen_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class VisualSettlementProbe:
    """Non-persisted visual state used only to detect screen settlement."""

    device: EvidenceDeviceIdentity
    application: EvidenceApplicationIdentity
    screenshot_to_driver_scale: CoordinateScale
    width_px: int
    height_px: int
    fingerprint: ScreenFingerprint

    @classmethod
    def from_observation(cls, observation: Observation) -> Self:
        """Project one persisted observation into the visual settlement contract."""
        return cls(
            device=observation.device,
            application=observation.application,
            screenshot_to_driver_scale=observation.screenshot_to_driver_scale,
            width_px=observation.screenshot.width_px,
            height_px=observation.screenshot.height_px,
            fingerprint=observation.screen_fingerprint,
        )


@dataclass(frozen=True, slots=True)
class ScreenCapture:
    """One capture of the screen, in the two forms a run needs.

    `image` is persisted and shown to a model.

    A caller hashes `comparable` to tell one screen from another, then drops the bytes. Nothing
    persists, sends or logs them.

    A source that changes the image leaves this view as the device gave it. So a change the
    edit would hide still counts, and an edit that differs between two identical screens cannot
    read as movement.
    """

    image: bytes
    comparable: bytes


class ObservationSource(Protocol):
    """Read evidence without exposing the underlying Appium client."""

    def screenshot(self) -> ScreenCapture: ...

    def comparable_screenshot(self) -> bytes:
        """Return only the bytes a caller hashes and drops.

        Separate from `screenshot`, because a source that changes the image pays for that work
        on every call. A caller that only wants to know whether the screen moved should not pay
        it at all.
        """
        ...

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
        *,
        policy: ObservationCapturePolicy | None = None,
        identifier_factory: Callable[[], str] = new_record_identifier,
        time_factory: Callable[[], datetime] = _utc_now,
        monotonic: Callable[[], float] = time.monotonic,
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
        self._policy = policy or ObservationCapturePolicy()
        self._identifier_factory = identifier_factory
        self._time_factory = time_factory
        self._monotonic = monotonic

    def capture_screen_fingerprint(self) -> ScreenFingerprint:
        """Fingerprint the screen now, without recording it as a step in the run.

        A caller that only needs to know whether the screen changed does not need a persisted
        observation. Recording one would place evidence about a dispatch into the lineage of the
        screens the run visited.
        """
        png = self._source.comparable_screenshot()
        calibration = self._source.screen_calibration()
        dimensions = _validated_dimensions(png, calibration)
        try:
            system_regions = _screenshot_regions(self._source.system_regions(), calibration)
            validate_screen_regions(system_regions, *dimensions)
        except (ValidationError, ValueError) as error:
            raise ObservationCaptureError(
                "The platform returned invalid system-region geometry."
            ) from error
        return build_screen_fingerprint(png, system_regions, source_artifact_id=None)

    def capture_settlement_probe(self) -> VisualSettlementProbe:
        """Capture visual stability without dynamic application metadata or persistence."""
        png = self._source.comparable_screenshot()
        calibration = self._source.screen_calibration()
        dimensions = _validated_dimensions(png, calibration)
        try:
            system_regions = _screenshot_regions(self._source.system_regions(), calibration)
            validate_screen_regions(system_regions, *dimensions)
        except (ValidationError, ValueError) as error:
            raise ObservationCaptureError(
                "The platform returned invalid system-region geometry."
            ) from error
        return VisualSettlementProbe(
            device=_device_identity(self._device),
            application=_application_identity(self._application),
            screenshot_to_driver_scale=CoordinateScale(
                x=calibration.scale_x,
                y=calibration.scale_y,
            ),
            width_px=dimensions[0],
            height_px=dimensions[1],
            fingerprint=build_screen_fingerprint(png, system_regions, source_artifact_id=None),
        )

    def capture(
        self,
        *,
        predecessor_observation_id: str | None = None,
    ) -> Observation:
        """Return screenshot evidence with one platform foreground reading."""
        started = self._monotonic()
        disposition = "failed"
        try:
            observation = self._capture(predecessor_observation_id)
            disposition = "completed"
            return observation
        finally:
            narrator = current_narrator()
            if narrator is not None:
                duration_ms = elapsed_milliseconds(started, self._monotonic())
                narrator.trace(
                    "device-observation",
                    f"capture={disposition} duration_ms={duration_ms:.1f}",
                )

    def _capture(
        self,
        predecessor_observation_id: str | None,
    ) -> Observation:
        """Capture and persist one observation."""
        observation_id, predecessor_id = self._validated_identifiers(predecessor_observation_id)
        phase_started = self._monotonic()
        capture = self._source.screenshot()
        self._trace_phase("source_screenshot", phase_started)
        phase_started = self._monotonic()
        foreground_application_id = self._source.active_application_id()
        calibration = self._source.screen_calibration()
        application_state = self._source.application_state()
        platform_context = self._source.application_context()
        keyboard_visible = self._source.keyboard_visible()
        orientation = _normalize_orientation(self._source.device_orientation())
        driver_regions = self._source.system_regions()
        dimensions = _validated_dimensions(capture.image, calibration)
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
        self._trace_phase("metadata", phase_started)
        references: list[ScreenshotReference] = []
        try:
            phase_started = self._monotonic()
            screenshot = self._screenshot_store.persist(
                capture.image,
                retention=self._policy.retention,
            )
            references.append(screenshot)
            self._trace_phase("screenshot_persist", phase_started)
            phase_started = self._monotonic()
            fingerprint = build_screen_fingerprint(
                capture.comparable,
                system_regions,
                source_artifact_id=screenshot.artifact_id,
            )
            self._trace_phase("fingerprint", phase_started)
            observation = Observation(
                observation_id=observation_id,
                device=_device_identity(self._device),
                application=_application_identity(self._application),
                foreground_application_id=foreground_application_id,
                application_state=application_state,
                platform_context=platform_context,
                keyboard_visible=keyboard_visible,
                orientation=orientation,
                screenshot_to_driver_scale=CoordinateScale(
                    x=calibration.scale_x,
                    y=calibration.scale_y,
                ),
                screen_fingerprint=fingerprint,
                screenshot=screenshot,
                system_regions=system_regions,
                captured_at=captured_at,
                predecessor_observation_id=predecessor_id,
            )
            phase_started = self._monotonic()
            self._observation_store.persist(observation)
            self._trace_phase("observation_persist", phase_started)
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

    def _trace_phase(self, phase: str, started: float) -> None:
        narrator = current_narrator()
        if narrator is None:
            return
        duration_ms = elapsed_milliseconds(started, self._monotonic())
        narrator.trace(
            "device-observation",
            f"phase={phase} duration_ms={duration_ms:.1f}",
        )

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


def _secret_field_edges(
    bounds: tuple[int, int, int, int],
    calibration: ScreenCalibration,
    dimensions: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    """Convert one driver rectangle to screenshot pixels, or report that none remain.

    Clamping a field that sits outside the current screenshot collapses it, and a collapsed
    rectangle is not a region. A field off the viewport has nothing on screen to cover.
    """
    width, height = dimensions
    left, top, right, bottom = bounds
    edges = (
        max(0, min(width, floor(left / calibration.scale_x))),
        max(0, min(height, floor(top / calibration.scale_y))),
        max(0, min(width, ceil(right / calibration.scale_x))),
        max(0, min(height, ceil(bottom / calibration.scale_y))),
    )
    if edges[2] <= edges[0] or edges[3] <= edges[1]:
        return None
    return edges


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
    normalized_value = value.strip().lower()
    if normalized_value.startswith("portrait"):
        return DeviceOrientation.PORTRAIT
    if normalized_value.startswith("landscape"):
        return DeviceOrientation.LANDSCAPE
    return DeviceOrientation.UNKNOWN
