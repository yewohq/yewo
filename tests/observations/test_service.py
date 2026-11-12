"""Tests for coherent observation capture."""

from collections.abc import Iterator
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from yewo.applications import ApplicationDescriptor
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.inventory import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.interaction import DeviceInteraction
from yewo.devices.screen import (
    ScreenCalibration,
    ScreenCoordinateSpace,
    ScreenRegionReading,
    SystemRegionKind,
)
from yewo.observations import (
    DeviceOrientation,
    ObservationCaptureError,
    ObservationService,
    ObservationStore,
    ScreenshotStore,
)
from yewo.observations.service import ScreenCapture, ObservationSource
from yewo.run_narration import RunNarrator, narration_scope


CAPTURE_TIME = datetime(2026, 10, 30, 12, 0, tzinfo=timezone.utc)


def _png(
    color: tuple[int, int, int] = (10, 20, 30),
    size: tuple[int, int] = (200, 400),
) -> bytes:
    image = Image.new("RGB", size, color=color)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class _SourceStub:
    def __init__(self) -> None:
        self.foreground_ids: list[str | None] = ["com.example.app"]
        self.foreground_read_count = 0
        self.dynamic_metadata_calls: list[str] = []
        self.screenshot_calls = 0
        self.comparable_calls = 0
        self.png = _png()
        self.calibration = ScreenCalibration(200, 400, 100, 200)
        self.regions: tuple[ScreenRegionReading, ...] | None = ()

    def screenshot(self) -> ScreenCapture:
        self.screenshot_calls += 1
        return ScreenCapture(image=self.png, comparable=self.png)

    def comparable_screenshot(self) -> bytes:
        self.comparable_calls += 1
        return self.png

    def screen_calibration(self) -> ScreenCalibration:
        return self.calibration

    def active_application_id(self) -> str | None:
        self.foreground_read_count += 1
        self.dynamic_metadata_calls.append("foreground")
        if len(self.foreground_ids) > 1:
            return self.foreground_ids.pop(0)
        return self.foreground_ids[0]

    def application_state(self) -> ApplicationState:
        self.dynamic_metadata_calls.append("application_state")
        return ApplicationState.FOREGROUND

    def application_context(self) -> str | None:
        self.dynamic_metadata_calls.append("application_context")
        return "com.example.MainActivity"

    def keyboard_visible(self) -> bool | None:
        self.dynamic_metadata_calls.append("keyboard")
        return False

    def device_orientation(self) -> str | None:
        self.dynamic_metadata_calls.append("orientation")
        return "PORTRAIT"

    def system_regions(self) -> tuple[ScreenRegionReading, ...] | None:
        return self.regions


def _device(platform: DevicePlatform = DevicePlatform.ANDROID) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=platform,
        target_id="pixel_8",
        connection_id="emulator-5554",
        display_name="Pixel 8",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="device",
        operating_system_version="16",
        device_model="Pixel 8",
        architecture="arm64-v8a",
    )


def _application(
    platform: DevicePlatform = DevicePlatform.ANDROID,
) -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=platform,
        application_id="com.example.app",
        version="2.1.0",
        build="42",
        artifact_sha256="a" * 64,
    )


def _identifier_factory(values: Iterator[str]):
    return lambda: next(values)


def _service(
    source: _SourceStub,
    root: Path,
    *,
    retain_raw: bool = True,
) -> ObservationService:
    screenshot_ids = iter(("shot_1",))
    return ObservationService(
        source,
        _device(),
        _application(),
        ScreenshotStore(root, identifier_factory=_identifier_factory(screenshot_ids)),
        ObservationStore(root),
        identifier_factory=lambda: "obs_1",
        time_factory=lambda: CAPTURE_TIME,
    )


def test_service_captures_and_persists_coherent_evidence(tmp_path: Path) -> None:
    source = _SourceStub()
    source.regions = (
        ScreenRegionReading(
            kind=SystemRegionKind.STATUS_BAR,
            coordinate_space=ScreenCoordinateSpace.DRIVER,
            left=0,
            top=0,
            right=100,
            bottom=20,
        ),
    )

    observation = _service(source, tmp_path).capture(
        predecessor_observation_id="obs_0",
    )

    assert observation.observation_id == "obs_1"
    assert observation.predecessor_observation_id == "obs_0"
    assert observation.device.operating_system_version == "16"
    assert observation.application.build == "42"
    assert observation.foreground_application_id == "com.example.app"
    assert source.foreground_read_count == 1
    assert observation.orientation is DeviceOrientation.PORTRAIT
    assert observation.screenshot_to_driver_scale.x == 0.5
    assert observation.screenshot_to_driver_scale.y == 0.5
    assert observation.screen_fingerprint.system_region_mask_sha256 is not None
    assert observation.screen_fingerprint.source_artifact_id == observation.screenshot.artifact_id
    assert observation.system_regions is not None
    assert observation.system_regions[0].right == 200
    assert observation.system_regions[0].bottom == 40
    assert (tmp_path / observation.screenshot.path).is_file()
    assert (tmp_path / observation.screenshot.path).is_file()
    observation_path = tmp_path / "observations.jsonl"
    assert observation_path.is_file()
    assert (
        observation.model_validate_json(observation_path.read_text().splitlines()[0]) == observation
    )


def test_service_traces_active_capture_duration(tmp_path: Path) -> None:
    source = _SourceStub()
    moments = iter(
        (
            10.0,
            10.0,
            10.010,
            10.010,
            10.020,
            10.020,
            10.030,
            10.030,
            10.040,
            10.040,
            10.050,
            10.125,
        )
    )
    service = ObservationService(
        source,
        _device(),
        _application(),
        ScreenshotStore(
            tmp_path,
            identifier_factory=_identifier_factory(iter(("shot_1",))),
        ),
        ObservationStore(tmp_path),
        identifier_factory=lambda: "obs_1",
        time_factory=lambda: CAPTURE_TIME,
        monotonic=lambda: next(moments),
    )

    with narration_scope(RunNarrator(tmp_path)):
        service.capture(predecessor_observation_id=None)

    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "device-observation: capture=completed duration_ms=125.0" in trace
    assert "device-observation: phase=source_screenshot duration_ms=10.0" in trace
    assert "device-observation: phase=metadata duration_ms=10.0" in trace
    assert "device-observation: phase=screenshot_persist duration_ms=10.0" in trace
    assert "device-observation: phase=fingerprint duration_ms=10.0" in trace
    assert "device-observation: phase=observation_persist duration_ms=10.0" in trace


def test_service_rejects_calibration_mismatch_without_writing(tmp_path: Path) -> None:
    source = _SourceStub()
    source.calibration = ScreenCalibration(100, 200, 100, 200)

    with pytest.raises(ObservationCaptureError, match="coordinate calibration"):
        _service(source, tmp_path).capture()

    assert not (tmp_path / "screenshots").exists()


def test_service_clamps_a_platform_region_to_the_screenshot(
    tmp_path: Path,
) -> None:
    source = _SourceStub()
    source.regions = (
        ScreenRegionReading(
            SystemRegionKind.STATUS_BAR,
            ScreenCoordinateSpace.DRIVER,
            0,
            0,
            101,
            20,
        ),
    )

    observation = _service(source, tmp_path).capture()

    assert observation.system_regions is not None
    assert observation.system_regions[0].right == 200


def test_service_clamps_float_rounding_at_the_screenshot_edge(
    tmp_path: Path,
) -> None:
    source = _SourceStub()
    source.png = _png(size=(1452, 2800))
    source.calibration = ScreenCalibration(1452, 2800, 415, 800)
    source.regions = (
        ScreenRegionReading(
            SystemRegionKind.STATUS_BAR,
            ScreenCoordinateSpace.DRIVER,
            0,
            0,
            415,
            20,
        ),
    )

    observation = _service(source, tmp_path).capture()

    assert observation.system_regions is not None
    assert observation.system_regions[0].right == 1452


def test_service_wraps_invalid_clamped_region_as_capture_error(
    tmp_path: Path,
) -> None:
    source = _SourceStub()
    source.regions = (
        ScreenRegionReading(
            SystemRegionKind.STATUS_BAR,
            ScreenCoordinateSpace.SCREENSHOT,
            201,
            0,
            202,
            20,
        ),
    )

    with pytest.raises(ObservationCaptureError, match="system-region geometry"):
        _service(source, tmp_path).capture()

    assert not (tmp_path / "screenshots").exists()


def test_failed_record_write_removes_screenshots_from_that_capture(
    tmp_path: Path,
) -> None:
    observation_path = tmp_path / "observations.jsonl"
    observation_path.parent.mkdir(parents=True, exist_ok=True)
    observation_path.write_text('{"observation_id":"obs_1"}\n')

    with pytest.raises(ObservationCaptureError, match="already exists"):
        _service(_SourceStub(), tmp_path).capture()

    screenshot_directory = tmp_path / "screenshots"
    assert tuple(screenshot_directory.iterdir()) == ()
    assert observation_path.read_text() == '{"observation_id":"obs_1"}\n'


def test_service_rejects_platform_mismatch_at_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same platform"):
        ObservationService(
            _SourceStub(),
            _device(DevicePlatform.ANDROID),
            _application(DevicePlatform.IOS),
            ScreenshotStore(tmp_path),
            ObservationStore(tmp_path),
        )


def test_service_rejects_stores_with_different_roots(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="same artifact root"):
        ObservationService(
            _SourceStub(),
            _device(),
            _application(),
            ScreenshotStore(tmp_path / "screenshots"),
            ObservationStore(tmp_path / "observations"),
        )


def _bind_device_interaction(
    interaction: DeviceInteraction,
) -> ObservationSource:
    return interaction


def test_a_settlement_probe_never_asks_for_an_image(tmp_path: Path) -> None:
    """A source that alters the image charges for the alteration on every call.

    This caller hashes the screen and drops it, so asking for an image it will not look at
    pays a masking cost per poll. One run spent 58 of 215 maskings here.
    """
    source = _SourceStub()
    service = _service(source, tmp_path)

    probe = service.capture_settlement_probe()

    assert probe.fingerprint.source_artifact_id is None
    assert source.comparable_calls == 1
    assert source.screenshot_calls == 0
    assert source.dynamic_metadata_calls == []


def test_settlement_probes_defer_dynamic_metadata_to_the_full_observation(
    tmp_path: Path,
) -> None:
    source = _SourceStub()
    service = _service(source, tmp_path)

    service.capture_settlement_probe()
    service.capture_settlement_probe()
    service.capture()

    assert source.dynamic_metadata_calls == [
        "foreground",
        "application_state",
        "application_context",
        "keyboard",
        "orientation",
    ]


def test_an_observation_takes_both_views_from_one_capture(tmp_path: Path) -> None:
    """Two captures would be two moments, and the stored image would not be the one hashed."""
    source = _SourceStub()
    service = _service(source, tmp_path)

    service.capture(predecessor_observation_id=None)

    assert source.screenshot_calls == 1
    assert source.comparable_calls == 0
