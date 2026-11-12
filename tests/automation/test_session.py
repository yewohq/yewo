"""Tests for composed automation-session ownership and rollback."""

import base64
from collections import deque
import json
from pathlib import Path

import pytest

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallationChange,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation import (
    AppiumServerLease,
    AutomationSession,
    AutomationSessionError,
    AutomationSessionManager,
    DriverSessionManager,
)
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DeviceLease,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.android import ANDROID_LOG_SOURCES, AndroidDriverOptionsBuilder
from yewo.devices.logs import (
    DeviceLogCaptureError,
    DeviceLogCheckpointKind,
    DeviceLogRollbackError,
)
from yewo.devices.recording import ScreenRecordingVideoMetadata
from yewo.devices.remote_files import RemoteTemporaryFileCleanupError
from yewo.resources import ResourceOwnership

from automation.session_stores import (
    FailFinalLogRollbackStore,
    appium_log_reader,
)


class _PresenceStub:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer

    def is_installed(self, device, application_id):
        del device, application_id
        return self.answer


class _DriverStub:
    def __init__(
        self,
        events: list[str],
        *,
        is_installed: bool = True,
        log_responses: tuple[tuple[object, ...], ...] | None = None,
    ) -> None:
        self.session_id = "session-123"
        self.events = events
        self.is_installed = is_installed
        self.remove_error: Exception | None = None
        self.script_error: Exception | None = None
        self.clipboard_error: Exception | None = None
        self.remote_file_content: dict[str, bytes] = {}
        self.log_types = ["logcat"]
        self._log_responses = deque(log_responses or (((),) * 10))

    def is_app_installed(self, application_id: str) -> bool:
        del application_id
        return self.is_installed

    def remove_app(self, application_id: str) -> object:
        self.events.append(f"remove:{application_id}")
        if self.remove_error is not None:
            raise self.remove_error
        return self

    def quit(self) -> None:
        self.events.append("driver")

    def set_clipboard_text(self, text: str, label: str | None = None) -> None:
        del label
        if self.clipboard_error is not None:
            raise self.clipboard_error
        self.events.append(f"clipboard:{text}")

    def get_log(self, log_type: str) -> tuple[object, ...]:
        self.events.append(f"log:{log_type}")
        return self._log_responses.popleft()

    def execute_script(self, script: str, arguments: dict[str, str]) -> object:
        remote_path = arguments["remotePath"]
        if self.script_error is not None:
            raise self.script_error
        if script == "mobile: pushFile":
            self.events.append(f"push:{remote_path}")
            self.remote_file_content[remote_path] = base64.b64decode(arguments["payload"])
            return None
        if script == "mobile: pullFile":
            self.events.append(f"pull:{remote_path}")
            return base64.b64encode(self.remote_file_content[remote_path]).decode("ascii")
        if script == "mobile: deleteFile":
            self.events.append(f"delete:{remote_path}")
            self.remote_file_content.pop(remote_path, None)
            return True
        raise AssertionError(f"Unexpected script: {script}")


class _RecordingDriverStub(_DriverStub):
    def start_recording_screen(self, **options: object) -> None:
        del options
        self.events.append("recording:start")

    def stop_recording_screen(self) -> str:
        self.events.append("recording:stop")
        video = b"\x00\x00\x00\x18ftypmp42video"
        return base64.b64encode(video).decode()


def _device(
    events: list[str],
    *,
    kind: DeviceKind = DeviceKind.VIRTUAL,
) -> DeviceLease:
    descriptor = DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9",
        connection_id="emulator-5554",
        display_name="Pixel 9",
        kind=kind,
        readiness=DeviceReadiness.READY,
        native_state="device",
    )
    return DeviceLease(
        descriptor,
        ResourceOwnership.RUN,
        lambda: events.append("device"),
    )


def _server(events: list[str]) -> AppiumServerLease:
    return AppiumServerLease(
        "http://127.0.0.1:4723",
        ResourceOwnership.RUN,
        release=lambda: events.append("server"),
    )


def _application() -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=DevicePlatform.ANDROID,
        application_id="com.example.app",
        launch_target="com.example.app.MainActivity",
        artifact_path=Path("build/app.apk"),
    )


def _request(
    install_mode: ApplicationInstallMode = ApplicationInstallMode.INSTALL,
    *,
    retain_installation: bool = False,
) -> ApplicationRequest:
    return ApplicationRequest(
        platform=DevicePlatform.ANDROID,
        install_mode=install_mode,
        artifact_path=(
            None if install_mode is ApplicationInstallMode.REUSE else Path("build/app.apk")
        ),
        application_id="com.example.app",
        retain_installation=retain_installation,
    )


def _manager(
    driver: _DriverStub,
    installed_before: bool | None,
) -> AutomationSessionManager:
    return AutomationSessionManager(
        DriverSessionManager(lambda server_url, options: driver),
        {DevicePlatform.ANDROID: AndroidDriverOptionsBuilder()},
        {DevicePlatform.ANDROID: _PresenceStub(installed_before)},
        {DevicePlatform.ANDROID: ANDROID_LOG_SOURCES},
        device_log_readers={DevicePlatform.ANDROID: appium_log_reader},
        recording_metadata_reader=lambda path: ScreenRecordingVideoMetadata(
            16,
            9,
            30.0,
            1.0,
        ),
    )


def _open(
    manager: AutomationSessionManager,
    device: DeviceLease,
    server: AppiumServerLease,
    request: ApplicationRequest,
    application: ApplicationDescriptor,
    artifact_root: Path,
) -> AutomationSession:
    return manager.open(
        device,
        server,
        request,
        application,
        artifact_root=artifact_root,
    )


def test_session_removes_only_an_application_absent_before_the_run(tmp_path: Path) -> None:
    events: list[str] = []
    driver = _DriverStub(events)

    session = _open(
        _manager(driver, installed_before=False),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )
    session.close()
    session.close()

    assert session.installation_ownership is ResourceOwnership.RUN
    assert session.installation_change is ApplicationInstallationChange.INSTALLED
    assert events == [
        "clipboard:",
        "log:logcat",
        "log:logcat",
        "remove:com.example.app",
        "driver",
        "server",
        "device",
    ]


def test_session_preserves_an_application_that_was_already_installed(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    session = _open(
        _manager(_DriverStub(events), installed_before=True),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )
    session.close()

    assert session.installation_ownership is ResourceOwnership.EXTERNAL
    assert session.installation_change is ApplicationInstallationChange.REINSTALLED
    assert events == [
        "clipboard:",
        "log:logcat",
        "log:logcat",
        "driver",
        "server",
        "device",
    ]


def test_session_honors_explicit_installation_retention(tmp_path: Path) -> None:
    events: list[str] = []

    session = _open(
        _manager(_DriverStub(events), installed_before=False),
        _device(events),
        _server(events),
        _request(retain_installation=True),
        _application(),
        tmp_path,
    )
    session.close()

    assert events == [
        "clipboard:",
        "log:logcat",
        "log:logcat",
        "driver",
        "server",
        "device",
    ]


def test_install_if_needed_records_no_change_for_an_existing_application(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    session = _open(
        _manager(_DriverStub(events), installed_before=True),
        _device(events),
        _server(events),
        _request(ApplicationInstallMode.INSTALL_IF_NEEDED),
        _application(),
        tmp_path,
    )

    assert session.installation_change is ApplicationInstallationChange.NONE
    session.close()


def test_install_records_unverified_change_when_prior_presence_is_unknown(
    tmp_path: Path,
) -> None:
    events: list[str] = []

    session = _open(
        _manager(_DriverStub(events), installed_before=None),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    assert session.installation_change is ApplicationInstallationChange.UNVERIFIED
    session.close()


def test_cleanup_continues_after_application_removal_failure(tmp_path: Path) -> None:
    events: list[str] = []
    driver = _DriverStub(events)
    driver.remove_error = RuntimeError("remove failed")
    session = _open(
        _manager(driver, installed_before=False),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    with pytest.raises(AutomationSessionError, match="1 stage"):
        session.close()

    assert session.closed
    assert events == [
        "clipboard:",
        "log:logcat",
        "log:logcat",
        "remove:com.example.app",
        "driver",
        "server",
        "device",
    ]


def test_session_preserves_log_rollback_error_after_other_cleanup(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    driver = _DriverStub(
        events,
        log_responses=((), ({"message": "final"},)),
    )
    manager = AutomationSessionManager(
        DriverSessionManager(lambda server_url, options: driver),
        {DevicePlatform.ANDROID: AndroidDriverOptionsBuilder()},
        {DevicePlatform.ANDROID: _PresenceStub(True)},
        {DevicePlatform.ANDROID: ANDROID_LOG_SOURCES},
        log_store_factory=FailFinalLogRollbackStore,
        device_log_readers={DevicePlatform.ANDROID: appium_log_reader},
    )
    session = _open(
        manager,
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    with pytest.raises(DeviceLogRollbackError, match="Could not roll back"):
        session.close()

    assert session.closed
    assert events[-3:] == ["driver", "server", "device"]


def test_session_persists_action_scenario_and_final_log_boundaries(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    driver = _DriverStub(
        events,
        log_responses=(
            ({"message": "stale"},),
            ({"message": "application setup"},),
            ({"message": "action"},),
            ({"message": "scenario"},),
            ({"message": "final"},),
        ),
    )
    session = _open(
        _manager(driver, installed_before=True),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    setup_receipt = session.drain_application_setup_logs("setup_1")
    action_receipt = session.drain_action_logs("action_1")
    scenario_receipt = session.drain_scenario_logs("scenario_1")
    session.close()

    assert setup_receipt.checkpoint_kind is DeviceLogCheckpointKind.APPLICATION_SETUP
    assert setup_receipt.checkpoint_id == "setup_1"
    assert action_receipt.checkpoint_kind is DeviceLogCheckpointKind.ACTION
    assert action_receipt.checkpoint_id == "action_1"
    assert scenario_receipt.checkpoint_kind is DeviceLogCheckpointKind.SCENARIO
    assert scenario_receipt.checkpoint_id == "scenario_1"
    assert session.device_logs.receipts[-1].checkpoint_kind is (DeviceLogCheckpointKind.SESSION_END)
    boundary_directory = tmp_path / "device-logs" / session.device_logs.boundary.boundary_id
    assert (boundary_directory / "boundary.json").is_file()
    assert len(tuple(boundary_directory.glob("*.json"))) == 1
    assert len(tuple(boundary_directory.glob("*.log"))) == 5
    drain_records = [
        json.loads(line) for line in (boundary_directory / "drains.jsonl").read_text().splitlines()
    ]
    assert len(drain_records) == 5
    action_record = next(
        record for record in drain_records if record["drain_id"] == action_receipt.drain_id
    )
    assert action_record["checkpoint_kind"] == "action"
    assert action_record["checkpoint_id"] == "action_1"
    persisted_log_text = "\n".join(
        log_path.read_text() for log_path in boundary_directory.glob("*.log")
    )
    assert "stale" in persisted_log_text

    with pytest.raises(DeviceLogCaptureError, match="closed"):
        session.drain_action_logs("action_2")


def test_session_checkpoints_and_closes_owned_screen_recording(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    driver = _RecordingDriverStub(events)
    session = _open(
        _manager(driver, installed_before=True),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    scenario = session.checkpoint_scenario_recording("scenario_1")
    session.close()

    assert scenario.artifact is not None
    assert len(session.recording.receipts) == 2
    assert len(tuple(tmp_path.glob("recordings/*/*.mp4"))) == 2
    assert events.count("recording:start") == 2
    assert events.count("recording:stop") == 2


def test_session_removes_owned_remote_files_before_closing_the_driver(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    driver = _DriverStub(events)
    session = _open(
        _manager(driver, installed_before=True),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )
    reference = session.reserve_remote_temporary_file("fixture.png")

    session.close()

    delete_event = f"delete:{reference.remote_path}"
    assert delete_event in events
    assert events.index(delete_event) < events.index("driver")
    assert session.remote_files.closed is True


def test_remote_file_cleanup_failure_does_not_skip_other_session_cleanup(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    driver = _DriverStub(events)
    driver.script_error = RuntimeError("delete failed")
    session = _open(
        _manager(driver, installed_before=True),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )
    session.reserve_remote_temporary_file("fixture.png")

    with pytest.raises(RemoteTemporaryFileCleanupError):
        session.close()

    assert events[-3:] == ["driver", "server", "device"]
    assert session.closed is True


def test_session_pushes_pulls_and_releases_owned_media(tmp_path: Path) -> None:
    events: list[str] = []
    driver = _DriverStub(events)
    session = _open(
        _manager(driver, installed_before=True),
        _device(events),
        _server(events),
        _request(),
        _application(),
        tmp_path,
    )

    reference = session.push_media("fixture.png", b"media")

    assert session.pull_media(reference) == b"media"
    assert session.release_media(reference) is True
    assert reference.remote_file.remote_path not in driver.remote_file_content
    session.close()
