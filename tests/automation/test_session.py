"""Tests for composed automation-session ownership and rollback."""

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
from yewo.devices.android import AndroidDriverOptionsBuilder
from yewo.resources import ResourceOwnership


class _PresenceStub:
    def __init__(self, answer: bool | None) -> None:
        self.answer = answer

    def is_installed(self, device, application_id):
        del device, application_id
        return self.answer


class _DriverStub:
    def __init__(self, events: list[str], *, installed: bool = True) -> None:
        self.session_id = "session-123"
        self.events = events
        self.installed = installed
        self.remove_error: Exception | None = None

    def is_app_installed(self, application_id: str) -> bool:
        del application_id
        return self.installed

    def remove_app(self, application_id: str) -> object:
        self.events.append(f"remove:{application_id}")
        if self.remove_error is not None:
            raise self.remove_error
        return self

    def quit(self) -> None:
        self.events.append("driver")


def _device(events: list[str]) -> DeviceLease:
    descriptor = DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9",
        connection_id="emulator-5554",
        display_name="Pixel 9",
        kind=DeviceKind.VIRTUAL,
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


def _manager(driver: _DriverStub, installed_before: bool | None) -> AutomationSessionManager:
    return AutomationSessionManager(
        DriverSessionManager(lambda server_url, options: driver),
        {DevicePlatform.ANDROID: AndroidDriverOptionsBuilder()},
        {DevicePlatform.ANDROID: _PresenceStub(installed_before)},
    )


def test_session_removes_only_an_application_absent_before_the_run() -> None:
    events: list[str] = []
    driver = _DriverStub(events)

    session = _manager(driver, installed_before=False).open(
        _device(events),
        _server(events),
        _request(),
        _application(),
    )
    session.close()
    session.close()

    assert session.installation_ownership is ResourceOwnership.RUN
    assert session.installation_change is ApplicationInstallationChange.INSTALLED
    assert events == ["remove:com.example.app", "driver", "server", "device"]


def test_session_preserves_an_application_that_was_already_installed() -> None:
    events: list[str] = []

    session = _manager(_DriverStub(events), installed_before=True).open(
        _device(events),
        _server(events),
        _request(),
        _application(),
    )
    session.close()

    assert session.installation_ownership is ResourceOwnership.EXTERNAL
    assert session.installation_change is ApplicationInstallationChange.REINSTALLED
    assert events == ["driver", "server", "device"]


def test_session_honors_explicit_installation_retention() -> None:
    events: list[str] = []

    session = _manager(_DriverStub(events), installed_before=False).open(
        _device(events),
        _server(events),
        _request(retain_installation=True),
        _application(),
    )
    session.close()

    assert events == ["driver", "server", "device"]


def test_install_if_needed_records_no_change_for_an_existing_application() -> None:
    events: list[str] = []

    session = _manager(_DriverStub(events), installed_before=True).open(
        _device(events),
        _server(events),
        _request(ApplicationInstallMode.INSTALL_IF_NEEDED),
        _application(),
    )

    assert session.installation_change is ApplicationInstallationChange.NONE
    session.close()


def test_install_records_unverified_change_when_prior_presence_is_unknown() -> None:
    events: list[str] = []

    session = _manager(_DriverStub(events), installed_before=None).open(
        _device(events),
        _server(events),
        _request(),
        _application(),
    )

    assert session.installation_change is ApplicationInstallationChange.UNVERIFIED
    session.close()


def test_strict_reuse_requires_positive_presence_evidence() -> None:
    events: list[str] = []

    with pytest.raises(AutomationSessionError, match="positive evidence"):
        _manager(_DriverStub(events), installed_before=None).open(
            _device(events),
            _server(events),
            _request(ApplicationInstallMode.REUSE),
            _application().model_copy(update={"artifact_path": None}),
        )

    assert events == ["server", "device"]


def test_missing_application_after_driver_open_rolls_back_every_lease() -> None:
    events: list[str] = []

    with pytest.raises(AutomationSessionError, match="not installed after"):
        _manager(_DriverStub(events, installed=False), installed_before=False).open(
            _device(events),
            _server(events),
            _request(),
            _application(),
        )

    assert events == ["driver", "server", "device"]


def test_cleanup_continues_after_application_removal_failure() -> None:
    events: list[str] = []
    driver = _DriverStub(events)
    driver.remove_error = RuntimeError("remove failed")
    session = _manager(driver, installed_before=False).open(
        _device(events),
        _server(events),
        _request(),
        _application(),
    )

    with pytest.raises(AutomationSessionError, match="1 stage"):
        session.close()

    assert session.closed
    assert events == ["remove:com.example.app", "driver", "server", "device"]
