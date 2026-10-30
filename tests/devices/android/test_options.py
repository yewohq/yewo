"""Tests for Android UiAutomator2 option building."""

from pathlib import Path

import pytest

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation import DriverOptionsBuilder, DriverOptionsError
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.android import AndroidDriverOptionsBuilder


def _device(*, readiness: DeviceReadiness = DeviceReadiness.READY) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.ANDROID,
        target_id="Pixel_9",
        connection_id="emulator-5554",
        display_name="Pixel 9",
        kind=DeviceKind.VIRTUAL,
        readiness=readiness,
        native_state="device",
    )


def _application(*, artifact_path: Path | None = Path("build/app.apk")) -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=DevicePlatform.ANDROID,
        application_id="com.example.app",
        launch_target="com.example.app.MainActivity",
        artifact_path=artifact_path,
    )


def _request(install_mode: ApplicationInstallMode) -> ApplicationRequest:
    return ApplicationRequest(
        platform=DevicePlatform.ANDROID,
        install_mode=install_mode,
        artifact_path=None
        if install_mode is ApplicationInstallMode.REUSE
        else Path("build/app.apk"),
        application_id="com.example.app",
    )


def test_android_install_options_target_the_prepared_connection() -> None:
    builder: DriverOptionsBuilder = AndroidDriverOptionsBuilder(new_command_timeout_seconds=240)

    capabilities = builder.build(
        _device(),
        _request(ApplicationInstallMode.INSTALL),
        _application(),
    ).to_capabilities()

    assert capabilities == {
        "platformName": "Android",
        "automationName": "UIAutomator2",
        "appium:deviceName": "Pixel 9",
        "appium:udid": "emulator-5554",
        "appium:newCommandTimeout": 240,
        "appium:appPackage": "com.example.app",
        "appium:appActivity": "com.example.app.MainActivity",
        "appium:disableSuppressAccessibilityService": True,
        "appium:app": "build/app.apk",
        "appium:enforceAppInstall": True,
    }
    assert "appium:avd" not in capabilities


def test_android_install_if_needed_does_not_force_reinstallation() -> None:
    capabilities = (
        AndroidDriverOptionsBuilder()
        .build(
            _device(),
            _request(ApplicationInstallMode.INSTALL_IF_NEEDED),
            _application(),
        )
        .to_capabilities()
    )

    assert capabilities["appium:enforceAppInstall"] is False


def test_android_reuse_targets_the_observed_installed_application() -> None:
    capabilities = (
        AndroidDriverOptionsBuilder()
        .build(
            _device(),
            _request(ApplicationInstallMode.REUSE),
            _application(artifact_path=None),
        )
        .to_capabilities()
    )

    assert capabilities["appium:appPackage"] == "com.example.app"
    assert capabilities["appium:appActivity"] == "com.example.app.MainActivity"
    assert capabilities["appium:noReset"] is True
    assert "appium:app" not in capabilities
    assert "appium:enforceAppInstall" not in capabilities


def test_android_options_require_a_prepared_device() -> None:
    with pytest.raises(DriverOptionsError, match="prepared device connection"):
        AndroidDriverOptionsBuilder().build(
            _device(readiness=DeviceReadiness.TRANSITIONING),
            _request(ApplicationInstallMode.INSTALL),
            _application(),
        )


def test_android_options_require_an_observed_launch_target() -> None:
    application = _application().model_copy(update={"launch_target": None})

    with pytest.raises(DriverOptionsError, match="require a launch target"):
        AndroidDriverOptionsBuilder().build(
            _device(),
            _request(ApplicationInstallMode.INSTALL),
            application,
        )


def test_android_options_reject_a_different_inspected_artifact() -> None:
    with pytest.raises(DriverOptionsError, match="does not match"):
        AndroidDriverOptionsBuilder().build(
            _device(),
            _request(ApplicationInstallMode.INSTALL),
            _application(artifact_path=Path("build/other.apk")),
        )
