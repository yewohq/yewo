"""Tests for iOS XCUITest option building."""

from pathlib import Path

import pytest

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation import DriverOptionsError
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.ios import IosDriverOptionsBuilder


def _device(*, kind: DeviceKind = DeviceKind.VIRTUAL) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.IOS,
        target_id="IOS-UDID",
        connection_id="IOS-UDID",
        display_name="iPhone 17",
        kind=kind,
        readiness=DeviceReadiness.READY,
        native_state="Booted" if kind is DeviceKind.VIRTUAL else "connected",
    )


def _application(
    *, artifact_path: Path | None = Path("build/Example.app")
) -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=DevicePlatform.IOS,
        application_id="com.example.app",
        artifact_path=artifact_path,
    )


def _request(install_mode: ApplicationInstallMode) -> ApplicationRequest:
    return ApplicationRequest(
        platform=DevicePlatform.IOS,
        install_mode=install_mode,
        artifact_path=(
            None if install_mode is ApplicationInstallMode.REUSE else Path("build/Example.app")
        ),
        application_id="com.example.app",
    )


def test_ios_install_options_target_the_prepared_connection() -> None:
    capabilities = (
        IosDriverOptionsBuilder(new_command_timeout_seconds=240)
        .build(
            _device(),
            _request(ApplicationInstallMode.INSTALL),
            _application(),
        )
        .to_capabilities()
    )

    assert capabilities == {
        "platformName": "iOS",
        "automationName": "XCUITest",
        "appium:deviceName": "iPhone 17",
        "appium:udid": "IOS-UDID",
        "appium:newCommandTimeout": 240,
        "appium:useNewWDA": True,
        "appium:app": "build/Example.app",
        "appium:enforceAppInstall": True,
    }


def test_ios_install_if_needed_does_not_force_reinstallation() -> None:
    capabilities = (
        IosDriverOptionsBuilder(use_new_wda=False)
        .build(
            _device(kind=DeviceKind.PHYSICAL),
            _request(ApplicationInstallMode.INSTALL_IF_NEEDED),
            _application(),
        )
        .to_capabilities()
    )

    assert capabilities["appium:enforceAppInstall"] is False
    assert capabilities["appium:useNewWDA"] is False


def test_ios_reuse_targets_the_observed_bundle_identifier() -> None:
    capabilities = (
        IosDriverOptionsBuilder()
        .build(
            _device(),
            _request(ApplicationInstallMode.REUSE),
            _application(artifact_path=None),
        )
        .to_capabilities()
    )

    assert capabilities["appium:bundleId"] == "com.example.app"
    assert capabilities["appium:noReset"] is True
    assert "appium:app" not in capabilities
    assert "appium:enforceAppInstall" not in capabilities


def test_ios_options_reject_an_android_application() -> None:
    application = _application().model_copy(update={"platform": DevicePlatform.ANDROID})

    with pytest.raises(DriverOptionsError, match="matching ios"):
        IosDriverOptionsBuilder().build(
            _device(),
            _request(ApplicationInstallMode.INSTALL),
            application,
        )


def test_ios_options_reject_a_different_requested_application_id() -> None:
    request = _request(ApplicationInstallMode.REUSE).model_copy(
        update={"application_id": "com.example.other"}
    )

    with pytest.raises(DriverOptionsError, match="application ID does not match"):
        IosDriverOptionsBuilder().build(
            _device(),
            request,
            _application(artifact_path=None),
        )
