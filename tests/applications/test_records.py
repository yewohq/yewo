"""Tests for application request and identity records."""

from pathlib import Path

from pydantic import ValidationError
import pytest

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.devices import DevicePlatform


def test_install_request_requires_an_artifact() -> None:
    request = ApplicationRequest(
        platform=DevicePlatform.ANDROID,
        artifact_path=Path("build/app.apk"),
    )

    assert request.install_mode is ApplicationInstallMode.INSTALL
    assert request.application_id is None
    assert not request.retain_installation


@pytest.mark.parametrize(
    "install_mode",
    [ApplicationInstallMode.INSTALL, ApplicationInstallMode.INSTALL_IF_NEEDED],
)
def test_install_modes_reject_a_missing_artifact(
    install_mode: ApplicationInstallMode,
) -> None:
    with pytest.raises(ValidationError, match="requires an artifact path"):
        ApplicationRequest(
            platform=DevicePlatform.ANDROID,
            install_mode=install_mode,
        )


def test_reuse_request_requires_an_application_id() -> None:
    with pytest.raises(ValidationError, match="requires an application ID"):
        ApplicationRequest(
            platform=DevicePlatform.IOS,
            install_mode=ApplicationInstallMode.REUSE,
        )


def test_reuse_request_does_not_require_an_artifact() -> None:
    request = ApplicationRequest(
        platform=DevicePlatform.IOS,
        install_mode=ApplicationInstallMode.REUSE,
        application_id="  com.example.app  ",
        retain_installation=True,
    )

    assert request.application_id == "com.example.app"
    assert request.artifact_path is None
    assert request.retain_installation


def test_application_descriptor_round_trips_without_losing_identity() -> None:
    descriptor = ApplicationDescriptor(
        platform=DevicePlatform.ANDROID,
        application_id="com.example.app",
        display_name="Example",
        version="2.1.0",
        build="42",
        launch_target="com.example.app.MainActivity",
        artifact_path=Path("build/app.apk"),
        artifact_sha256="a" * 64,
    )

    restored = ApplicationDescriptor.model_validate_json(descriptor.model_dump_json())

    assert restored == descriptor


def test_application_descriptor_rejects_an_invalid_digest() -> None:
    with pytest.raises(ValidationError, match="artifact_sha256"):
        ApplicationDescriptor(
            platform=DevicePlatform.ANDROID,
            application_id="com.example.app",
            artifact_sha256="not-a-sha256-digest",
        )


def test_application_records_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="unknown"):
        ApplicationDescriptor.model_validate(
            {
                "platform": "android",
                "application_id": "com.example.app",
                "unknown": "value",
            }
        )
