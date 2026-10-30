"""Tests for application-request resolution."""

from pathlib import Path

import pytest

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInspectionError,
    ApplicationInstallMode,
    ApplicationRequest,
    RequestedApplicationResolver,
)
from yewo.devices import DevicePlatform


class _InspectorStub:
    def __init__(self, descriptor: ApplicationDescriptor) -> None:
        self.descriptor = descriptor
        self.paths: list[Path] = []

    def inspect(self, artifact_path: Path) -> ApplicationDescriptor:
        self.paths.append(artifact_path)
        return self.descriptor


def test_reuse_resolves_the_explicit_installed_identity() -> None:
    request = ApplicationRequest(
        platform=DevicePlatform.ANDROID,
        install_mode=ApplicationInstallMode.REUSE,
        application_id="com.example.app",
    )

    descriptor = RequestedApplicationResolver({}).resolve(request)

    assert descriptor.platform is DevicePlatform.ANDROID
    assert descriptor.application_id == "com.example.app"
    assert descriptor.artifact_path is None


def test_artifact_resolution_uses_only_the_matching_platform_inspector() -> None:
    artifact_path = Path("application.apk")
    inspector = _InspectorStub(
        ApplicationDescriptor(
            platform=DevicePlatform.ANDROID,
            application_id="com.example.app",
            artifact_path=artifact_path,
        )
    )
    request = ApplicationRequest(
        platform=DevicePlatform.ANDROID,
        artifact_path=artifact_path,
    )

    descriptor = RequestedApplicationResolver({DevicePlatform.ANDROID: inspector}).resolve(request)

    assert descriptor.application_id == "com.example.app"
    assert inspector.paths == [artifact_path]


def test_artifact_resolution_rejects_an_identity_mismatch() -> None:
    artifact_path = Path("application.ipa")
    inspector = _InspectorStub(
        ApplicationDescriptor(
            platform=DevicePlatform.IOS,
            application_id="com.other.app",
            artifact_path=artifact_path,
        )
    )
    request = ApplicationRequest(
        platform=DevicePlatform.IOS,
        artifact_path=artifact_path,
        application_id="com.example.app",
    )

    with pytest.raises(ApplicationInspectionError, match="does not match"):
        RequestedApplicationResolver({DevicePlatform.IOS: inspector}).resolve(request)
