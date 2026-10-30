"""Tests for iOS application artifact inspection."""

from pathlib import Path
import plistlib
import zipfile

import pytest

from yewo.applications import (
    ApplicationArtifactInspector,
    ApplicationInspectionError,
)
from yewo.devices import DevicePlatform
from yewo.devices.ios import IosApplicationInspector
from yewo.devices.ios import DevicectlClient, IosApplicationPresence, SimctlClient
from devices.ios.test_devicectl_client import DevicectlRunnerStub
from devices._runner import CommandRunnerStub
from yewo.devices import DeviceDescriptor, DeviceKind, DeviceReadiness


IOS_METADATA = {
    "CFBundleIdentifier": "com.example.app",
    "CFBundleDisplayName": "Example",
    "CFBundleShortVersionString": "2.1.0",
    "CFBundleVersion": "42",
}


def _app_bundle(tmp_path: Path, metadata: dict[str, str] | None = None) -> Path:
    bundle = tmp_path / "Example.app"
    bundle.mkdir()
    (bundle / "Info.plist").write_bytes(plistlib.dumps(metadata or IOS_METADATA))
    return bundle


def test_app_bundle_inspection_returns_observed_identity(tmp_path: Path) -> None:
    bundle = _app_bundle(tmp_path)
    inspector: ApplicationArtifactInspector = IosApplicationInspector()

    descriptor = inspector.inspect(bundle)

    assert descriptor.platform is DevicePlatform.IOS
    assert descriptor.application_id == "com.example.app"
    assert descriptor.display_name == "Example"
    assert descriptor.version == "2.1.0"
    assert descriptor.build == "42"
    assert descriptor.launch_target is None
    assert descriptor.artifact_path == bundle
    assert descriptor.artifact_sha256 is not None


def test_ipa_inspection_reads_only_the_top_level_application(tmp_path: Path) -> None:
    ipa_path = tmp_path / "Example.ipa"
    with zipfile.ZipFile(ipa_path, "w") as archive:
        archive.writestr("Payload/Example.app/Info.plist", plistlib.dumps(IOS_METADATA))
        archive.writestr(
            "Payload/Example.app/PlugIns/Share.appex/Info.plist",
            plistlib.dumps({"CFBundleIdentifier": "com.example.app.share"}),
        )

    descriptor = IosApplicationInspector().inspect(ipa_path)

    assert descriptor.application_id == "com.example.app"
    assert descriptor.artifact_path == ipa_path


def test_ipa_rejects_multiple_top_level_applications(tmp_path: Path) -> None:
    ipa_path = tmp_path / "Multiple.ipa"
    with zipfile.ZipFile(ipa_path, "w") as archive:
        archive.writestr("Payload/First.app/Info.plist", plistlib.dumps(IOS_METADATA))
        archive.writestr("Payload/Second.app/Info.plist", plistlib.dumps(IOS_METADATA))

    with pytest.raises(ApplicationInspectionError, match="exactly one"):
        IosApplicationInspector().inspect(ipa_path)


def test_ios_inspection_rejects_missing_bundle_identifier(tmp_path: Path) -> None:
    bundle = _app_bundle(tmp_path, {"CFBundleName": "Example"})

    with pytest.raises(ApplicationInspectionError, match="no bundle identifier"):
        IosApplicationInspector().inspect(bundle)


def test_ios_inspection_rejects_unsupported_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "Example.zip"
    artifact.write_bytes(b"not an application")

    with pytest.raises(ApplicationInspectionError, match="app bundle or IPA"):
        IosApplicationInspector().inspect(artifact)


def test_ios_inspection_rejects_oversized_metadata(tmp_path: Path) -> None:
    bundle = tmp_path / "Oversized.app"
    bundle.mkdir()
    (bundle / "Info.plist").write_bytes(b"x" * (4 * 1024 * 1024 + 1))

    with pytest.raises(ApplicationInspectionError, match="metadata is too large"):
        IosApplicationInspector().inspect(bundle)


def _ios_device(*, kind: DeviceKind = DeviceKind.VIRTUAL) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=DevicePlatform.IOS,
        target_id="IOS-UDID",
        connection_id="IOS-UDID",
        display_name="iPhone 17",
        kind=kind,
        readiness=DeviceReadiness.READY,
        native_state="Booted" if kind is DeviceKind.VIRTUAL else "connected",
    )


def test_ios_simulator_presence_reads_listapps_json() -> None:
    runner = CommandRunnerStub(
        stdout='{"com.example.app":{"CFBundleIdentifier":"com.example.app"}}'
    )

    installed = IosApplicationPresence(SimctlClient(runner)).is_installed(
        _ios_device(),
        "com.example.app",
    )

    assert installed is True
    assert runner.calls == [(("xcrun", "simctl", "listapps", "IOS-UDID", "--json"), 20)]


def test_ios_physical_presence_is_unknown_without_supported_host_evidence() -> None:
    runner = CommandRunnerStub()

    installed = IosApplicationPresence(SimctlClient(runner)).is_installed(
        _ios_device(kind=DeviceKind.PHYSICAL),
        "com.example.app",
    )

    assert installed is None
    assert runner.calls == []


def test_ios_physical_presence_uses_devicectl_evidence() -> None:
    simctl_runner = CommandRunnerStub()
    devicectl_runner = DevicectlRunnerStub(
        {"result": {"apps": [{"bundleIdentifier": "com.example.app"}]}}
    )

    installed = IosApplicationPresence(
        SimctlClient(simctl_runner),
        DevicectlClient(devicectl_runner),
    ).is_installed(
        _ios_device(kind=DeviceKind.PHYSICAL),
        "com.example.app",
    )

    assert installed is True
    assert simctl_runner.calls == []
