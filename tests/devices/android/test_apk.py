"""Tests for Android APK and launcher inspection."""

from hashlib import sha256
from pathlib import Path

import pytest

from devices._runner import CommandRunnerStub
from yewo.applications import ApplicationInspectionError
from yewo.devices.android import AdbClient, ApkInspector
from yewo.devices.android.manifest import enabled_launcher
from yewo.devices.errors import DeviceCommandError


ANDROID_NAMESPACE = "http://schemas.android.com/apk/res/android"


def _launcher_alias(name: str, *, enabled: bool = True) -> str:
    return "\n".join(
        (
            "          E: activity-alias",
            f'            A: {ANDROID_NAMESPACE}:name(0x01010003)="{name}"',
            f"            A: {ANDROID_NAMESPACE}:enabled(0x0101000e)={str(enabled).lower()}",
            "              E: intent-filter",
            "                E: action",
            f'                    A: {ANDROID_NAMESPACE}:name(0x01010003)="android.intent.action.MAIN"',
            "                E: category",
            f'                    A: {ANDROID_NAMESPACE}:name(0x01010003)="android.intent.category.LAUNCHER"',
        )
    )


def _android_home(tmp_path: Path) -> Path:
    build_tools = tmp_path / "build-tools" / "35.0.0"
    build_tools.mkdir(parents=True)
    (build_tools / "aapt").touch()
    (build_tools / "aapt2").touch()
    return tmp_path


def test_manifest_selects_the_only_enabled_launcher() -> None:
    xml_tree = "\n".join(
        (
            _launcher_alias("example.Disabled", enabled=False),
            _launcher_alias(".MainActivity"),
        )
    )

    assert enabled_launcher(xml_tree, "example.app") == "example.app.MainActivity"


def test_apk_inspection_selects_the_highest_numeric_build_tools_version(
    tmp_path: Path,
) -> None:
    for version in ("9.0.0", "34.0.0", "35.0.0-rc1", "35.0.0"):
        executable = tmp_path / "build-tools" / version / "aapt"
        executable.parent.mkdir(parents=True)
        executable.touch()
    runner = CommandRunnerStub()
    inspector = ApkInspector(AdbClient(runner), runner, android_home=tmp_path)

    inspector.inspect_badging(tmp_path / "app.apk")

    assert runner.calls[0][0][0] == str(tmp_path / "build-tools/35.0.0/aapt")


def test_live_device_launcher_outranks_apk_metadata(tmp_path: Path) -> None:
    runner = CommandRunnerStub(stdout="metadata\nexample.app/.CurrentLauncher\n")
    inspector = ApkInspector(
        AdbClient(runner),
        runner,
        android_home=_android_home(tmp_path),
    )

    activity = inspector.resolve_launch_activity(
        tmp_path / "app.apk",
        "example.app",
        connection_id="emulator-5554",
    )

    assert activity == "example.app.CurrentLauncher"
    assert len(runner.calls) == 1


def test_badging_is_the_final_launcher_source(tmp_path: Path) -> None:
    runner = CommandRunnerStub(
        responses=(
            ("", "", 1),
            (
                "package: name='example.app'\nlaunchable-activity: name='example.app.Badged'\n",
                "",
                0,
            ),
        )
    )
    inspector = ApkInspector(
        AdbClient(runner),
        runner,
        android_home=_android_home(tmp_path),
    )

    activity = inspector.resolve_launch_activity(tmp_path / "app.apk", "example.app")

    assert activity == "example.app.Badged"


def test_missing_launcher_fails_before_session_start(tmp_path: Path) -> None:
    runner = CommandRunnerStub(responses=(("", "", 1), ("package: name='example.app'\n", "", 0)))
    inspector = ApkInspector(
        AdbClient(runner),
        runner,
        android_home=_android_home(tmp_path),
    )

    with pytest.raises(DeviceCommandError, match="No launcher activity"):
        inspector.resolve_launch_activity(tmp_path / "app.apk", "example.app")


def test_apk_inspection_returns_observed_application_identity(tmp_path: Path) -> None:
    apk_path = tmp_path / "app.apk"
    apk_path.write_bytes(b"apk artifact")
    runner = CommandRunnerStub(
        responses=(
            (
                "package: name='example.app' versionCode='42' versionName='2.1.0'\n"
                "application-label:'Example'\n"
                "launchable-activity: name='example.app.MainActivity'\n",
                "",
                0,
            ),
            ("", "", 1),
        )
    )
    inspector = ApkInspector(
        AdbClient(runner),
        runner,
        android_home=_android_home(tmp_path),
    )

    descriptor = inspector.inspect(apk_path)

    assert descriptor.application_id == "example.app"
    assert descriptor.display_name == "Example"
    assert descriptor.version == "2.1.0"
    assert descriptor.build == "42"
    assert descriptor.launch_target == "example.app.MainActivity"
    assert descriptor.artifact_path == apk_path
    assert descriptor.artifact_sha256 == sha256(b"apk artifact").hexdigest()
    assert len(runner.calls) == 2


def test_apk_inspection_rejects_missing_package_identity(tmp_path: Path) -> None:
    apk_path = tmp_path / "app.apk"
    apk_path.write_bytes(b"apk artifact")
    runner = CommandRunnerStub(stdout="application-label:'Example'\n")
    inspector = ApkInspector(
        AdbClient(runner),
        runner,
        android_home=_android_home(tmp_path),
    )

    with pytest.raises(ApplicationInspectionError, match="no package ID"):
        inspector.inspect(apk_path)
