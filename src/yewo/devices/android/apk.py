"""Inspect Android packages and resolve their launcher activity."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re

from yewo.applications.errors import ApplicationInspectionError
from yewo.applications.inspection import artifact_sha256
from yewo.applications.records import ApplicationDescriptor
from yewo.devices._commands import (
    CommandRunner,
    capture_optional,
    execute_command,
    run_command,
)
from yewo.devices.android.adb import AdbClient
from yewo.devices.android.manifest import LAUNCHER_CATEGORY, enabled_launcher
from yewo.devices.errors import DeviceCommandError
from yewo.host import resolve_build_tools_executables
from yewo.devices.inventory import DevicePlatform


@dataclass(frozen=True)
class ApkInfo:
    """Package and launcher values reported by AAPT badging."""

    package_id: str | None
    launch_activity: str | None
    display_name: str | None
    version: str | None
    build: str | None


def _build_tools_version(executable: Path) -> tuple[int, tuple[int, ...], bool, str]:
    version = executable.parent.name
    version_match = re.fullmatch(r"(\d+(?:\.\d+)*)(.*)", version)
    if version_match is None:
        return 0, (), False, version
    numbers = tuple(int(part) for part in version_match.group(1).split("."))
    is_stable = not version_match.group(2)
    return 1, numbers, is_stable, version


class ApkInspector:
    """Read APK metadata with Android SDK build tools and a live device."""

    def __init__(
        self,
        adb: AdbClient,
        runner: CommandRunner = run_command,
        *,
        android_home: Path | None = None,
    ) -> None:
        self._adb = adb
        self._runner = runner
        configured_home = android_home or (
            Path(value) if (value := os.getenv("ANDROID_HOME")) else None
        )
        self._android_home = configured_home

    def _build_tool(self, executable: str) -> str:
        if self._android_home is None:
            raise DeviceCommandError("ANDROID_HOME is not set.")
        candidates = sorted(
            resolve_build_tools_executables(self._android_home, executable),
            key=_build_tools_version,
            reverse=True,
        )
        if not candidates:
            raise DeviceCommandError(
                f"No {executable} executable exists under Android build-tools."
            )
        return str(candidates[0])

    def inspect_badging(self, apk_path: Path) -> ApkInfo:
        """Return the package and launch activity from AAPT badging."""
        output = (
            execute_command(
                self._runner,
                [self._build_tool("aapt"), "dump", "badging", str(apk_path)],
                purpose="inspect APK badging",
            ).stdout
            or ""
        )
        package_line = next(
            (line for line in output.splitlines() if line.startswith("package: ")),
            "",
        )
        package_match = re.search(r"name='([^']+)'", package_line)
        activity_match = re.search(r"launchable-activity: name='([^']+)'", output)
        display_name_match = re.search(r"^application-label:'([^']+)'", output, re.MULTILINE)
        return ApkInfo(
            package_id=package_match.group(1) if package_match else None,
            launch_activity=activity_match.group(1) if activity_match else None,
            display_name=display_name_match.group(1) if display_name_match else None,
            version=self._badging_attribute(package_line, "versionName"),
            build=self._badging_attribute(package_line, "versionCode"),
        )

    @staticmethod
    def _badging_attribute(package_line: str, name: str) -> str | None:
        value_match = re.search(rf"{name}='([^']+)'", package_line)
        return value_match.group(1) if value_match else None

    def inspect(self, artifact_path: Path) -> ApplicationDescriptor:
        """Return application identity observed from one APK."""
        digest = artifact_sha256(artifact_path)
        badging = self.inspect_badging(artifact_path)
        package_id = badging.package_id
        if package_id is None:
            raise ApplicationInspectionError(f"APK metadata has no package ID: {artifact_path}")
        launch_activity = self._resolve_launch_activity(
            artifact_path,
            package_id,
            badging=badging,
        )
        return ApplicationDescriptor(
            platform=DevicePlatform.ANDROID,
            application_id=package_id,
            display_name=badging.display_name,
            version=badging.version,
            build=badging.build,
            launch_target=launch_activity,
            artifact_path=artifact_path,
            artifact_sha256=digest,
        )

    def manifest_tree(self, apk_path: Path) -> str | None:
        """Return the AAPT2 manifest tree when it can be read."""
        try:
            aapt2 = self._build_tool("aapt2")
        except DeviceCommandError:
            return None
        return capture_optional(
            self._runner,
            [
                aapt2,
                "dump",
                "xmltree",
                "--file",
                "AndroidManifest.xml",
                str(apk_path),
            ],
        )

    def launcher_activity_on_device(self, connection_id: str, package_id: str) -> str | None:
        """Return the component that the installed package resolves."""
        output = self._adb.ask(
            connection_id,
            "cmd",
            "package",
            "resolve-activity",
            "--brief",
            "-c",
            LAUNCHER_CATEGORY,
            package_id,
        )
        lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
        if not lines or "/" not in lines[-1]:
            return None
        resolved_package, _, activity = lines[-1].partition("/")
        if resolved_package != package_id or not activity:
            return None
        return f"{package_id}{activity}" if activity.startswith(".") else activity

    def resolve_launch_activity(
        self,
        apk_path: Path,
        package_id: str,
        *,
        connection_id: str | None = None,
    ) -> str:
        """Resolve a launcher from the device, manifest, or AAPT badging."""
        return self._resolve_launch_activity(
            apk_path,
            package_id,
            connection_id=connection_id,
        )

    def _resolve_launch_activity(
        self,
        apk_path: Path,
        package_id: str,
        *,
        connection_id: str | None = None,
        badging: ApkInfo | None = None,
    ) -> str:
        if connection_id:
            on_device = self.launcher_activity_on_device(connection_id, package_id)
            if on_device:
                return on_device
        xml_tree = self.manifest_tree(apk_path)
        if xml_tree and (declared := enabled_launcher(xml_tree, package_id)):
            return declared
        inspected = badging or self.inspect_badging(apk_path)
        if inspected.launch_activity:
            return inspected.launch_activity
        raise DeviceCommandError(f"No launcher activity was found for {package_id} in {apk_path}.")
