"""Inspect identity metadata in iOS application artifacts."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import plistlib
import zipfile

from yewo.applications.errors import ApplicationInspectionError
from yewo.applications.inspection import artifact_sha256
from yewo.applications.records import ApplicationDescriptor
from yewo.devices.inventory import DeviceDescriptor, DeviceKind, DevicePlatform
from yewo.devices.ios.devicectl import DevicectlClient
from yewo.devices.ios.simctl import SimctlClient


_MAX_METADATA_BYTES = 4 * 1024 * 1024


def _text(metadata: dict[object, object], key: str) -> str | None:
    value = metadata.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _descriptor(metadata: dict[object, object], artifact_path: Path) -> ApplicationDescriptor:
    application_id = _text(metadata, "CFBundleIdentifier")
    if application_id is None:
        raise ApplicationInspectionError(
            f"iOS application metadata has no bundle identifier: {artifact_path}"
        )
    return ApplicationDescriptor(
        platform=DevicePlatform.IOS,
        application_id=application_id,
        display_name=_text(metadata, "CFBundleDisplayName") or _text(metadata, "CFBundleName"),
        version=_text(metadata, "CFBundleShortVersionString"),
        build=_text(metadata, "CFBundleVersion"),
        artifact_path=artifact_path,
        artifact_sha256=artifact_sha256(artifact_path),
    )


def _load_plist(data: bytes, artifact_path: Path) -> dict[object, object]:
    if len(data) > _MAX_METADATA_BYTES:
        raise ApplicationInspectionError(f"iOS application metadata is too large: {artifact_path}")
    try:
        metadata = plistlib.loads(data)
    except (plistlib.InvalidFileException, ValueError, TypeError) as error:
        raise ApplicationInspectionError(
            f"iOS application metadata is invalid: {artifact_path}"
        ) from error
    if not isinstance(metadata, dict):
        raise ApplicationInspectionError(
            f"iOS application metadata is not a dictionary: {artifact_path}"
        )
    return metadata


class IosApplicationInspector:
    """Read application identity from an app bundle or IPA archive."""

    def inspect(self, artifact_path: Path) -> ApplicationDescriptor:
        suffix = artifact_path.suffix.lower()
        if suffix == ".app" and artifact_path.is_dir():
            metadata = self._read_app_bundle(artifact_path)
        elif suffix == ".ipa" and artifact_path.is_file():
            metadata = self._read_ipa(artifact_path)
        else:
            raise ApplicationInspectionError(
                "An iOS application artifact must be an app bundle or IPA archive."
            )
        return _descriptor(metadata, artifact_path)

    @staticmethod
    def _read_app_bundle(artifact_path: Path) -> dict[object, object]:
        info_path = artifact_path / "Info.plist"
        try:
            if info_path.stat().st_size > _MAX_METADATA_BYTES:
                raise ApplicationInspectionError(
                    f"iOS application metadata is too large: {artifact_path}"
                )
            data = info_path.read_bytes()
        except OSError as error:
            raise ApplicationInspectionError(
                f"Could not read iOS application metadata: {info_path}"
            ) from error
        return _load_plist(data, artifact_path)

    @staticmethod
    def _read_ipa(artifact_path: Path) -> dict[object, object]:
        try:
            with zipfile.ZipFile(artifact_path) as archive:
                candidates = [
                    name
                    for name in archive.namelist()
                    if IosApplicationInspector._is_app_plist(name)
                ]
                if len(candidates) != 1:
                    raise ApplicationInspectionError(
                        "An IPA archive must contain exactly one application bundle."
                    )
                plist_info = archive.getinfo(candidates[0])
                if plist_info.file_size > _MAX_METADATA_BYTES:
                    raise ApplicationInspectionError(
                        f"iOS application metadata is too large: {artifact_path}"
                    )
                data = archive.read(plist_info)
        except (OSError, zipfile.BadZipFile, KeyError) as error:
            raise ApplicationInspectionError(
                f"Could not read IPA archive: {artifact_path}"
            ) from error
        return _load_plist(data, artifact_path)

    @staticmethod
    def _is_app_plist(name: str) -> bool:
        parts = PurePosixPath(name).parts
        return (
            len(parts) == 3
            and parts[0] == "Payload"
            and parts[1].endswith(".app")
            and parts[2] == "Info.plist"
        )


class IosApplicationPresence:
    """Read installed iOS application IDs from the matching platform tool."""

    def __init__(
        self,
        simctl: SimctlClient,
        devicectl: DevicectlClient | None = None,
    ) -> None:
        self._simctl = simctl
        self._devicectl = devicectl

    def is_installed(
        self,
        device: DeviceDescriptor,
        application_id: str,
    ) -> bool | None:
        if device.platform is not DevicePlatform.IOS or device.connection_id is None:
            return None
        if device.kind is DeviceKind.VIRTUAL:
            applications = self._simctl.installed_applications(device.connection_id)
            return None if applications is None else application_id in applications
        if device.kind is DeviceKind.PHYSICAL and self._devicectl is not None:
            return self._devicectl.is_application_installed(
                device.connection_id,
                application_id,
            )
        return None
