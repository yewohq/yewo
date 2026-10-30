"""Tests for remote temporary-file identities."""

from pydantic import ValidationError
import pytest

from yewo.devices import DevicePlatform
from yewo.devices.remote_files import (
    RemoteTemporaryFileReference,
    remote_temporary_path,
)
from yewo.resources import ResourceOwnership


def test_platform_paths_are_scoped_to_one_yewo_namespace() -> None:
    assert (
        remote_temporary_path(
            DevicePlatform.ANDROID,
            "com.example.app",
            "namespace_1",
            "photo.png",
        )
        == "/sdcard/Download/.yewo_namespace_1_photo.png"
    )
    assert (
        remote_temporary_path(
            DevicePlatform.IOS,
            "com.example.app",
            "namespace_1",
            "photo.png",
        )
        == "@com.example.app:data/tmp/yewo_namespace_1_photo.png"
    )


def test_reference_requires_the_derived_run_owned_path() -> None:
    with pytest.raises(ValidationError, match="must match its run namespace"):
        RemoteTemporaryFileReference(
            file_id="file_1",
            namespace_id="namespace_1",
            platform=DevicePlatform.ANDROID,
            target_id="emulator-5554",
            application_id="com.example.app",
            name="photo.png",
            remote_path="/sdcard/photo.png",
            ownership=ResourceOwnership.RUN,
        )


@pytest.mark.parametrize("name", ("../photo.png", "folder/photo.png", ".hidden", ""))
def test_reference_rejects_unsafe_remote_names(name: str) -> None:
    with pytest.raises(ValidationError):
        remote_temporary_path(
            DevicePlatform.ANDROID,
            "com.example.app",
            "namespace_1",
            name,
        )

    with pytest.raises(ValidationError):
        RemoteTemporaryFileReference(
            file_id="file_1",
            namespace_id="namespace_1",
            platform=DevicePlatform.ANDROID,
            target_id="emulator-5554",
            application_id="com.example.app",
            name=name,
            remote_path=f"/sdcard/Download/.yewo_namespace_1_{name}",
        )
