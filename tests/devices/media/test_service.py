"""Tests for owned and integrity-checked media transfer."""

from collections import deque

import pytest

from yewo.applications import ApplicationDescriptor
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.media import (
    MediaTransferError,
    MediaTransferPolicy,
    MediaTransferRollbackError,
    MediaTransferService,
)
from yewo.devices.remote_files import (
    RemoteTemporaryFileError,
    RemoteTemporaryFileLease,
)


class _RemoverStub:
    def __init__(self) -> None:
        self.paths: list[str] = []
        self.error: Exception | None = None

    def remove_file(self, remote_path: str) -> None:
        self.paths.append(remote_path)
        if self.error is not None:
            raise self.error


class _TransportStub:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.push_error: Exception | None = None

    def push(self, remote_path: str, content: bytes) -> None:
        if self.push_error is not None:
            raise self.push_error
        self.files[remote_path] = content

    def pull(self, remote_path: str, maximum_bytes: int) -> bytes:
        del maximum_bytes
        return self.files[remote_path]


def _lease(remover: _RemoverStub) -> RemoteTemporaryFileLease:
    identifiers = deque(("namespace_1", "file_1", "file_2"))
    return RemoteTemporaryFileLease.open(
        remover,
        DeviceDescriptor(
            platform=DevicePlatform.ANDROID,
            target_id="device_1",
            connection_id="connection_1",
            display_name="Device",
            kind=DeviceKind.VIRTUAL,
            readiness=DeviceReadiness.READY,
            native_state="ready",
        ),
        ApplicationDescriptor(
            platform=DevicePlatform.ANDROID,
            application_id="com.example.app",
        ),
        identifier_factory=identifiers.popleft,
    )


def test_push_pull_and_release_preserve_content_identity() -> None:
    remover = _RemoverStub()
    transport = _TransportStub()
    lease = _lease(remover)
    service = MediaTransferService(transport, lease)

    reference = service.push("photo.png", b"media")

    assert service.pull(reference) == b"media"
    assert reference.byte_count == 5
    assert lease.owns(reference.remote_file)
    assert service.release(reference) is True
    assert remover.paths == [reference.remote_file.remote_path]


def test_pull_rejects_changed_or_foreign_content() -> None:
    transport = _TransportStub()
    service = MediaTransferService(transport, _lease(_RemoverStub()))
    reference = service.push("photo.png", b"media")
    transport.files[reference.remote_file.remote_path] = b"changed"

    with pytest.raises(MediaTransferError, match="does not match"):
        service.pull(reference)

    foreign_service = MediaTransferService(transport, _lease(_RemoverStub()))
    with pytest.raises(MediaTransferError, match="not owned"):
        foreign_service.pull(reference)


def test_limits_are_checked_before_remote_mutation() -> None:
    transport = _TransportStub()
    lease = _lease(_RemoverStub())
    service = MediaTransferService(
        transport,
        lease,
        policy=MediaTransferPolicy(maximum_bytes=4),
    )

    with pytest.raises(MediaTransferError, match="byte limit"):
        service.push("photo.png", b"media")

    assert transport.files == {}
    assert lease.references == ()


def test_failed_push_cleans_its_reserved_path() -> None:
    remover = _RemoverStub()
    transport = _TransportStub()
    transport.push_error = MediaTransferError("push failed")
    lease = _lease(remover)

    with pytest.raises(MediaTransferError, match="push failed"):
        MediaTransferService(transport, lease).push("photo.png", b"media")

    assert len(remover.paths) == 1
    assert lease.references == ()


def test_failed_push_wraps_an_injected_transport_error_and_cleans_the_path() -> None:
    remover = _RemoverStub()
    transport = _TransportStub()
    transport.push_error = RuntimeError("private transport detail")
    lease = _lease(remover)

    with pytest.raises(MediaTransferError, match="transport failed") as captured:
        MediaTransferService(transport, lease).push("photo.png", b"media")

    assert isinstance(captured.value.__cause__, RuntimeError)
    assert "private transport detail" not in str(captured.value)
    assert lease.references == ()


def test_failed_push_reports_an_uncertain_remote_file() -> None:
    remover = _RemoverStub()
    remover.error = RemoteTemporaryFileError("remove failed")
    transport = _TransportStub()
    transport.push_error = MediaTransferError("push failed")

    with pytest.raises(MediaTransferRollbackError, match="uncertain") as captured:
        MediaTransferService(transport, _lease(remover)).push("photo.png", b"media")

    assert isinstance(captured.value.__cause__, MediaTransferError)
    assert captured.value.__notes__ == ["Remote cleanup also failed: remove failed"]
