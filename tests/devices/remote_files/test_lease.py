"""Tests for exact remote temporary-file ownership."""

from collections import deque

import pytest

from yewo.applications import ApplicationDescriptor
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DevicePlatform,
    DeviceReadiness,
)
from yewo.devices.remote_files import (
    RemoteTemporaryFileCleanupError,
    RemoteTemporaryFileError,
    RemoteTemporaryFileLease,
)


class _RemoverStub:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failures: set[str] = set()

    def remove_file(self, remote_path: str) -> None:
        self.calls.append(remote_path)
        if remote_path in self.failures:
            raise RemoteTemporaryFileError("remove failed")


def _device(platform: DevicePlatform = DevicePlatform.ANDROID) -> DeviceDescriptor:
    return DeviceDescriptor(
        platform=platform,
        target_id="device_1",
        connection_id="connection_1",
        display_name="Device",
        kind=DeviceKind.VIRTUAL,
        readiness=DeviceReadiness.READY,
        native_state="ready",
    )


def _application(platform: DevicePlatform = DevicePlatform.ANDROID) -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=platform,
        application_id="com.example.app",
    )


def _lease(remover: _RemoverStub) -> RemoteTemporaryFileLease:
    identifiers = deque(("namespace_1", "file_1", "file_2"))
    return RemoteTemporaryFileLease.open(
        remover,
        _device(),
        _application(),
        identifier_factory=identifiers.popleft,
    )


def test_reservation_registers_cleanup_before_the_remote_write() -> None:
    remover = _RemoverStub()
    lease = _lease(remover)

    reference = lease.reserve("fixture.png")

    assert lease.references == (reference,)
    assert reference.remote_path == "/sdcard/Download/.yewo_namespace_1_fixture.png"
    assert remover.calls == []


def test_close_removes_every_exact_path_in_reverse_order() -> None:
    remover = _RemoverStub()
    lease = _lease(remover)
    first = lease.reserve("first.png")
    second = lease.reserve("second.png")

    lease.close()
    lease.close()

    assert remover.calls == [second.remote_path, first.remote_path]
    assert lease.references == ()
    assert lease.closed is True


def test_cleanup_continues_and_a_retry_targets_only_remaining_files() -> None:
    remover = _RemoverStub()
    lease = _lease(remover)
    first = lease.reserve("first.png")
    second = lease.reserve("second.png")
    remover.failures.add(second.remote_path)

    with pytest.raises(RemoteTemporaryFileCleanupError) as captured:
        lease.close()

    assert captured.value.failed_paths == (second.remote_path,)
    assert isinstance(captured.value.__cause__, RemoteTemporaryFileError)
    assert remover.calls == [second.remote_path, first.remote_path]
    assert lease.references == (second,)
    assert lease.closed is True
    assert lease.close_attempted is True
    assert lease.cleanup_complete is False
    with pytest.raises(RemoteTemporaryFileError, match="closed"):
        lease.reserve("third.png")

    remover.failures.clear()
    lease.close()

    assert remover.calls[-1] == second.remote_path
    assert lease.closed is True
    assert lease.cleanup_complete is True


def test_release_is_idempotent_but_rejects_a_changed_reference() -> None:
    remover = _RemoverStub()
    lease = _lease(remover)
    reference = lease.reserve("fixture.png")
    changed = reference.model_copy(update={"name": "other.png"})

    with pytest.raises(RemoteTemporaryFileError, match="does not match"):
        lease.release(changed)

    assert lease.release(reference) is True
    assert lease.release(reference) is False


def test_lease_rejects_duplicate_and_post_close_reservations() -> None:
    lease = _lease(_RemoverStub())
    lease.reserve("fixture.png")

    with pytest.raises(RemoteTemporaryFileError, match="already reserved"):
        lease.reserve("FIXTURE.PNG")

    lease.close()
    with pytest.raises(RemoteTemporaryFileError, match="closed"):
        lease.reserve("other.png")
