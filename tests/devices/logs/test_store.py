"""Tests for atomic device-log artifact persistence."""

from hashlib import sha256
from pathlib import Path

import pytest

from yewo.artifacts import ArtifactRetention
from yewo.devices.logs import (
    DeviceLogArtifactKind,
    DeviceLogCaptureError,
    DeviceLogSource,
    DeviceLogStore,
)


SOURCE = DeviceLogSource(log_type="logcat", artifact_stem="logcat")


def _persist(store: DeviceLogStore, *, boundary_id: str = "boundary_1"):
    return store.persist_artifact(
        b"one\ntwo\n",
        boundary_id=boundary_id,
        drain_id="drain_1",
        drain_index=1,
        source=SOURCE,
        entry_count=2,
        kind=DeviceLogArtifactKind.TEXT_STREAM,
        source_name=None,
        retention=ArtifactRetention.RUN,
    )


def test_store_persists_one_immutable_log_segment(tmp_path: Path) -> None:
    store = DeviceLogStore(tmp_path, identifier_factory=lambda: "log_1")

    reference = _persist(store)

    assert reference.path == Path("device-logs/boundary_1/000001-drain_1-logcat-log_1.log")
    assert reference.sha256 == sha256(b"one\ntwo\n").hexdigest()
    assert (tmp_path / reference.path).read_bytes() == b"one\ntwo\n"


def test_store_discards_an_unchanged_incomplete_artifact(tmp_path: Path) -> None:
    store = DeviceLogStore(tmp_path, identifier_factory=lambda: "log_1")
    reference = _persist(store)

    store.discard(reference)

    assert not (tmp_path / reference.path).exists()
    assert not (tmp_path / "device-logs/boundary_1").exists()


def test_store_refuses_to_discard_a_changed_artifact(tmp_path: Path) -> None:
    store = DeviceLogStore(tmp_path, identifier_factory=lambda: "log_1")
    reference = _persist(store)
    (tmp_path / reference.path).write_bytes(b"changed")

    with pytest.raises(DeviceLogCaptureError, match="Refused to remove"):
        store.discard(reference)


def test_store_rejects_unsafe_boundary_before_writing(tmp_path: Path) -> None:
    store = DeviceLogStore(tmp_path, identifier_factory=lambda: "log_1")

    with pytest.raises(DeviceLogCaptureError, match="persist"):
        _persist(store, boundary_id="../outside")

    assert not (tmp_path / "device-logs").exists()
