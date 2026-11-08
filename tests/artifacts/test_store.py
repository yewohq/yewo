"""Tests for immutable writes shared by all artifact stores."""

from pathlib import Path

import pytest

from symlink_support import requires_symlinks

from yewo.artifacts import ARTIFACTS_DIRECTORY
from yewo.artifacts.errors import ArtifactWriteError
from yewo.artifacts.store import AppendOnlyJsonlWriter, AtomicArtifactWriter


def test_shared_artifact_directory_is_run_relative() -> None:
    assert Path() == ARTIFACTS_DIRECTORY
    assert not ARTIFACTS_DIRECTORY.is_absolute()


def test_writer_never_overwrites_existing_evidence(tmp_path: Path) -> None:
    writer = AtomicArtifactWriter(tmp_path)
    relative_path = ARTIFACTS_DIRECTORY / "evidence.bin"
    writer.write(relative_path, b"first")

    with pytest.raises(ArtifactWriteError, match="already exists"):
        writer.write(relative_path, b"second")

    assert (tmp_path / relative_path).read_bytes() == b"first"


def test_writer_streams_a_regular_file_without_changing_the_source(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"streamed evidence")
    artifact_root = tmp_path / "run"
    relative_path = ARTIFACTS_DIRECTORY / "evidence.bin"

    AtomicArtifactWriter(artifact_root).write_file(relative_path, source)

    assert (artifact_root / relative_path).read_bytes() == b"streamed evidence"
    assert source.read_bytes() == b"streamed evidence"


@requires_symlinks
def test_writer_rejects_a_symlinked_artifact_directory(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    artifact_root.mkdir()
    outside.mkdir()
    (artifact_root / "linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactWriteError, match="outside the artifact root"):
        AtomicArtifactWriter(artifact_root).write(
            Path("linked/evidence.bin"),
            b"evidence",
        )

    assert tuple(outside.iterdir()) == ()


def test_ledger_appends_each_uniquely_identified_record(tmp_path: Path) -> None:
    writer = AppendOnlyJsonlWriter(tmp_path)
    relative_path = ARTIFACTS_DIRECTORY / "ledger.jsonl"

    first = writer.append(relative_path, {"record_id": "a"}, identity_field="record_id")
    second = writer.append(relative_path, {"record_id": "b"}, identity_field="record_id")

    assert (tmp_path / relative_path).read_bytes() == first + second


def test_ledger_locks_a_sidecar_so_readers_are_never_blocked(tmp_path: Path) -> None:
    relative_path = ARTIFACTS_DIRECTORY / "ledger.jsonl"

    AppendOnlyJsonlWriter(tmp_path).append(
        relative_path,
        {"record_id": "a"},
        identity_field="record_id",
    )

    assert (tmp_path / "ledger.jsonl.lock").is_file()
    assert (tmp_path / relative_path).read_text(encoding="utf-8") == '{"record_id":"a"}\n'


def test_ledger_rejects_a_repeated_identity(tmp_path: Path) -> None:
    writer = AppendOnlyJsonlWriter(tmp_path)
    relative_path = ARTIFACTS_DIRECTORY / "ledger.jsonl"
    writer.append(relative_path, {"record_id": "a"}, identity_field="record_id")

    with pytest.raises(ArtifactWriteError, match="already exists"):
        writer.append(relative_path, {"record_id": "a"}, identity_field="record_id")


def test_ledger_rejects_a_missing_identity(tmp_path: Path) -> None:
    with pytest.raises(ArtifactWriteError, match="nonempty"):
        AppendOnlyJsonlWriter(tmp_path).append(
            ARTIFACTS_DIRECTORY / "ledger.jsonl",
            {"record_id": ""},
            identity_field="record_id",
        )


def test_ledger_rejects_an_incomplete_final_line(tmp_path: Path) -> None:
    (tmp_path / "ledger.jsonl").write_bytes(b'{"record_id":"a"}')

    with pytest.raises(ArtifactWriteError, match="incomplete line: 1"):
        AppendOnlyJsonlWriter(tmp_path).append(
            ARTIFACTS_DIRECTORY / "ledger.jsonl",
            {"record_id": "b"},
            identity_field="record_id",
        )


def test_ledger_rejects_an_unreadable_line(tmp_path: Path) -> None:
    (tmp_path / "ledger.jsonl").write_bytes(b'{"record_id":"a"}\nnot json\n')

    with pytest.raises(ArtifactWriteError, match="invalid line: 2"):
        AppendOnlyJsonlWriter(tmp_path).append(
            ARTIFACTS_DIRECTORY / "ledger.jsonl",
            {"record_id": "b"},
            identity_field="record_id",
        )
