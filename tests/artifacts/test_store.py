"""Tests for immutable writes shared by all artifact stores."""

from pathlib import Path

import pytest

from yewo.artifacts import ARTIFACTS_DIRECTORY
from yewo.artifacts.errors import ArtifactWriteError
from yewo.artifacts.store import AtomicArtifactWriter


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
