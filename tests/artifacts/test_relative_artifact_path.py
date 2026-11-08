"""Tests that a stored artifact path reads the same on every host."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import BaseModel

from yewo.artifacts import RelativeArtifactPath, format_relative_artifact_path


class _StoredRecord(BaseModel):
    path: RelativeArtifactPath


def test_a_windows_path_renders_with_posix_separators() -> None:
    rendered = format_relative_artifact_path(PureWindowsPath(r"agent\steps.jsonl"))

    assert rendered == "agent/steps.jsonl"


def test_a_posix_path_renders_unchanged() -> None:
    rendered = format_relative_artifact_path(PurePosixPath("agent/steps.jsonl"))

    assert rendered == "agent/steps.jsonl"


def test_a_stored_record_serializes_through_the_shared_formatter() -> None:
    record = _StoredRecord(path=Path("observations/screen.png"))

    assert record.model_dump(mode="json") == {"path": "observations/screen.png"}


def test_a_stored_record_keeps_a_real_path_for_python_callers() -> None:
    record = _StoredRecord(path=Path("observations/screen.png"))

    assert record.model_dump()["path"].parts == ("observations", "screen.png")
