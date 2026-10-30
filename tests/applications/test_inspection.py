"""Tests for shared application artifact inspection."""

from pathlib import Path

import pytest

from yewo.applications import ApplicationInspectionError
from yewo.applications.inspection import artifact_sha256


def test_directory_digest_changes_with_member_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.app"
    second = tmp_path / "second.app"
    first.mkdir()
    second.mkdir()
    (first / "one").write_bytes(b"same")
    (second / "two").write_bytes(b"same")

    assert artifact_sha256(first) != artifact_sha256(second)


def test_artifact_digest_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(ApplicationInspectionError, match="does not exist"):
        artifact_sha256(tmp_path / "missing.apk")
