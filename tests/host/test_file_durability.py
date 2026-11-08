"""Tests for durable replacement across hosts with different flush guarantees."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from host.windows_simulation import ImmediateClock, load_host_module_as_windows
from yewo.host import flush_directory_entry, replace_file_durably


def test_replace_puts_the_temporary_content_in_place(tmp_path: Path) -> None:
    temporary_path = tmp_path / ".snapshot-tmp"
    destination = tmp_path / "snapshot.json"
    destination.write_bytes(b"previous")
    temporary_path.write_bytes(b"current")

    replace_file_durably(temporary_path, destination)

    assert destination.read_bytes() == b"current"
    assert not temporary_path.exists()


def test_flush_reports_a_missing_directory_to_its_caller(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        flush_directory_entry(tmp_path / "absent")


def test_windows_flush_skips_the_unavailable_directory_handle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_host_module_as_windows(monkeypatch, "file_durability")

    module.flush_directory_entry(tmp_path / "absent")


def test_windows_replace_retries_while_a_reader_holds_the_destination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_host_module_as_windows(monkeypatch, "file_durability")
    clock = ImmediateClock(step_seconds=0.0)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    attempts = 0
    real_replace = os.replace

    def replace_after_two_sharing_violations(source: Path, target: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts <= 2:
            raise PermissionError("the destination is open")
        real_replace(source, target)

    monkeypatch.setattr(module.os, "replace", replace_after_two_sharing_violations)
    temporary_path = tmp_path / ".snapshot-tmp"
    temporary_path.write_bytes(b"current")
    destination = tmp_path / "snapshot.json"

    module.replace_file_durably(temporary_path, destination)

    assert attempts == 3
    assert destination.read_bytes() == b"current"
    assert len(clock.sleeps) == 2


def test_windows_replace_gives_up_when_the_destination_stays_open(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = load_host_module_as_windows(monkeypatch, "file_durability")
    clock = ImmediateClock(step_seconds=1.0)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)

    def always_refuse(source: Path, target: Path) -> None:
        del source, target
        raise PermissionError("the destination is open")

    monkeypatch.setattr(module.os, "replace", always_refuse)

    with pytest.raises(PermissionError, match="the destination is open"):
        module.replace_file_durably(tmp_path / ".snapshot-tmp", tmp_path / "snapshot.json")
