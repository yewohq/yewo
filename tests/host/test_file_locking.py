"""Tests for the exclusive file lock shared by every durable store."""

from __future__ import annotations

from pathlib import Path
import threading

import pytest

from symlink_support import requires_symlinks
from host.windows_simulation import (
    FakeWindowsLocking,
    ImmediateClock,
    load_host_module_as_windows,
)
from yewo.host import LOCK_FILE_SUFFIX, exclusive_file_lock


def test_lock_suffix_identifies_internal_bookkeeping() -> None:
    assert LOCK_FILE_SUFFIX == ".lock"
    assert Path(f"ledger.jsonl{LOCK_FILE_SUFFIX}").suffix == LOCK_FILE_SUFFIX


def test_lock_is_released_so_a_later_holder_can_acquire_it(tmp_path: Path) -> None:
    lock_path = tmp_path / "ledger.jsonl.lock"

    with exclusive_file_lock(lock_path):
        assert lock_path.is_file()
    with exclusive_file_lock(lock_path):
        assert lock_path.is_file()


def test_lock_is_released_when_the_held_block_raises(tmp_path: Path) -> None:
    lock_path = tmp_path / "ledger.jsonl.lock"

    with pytest.raises(ValueError, match="store failure"), exclusive_file_lock(lock_path):
        raise ValueError("store failure")

    with exclusive_file_lock(lock_path):
        assert lock_path.is_file()


def test_lock_serializes_concurrent_holders(tmp_path: Path) -> None:
    lock_path = tmp_path / "ledger.jsonl.lock"
    start = threading.Barrier(4)
    held_count = 0
    highest_held_count = 0
    guard = threading.Lock()

    def hold_the_lock() -> None:
        nonlocal held_count, highest_held_count
        start.wait()
        for _ in range(20):
            with exclusive_file_lock(lock_path):
                with guard:
                    held_count += 1
                    highest_held_count = max(highest_held_count, held_count)
                with guard:
                    held_count -= 1

    holders = [threading.Thread(target=hold_the_lock) for _ in range(4)]
    for holder in holders:
        holder.start()
    for holder in holders:
        holder.join(timeout=30)

    assert not any(holder.is_alive() for holder in holders)
    assert highest_held_count == 1


def test_lock_rejects_a_timeout_that_cannot_elapse(tmp_path: Path) -> None:
    with (
        pytest.raises(ValueError, match="greater than zero"),
        exclusive_file_lock(tmp_path / "ledger.lock", timeout_seconds=0),
    ):
        pass


@requires_symlinks
def test_lock_rejects_a_symlinked_lock_path(tmp_path: Path) -> None:
    target = tmp_path / "outside.lock"
    target.write_bytes(b"")
    lock_path = tmp_path / "ledger.jsonl.lock"
    lock_path.symlink_to(target)

    with pytest.raises(OSError), exclusive_file_lock(lock_path):
        pass


def test_windows_branch_locks_and_unlocks_one_byte_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    locking = FakeWindowsLocking()
    module = load_host_module_as_windows(monkeypatch, "file_locking", locking)

    with module.exclusive_file_lock(tmp_path / "ledger.jsonl.lock"):
        assert locking.modes == (FakeWindowsLocking.LK_NBLCK,)

    assert locking.modes == (FakeWindowsLocking.LK_NBLCK, FakeWindowsLocking.LK_UNLCK)
    assert {byte_count for _, byte_count in locking.calls} == {1}


def test_windows_branch_retries_until_the_holder_releases(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    locking = FakeWindowsLocking(failures_before_success=2)
    module = load_host_module_as_windows(monkeypatch, "file_locking", locking)
    clock = ImmediateClock(step_seconds=0.0)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)

    with module.exclusive_file_lock(tmp_path / "ledger.jsonl.lock"):
        pass

    assert locking.modes.count(FakeWindowsLocking.LK_NBLCK) == 3
    assert len(clock.sleeps) == 2


def test_windows_branch_gives_up_when_the_timeout_elapses(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    locking = FakeWindowsLocking(failures_before_success=1000)
    module = load_host_module_as_windows(monkeypatch, "file_locking", locking)
    clock = ImmediateClock(step_seconds=10.0)
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)

    with (
        pytest.raises(PermissionError, match="the lock is held"),
        module.exclusive_file_lock(tmp_path / "ledger.jsonl.lock", timeout_seconds=30),
    ):
        pass

    assert locking.modes.count(FakeWindowsLocking.LK_NBLCK) == 4
