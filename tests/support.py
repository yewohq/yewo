"""Fixtures every area of the suite shares."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO

from PIL import Image

# One dispatch time for records that need a timestamp but do not measure one. Tests that do
# measure a duration pass their own, derived from the observation the step started on.
DISPATCHED_AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


class ClockStub:
    """A clock the test advances, so a poll loop costs no wall time.

    Eight copies of this stood in eight test modules, and a recording test that forgot one spent
    a real second sleeping.
    """

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class ProcessStub:
    """A started process the test drives, instead of one the host has to spawn."""

    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.pid = 4321

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.returncode or 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def png(size: tuple[int, int] = (3, 2), colour: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    """Encode the smallest image these tests can carry."""
    image = Image.new("RGB", size, color=colour)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


__all__ = ["DISPATCHED_AT", "ClockStub", "ProcessStub", "png"]
