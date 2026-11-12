"""Owned local files produced by a screen recorder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from yewo.devices.recording.errors import ScreenRecordingError


@dataclass(frozen=True)
class ScreenRecordingStart:
    """Requested and effective device-side limits for one segment."""

    requested_maximum_seconds: int
    effective_maximum_seconds: int
    maximum_stop_seconds: float = 10.0
    maximum_pull_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.requested_maximum_seconds < 1:
            raise ValueError("The requested recording limit must be positive.")
        if not 1 <= self.effective_maximum_seconds <= self.requested_maximum_seconds:
            raise ValueError("The effective recording limit is invalid.")
        if self.maximum_stop_seconds <= 0 or self.maximum_pull_seconds <= 0:
            raise ValueError("Recording finalization timeouts must be positive.")


class ScreenRecordingCapture:
    """Own one local MP4 until persistence accepts or rejects it."""

    def __init__(
        self,
        path: Path,
        start: ScreenRecordingStart,
        *,
        acknowledge: Callable[[], None] | None = None,
        cleanup: Callable[[], None] | None = None,
    ) -> None:
        self.path = path
        self.start = start
        self._acknowledge = acknowledge
        self._cleanup = cleanup
        self._accepted = False
        self._closed = False

    @property
    def accepted(self) -> bool:
        """Return whether durable persistence accepted this capture."""
        return self._accepted

    @property
    def requires_acknowledgement(self) -> bool:
        """Return whether acceptance releases a remote cleanup obligation."""
        return self._acknowledge is not None

    def accept(self) -> None:
        """Acknowledge durable persistence before releasing remote recovery."""
        if self._accepted:
            return
        try:
            if self._acknowledge is not None:
                self._acknowledge()
        except Exception as error:  # noqa: BLE001 - injected ownership boundary
            raise ScreenRecordingError(
                "Could not acknowledge the persisted screen recording."
            ) from error
        self._accepted = True

    def close(self) -> None:
        """Release only the local temporary capture."""
        if self._closed:
            return
        try:
            if self._cleanup is not None:
                self._cleanup()
        except Exception as error:  # noqa: BLE001 - injected ownership boundary
            raise ScreenRecordingError("Could not remove a local temporary recording.") from error
        self._closed = True
