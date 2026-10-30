"""Own one checkpointed screen-recording interval."""

from __future__ import annotations

from typing import Self

from yewo.devices.recording.errors import ScreenRecordingError
from yewo.devices.recording.records import (
    ScreenRecordingBoundary,
    ScreenRecordingCheckpointKind,
    ScreenRecordingCheckpointReceipt,
)
from yewo.devices.recording.service import ScreenRecordingService
from yewo.observations.records import RedactionRegion


class ScreenRecordingLease:
    """Checkpoint and close one run-owned recording interval."""

    def __init__(
        self,
        service: ScreenRecordingService,
        boundary: ScreenRecordingBoundary,
    ) -> None:
        self._service = service
        self.boundary = boundary
        self._receipts: list[ScreenRecordingCheckpointReceipt] = []
        self._closed = False

    @classmethod
    def open(cls, service: ScreenRecordingService) -> Self:
        """Start a recording boundary and return its owning lease."""
        return cls(service, service.begin())

    @property
    def closed(self) -> bool:
        """Return whether the final recording checkpoint was attempted."""
        return self._closed

    @property
    def receipts(self) -> tuple[ScreenRecordingCheckpointReceipt, ...]:
        """Return completed recording checkpoints in segment order."""
        return tuple(self._receipts)

    def checkpoint_scenario(self, scenario_id: str) -> ScreenRecordingCheckpointReceipt:
        """Close the current scenario segment and start the next segment."""
        if self._closed:
            raise ScreenRecordingError("A closed recording lease cannot checkpoint video.")
        receipt = self._service.checkpoint(
            self.boundary,
            ScreenRecordingCheckpointKind.SCENARIO,
            scenario_id,
        )
        self._receipts.append(receipt)
        return receipt

    def add_redaction_region(
        self,
        region: RedactionRegion,
        source_width_px: int,
        source_height_px: int,
    ) -> None:
        """Start a persistent mask before sensitive pixels appear."""
        if self._closed:
            raise ScreenRecordingError("A closed recording lease cannot add a redaction region.")
        self._service.add_redaction_region(
            region,
            source_width_px,
            source_height_px,
        )

    def clear_redaction_regions(self) -> None:
        """Stop carrying the current masks into later segments."""
        if self._closed:
            raise ScreenRecordingError("A closed recording lease cannot clear redaction regions.")
        self._service.clear_redaction_regions()

    def close(self) -> None:
        """Close the final segment and release the recording interval."""
        if self._closed:
            return
        try:
            self._receipts.append(self._service.finish(self.boundary))
        finally:
            self._closed = True
