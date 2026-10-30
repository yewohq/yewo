"""Own one device-log interval for an Appium driver session."""

from __future__ import annotations

from typing import Self

from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.records import (
    DeviceLogBoundary,
    DeviceLogCheckpointKind,
    DeviceLogDrainReceipt,
)
from yewo.devices.logs.service import DeviceLogCaptureService


class DeviceLogCaptureLease:
    """Drain and close one run-owned device-log interval."""

    def __init__(
        self,
        service: DeviceLogCaptureService,
        boundary: DeviceLogBoundary,
        startup_receipt: DeviceLogDrainReceipt,
    ) -> None:
        self._service = service
        self.boundary = boundary
        self._receipts: list[DeviceLogDrainReceipt] = [startup_receipt]
        self._closed = False

    @classmethod
    def open(cls, service: DeviceLogCaptureService) -> Self:
        """Start a boundary and return its owning lease."""
        started = service.begin()
        return cls(service, started.boundary, started.startup_receipt)

    @property
    def closed(self) -> bool:
        """Return whether the final drain was attempted."""
        return self._closed

    @property
    def receipts(self) -> tuple[DeviceLogDrainReceipt, ...]:
        """Return completed drain receipts in capture order."""
        return tuple(self._receipts)

    def drain_action(self, action_id: str) -> DeviceLogDrainReceipt:
        """Drain entries produced through one completed action."""
        return self._drain(DeviceLogCheckpointKind.ACTION, action_id)

    def drain_scenario(self, scenario_id: str) -> DeviceLogDrainReceipt:
        """Drain entries produced through one completed scenario."""
        return self._drain(DeviceLogCheckpointKind.SCENARIO, scenario_id)

    def close(self) -> None:
        """Perform one final drain and close the interval."""
        if self._closed:
            return
        try:
            self._receipts.append(self._service.finish(self.boundary))
        finally:
            self._closed = True

    def _drain(
        self,
        checkpoint_kind: DeviceLogCheckpointKind,
        checkpoint_id: str,
    ) -> DeviceLogDrainReceipt:
        if self._closed:
            raise DeviceLogCaptureError(
                "A closed device-log capture lease cannot drain logs."
            )
        receipt = self._service.drain(
            self.boundary,
            checkpoint_kind,
            checkpoint_id,
        )
        self._receipts.append(receipt)
        return receipt
