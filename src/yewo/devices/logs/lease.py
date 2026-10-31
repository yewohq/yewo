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
from yewo.run_narration import current_narrator


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
        lease = cls(service, started.boundary, started.startup_receipt)
        _narrate_boundary(started.boundary, started.startup_receipt)
        return lease

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
            receipt = self._service.finish(self.boundary)
            self._receipts.append(receipt)
            _narrate_drain(receipt, finished=True)
        finally:
            self._closed = True

    def _drain(
        self,
        checkpoint_kind: DeviceLogCheckpointKind,
        checkpoint_id: str,
    ) -> DeviceLogDrainReceipt:
        if self._closed:
            raise DeviceLogCaptureError("A closed device-log capture lease cannot drain logs.")
        receipt = self._service.drain(
            self.boundary,
            checkpoint_kind,
            checkpoint_id,
        )
        self._receipts.append(receipt)
        _narrate_drain(receipt, finished=False)
        return receipt


def _narrate_boundary(
    boundary: DeviceLogBoundary,
    startup_receipt: DeviceLogDrainReceipt,
) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    ready = sum(source.state.value == "ready" for source in boundary.sources)
    narrator.record(
        "device-logs",
        f"Device-log capture started with {ready} of {len(boundary.sources)} source(s) ready.",
    )
    narrator.trace(
        "device-logs",
        f"boundary={boundary.boundary_id} sources={len(boundary.sources)} ready={ready}",
    )
    _trace_drain(startup_receipt)


def _narrate_drain(receipt: DeviceLogDrainReceipt, *, finished: bool) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    retained = sum(source.retained_item_count for source in receipt.sources)
    if finished:
        narrator.record(
            "device-logs",
            f"Device-log capture finished with {retained} retained item(s) in the final drain.",
        )
    _trace_drain(receipt)


def _trace_drain(receipt: DeviceLogDrainReceipt) -> None:
    narrator = current_narrator()
    if narrator is None:
        return
    states = ",".join(f"{item.source.log_type}:{item.state.value}" for item in receipt.sources)
    retained = sum(source.retained_item_count for source in receipt.sources)
    artifacts = ",".join(
        str(artifact.path) for source in receipt.sources for artifact in source.artifacts
    )
    narrator.trace(
        "device-logs",
        f"boundary={receipt.boundary_id} drain={receipt.drain_id} "
        f"index={receipt.drain_index} checkpoint={receipt.checkpoint_kind.value} "
        f"retained={retained} states={states} artifacts={artifacts or 'none'}",
    )
