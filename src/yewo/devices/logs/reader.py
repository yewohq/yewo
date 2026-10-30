"""Typed values exchanged at device-log reader boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, TypeAlias

from yewo.devices.logs.records import DeviceLogCheckpointKind
from yewo.devices.diagnostics import ApplicationFailureKind


@dataclass(frozen=True)
class DeviceLogTextEntry:
    """One normalized entry from a destructive text-log stream."""

    text: str


@dataclass(frozen=True)
class ApplicationDiagnosticReport:
    """One platform report with a deterministic failure classification."""

    source_name: str
    text: str
    failure_kind: ApplicationFailureKind
    occurred_at: datetime | None = None
    platform_reason: str | None = None
    platform_code: str | None = None


DeviceLogItem: TypeAlias = DeviceLogTextEntry | ApplicationDiagnosticReport


@dataclass(frozen=True)
class DeviceLogBatch:
    """One source read with explicit filtering and completeness metadata."""

    items: tuple[DeviceLogItem, ...]
    observed_item_count: int
    filtered_item_count: int = 0
    unparsed_item_count: int = 0
    may_have_omitted_earlier_items: bool = False

    def __post_init__(self) -> None:
        accounted_item_count = len(self.items) + self.filtered_item_count + self.unparsed_item_count
        if self.observed_item_count < accounted_item_count:
            raise ValueError("A device-log batch cannot account for more items than it observed.")
        if self.filtered_item_count < 0 or self.unparsed_item_count < 0:
            raise ValueError("A device-log excluded-item count cannot be negative.")


class DeviceLogReader(Protocol):
    """Own destructive reads for configured sources in one device session."""

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        """Return the source baseline, or no value when the source is unsupported."""
        ...

    def read(self, log_type: str) -> DeviceLogBatch:
        """Return items observed since the source's previous acknowledged read."""
        ...

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        """Return whether the source must run at this checkpoint."""
        ...

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        """Commit a non-destructive source cursor after durable persistence."""
        ...
