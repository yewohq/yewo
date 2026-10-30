"""Route device-log sources to their owning readers."""

from __future__ import annotations

from collections.abc import Mapping

from yewo.devices.logs.errors import DeviceLogCaptureError
from yewo.devices.logs.reader import DeviceLogBatch, DeviceLogReader
from yewo.devices.logs.records import DeviceLogCheckpointKind


class RoutedDeviceLogReader:
    """Expose several source-specific readers through one log boundary."""

    def __init__(self, readers_by_log_type: Mapping[str, DeviceLogReader]) -> None:
        if not readers_by_log_type:
            raise ValueError("A routed device-log reader requires at least one source.")
        if any(not log_type for log_type in readers_by_log_type):
            raise ValueError("A routed device-log source cannot be empty.")
        self._readers_by_log_type = dict(readers_by_log_type)

    def begin(self, log_type: str) -> DeviceLogBatch | None:
        return self._reader(log_type).begin(log_type)

    def read(self, log_type: str) -> DeviceLogBatch:
        return self._reader(log_type).read(log_type)

    def captures_checkpoint(
        self,
        log_type: str,
        checkpoint_kind: DeviceLogCheckpointKind,
    ) -> bool:
        return self._reader(log_type).captures_checkpoint(log_type, checkpoint_kind)

    def acknowledge(self, log_type: str, batch: DeviceLogBatch) -> None:
        self._reader(log_type).acknowledge(log_type, batch)

    def _reader(self, log_type: str) -> DeviceLogReader:
        reader = self._readers_by_log_type.get(log_type)
        if reader is None:
            raise DeviceLogCaptureError("No reader owns this device-log source.")
        return reader
