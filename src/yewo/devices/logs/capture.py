"""Configuration and startup result for one device-log interval."""

from dataclasses import dataclass

from yewo.artifacts.records import ArtifactRetention
from yewo.devices.logs.records import DeviceLogBoundary, DeviceLogDrainReceipt


DEFAULT_MAXIMUM_DRAIN_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class DeviceLogCapturePolicy:
    """Retention and size bounds for device-log artifacts."""

    retention: ArtifactRetention = ArtifactRetention.RUN
    maximum_drain_bytes: int = DEFAULT_MAXIMUM_DRAIN_BYTES

    def __post_init__(self) -> None:
        if self.maximum_drain_bytes < 1:
            raise ValueError("The maximum device-log drain size must be positive.")


@dataclass(frozen=True)
class DeviceLogCaptureStart:
    """The durable boundary and startup receipt for one log interval."""

    boundary: DeviceLogBoundary
    startup_receipt: DeviceLogDrainReceipt
