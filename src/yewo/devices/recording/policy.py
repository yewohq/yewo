"""Policy for one screen-recording interval."""

from dataclasses import dataclass

from yewo.artifacts.records import ArtifactRetention, ArtifactSensitivity


DEFAULT_RECORDING_TIME_LIMIT_SECONDS = 1_800
DEFAULT_RECORDING_STOP_TIMEOUT_SECONDS = 10.0
DEFAULT_RECORDING_PULL_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class ScreenRecordingPolicy:
    """Enable recording and bound each device-side segment."""

    enabled: bool = True
    maximum_segment_seconds: int = DEFAULT_RECORDING_TIME_LIMIT_SECONDS
    maximum_stop_seconds: float = DEFAULT_RECORDING_STOP_TIMEOUT_SECONDS
    maximum_pull_seconds: float = DEFAULT_RECORDING_PULL_TIMEOUT_SECONDS
    sensitivity: ArtifactSensitivity = ArtifactSensitivity.DEVICE_CONTENT
    retention: ArtifactRetention = ArtifactRetention.RUN

    def __post_init__(self) -> None:
        if self.maximum_segment_seconds < 1:
            raise ValueError("The recording segment limit must be positive.")
        if self.maximum_stop_seconds <= 0 or self.maximum_pull_seconds <= 0:
            raise ValueError("Recording finalization timeouts must be positive.")
