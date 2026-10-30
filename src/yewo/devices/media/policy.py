"""Limits for one media-transfer boundary."""

from dataclasses import dataclass


DEFAULT_MAXIMUM_MEDIA_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class MediaTransferPolicy:
    """Bound binary content handled in memory by Appium."""

    maximum_bytes: int = DEFAULT_MAXIMUM_MEDIA_BYTES

    def __post_init__(self) -> None:
        if self.maximum_bytes < 1:
            raise ValueError("The media-transfer byte limit must be positive.")
