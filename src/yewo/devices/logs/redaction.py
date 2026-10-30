"""Redaction result required before device-log persistence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DeviceLogRedactionResult:
    """Redacted text and the policy evidence that produced it."""

    text: str
    policy_id: str
    redaction_count: int

    def __post_init__(self) -> None:
        if not self.policy_id or self.redaction_count < 0:
            raise ValueError("Device-log redaction provenance is invalid.")


class DeviceLogTextScrubber(Protocol):
    """Redact one device-log value and report policy provenance."""

    def __call__(self, text: str) -> DeviceLogRedactionResult: ...
