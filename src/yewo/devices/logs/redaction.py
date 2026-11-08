"""Redaction result required before device-log persistence."""

from __future__ import annotations

from dataclasses import dataclass
import re
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


class DeclaredValueDeviceLogScrubber:
    """Replace exact caller-declared sensitive values before persistence."""

    POLICY_ID = "yewo-declared-values-v1"
    REPLACEMENT = "[yewo-redacted]"

    def __init__(self, sensitive_values: tuple[str, ...]) -> None:
        if any(not value for value in sensitive_values):
            raise ValueError("A device-log sensitive value cannot be empty.")
        if len(sensitive_values) != len(set(sensitive_values)):
            raise ValueError("A device-log sensitive value cannot be repeated.")
        ordered_values = sorted(sensitive_values, key=lambda value: (-len(value), value))
        self._pattern = (
            re.compile("|".join(re.escape(value) for value in ordered_values)) if ordered_values else None
        )

    def __call__(self, text: str) -> DeviceLogRedactionResult:
        """Return text after exact-value replacement with explicit provenance."""
        if self._pattern is None:
            redacted, redaction_count = text, 0
        else:
            redacted, redaction_count = self._pattern.subn(self.REPLACEMENT, text)
        return DeviceLogRedactionResult(
            text=redacted,
            policy_id=self.POLICY_ID,
            redaction_count=redaction_count,
        )
