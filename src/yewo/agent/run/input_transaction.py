"""Secret-free evidence for one declared-value input transaction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class FocusReceipt:
    """Focus and keyboard state around dispatch, without any field value."""

    before_bounds: tuple[int, int, int, int] | None = None
    after_bounds: tuple[int, int, int, int] | None = None
    keyboard_before: bool | None = None
    keyboard_after: bool | None = None


class InputTransactionStatus(str, Enum):
    COMPLETED = "completed"
    VALUE_FAILED = "value_failed"
    CONTEXT_LOST = "context_lost"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class InputTransactionReceipt:
    """Final, replayable result of a text-entry transaction."""

    transaction_id: str
    value_name: str
    status: InputTransactionStatus
    value_verified: bool
    context_retained: bool
    reason: str
    focus: FocusReceipt
    source_observation_id: str | None = None
    destination_observation_id: str | None = None
    source_context_id: str | None = None
    source_phash: str | None = None
    destination_phash: str | None = None
