"""Lifecycle operations as concise report-page rows."""

from typing import Sequence

from yewo.testing.lifecycle import LifecycleOperation


def lifecycle_rows(operations: Sequence[LifecycleOperation]) -> tuple[str, ...]:
    """Format setup and restoration evidence without deciding its status."""
    rows = []
    for operation in operations:
        parts = (
            operation.phase.value.title(),
            operation.requested,
            operation.status.value.title(),
            operation.failure_reason or "",
        )
        rows.append(" · ".join(part for part in parts if part))
    return tuple(rows)
