"""Bounded concurrent execution and recovery for scenario-audit batches."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from yewo.models import ModelCallDisposition, ModelImageConstraints
from yewo.scenarios.auditor_plan import (
    AuditRequestBatch,
    split_audit_request_batch,
)


_Output = TypeVar("_Output")
_MAXIMUM_PARALLEL_AUDITS = 4
_MAXIMUM_REASONING_RECOVERY_DEPTH = 2


@dataclass(frozen=True)
class AuditBatchAttempt(Generic[_Output]):
    """One normalized batch result with its model disposition."""

    output: _Output | None
    disposition: ModelCallDisposition | None = None


def execute_audit_batches(
    root: Path,
    constraints: ModelImageConstraints,
    requests: tuple[AuditRequestBatch, ...],
    audit_batch: Callable[[AuditRequestBatch], AuditBatchAttempt[_Output]],
) -> tuple[tuple[AuditRequestBatch, AuditBatchAttempt[_Output]], ...]:
    """Run batches concurrently and split exhausted visual evidence twice."""
    pending = requests
    completed: list[tuple[AuditRequestBatch, AuditBatchAttempt[_Output]]] = []
    while pending:
        workers = min(_MAXIMUM_PARALLEL_AUDITS, len(pending))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            attempts = tuple(executor.map(audit_batch, pending))
        recovery: list[AuditRequestBatch] = []
        for batch, attempt in zip(pending, attempts, strict=True):
            if attempt.disposition is ModelCallDisposition.REASONING_EXHAUSTED:
                split = split_audit_request_batch(
                    root,
                    constraints,
                    batch,
                    maximum_recovery_depth=_MAXIMUM_REASONING_RECOVERY_DEPTH,
                )
                if split:
                    recovery.extend(split)
                    continue
            completed.append((batch, attempt))
        pending = tuple(recovery)
    return tuple(completed)
