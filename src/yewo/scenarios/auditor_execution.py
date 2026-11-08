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
    pending_requests = requests
    completed_batches: list[tuple[AuditRequestBatch, AuditBatchAttempt[_Output]]] = []
    while pending_requests:
        workers = min(_MAXIMUM_PARALLEL_AUDITS, len(pending_requests))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            attempts = tuple(executor.map(audit_batch, pending_requests))
        recovery: list[AuditRequestBatch] = []
        for batch, attempt in zip(pending_requests, attempts, strict=True):
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
            completed_batches.append((batch, attempt))
        pending_requests = tuple(recovery)
    return tuple(completed_batches)
