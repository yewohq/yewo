"""Failures owned by the run-coordination boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yewo.runs.records import RunTerminal


class RunCoordinationError(RuntimeError):
    """A run could not establish or preserve truthful ownership."""


class RunTerminalPersistenceError(RunCoordinationError):
    """A computed terminal exists in memory but could not be persisted."""

    def __init__(self, terminal: RunTerminal) -> None:
        super().__init__("The computed run terminal could not be persisted.")
        self.terminal = terminal


class RunWorkflowInterrupted(RuntimeError):
    """A workflow observed cooperative interruption between owned operations."""


class RunWorkflowEvidenceError(RuntimeError):
    """A workflow could not preserve partial evidence while propagating a failure."""

    def __init__(self, *, interrupted: bool, original_error: BaseException) -> None:
        super().__init__("The workflow could not persist partial failure evidence.")
        self.interrupted = interrupted
        self.original_error = original_error
