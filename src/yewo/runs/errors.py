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
