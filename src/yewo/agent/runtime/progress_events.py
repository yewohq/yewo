"""Persist append-only same-run workflow progress events."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.planning.records import PlanningFocusContext, ProgressRouteAttempt
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AppendOnlyJsonlWriter,
    ArtifactWriteError,
    RecordIdentifier,
)


MAXIMUM_FEEDBACK_CHARACTERS = 4_000


class RunProgressPersistenceError(RuntimeError):
    """The runtime could not append same-run progress evidence."""


class InvalidRunProgressAssessment(ValueError):
    """A verifier assessment referenced work outside the current task."""


class RunProgressEvent(BaseModel):
    """One immutable update to the same-run workflow ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: RecordIdentifier
    event_index: int = Field(ge=1)
    kind: Literal["initialized", "assessment", "refusal", "route", "planning_focus", "stage"]
    observation_id: RecordIdentifier
    work_items: tuple[str, ...] = Field(default=(), max_length=100)
    completed_work_items: tuple[str, ...] = Field(default=(), max_length=100)
    proposal_id: RecordIdentifier | None = None
    proposal_work_item: str | None = Field(default=None, min_length=1, max_length=4_000)
    restarts_completed_work_item: str | None = Field(
        default=None,
        min_length=1,
        max_length=4_000,
    )
    interaction_intent_key: str | None = Field(default=None, min_length=1, max_length=4_000)
    refusal_code: str | None = Field(default=None, min_length=1, max_length=100)
    feedback: str | None = Field(default=None, min_length=1, max_length=MAXIMUM_FEEDBACK_CHARACTERS)
    route: ProgressRouteAttempt | None = None
    planning_focus: PlanningFocusContext | None = None


class RunProgressEventStore:
    """Append immutable progress events with one monotonic run-local index."""

    def __init__(self, root: Path) -> None:
        self._writer = AppendOnlyJsonlWriter(root)
        self._event_index = 0

    def next_index(self) -> int:
        """Return the next event index for this run."""
        self._event_index += 1
        return self._event_index

    def append(self, event: RunProgressEvent) -> None:
        """Persist one event without replacing prior evidence."""
        try:
            self._writer.append(
                ARTIFACTS_DIRECTORY / "agent" / "progress-ledger.jsonl",
                event.model_dump(mode="json"),
                identity_field="event_id",
            )
        except ArtifactWriteError as error:
            raise RunProgressPersistenceError(
                "Could not persist same-run progress evidence."
            ) from error
