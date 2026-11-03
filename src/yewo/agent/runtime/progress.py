"""Evidence-linked same-run workflow progress and route history."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yewo.agent.actuation import ActionOutcome, ActionProposal
from yewo.agent.planning.planner import ProgressRouteAttempt, TaskProgressContext
from yewo.agent.runtime.records import AgentStep
from yewo.agent.runtime.verification import PreDispatchVerification
from yewo.artifacts import (
    ARTIFACTS_DIRECTORY,
    AppendOnlyJsonlWriter,
    ArtifactWriteError,
    RecordIdentifier,
    new_record_identifier,
)
from yewo.observations import Observation


class RunProgressPersistenceError(RuntimeError):
    """The runtime could not append same-run progress evidence."""


class RunProgressEvent(BaseModel):
    """One immutable update to the same-run workflow ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    event_id: RecordIdentifier
    event_index: int = Field(ge=1)
    kind: Literal["initialized", "assessment", "route"]
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
    route: ProgressRouteAttempt | None = None


class RunProgressLedger:
    """Retain completed work and observed routes for one Device Agent run."""

    def __init__(
        self,
        root: Path,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._writer = AppendOnlyJsonlWriter(root)
        self._identifier_factory = identifier_factory
        self._event_index = 0
        self._work_items: tuple[str, ...] = ()
        self._completed_work_items: list[str] = []
        self._route_attempts: list[ProgressRouteAttempt] = []

    def start(
        self,
        work_items: tuple[str, ...],
        observation: Observation,
        base: TaskProgressContext | None,
    ) -> None:
        """Initialize the ledger from the task and any accepted preparation context."""
        self._work_items = tuple(dict.fromkeys(work_items))
        if base is not None:
            self._completed_work_items = list(
                dict.fromkeys(
                    item for item in base.completed_work_items if item in self._work_items
                )
            )
        self._append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._next_event_index(),
                kind="initialized",
                observation_id=observation.observation_id,
                work_items=self._work_items,
                completed_work_items=tuple(self._completed_work_items),
            )
        )

    def progress_context(self, base: TaskProgressContext | None) -> TaskProgressContext:
        """Return compact ledger state for planner and verifier prompts."""
        context = base or TaskProgressContext()
        completed = tuple(
            dict.fromkeys((*context.completed_work_items, *self._completed_work_items))
        )
        candidates = context.remaining_work or self._work_items
        remaining = tuple(item for item in candidates if item not in completed)
        routes = tuple((*context.route_attempts, *self._route_attempts)[-100:])
        return context.model_copy(
            update={
                "completed_work_items": completed,
                "remaining_work": remaining,
                "route_attempts": routes,
            }
        )

    def record_assessment(
        self,
        observation: Observation,
        proposal: ActionProposal,
        completed_work_items: tuple[str, ...],
        proposal_work_item: str | None,
        restarts_completed_work_item: str | None,
    ) -> None:
        """Record one verifier assessment and accept evidence-linked completions."""
        unknown = set(completed_work_items) - set(self._work_items)
        if unknown:
            raise ValueError("Run progress contains a work item outside the task.")
        self._completed_work_items = list(
            dict.fromkeys((*self._completed_work_items, *completed_work_items))
        )
        self._append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._next_event_index(),
                kind="assessment",
                observation_id=observation.observation_id,
                completed_work_items=completed_work_items,
                proposal_id=proposal.proposal_id,
                proposal_work_item=proposal_work_item,
                restarts_completed_work_item=restarts_completed_work_item,
            )
        )

    def record_verification(
        self,
        verification: PreDispatchVerification,
        observation: Observation,
        proposal: ActionProposal,
    ) -> None:
        """Record one optional pre-dispatch progress assessment."""
        if not verification.progress_assessed:
            return
        self.record_assessment(
            observation,
            proposal,
            verification.completed_work_items,
            verification.proposal_work_item,
            verification.restarts_completed_work_item,
        )

    def record_route(
        self,
        step_index: int,
        proposal: ActionProposal,
        proposal_work_item: str | None,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
    ) -> None:
        """Record one observed route independently from bounded planner history."""
        if proposal_work_item is None:
            return
        route = ProgressRouteAttempt(
            step_index=step_index,
            work_item=proposal_work_item,
            source_context=before.platform_context,
            action=proposal.action,
            target=proposal.target_expression,
            destination_context=after.platform_context,
            outcome=outcome.disposition.value,
        )
        self._route_attempts.append(route)
        self._append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._next_event_index(),
                kind="route",
                observation_id=after.observation_id,
                proposal_id=proposal.proposal_id,
                proposal_work_item=proposal_work_item,
                route=route,
            )
        )

    def record_step(
        self,
        step: AgentStep,
        verification: PreDispatchVerification,
        before: Observation,
        after: Observation,
    ) -> None:
        """Record the observed route from one settled agent step."""
        if step.outcome is None:
            return
        self.record_route(
            step.step_index,
            step.proposal,
            verification.proposal_work_item,
            before,
            after,
            step.outcome,
        )

    def _next_event_index(self) -> int:
        self._event_index += 1
        return self._event_index

    def _append(self, event: RunProgressEvent) -> None:
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


__all__ = ["RunProgressEvent", "RunProgressLedger", "RunProgressPersistenceError"]
