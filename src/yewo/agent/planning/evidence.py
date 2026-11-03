"""Structured provenance for model and stored-flow planning decisions."""

from __future__ import annotations

from enum import StrEnum
from collections.abc import Callable
from typing import Protocol, Self, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.agent.actuation import ActionKind, ActionProposal


class PlannerDecisionSource(StrEnum):
    """The source of one runtime planning decision."""

    MODEL = "model"
    SYSTEM = "system"
    COMPILED_PROCEDURE = "compiled_procedure"


class PlannerDecisionMetadata(BaseModel):
    """Canonical provenance for one runtime planning decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: PlannerDecisionSource
    reused_flow_id: RecordIdentifier | None = None
    reused_procedure_step_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    reused_segment_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_reuse_provenance(self) -> Self:
        if self.source in {PlannerDecisionSource.MODEL, PlannerDecisionSource.SYSTEM}:
            if (
                self.reused_flow_id is not None
                or self.reused_procedure_step_ids
                or self.reused_segment_index is not None
            ):
                raise ValueError("A model or system decision cannot claim stored-flow evidence.")
            return self
        if self.reused_flow_id is None:
            raise ValueError("A reused decision must identify its stored flow.")
        if not self.reused_procedure_step_ids and self.reused_segment_index is None:
            return self
        if not self.reused_procedure_step_ids:
            raise ValueError("A compiled procedure decision requires procedure-step evidence.")
        if self.reused_segment_index is None:
            raise ValueError("A reused decision requires a segment index.")
        return self


@runtime_checkable
class PlannerDecisionMetadataSource(Protocol):
    """Expose dynamic decision provenance after proposal selection."""

    def decision_metadata(self) -> PlannerDecisionMetadata: ...


@runtime_checkable
class PlanningModelUsageSource(Protocol):
    """Expose actual model use when deterministic planning can avoid a reserved call."""

    def last_planning_model_request_count(self) -> int: ...


def resolve_planner_decision_metadata(source: object) -> PlannerDecisionMetadata:
    """Return delegated decision evidence or the default model provenance."""
    if isinstance(source, PlannerDecisionMetadataSource):
        return source.decision_metadata()
    return PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL)


def resolve_planning_model_request_count(source: object, reserved_count: int) -> int:
    """Return actual model calls after one planning operation."""
    if not isinstance(source, PlanningModelUsageSource):
        return reserved_count
    consumed_count = source.last_planning_model_request_count()
    if consumed_count < 0 or consumed_count > reserved_count:
        raise ValueError("Planning model use must remain within the reserved request count.")
    return consumed_count


def guard_planner_decision(
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    identifier_factory: Callable[[], str],
) -> tuple[ActionProposal, PlannerDecisionMetadata]:
    """Stop before dispatch when retained-action provenance is incomplete."""
    if (
        decision.source is PlannerDecisionSource.COMPILED_PROCEDURE
        and proposal.action not in {ActionKind.DONE, ActionKind.CANNOT_PROGRESS}
        and not decision.reused_procedure_step_ids
    ):
        return (
            ActionProposal(
                proposal_id=identifier_factory(),
                source_observation_id=proposal.source_observation_id,
                action=ActionKind.CANNOT_PROGRESS,
                reason="Planning stopped before dispatch because retained-action evidence is incomplete.",
            ),
            PlannerDecisionMetadata(source=PlannerDecisionSource.SYSTEM),
        )
    return proposal, decision
