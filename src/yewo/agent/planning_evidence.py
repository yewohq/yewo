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
    VERIFIED_ROUTE = "verified_route"
    DISCOVERY_CHECKPOINT = "discovery_checkpoint"


class PlannerDecisionMetadata(BaseModel):
    """Canonical provenance for one runtime planning decision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: PlannerDecisionSource
    reused_flow_id: RecordIdentifier | None = None
    reused_transition_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    reused_segment_index: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_reuse_provenance(self) -> Self:
        if self.source in {PlannerDecisionSource.MODEL, PlannerDecisionSource.SYSTEM}:
            if (
                self.reused_flow_id is not None
                or self.reused_transition_ids
                or self.reused_segment_index is not None
            ):
                raise ValueError("A model or system decision cannot claim stored-flow evidence.")
            return self
        if self.reused_flow_id is None:
            raise ValueError("A reused decision must identify its stored flow.")
        if bool(self.reused_transition_ids) != (self.reused_segment_index is not None):
            raise ValueError("Reused transition identifiers require a segment index.")
        return self


@runtime_checkable
class PlannerDecisionMetadataSource(Protocol):
    """Expose dynamic decision provenance after proposal selection."""

    def decision_metadata(self) -> PlannerDecisionMetadata: ...


def resolve_planner_decision_metadata(source: object) -> PlannerDecisionMetadata:
    """Return delegated decision evidence or the default model provenance."""
    if isinstance(source, PlannerDecisionMetadataSource):
        return source.decision_metadata()
    return PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL)


def guard_planner_decision(
    proposal: ActionProposal,
    decision: PlannerDecisionMetadata,
    identifier_factory: Callable[[], str],
) -> tuple[ActionProposal, PlannerDecisionMetadata]:
    """Stop before dispatch when retained-action provenance is incomplete."""
    if (
        decision.source in {
            PlannerDecisionSource.VERIFIED_ROUTE,
            PlannerDecisionSource.DISCOVERY_CHECKPOINT,
        }
        and proposal.action not in {ActionKind.DONE, ActionKind.CANNOT_PROGRESS}
        and not decision.reused_transition_ids
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
