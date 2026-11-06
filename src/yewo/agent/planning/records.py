"""Contracts for the evidence one planner decision consumes."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.artifacts import RecordIdentifier
from yewo.observations import Observation
from yewo.agent.planning.history import PlanningHistoryEntry, PlanningScreenFact
from yewo.agent.screen_analysis import CurrentScreenAnalysis
from yewo.agent.actuation import ActionKind
from yewo.agent.runtime.limits import (
    DEFAULT_MAXIMUM_STEPS,
    MAXIMUM_CREDITED_RECOVERY_STEPS,
    MAXIMUM_PHYSICAL_STEPS,
)

PlanningText = Annotated[str, Field(min_length=1, max_length=4_000)]
MAXIMUM_PLANNING_HISTORY_ENTRIES = MAXIMUM_PHYSICAL_STEPS
# One step records at most one attempt, refusal, or exhausted transition.
MAXIMUM_DEFAULT_RUN_STEPS = DEFAULT_MAXIMUM_STEPS + MAXIMUM_CREDITED_RECOVERY_STEPS
MAXIMUM_PROHIBITED_INTENTS = MAXIMUM_DEFAULT_RUN_STEPS
# One observed step can report both a repeated transition and a revisited screen.
MAXIMUM_SCREEN_FACTS = MAXIMUM_DEFAULT_RUN_STEPS * 2
MAXIMUM_RECOVERY_GUIDANCE = 20



class PlanningFocusCause(StrEnum):
    """Why a screen earned an isolated planning view."""

    SCREEN_NO_PROGRESS = "screen_no_progress"
    REPEATED_TRANSITION = "repeated_transition"


class TaskProgressState(StrEnum):
    """How reconciled evidence relates to one original work item."""

    CURRENT_PROGRESS = "current_progress"
    KNOWN_PROCEDURE = "known_procedure"
    UNRESOLVED = "unresolved"


class ProgressRouteAttempt(BaseModel):
    """One observed same-run route retained outside bounded planner history."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    step_index: int = Field(ge=1)
    work_item: PlanningText | None = None
    source_context: PlanningText | None = None
    source_screen_fingerprint: str | None = Field(default=None, min_length=16, max_length=16)
    action: ActionKind
    target: PlanningText | None = None
    interaction_intent_key: str | None = Field(default=None, min_length=1, max_length=4_000)
    destination_context: PlanningText | None = None
    destination_screen_fingerprint: str | None = Field(default=None, min_length=16, max_length=16)
    outcome: PlanningText


class PlanningFocusContext(BaseModel):
    """One bounded fresh planning view after deterministic no-progress evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    focus_id: RecordIdentifier
    cause: PlanningFocusCause
    reason: PlanningText
    unresolved_work: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    prohibited_interaction_intents: tuple[PlanningText, ...] = Field(
        default=(),
        max_length=MAXIMUM_PROHIBITED_INTENTS,
    )
    recent_routes: tuple[ProgressRouteAttempt, ...] = Field(default=(), max_length=12)


class TaskProgressContext(BaseModel):
    """Run-scoped memory guidance without changing the original task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_progress: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    known_procedures: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    completed_work_items: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    remaining_work: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    route_attempts: tuple[ProgressRouteAttempt, ...] = Field(default=(), max_length=100)
    historical_memory_facts: tuple[PlanningText, ...] = Field(default=(), max_length=20)
    exhausted_branches: tuple[PlanningText, ...] = Field(default=(), max_length=20)
    recovery_guidance: tuple[PlanningText, ...] = Field(
        default=(),
        max_length=MAXIMUM_RECOVERY_GUIDANCE,
    )
    preparation_planning_requests: int = Field(default=0, ge=0, exclude=True)


class PlanningRequest(BaseModel):
    """Current evidence and task state needed to select one action."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    observation: Observation
    current_screen_analysis: CurrentScreenAnalysis | None = None
    objective: PlanningText
    known_facts: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    screen_facts: tuple[PlanningScreenFact, ...] = Field(
        default=(),
        max_length=MAXIMUM_SCREEN_FACTS,
    )
    open_work: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    completion_conditions: tuple[PlanningText, ...] = Field(default=(), max_length=100)
    declared_value_names: tuple[RecordIdentifier, ...] = Field(default=(), max_length=100)
    history: tuple[PlanningHistoryEntry, ...] = Field(
        default=(), max_length=MAXIMUM_PLANNING_HISTORY_ENTRIES
    )
    available_actions: frozenset[ActionKind] = Field(min_length=1)
    progress_context: TaskProgressContext | None = None
    planning_focus: PlanningFocusContext | None = None
    discovery_mode: bool = False

    @model_validator(mode="after")
    def _validate_unique_declared_values(self) -> Self:
        if len(set(self.declared_value_names)) != len(self.declared_value_names):
            raise ValueError("Declared value names must be unique.")
        if self.current_screen_analysis is not None and not self.current_screen_analysis.matches(
            self.observation
        ):
            raise ValueError("Current screen analysis must match the planning observation.")
        return self


def planning_progress_context(context: TaskProgressContext) -> dict[str, object]:
    """Project progress without duplicated work or runtime-only route fields."""
    projected = context.model_dump(
        mode="json",
        exclude={"remaining_work", "route_attempts"},
    )
    projected["route_attempts"] = tuple(
        _planning_route_context(route) for route in context.route_attempts
    )
    return projected


def _planning_route_context(route: ProgressRouteAttempt) -> dict[str, object]:
    """Keep the work-linked route evidence used by model prompts."""
    return {
        key: value
        for key, value in {
            "work_item": route.work_item,
            "source_context": route.source_context,
            "action": route.action.value,
            "target": route.target,
            "destination_context": route.destination_context,
            "outcome": route.outcome,
        }.items()
        if value is not None
    }


def planning_focus_context(context: PlanningFocusContext) -> dict[str, object]:
    """Project focus evidence through the standard route boundary."""
    projected = context.model_dump(mode="json", exclude={"recent_routes"})
    projected["recent_routes"] = tuple(
        _planning_route_context(route) for route in context.recent_routes
    )
    return projected
