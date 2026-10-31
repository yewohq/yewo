"""Freshly grounded replay of YewoAgent's verified semantic routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.composition import build_device_agent
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentResult,
    AgentTask,
    AgentTerminalReason,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionProposal,
    DeclaredValueReference,
    LiteralText,
)
from yewo.agent.planner import PlanningRequest
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient
from yewo.observations import RedactionRegion
from yewo.runs import InterruptionSignal, RunExecutionContext

from yewo.agent.app_memory import (
    AppMemory,
    CapabilityEvidence,
    RouteSegment,
    SemanticRouteAction,
    VerifiedRoute,
    derived_memory_identifier,
)
from yewo.agent.screen_memory import observation_matches_screen


class RouteReplayTerminal(StrEnum):
    """Why one verified-route replay stopped."""

    COMPLETED = "completed"
    DIVERGED = "diverged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    APPLICATION_FAILURE = "application_failure"
    DEVICE_REFUSED = "device_refused"


class RouteReplayResult(BaseModel):
    """Outcome and canonical Device Agent evidence for one route replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    route_id: RecordIdentifier
    terminal: RouteReplayTerminal
    completed_segment_count: int = Field(ge=0)
    failed_segment_index: int | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1, max_length=4_000)
    agent_result: AgentResult

    @model_validator(mode="after")
    def _validate_replay_terminal(self) -> Self:
        if self.terminal is RouteReplayTerminal.COMPLETED:
            if self.failed_segment_index is not None:
                raise ValueError("Completed route replay cannot identify a failed segment.")
        elif self.failed_segment_index is None and self.terminal is RouteReplayTerminal.DIVERGED:
            raise ValueError("Diverged route replay must identify its failed segment.")
        return self


class RouteActionSource:
    """Supply deterministic semantic actions from one verified route."""

    def __init__(
        self,
        memory: AppMemory,
        route: VerifiedRoute,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        routes = {item.route_id: item for item in memory.routes}
        if routes.get(route.route_id) != route:
            raise ValueError("Route replay requires a route from the supplied app memory.")
        transitions = {item.transition_id: item for item in memory.transitions}
        for segment in route.segments:
            transition = transitions.get(segment.transition_id)
            if transition is None or (
                transition.source_screen_id != segment.source_screen_id
                or transition.destination_screen_id != segment.destination_screen_id
                or transition.action != segment.action
            ):
                raise ValueError("Route replay segment evidence does not match app memory.")
        self._route = route
        self._screens = {screen.screen_id: screen for screen in memory.screens}
        self._screen_comparison = memory.compatibility.screen_comparison
        self._identifier_factory = identifier_factory
        self._next_segment_index = 0
        self._pending_proposal_id: str | None = None
        self._pending_segment_count = 0
        self._failed_segment_index: int | None = None

    @property
    def completed_segment_count(self) -> int:
        """Return the number of destination guards already verified."""
        return self._next_segment_index

    @property
    def failed_segment_index(self) -> int | None:
        """Return the zero-based segment that could not continue."""
        return self._failed_segment_index

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Return the next stored action only after its current guard matches."""
        if self._pending_proposal_id is not None:
            failure = self._settle_pending(request)
            if failure is not None:
                return failure
        if self._next_segment_index >= len(self._route.segments):
            return self._terminal(
                request,
                ActionKind.DONE,
                "Every verified route segment reached its destination guard.",
            )
        segment = self._route.segments[self._next_segment_index]
        source_screen = self._screens[segment.source_screen_id]
        if not observation_matches_screen(
            request.observation,
            source_screen,
            self._screen_comparison,
        ):
            return self._diverged(
                request,
                "The current observation does not match the route source guard.",
            )
        action, segment_count = self._replay_action(segment)
        if action.action not in request.available_actions:
            return self._diverged(
                request,
                "The next route action is unavailable under the active policy.",
            )
        proposal_id = self._identifier_factory()
        self._pending_proposal_id = proposal_id
        self._pending_segment_count = segment_count
        return _action_proposal(
            action,
            proposal_id,
            request.observation.observation_id,
        )

    def _settle_pending(self, request: PlanningRequest) -> ActionProposal | None:
        if not request.history:
            return self._diverged(request, "Route action feedback is unavailable.")
        feedback = request.history[-1]
        if feedback.proposal.proposal_id != self._pending_proposal_id:
            return self._diverged(request, "Route action feedback does not match the route.")
        if feedback.refusal is not None:
            return self._diverged(request, feedback.refusal.reason)
        outcome = feedback.outcome
        if outcome is None or outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
            return self._diverged(
                request,
                "The route action did not produce accepted observable evidence.",
            )
        destination_index = self._next_segment_index + self._pending_segment_count - 1
        segment = self._route.segments[destination_index]
        destination = self._screens[segment.destination_screen_id]
        if not observation_matches_screen(
            request.observation,
            destination,
            self._screen_comparison,
        ):
            return self._diverged(
                request,
                "The current observation does not match the route destination guard.",
            )
        self._next_segment_index += self._pending_segment_count
        self._pending_proposal_id = None
        self._pending_segment_count = 0
        return None

    def _replay_action(self, segment: RouteSegment) -> tuple[SemanticRouteAction, int]:
        next_index = self._next_segment_index + 1
        if segment.action.action is not ActionKind.FOCUS or next_index >= len(self._route.segments):
            return segment.action, 1
        next_segment = self._route.segments[next_index]
        if (
            next_segment.action.action is ActionKind.TYPE
            and next_segment.action.input_contract is not None
            and next_segment.action.target_description == segment.action.target_description
            and segment.destination_screen_id == next_segment.source_screen_id
        ):
            return next_segment.action, 2
        return segment.action, 1

    def _diverged(self, request: PlanningRequest, reason: str) -> ActionProposal:
        self._failed_segment_index = self._next_segment_index
        return self._terminal(request, ActionKind.CANNOT_PROGRESS, reason)

    def _terminal(
        self,
        request: PlanningRequest,
        action: ActionKind,
        reason: str,
    ) -> ActionProposal:
        return ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=action,
            reason=reason,
        )


def combine_repaired_route(
    memory: AppMemory,
    original: VerifiedRoute,
    replay: RouteReplayResult,
    repaired_route_id: str | None,
) -> tuple[AppMemory, VerifiedRoute, CapabilityEvidence] | None:
    """Combine a verified replay prefix with one independently verified repair."""
    if replay.completed_segment_count == 0 or repaired_route_id is None:
        return None
    repaired = next(
        (route for route in memory.routes if route.route_id == repaired_route_id),
        None,
    )
    prefix = original.segments[: replay.completed_segment_count]
    if (
        repaired is None
        or not prefix
        or prefix[-1].destination_screen_id != repaired.start_screen_id
    ):
        return None
    segments = (*prefix, *repaired.segments)
    transitions = {item.transition_id: item for item in memory.transitions}
    observation_ids = (
        transitions[prefix[0].transition_id].source_observation_id,
        *(transitions[item.transition_id].destination_observation_id for item in segments),
    )
    combined = VerifiedRoute(
        route_id=derived_memory_identifier(
            "repaired-route",
            original.route_id,
            repaired.route_id,
            *(item.transition_id for item in segments),
        ),
        objective_id=repaired.objective_id,
        objective=repaired.objective,
        start_screen_id=original.start_screen_id,
        destination_screen_id=repaired.destination_screen_id,
        segments=segments,
        evidence_observation_ids=tuple(dict.fromkeys(observation_ids)),
    )
    capability = CapabilityEvidence(
        capability_id=derived_memory_identifier(
            "repaired-capability",
            repaired.objective_id,
            combined.route_id,
        ),
        objective_id=repaired.objective_id,
        statement=f"The application can reach: {repaired.objective}",
        route_id=combined.route_id,
        evidence_observation_ids=combined.evidence_observation_ids,
    )
    updated = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "routes": (*memory.routes, combined),
            "capabilities": (*memory.capabilities, capability),
        }
    )
    return updated, combined, capability


def replay_route(
    context: RunExecutionContext,
    memory: AppMemory,
    route: VerifiedRoute,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    interruption: InterruptionSignal,
    *,
    redaction_regions: Sequence[RedactionRegion] = (),
    recover_application: bool = False,
    declared_value_resolver: Callable[[str], str] | None = None,
) -> RouteReplayResult:
    """Replay one route through the same runtime used by normal agent tasks."""
    required_value_names = tuple(
        dict.fromkeys(
            segment.action.declared_value_name
            for segment in route.segments
            if segment.action.declared_value_name is not None
        )
    )
    if required_value_names and declared_value_resolver is None:
        raise ValueError("This verified route requires declared-value resolution.")
    action_source = RouteActionSource(memory, route)
    agent_result = build_device_agent(
        context,
        model_client,
        grounding_profile,
        None,
        redaction_regions=redaction_regions,
        proposal_source=action_source,
        recover_application=recover_application,
        declared_value_resolver=declared_value_resolver,
        screen_comparison=memory.compatibility.screen_comparison,
    ).run(
        AgentTask(
            objective=f"Replay the verified route for: {route.objective}",
            known_facts=("Use only the supplied verified route mechanics.",),
            completion_conditions=("Every route destination guard matches.",),
            declared_value_names=required_value_names,
            targeted_type_available=any(
                segment.action.input_contract is not None for segment in route.segments
            ),
        ),
        mode_policy,
        interruption,
    )
    terminal = _replay_terminal(agent_result.terminal_reason)
    failed_index = (
        action_source.failed_segment_index if terminal is RouteReplayTerminal.DIVERGED else None
    )
    return RouteReplayResult(
        route_id=route.route_id,
        terminal=terminal,
        completed_segment_count=action_source.completed_segment_count,
        failed_segment_index=failed_index,
        reason=agent_result.reason,
        agent_result=agent_result,
    )


def _action_proposal(
    action: SemanticRouteAction,
    proposal_id: str,
    observation_id: str,
) -> ActionProposal:
    return ActionProposal(
        proposal_id=proposal_id,
        source_observation_id=observation_id,
        action=action.action,
        target_expression=action.target_description,
        destination_expression=action.destination_description,
        expected_outcome=action.expected_outcome,
        reason="Follow the independently verified route toward its destination.",
        direction=action.direction,
        distance=action.distance,
        duration_seconds=action.duration_seconds,
        text=_replay_text(action),
    )


def _replay_text(
    action: SemanticRouteAction,
) -> DeclaredValueReference | LiteralText | None:
    if action.declared_value_name is not None:
        return DeclaredValueReference(value_name=action.declared_value_name)
    if action.disposable_text is not None:
        return LiteralText(value=action.disposable_text)
    return None


def _replay_terminal(terminal: AgentTerminalReason) -> RouteReplayTerminal:
    return {
        AgentTerminalReason.DONE_REQUESTED: RouteReplayTerminal.COMPLETED,
        AgentTerminalReason.CANNOT_PROGRESS: RouteReplayTerminal.DIVERGED,
        AgentTerminalReason.BUDGET_EXHAUSTED: RouteReplayTerminal.BUDGET_EXHAUSTED,
        AgentTerminalReason.APPLICATION_EXITED: RouteReplayTerminal.APPLICATION_FAILURE,
        AgentTerminalReason.APPLICATION_INACTIVE: RouteReplayTerminal.APPLICATION_FAILURE,
        AgentTerminalReason.DEVICE_REFUSED: RouteReplayTerminal.DEVICE_REFUSED,
    }[terminal]
