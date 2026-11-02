"""Freshly grounded replay of YewoAgent's verified semantic routes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.composition import build_device_agent
from yewo.agent.discovery_checkpoint import (
    DiscoveryCheckpoint,
    checkpoint_segments,
)
from yewo.agent.runtime_records import (
    AgentModePolicy,
    AgentResult,
    AgentTask,
    AgentTerminalReason,
)
from yewo.agent.planning_evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionProposal,
)
from yewo.agent.planner import PlanningRequest, TaskProgressContext
from yewo.agent.replay_rejoin import (
    build_replay_action_proposal,
    replay_semantic_action,
    select_checkpoint_replay_segments,
)
from yewo.artifacts import RecordIdentifier, new_record_identifier
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient
from yewo.observations import Observation, RedactionRegion
from yewo.runs import InterruptionSignal, RunExecutionContext

from yewo.agent.app_memory import (
    AppMemory,
    RouteSegment,
    VerifiedRoute,
)
from yewo.agent.screen_memory import observation_matches_replay_variant


class RouteReplayTerminal(StrEnum):
    """Why one verified-route replay stopped."""

    COMPLETED = "completed"
    DIVERGED = "diverged"
    BUDGET_EXHAUSTED = "budget_exhausted"
    APPLICATION_FAILURE = "application_failure"
    DEVICE_REFUSED = "device_refused"


class ReplayPathKind(StrEnum):
    """The retained path used by one replay invocation."""

    VERIFIED_ROUTE = "verified_route"
    DISCOVERY_CHECKPOINT = "discovery_checkpoint"


class RouteReplayResult(BaseModel):
    """Outcome and canonical Device Agent evidence for one route replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_id: RecordIdentifier
    path_kind: ReplayPathKind
    terminal: RouteReplayTerminal
    completed_segment_count: int = Field(ge=0)
    failed_segment_index: int | None = Field(default=None, ge=0)
    replay_transition_ids: tuple[RecordIdentifier, ...] = Field(default=(), max_length=500)
    downstream_rejoin_transition_ids: tuple[RecordIdentifier, ...] = Field(
        default=(),
        max_length=500,
    )
    rejoin_segment_index: int | None = Field(default=None, ge=0)
    progress_context: TaskProgressContext | None = None
    reason: str = Field(min_length=1, max_length=4_000)
    agent_result: AgentResult

    @model_validator(mode="after")
    def _validate_replay_terminal(self) -> Self:
        if self.terminal is RouteReplayTerminal.COMPLETED:
            if self.failed_segment_index is not None:
                raise ValueError("Completed route replay cannot identify a failed segment.")
        elif self.failed_segment_index is None and self.terminal is RouteReplayTerminal.DIVERGED:
            raise ValueError("Diverged route replay must identify its failed segment.")
        if self.replay_transition_ids and self.completed_segment_count > len(
            self.replay_transition_ids
        ):
            raise ValueError("Replay completion exceeds the attempted retained path.")
        if self.rejoin_segment_index is not None and self.rejoin_segment_index > len(
            self.replay_transition_ids
        ):
            raise ValueError("Replay rejoin index exceeds the attempted retained path.")
        return self


class RouteActionSource:
    """Supply deterministic semantic actions from one retained path."""

    def __init__(
        self,
        memory: AppMemory,
        route: VerifiedRoute | DiscoveryCheckpoint,
        *,
        known_divergent_transition_id: RecordIdentifier | None = None,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        if isinstance(route, VerifiedRoute):
            routes = {item.route_id: item for item in memory.routes}
            if routes.get(route.route_id) != route:
                raise ValueError("Route replay requires a route from the supplied app memory.")
            segments = route.segments
            checkpoint = None
            decision_source = PlannerDecisionSource.VERIFIED_ROUTE
            path_id = route.route_id
        else:
            checkpoints = {item.checkpoint_id: item for item in memory.checkpoints}
            if checkpoints.get(route.checkpoint_id) != route:
                raise ValueError("Checkpoint replay requires a checkpoint from app memory.")
            segments = checkpoint_segments(memory, route)
            checkpoint = route
            decision_source = PlannerDecisionSource.DISCOVERY_CHECKPOINT
            path_id = route.checkpoint_id
        self._memory = memory
        self._checkpoint = checkpoint
        self._segments_resolved = checkpoint is None
        self._transitions = {item.transition_id: item for item in memory.transitions}
        self._variants = {
            variant.observation_id: variant
            for screen in memory.screens
            for variant in screen.variants
        }
        self._validate_segments(segments)
        self._segments = segments
        self._identifier_factory = identifier_factory
        self._next_segment_index = 0
        self._pending_proposal_id: str | None = None
        self._pending_segment_count = 0
        self._failed_segment_index: int | None = None
        self._rejoin_segment_index: int | None = None
        self._downstream_rejoin_transition_ids: tuple[RecordIdentifier, ...] = ()
        self._decision_source = decision_source
        self._path_id = path_id
        self._known_divergent_transition_id = known_divergent_transition_id
        if known_divergent_transition_id is not None and known_divergent_transition_id not in {
            segment.transition_id for segment in segments
        }:
            raise ValueError("A known divergent transition must belong to the retained path.")
        self._last_metadata = PlannerDecisionMetadata(
            source=decision_source,
            reused_flow_id=path_id,
        )
        self._progress_context: TaskProgressContext | None = None

    @property
    def completed_segment_count(self) -> int:
        """Return the number of destination guards already verified."""
        return self._next_segment_index

    @property
    def failed_segment_index(self) -> int | None:
        """Return the zero-based segment that could not continue."""
        return self._failed_segment_index

    @property
    def rejoin_segment_index(self) -> int | None:
        """Return the first downstream segment eligible after divergence."""
        return self._rejoin_segment_index

    @property
    def replay_transition_ids(self) -> tuple[RecordIdentifier, ...]:
        """Return the retained transitions selected for this replay."""
        return tuple(segment.transition_id for segment in self._segments)

    @property
    def downstream_rejoin_transition_ids(self) -> tuple[RecordIdentifier, ...]:
        """Return a disconnected checkpoint suffix retained for repair rejoin."""
        return self._downstream_rejoin_transition_ids

    @property
    def progress_context(self) -> TaskProgressContext | None:
        """Return the reconciliation applied before checkpoint replay."""
        return self._progress_context

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Return the retained evidence behind the latest proposal."""
        return self._last_metadata

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Count no model request for deterministic route replay."""
        del request
        return 0

    def plan(self, request: PlanningRequest) -> ActionProposal:
        """Return the next stored action only after its current guard matches."""
        if self._progress_context is None:
            self._progress_context = request.progress_context
        self._resolve_checkpoint_segments(request.observation, request.progress_context)
        if self._pending_proposal_id is not None:
            failure = self._settle_pending(request)
            if failure is not None:
                return failure
        if self._next_segment_index >= len(self._segments):
            if self._downstream_rejoin_transition_ids:
                return self._diverged(
                    request,
                    "No accepted approach reaches the retained checkpoint suffix.",
                )
            return self._terminal(
                request,
                ActionKind.DONE,
                "Every stored path segment reached its destination guard.",
            )
        segment = self._segments[self._next_segment_index]
        transition = self._transitions[segment.transition_id]
        source_variant = self._variants[transition.source_observation_id]
        if not observation_matches_replay_variant(
            request.observation,
            source_variant,
            self._memory.compatibility.screen_comparison,
        ):
            return self._diverged(
                request,
                "The current observation does not match the route source guard.",
            )
        if segment.transition_id == self._known_divergent_transition_id:
            # Skip only the transition with repeated divergence evidence. A collapsed partner
            # remains eligible only if repair later establishes its own source guard.
            return self._diverged(
                request,
                "Prior replay evidence repeatedly diverged at this retained transition.",
                rejoin_segment_offset=1,
            )
        action, segment_count = replay_semantic_action(self._segments, self._next_segment_index)
        if action.action not in request.available_actions:
            return self._diverged(
                request,
                "The next route action is unavailable under the active policy.",
            )
        proposal_id = self._identifier_factory()
        segment_index = self._next_segment_index
        self._pending_proposal_id = proposal_id
        self._pending_segment_count = segment_count
        self._last_metadata = PlannerDecisionMetadata(
            source=self._decision_source,
            reused_flow_id=self._path_id,
            reused_transition_ids=tuple(
                segment.transition_id
                for segment in self._segments[segment_index : segment_index + segment_count]
            ),
            reused_segment_index=segment_index,
        )
        return build_replay_action_proposal(
            action,
            proposal_id,
            request.observation.observation_id,
        )

    def _settle_pending(self, request: PlanningRequest) -> ActionProposal | None:
        if not request.history:
            return self._diverged(
                request,
                "Route action feedback is unavailable.",
                rejoin_segment_offset=self._pending_segment_count,
            )
        feedback = request.history[-1]
        if feedback.proposal.proposal_id != self._pending_proposal_id:
            return self._diverged(
                request,
                "Route action feedback does not match the route.",
                rejoin_segment_offset=self._pending_segment_count,
            )
        if feedback.refusal is not None:
            return self._diverged(
                request,
                feedback.refusal.reason,
                rejoin_segment_offset=self._pending_segment_count,
            )
        outcome = feedback.outcome
        if outcome is None or outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
            return self._diverged(
                request,
                "The route action did not produce accepted observable evidence.",
                rejoin_segment_offset=self._pending_segment_count,
            )
        destination_index = self._next_segment_index + self._pending_segment_count - 1
        segment = self._segments[destination_index]
        transition = self._transitions[segment.transition_id]
        destination_variant = self._variants[transition.destination_observation_id]
        if not observation_matches_replay_variant(
            request.observation,
            destination_variant,
            self._memory.compatibility.screen_comparison,
        ):
            return self._diverged(
                request,
                "The current observation does not match the route destination guard.",
                rejoin_segment_offset=self._pending_segment_count,
            )
        self._next_segment_index += self._pending_segment_count
        self._pending_proposal_id = None
        self._pending_segment_count = 0
        return None

    def _diverged(
        self,
        request: PlanningRequest,
        reason: str,
        *,
        rejoin_segment_offset: int = 0,
    ) -> ActionProposal:
        self._failed_segment_index = self._next_segment_index
        self._rejoin_segment_index = self._next_segment_index + rejoin_segment_offset
        return self._terminal(request, ActionKind.CANNOT_PROGRESS, reason)

    def _resolve_checkpoint_segments(
        self,
        observation: Observation,
        progress_context: TaskProgressContext | None,
    ) -> None:
        if self._segments_resolved or self._checkpoint is None:
            return
        selection = select_checkpoint_replay_segments(
            self._memory,
            self._checkpoint,
            observation,
            allow_declared_values=False,
            target_transition_id=(
                progress_context.replay_target_transition_id
                if progress_context is not None
                else None
            ),
        )
        self._segments = selection.replay_segments
        self._downstream_rejoin_transition_ids = tuple(
            segment.transition_id for segment in selection.downstream_segments
        )
        self._segments_resolved = True

    def _validate_segments(self, segments: Sequence[RouteSegment]) -> None:
        for segment in segments:
            transition = self._transitions.get(segment.transition_id)
            if transition is None or (
                transition.source_screen_id != segment.source_screen_id
                or transition.destination_screen_id != segment.destination_screen_id
                or transition.action != segment.action
            ):
                raise ValueError("Route replay segment evidence does not match app memory.")

    def _terminal(
        self,
        request: PlanningRequest,
        action: ActionKind,
        reason: str,
    ) -> ActionProposal:
        self._last_metadata = PlannerDecisionMetadata(
            source=self._decision_source,
            reused_flow_id=self._path_id,
        )
        return ActionProposal(
            proposal_id=self._identifier_factory(),
            source_observation_id=request.observation.observation_id,
            action=action,
            reason=reason,
        )

def replay_route(
    context: RunExecutionContext,
    memory: AppMemory,
    route: VerifiedRoute,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    interruption: InterruptionSignal,
    *,
    recovery_profile_id: str | None = None,
    action_verifier_profile_id: str | None = None,
    redaction_regions: Sequence[RedactionRegion] = (),
    recover_application: bool = False,
    declared_value_resolver: Callable[[str], str] | None = None,
) -> RouteReplayResult:
    """Replay one route through the same runtime used by normal agent tasks."""
    return _replay_path(
        context,
        memory,
        route,
        mode_policy,
        model_client,
        grounding_profile,
        interruption,
        redaction_regions,
        recover_application,
        declared_value_resolver,
        recovery_profile_id=recovery_profile_id,
        action_verifier_profile_id=action_verifier_profile_id,
    )

def replay_discovery_checkpoint(
    context: RunExecutionContext,
    memory: AppMemory,
    checkpoint: DiscoveryCheckpoint,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    interruption: InterruptionSignal,
    *,
    planner_profile_id: str,
    recovery_profile_id: str | None = None,
    action_verifier_profile_id: str | None = None,
    objective_id: RecordIdentifier,
    task: AgentTask,
    redaction_regions: Sequence[RedactionRegion] = (),
    recover_application: bool = False,
    declared_value_resolver: Callable[[str], str] | None = None,
    known_divergent_transition_id: RecordIdentifier | None = None,
) -> RouteReplayResult:
    """Replay accepted incomplete discovery evidence without claiming success."""
    return _replay_path(
        context,
        memory,
        checkpoint,
        mode_policy,
        model_client,
        grounding_profile,
        interruption,
        redaction_regions,
        recover_application,
        declared_value_resolver,
        planner_profile_id=planner_profile_id,
        recovery_profile_id=recovery_profile_id,
        action_verifier_profile_id=action_verifier_profile_id,
        objective_id=objective_id,
        reconciliation_task=task,
        known_divergent_transition_id=known_divergent_transition_id,
    )


def _replay_path(
    context: RunExecutionContext,
    memory: AppMemory,
    path: VerifiedRoute | DiscoveryCheckpoint,
    mode_policy: AgentModePolicy,
    model_client: ModelClient,
    grounding_profile: GroundingProfile,
    interruption: InterruptionSignal,
    redaction_regions: Sequence[RedactionRegion],
    recover_application: bool,
    declared_value_resolver: Callable[[str], str] | None,
    *,
    planner_profile_id: str | None = None,
    recovery_profile_id: str | None = None,
    action_verifier_profile_id: str | None = None,
    objective_id: RecordIdentifier | None = None,
    reconciliation_task: AgentTask | None = None,
    known_divergent_transition_id: RecordIdentifier | None = None,
) -> RouteReplayResult:
    segments = (
        path.segments if isinstance(path, VerifiedRoute) else checkpoint_segments(memory, path)
    )
    required_value_names = tuple(
        dict.fromkeys(
            segment.action.declared_value_name
            for segment in segments
            if segment.action.declared_value_name is not None
        )
    )
    if required_value_names and declared_value_resolver is None:
        raise ValueError("This stored path requires declared-value resolution.")
    action_source = RouteActionSource(
        memory,
        path,
        known_divergent_transition_id=known_divergent_transition_id,
    )
    planner_decision_source = (
        PlannerDecisionSource.VERIFIED_ROUTE
        if isinstance(path, VerifiedRoute)
        else PlannerDecisionSource.DISCOVERY_CHECKPOINT
    )
    reused_flow_id = path.route_id if isinstance(path, VerifiedRoute) else path.checkpoint_id
    reconcile_checkpoint = isinstance(path, DiscoveryCheckpoint)
    agent_result = build_device_agent(
        context,
        model_client,
        grounding_profile,
        planner_profile_id if reconcile_checkpoint else None,
        recovery_profile_id=recovery_profile_id,
        action_verifier_profile_id=action_verifier_profile_id,
        redaction_regions=redaction_regions,
        proposal_source=action_source,
        planning_memory=memory if reconcile_checkpoint else None,
        planning_memory_objective_id=objective_id if reconcile_checkpoint else None,
        grounding_memory=memory,
        recover_application=recover_application,
        declared_value_resolver=declared_value_resolver,
        screen_comparison=memory.compatibility.screen_comparison,
        planner_decision_source=planner_decision_source,
        reused_flow_id=reused_flow_id,
    ).run(
        reconciliation_task
        if reconcile_checkpoint and reconciliation_task is not None
        else AgentTask(
            objective=f"Replay: {path.objective}",
            known_facts=("Use only the supplied stored path mechanics.",),
            completion_conditions=("Every route destination guard matches.",),
            declared_value_names=required_value_names,
            targeted_type_available=any(
                transition.action.input_contract is not None for transition in segments
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
        path_id=(path.route_id if isinstance(path, VerifiedRoute) else path.checkpoint_id),
        path_kind=(
            ReplayPathKind.VERIFIED_ROUTE
            if isinstance(path, VerifiedRoute)
            else ReplayPathKind.DISCOVERY_CHECKPOINT
        ),
        terminal=terminal,
        completed_segment_count=action_source.completed_segment_count,
        failed_segment_index=failed_index,
        replay_transition_ids=action_source.replay_transition_ids,
        downstream_rejoin_transition_ids=(action_source.downstream_rejoin_transition_ids),
        rejoin_segment_index=(
            action_source.rejoin_segment_index if terminal is RouteReplayTerminal.DIVERGED else None
        ),
        progress_context=action_source.progress_context,
        reason=agent_result.reason,
        agent_result=agent_result,
    )
def _replay_terminal(terminal: AgentTerminalReason) -> RouteReplayTerminal:
    return {
        AgentTerminalReason.DONE_REQUESTED: RouteReplayTerminal.COMPLETED,
        AgentTerminalReason.CANNOT_PROGRESS: RouteReplayTerminal.DIVERGED,
        AgentTerminalReason.BUDGET_EXHAUSTED: RouteReplayTerminal.BUDGET_EXHAUSTED,
        AgentTerminalReason.APPLICATION_EXITED: RouteReplayTerminal.APPLICATION_FAILURE,
        AgentTerminalReason.APPLICATION_INACTIVE: RouteReplayTerminal.APPLICATION_FAILURE,
        AgentTerminalReason.DEVICE_REFUSED: RouteReplayTerminal.DEVICE_REFUSED,
    }[terminal]
