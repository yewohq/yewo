"""Promote accepted discovery evidence into reusable YewoAgent memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import Mapping

from pydantic import ValidationError

from yewo.agent.runtime_records import AgentResult, AgentStep, AgentTerminalReason
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    DeclaredValueReference,
    DispatchDisposition,
)
from yewo.artifacts import RecordIdentifier
from yewo.grounding import GroundedPoint
from yewo.observations import Observation
from yewo.agent.app_memory import (
    AcceptedTransition,
    AppMemory,
    CapabilityEvidence,
    InputContract,
    InputValueShape,
    LogicalScreen,
    MemoryCompatibility,
    RouteSegment,
    SemanticRouteAction,
    UnsuccessfulAttempt,
    UnsuccessfulAttemptDisposition,
    VerifiedRoute,
    derived_memory_identifier,
)
from yewo.agent.screen_memory import add_observation_screen, validate_memory_observations
from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
)


@dataclass(frozen=True)
class DiscoveryPromotion:
    """Reusable evidence and truthful objective coverage from one agent run."""

    memory: AppMemory | None
    coverage: DiscoveryCoverage


class AppMemoryBuilder:
    """Build memory only from internally consistent canonical step evidence."""

    def promote(
        self,
        request: DiscoveryRequest,
        agent_result: AgentResult,
        observations: Mapping[str, Observation],
        compatibility: MemoryCompatibility,
        *,
        verified_objective_ids: frozenset[RecordIdentifier],
        source_run_id: RecordIdentifier,
        updated_at: datetime,
        existing: AppMemory | None = None,
    ) -> DiscoveryPromotion:
        """Return accepted memory and coverage without upgrading ambiguous evidence."""
        if existing is not None and existing.compatibility != compatibility:
            raise ValueError("Existing app memory is incompatible with this discovery run.")
        requested_objective_ids = {objective.objective_id for objective in request.objectives}
        if not verified_objective_ids.issubset(requested_objective_ids):
            raise ValueError("Verification names an objective outside this discovery request.")
        validate_memory_observations(observations, compatibility)
        screens = list(existing.screens if existing is not None else ())
        transitions = list(existing.transitions if existing is not None else ())
        unsuccessful_attempts = list(existing.unsuccessful_attempts if existing is not None else ())
        run_transitions: list[AcceptedTransition] = []
        new_unsuccessful_attempts: list[UnsuccessfulAttempt] = []

        for step in agent_result.steps:
            unsuccessful_attempt = _unsuccessful_attempt(
                step,
                observations,
                screens,
                source_run_id,
                compatibility,
            )
            if unsuccessful_attempt is not None and not any(
                item.has_same_evidence(unsuccessful_attempt) for item in unsuccessful_attempts
            ):
                unsuccessful_attempts.append(unsuccessful_attempt)
                new_unsuccessful_attempts.append(unsuccessful_attempt)
            transition = self._accepted_transition(
                step,
                observations,
                screens,
                source_run_id,
                compatibility,
            )
            if transition is None:
                continue
            if run_transitions:
                transition = _add_input_contract(run_transitions[-1], transition)
            equivalent = next(
                (item for item in transitions if _same_semantic_transition(item, transition)),
                None,
            )
            if equivalent is None:
                transitions.append(transition)
            run_transitions.append(transition if equivalent is None else equivalent)

        routes = list(existing.routes if existing is not None else ())
        capabilities = list(existing.capabilities if existing is not None else ())
        coverage_entries: list[DiscoveryCoverageEntry] = []
        route = _verified_route(request, agent_result, run_transitions)
        for objective in request.objectives:
            entry, objective_route, capability = _objective_evidence(
                objective,
                agent_result,
                route,
                objective.objective_id in verified_objective_ids,
            )
            coverage_entries.append(entry)
            if objective_route is not None and objective_route.route_id not in {
                item.route_id for item in routes
            }:
                routes.append(objective_route)
            if capability is not None and capability.capability_id not in {
                item.capability_id for item in capabilities
            }:
                capabilities.append(capability)

        coverage = DiscoveryCoverage(entries=tuple(coverage_entries))
        if not run_transitions and not new_unsuccessful_attempts:
            return DiscoveryPromotion(memory=existing, coverage=coverage)
        source_run_ids = tuple(
            dict.fromkeys((*(existing.source_run_ids if existing else ()), source_run_id))
        )
        retained_transitions = _retain_referenced_transitions(
            transitions,
            routes,
            compatibility.retention.maximum_transitions,
        )
        maximum_attempts = compatibility.retention.maximum_unsuccessful_attempts
        memory = AppMemory(
            memory_id=(
                existing.memory_id
                if existing is not None
                else derived_memory_identifier("memory", compatibility.model_dump_json())
            ),
            compatibility=compatibility,
            source_run_ids=source_run_ids,
            screens=tuple(screens),
            transitions=retained_transitions,
            unsuccessful_attempts=tuple(unsuccessful_attempts[-maximum_attempts:]),
            routes=tuple(routes),
            capabilities=tuple(capabilities),
            updated_at=updated_at,
        )
        return DiscoveryPromotion(memory=memory, coverage=coverage)

    def _accepted_transition(
        self,
        step: AgentStep,
        observations: Mapping[str, Observation],
        screens: list[LogicalScreen],
        source_run_id: str,
        compatibility: MemoryCompatibility,
    ) -> AcceptedTransition | None:
        authorization = step.authorization
        receipt = step.receipt
        outcome = step.outcome
        if authorization is None or receipt is None or outcome is None:
            return None
        if (
            receipt.disposition is not DispatchDisposition.DISPATCHED
            or outcome.disposition is not ActionOutcomeDisposition.OBSERVED
            or outcome.signals
            & {
                ActionOutcomeSignal.APP_BACKGROUNDED,
                ActionOutcomeSignal.APP_EXITED,
            }
            or step.settled_observation_id is None
        ):
            return None
        source = observations.get(step.source_observation_id)
        destination = observations.get(step.settled_observation_id)
        if source is None or destination is None:
            raise ValueError("Accepted step evidence references an unavailable observation.")
        try:
            semantic_action = _semantic_action(step)
        except (TypeError, ValidationError, ValueError):
            return None
        source_screen = add_observation_screen(
            source,
            screens,
            source_run_id,
            compatibility,
        )
        destination_screen = add_observation_screen(
            destination,
            screens,
            source_run_id,
            compatibility,
        )
        if not _transition_has_independent_signal(
            outcome.signals,
            source_screen.screen_id,
            destination_screen.screen_id,
        ):
            return None
        return AcceptedTransition(
            transition_id=derived_memory_identifier(
                "transition",
                step.step_id,
                outcome.outcome_id,
            ),
            source_screen_id=source_screen.screen_id,
            destination_screen_id=destination_screen.screen_id,
            source_observation_id=source.observation_id,
            destination_observation_id=destination.observation_id,
            action=semantic_action,
            side_effect=authorization.side_effect,
            authorization_id=authorization.authorization_id,
            receipt_id=receipt.receipt_id,
            outcome_id=outcome.outcome_id,
            outcome_signals=outcome.signals,
            grounding_profile_ids=tuple(
                evidence.result.profile_id
                for evidence in (step.target_grounding, step.destination_grounding)
                if evidence is not None
            ),
            grounder_model_profile_ids=tuple(
                evidence.result.model_profile_id
                for evidence in (step.target_grounding, step.destination_grounding)
                if evidence is not None
            ),
            requested_grounder_models=tuple(
                evidence.result.requested_model
                for evidence in (step.target_grounding, step.destination_grounding)
                if evidence is not None
            ),
            source_step_id=step.step_id,
        )


def _unsuccessful_attempt(
    step: AgentStep,
    observations: Mapping[str, Observation],
    screens: list[LogicalScreen],
    source_run_id: str,
    compatibility: MemoryCompatibility,
) -> UnsuccessfulAttempt | None:
    receipt = step.receipt
    outcome = step.outcome
    if receipt is not None and receipt.disposition is DispatchDisposition.TARGET_REJECTED:
        disposition = UnsuccessfulAttemptDisposition.TARGET_REJECTED
        reason = receipt.reason
    elif (
        outcome is not None and outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
    ):
        disposition = UnsuccessfulAttemptDisposition.NO_OBSERVABLE_CHANGE
        reason = outcome.reason
    else:
        return None
    source = observations.get(step.source_observation_id)
    if source is None or reason is None:
        raise ValueError("Unsuccessful step evidence is incomplete.")
    screen = add_observation_screen(source, screens, source_run_id, compatibility)
    grounding = step.target_grounding
    grounding_result = grounding.result if grounding is not None else None
    grounded_point = None
    width_px = None
    height_px = None
    if grounding is not None and isinstance(grounding.result.location, GroundedPoint):
        grounded_point = grounding.result.location.pixel
        width_px = grounding.request.screenshot.width_px
        height_px = grounding.request.screenshot.height_px
    return UnsuccessfulAttempt(
        attempt_id=derived_memory_identifier(
            "unsuccessful-attempt", source_run_id, step.step_id, disposition
        ),
        screen_id=screen.screen_id,
        source_observation_id=source.observation_id,
        action=step.proposal.action,
        target_description=step.proposal.target_expression,
        destination_description=step.proposal.destination_expression,
        direction=step.proposal.direction,
        distance=step.proposal.distance,
        disposition=disposition,
        reason=reason,
        grounded_point=grounded_point,
        screenshot_width_px=width_px,
        screenshot_height_px=height_px,
        grounding_profile_id=grounding_result and grounding_result.profile_id,
        grounder_model_profile_id=grounding_result and grounding_result.model_profile_id,
        requested_grounder_model=grounding_result and grounding_result.requested_model,
        source_step_id=step.step_id,
        source_run_id=source_run_id,
    )


def _semantic_action(step: AgentStep) -> SemanticRouteAction:
    proposal = step.proposal
    declared_value_name = None
    disposable_text = None
    if proposal.text is not None:
        if isinstance(proposal.text, DeclaredValueReference):
            declared_value_name = proposal.text.value_name
        else:
            disposable_text = proposal.text.value
    if proposal.expected_outcome is None:
        raise ValueError("Reusable route actions require an expected outcome.")
    return SemanticRouteAction(
        action=proposal.action,
        target_description=proposal.target_expression,
        destination_description=proposal.destination_expression,
        expected_outcome=proposal.expected_outcome,
        direction=proposal.direction,
        distance=proposal.distance,
        duration_seconds=proposal.duration_seconds,
        declared_value_name=declared_value_name,
        disposable_text=disposable_text,
    )


def _add_input_contract(
    previous: AcceptedTransition,
    current: AcceptedTransition,
) -> AcceptedTransition:
    action = current.action
    if (
        previous.action.action is not ActionKind.FOCUS
        or action.action is not ActionKind.TYPE
        or action.disposable_text is None
        or previous.destination_screen_id != current.source_screen_id
        or previous.action.target_description is None
    ):
        return current
    value_shape, format_pattern = _input_shape(action.disposable_text)
    return current.model_copy(
        update={
            "action": action.model_copy(
                update={
                    "target_description": previous.action.target_description,
                    "input_contract": InputContract(
                        value_shape=value_shape,
                        format_pattern=format_pattern,
                    ),
                }
            )
        }
    )


def _input_shape(value: str) -> tuple[InputValueShape, str | None]:
    if re.fullmatch(r"[+-]?\d+", value):
        return InputValueShape.INTEGER, None
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\d*\.\d+)", value):
        return InputValueShape.DECIMAL, None
    if re.fullmatch(r"\d+(?:[^\w\s]+\d+)+", value):
        pattern = re.sub(r"\d", "#", value)
        return InputValueShape.FORMATTED, pattern
    return InputValueShape.TEXT, None


def _transition_has_independent_signal(
    signals: frozenset[ActionOutcomeSignal],
    source_screen_id: str,
    destination_screen_id: str,
) -> bool:
    meaningful = signals - {ActionOutcomeSignal.SCREEN_CHANGED}
    return bool(meaningful) or source_screen_id != destination_screen_id


def _same_semantic_transition(
    first: AcceptedTransition,
    second: AcceptedTransition,
) -> bool:
    return (
        first.source_screen_id == second.source_screen_id
        and first.destination_screen_id == second.destination_screen_id
        and first.action == second.action
        and first.outcome_signals == second.outcome_signals
    )


def _retain_referenced_transitions(
    transitions: list[AcceptedTransition],
    routes: list[VerifiedRoute],
    maximum: int,
) -> tuple[AcceptedTransition, ...]:
    required = {segment.transition_id for route in routes for segment in route.segments}
    retained = [item for item in transitions if item.transition_id in required]
    optional = [item for item in transitions if item.transition_id not in required]
    available = maximum - len(retained)
    if available < 0:
        raise ValueError("Verified routes exceed the transition retention limit.")
    return tuple((*retained, *optional[-available:])) if available else tuple(retained)


def _verified_route(
    request: DiscoveryRequest,
    agent_result: AgentResult,
    transitions: list[AcceptedTransition],
) -> VerifiedRoute | None:
    if agent_result.terminal_reason is not AgentTerminalReason.DONE_REQUESTED or not transitions:
        return None
    represented_step_ids = {transition.source_step_id for transition in transitions}
    for step in agent_result.steps:
        if step.receipt is None or step.receipt.disposition is not DispatchDisposition.DISPATCHED:
            continue
        if step.step_id in represented_step_ids:
            continue
        if (
            step.outcome is not None
            and step.outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
        ):
            continue
        return None
    if any(
        previous.destination_screen_id != current.source_screen_id
        for previous, current in zip(transitions, transitions[1:], strict=False)
    ):
        return None
    objective_text = " ".join(objective.statement for objective in request.objectives)
    transition_ids = tuple(transition.transition_id for transition in transitions)
    observations = (
        transitions[0].source_observation_id,
        *(transition.destination_observation_id for transition in transitions),
    )
    return VerifiedRoute(
        route_id=derived_memory_identifier("route", request.request_id, *transition_ids),
        objective_id=request.objectives[0].objective_id,
        objective=objective_text,
        start_screen_id=transitions[0].source_screen_id,
        destination_screen_id=transitions[-1].destination_screen_id,
        segments=tuple(
            RouteSegment(
                transition_id=transition.transition_id,
                source_screen_id=transition.source_screen_id,
                destination_screen_id=transition.destination_screen_id,
                action=transition.action,
            )
            for transition in transitions
        ),
        evidence_observation_ids=tuple(dict.fromkeys(observations)),
    )


def _objective_evidence(
    objective: DiscoveryObjective,
    agent_result: AgentResult,
    shared_route: VerifiedRoute | None,
    independently_verified: bool,
) -> tuple[DiscoveryCoverageEntry, VerifiedRoute | None, CapabilityEvidence | None]:
    if shared_route is not None and independently_verified:
        route = shared_route.model_copy(
            update={
                "route_id": derived_memory_identifier(
                    "route", objective.objective_id, shared_route.route_id
                ),
                "objective_id": objective.objective_id,
                "objective": objective.statement,
            }
        )
        capability = CapabilityEvidence(
            capability_id=derived_memory_identifier(
                "capability", objective.objective_id, route.route_id
            ),
            objective_id=objective.objective_id,
            statement=f"The application can reach: {objective.statement}",
            route_id=route.route_id,
            evidence_observation_ids=route.evidence_observation_ids,
        )
        return (
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.SUPPORTED,
                capability_id=capability.capability_id,
                route_id=route.route_id,
                evidence_observation_ids=route.evidence_observation_ids,
                reason="A contiguous accepted route reached the requested state.",
            ),
            route,
            capability,
        )
    if shared_route is not None:
        return (
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.UNRESOLVED,
                reason=(
                    "A reusable route was observed, but independent final-state verification "
                    "did not support the objective."
                ),
            ),
            shared_route.model_copy(
                update={
                    "route_id": derived_memory_identifier(
                        "route", objective.objective_id, shared_route.route_id
                    ),
                    "objective_id": objective.objective_id,
                    "objective": objective.statement,
                }
            ),
            None,
        )
    state = (
        DiscoveryObjectiveState.BLOCKED
        if agent_result.terminal_reason
        in {
            AgentTerminalReason.APPLICATION_EXITED,
            AgentTerminalReason.APPLICATION_INACTIVE,
            AgentTerminalReason.DEVICE_REFUSED,
        }
        else DiscoveryObjectiveState.UNRESOLVED
    )
    return (
        DiscoveryCoverageEntry(
            objective_id=objective.objective_id,
            state=state,
            reason=_coverage_reason(agent_result),
        ),
        None,
        None,
    )


def _coverage_reason(agent_result: AgentResult) -> str:
    return {
        AgentTerminalReason.DONE_REQUESTED: (
            "The planner requested completion without a contiguous reusable route."
        ),
        AgentTerminalReason.CANNOT_PROGRESS: (
            "The agent found no further permitted useful action. The objective remains unresolved."
        ),
        AgentTerminalReason.BUDGET_EXHAUSTED: (
            "The discovery budget ended. Evidence does not support the objective."
        ),
        AgentTerminalReason.APPLICATION_EXITED: "The application exited during discovery.",
        AgentTerminalReason.APPLICATION_INACTIVE: "The application became inactive during discovery.",
        AgentTerminalReason.DEVICE_REFUSED: "The device refused a discovery action.",
    }[agent_result.terminal_reason]
