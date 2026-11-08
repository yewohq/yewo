"""Promote accepted discovery evidence into reusable YewoAgent memory."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from pydantic import ValidationError

from yewo.agent.runtime.records import AgentResult, AgentStep
from yewo.agent.memory.grounding import AcceptedGrounding
from yewo.agent.memory.knowledge import AppKnowledge
from yewo.agent.memory.input_contracts import build_literal_input_contract
from yewo.agent.memory.retention import retain_transitions
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    DeclaredValueReference,
    DispatchDisposition,
)
from yewo.artifacts import RecordIdentifier
from yewo.grounding import GroundedPoint, GroundingEvidence
from yewo.observations import Observation
from yewo.agent.memory.records import (
    AcceptedTransition,
    AppMemory,
    LogicalScreen,
    MemoryCompatibility,
    SemanticRouteAction,
    UnsuccessfulAttempt,
    UnsuccessfulAttemptDisposition,
    derived_memory_identifier,
)
from yewo.agent.memory.screens import (
    add_observation_screen,
    retain_observation_variants,
    validate_memory_observations,
)
from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjectiveState,
    DiscoveryRequest,
)
from yewo.agent.discovery.coverage import build_objective_evidence
from yewo.agent.replay.reconciliation.records import ReplayProcedure


@dataclass(frozen=True)
class DiscoveryPromotion:
    """Reusable evidence and truthful objective coverage from one agent run."""

    memory: AppMemory | None
    coverage: DiscoveryCoverage


def _build_discovery_coverage(
    request: DiscoveryRequest,
    agent_result: AgentResult,
    verified_objective_ids: frozenset[RecordIdentifier] | None,
    compiled_procedure: ReplayProcedure | None,
    evidence_observation_ids: tuple[RecordIdentifier, ...],
) -> DiscoveryCoverage:
    if verified_objective_ids is None:
        entries = tuple(
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.UNRESOLVED,
                reason="The run did not independently verify objective coverage.",
            )
            for objective in request.objectives
        )
    else:
        entries = tuple(
            build_objective_evidence(
                objective,
                agent_result,
                compiled_procedure,
                evidence_observation_ids,
                objective.objective_id in verified_objective_ids,
            )
            for objective in request.objectives
        )
    return DiscoveryCoverage(entries=entries)


def _merge_transition(
    transitions: list[AcceptedTransition],
    candidate: AcceptedTransition,
) -> AcceptedTransition:
    equivalent_index = next(
        (
            index
            for index, transition in enumerate(transitions)
            if _same_semantic_transition(transition, candidate)
        ),
        None,
    )
    if equivalent_index is None:
        transitions.append(candidate)
        return candidate
    retained = transitions[equivalent_index]
    merged = retained.model_copy(
        update={
            "target_grounding": retained.target_grounding or candidate.target_grounding,
            "destination_grounding": (
                retained.destination_grounding or candidate.destination_grounding
            ),
        }
    )
    transitions[equivalent_index] = merged
    return merged


class AppMemoryBuilder:
    """Build memory only from internally consistent canonical step evidence."""

    def promote(
        self,
        request: DiscoveryRequest,
        agent_result: AgentResult,
        observations: Mapping[str, Observation],
        compatibility: MemoryCompatibility,
        *,
        verified_objective_ids: frozenset[RecordIdentifier] | None,
        source_run_id: RecordIdentifier,
        updated_at: datetime,
        existing_memory: AppMemory | None = None,
        compiled_procedure: ReplayProcedure | None = None,
        evidence_observation_ids: tuple[RecordIdentifier, ...] = (),
    ) -> DiscoveryPromotion:
        """Return accepted memory and coverage without upgrading ambiguous evidence."""
        if existing_memory is not None and existing_memory.compatibility != compatibility:
            raise ValueError("Existing app memory is incompatible with this discovery run.")
        requested_objective_ids = {objective.objective_id for objective in request.objectives}
        if verified_objective_ids is not None and not verified_objective_ids.issubset(
            requested_objective_ids
        ):
            raise ValueError("Verification names an objective outside this discovery request.")
        validate_memory_observations(observations, compatibility)
        screens = list(existing_memory.screens if existing_memory is not None else ())
        transitions = list(existing_memory.transitions if existing_memory is not None else ())
        unsuccessful_attempts = list(existing_memory.unsuccessful_attempts if existing_memory is not None else ())
        run_transitions, new_unsuccessful_attempts = self._promote_steps(
            agent_result,
            observations,
            compatibility,
            source_run_id,
            screens,
            transitions,
            unsuccessful_attempts,
        )
        coverage = _build_discovery_coverage(
            request,
            agent_result,
            verified_objective_ids,
            compiled_procedure,
            evidence_observation_ids,
        )
        if not run_transitions and not new_unsuccessful_attempts:
            return DiscoveryPromotion(memory=existing_memory, coverage=coverage)
        source_run_ids = tuple(
            dict.fromkeys((*(existing_memory.source_run_ids if existing_memory else ()), source_run_id))
        )
        retained_transitions = retain_transitions(
            transitions,
            compatibility.retention.maximum_transitions,
        )
        maximum_attempts = compatibility.retention.maximum_unsuccessful_attempts
        retained_attempts = tuple(unsuccessful_attempts[-maximum_attempts:])
        retained_screens = retain_observation_variants(
            screens,
            retained_transitions,
            retained_attempts,
            compatibility.retention.maximum_variants_per_screen,
        )
        memory = AppMemory(
            memory_id=(
                existing_memory.memory_id
                if existing_memory is not None
                else derived_memory_identifier("memory", compatibility.model_dump_json())
            ),
            compatibility=compatibility,
            source_run_ids=source_run_ids,
            screens=retained_screens,
            transitions=retained_transitions,
            unsuccessful_attempts=retained_attempts,
            app_knowledge=existing_memory.app_knowledge if existing_memory is not None else AppKnowledge(),
            updated_at=updated_at,
        )
        return DiscoveryPromotion(memory=memory, coverage=coverage)

    def _promote_steps(
        self,
        agent_result: AgentResult,
        observations: Mapping[str, Observation],
        compatibility: MemoryCompatibility,
        source_run_id: RecordIdentifier,
        screens: list[LogicalScreen],
        transitions: list[AcceptedTransition],
        unsuccessful_attempts: list[UnsuccessfulAttempt],
    ) -> tuple[list[AcceptedTransition], list[UnsuccessfulAttempt]]:
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
            transition = _merge_transition(transitions, transition)
            run_transitions.append(transition)
        return run_transitions, new_unsuccessful_attempts

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
            target_grounding=_accepted_grounding(step.target_grounding, source),
            destination_grounding=_accepted_grounding(step.destination_grounding, source),
            source_step_id=step.step_id,
        )


def _accepted_grounding(
    evidence: GroundingEvidence | None,
    source: Observation,
) -> AcceptedGrounding | None:
    if evidence is None or evidence.result.location is None:
        return None
    return AcceptedGrounding(
        location=evidence.result.location,
        source_observation_id=source.observation_id,
        source_masked_pixels_sha256=source.screen_fingerprint.masked_pixels_sha256,
        source_system_region_mask_sha256=source.screen_fingerprint.system_region_mask_sha256,
        resolution=evidence.result.resolution,
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
        source_step_id=step.step_id,
        source_run_id=source_run_id,
    )


def _semantic_action(step: AgentStep) -> SemanticRouteAction:
    proposal = step.proposal
    declared_value_name = None
    disposable_text = None
    input_contract = None
    if proposal.text is not None:
        if isinstance(proposal.text, DeclaredValueReference):
            declared_value_name = proposal.text.value_name
        else:
            disposable_text = proposal.text.value
    if proposal.action is ActionKind.TYPE and disposable_text is not None:
        input_contract = build_literal_input_contract(disposable_text)
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
        input_contract=input_contract,
    )


def _add_input_contract(
    previous: AcceptedTransition,
    current: AcceptedTransition,
) -> AcceptedTransition:
    action = current.action
    if (
        previous.action.action is not ActionKind.FOCUS
        or action.action is not ActionKind.TYPE
        or action.input_contract is None
        or previous.destination_screen_id != current.source_screen_id
        or previous.action.target_description is None
    ):
        return current
    return current.model_copy(
        update={
            "action": action.model_copy(
                update={
                    "target_description": previous.action.target_description,
                    "input_contract": action.input_contract,
                }
            )
        }
    )


def _transition_has_independent_signal(
    signals: frozenset[ActionOutcomeSignal],
    source_screen_id: str,
    destination_screen_id: str,
) -> bool:
    meaningful = signals - {ActionOutcomeSignal.PIXELS_CHANGED}
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
