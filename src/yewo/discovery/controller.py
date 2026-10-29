"""Deterministic bounded controller for one app-discovery run."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Literal

from yewo.agent.actuation.actions import ActionType
from yewo.agent.actuation.safety import (
    ActionAuthorization,
    ActionSafetyPolicy,
    ActionSource,
)
from yewo.agent.memory.control_identity import control_identity_id
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.run.settlement import SettlementStatus
from yewo.discovery.artifacts import seal_artifact, validate_shared_knowledge_payload
from yewo.discovery import budgeting
from yewo.discovery.contracts import (
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryRun,
    DiscoveryTerminationSignals,
    DiscoveryTransitionEvidence,
    FrontierState,
    RelevantFrontierEntry,
    SideEffectClass,
    finalize_discovery,
)
from yewo.discovery.runtime import (
    CapabilityEvidenceReceipt,
    DiscoveryActionCandidate,
    DiscoveryFinding,
    DiscoveryExecutionReceipt,
    DiscoveryObservationBatch,
    DiscoveryProposer,
    DiscoveryProposalReceipt,
    DiscoveryResult,
    DiscoveryRuntime,
)
from yewo.identity import derived_uuid

DEFAULT_DISCOVERY_SIDE_EFFECTS = frozenset(
    {SideEffectClass.READ_ONLY, SideEffectClass.REVERSIBLE}
)


class DiscoveryController:
    """Own authorization, evidence acceptance, budgets, reset, and termination."""

    def __init__(
        self,
        run: DiscoveryRun,
        *,
        allowed_side_effects: frozenset[SideEffectClass] = DEFAULT_DISCOVERY_SIDE_EFFECTS,
        forbidden_scenario_values: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ):
        if run.terminal is not None:
            raise ValueError("Discovery requires an open run.")
        self.run = run
        self.allowed_side_effects = allowed_side_effects
        self.forbidden_scenario_values = tuple(forbidden_scenario_values)
        self.clock = clock
        self.action_policy = ActionSafetyPolicy()
        self._findings: list[DiscoveryFinding] = []
        self._started_at = 0.0
        self._root_screen_state_id: str | None = None
        self._reset_since_action = False

    def run_to_terminal(
        self,
        runtime: DiscoveryRuntime,
        proposer: DiscoveryProposer,
        *,
        interrupted: Callable[[], bool] = lambda: False,
    ) -> DiscoveryResult:
        """Execute at most one authorized action per observation until terminal."""
        self._started_at = self.clock()
        initial = self._initial_observation(runtime)
        if isinstance(initial, DiscoveryResult):
            return initial
        current = initial

        while True:
            loop_terminal = self._loop_terminal(interrupted)
            if loop_terminal is not None:
                return loop_terminal

            batch_result = self._accept_batch_or_finish(current)
            if batch_result is not None:
                return batch_result
            if self._required_questions_succeeded():
                return self._finish()

            authorized = self._authorize_current_candidates(current)
            if not authorized:
                relevant = self._relevant_candidates(current)
                if relevant:
                    self._refuse_frontier(relevant)
                    return self._finish(blocker=DiscoveryBlocker.POLICY)
                if (
                    current.observation.screen_state_id != self._root_screen_state_id
                    and not self._reset_since_action
                ):
                    reset_result = self._reset(runtime)
                    if isinstance(reset_result, DiscoveryResult):
                        return reset_result
                    current = reset_result
                    continue
                self._close_unresolved(CapabilityState.UNAVAILABLE)
                return self._finish(frontier_remaining=False)

            authorized, action_budget = budgeting.budget_authorized(self.run, authorized)
            if action_budget is not None:
                return self._finish(exhausted=action_budget)

            proposal_budget = budgeting.proposal_budget(self.run, proposer)
            if proposal_budget is not None:
                return self._finish(exhausted=proposal_budget)
            proposal = proposer.select(
                self.run, tuple(candidate for candidate, _ in authorized.values())
            )
            proposal_error = self._record_proposal(proposal, proposer)
            if proposal_error is not None:
                return self._finish(blocker=proposal_error)
            if proposal.selected_frontier_id not in authorized:
                self._mark_available_frontier_attempted()
                self._close_unresolved(CapabilityState.AMBIGUOUS)
                return self._finish(frontier_remaining=False)

            candidate, authorization = authorized[proposal.selected_frontier_id]
            action_budget = budgeting.action_budget(self.run, candidate)
            if action_budget is not None:
                return self._finish(exhausted=action_budget)
            self._record_action_spend(candidate)
            self._reset_since_action = False
            try:
                execution = runtime.execute(authorization, candidate)
            except Exception:  # noqa: BLE001 - runtime is an external boundary
                self._close_frontier(candidate.frontier_id, FrontierState.ATTEMPTED)
                return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)

            self._record_findings(execution.findings)
            batch_result = self._accept_batch_or_finish(execution.destination)
            if batch_result is not None:
                return batch_result
            transition_identity = self._accept_transition(candidate, authorization, execution)
            if transition_identity is None:
                self._close_frontier(candidate.frontier_id, FrontierState.ATTEMPTED)
            else:
                self._close_frontier(
                    candidate.frontier_id,
                    FrontierState.VERIFIED,
                    transition_identity,
                )
                self._accept_capability_evidence(
                    execution.capability_evidence,
                    execution.destination.observation.observation_id,
                    transition_identity,
                )
            if self._required_questions_succeeded():
                return self._finish()

            if candidate.reset_after:
                reset_result = self._reset(runtime)
                if isinstance(reset_result, DiscoveryResult):
                    return reset_result
                current = reset_result
            else:
                current = execution.destination

    def _loop_terminal(
        self, interrupted: Callable[[], bool]
    ) -> DiscoveryResult | None:
        duration_result = self._record_duration_or_finish()
        if duration_result is not None:
            return duration_result
        return self._finish(interrupted=True) if interrupted() else None

    def _initial_observation(
        self, runtime: DiscoveryRuntime
    ) -> DiscoveryObservationBatch | DiscoveryResult:
        expected_setup = self.run.header.compatibility.setup_contract_sha256
        try:
            setup = runtime.prepare(expected_setup)
        except Exception:  # noqa: BLE001 - runtime is an external boundary
            return self._finish(blocker=DiscoveryBlocker.SETUP)
        if not setup.success or setup.setup_contract_sha256 != expected_setup:
            return self._finish(blocker=DiscoveryBlocker.SETUP)
        if not self._can_spend(DiscoveryBudgetDimension.OBSERVATIONS, 1):
            return self._finish(exhausted=DiscoveryBudgetDimension.OBSERVATIONS)
        try:
            return runtime.observe()
        except Exception:  # noqa: BLE001 - runtime is an external boundary
            return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)

    def _validated_run(self, **updates) -> None:
        candidate = self.run.model_copy(update=updates)
        self.run = DiscoveryRun.model_validate(candidate.model_dump(mode="json"))

    def _record_duration_or_finish(self) -> DiscoveryResult | None:
        elapsed_ms = max(0, round((self.clock() - self._started_at) * 1000))
        limit = self.run.budgets.duration_ms
        usage = self.run.usage.model_copy(update={"duration_ms": min(elapsed_ms, limit)})
        self._validated_run(usage=usage)
        if elapsed_ms >= limit:
            return self._finish(exhausted=DiscoveryBudgetDimension.DURATION_MS)
        return None

    def _can_spend(self, dimension: DiscoveryBudgetDimension, amount: int) -> bool:
        return budgeting.can_spend(self.run, dimension, amount)

    def _spend(self, dimension: DiscoveryBudgetDimension, amount: int) -> None:
        usage = budgeting.spend(self.run.usage, dimension, amount)
        self._validated_run(usage=usage)

    def _record_observation_batch(self, batch: DiscoveryObservationBatch) -> None:
        observation = batch.observation
        if observation.observation_id in {
            existing.observation_id for existing in self.run.observations
        }:
            return
        if not self._can_spend(DiscoveryBudgetDimension.OBSERVATIONS, 1):
            raise ValueError("The runtime returned an observation beyond its budget.")
        self._spend(DiscoveryBudgetDimension.OBSERVATIONS, 1)
        depth = max(self.run.usage.state_depth, batch.state_depth)
        if depth > self.run.budgets.state_depth:
            raise ValueError("The runtime returned a state beyond the depth budget.")
        usage = self.run.usage.model_copy(update={"state_depth": depth})
        self._validated_run(
            observations=(*self.run.observations, observation),
            usage=usage,
        )
        if self._root_screen_state_id is None:
            self._root_screen_state_id = observation.screen_state_id
        self._record_findings(batch.findings)
        self._accept_capability_evidence(
            batch.capability_evidence,
            observation.observation_id,
            None,
        )
        self._record_frontier(batch)

    def _accept_batch_or_finish(
        self, batch: DiscoveryObservationBatch
    ) -> DiscoveryResult | None:
        known_ids = {observation.observation_id for observation in self.run.observations}
        if batch.observation.observation_id not in known_ids:
            if not self._can_spend(DiscoveryBudgetDimension.OBSERVATIONS, 1):
                return self._finish(exhausted=DiscoveryBudgetDimension.OBSERVATIONS)
            if batch.state_depth > self.run.budgets.state_depth:
                return self._finish(exhausted=DiscoveryBudgetDimension.STATE_DEPTH)
        try:
            self._record_observation_batch(batch)
        except (TypeError, ValueError):
            return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)
        return None

    def _record_frontier(self, batch: DiscoveryObservationBatch) -> None:
        known_controls = {
            control.control_identity
            for control in batch.observation.controls
            if control.actionable
        }
        questions_by_id = {question.question_id: question for question in self.run.questions}
        frontier = list(self.run.frontier)
        known_frontier_ids = {entry.frontier_id for entry in frontier}
        questions = list(self.run.questions)
        for candidate in batch.candidates:
            if candidate.source_observation_id != batch.observation.observation_id:
                raise ValueError("A frontier candidate requires its exact source observation.")
            if candidate.source_screen_state_id != batch.observation.screen_state_id:
                raise ValueError("A frontier candidate requires its exact source state.")
            if candidate.control_identity not in known_controls:
                raise ValueError("A frontier candidate requires an actionable observed control.")
            if not candidate.relevant_question_ids or not set(
                candidate.relevant_question_ids
            ).issubset(questions_by_id):
                raise ValueError("A frontier candidate requires known capability questions.")
            unresolved_ids = tuple(
                question_id
                for question_id in candidate.relevant_question_ids
                if not questions_by_id[question_id].is_terminal
            )
            if not unresolved_ids or candidate.frontier_id in known_frontier_ids:
                continue
            for question_id in unresolved_ids:
                question = questions_by_id[question_id]
                if question.state is CapabilityState.PENDING:
                    activated = question.activate()
                    questions_by_id[question_id] = activated
                    questions[questions.index(question)] = activated
            frontier.append(
                RelevantFrontierEntry(
                    frontier_id=candidate.frontier_id,
                    source_observation_id=candidate.source_observation_id,
                    control_identity=candidate.control_identity,
                    relevant_question_ids=unresolved_ids,
                )
            )
            known_frontier_ids.add(candidate.frontier_id)
        self._validated_run(questions=tuple(questions), frontier=tuple(frontier))

    def _accept_capability_evidence(
        self,
        receipts: Sequence[CapabilityEvidenceReceipt],
        observation_id: str,
        transition_identity: str | None,
    ) -> None:
        questions = list(self.run.questions)
        questions_by_id = {question.question_id: question for question in questions}
        question_indexes = {
            question.question_id: index for index, question in enumerate(questions)
        }
        for receipt in receipts:
            if receipt.observation_id != observation_id:
                raise ValueError("Capability evidence requires its exact observation.")
            question = questions_by_id.get(receipt.question_id)
            if question is None:
                raise ValueError("Capability evidence names an unknown question.")
            if question.is_terminal:
                continue
            if receipt.kind == "route_verified":
                if transition_identity is None or receipt.transition_identity != transition_identity:
                    raise ValueError("Route capability evidence requires its accepted transition.")
                state = CapabilityState.ROUTE_VERIFIED
            else:
                if receipt.transition_identity is not None:
                    raise ValueError("Observed capability evidence cannot claim a transition.")
                state = CapabilityState.CAPABILITY_OBSERVED
            if question.state is CapabilityState.PENDING:
                question = question.activate()
            closed = question.close(state, (observation_id,))
            questions_by_id[question.question_id] = closed
            questions[question_indexes[question.question_id]] = closed
        self._validated_run(questions=tuple(questions))

    def _relevant_candidates(
        self, batch: DiscoveryObservationBatch
    ) -> tuple[DiscoveryActionCandidate, ...]:
        frontier_by_id = {entry.frontier_id: entry for entry in self.run.frontier}
        return tuple(
            candidate
            for candidate in batch.candidates
            if candidate.frontier_id in frontier_by_id
            and frontier_by_id[candidate.frontier_id].state is FrontierState.AVAILABLE
        )

    def _authorize_current_candidates(
        self, batch: DiscoveryObservationBatch
    ) -> dict[str, tuple[DiscoveryActionCandidate, ActionAuthorization]]:
        authorized: dict[
            str, tuple[DiscoveryActionCandidate, ActionAuthorization]
        ] = {}
        for candidate in self._relevant_candidates(batch):
            if candidate.side_effect not in self.allowed_side_effects:
                continue
            if not self.run.usage.can_attempt_transition(
                candidate.attempt_identity, self.run.budgets
            ):
                continue
            try:
                authorization = self.action_policy.authorize(
                    candidate.action,
                    source=ActionSource.DISCOVERY,
                    observation_id=candidate.source_observation_id,
                    screen_phash=candidate.source_phash,
                    screen_size=candidate.screen_size,
                    context_id=self.run.header.artifact_id,
                    keyboard_visible=batch.observation.keyboard_visible,
                    expected_destination_phash=candidate.expected_destination_phash,
                )
            except (TypeError, ValueError):
                continue
            if control_identity_id(authorization.control_identity) != candidate.control_identity:
                continue
            authorized[candidate.frontier_id] = (candidate, authorization)
        return authorized

    def _record_proposal(
        self,
        proposal: DiscoveryProposalReceipt,
        proposer: DiscoveryProposer,
    ) -> DiscoveryBlocker | None:
        usage, error = budgeting.record_proposal(self.run, proposal, proposer)
        self._validated_run(usage=usage)
        return error

    def _record_action_spend(self, candidate: DiscoveryActionCandidate) -> None:
        usage = budgeting.record_action_spend(self.run, candidate)
        self._validated_run(usage=usage)

    def _accept_transition(
        self,
        candidate: DiscoveryActionCandidate,
        authorization: ActionAuthorization,
        execution: DiscoveryExecutionReceipt,
    ) -> str | None:
        dispatch = execution.dispatch
        settlement = execution.settlement
        destination = execution.destination.observation
        if dispatch.authorization_id != authorization.authorization_id or not dispatch.success:
            return None
        if settlement.status is not SettlementStatus.DURABLE_PROGRESS:
            return None
        if not settlement.control_identity.matches(authorization.control_identity):
            return None
        if not {
            candidate.source_observation_id,
            destination.observation_id,
        }.issubset(set(settlement.observation_ids)):
            return None
        if not is_same_screen(settlement.destination_phash, destination.phash):
            return None
        transition_identity = derived_uuid(
            "discovery-transition-v1",
            candidate.source_observation_id,
            candidate.control_identity,
            destination.observation_id,
        )
        transition = DiscoveryTransitionEvidence(
            transition_identity=transition_identity,
            source_observation_id=candidate.source_observation_id,
            destination_observation_id=destination.observation_id,
            control_identity=candidate.control_identity,
            action_kind=(
                candidate.action.type.value
                if isinstance(candidate.action.type, ActionType)
                else candidate.action.type
            ),
            authorization_receipt_id=authorization.authorization_id,
            dispatch_receipt_id=dispatch.dispatch_id,
            settlement_receipt_id=settlement.settlement_id,
            side_effect=candidate.side_effect,
            reversible=candidate.reversible,
        )
        self._validated_run(transitions=(*self.run.transitions, transition))
        return transition_identity

    def _close_frontier(
        self,
        frontier_id: str,
        state: Literal[
            FrontierState.ATTEMPTED,
            FrontierState.VERIFIED,
            FrontierState.POLICY_REFUSED,
        ],
        transition_identity: str | None = None,
    ) -> None:
        frontier = tuple(
            entry.close(state, transition_identity)
            if entry.frontier_id == frontier_id
            else entry
            for entry in self.run.frontier
        )
        self._validated_run(frontier=frontier)

    def _refuse_frontier(self, candidates: Sequence[DiscoveryActionCandidate]) -> None:
        for candidate in candidates:
            self._close_frontier(candidate.frontier_id, FrontierState.POLICY_REFUSED)

    def _mark_available_frontier_attempted(self) -> None:
        frontier = tuple(
            entry.close(FrontierState.ATTEMPTED)
            if entry.state is FrontierState.AVAILABLE
            else entry
            for entry in self.run.frontier
        )
        self._validated_run(frontier=frontier)

    def _close_unresolved(self, state: CapabilityState) -> None:
        questions = []
        for question in self.run.questions:
            if question.is_terminal:
                questions.append(question)
                continue
            if question.state is CapabilityState.PENDING and state not in (
                CapabilityState.NOT_EVALUATED,
                CapabilityState.BLOCKED_BY_POLICY,
                CapabilityState.BLOCKED_BY_SETUP,
            ):
                question = question.activate()
            questions.append(question.close(state))
        self._validated_run(questions=tuple(questions))

    def _required_questions_succeeded(self) -> bool:
        return all(
            not question.required
            or question.state
            in (CapabilityState.ROUTE_VERIFIED, CapabilityState.CAPABILITY_OBSERVED)
            for question in self.run.questions
        )

    def _reset(self, runtime: DiscoveryRuntime) -> DiscoveryObservationBatch | DiscoveryResult:
        if not self._can_spend(DiscoveryBudgetDimension.RESETS, 1):
            return self._finish(exhausted=DiscoveryBudgetDimension.RESETS)
        if not self._can_spend(DiscoveryBudgetDimension.OBSERVATIONS, 1):
            return self._finish(exhausted=DiscoveryBudgetDimension.OBSERVATIONS)
        self._spend(DiscoveryBudgetDimension.RESETS, 1)
        try:
            receipt = runtime.reset()
        except Exception:  # noqa: BLE001 - runtime is an external boundary
            return self._finish(blocker=DiscoveryBlocker.RESET)
        if not receipt.success or receipt.observation is None:
            return self._finish(blocker=DiscoveryBlocker.RESET)
        self._reset_since_action = True
        return receipt.observation

    def _record_findings(self, findings: Sequence[DiscoveryFinding]) -> None:
        for finding in findings:
            validate_shared_knowledge_payload(
                finding.model_dump(mode="json"),
                forbidden_scenario_values=self.forbidden_scenario_values,
            )
            self._findings.append(finding)

    def _finish(
        self,
        *,
        interrupted: bool = False,
        exhausted: DiscoveryBudgetDimension | None = None,
        blocker: DiscoveryBlocker | None = None,
        frontier_remaining: bool = True,
    ) -> DiscoveryResult:
        if blocker is DiscoveryBlocker.POLICY:
            self._close_unresolved(CapabilityState.BLOCKED_BY_POLICY)
        terminal = finalize_discovery(
            self.run,
            DiscoveryTerminationSignals(
                externally_interrupted=interrupted,
                exhausted_budget=exhausted,
                blocker=blocker,
                relevant_frontier_remaining=frontier_remaining,
            ),
        )
        sealed = seal_artifact(
            terminal,
            forbidden_scenario_values=self.forbidden_scenario_values,
        )
        return DiscoveryResult(run=sealed, findings=tuple(self._findings))
