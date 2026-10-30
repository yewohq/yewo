"""Deterministic bounded controller for one app-discovery run."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable, Sequence

from yewo.agent.actuation.safety import (
    ActionAuthorization,
    ActionSafetyPolicy,
)
from yewo.discovery.artifacts import seal_artifact, validate_shared_knowledge_payload
from yewo.discovery import budgeting
from yewo.discovery.candidate_selection import (
    CandidateSelection,
    authorize_candidates,
    prioritize_authorized_candidates,
    select_available_candidates,
)
from yewo.discovery.contracts import (
    CapabilityState,
    DiscoveryBlocker,
    DiscoveryBudgetDimension,
    DiscoveryRun,
    DiscoveryTerminationSignals,
    FrontierState,
    SideEffectClass,
    finalize_discovery,
)
from yewo.discovery.frontier import extend_frontier
from yewo.discovery.frontier_navigation import (
    FrontierReachability,
    select_frontier_replay_candidate,
    stranded_frontier_ids,
)
from yewo.discovery.frontier_state import (
    ClosableFrontierState,
    WindowClosureState,
    close_all_available_frontier,
    close_available_frontier_ids,
    close_frontier_entry,
    close_unresolved_goals,
)
from yewo.discovery.goal_settlement import settle_goals
from yewo.discovery.reached_screens import merge_reached_screen_candidates
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
from yewo.discovery.transition_acceptance import build_accepted_transition

logger = logging.getLogger(__name__)

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
        scenario_owned_values: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ):
        if run.terminal is not None:
            raise ValueError("Discovery requires an open run.")
        self.run = run
        self.allowed_side_effects = allowed_side_effects
        self.scenario_owned_values = tuple(scenario_owned_values)
        self.clock = clock
        self.action_policy = ActionSafetyPolicy()
        self._findings: list[DiscoveryFinding] = []
        self._started_at = 0.0
        self._root_screen_state_id: str | None = None
        self._reset_since_action = False
        self._proposer: DiscoveryProposer | None = None

    def run_to_terminal(
        self,
        runtime: DiscoveryRuntime,
        proposer: DiscoveryProposer,
        *,
        interrupted: Callable[[], bool] = lambda: False,
    ) -> DiscoveryResult:
        """Execute at most one authorized action per observation until terminal."""
        self._started_at = self.clock()
        self._proposer = proposer
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
            if self._required_goals_succeeded():
                return self._finish()

            selection = self._select_candidate(current, proposer)
            if isinstance(selection, DiscoveryResult):
                return selection
            if selection is None:
                outcome = self._leave_dead_end(runtime, current)
                if isinstance(outcome, DiscoveryResult):
                    return outcome
                if isinstance(outcome, CandidateSelection):
                    selection = outcome
                else:
                    current = outcome
                    continue

            candidate = selection.candidate
            authorization = selection.authorization
            action_budget = budgeting.action_budget(self.run, candidate)
            if action_budget is not None:
                return self._finish(exhausted=action_budget)
            self._record_action_spend(candidate)
            self._reset_since_action = False
            try:
                execution = runtime.execute(authorization, candidate)
            except Exception:  # noqa: BLE001 - runtime is an external boundary
                return self._finish_execution_failure(selection)

            self._record_findings(execution.findings)
            batch_result = self._accept_batch_or_finish(execution.destination)
            if batch_result is not None:
                return batch_result
            transition_identity = self._accept_transition(candidate, authorization, execution)
            replay_destination = selection.replay_destination_state_id
            if replay_destination is not None and (
                transition_identity is None
                or execution.destination.observation.screen_state_id
                != replay_destination
            ):
                return self._finish_navigation_unavailable()
            if transition_identity is None:
                self._close_frontier(candidate.frontier_id, FrontierState.ATTEMPTED)
            elif replay_destination is None:
                self._close_frontier(
                    candidate.frontier_id,
                    FrontierState.VERIFIED,
                    transition_identity,
                )
                try:
                    self._accept_capability_evidence(
                        execution.capability_evidence,
                        execution.destination.observation.observation_id,
                        transition_identity,
                    )
                except (TypeError, ValueError):
                    return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)
            if self._required_goals_succeeded():
                return self._finish()

            if candidate.reset_after:
                reset_result = self._reset(runtime)
                if isinstance(reset_result, DiscoveryResult):
                    return reset_result
                current = reset_result
            else:
                current = execution.destination

    def _finish_execution_failure(
        self, selection: CandidateSelection
    ) -> DiscoveryResult:
        if selection.replay_destination_state_id is not None:
            return self._finish_navigation_unavailable()
        self._close_frontier(selection.candidate.frontier_id, FrontierState.ATTEMPTED)
        return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)

    def _finish_navigation_unavailable(self) -> DiscoveryResult:
        """End a run that still holds work no route of its own can reach.

        Ambiguous rather than unreachable, because the run never found out.
        Every open offer closes with the same reading, so a later reader does
        not take an unvisited screen for a tried one.
        """
        self._validated_run(
            frontier=close_all_available_frontier(
                self.run.frontier,
                FrontierState.NAVIGATION_UNAVAILABLE,
            )
        )
        self._close_unresolved(CapabilityState.AMBIGUOUS)
        return self._finish(
            frontier_remaining=False,
            navigation_unavailable=True,
        )

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
        except Exception:
            # A raise is a broken runtime, not a device reporting it is not ready.
            logger.exception("The discovery runtime failed to prepare.")
            return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)
        if not setup.success or setup.setup_contract_sha256 != expected_setup:
            logger.error("Discovery setup was refused: %s", setup.reason or "no reason given")
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
        self._record_capability_candidates(batch)
        self._accept_capability_evidence(
            batch.capability_evidence,
            observation.observation_id,
            None,
        )
        self._record_frontier(batch)

    def _charge_unreported_spend(
        self,
    ) -> tuple[DiscoveryBlocker | None, DiscoveryBudgetDimension | None]:
        """Charge pending model spend before the run is sealed."""
        if self._proposer is None:
            return None, None
        usage, blocker, exhausted = budgeting.charge_unreported_proposal_spend(
            self.run, self._proposer
        )
        self._proposer = None
        if usage is not None:
            self._validated_run(usage=usage)
        return blocker, exhausted

    def _record_capability_candidates(self, batch: DiscoveryObservationBatch) -> None:
        kept = merge_reached_screen_candidates(
            self.run, batch.capability_candidates, self.scenario_owned_values
        )
        if kept is not None:
            self._validated_run(capability_candidates=kept)

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

    def _select_candidate(
        self,
        batch: DiscoveryObservationBatch,
        proposer: DiscoveryProposer,
    ) -> CandidateSelection | DiscoveryResult | None:
        """Select one authorized candidate or return a typed terminal."""
        relevant = select_available_candidates(self.run, batch)
        authorized = authorize_candidates(
            self.run,
            batch,
            relevant,
            self.action_policy,
            self.allowed_side_effects,
        )
        if not authorized:
            if not relevant:
                return None
            self._refuse_frontier(relevant)
            return self._finish(blocker=DiscoveryBlocker.POLICY)

        authorized, action_budget = budgeting.budget_authorized(self.run, authorized)
        if action_budget is not None:
            return self._finish(exhausted=action_budget)
        proposal_budget = budgeting.proposal_budget(self.run, proposer)
        if proposal_budget is not None:
            return self._finish(exhausted=proposal_budget)
        prioritized = prioritize_authorized_candidates(batch, authorized)
        window = prioritized[: self.run.budgets.proposal_window_size]
        window_by_id = {
            candidate.frontier_id: (candidate, receipt)
            for candidate, receipt in window
        }
        proposal = proposer.select(
            self.run, tuple(candidate for candidate, _ in window)
        )
        proposal_error = self._record_proposal(proposal, proposer)
        if proposal_error is not None:
            return self._finish(blocker=proposal_error)
        if proposal.provider_failed:
            return self._finish(blocker=DiscoveryBlocker.INFRASTRUCTURE)
        selected_frontier_id = proposal.selected_frontier_id
        if selected_frontier_id is None:
            self._close_frontier_ids(window_by_id, FrontierState.DECLINED)
            return self._select_candidate(batch, proposer)
        if selected_frontier_id not in window_by_id:
            self._close_frontier_ids(window_by_id, FrontierState.DECLINED)
            return self._finish_navigation_unavailable()
        candidate, authorization = window_by_id[selected_frontier_id]
        return CandidateSelection(candidate, authorization)

    def _record_frontier(self, batch: DiscoveryObservationBatch) -> None:
        extension = extend_frontier(
            batch.observation,
            batch.candidates,
            self.run.goals,
            self.run.frontier,
        )
        self._validated_run(goals=extension.goals, frontier=extension.frontier)

    def _accept_capability_evidence(
        self,
        receipts: Sequence[CapabilityEvidenceReceipt],
        observation_id: str,
        transition_identity: str | None,
    ) -> None:
        settlement = settle_goals(
            receipts,
            self.run.goals,
            self.run.frontier,
            observation_id,
            transition_identity,
        )
        self._validated_run(goals=settlement.goals, frontier=settlement.frontier)

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
        transition = build_accepted_transition(candidate, authorization, execution)
        if transition is None:
            return None
        self._validated_run(transitions=(*self.run.transitions, transition))
        return transition.transition_identity

    def _close_frontier(
        self,
        frontier_id: str,
        state: ClosableFrontierState,
        transition_identity: str | None = None,
    ) -> None:
        self._validated_run(
            frontier=close_frontier_entry(
                self.run.frontier,
                frontier_id,
                state,
                transition_identity,
            )
        )

    def _close_frontier_ids(
        self,
        frontier_ids: Iterable[str],
        state: WindowClosureState,
    ) -> None:
        self._validated_run(
            frontier=close_available_frontier_ids(
                self.run.frontier,
                set(frontier_ids),
                state,
            )
        )

    def _refuse_frontier(self, candidates: Sequence[DiscoveryActionCandidate]) -> None:
        for candidate in candidates:
            self._close_frontier(candidate.frontier_id, FrontierState.POLICY_REFUSED)

    def _close_unresolved(self, state: CapabilityState) -> None:
        self._validated_run(goals=close_unresolved_goals(self.run.goals, state))

    def _required_goals_succeeded(self) -> bool:
        return all(
            not goal.required
            or goal.state
            in (CapabilityState.ROUTE_VERIFIED, CapabilityState.REACHED)
            for goal in self.run.goals
        )

    def _leave_dead_end(
        self,
        runtime: DiscoveryRuntime,
        current: DiscoveryObservationBatch,
    ) -> DiscoveryObservationBatch | CandidateSelection | DiscoveryResult:
        """Replay known navigation, reset, or close a genuinely spent frontier."""
        replay = self._select_replay(current)
        if replay is FrontierReachability.WORK_IS_HERE:
            self._close_frontier_ids(
                stranded_frontier_ids(self.run, current),
                FrontierState.NAVIGATION_UNAVAILABLE,
            )
            replay = self._select_replay(current)
        if isinstance(replay, CandidateSelection):
            return replay
        if (
            current.observation.screen_state_id != self._root_screen_state_id
            and not self._reset_since_action
        ):
            return self._reset(runtime)
        if any(entry.state is FrontierState.AVAILABLE for entry in self.run.frontier):
            return self._finish_navigation_unavailable()
        self._close_unresolved(CapabilityState.UNREACHABLE)
        return self._finish(frontier_remaining=False)

    def _select_replay(
        self,
        current: DiscoveryObservationBatch,
    ) -> CandidateSelection | FrontierReachability:
        return select_frontier_replay_candidate(
            self.run,
            current,
            self.action_policy,
            self.allowed_side_effects,
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
                scenario_owned_values=self.scenario_owned_values,
            )
            self._findings.append(finding)

    def _finish(
        self,
        *,
        interrupted: bool = False,
        exhausted: DiscoveryBudgetDimension | None = None,
        blocker: DiscoveryBlocker | None = None,
        frontier_remaining: bool = True,
        navigation_unavailable: bool = False,
    ) -> DiscoveryResult:
        spend_blocker, spend_exhausted = self._charge_unreported_spend()
        blocker = blocker or spend_blocker
        exhausted = exhausted or spend_exhausted
        if blocker is DiscoveryBlocker.POLICY:
            self._close_unresolved(CapabilityState.BLOCKED_BY_POLICY)
        terminal = finalize_discovery(
            self.run,
            DiscoveryTerminationSignals(
                externally_interrupted=interrupted,
                exhausted_budget=exhausted,
                blocker=blocker,
                relevant_frontier_remaining=frontier_remaining,
                navigation_unavailable=navigation_unavailable,
            ),
        )
        sealed = seal_artifact(
            terminal,
            scenario_owned_values=self.scenario_owned_values,
        )
        return DiscoveryResult(run=sealed, findings=tuple(self._findings))
