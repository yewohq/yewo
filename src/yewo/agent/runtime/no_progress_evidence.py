"""Own run-scoped no-progress attempts, transitions, and screen facts."""

from __future__ import annotations

from yewo.agent.actuation import ActionProposal, ActionRefusal
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.memory.screens import exact_observation_state_key, observations_match
from yewo.agent.planning.history import (
    MAXIMUM_SCREEN_FACT_CHARACTERS,
    PlanningScreenFact,
    planning_screen_reference,
)
from yewo.agent.runtime.interaction_intent import same_interaction_family
from yewo.agent.runtime.prohibited_intents import collect_prohibited_intents
from yewo.agent.runtime.interaction_intent import AmbiguousInteractionState
from yewo.agent.runtime.workflow_routes import WorkflowRouteExhaustion
from yewo.observations import Observation


class NoProgressEvidence:
    """Store and query evidence shared by the no-progress policies."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._comparison = comparison
        self.reset()

    def reset(self, initial_observation: Observation | None = None) -> None:
        """Drop evidence from an earlier application lifetime."""
        self._attempts: list[tuple[ActionProposal, Observation]] = []
        self._refusals: list[tuple[ActionProposal, str, Observation, int]] = []
        self._transitions: list[tuple[ActionProposal, Observation, Observation]] = []
        self._exhausted_transitions: list[tuple[ActionProposal, Observation]] = []
        self._focus_prohibitions: list[tuple[ActionProposal, Observation]] = []
        self._exhausted_actions: list[tuple[str, Observation]] = []
        self._visited = [initial_observation] if initial_observation is not None else []
        self._reported_revisited_screens: list[Observation] = []
        self._facts: list[tuple[Observation, PlanningScreenFact]] = []

    def known_facts(self) -> tuple[str, ...]:
        """Return every retained deterministic planning fact."""
        return tuple(fact.fact for unused_screen, fact in self._facts)

    def screen_facts(
        self,
        observation: Observation,
        recent_proposal_ids: frozenset[str],
        maximum_facts: int,
    ) -> tuple[PlanningScreenFact, ...]:
        """Return current-screen facts not duplicated by recent history."""
        matched = tuple(
            fact
            for screen, fact in self._facts
            if self._matches(observation, screen)
            if fact.is_durable_constraint or fact.origin_proposal_id not in recent_proposal_ids
        )
        return matched[-maximum_facts:]

    def record_fact(
        self,
        observation: Observation,
        fact: str,
        *,
        origin_proposal_id: str,
        is_durable_constraint: bool = False,
        supersedes_recovery_guidance: bool = False,
    ) -> None:
        """Retain one bounded screen-scoped planning fact."""
        screen_fact = PlanningScreenFact(
            screen=planning_screen_reference(observation),
            fact=fact[:MAXIMUM_SCREEN_FACT_CHARACTERS],
            origin_proposal_id=origin_proposal_id,
            is_durable_constraint=is_durable_constraint,
            supersedes_recovery_guidance=supersedes_recovery_guidance,
        )
        self._facts.append((observation, screen_fact))

    def exhausted_action(self, proposal: ActionProposal, observation: Observation) -> bool:
        """Return whether this action exhausted one exact transition on the screen."""
        return any(
            proposal.action.value == action and self._matches(observation, screen)
            for action, screen in self._exhausted_actions
        )

    def exhausted_transition(self, proposal: ActionProposal, observation: Observation) -> bool:
        """Return whether this interaction exhausted an equivalent transition."""
        return any(
            same_interaction_family(proposal, prior) and self._matches(observation, screen)
            for prior, screen in self._exhausted_transitions
        )

    def prior_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> tuple[str, int] | None:
        """Return matching refusal code and count for this interaction."""
        return next(
            (
                (code, count)
                for prior, code, screen, count in self._refusals
                if same_interaction_family(proposal, prior)
                and self._matches(observation, screen)
            ),
            None,
        )

    def has_attempt(self, proposal: ActionProposal, observation: Observation) -> bool:
        """Return whether this interaction already produced no visible change."""
        return any(
            same_interaction_family(proposal, prior) and self._matches(observation, screen)
            for prior, screen in self._attempts
        )

    def record_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
        refusal: ActionRefusal,
    ) -> int:
        """Record a refusal and return its equivalent-interaction count."""
        matched_index = next(
            (
                index
                for index, (prior, code, screen, unused_count) in enumerate(self._refusals)
                if same_interaction_family(proposal, prior)
                and code == refusal.code.value
                and self._matches(observation, screen)
            ),
            None,
        )
        if matched_index is None:
            self._refusals.append((proposal, refusal.code.value, observation, 1))
            return 1
        prior, code, screen, previous_count = self._refusals[matched_index]
        count = previous_count + 1
        self._refusals[matched_index] = (prior, code, screen, count)
        return count

    def record_attempt(self, proposal: ActionProposal, observation: Observation) -> None:
        """Record one dispatched interaction with no observable change."""
        self._attempts.append((proposal, observation))

    def record_exited_transition(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> None:
        """Exhaust an interaction that caused the application to exit."""
        self._exhausted_transitions.append((proposal, observation))

    def record_transition(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
    ) -> int:
        """Record an observed transition and return its exact repetition count."""
        count = 1 + sum(
            proposal.action is prior_proposal.action
            and exact_observation_state_key(before) == exact_observation_state_key(prior_before)
            and exact_observation_state_key(after) == exact_observation_state_key(prior_after)
            for prior_proposal, prior_before, prior_after in self._transitions
        )
        self._transitions.append((proposal, before, after))
        if count == 2:
            self._exhausted_transitions.append((proposal, before))
            self._focus_prohibitions.append((proposal, after))
        elif count >= 3:
            self._exhausted_actions.append((proposal.action.value, before))
        return count

    def record_revisit(self, observation: Observation) -> bool:
        """Return true once for each revisited logical screen."""
        prior_screens = self._visited[:-1] if self._visited else ()
        revisited = any(self._matches(observation, prior) for prior in prior_screens)
        self._visited.append(observation)
        already_reported = any(
            self._matches(observation, reported) for reported in self._reported_revisited_screens
        )
        if not revisited or already_reported:
            return False
        self._reported_revisited_screens.append(observation)
        return True

    def prohibited_intents(
        self,
        observation: Observation,
        ambiguous: AmbiguousInteractionState,
        workflow_routes: WorkflowRouteExhaustion,
        work_items: tuple[str, ...],
        maximum_intents: int,
    ) -> tuple[str, ...]:
        """Collect bounded interaction prohibitions from all evidence groups."""
        return collect_prohibited_intents(
            observation,
            self._matches,
            self._attempts,
            self._refusals,
            self._exhausted_transitions,
            self._focus_prohibitions,
            ambiguous,
            workflow_routes.prohibited_intents(observation, work_items),
            maximum_intents,
        )

    def _matches(self, first: Observation, second: Observation) -> bool:
        return observations_match(first, second, self._comparison)


__all__ = ["NoProgressEvidence"]
