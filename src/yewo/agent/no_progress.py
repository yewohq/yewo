"""Detect repeated semantic no-progress attempts and refusals."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation import (
    ActionBudget,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionPolicy,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.agent.app_memory import ScreenComparisonPolicy
from yewo.agent.planner import PlanningHistoryEntry
from yewo.agent.screen_memory import observations_match
from yewo.grounding import GroundingCorrection
from yewo.observations import Observation


def proposal_refusal(
    tracker: ActionNoProgressTracker,
    policy: ActionPolicy,
    proposal: ActionProposal,
    observation: Observation,
    budget: ActionBudget,
) -> ActionRefusal | None:
    """Return deterministic no-progress or state-policy feedback."""
    return tracker.refusal_for(proposal, observation) or policy.refusal_before_grounding(
        proposal, observation, budget
    )


def record_refusal_with_facts(
    tracker: ActionNoProgressTracker,
    proposal: ActionProposal,
    observation: Observation,
    refusal: ActionRefusal,
    narrate_fact: Callable[[str], None],
) -> str | None:
    """Record a refusal and narrate only newly derived facts."""
    facts_before = tracker.known_facts()
    reason = tracker.record_refusal(proposal, observation, refusal)
    for fact in tracker.known_facts()[len(facts_before) :]:
        narrate_fact(fact)
    return reason


def record_outcome_with_facts(
    tracker: ActionNoProgressTracker,
    proposal: ActionProposal,
    before: Observation,
    after: Observation,
    outcome: ActionOutcome,
    narrate_fact: Callable[[str], None],
) -> str | None:
    """Record an outcome and narrate only newly derived facts."""
    facts_before = tracker.known_facts()
    reason = tracker.record_outcome(proposal, before, after, outcome)
    for fact in tracker.known_facts()[len(facts_before) :]:
        narrate_fact(fact)
    return reason


def grounding_correction_for_history(
    proposal: ActionProposal,
    history: list[PlanningHistoryEntry],
) -> GroundingCorrection | None:
    """Resolve one declared correction from trusted no-progress history."""
    failed_proposal_id = proposal.grounding_correction_for_proposal_id
    for entry in reversed(history):
        if entry.proposal.proposal_id != failed_proposal_id:
            continue
        point = entry.grounded_target_point
        return point.correction() if point is not None else None
    return None


class ActionNoProgressTracker:
    """Bound repeated semantic failures across equivalent observed screens."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._comparison = comparison
        self._attempts: list[tuple[tuple[str, ...], Observation]] = []
        self._refusals: list[tuple[tuple[str, ...], str, Observation, int]] = []
        self._transitions: list[tuple[str, Observation, Observation]] = []
        self._visited: list[Observation] = []
        self._reported_revisit = False
        self._facts: list[str] = []

    def start(self, observation: Observation) -> None:
        """Record the initial logical screen."""
        self._transitions = []
        self._visited = [observation]
        self._reported_revisit = False

    def known_facts(self) -> tuple[str, ...]:
        """Return deterministic feedback for the next planning request."""
        return tuple(self._facts)

    def refusal_for(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> ActionRefusal | None:
        """Refuse a known no-progress action on an equivalent screen."""
        semantics = _action_semantics(proposal)
        prior_refusal = next(
            (
                code
                for prior, code, screen, unused_count in self._refusals
                if semantics == prior and observations_match(observation, screen, self._comparison)
            ),
            None,
        )
        if prior_refusal is not None:
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode(prior_refusal),
                reason=(
                    "The same semantic action was already refused on an equivalent screen. "
                    "Choose a different action."
                ),
            )
        if not any(
            semantics == prior and observations_match(observation, screen, self._comparison)
            for prior, screen in self._attempts
        ):
            return None
        return ActionRefusal(
            proposal_id=proposal.proposal_id,
            code=ActionRefusalCode.REPEATED_NO_PROGRESS,
            reason=(
                "The same semantic action already produced no observable change "
                "on an equivalent screen."
            ),
        )

    def record_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
        refusal: ActionRefusal,
    ) -> str | None:
        """Return a terminal reason after the same refusal repeats twice."""
        semantics = _action_semantics(proposal)
        matched_index = next(
            (
                index
                for index, (prior, code, screen, unused_count) in enumerate(self._refusals)
                if prior == semantics
                and code == refusal.code.value
                and observations_match(observation, screen, self._comparison)
            ),
            None,
        )
        if matched_index is None:
            count = 1
            self._refusals.append((semantics, refusal.code.value, observation, count))
        else:
            prior, code, screen, previous_count = self._refusals[matched_index]
            count = previous_count + 1
            self._refusals[matched_index] = (prior, code, screen, count)
        if count == 1:
            self._facts.append(f"Do not repeat {proposal.action.value}: {refusal.reason}")
            return None
        return "The same semantic action was repeatedly refused on an equivalent screen."

    def record_outcome(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
    ) -> str | None:
        """Record trusted no-progress and advise after a screen revisit."""
        if outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE:
            self._attempts.append((_action_semantics(proposal), before))
            return None
        if outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
            return None
        repeated_transition_count = 1 + sum(
            proposal.action.value == action
            and observations_match(before, prior_before, self._comparison)
            and observations_match(after, prior_after, self._comparison)
            for action, prior_before, prior_after in self._transitions
        )
        self._transitions.append((proposal.action.value, before, after))
        if repeated_transition_count == 2:
            self._facts.append(
                "The same action repeated an already observed screen transition. "
                "Choose a different route."
            )
        elif repeated_transition_count >= 3:
            return "The same action repeatedly produced an already observed screen transition."
        prior_screens = self._visited[:-1] if self._visited else ()
        revisited = any(
            observations_match(after, prior, self._comparison) for prior in prior_screens
        )
        self._visited.append(after)
        if not revisited or self._reported_revisit:
            return None
        self._reported_revisit = True
        self._facts.append(
            "The recent actions returned to an already visited screen. Choose a different route."
        )
        return None


def _action_semantics(proposal: ActionProposal) -> tuple[str, ...]:
    return (
        proposal.action.value,
        proposal.target_expression or "",
        proposal.destination_expression or "",
        proposal.direction.value if proposal.direction is not None else "",
        proposal.distance.value if proposal.distance is not None else "",
        str(proposal.duration_seconds) if proposal.duration_seconds is not None else "",
    )
