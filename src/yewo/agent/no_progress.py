"""Detect repeated semantic no-progress attempts and refusals."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation import (
    ActionBudget,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionPolicy,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.agent.app_memory import ScreenComparisonPolicy
from yewo.agent.planner import PlanningHistoryEntry
from yewo.agent.planning_history import PlanningScreenFact, planning_screen_reference
from yewo.agent.screen_memory import exact_observation_state_key, observations_match
from yewo.grounding import GroundingCorrection
from yewo.observations import Observation


CONDITIONAL_REFUSAL_CODES = frozenset({ActionRefusalCode.DISCOVERY_INCOMPLETE})
MAXIMUM_CONSECUTIVE_REFUSALS = 5


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
        self._no_change_proposals: list[tuple[ActionProposal, Observation]] = []
        self._refusals: list[tuple[tuple[str, ...], str, Observation, int]] = []
        self._transitions: list[tuple[str, Observation, Observation]] = []
        self._exhausted_transitions: list[tuple[tuple[str, ...], Observation]] = []
        self._exhausted_actions: list[tuple[str, Observation]] = []
        self._visited: list[Observation] = []
        self._reported_revisit = False
        self._facts: list[PlanningScreenFact] = []
        self._latest_feedback_proposal_id: str | None = None
        self._consecutive_refusals = 0

    def start(self, observation: Observation) -> None:
        """Record the initial logical screen."""
        self._transitions = []
        self._exhausted_transitions = []
        self._exhausted_actions = []
        self._no_change_proposals = []
        self._visited = [observation]
        self._reported_revisit = False
        self._latest_feedback_proposal_id = None
        self._consecutive_refusals = 0

    def known_facts(self) -> tuple[str, ...]:
        """Return deterministic feedback for the next planning request."""
        return tuple(fact.fact for fact in self._facts)

    def screen_facts(self) -> tuple[PlanningScreenFact, ...]:
        """Return deterministic feedback with the screen that owns each fact."""
        return tuple(self._facts)

    def refusal_for(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> ActionRefusal | None:
        """Refuse a known no-progress action on an equivalent screen."""
        refinement_refusal = self._interaction_refinement_refusal(proposal, observation)
        if refinement_refusal is not None:
            return refinement_refusal
        semantics = _action_semantics(proposal)
        if any(
            proposal.action.value == action
            and observations_match(observation, screen, self._comparison)
            for action, screen in self._exhausted_actions
        ):
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    f"{proposal.action.value} repeated one observed screen transition "
                    "three times on this screen. Choose a different action."
                ),
            )
        if any(
            semantics == prior and observations_match(observation, screen, self._comparison)
            for prior, screen in self._exhausted_transitions
        ):
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    "This semantic action already repeated an equivalent screen transition. "
                    "Choose a different route."
                ),
            )
        prior_refusal = next(
            (
                (code, count)
                for prior, code, screen, count in self._refusals
                if semantics == prior and observations_match(observation, screen, self._comparison)
            ),
            None,
        )
        if prior_refusal is not None:
            code, count = prior_refusal
            if code == ActionRefusalCode.INVALID_INPUT.value and count == 1:
                # The planner gets one bounded chance to replace an incompatible value.
                # Text is intentionally absent from action semantics, so refusing here
                # would also reject a corrected value without ever trying it.
                return None
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode(code),
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
        """Bound one refused action and report only run-wide refusal deadlock."""
        self._latest_feedback_proposal_id = proposal.proposal_id
        self._consecutive_refusals += 1
        if refusal.code in CONDITIONAL_REFUSAL_CODES:
            self._record_fact(
                observation,
                f"Resolve this before you repeat {proposal.action.value}: {refusal.reason}",
            )
            return self._refusal_deadlock_reason()
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
            self._record_fact(
                observation,
                f"Do not repeat {proposal.action.value}: {refusal.reason}",
            )
        return self._refusal_deadlock_reason()

    def _refusal_deadlock_reason(self) -> str | None:
        if self._consecutive_refusals < MAXIMUM_CONSECUTIVE_REFUSALS:
            return None
        return (
            f"The agent reached {MAXIMUM_CONSECUTIVE_REFUSALS} consecutive refusals "
            "without a dispatched action."
        )

    def record_outcome(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
    ) -> str | None:
        """Record trusted no-progress and advise after a screen revisit."""
        self._latest_feedback_proposal_id = proposal.proposal_id
        self._consecutive_refusals = 0
        if ActionOutcomeSignal.APP_EXITED in outcome.signals:
            self._exhausted_transitions.append((_action_semantics(proposal), before))
            self._record_fact(
                before,
                f"Do not repeat {proposal.action.value}: the application exited after this action.",
            )
            return None
        if outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE:
            self._attempts.append((_action_semantics(proposal), before))
            self._no_change_proposals.append((proposal, before))
            if proposal.interaction_refinement_for_proposal_id is not None:
                self._record_fact(
                    before,
                    "The original target and its one visible interaction refinement had no "
                    "observable effect. Treat this interaction family as ineffective on the "
                    "current screen and choose another route.",
                )
            return None
        if outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
            return None
        repeated_transition_count = 1 + sum(
            proposal.action.value == action
            and exact_observation_state_key(before) == exact_observation_state_key(prior_before)
            and exact_observation_state_key(after) == exact_observation_state_key(prior_after)
            for action, prior_before, prior_after in self._transitions
        )
        self._transitions.append((proposal.action.value, before, after))
        if repeated_transition_count == 2:
            self._exhausted_transitions.append((_action_semantics(proposal), before))
            self._record_fact(
                before,
                "The same action repeated an already observed screen transition. "
                "Choose a different route.",
            )
        elif repeated_transition_count >= 3:
            self._exhausted_actions.append((proposal.action.value, before))
            self._record_fact(
                before,
                f"Stop using {proposal.action.value} on this screen. It repeated one "
                "observed screen transition three times. A new target description does "
                "not change the result.",
            )
        prior_screens = self._visited[:-1] if self._visited else ()
        revisited = any(
            observations_match(after, prior, self._comparison) for prior in prior_screens
        )
        self._visited.append(after)
        if not revisited or self._reported_revisit:
            return None
        self._reported_revisit = True
        self._record_fact(
            after,
            "The recent actions returned to an already visited screen. Choose a different route.",
        )
        return None

    def _record_fact(self, observation: Observation, fact: str) -> None:
        self._facts.append(
            PlanningScreenFact(
                screen=planning_screen_reference(observation),
                fact=fact,
            )
        )

    def _interaction_refinement_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> ActionRefusal | None:
        original_id = proposal.interaction_refinement_for_proposal_id
        if original_id is None:
            return None
        original = next(
            (
                candidate
                for candidate, screen in reversed(self._no_change_proposals)
                if candidate.proposal_id == original_id
                and observations_match(observation, screen, self._comparison)
            ),
            None,
        )
        reason = None
        if original is None or original_id != self._latest_feedback_proposal_id:
            reason = "An interaction refinement must reference the immediately preceding no-effect proposal."
        elif original.interaction_refinement_for_proposal_id is not None:
            reason = "An interaction refinement cannot refine another interaction refinement."
        elif original.grounding_correction_for_proposal_id is not None:
            reason = "An interaction refinement cannot refine a grounding correction."
        elif proposal.action is not original.action:
            reason = "An interaction refinement must use the original action."
        elif proposal.target_expression == original.target_expression:
            reason = "An interaction refinement must identify a distinct visible surface."
        if reason is None:
            return None
        return ActionRefusal(
            proposal_id=proposal.proposal_id,
            code=ActionRefusalCode.INVALID_CONSTRAINT,
            reason=reason,
        )


def _action_semantics(proposal: ActionProposal) -> tuple[str, ...]:
    return (
        proposal.action.value,
        proposal.target_expression or "",
        proposal.destination_expression or "",
        proposal.direction.value if proposal.direction is not None else "",
        proposal.distance.value if proposal.distance is not None else "",
        str(proposal.duration_seconds) if proposal.duration_seconds is not None else "",
    )
