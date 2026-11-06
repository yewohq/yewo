"""Detect repeated semantic no-progress attempts and refusals."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.planning.planner import TERMINAL_ACTIONS
from yewo.agent.planning.records import (
    MAXIMUM_PROHIBITED_INTENTS,
    MAXIMUM_SCREEN_FACTS,
    PlanningFocusCause,
    PlanningFocusContext,
    ProgressRouteAttempt,
)
from yewo.agent.planning.history import PlanningScreenFact
from yewo.agent.runtime.attempt_corrections import FailedAttemptCorrections
from yewo.agent.runtime.interaction_intent import AmbiguousInteractionState
from yewo.agent.runtime.planning_focus_state import PlanningFocusState
from yewo.agent.runtime.refusal_streak import RefusalStreak
from yewo.agent.runtime.no_progress_evidence import NoProgressEvidence
from yewo.agent.runtime.screen_no_progress import ScreenNoProgressCounts
from yewo.agent.runtime.workflow_routes import WorkflowRouteExhaustion
from yewo.observations import Observation


CONDITIONAL_REFUSAL_CODES = frozenset(
    {
        ActionRefusalCode.COMPLETED_WORK_REGRESSION,
        ActionRefusalCode.DISCOVERY_INCOMPLETE,
    }
)
TRANSIENT_REFUSAL_CODES = frozenset(
    {
        ActionRefusalCode.PROVIDER_UNAVAILABLE,
        ActionRefusalCode.VERIFICATION_UNAVAILABLE,
    }
)
MAXIMUM_CONSECUTIVE_REFUSALS = 5
MAXIMUM_SCREEN_NO_PROGRESS_EVENTS = 3
_SCREEN_EXIT_ACTIONS = TERMINAL_ACTIONS | {ActionKind.BACK, ActionKind.DISMISS_KEYBOARD}


class ActionNoProgressTracker:
    """Bound repeated semantic failures across equivalent observed screens."""

    def __init__(self, comparison: ScreenComparisonPolicy) -> None:
        self._ambiguous = AmbiguousInteractionState(comparison)
        self._corrections = FailedAttemptCorrections(comparison)
        self._evidence = NoProgressEvidence(comparison)
        self._refusal_streak = RefusalStreak()
        self._screen_counts = ScreenNoProgressCounts(comparison)
        self._workflow_routes = WorkflowRouteExhaustion()
        self._focus_state = PlanningFocusState(comparison)

    def start(self, observation: Observation) -> None:
        """Record the initial logical screen."""
        self._ambiguous.reset()
        self._corrections.start()
        self._evidence.reset(observation)
        self._corrections.record_feedback(None)
        self._refusal_streak.reset()
        self._screen_counts.reset()
        self._workflow_routes.reset()
        self._focus_state.reset()

    def known_facts(self) -> tuple[str, ...]:
        """Return deterministic feedback for the next planning request."""
        return self._evidence.known_facts()

    def screen_facts(
        self,
        observation: Observation,
        recent_proposal_ids: frozenset[str] = frozenset(),
    ) -> tuple[PlanningScreenFact, ...]:
        """Return deterministic feedback learned on this logical screen."""
        return self._evidence.screen_facts(
            observation,
            recent_proposal_ids,
            MAXIMUM_SCREEN_FACTS,
        )

    def pending_planning_focus(
        self,
        observation: Observation,
        identifier_factory: Callable[[], str],
        unresolved_work: tuple[str, ...],
        recent_routes: tuple[ProgressRouteAttempt, ...],
    ) -> PlanningFocusContext | None:
        """Return the focus view owed to this screen without claiming it."""
        return self._focus_state.pending_context(
            observation,
            identifier_factory,
            unresolved_work,
            self._prohibited_intents(observation, unresolved_work),
            recent_routes,
        )

    def commit_planning_focus(self, observation: Observation) -> None:
        """Claim the focus owed to this screen once a planner will receive it."""
        self._focus_state.commit(observation)

    def correctable_grounding_proposal_id(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> str | None:
        """Name the no-effect attempt this repeated action may re-ground once."""
        return self._corrections.correctable_proposal_id(proposal, observation)

    def refusal_for(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> ActionRefusal | None:
        """Refuse a known no-progress action on an equivalent screen."""
        focus_action = self._focus_state.action_armed(observation)
        refinement_refusal = self._corrections.refinement_refusal(proposal, observation)
        if refinement_refusal is not None:
            return refinement_refusal
        if self._corrections.corrects_one_failed_grounding(proposal, observation):
            # A correction carries a minimum distance from the recorded point, so it
            # dispatches somewhere the repeated attempt never reached.
            return None
        if (
            self._screen_counts.count(observation) >= MAXIMUM_SCREEN_NO_PROGRESS_EVENTS
            and not focus_action
            and proposal.action not in _SCREEN_EXIT_ACTIONS
        ):
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    "This screen reached the no-progress limit. Leave this screen or stop the "
                    "task. Do not try another interaction on the same screen state."
                ),
            )
        if self._evidence.exhausted_action(proposal, observation):
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    f"{proposal.action.value} repeated one observed screen transition "
                    "three times on this screen. Choose a different action."
                ),
            )
        if self._ambiguous.count(proposal, observation) >= 2:
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    "The same semantic interaction repeatedly produced an ambiguous outcome. "
                    "Choose a different target or route."
                ),
            )
        if self._evidence.exhausted_transition(proposal, observation):
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    "This semantic action already repeated an equivalent screen transition. "
                    "Choose a different route."
                ),
            )
        prior_refusal = self._evidence.prior_refusal(proposal, observation)
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
        if not self._evidence.has_attempt(proposal, observation):
            return None
        correctable_proposal_id = self._corrections.correctable_proposal_id(proposal, observation)
        if correctable_proposal_id is None:
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    "The same semantic action already produced no observable change "
                    "on an equivalent screen."
                ),
            )
        return ActionRefusal(
            proposal_id=proposal.proposal_id,
            code=ActionRefusalCode.REPEATED_NO_PROGRESS,
            reason=(
                "The same semantic action already produced no observable change on an "
                "equivalent screen. Set grounding_correction_for_proposal_id to "
                f"{correctable_proposal_id} when the recorded point missed the target."
            ),
        )

    def workflow_route_is_exhausted(
        self,
        proposal: ActionProposal,
        observation: Observation,
        work_item: str,
    ) -> bool:
        """Return whether this interaction is exhausted for the verified work item."""
        return self._workflow_routes.prohibits(proposal, observation, work_item)

    def record_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
        refusal: ActionRefusal,
    ) -> str | None:
        """Bound one refused action and report only run-wide refusal deadlock."""
        self._corrections.record_feedback(proposal.proposal_id)
        is_transient = refusal.code in TRANSIENT_REFUSAL_CODES
        self._refusal_streak.record(is_transient)
        if is_transient:
            # Transport availability says nothing about the proposed target or
            # current screen. Keep the run-wide deadlock bound without poisoning
            # target-specific or screen-specific no-progress evidence.
            return self._refusal_deadlock_reason()
        if refusal.code in CONDITIONAL_REFUSAL_CODES:
            self._record_fact(
                observation,
                f"Resolve this before you repeat {proposal.action.value}: {refusal.reason}",
                origin_proposal_id=proposal.proposal_id,
            )
            return self._refusal_deadlock_reason()
        self._record_screen_no_progress(observation, proposal.proposal_id)
        count = self._evidence.record_refusal(proposal, observation, refusal)
        if count == 1:
            self._record_fact(
                observation,
                f"Do not repeat {proposal.action.value}: {refusal.reason}",
                origin_proposal_id=proposal.proposal_id,
            )
        return self._refusal_deadlock_reason()

    def record_system_refusal(
        self,
        proposal: ActionProposal,
        refusal: ActionRefusal,
    ) -> str | None:
        """Count a refused runtime action against deadlock only: no model chose it."""
        self._corrections.record_feedback(proposal.proposal_id)
        self._refusal_streak.record(refusal.code in TRANSIENT_REFUSAL_CODES)
        return self._refusal_deadlock_reason()

    def record_system_outcome(self, proposal: ActionProposal) -> None:
        """Clear deadlock for a dispatched runtime action the screen guard ignores."""
        self._corrections.record_feedback(proposal.proposal_id)
        self._refusal_streak.reset()

    @property
    def refusal_streak_is_entirely_transient(self) -> bool:
        """Return whether every refusal in the current streak was transient."""
        return self._refusal_streak.entirely_transient is True

    def _refusal_deadlock_reason(self) -> str | None:
        if self._refusal_streak.count < MAXIMUM_CONSECUTIVE_REFUSALS:
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
        *,
        task_advanced: bool = False,
    ) -> str | None:
        """Record trusted no-progress and advise after a screen revisit."""
        self._focus_state.consume_action(before)
        self._corrections.record_feedback(proposal.proposal_id)
        self._refusal_streak.reset()
        if ActionOutcomeSignal.APP_EXITED in outcome.signals:
            self._record_screen_no_progress(before, proposal.proposal_id)
            self._evidence.record_exited_transition(proposal, before)
            self._record_fact(
                before,
                f"Do not repeat {proposal.action.value}: the application exited after this action.",
                origin_proposal_id=proposal.proposal_id,
                is_durable_constraint=True,
            )
            return None
        if outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE:
            self._record_screen_no_progress(before, proposal.proposal_id)
            self._evidence.record_attempt(proposal, before)
            self._corrections.record_no_change(proposal, before)
            if proposal.interaction_refinement_for_proposal_id is not None:
                self._record_fact(
                    before,
                    "The original target and its one visible interaction refinement had no "
                    "observable effect. Treat this interaction family as ineffective on the "
                    "current screen and choose another route.",
                    origin_proposal_id=proposal.proposal_id,
                    is_durable_constraint=True,
                )
            return None
        if outcome.disposition is ActionOutcomeDisposition.AMBIGUOUS:
            self._record_screen_no_progress(before, proposal.proposal_id)
            if self._ambiguous.record(proposal, before) == 2:
                fact = (
                    "Repeated attempts did not establish the intended effect. Treat this target "
                    "as unverified and choose a different target or route."
                )
                if proposal.action is ActionKind.FOCUS:
                    fact = (
                        "Repeated focus attempts did not establish an editable control for this "
                        "target. Treat the target as unverified."
                    )
                self._record_fact(
                    before,
                    fact,
                    origin_proposal_id=proposal.proposal_id,
                    is_durable_constraint=True,
                    supersedes_recovery_guidance=True,
                )
                self._focus_state.request(
                    before,
                    PlanningFocusCause.SCREEN_NO_PROGRESS,
                    "The same interaction twice failed to establish its intended effect.",
                )
            return None
        if outcome.disposition is not ActionOutcomeDisposition.OBSERVED:
            return None
        self._ambiguous.clear_observed(proposal, before)
        if task_advanced:
            self._screen_counts.clear(before)
        repeated_transition_count = self._evidence.record_transition(proposal, before, after)
        if repeated_transition_count == 2:
            self._record_fact(
                before,
                "The same action repeated an already observed screen transition. "
                "Choose a different route.",
                origin_proposal_id=proposal.proposal_id,
                is_durable_constraint=True,
            )
            self._focus_state.request(
                after,
                PlanningFocusCause.REPEATED_TRANSITION,
                "An observed screen transition repeated without advancing the task.",
            )
        elif repeated_transition_count >= 3:
            self._record_fact(
                before,
                f"Stop using {proposal.action.value} on this screen. It repeated one "
                "observed screen transition three times. A new target description does "
                "not change the result.",
                origin_proposal_id=proposal.proposal_id,
                is_durable_constraint=True,
            )
        if not self._evidence.record_revisit(after):
            return None
        self._record_fact(
            after,
            "The recent actions returned to an already visited screen. Choose a different route.",
            origin_proposal_id=proposal.proposal_id,
            is_durable_constraint=True,
        )
        return None

    def record_route(
        self,
        proposal: ActionProposal,
        before: Observation,
        after: Observation,
        route: ProgressRouteAttempt,
        *,
        task_advanced: bool,
    ) -> bool:
        """Record one work-linked route and exhaust a repeated closed workflow."""
        if not self._workflow_routes.record(route, task_advanced=task_advanced):
            return False
        self._record_fact(
            after,
            "This workflow route returned to its starting stage twice without advancing the "
            "task. Do not repeat or rename an interaction from that route.",
            origin_proposal_id=proposal.proposal_id,
            is_durable_constraint=True,
        )
        self._focus_state.request(
            after,
            PlanningFocusCause.REPEATED_TRANSITION,
            "A complete workflow route repeated without task progress.",
        )
        return True

    def _record_fact(
        self,
        observation: Observation,
        fact: str,
        *,
        origin_proposal_id: str,
        is_durable_constraint: bool = False,
        supersedes_recovery_guidance: bool = False,
    ) -> None:
        self._evidence.record_fact(
            observation,
            fact,
            origin_proposal_id=origin_proposal_id,
            is_durable_constraint=is_durable_constraint,
            supersedes_recovery_guidance=supersedes_recovery_guidance,
        )

    def _record_screen_no_progress(
        self,
        observation: Observation,
        origin_proposal_id: str,
    ) -> None:
        count = self._screen_counts.record(observation)
        if count == MAXIMUM_SCREEN_NO_PROGRESS_EVENTS:
            self._record_fact(
                observation,
                "This screen reached the no-progress limit. Leave this screen or stop the task.",
                origin_proposal_id=origin_proposal_id,
                is_durable_constraint=True,
            )
            self._focus_state.request(
                observation,
                PlanningFocusCause.SCREEN_NO_PROGRESS,
                "Three interactions or refusals produced no task progress on this screen.",
            )

    def _prohibited_intents(
        self,
        observation: Observation,
        work_items: tuple[str, ...],
    ) -> tuple[str, ...]:
        return self._evidence.prohibited_intents(
            observation,
            self._ambiguous,
            self._workflow_routes,
            work_items,
            MAXIMUM_PROHIBITED_INTENTS,
        )
