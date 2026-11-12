"""Evidence-linked same-run workflow progress and route history."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from yewo.agent.actuation import (
    ACTION_DEFINITIONS,
    ActionBudget,
    ActionOutcome,
    ActionPolicy,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
)
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.planning.evidence import PlannerDecisionSource
from yewo.agent.planning.planner import (
    source_accepts_planning_focus,
)
from yewo.agent.planning.records import (
    PlanningFocusContext,
    ProgressRouteAttempt,
    TaskProgressContext,
)
from yewo.agent.narration import trace
from yewo.agent.planning.history import (
    grounding_correction_failure_reason,
    RECENT_PLANNING_HISTORY_LIMIT,
    PlanningHistoryEntry,
    PlanningScreenFact,
)
from yewo.agent.runtime.interaction_intent import interaction_intent_key
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.agent.runtime.progress_events import (
    InvalidRunProgressAssessment,
    MAXIMUM_FEEDBACK_CHARACTERS,
    RunProgressEvent,
    RunProgressEventStore,
    RunProgressPersistenceError,
)
from yewo.agent.runtime.records import AgentStep
from yewo.agent.runtime.verification import PreDispatchVerification
from yewo.agent.screen_analysis import CurrentScreenAnalysis
from yewo.agent.runtime.workflow_stages import confirmed_stage
from yewo.artifacts import new_record_identifier
from yewo.grounding import GroundingTargetOperation
from yewo.observations import Observation


class RunProgressLedger:
    """Retain completed work and observed routes for one Device Agent run."""

    def __init__(
        self,
        root: Path,
        *,
        screen_comparison: ScreenComparisonPolicy,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._events = RunProgressEventStore(root)
        self._identifier_factory = identifier_factory
        self._work_items: tuple[str, ...] = ()
        self._completed_work_items: list[str] = []
        self._advancing_proposal_ids: set[str] = set()
        self._current_progress: list[str] = []
        self._route_attempts: list[ProgressRouteAttempt] = []
        self._no_progress = ActionNoProgressTracker(screen_comparison)

    def start(
        self,
        work_items: tuple[str, ...],
        observation: Observation,
        base: TaskProgressContext | None,
    ) -> None:
        """Initialize the ledger from the task and any accepted preparation context."""
        self._work_items = tuple(dict.fromkeys(work_items))
        self._no_progress.start(observation)
        self._route_attempts = []
        self._advancing_proposal_ids = set()
        self._current_progress = []
        self._completed_work_items = (
            list(
                dict.fromkeys(
                    item for item in base.completed_work_items if item in self._work_items
                )
            )
            if base is not None
            else []
        )
        self._events.append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._events.next_index(),
                kind="initialized",
                observation_id=observation.observation_id,
                work_items=self._work_items,
                completed_work_items=tuple(self._completed_work_items),
            )
        )

    def progress_context(self, base: TaskProgressContext | None) -> TaskProgressContext:
        """Return compact ledger state for planner and verifier prompts."""
        context = base or TaskProgressContext()
        completed_items = tuple(
            dict.fromkeys(
                item
                for item in (*context.completed_work_items, *self._completed_work_items)
                if item in self._work_items
            )
        )
        remaining_items = tuple(item for item in self._work_items if item not in completed_items)
        routes = tuple((*context.route_attempts, *self._route_attempts)[-100:])
        return context.model_copy(
            update={
                "current_progress": tuple(
                    dict.fromkeys((*context.current_progress, *self._current_progress))
                )[-100:],
                "completed_work_items": completed_items,
                "remaining_work": remaining_items,
                "route_attempts": routes,
            }
        )

    def remaining_work(self) -> tuple[str, ...]:
        """Return canonical task work that has not been completed."""
        completed_items = set(self._completed_work_items)
        return tuple(item for item in self._work_items if item not in completed_items)

    def record_assessment(
        self,
        observation: Observation,
        proposal: ActionProposal,
        completed_work_items: tuple[str, ...],
        proposal_work_item: str | None,
        restarts_completed_work_item: str | None,
    ) -> None:
        """Record one verifier assessment and accept evidence-linked completions."""
        reported_work_items = set(completed_work_items)
        if proposal_work_item is not None:
            reported_work_items.add(proposal_work_item)
        if restarts_completed_work_item is not None:
            reported_work_items.add(restarts_completed_work_item)
        unknown = reported_work_items - set(self._work_items)
        if unknown:
            raise InvalidRunProgressAssessment(
                "Run progress contains a work item outside the task."
            )
        previous_completed = set(self._completed_work_items)
        self._completed_work_items = list(
            dict.fromkeys((*self._completed_work_items, *completed_work_items))
        )
        if set(self._completed_work_items) != previous_completed:
            self._advancing_proposal_ids.add(proposal.proposal_id)
        self._events.append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._events.next_index(),
                kind="assessment",
                observation_id=observation.observation_id,
                completed_work_items=completed_work_items,
                proposal_id=proposal.proposal_id,
                proposal_work_item=proposal_work_item,
                restarts_completed_work_item=restarts_completed_work_item,
            )
        )

    def record_verification(
        self,
        verification: PreDispatchVerification,
        observation: Observation,
        proposal: ActionProposal,
    ) -> ActionRefusal | None:
        """Record valid progress or return a bounded verifier refusal."""
        if not verification.progress_assessed:
            return None
        if (
            verification.proposal_work_item is not None
            and self._no_progress.workflow_route_is_exhausted(
                proposal,
                observation,
                verification.proposal_work_item,
            )
        ):
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.REPEATED_NO_PROGRESS,
                reason=(
                    "This interaction belongs to a workflow route that already returned to its "
                    "starting stage twice for this work item. Choose a different route."
                ),
            )
        try:
            self.record_assessment(
                observation,
                proposal,
                verification.completed_work_items,
                verification.proposal_work_item,
                verification.restarts_completed_work_item,
            )
        except InvalidRunProgressAssessment as error:
            return ActionRefusal(
                proposal_id=proposal.proposal_id,
                code=ActionRefusalCode.VERIFICATION_UNAVAILABLE,
                reason=str(error),
            )
        return None

    def screen_facts(
        self,
        observation: Observation,
        history: tuple[PlanningHistoryEntry, ...] = (),
    ) -> tuple[PlanningScreenFact, ...]:
        """Return deterministic screen facts derived from ledger events."""
        recent_proposal_ids = frozenset(
            entry.proposal.proposal_id for entry in history[-RECENT_PLANNING_HISTORY_LIMIT:]
        )
        return self._no_progress.screen_facts(observation, recent_proposal_ids)

    def pending_planning_focus(
        self,
        observation: Observation,
        source: object,
    ) -> PlanningFocusContext | None:
        """Return the fresh planning view owed to a source that can receive it."""
        if not source_accepts_planning_focus(source):
            return None
        return self._no_progress.pending_planning_focus(
            observation,
            self._identifier_factory,
            self.remaining_work(),
            tuple(self._route_attempts),
        )

    def commit_planning_focus(
        self,
        observation: Observation,
        focus: PlanningFocusContext | None,
        decision_source: PlannerDecisionSource,
    ) -> PlanningFocusContext | None:
        """Claim one focus view only once a planner has actually consumed it."""
        if focus is None or decision_source is not PlannerDecisionSource.MODEL:
            return None
        self._no_progress.commit_planning_focus(observation)
        self._events.append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._events.next_index(),
                kind="planning_focus",
                observation_id=observation.observation_id,
                planning_focus=focus,
            )
        )
        return focus

    def correctable_grounding_proposal_id(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> str | None:
        """Name the no-effect attempt this repeated action may re-ground once."""
        return self._no_progress.correctable_grounding_proposal_id(proposal, observation)

    def refusal_for(
        self,
        proposal: ActionProposal,
        observation: Observation,
    ) -> ActionRefusal | None:
        """Refuse an interaction family that ledger evidence already exhausted."""
        return self._no_progress.refusal_for(proposal, observation)

    @property
    def refusal_deadlock_is_transient(self) -> bool:
        """Return whether the reached refusal deadlock is entirely transient."""
        return self._no_progress.refusal_streak_is_entirely_transient

    def record_refusal(
        self,
        proposal: ActionProposal,
        observation: Observation,
        refusal: ActionRefusal,
        decision_source: PlannerDecisionSource,
    ) -> tuple[tuple[str, ...], str | None]:
        """Persist a refusal and return newly derived facts and terminal feedback."""
        facts_before = self._no_progress.known_facts()
        terminal_reason = (
            self._no_progress.record_system_refusal(proposal, refusal)
            if decision_source is PlannerDecisionSource.SYSTEM
            else self._no_progress.record_refusal(proposal, observation, refusal)
        )
        self._events.append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._events.next_index(),
                kind="refusal",
                observation_id=observation.observation_id,
                proposal_id=proposal.proposal_id,
                interaction_intent_key=interaction_intent_key(proposal),
                refusal_code=refusal.code.value,
                feedback=refusal.reason[:MAXIMUM_FEEDBACK_CHARACTERS],
            )
        )
        return self._no_progress.known_facts()[len(facts_before) :], terminal_reason

    def record_route(
        self,
        step_index: int,
        proposal: ActionProposal,
        proposal_work_item: str | None,
        before: Observation,
        after: Observation,
        outcome: ActionOutcome,
        *,
        task_advanced: bool = False,
    ) -> None:
        """Record one observed route independently from bounded planner history."""
        if proposal_work_item is None:
            return
        route = ProgressRouteAttempt(
            step_index=step_index,
            work_item=proposal_work_item,
            source_context=before.platform_context,
            source_screen_fingerprint=before.screen_fingerprint.value,
            action=proposal.action,
            target=proposal.target_expression,
            interaction_intent_key=interaction_intent_key(proposal),
            destination_context=after.platform_context,
            destination_screen_fingerprint=after.screen_fingerprint.value,
            outcome=outcome.disposition.value,
        )
        self._route_attempts.append(route)
        self._no_progress.record_route(
            proposal,
            before,
            after,
            route,
            task_advanced=task_advanced,
        )
        self._events.append(
            RunProgressEvent(
                event_id=self._identifier_factory(),
                event_index=self._events.next_index(),
                kind="route",
                observation_id=after.observation_id,
                proposal_id=proposal.proposal_id,
                proposal_work_item=proposal_work_item,
                interaction_intent_key=route.interaction_intent_key,
                route=route,
            )
        )

    def record_step(
        self,
        step: AgentStep,
        verification: PreDispatchVerification,
        before: Observation,
        after: Observation,
        current_screen_analysis: CurrentScreenAnalysis | None = None,
    ) -> tuple[tuple[str, ...], str | None]:
        """Record one outcome and return newly derived facts and terminal feedback."""
        if step.outcome is None:
            return (), None
        facts_before = self._no_progress.known_facts()
        terminal_reason = None
        stage = confirmed_stage(
            step.proposal,
            verification.proposal_work_item,
            current_screen_analysis,
            before,
            after,
            step.outcome,
        )
        if stage is not None:
            self._current_progress.append(stage)
            self._events.append(
                RunProgressEvent(
                    event_id=self._identifier_factory(),
                    event_index=self._events.next_index(),
                    kind="stage",
                    observation_id=after.observation_id,
                    proposal_id=step.proposal.proposal_id,
                    proposal_work_item=verification.proposal_work_item,
                    feedback=stage,
                )
            )
        task_advanced = (
            step.proposal.proposal_id in self._advancing_proposal_ids or stage is not None
        )
        if step.decision.source is PlannerDecisionSource.SYSTEM:
            self._no_progress.record_system_outcome(step.proposal)
        else:
            terminal_reason = self._no_progress.record_outcome(
                step.proposal,
                before,
                after,
                step.outcome,
                task_advanced=task_advanced,
            )
        self.record_route(
            step.step_index,
            step.proposal,
            verification.proposal_work_item,
            before,
            after,
            step.outcome,
            task_advanced=task_advanced,
        )
        self._advancing_proposal_ids.discard(step.proposal.proposal_id)
        return self._no_progress.known_facts()[len(facts_before) :], terminal_reason


def _accepts_grounding_correction(proposal: ActionProposal) -> bool:
    """Match the proposal rule that a correction needs one visual point target."""
    definition = ACTION_DEFINITIONS[proposal.action]
    return (
        definition.target_operation is not None
        and definition.target_operation is not GroundingTargetOperation.SCROLL_REGION
        and definition.destination_operation is None
        and (proposal.target_expression is not None or proposal.target_element_ref is not None)
    )


def correct_repeated_grounding(
    progress_ledger: RunProgressLedger,
    proposal: ActionProposal,
    observation: Observation,
    decision_source: PlannerDecisionSource,
) -> tuple[ActionProposal, bool]:
    """Re-ground a repeated target whose recorded point produced no change.

    The planner states the target it wants. The ledger owns the point that already
    failed. Declaring the correction here pairs them deterministically, rather than
    waiting for the planner to cite evidence the runtime already handed it. The
    second value reports whether this call declared the correction.
    """
    if decision_source is PlannerDecisionSource.SYSTEM:
        return proposal, False
    if proposal.grounding_correction_for_proposal_id is not None:
        return proposal, False
    if proposal.interaction_refinement_for_proposal_id is not None:
        return proposal, False
    if not _accepts_grounding_correction(proposal):
        return proposal, False
    failed_proposal_id = progress_ledger.correctable_grounding_proposal_id(proposal, observation)
    if failed_proposal_id is None or failed_proposal_id == proposal.proposal_id:
        return proposal, False
    corrected = proposal.model_copy(
        update={"grounding_correction_for_proposal_id": failed_proposal_id}
    )
    return corrected, True


def gate_before_grounding(
    progress_ledger: RunProgressLedger,
    action_policy: ActionPolicy,
    proposal: ActionProposal,
    observation: Observation,
    action_budget: ActionBudget,
    decision_source: PlannerDecisionSource,
) -> tuple[ActionProposal, ActionRefusal | None]:
    """Re-ground one repeated target, then apply progress and action policy."""
    corrected, _ = correct_repeated_grounding(
        progress_ledger, proposal, observation, decision_source
    )
    refusal = refusal_before_grounding(
        progress_ledger, action_policy, corrected, observation, action_budget, decision_source
    )
    return corrected, refusal


def trace_ineffective_grounding_correction(
    proposal: ActionProposal,
    outcome: ActionOutcome,
) -> None:
    """Record that re-grounding one repeated target changed nothing.

    A correction that changes nothing shows that one target is out of reach. It does not show
    that the objective is, and the planner is often the only component holding evidence of
    another route. The progress ledger already refuses the repeated route, so the run ends when
    the routes run out rather than when one aim fails.
    """
    if grounding_correction_failure_reason(proposal, outcome) is not None:
        trace(f"grounding_correction=ineffective proposal={proposal.proposal_id}")


def refusal_before_grounding(
    progress_ledger: RunProgressLedger,
    action_policy: ActionPolicy,
    proposal: ActionProposal,
    observation: Observation,
    action_budget: ActionBudget,
    decision_source: PlannerDecisionSource,
) -> ActionRefusal | None:
    """Apply run progress before the general action policy."""
    progress_refusal = (
        None
        if decision_source is PlannerDecisionSource.SYSTEM
        else progress_ledger.refusal_for(proposal, observation)
    )
    return progress_refusal or action_policy.refusal_before_grounding(
        proposal,
        observation,
        action_budget,
    )


__all__ = [
    "RunProgressEvent",
    "RunProgressLedger",
    "RunProgressPersistenceError",
    "correct_repeated_grounding",
    "gate_before_grounding",
    "trace_ineffective_grounding_correction",
    "refusal_before_grounding",
]
