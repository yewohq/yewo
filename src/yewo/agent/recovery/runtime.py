"""Run-scoped recovery guidance and capped logical budget credit."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.memory.records import ScreenComparisonPolicy
from yewo.agent.narration import narrate_recovery_unavailable
from yewo.agent.planning.planner import PlanningRequest, TaskProgressContext
from yewo.agent.recovery.advisor import (
    RecoveryAdvisor,
    RecoveryAssessment,
    RecoveryDisposition,
)
from yewo.agent.recovery.records import RecoveryBudgetCredit
from yewo.agent.runtime.records import AgentStep
from yewo.agent.memory.screens import observations_match
from yewo.observations import Observation


MAXIMUM_CREDITED_RECOVERY_STEPS = 10


class RecoveryRunState:
    """Own one advisor decision and its bounded effective-budget adjustment."""

    def __init__(
        self,
        advisor: RecoveryAdvisor | None,
        screen_comparison: ScreenComparisonPolicy,
    ) -> None:
        self._advisor = advisor
        self._screen_comparison = screen_comparison
        self._assessment: RecoveryAssessment | None = None
        self._frontier: Observation | None = None
        self._active = False
        self._baseline_step_count = 0
        self._credited_step_count = 0
        self._planning_requests = 0
        self._grounding_requests = 0
        self._dispatched_actions = 0
        self._started_at: float | None = None
        self._finished_at: float | None = None

    @property
    def assessment(self) -> RecoveryAssessment | None:
        """Return the single advisor decision when recovery was assessed."""
        return self._assessment

    @property
    def is_active(self) -> bool:
        """Return whether the runtime is restoring the pre-failure frontier."""
        return self._active

    def assess(
        self,
        planning_request: PlanningRequest,
        failed_step: AgentStep,
        recovery_observation: Observation,
        completed_step_count: int,
        now: float,
    ) -> RecoveryAssessment | None:
        """Run the advisor once and start a bounded frontier-restoration interval."""
        if self._advisor is None or self._assessment is not None:
            return None
        try:
            assessment = self._advisor.assess(
                planning_request,
                failed_step,
                recovery_observation,
            )
        except Exception:  # noqa: BLE001 - advice must never end one run
            narrate_recovery_unavailable()
            self._planning_requests += 1
            self._finished_at = now
            return None
        self._assessment = assessment
        self._planning_requests += 1
        self._baseline_step_count = completed_step_count
        self._started_at = now
        if assessment.disposition is RecoveryDisposition.CONTINUE:
            pre_failure_observation = planning_request.observation
            self._frontier = pre_failure_observation
            self._active = not observations_match(
                pre_failure_observation,
                recovery_observation,
                self._screen_comparison,
            )
        if not self._active:
            self._finished_at = now
        return assessment

    def progress_context(
        self,
        base: TaskProgressContext | None = None,
    ) -> TaskProgressContext | None:
        """Convert continuing recovery guidance into planner-owned run context."""
        assessment = self._assessment
        if assessment is None or assessment.disposition is RecoveryDisposition.STOP:
            return None
        context = base or TaskProgressContext(remaining_work=assessment.remaining_work)
        failed_branch = (
            "Do not repeat the application-failing action without a material state change."
        )
        return context.model_copy(
            update={
                "remaining_work": assessment.remaining_work,
                "exhausted_branches": tuple(
                    dict.fromkeys((*context.exhausted_branches, failed_branch))
                ),
                "recovery_guidance": assessment.planner_guidance,
            }
        )

    def credit_planning_requests(self, count: int) -> None:
        """Credit planner calls used while restoring the recorded frontier."""
        if self._active:
            self._planning_requests += count

    def credit_grounding_requests(self, count: int) -> None:
        """Credit grounding calls used while restoring the recorded frontier."""
        if self._active:
            self._grounding_requests += count

    def credit_dispatched_action(self) -> None:
        """Credit one dispatched action used while restoring the frontier."""
        if self._active:
            self._dispatched_actions += 1

    def observe(self, observation: Observation, completed_step_count: int, now: float) -> None:
        """End credit at the matched frontier or at the fixed recovery-step cap."""
        if not self._active:
            return
        self._credited_step_count = min(
            completed_step_count - self._baseline_step_count,
            MAXIMUM_CREDITED_RECOVERY_STEPS,
        )
        frontier = self._frontier
        if (
            frontier is not None
            and observations_match(frontier, observation, self._screen_comparison)
        ) or self._credited_step_count >= MAXIMUM_CREDITED_RECOVERY_STEPS:
            self._active = False
            self._finished_at = now

    def observe_if_active(
        self,
        observation: Observation,
        completed_step_count: int,
        clock: Callable[[], float],
    ) -> None:
        """Observe a recovery step without reading the clock outside recovery."""
        if self._active:
            self.observe(observation, completed_step_count, clock())

    def credited_steps(self, completed_step_count: int) -> int:
        """Return the frozen or currently eligible recovery-step count."""
        if self._assessment is None:
            return 0
        if self._active:
            return min(
                completed_step_count - self._baseline_step_count,
                MAXIMUM_CREDITED_RECOVERY_STEPS,
            )
        return self._credited_step_count

    def settle(self, completed_step_count: int, now: float) -> None:
        """End the recovery interval when non-observation steps reach its cap."""
        if not self._active:
            return
        credited_steps = self.credited_steps(completed_step_count)
        if credited_steps < MAXIMUM_CREDITED_RECOVERY_STEPS:
            return
        self._credited_step_count = credited_steps
        self._active = False
        self._finished_at = now

    def credit(self, completed_step_count: int, now: float) -> RecoveryBudgetCredit:
        """Build persisted gross-to-effective accounting without changing recovery state."""
        assessment = self._assessment
        if assessment is None:
            return RecoveryBudgetCredit()
        credited_steps = self.credited_steps(completed_step_count)
        finished_at = self._finished_at or now
        started_at = self._started_at or finished_at
        return RecoveryBudgetCredit(
            assessment_id=assessment.assessment_id,
            steps=credited_steps,
            planning_requests=min(self._planning_requests, 11),
            grounding_requests=min(self._grounding_requests, 20),
            dispatched_actions=min(self._dispatched_actions, 10),
            duration_seconds=min(max(0.0, finished_at - started_at), 600.0),
        )
