"""Apply fresh screen audits while either a saved or planned path runs."""

import logging
from typing import Sequence

from yewo.agent.actuation.actions import Action, ActionType, WaitAction
from yewo.agent.run.outcomes import ObservationDirective
from yewo.testing.judge import rules
from yewo.testing.judge.qa_auditor import ScreenAssessment, ScreenAuditSession

logger = logging.getLogger(__name__)

LAUNCH_RECHECK_WAIT = 1.5
CHECKPOINTED_ACTIONS = frozenset(
    {
        ActionType.TAP,
        ActionType.DOUBLE_TAP,
        ActionType.LONG_PRESS,
        ActionType.DRAG_AND_DROP,
        ActionType.KEY_EVENT,
        ActionType.CAPABILITY,
    }
)


def record_issue_step(assessments: Sequence[ScreenAssessment], step: int) -> None:
    """Stamp new issues with the scenario step that exposed them."""
    for assessment in assessments:
        for issue in assessment.detected_issues:
            if issue.step is None:
                issue.step = step


class LiveScreenAudits:
    """Send every observation through one run-local audit session."""

    def __init__(
        self,
        session: ScreenAuditSession,
        obligation_session: ScreenAuditSession | None = None,
    ):
        self._session = session
        self._obligation_session = obligation_session
        self._last_snapshot = None
        self._observations = 0
        self._launch_recheck_pending = False
        self._last_step = 0
        self.assessments: list[ScreenAssessment] = []
        self.stop_reason: str | None = None

    def observe(self, snapshot, step: int) -> ObservationDirective:
        """Audit this reading before the run chooses or executes an action."""
        self._observations += 1
        self._last_step = step
        current = self._session.audit_snapshot(snapshot)
        if self._obligation_session is not None:
            current.extend(self._obligation_session.audit_snapshot(snapshot))
        self._last_snapshot = snapshot
        return self._record(current, step, allow_launch_recheck=True)

    def before_action(self, action: Action, step: int) -> ObservationDirective:
        """Audit pending app state before a potentially irreversible action."""
        if action.type not in CHECKPOINTED_ACTIONS:
            return ObservationDirective.CONTINUE
        return self._record(
            self._session.audit_pending(), step, allow_launch_recheck=False
        )

    def _record(
        self,
        current: list[ScreenAssessment],
        step: int,
        *,
        allow_launch_recheck: bool,
    ) -> ObservationDirective:
        """Store fresh assessments and apply their immediate directive."""
        record_issue_step(current, step)
        crashes = [
            issue
            for assessment in current
            for issue in assessment.detected_issues
            if issue.issue_type == "crash_screen"
        ]
        if crashes and allow_launch_recheck and self._observations == 1:
            logger.warning("Crash screen on the first look — re-checking next step.")
            for assessment in current:
                assessment.detected_issues = [
                    issue
                    for issue in assessment.detected_issues
                    if issue.issue_type != "crash_screen"
                ]
            self._session.request_repeat_audit()
            self._launch_recheck_pending = True
            self.assessments.extend(current)
            return ObservationDirective.PLAN

        self.assessments.extend(current)
        if crashes:
            logger.error("App crash screen detected — stopping scenario.")
            self.stop_reason = "App crashed: crash screen detected."
            return ObservationDirective.STOP
        critical = next(
            (
                issue
                for assessment in current
                for issue in assessment.detected_issues
                if issue.severity == "critical"
            ),
            None,
        )
        if critical is not None:
            logger.error("Critical current-screen issue detected — stopping scenario.")
            self.stop_reason = f"Critical QA issue: {critical.description}"
            return ObservationDirective.STOP
        violated_security_rule = next(
            (
                verification
                for assessment in current
                for verification in assessment.rule_verifications
                if verification.applicable
                and verification.status == "failed"
                and rules.is_kind(verification.rule_id, rules.SECURITY)
            ),
            None,
        )
        if violated_security_rule is not None:
            logger.error("A security rule failed — stopping before the next action.")
            self.stop_reason = (
                f"Security rule violated: {violated_security_rule.rule}"
            )
            return ObservationDirective.STOP
        return ObservationDirective.CONTINUE

    def apply_launch_recheck(self, step_response) -> bool:
        """Replace the action selected for a withdrawn first-look crash."""
        if not self._launch_recheck_pending:
            return False
        self._launch_recheck_pending = False
        if not getattr(step_response, "is_scenario_feasible", True):
            logger.warning("Infeasible on the first look — re-checking next step.")
            step_response.is_scenario_feasible = True
        self._wait(step_response)
        return True

    def defer_first_infeasible(self, step_response) -> None:
        """Re-read an app that may still be drawing before calling it infeasible."""
        logger.warning("Infeasible on the first look — re-checking next step.")
        step_response.is_scenario_feasible = True
        self._session.request_repeat_audit()
        self._wait(step_response)

    def finalize(self, final_obligation_session: ScreenAuditSession | None = None) -> None:
        """Audit run-end obligations against the final observed state."""
        final = self._session.finalize_audit()
        if final_obligation_session is not None and self._last_snapshot is not None:
            final.extend(final_obligation_session.audit_snapshot(self._last_snapshot))
        record_issue_step(final, max(1, self._last_step))
        self.assessments.extend(final)

    @staticmethod
    def _wait(step_response) -> None:
        step_response.action = WaitAction(
            duration=LAUNCH_RECHECK_WAIT,
            reason="Looking again before judging a screen the app may still be drawing.",
        )
