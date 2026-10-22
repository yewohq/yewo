"""
QAAgent — testing layer built on top of DeviceAgent.
Injects QAPlanner per-scenario and collects QA outputs via the step hook.
Returns ScenarioReport with all verifications, detected issues, and status.
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

from yewo.agent.run.agent import DeviceAgent, DeviceRunResult
from yewo.agent.plan.planner import DeviceStepResponse
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.device.platform import Platform
from yewo.agent.memory.replay import ReplayLedger

from yewo.defaults import STEP_BUDGET_FLOOR, STEP_BUDGET_PER_INSTRUCTION
from yewo.llm.failures import reportable
from yewo.llm.llm_client import LLMClient
from yewo.testing.models import (
    ExpectedOutcome,
    GeneratedTestCase,
    QAIssue,
    ReceiptRecord,
    RuleVerification,
    ScenarioReport,
    SensitiveFinding,
    VerdictInputs,
)
from yewo.agent.device import restoration
from yewo.testing.judge.capabilities import target_of, unsupported_reason
from yewo.agent.plan.planner import GROUNDING_COORDINATES
from yewo.testing.judge import decide
from yewo.testing.judge.qa_auditor import Assessment, QAAuditor, ScreenWatch
from yewo.testing.qa_planner import QAPlanner, QAStepResponse
from yewo.errors import Cause, cause_of
from yewo.testing import steps, taxonomy
from yewo.testing.judge import transitions

from yewo.agent.actuation.actions import WaitAction

logger = logging.getLogger(__name__)

LAUNCH_RECHECK_WAIT = 1.5


def _verdict_inputs(
    run_result, stop_reason, qa_outputs, assessments, test_case,
    security_rules, all_findings,
) -> VerdictInputs:
    """What the judge was given, kept so the verdict can be reached again.

    Written beside the verdict rather than derived later. Nothing else in the
    record holds the walk's own status, what the planner last said about
    proceeding, or which device states the target reported back.
    """
    last = qa_outputs[-1] if qa_outputs else None
    return VerdictInputs(
        run_status=run_result.status,
        stop_reason=stop_reason,
        is_scenario_feasible=last.is_scenario_feasible if last else True,
        infeasibility_reason=last.infeasibility_reason if last else None,
        screens_judged=len(assessments),
        expected_outcomes=list(test_case.expected_outcomes),
        security_rules=list(security_rules),
        sensitive_findings=list(all_findings),
        required_capabilities=list(test_case.required_capabilities),
        receipts=[
            ReceiptRecord(capability=receipt.capability, confirmed=receipt.confirmed)
            for receipt in run_result.capability_receipts
        ],
    )


def _defer_first_look(step_response, assessment, first_look: bool) -> bool:
    """Withdraws a terminal verdict taken before the app may have finished drawing.

    A launching app shows an empty screen for a moment, which reads as both
    crashed and infeasible. One more step costs little. A false crash report
    costs a red build.

    The response's own action has to be withdrawn too. It is almost always
    `done`, and the run loop acts on it the moment this returns. Clearing the
    verdict alone would change the label and nothing else.

    Returns whether a verdict was withdrawn.
    """
    if not first_look:
        return False

    deferred = False
    if any(i.issue_type == "crash_screen" for i in assessment.detected_issues):
        logger.warning("Crash screen on the first look — re-checking next step.")
        assessment.detected_issues = [
            i for i in assessment.detected_issues if i.issue_type != "crash_screen"
        ]
        deferred = True
    if not step_response.is_scenario_feasible:
        logger.warning("Infeasible on the first look — re-checking next step.")
        step_response.is_scenario_feasible = True
        deferred = True

    if deferred:
        # Whatever the action was, it was chosen from the same screen the
        # verdict came from. Acting on it would trust a reading we just said
        # not to trust — and `done` would end the run before the second look.
        step_response.action = WaitAction(
            duration=LAUNCH_RECHECK_WAIT,
            reason="Looking again before judging a screen the app may still be drawing.",
        )
    return deferred


def _seen_at(assessments, step: int) -> None:
    """Records which step each issue was found after.

    An auditor is shown one screen and does not know where in a walk it
    sits, so the run stamps it. Without this a report can only match on
    the screen name. That finds the first visit to a screen rather than
    the visit that saw the defect.

    An issue already stamped keeps its step. A closing look repeats what
    an earlier look returned, and the earlier step is the true one.
    """
    for assessment in assessments:
        for issue in assessment.detected_issues:
            if issue.step is None:
                issue.step = step


class QAAgent:
    """Runs one QA scenario, and keeps the clients a journey shares.

    One QAAgent serves every scenario in a journey, so the model clients are
    built once. Each scenario gets its own QAPlanner and its own QAAuditor,
    because each carries state that must not reach the next scenario.
    """

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 40,
        settle_time: float = 2.0,
        headless: bool = True,
        fast_model: str = None,
        replay_ledger: Optional[ReplayLedger] = None,
        grounding: str = GROUNDING_COORDINATES,
    ):
        self._grounding = grounding

        self._device_agent = DeviceAgent(
            model=model,
            max_steps=max_steps,
            settle_time=settle_time,
            fast_model=fast_model,
            replay_ledger=replay_ledger,
            grounding=grounding,
        )
        self._fast_llm: LLMClient = self._device_agent.fast_llm

    def _step_budget(self, test_case: GeneratedTestCase) -> int:
        """How many agent steps a scenario may use.

        Scenarios differ widely in size, and a short check cannot share one
        number with a long end-to-end order. So each gets a budget from its own
        instructions, under the configured ceiling.
        """
        wanted = len(test_case.steps_instructions) * STEP_BUDGET_PER_INSTRUCTION
        ceiling = self._device_agent.max_steps
        return min(ceiling, max(STEP_BUDGET_FLOOR, wanted))

    def execute_scenario(
        self,
        test_case: GeneratedTestCase,
        security_rules: List[str],
        device_control,
        perception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
    ) -> ScenarioReport:
        """Execute one test scenario and return a complete ScenarioReport."""
        start_time = time.time()

        qa_outputs: List[QAStepResponse] = []
        assessments: List[Assessment] = []
        stop_reason: dict = {"value": None}  # mutable so the nested hook can write out

        # Only what one screen can settle is put to the auditor. Asked whether
        # the app hung, it answers from pixels that cannot carry the fact.
        auditor = QAAuditor(
            self._fast_llm,
            test_case.expected_outcomes,
            security_rules,
            scrubber=self._device_agent.scrubber,
        )

        watch = ScreenWatch(auditor)

        def step_hook(step_response: DeviceStepResponse, snapshot) -> Optional[bool]:
            if not isinstance(step_response, QAStepResponse):
                return None

            qa_outputs.append(step_response)
            # The screen is audited by something that was not told the plan, so
            # a verdict here is capable of contradicting the action above it.
            fresh = watch.look(snapshot)
            for assessment in fresh:
                assessment.screen_name = (
                    assessment.screen_name or step_response.current_screen
                )
            _seen_at(fresh, len(qa_outputs))
            assessments.extend(fresh)

            if fresh and _defer_first_look(
                step_response, fresh[-1], len(qa_outputs) == 1
            ):
                # The withdrawal promised another look; this is what keeps it.
                watch.look_again()
                return None

            if any(
                issue.issue_type == "crash_screen"
                for assessment in fresh
                for issue in assessment.detected_issues
            ):
                logger.error("App crash screen detected — stopping scenario.")
                stop_reason["value"] = "App crashed: crash screen detected."
                return False

            if not step_response.is_scenario_feasible:
                logger.warning(
                    "Scenario flagged infeasible: %s",
                    self._device_agent.scrubber.scrub(step_response.infeasibility_reason),
                )
                stop_reason["value"] = (
                    "Scenario is infeasible: "
                    + self._device_agent.scrubber.scrub(
                        step_response.infeasibility_reason or "No reason provided"
                    )
                )
                return False

            return None

        # Read against the device in front of us, so a scenario whose
        # capability this target cannot apply is skipped rather than failing a
        # step inside itself.
        target = target_of(device_control)
        unsupported = unsupported_reason(test_case, target)
        if unsupported:
            return ScenarioReport(
                scenario_uuid=test_case.scenario_uuid,
                test_id=test_case.test_id,
                scenario_name=test_case.scenario_name,
                cell=test_case.cell,
                status="UNSUPPORTED",
                cause=Cause.SCENARIO.value,
                duration=round(time.time() - start_time, 2),
                failure_reason=unsupported,
            )

        # OCR caches are useful within a scenario but are evidence from the
        # wrong run once the app has been restarted.
        perception.reset()

        qa_planner = QAPlanner(
            llm_client=self._fast_llm,
            setup=test_case.setup,
            scrubber=self._device_agent.scrubber,
            grounding=self._grounding,
        )

        replay = self._device_agent.replay_ledger
        # Replay skips perception and auditing on the screens it walks, so a
        # cell whose claims apply mid-navigation must not use it.
        audits_every_screen = taxonomy.audits_every_screen(
            test_case.category, test_case.scenario_type
        )
        if audits_every_screen:
            # Replayed screens are never perceived or audited, which is the
            # whole point of these scenarios.
            self._device_agent.replay_ledger = None

        step_budget = self._step_budget(test_case)
        logger.info(
            "Scenario %s: %d instruction(s), %d step budget",
            test_case.test_id,
            len(test_case.steps_instructions),
            step_budget,
        )

        try:
            # Scenarios share one device, and SetupMode restores the app and
            # not the system around it. Without this, one scenario that goes
            # offline decides what every later scenario was testing.
            # The typed capabilities say which states could move, so iOS
            # reads only those rather than all of them. Read off the steps,
            # because a declared kind with no step applies nothing.
            with restoration.preserved(
                device_control, capabilities=steps.capabilities_in(test_case.steps)
            ):
                run_result: DeviceRunResult = self._device_agent.run(
                    goal=test_case.steps_instructions,
                    device_control=device_control,
                    perception=perception,
                    session_dir=session_dir,
                    session_trail=session_trail,
                    step_hook=step_hook,
                    planner_override=qa_planner,
                    max_steps=step_budget,
                    setup=test_case.setup,
                )
        finally:
            self._device_agent.replay_ledger = replay

        # Nothing ever leaves the screen a scenario ended on, so it is read here.
        last = watch.close()
        _seen_at(last, len(qa_outputs))
        assessments.extend(last)

        return self._build_report(
            test_case=test_case,
            run_result=run_result,
            qa_outputs=qa_outputs,
            assessments=assessments,
            stop_reason=stop_reason["value"],
            duration=time.time() - start_time,
            security_rules=security_rules,
        )

    def _build_report(
        self,
        test_case: GeneratedTestCase,
        run_result: DeviceRunResult,
        qa_outputs: List[QAStepResponse],
        assessments: List[Assessment],
        stop_reason: Optional[str],
        duration: float,
        security_rules: List[str],
    ) -> ScenarioReport:
        all_verifications: List[RuleVerification] = []
        all_issues: List[QAIssue] = []
        all_findings: List[SensitiveFinding] = []

        scrub = self._device_agent.scrubber.scrub

        for assessment in assessments:
            for verify in assessment.rule_verifications:
                if not verify.screen_name:
                    verify.screen_name = assessment.screen_name
                # An audit naming the value it found — "password123 is visible
                # unmasked" — would otherwise write it into logs and reports.
                verify.rule = scrub(verify.rule)
                verify.reasoning = scrub(verify.reasoning)
                all_verifications.append(verify)

            for finding in assessment.sensitive_findings:
                finding.screen_name = assessment.screen_name
                finding.quoted = scrub(finding.quoted)
                all_findings.append(finding)

            for issue in assessment.detected_issues:
                issue.current_screen = assessment.screen_name
                issue.description = scrub(issue.description)
                all_issues.append(issue)
                logger.info(
                    "QA issue [%s/%s] on '%s': %s",
                    issue.severity, issue.issue_type,
                    assessment.screen_name, issue.description,
                )

        all_verifications.extend(
            self._settled_transitions(test_case, assessments)
        )

        status, failure_reason, cause = self._determine_status(
            run_result=run_result,
            stop_reason=stop_reason,
            all_verifications=all_verifications,
            all_issues=all_issues,
            qa_outputs=qa_outputs,
            assessments=assessments,
            required_outcomes=test_case.expected_outcomes,
            required_capabilities=test_case.required_capabilities,
            security_rules=security_rules,
            all_findings=all_findings,
        )

        return ScenarioReport(
            scenario_uuid=test_case.scenario_uuid,
            test_id=test_case.test_id,
            scenario_name=test_case.scenario_name,
            cell=test_case.cell,
            status=status,
            cause=cause.value if cause else None,
            duration=round(duration, 2),
            steps_executed=run_result.steps_executed,
            rule_verifications=all_verifications,
            detected_issues=all_issues,
            steps=list(run_result.steps),
            failure_reason=scrub(failure_reason or "") or None,
            verdict_inputs=_verdict_inputs(
                run_result, stop_reason, qa_outputs, assessments,
                test_case, security_rules, all_findings,
            ),
        )

    def _settled_transitions(
        self, test_case: GeneratedTestCase, assessments: List[Assessment]
    ) -> List[RuleVerification]:
        """Evidence for the outcomes no single screenshot could carry.

        The gate requires every expected outcome, and the auditor is never
        shown the ones about reaching a screen. Without this they are demanded
        and unanswerable, and a scenario that did everything asked of it
        reports missing evidence.

        A failure here leaves them unevidenced, which is the honest answer:
        the judgement did not happen, so nothing was judged.

        A failure of the service is the exception, whether it was spent or
        refused. Read as missing evidence it would blame the run for an
        outage, so it ends the scenario on the provider instead.
        """
        try:
            return transitions.settle(
                self._fast_llm,
                test_case.expected_outcomes,
                assessments,
                scrubber=self._device_agent.scrubber,
            )
        except Exception as exc:  # noqa: BLE001 - model boundary
            if cause_of(exc) is Cause.PROVIDER:
                raise
            logger.warning(
                "Could not judge the transitions of this scenario: %s",
                reportable(exc),
            )
            return []

    def _determine_status(
        self,
        run_result: DeviceRunResult,
        stop_reason: Optional[str],
        all_verifications: List[RuleVerification],
        all_issues: List[QAIssue],
        qa_outputs: List[QAStepResponse],
        assessments: List[Assessment],
        required_outcomes: Sequence[ExpectedOutcome | str],
        security_rules: Sequence[str],
        all_findings: Optional[List[SensitiveFinding]] = None,
        required_capabilities: Sequence = (),
    ):
        """The verdict for this scenario, decided in `judge.decide`.

        Kept as a method because the decision is read here. Moved out because
        it reads no part of this agent. The eval corpus calls the function
        with stored evidence, and with no device.
        """
        return decide.determine_status(
            run_result=run_result,
            stop_reason=stop_reason,
            all_verifications=all_verifications,
            all_issues=all_issues,
            qa_outputs=qa_outputs,
            assessments=assessments,
            required_outcomes=required_outcomes,
            security_rules=security_rules,
            all_findings=all_findings,
            required_capabilities=required_capabilities,
        )
