"""The testing layer around DeviceAgent."""

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
    ReportStatus,
    RuleVerification,
    ScenarioReport,
    SensitiveFinding,
    Measurement,
    VerdictInputs,
)
from yewo.agent.device import restoration
from yewo.agent.device.capability_controller import CapabilityController
from yewo.agent.device.launch import SetupMode
from yewo.agent.device.receipts import Receipt
from yewo.testing.judge.capabilities import (
    target_of,
    unavailable_metric_reason,
    unsupported_reason,
)
from yewo.agent.plan.planner import GROUNDING_COORDINATES
from yewo.testing.judge import decide
from yewo.testing.judge.qa_auditor import QAAuditor, ScreenAssessment, ScreenAuditSession
from yewo.testing.qa_planner import QAPlanner, QAStepResponse
from yewo.errors import Cause, cause_of
from yewo.testing import steps, taxonomy
from yewo.testing.lifecycle import (
    LifecyclePhase,
    LifecycleStatus,
    apply_app_precondition,
    apply_capability_precondition,
    failed_operation,
    restore_app_precondition,
    restoration_operations,
)
from yewo.testing.judge import transitions
from yewo.testing.judge.screen_audits import LiveScreenAudits

logger = logging.getLogger(__name__)


def _verdict_inputs(
    run_result,
    stop_reason,
    qa_outputs,
    assessments,
    test_case,
    security_rules,
    all_findings,
    measurement_samples=(),
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
        measurements=list(measurement_samples),
        receipts=[
            ReceiptRecord(capability=receipt.capability, confirmed=receipt.confirmed)
            for receipt in run_result.capability_receipts
        ],
        postcondition_verifications=list(run_result.postcondition_verifications),
    )


class QAAgent:
    """Runs one QA scenario, and keeps the clients a journey shares.

    One QAAgent serves every scenario in a journey, so the model clients are
    built once. Each scenario gets its own QAPlanner and its own QAAuditor,
    because each carries state that must not reach the next scenario.
    """

    _trace_store = None
    """Where a judged path is kept between runs.

    A class default, so an agent built without __init__ still answers
    the question a scenario asks of it. None is durable replay off.
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
        trace_store=None,
        cache_affinity: Optional[str] = None,
    ):
        self._grounding = grounding
        # Where a judged path is kept between runs. None turns durable
        # replay off, which is what --no-replay gives this.
        self._trace_store = trace_store

        self._device_agent = DeviceAgent(
            model=model,
            max_steps=max_steps,
            settle_time=settle_time,
            fast_model=fast_model,
            replay_ledger=replay_ledger,
            grounding=grounding,
        )
        self._device_agent.llm.set_cache_affinity(cache_affinity)
        self._device_agent.fast_llm.set_cache_affinity(cache_affinity)
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
        stop_reason: dict = {"value": None}  # mutable so the nested hook can write out

        # Only what one screen can settle is put to the auditor. Asked whether
        # the app hung, it answers from pixels that cannot carry the fact.
        auditor = QAAuditor(
            self._fast_llm,
            test_case.expected_outcomes,
            security_rules,
            scrubber=self._device_agent.scrubber,
        )

        replay = self._build_replay(test_case)
        screen_audits = LiveScreenAudits(ScreenAuditSession(auditor))

        def step_hook(step_response: DeviceStepResponse, snapshot) -> Optional[bool]:
            if not isinstance(step_response, QAStepResponse):
                return None

            qa_outputs.append(step_response)
            screen_audits.label_latest(step_response.current_screen)
            if screen_audits.apply_launch_recheck(step_response):
                return None

            if not step_response.is_scenario_feasible:
                if len(qa_outputs) == 1:
                    screen_audits.defer_first_infeasible(step_response)
                    return None
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
        preflight_reason = unsupported_reason(test_case, target)
        preflight_status = ReportStatus.UNSUPPORTED
        if preflight_reason is None:
            preflight_reason = unavailable_metric_reason(test_case)
            preflight_status = ReportStatus.UNVERIFIED
        if preflight_reason:
            return ScenarioReport(
                scenario_uuid=test_case.scenario_uuid,
                test_id=test_case.test_id,
                scenario_name=test_case.scenario_name,
                cell=test_case.cell,
                status=preflight_status,
                cause=Cause.SCENARIO.value,
                duration=round(time.time() - start_time, 2),
                failure_reason=preflight_reason,
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

        in_run_ledger = self._device_agent.replay_ledger
        # The in-run ledger walks screens without reading them. A security
        # rule becomes required only once a screen reported it applicable,
        # so a screen it skipped takes that rule out of the required set
        # and a violation there passes. Durable replay audits what it
        # arrives at, which is why only this one is dropped.
        audits_every_screen = taxonomy.must_audit_every_screen(
            test_case.category, test_case.scenario_type, security_rules
        )
        if audits_every_screen:
            self._device_agent.replay_ledger = None

        self._device_agent.durable_replay = replay

        step_budget = self._step_budget(test_case)
        logger.info(
            "Scenario %s: %d instruction(s), %d step budget",
            test_case.test_id,
            len(test_case.steps_instructions),
            step_budget,
        )

        lifecycle_operations = []
        initial_receipts: list[Receipt] = []
        restoration_events: list[restoration.RestorationEvent] = []
        precondition_failure: Optional[Exception] = None
        try:
            # Preserve every typed state this scenario can change.
            with restoration.preserved(
                device_control,
                capabilities=[
                    *test_case.initial_capabilities,
                    *steps.capabilities_in(test_case.steps),
                ],
                events=restoration_events,
            ):
                try:
                    try:
                        lifecycle_operations.append(
                            apply_app_precondition(
                                device_control, test_case.setup, test_case.setup_origin
                            )
                        )
                        controller = CapabilityController(device_control)
                        for capability in test_case.initial_capabilities:
                            operation, receipt = apply_capability_precondition(
                                controller, capability
                            )
                            lifecycle_operations.append(operation)
                            initial_receipts.append(receipt)
                    except Exception as exc:  # noqa: BLE001 - device boundaries vary
                        precondition_failure = exc
                        lifecycle_operations.append(
                            failed_operation(
                                "scenario_preconditions.setup",
                                LifecyclePhase.SETUP,
                                "device_state",
                                "declared scenario preconditions",
                                "device control",
                                reportable(exc),
                            )
                        )
                    if precondition_failure is None:
                        run_result: DeviceRunResult = self._device_agent.run(
                            goal=test_case.steps_instructions,
                            device_control=device_control,
                            perception=perception,
                            session_dir=session_dir,
                            session_trail=session_trail,
                            observation_hook=screen_audits.observe,
                            before_action_hook=screen_audits.before_action,
                            step_hook=step_hook,
                            planner_override=qa_planner,
                            max_steps=step_budget,
                            setup=SetupMode.PRESERVE,
                        )
                finally:
                    app_restoration = restore_app_precondition(device_control, test_case.setup)
                    if app_restoration is not None:
                        lifecycle_operations.append(app_restoration)
        finally:
            self._device_agent.replay_ledger = in_run_ledger

        lifecycle_operations.extend(restoration_operations(restoration_events))
        if precondition_failure is not None:
            return ScenarioReport(
                scenario_uuid=test_case.scenario_uuid,
                test_id=test_case.test_id,
                scenario_name=test_case.scenario_name,
                cell=test_case.cell,
                status=ReportStatus.ERROR,
                cause=Cause.DEVICE.value,
                duration=round(time.time() - start_time, 2),
                failure_reason=reportable(precondition_failure),
                lifecycle_operations=lifecycle_operations,
            )
        run_result.capability_receipts[0:0] = initial_receipts

        # Nothing ever leaves the screen a scenario ended on, so read it here.
        screen_audits.finalize()
        stop_reason["value"] = stop_reason["value"] or screen_audits.stop_reason

        report = self._build_report(
            test_case=test_case,
            run_result=run_result,
            qa_outputs=qa_outputs,
            assessments=screen_audits.assessments,
            stop_reason=stop_reason["value"],
            duration=time.time() - start_time,
            security_rules=security_rules,
        )
        report.lifecycle_operations = lifecycle_operations
        failed_restorations = [
            operation
            for operation in lifecycle_operations
            if operation.phase is LifecyclePhase.RESTORATION
            and operation.status is LifecycleStatus.FAILED
        ]
        if failed_restorations:
            report.status = ReportStatus.ERROR
            report.cause = Cause.DEVICE.value
            report.failure_reason = "Device state restoration failed: " + "; ".join(
                operation.failure_reason or operation.requested for operation in failed_restorations
            )
        self._save_trace(test_case, replay, report)
        return report

    def _build_report(
        self,
        test_case: GeneratedTestCase,
        run_result: DeviceRunResult,
        qa_outputs: List[QAStepResponse],
        assessments: List[ScreenAssessment],
        stop_reason: Optional[str],
        duration: float,
        security_rules: List[str],
        settle_transitions: bool = True,
        measurement_samples: Sequence[Measurement] = (),
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
                    issue.severity,
                    issue.issue_type,
                    assessment.screen_name,
                    issue.description,
                )

        if settle_transitions:
            all_verifications.extend(self._settled_transitions(test_case, assessments))

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
            measurement_samples=measurement_samples,
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
            postcondition_verifications=list(run_result.postcondition_verifications),
            measurements=list(measurement_samples),
            failure_reason=scrub(failure_reason or "") or None,
            verdict_inputs=_verdict_inputs(
                run_result,
                stop_reason,
                qa_outputs,
                assessments,
                test_case,
                security_rules,
                all_findings,
                measurement_samples,
            ),
        )

    def _settled_transitions(
        self, test_case: GeneratedTestCase, assessments: List[ScreenAssessment]
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

    def _build_replay(self, test_case: GeneratedTestCase):
        """The path an earlier run saved for this scenario, if there is one.

        None when durable replay is off, or when nothing saved one. None
        too when this run's identity does not match the path's.

        The path supplies actions only. This run's screen hook independently
        earns all evidence used by its verdict.
        """
        if self._trace_store is None:
            return None

        identity = self._trace_store.resolve_identity(test_case)
        if identity is None:
            return None

        return self._trace_store.build_replay(identity)

    def _save_trace(self, test_case, replay, report) -> None:
        """Writes the path this scenario walked, once it has been judged.

        Only after a pass. A path to a failure is a path to a result already
        known, and walking it again would spend a run reaching it.
        """
        if self._trace_store is None or replay is None:
            return
        self._trace_store.save_trace(test_case, replay, report.status)

    def _determine_status(
        self,
        run_result: DeviceRunResult,
        stop_reason: Optional[str],
        all_verifications: List[RuleVerification],
        all_issues: List[QAIssue],
        qa_outputs: List[QAStepResponse],
        assessments: List[ScreenAssessment],
        required_outcomes: Sequence[ExpectedOutcome | str],
        security_rules: Sequence[str],
        all_findings: Optional[List[SensitiveFinding]] = None,
        required_capabilities: Sequence = (),
        measurement_samples: Sequence[Measurement] = (),
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
            measurement_samples=measurement_samples,
        )
