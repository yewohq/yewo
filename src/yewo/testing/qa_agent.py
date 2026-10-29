"""The testing layer around DeviceAgent."""

import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

from yewo.agent.run.agent import DeviceAgent, DeviceRunResult
from yewo.agent.plan.planner import DeviceStepResponse
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.device.platform import Platform
from yewo.agent.device.crash_diagnostics import CrashDiagnosticSource
from yewo.agent.memory.replay import ReplayLedger

from yewo.llm.failures import reportable
from yewo.llm.llm_client import LLMClient
from yewo.testing.models import (
    CrashDiagnostic,
    ExpectedOutcome,
    GeneratedTestCase,
    QAIssue,
    ReportStatus,
    RuleVerification,
    ScenarioReport,
    SensitiveFinding,
    Measurement,
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
from yewo.testing.judge.qa_auditor import ScreenAssessment
from yewo.testing.qa_planner import QAPlanner, QAStepResponse
from yewo.errors import Cause
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
from yewo.testing.measurement_collection import ScenarioMeasurementCollector
from yewo.testing.report.measurement_log import MeasurementLog
from yewo.testing.verdict_inputs import build_verdict_inputs
from yewo.testing import obligation_auditing
from yewo.testing.obligations import ObligationController, ObligationRunEvidence
from yewo.testing.warm_execution import (
    ScenarioRoutePreparer,
    ScenarioWarmExecution,
    WarmExecutionTerminal,
)
from yewo.testing.warm_execution_reporting import (
    failed_warm_execution_report,
    prepare_scenario_route,
)
from yewo.testing.scenario_execution_support import (
    build_scenario_replay,
    save_scenario_trace,
    scenario_step_budget,
)

logger = logging.getLogger(__name__)


class QAAgent:
    """Share model clients while isolating each scenario's planner and auditor."""

    _trace_store = None
    """A class default keeps agents built without ``__init__`` replay-safe."""
    _platform = Platform.ANDROID
    _active_obligation_controller: ObligationController | None = None

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
            platform=platform,
        )
        self._platform = platform
        self._device_agent.llm.set_cache_affinity(cache_affinity)
        self._device_agent.fast_llm.set_cache_affinity(cache_affinity)
        self._fast_llm: LLMClient = self._device_agent.fast_llm

    def _step_budget(self, test_case: GeneratedTestCase) -> int:
        """Bound this scenario from its instruction count and the run ceiling."""
        return scenario_step_budget(test_case, self._device_agent.max_steps)

    def execute_scenario(
        self,
        test_case: GeneratedTestCase,
        security_rules: List[str],
        device_control,
        perception,
        session_dir: Path,
        session_trail: Optional[SessionTrail] = None,
        telemetry_dir: Optional[Path] = None,
        crash_diagnostic_source: Optional[CrashDiagnosticSource] = None,
        route_preparer: ScenarioRoutePreparer | None = None,
    ) -> ScenarioReport:
        """Execute one test scenario and return a complete ScenarioReport."""
        start_time = time.time()

        qa_outputs: List[QAStepResponse] = []
        stop_reason: dict = {"value": None}  # mutable so the nested hook can write out

        obligation_controller, screen_audits = obligation_auditing.build_audit_lanes(
            self._fast_llm,
            test_case,
            security_rules,
            self._device_agent.scrubber,
        )
        self._active_obligation_controller = obligation_controller

        replay = self._build_replay(test_case)

        def step_hook(step_response: DeviceStepResponse, snapshot) -> Optional[bool]:
            if not isinstance(step_response, QAStepResponse):
                return None

            qa_outputs.append(step_response)
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
            preflight_reason = unavailable_metric_reason(test_case, target)
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

        measurement_log = (
            MeasurementLog(telemetry_dir, self._device_agent.scrubber)
            if telemetry_dir is not None
            else None
        )
        measurement_collector = ScenarioMeasurementCollector(
            device_control,
            target,
            test_case.scenario_uuid,
            test_case.setup,
            test_case.required_metrics,
            measurement_log,
        )

        def observe_screen(snapshot, step: int):
            measurement_collector.observe_screen(snapshot.current_phash, step)
            return screen_audits.observe(snapshot, step)

        # OCR caches are useful within a scenario but are evidence from the
        # wrong run once the app has been restarted.
        perception.reset()

        qa_planner = QAPlanner(
            llm_client=self._fast_llm,
            setup=test_case.setup,
            scrubber=self._device_agent.scrubber,
            grounding=self._grounding,
            platform=self._platform,
        )

        in_run_ledger = self._device_agent.replay_ledger
        # Security scenarios must audit every reached screen, including replay.
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
        warm_execution: ScenarioWarmExecution | None = None
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
                        setup_evidence = apply_app_precondition(
                            device_control,
                            test_case.setup,
                            test_case.setup_origin,
                            collect_launch_ttid=(
                                device_control.platform is Platform.ANDROID
                                and test_case.setup is not SetupMode.PRESERVE
                            ),
                        )
                        lifecycle_operations.append(setup_evidence.operation)
                        if setup_evidence.launch_timing is not None:
                            measurement_collector.record_launch(setup_evidence.launch_timing)
                        if setup_evidence.measurement_failure:
                            measurement_collector.record_failure(setup_evidence.measurement_failure)
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
                    if precondition_failure is None and test_case.route_memory_ids:
                        warm_execution = prepare_scenario_route(
                            test_case,
                            route_preparer,
                            device_control,
                            perception,
                        )
                    if precondition_failure is None and (
                        warm_execution is None
                        or warm_execution.terminal is WarmExecutionTerminal.READY
                    ):
                        measurement_collector.begin_frame_diagnostics()
                        try:
                            run_result: DeviceRunResult = self._device_agent.run(
                                goal=test_case.steps_instructions,
                                device_control=device_control,
                                perception=perception,
                                session_dir=session_dir,
                                session_trail=session_trail,
                                observation_hook=observe_screen,
                                before_action_hook=screen_audits.before_action,
                                step_hook=step_hook,
                                planner_override=qa_planner,
                                max_steps=step_budget,
                                setup=SetupMode.PRESERVE,
                            )
                        finally:
                            measurement_collector.finish_frame_diagnostics()
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
                measurements=list(measurement_collector.samples),
                lifecycle_operations=lifecycle_operations,
            )
        if (
            warm_execution is not None
            and warm_execution.terminal is not WarmExecutionTerminal.READY
        ):
            return failed_warm_execution_report(
                test_case,
                warm_execution,
                duration=time.time() - start_time,
                measurements=measurement_collector.samples,
                lifecycle_operations=lifecycle_operations,
            )
        run_result.capability_receipts[0:0] = initial_receipts

        obligation_auditing.finalize_run_end(
            self._fast_llm,
            test_case,
            self._device_agent.scrubber,
            obligation_controller,
            screen_audits,
        )
        stop_reason["value"] = stop_reason["value"] or screen_audits.stop_reason

        crash_diagnostic = (
            crash_diagnostic_source.collect_crash_diagnostic()
            if crash_diagnostic_source is not None
            else None
        )

        report = self._build_report(
            test_case=test_case,
            run_result=run_result,
            qa_outputs=qa_outputs,
            assessments=screen_audits.assessments,
            stop_reason=stop_reason["value"],
            duration=time.time() - start_time,
            security_rules=security_rules,
            measurement_samples=measurement_collector.samples,
            crash_diagnostic=crash_diagnostic,
            obligation_controller=obligation_controller,
        )
        self._active_obligation_controller = None
        report.lifecycle_operations = lifecycle_operations
        report.warm_execution = warm_execution
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

    def _build_replay(self, test_case: GeneratedTestCase):
        return build_scenario_replay(self._trace_store, test_case)

    def _save_trace(self, test_case, replay, report) -> None:
        save_scenario_trace(self._trace_store, test_case, replay, report.status)

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
        crash_diagnostic: Optional[CrashDiagnostic] = None,
        obligation_controller: ObligationController | None = None,
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
            if obligation_controller is not None:
                obligation_auditing.route_aggregate_evidence(obligation_controller)
            all_verifications.extend(
                obligation_auditing.settle_transitions(
                    self._fast_llm,
                    test_case,
                    assessments,
                    self._device_agent.scrubber,
                )
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
            measurement_samples=measurement_samples,
            crash_diagnostic=crash_diagnostic,
        )
        obligation_evidence = (
            obligation_auditing.settle_obligations(obligation_controller, all_verifications)
            if obligation_controller is not None
            else ObligationRunEvidence()
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
            intervention_steps=run_result.intervention_steps,
            rule_verifications=all_verifications,
            detected_issues=all_issues,
            steps=list(run_result.steps),
            postcondition_verifications=list(run_result.postcondition_verifications),
            recovery_transactions=list(run_result.recovery_transactions),
            settlement_receipts=list(run_result.settlement_receipts),
            input_transactions=list(run_result.input_transactions),
            obligation_evidence=obligation_evidence,
            measurements=list(measurement_samples),
            failure_reason=scrub(failure_reason or "") or None,
            crash_diagnostic=crash_diagnostic,
            verdict_inputs=build_verdict_inputs(
                run_result,
                stop_reason,
                qa_outputs,
                assessments,
                test_case,
                security_rules,
                all_findings,
                measurement_samples,
                crash_diagnostic,
            ),
        )

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
        crash_diagnostic: Optional[CrashDiagnostic] = None,
    ):
        """Decide this scenario through the same function used by evaluation."""
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
            crash_diagnostic=crash_diagnostic,
        )
