import logging
import time
from functools import partial
from pathlib import Path
from typing import List, Optional

from yewo import __version__, identity
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.device.session import DeviceSession
from yewo.agent.device.platform import Platform
from yewo.agent.device.readiness import ensure_device_ready
from yewo.agent.perception.blind import for_grounding
from yewo.agent.plan.planner import GROUNDING_MARKS
from yewo.agent.memory import replay_scenarios
from yewo.agent.memory.replay import ReplayLedger
from yewo.agent.memory.replay_store import ValueNames
from yewo.errors import ProviderUnavailable
from yewo.llm.failures import reportable
from .judge.coverage import unevidenced_security_rules
from . import interruption, lock_resolution, replay_paths
from .report.run_record import create_manifest, ensure_interrupted_manifest
from .report import infrastructure
from .report.aggregation import aggregate_journey_status, tally
from . import sensitivity
from yewo.llm.prompts import PromptStore
from yewo.defaults import DEFAULT_PERCEPTION_MODEL
from .models import (
    JourneyReport,
    ReportStatus,
    RunEnvironment,
    ScenarioReport,
    UserJourney,
)
from .qa_agent import QAAgent
from .metric_preflight import prepare_execution, record_refused_reports
from .report.usage_tracker import LLMUsageTracker
from yewo.utils.secrets import SecretScrubber
from yewo.package_policy import evaluate_package_policy
from .report import writing
from .run_usage import (
    GENERATION,
    PLANNER,
    SCENARIO_COMPONENTS,
    SEMANTIC_GROUNDER,
    component_for,
    record_scenario_execution_cost,
)
from .warm_execution import ScenarioRoutePreparer
from .runner_session import is_session_responsive
from .generation_evidence import ScenarioGenerationEvidence
from .runner_knowledge import RunnerDiscoveryProvider
from .runner_startup import DiscoveryProviderFactory, build_runner_startup_authority
from .cold_start import ColdStartEvidence

logger = logging.getLogger(__name__)


class YewoTestRunner:
    """Orchestrates E2E testing by decomposing journeys, executing scenarios, and emitting JUnit reports."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        perception_model: str | None = None,
        assets_dir: Path = Path("assets"),
        max_steps: int = 40,
        replay: bool = False,
        durable_replay: bool = True,
        update_paths: bool = False,
        reuse_app: bool = False,
        keep_app: bool = False,
        settle_time: float = 2.0,
        headless: bool = True,
        fast_model: str = None,
        record_session: bool = True,
        grounding: str = GROUNDING_MARKS,
        journey_file: Optional[Path] = None,
        workflow: str = "",
        locked: bool = False,
        persist_inputs: bool = True,
        allow_secret_exposure: bool = False,
        policy_waivers: Optional[dict[str, str]] = None,
        run_id: Optional[str] = None,
        execution_id: Optional[str] = None,
        target_id: Optional[str] = None,
        route_preparer: ScenarioRoutePreparer | None = None,
        generation_evidence: ScenarioGenerationEvidence | None = None,
        knowledge_root: Path | None = None,
        discovery_provider: RunnerDiscoveryProvider | None = None,
        discovery_factory: DiscoveryProviderFactory | None = None,
        cold_start: ColdStartEvidence | None = None,
    ):
        self.journey_file = journey_file
        self.workflow = workflow
        self.locked = locked
        self.persist_inputs = persist_inputs
        self.allow_secret_exposure = allow_secret_exposure
        self.policy_waivers = dict(policy_waivers or {})
        self.run_id, self.execution_id, self.target_id = run_id, execution_id, target_id
        self.knowledge_root = knowledge_root
        self.cold_start = cold_start
        self.startup = build_runner_startup_authority(
            partial(sensitivity.require_safe_secret_policy, opted_in=allow_secret_exposure),
            journey_file,
            workflow,
            locked,
            knowledge_root,
            discovery_provider,
            generation_evidence,
            route_preparer,
            discovery_factory,
        )
        self.grounding = grounding
        self.reuse_app = reuse_app
        self.keep_app = keep_app
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        self.model = model
        self.perception_model = perception_model
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.replay = replay
        # On unless a caller turns it off. A saved path only ever
        # removes planning, and every screen it walks is still read.
        self.durable_replay = durable_replay
        self.update_paths = update_paths
        self.settle_time = settle_time
        self.headless = headless
        self.fast_model = fast_model
        self.record_session = record_session
        self._app: Optional[identity.AppIdentity] = None
        self._device = identity.DeviceIdentity()
        self._git: Optional[identity.GitIdentity] = None
        self.last_report_paths: dict[str, Path] = {}

    def _trace_store_for(self, journey):
        """Build this journey's saved-path store when replay is enabled."""
        if not self.durable_replay:
            return None
        root = self.journey_file.parent if self.journey_file else Path.cwd()
        return replay_scenarios.build_trace_store(
            root=root,
            values=dict(journey.values),
            verification_policies=dict(journey.value_verifications),
            environment=self._environment(),
            grounding=self.grounding,
            allow_path_creation=not replay_paths.is_ci(),
            approve_path_update=lambda path: replay_paths.approve_update(path, self.update_paths),
        )

    def _environment(self) -> RunEnvironment:
        """Cache the immutable app and commit identity used by every checkpoint."""
        if self._app is None:
            self._app = identity.app_identity(self.app_path, installed=not self.reuse_app)
        if self._git is None:
            self._git = identity.git_identity(
                self.journey_file.parent if self.journey_file else None
            )
        return RunEnvironment(
            yewo_version=__version__,
            model=self.model,
            fast_model=self.fast_model,
            platform=self.platform.value,
            grounding=self.grounding,
            detector=(
                self.perception_model or DEFAULT_PERCEPTION_MODEL
                if self.grounding == GROUNDING_MARKS
                else None
            ),
            app=self._app,
            device=self._device,
            git=self._git,
            ci=identity.ci_identity(),
            execution_id=self.execution_id,
            run_id=self.run_id,
            target_id=self.target_id,
        )

    def _suite(self, journey: UserJourney, llm) -> lock_resolution.LockResolution:
        """Resolve scenarios only through the fail-closed startup authority."""
        return self.startup.resolve_suite(journey, llm)

    def run_journey(self, journey: UserJourney) -> JourneyReport:
        """Run one journey, refusing invalid inputs before opening the device."""
        start_time = time.time()
        self.startup.inspect_knowledge(journey)
        if self.durable_replay:
            replay_paths.validate_path_key(
                self.journey_file.parent if self.journey_file else Path.cwd()
            )

        if self.locked:
            lock_resolution.require_lock(
                journey,
                self.startup.lock_path,
                self.workflow,
                self.startup.generation_evidence,
            )

        record_session = self.record_session
        redact_recording = bool(journey.secrets) and not self.allow_secret_exposure
        session_root = self.assets_dir / "sessions" if self.run_id else self.assets_dir
        session = DeviceSession(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            record_session=record_session,
            redact_recording=redact_recording,
            session_dir=session_root,
            reuse_app=self.reuse_app,
            keep_app=self.keep_app,
        )

        agent = QAAgent(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            model=self.model,
            assets_dir=self.assets_dir,
            max_steps=self.max_steps,
            settle_time=self.settle_time,
            headless=self.headless,
            fast_model=self.fast_model,
            replay_ledger=ReplayLedger() if self.replay else None,
            grounding=self.grounding,
            cache_affinity=session.session_id,
        )

        artifact_scrubber = SecretScrubber()
        usage_tracker = LLMUsageTracker(
            output_dir=session.telemetry_dir, scrubber=artifact_scrubber
        )
        prompt_store = PromptStore(
            session.prompts_dir,
            scrub=artifact_scrubber.scrub,
            keep_bodies=self.persist_inputs,
        )
        current_component = GENERATION
        device_agent = agent._device_agent
        device_agent.llm.usage_callback = lambda call: usage_tracker.add_call(
            component_for(current_component, call.answering), call
        )
        if device_agent.fast_llm is not device_agent.llm:
            device_agent.fast_llm.usage_callback = lambda call: usage_tracker.add_call(
                component_for(current_component, call.answering), call
            )
        device_agent.grounding_usage_callback = lambda call: usage_tracker.add_call(
            SEMANTIC_GROUNDER, call
        )
        device_agent.step_opened = usage_tracker.mark
        device_agent.record_prompts(prompt_store)

        scenarios = []
        scenario_reports: List[ScenarioReport] = []
        manifest = None
        policy_checks = []
        run_was_aborted = False
        current_scenario = None
        artifact_scrubber.learn_secrets(journey.secrets)
        device_agent.scrubber.learn_secrets(
            journey.secrets,
            journey.secret_displays,
            allow_model_exposure=self.allow_secret_exposure,
        )
        device_agent.declared_values = dict(journey.values)
        device_agent.value_verifications = dict(journey.value_verifications)
        device_agent.value_names = ValueNames(device_agent.declared_values)

        try:
            policy_checks = evaluate_package_policy(
                Path(self.app_path), self.platform, self.policy_waivers, artifact_scrubber
            )
            prepared = prepare_execution(
                self,
                journey,
                device_agent,
                session,
                ensure_device_ready,
                for_grounding,
                policy_checks,
            )
            suite = prepared.suite
            scenarios = prepared.scenarios
            preflight_reports = prepared.refused_reports
            scenario_reports.extend(preflight_reports)
            device_control = prepared.device_control
            perception = prepared.perception
            self._device = prepared.device_identity
            if redact_recording:
                device_agent.scrubber.protected_region_callback = session.start_recording_mask
            agent._trace_store = prepared.trace_store
            logger.info("Scenarios: %s", suite.note)
            logger.info(
                "Successfully decomposed journey '%s' into %d scenarios:",
                journey.id,
                len(scenarios),
            )
            for idx, s in enumerate(scenarios):
                logger.info(
                    "  %d. [%s %s] %s (%s)",
                    idx + 1,
                    s.priority.value,
                    s.cell,
                    s.scenario_name,
                    s.test_id,
                )

            manifest = create_manifest(
                session.manifest_path,
                journey,
                scenarios,
                self._environment(),
                suite,
                artifact_scrubber,
                self.run_id,
                self.allow_secret_exposure,
                self.cold_start,
            )
            # Generation happened before this file existed, and it is the one
            # call a run makes that no scenario owns.
            manifest.record_unattached_calls(usage_tracker.calls_outside_steps(GENERATION))
            # A cell nothing covered is recorded before the first scenario
            # runs, so a reader sees the gap even if the run dies next. It
            # goes in the record as well as the report: the record is the one
            # a reader is told to trust.
            for gap in suite.uncovered:
                uncovered = infrastructure.uncovered_cell(gap)
                scenario_reports.append(uncovered)
                manifest.add(uncovered)
            record_refused_reports(manifest, preflight_reports)
            manifest.write()
            logger.info("Run manifest: %s", manifest.path)
            current_component = PLANNER
            total_scenarios = len(scenarios)
            for idx, scenario in enumerate(scenarios):
                current_scenario = scenario
                # Scenario evidence must not inherit screen names, actions, or
                # navigation claims from a previous scenario.
                trail = SessionTrail()
                manifest.start(scenario.test_id)
                logger.info("========================================")
                current_component = PLANNER
                logger.info(
                    "Executing Scenario %d/%d: %s (%s) - Cell: %s",
                    idx + 1,
                    total_scenarios,
                    scenario.scenario_name,
                    scenario.test_id,
                    scenario.cell,
                )
                logger.info("========================================")
                if not is_session_responsive(device_control):
                    logger.warning(
                        "WebDriver/instrumentation session is unresponsive. Attempting session recovery..."
                    )
                    try:
                        # Recovery, not teardown: the scenarios still to run
                        # need the app that this session was testing.
                        session.close(final=False)
                    except Exception as close_err:
                        logger.warning("Error closing old session during recovery: %s", close_err)

                    session = DeviceSession(
                        device_name=self.device_name,
                        app_path=self.app_path,
                        platform=self.platform,
                        record_session=record_session,
                        redact_recording=redact_recording,
                        session_dir=session_root,
                        session_id=session.session_id,
                        reuse_app=self.reuse_app,
                        keep_app=self.keep_app,
                        # The app is on the device by now, so a fresh look
                        # would call it preexisting and leave it behind.
                        installed_by_run=session.installed_by_run,
                    )
                    logger.info(
                        "Starting a new driver connection for session %s...",
                        session.session_id,
                    )
                    device_control = session.start(is_headless=self.headless)
                    perception = for_grounding(
                        self.grounding,
                        session,
                        self.perception_model,
                    )

                    usage_tracker.output_dir = session.telemetry_dir
                try:
                    scen_report = agent.execute_scenario(
                        test_case=scenario,
                        security_rules=journey.security_rules,
                        device_control=device_control,
                        perception=perception,
                        session_dir=session.screenshots_dir,
                        session_trail=trail,
                        telemetry_dir=session.telemetry_dir,
                        crash_diagnostic_source=session,
                        route_preparer=self.startup.route_preparer,
                    )
                except ProviderUnavailable as exc:
                    # The service was asked again and still would not answer.
                    # The scenario is over, and the next one is not: a rate
                    # limit that has passed by then costs nothing here, and
                    # the device state was restored on the way out.
                    scen_report = infrastructure.provider_error(scenario, exc, artifact_scrubber)

                scenario_level_calls = usage_tracker.take_calls_outside_steps(SCENARIO_COMPONENTS)
                record_scenario_execution_cost(scen_report, scenario_level_calls)
                manifest.record(
                    scenario.test_id,
                    scen_report,
                    calls=scenario_level_calls,
                )

                scenario_reports.append(scen_report)
                self._save_checkpoint(
                    journey,
                    scenario_reports,
                    *tally(scenario_reports),
                    start_time,
                    session,
                    artifact_scrubber,
                )
                usage_tracker.checkpoint()
                try:
                    session.checkpoint_recording()
                except Exception as record_err:
                    logger.warning("Failed to save the scenario recording: %s", record_err)
                try:
                    session.checkpoint_device_logs()
                except Exception as log_err:  # noqa: BLE001 - evidence, not the run
                    logger.warning("Failed to drain the device log: %s", log_err)

        except BaseException as e:
            run_was_aborted = True
            safe_error = artifact_scrubber.scrub(reportable(e))
            logger.error("Fatal run interruption (%s): %s", type(e).__name__, safe_error)
            manifest, scenarios = ensure_interrupted_manifest(
                manifest,
                self,
                journey,
                scenarios,
                device_agent.llm,
                session,
                artifact_scrubber,
                usage_tracker.calls_outside_steps(GENERATION),
            )
            interruption.record_active_scenario(
                manifest,
                current_scenario,
                agent,
                e,
                scenario_reports,
                usage_tracker,
                SCENARIO_COMPONENTS,
            )
            prior_reports = len(scenario_reports)
            infrastructure.record_aborted(scenarios, scenario_reports, e)
            for aborted_report in scenario_reports[prior_reports:]:
                manifest.record_or_add(aborted_report)
            manifest.abort(f"Aborted due to run failure/interrupt: {safe_error}")
            if isinstance(e, KeyboardInterrupt):
                raise
        finally:
            if session is not None:
                try:
                    session.close(interrupted=run_was_aborted)
                except Exception as cleanup_err:
                    logger.warning("Session cleanup failed: %s", cleanup_err)

            try:
                usage_tracker.save_report()
            except Exception as usage_err:
                logger.error("Failed to serialize LLM usage: %s", usage_err)
                failed_usage = infrastructure.usage_failure(usage_err)
                scenario_reports.append(failed_usage)
                if manifest is not None:
                    manifest.add(failed_usage)

            # persist reports even on Ctrl+C or mid-run crash
            duration = time.time() - start_time
            unevidenced = unevidenced_security_rules(journey.security_rules, scenario_reports)
            for rule in unevidenced:
                logger.warning("No scenario evidenced security rule: %s", rule)
            passed_count, failed_count, skipped_count = tally(scenario_reports)
            journey_status = (
                ReportStatus.ABORTED
                if run_was_aborted
                else aggregate_journey_status(passed_count, failed_count)
            )
            report = JourneyReport(
                journey_id=journey.id,
                journey_name=journey.name,
                description=artifact_scrubber.scrub(journey.description),
                status=journey_status,
                total_duration=round(duration, 2),
                scenarios=scenario_reports,
                passed_count=passed_count,
                failed_count=failed_count,
                skipped_count=skipped_count,
                unevidenced_security_rules=unevidenced,
                policy_checks=policy_checks,
                environment=self._environment(),
            )
            self.last_report_paths = writing.persist(
                report,
                self.assets_dir,
                session,
                artifact_scrubber,
                manifest,
                unevidenced,
                usage_tracker.summary(),
            )

        return report

    def _save_checkpoint(
        self,
        journey: UserJourney,
        reports: List[ScenarioReport],
        passed: int,
        failed: int,
        skipped: int,
        started_at: float,
        session: DeviceSession,
        scrubber: SecretScrubber,
    ) -> None:
        checkpoint = JourneyReport(
            journey_id=journey.id,
            journey_name=journey.name,
            description=scrubber.scrub(journey.description),
            status=ReportStatus.RUNNING,
            total_duration=round(time.time() - started_at, 2),
            scenarios=list(reports),
            passed_count=passed,
            failed_count=failed,
            skipped_count=skipped,
            environment=self._environment(),
        )
        writing.checkpoint(checkpoint, self.assets_dir, session, scrubber)
