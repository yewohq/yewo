import logging
import time
from pathlib import Path
from typing import List, Optional

from yewo import __version__, identity
from yewo.agent.memory.trail import SessionTrail
from yewo.agent.device.session import DeviceSession
from yewo.agent.device.platform import Platform
from yewo.agent.device.readiness import ensure_device_ready
from yewo.agent.perception.detector import YOLO_REPO_ID, YOLO_REVISION
from yewo.agent.perception.blind import for_grounding
from yewo.agent.plan.planner import GROUNDING_COORDINATES
from yewo.agent.memory.replay import ReplayLedger
from yewo.errors import ProviderUnavailable
from yewo.llm.failures import reportable
from .judge.coverage import unevidenced_security_rules
from . import lock_file, lock_resolution
from .generator import ScenarioGenerator
from .report.manifest import RunInputs, RunManifest, RunSubject
from .report import infrastructure
from . import sensitivity
from yewo.llm.prompts import PromptStore
from .models import (
    JourneyReport,
    JourneyStatus,
    ReportStatus,
    RunEnvironment,
    ScenarioReport,
    UserJourney,
)
from .qa_agent import QAAgent
from .report.usage_tracker import LLMUsageTracker
from yewo.utils.secrets import SecretScrubber
from .report import writing

logger = logging.getLogger(__name__)

GENERATION = "generator"
"""The call that wrote the suite. It happens once, before any scenario."""

PLANNER = "agent"
"""Every call a scenario makes: the planner, the auditor and the judges.

They share one client unless a fast model is configured, so this is as far as
the call site can tell them apart. What a call asked is in its prompt, and
which step it belongs to is in the record it nests under.
"""


def is_session_responsive(device_control) -> bool:
    """Checks whether the driver connection and the instrumentation server are alive."""
    if not device_control or not device_control.driver:
        return False
    try:
        if device_control.platform == Platform.ANDROID:
            _ = device_control.driver.current_package
        else:
            _ = device_control.driver.orientation
        return True
    except Exception as e:
        logger.warning("Session responsiveness check failed: %s", e)
        return False


def tally(reports: List[ScenarioReport]) -> tuple[int, int, int]:
    """Passed, failed and skipped, where anything else that ran counts as failed."""
    passed = sum(1 for report in reports if report.status == ReportStatus.PASSED)
    skipped = sum(1 for report in reports if report.status == ReportStatus.SKIPPED)
    return passed, len(reports) - passed - skipped, skipped


def aggregate_journey_status(
    passed_count: int, failed_count: int
) -> JourneyStatus:
    """A suite passes only when at least one scenario has verified evidence."""
    return (
        ReportStatus.PASSED
        if passed_count > 0 and failed_count == 0
        else ReportStatus.FAILED
    )


class YewoTestRunner:
    """Orchestrates E2E testing by decomposing journeys, executing scenarios, and emitting JUnit reports."""

    def __init__(
        self,
        device_name: str,
        app_path: str,
        platform: Platform = Platform.ANDROID,
        model: str = "openrouter/google/gemini-2.5-flash",
        assets_dir: Path = Path("assets"),
        max_steps: int = 40,
        replay: bool = False,
        reuse_app: bool = False,
        keep_app: bool = False,
        settle_time: float = 2.0,
        headless: bool = True,
        fast_model: str = None,
        record_session: bool = True,
        grounding: str = GROUNDING_COORDINATES,
        journey_file: Optional[Path] = None,
        workflow: str = "",
        locked: bool = False,
        persist_inputs: bool = True,
        record_secrets: bool = False,
    ):
        self.journey_file = journey_file
        self.workflow = workflow
        self.locked = locked
        self.persist_inputs = persist_inputs
        self.record_secrets = record_secrets
        self.grounding = grounding
        self.reuse_app = reuse_app
        self.keep_app = keep_app
        self.platform = platform
        self.device_name = device_name
        self.app_path = app_path
        self.model = model
        self.assets_dir = assets_dir
        self.max_steps = max_steps
        self.replay = replay
        self.settle_time = settle_time
        self.headless = headless
        self.fast_model = fast_model
        self.record_session = record_session
        self._app: Optional[identity.AppIdentity] = None
        self._device = identity.DeviceIdentity()
        self._git: Optional[identity.GitIdentity] = None

    def _environment(self) -> RunEnvironment:
        """What was in play for this run, so a reader can explain a difference.

        The app and the commit are read once and kept. Hashing a build costs
        seconds, and a checkpoint is written after every scenario. The commit
        is kept for a second reason. A checkout that changes mid-run would
        give the record one answer and the report another, and the two are
        one page.
        """
        if self._app is None:
            self._app = identity.app_identity(
                self.app_path, installed=not self.reuse_app
            )
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
            detector=f"{YOLO_REPO_ID}@{YOLO_REVISION}",
            app=self._app,
            device=self._device,
            git=self._git,
            ci=identity.ci_identity(),
        )

    def _suite(self, journey: UserJourney, llm) -> lock_resolution.LockResolution:
        """The scenarios to run, from the lock where the lock still holds.

        Already in priority order, so a run that dies part way has done the
        P0 cells.
        """
        generator = ScenarioGenerator(llm)
        return lock_resolution.suite_for(
            journey,
            lambda wanted, cells: generator.generate_suite(wanted, cells),
            path=self._lock_path(),
            workflow=self.workflow,
            locked=self.locked,
        )

    def _lock_path(self) -> Optional[Path]:
        """Where this run's lock is, or None for a journey file off disk."""
        return lock_file.path_beside(self.journey_file) if self.journey_file else None

    def run_journey(self, journey: UserJourney) -> JourneyReport:
        """Generates scenarios from the journey, executes them, and builds reports & LLM usage metrics.

        Raises:
            UsageError: If `--locked` was asked for and the lock does not
                hold every declared cell. Raised before anything is opened or
                written, because the answer is the same either way.
        """
        start_time = time.time()

        # First, because reading a lock costs nothing. A stale lock in CI has
        # to fail in a second rather than after a device boots.
        if self.locked:
            lock_resolution.require_lock(journey, self._lock_path(), self.workflow)

        # Nothing masks a video frame, so a journey that types a declared
        # secret on screen is not recorded unless the caller asked for it.
        refused = sensitivity.recording_refused(journey, self.record_secrets)
        if refused:
            logger.warning("%s", refused)
        record_session = self.record_session and refused is None

        session = DeviceSession(
            device_name=self.device_name,
            app_path=self.app_path,
            platform=self.platform,
            record_session=record_session,
            session_dir=self.assets_dir,
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
        # Named for the component, not the scenario. The scenario is already
        # said by which entry the call is nested under, and two components
        # share one client when no fast model is configured.
        device_agent.llm.usage_callback = lambda call: usage_tracker.add_call(
            current_component, call
        )
        if device_agent.fast_llm is not device_agent.llm:
            device_agent.fast_llm.usage_callback = lambda call: usage_tracker.add_call(
                current_component, call
            )
        # So each call lands under the step that paid for it. The loop knows
        # which step it is on; the planner that makes the call does not.
        device_agent.step_opened = usage_tracker.mark
        device_agent.record_prompts(prompt_store)

        scenarios = []
        scenario_reports: List[ScenarioReport] = []
        manifest = None
        artifact_scrubber.learn_secrets(journey.secrets)
        device_agent.scrubber.learn_secrets(journey.secrets)

        try:
            # Both before the model call. Asking adb is cheap and names what is
            # available when the device is wrong. Opening the session proves
            # the device boots, the app installs and the driver answers, and a
            # generated suite is paid for before any of the three is known.
            ensure_device_ready(self.platform, self.device_name)

            logger.info(
                "Initializing emulator and starting remote webdriver session..."
            )
            device_control = session.start(is_headless=self.headless)
            perception = for_grounding(self.grounding, session)
            # Recorded as soon as the driver answers, so a report written from
            # the finally block still names the target it ran against.
            self._device = identity.device_identity(device_control)

            suite = self._suite(journey, device_agent.llm)
            logger.info("Scenarios: %s", suite.note)
            scenarios = suite.scenarios
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

            manifest = RunManifest(
                session.manifest_path,
                RunSubject(
                    kind="journey",
                    id=journey.id,
                    name=artifact_scrubber.scrub(journey.name),
                    description=artifact_scrubber.scrub(journey.description),
                ),
                scenarios,
                environment=self._environment(),
                inputs=RunInputs(
                    subject=journey.id,
                    subject_sha256=lock_file.journey_sha256(journey),
                    coverage_declared=[cell.key for cell in journey.coverage],
                    lock_sha256=suite.lock_sha256,
                    scenarios_from_lock=suite.reused_count,
                    scenarios_generated=suite.generated_count,
                ),
                scrubber=artifact_scrubber,
                artifacts=sensitivity.declared_for(journey),
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

            manifest.write()
            logger.info("Run manifest: %s", manifest.path)

            current_component = PLANNER
            total_scenarios = len(scenarios)
            for idx, scenario in enumerate(scenarios):
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
                        logger.warning(
                            "Error closing old session during recovery: %s", close_err
                        )

                    session = DeviceSession(
                        device_name=self.device_name,
                        app_path=self.app_path,
                        platform=self.platform,
                        record_session=record_session,
                        session_dir=self.assets_dir,
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
                    perception = for_grounding(self.grounding, session)

                    usage_tracker.output_dir = session.telemetry_dir

                try:
                    scen_report = agent.execute_scenario(
                        test_case=scenario,
                        security_rules=journey.security_rules,
                        device_control=device_control,
                        perception=perception,
                        session_dir=session.screenshots_dir,
                        session_trail=trail,
                    )
                except ProviderUnavailable as exc:
                    # The service was asked again and still would not answer.
                    # The scenario is over, and the next one is not: a rate
                    # limit that has passed by then costs nothing here, and
                    # the device state was restored on the way out.
                    scen_report = infrastructure.provider_error(
                        scenario, exc, artifact_scrubber
                    )

                manifest.record(
                    scenario.test_id,
                    scen_report,
                    calls=usage_tracker.calls_outside_steps(PLANNER),
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
                    logger.warning(
                        "Failed to save the scenario recording: %s", record_err
                    )

        except BaseException as e:
            safe_error = artifact_scrubber.scrub(reportable(e))
            logger.error(
                "Fatal run interruption (%s): %s", type(e).__name__, safe_error
            )
            if manifest is not None:
                manifest.abort(f"Aborted due to run failure/interrupt: {safe_error}")
            infrastructure.record_aborted(scenarios, scenario_reports, e)
            if isinstance(e, KeyboardInterrupt):
                raise e
        finally:
            if session is not None:
                try:
                    session.close()
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
            unevidenced = unevidenced_security_rules(
                journey.security_rules, scenario_reports
            )
            for rule in unevidenced:
                logger.warning("No scenario evidenced security rule: %s", rule)
            passed_count, failed_count, skipped_count = tally(scenario_reports)
            journey_status = aggregate_journey_status(passed_count, failed_count)
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
                environment=self._environment(),
            )
            writing.persist(
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
