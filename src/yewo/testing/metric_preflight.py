"""Refuse known measurement requirements before Appium starts."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from yewo import identity
from yewo.agent.device.platform import Platform
from yewo.errors import Cause
from yewo.testing import lock_file, lock_resolution
from yewo.testing.generator import with_journey_preconditions, with_values_filled
from yewo.testing.judge.capabilities import (
    CapabilityTarget,
    unavailable_metric_reason,
    unsupported_metric,
)
from yewo.testing.models import GeneratedTestCase, ReportStatus, ScenarioReport, UserJourney
from yewo.testing.taxonomy import sort_cells_by_priority
from yewo.policy import PolicyCheckReport, policy_checks_succeeded

logger = logging.getLogger(__name__)


class RunnerBoundary(Protocol):
    """The runner state required to prepare scenario execution."""

    platform: Platform
    device_name: str
    workflow: str
    locked: bool
    headless: bool
    grounding: str
    _device: identity.DeviceIdentity

    def _lock_path(self) -> Path | None: ...

    def _suite(self, journey: UserJourney, llm) -> lock_resolution.LockResolution: ...

    def _trace_store_for(self, journey: UserJourney) -> object | None: ...


@dataclass
class PreparedExecution:
    """The suite and optional device boundaries ready for the scenario loop."""

    suite: lock_resolution.LockResolution
    scenarios: list[GeneratedTestCase]
    refused_reports: list[ScenarioReport]
    device_control: object | None
    perception: object | None
    device_identity: identity.DeviceIdentity
    trace_store: object | None


def preflight_metrics_before_device(
    scenarios: list[GeneratedTestCase],
    platform: Platform,
) -> tuple[list[GeneratedTestCase], list[ScenarioReport]]:
    """Separate scenarios whose measurements cannot run on this platform."""
    runnable = []
    refused = []
    target = CapabilityTarget(platform=platform)
    for scenario in scenarios:
        platform_reasons = [
            reason
            for metric in scenario.required_metrics
            if (reason := unsupported_metric(metric, target)) is not None
        ]
        reason = (
            "Unsupported evidence requirement(s): " + "; ".join(platform_reasons)
            if platform_reasons
            else unavailable_metric_reason(scenario, target)
        )
        if reason is None:
            runnable.append(scenario)
            continue
        refused.append(
            ScenarioReport(
                scenario_uuid=scenario.scenario_uuid,
                test_id=scenario.test_id,
                scenario_name=scenario.scenario_name,
                cell=scenario.cell,
                status=(
                    ReportStatus.UNSUPPORTED
                    if platform_reasons
                    else ReportStatus.UNVERIFIED
                ),
                cause=Cause.SCENARIO.value,
                failure_reason=reason,
            )
        )
    return runnable, refused


def record_refused_reports(manifest, reports: list[ScenarioReport]) -> None:
    """Add every preflight refusal to the canonical run record."""
    for report in reports:
        manifest.add(report)


def _known_locked_scenarios(
    runner: RunnerBoundary,
    journey: UserJourney,
) -> tuple[list[GeneratedTestCase], bool]:
    """Return fresh locked scenarios and whether they cover the whole journey."""
    path = runner._lock_path()
    if path is None:
        return [], False
    stored = lock_file.read(path)
    if stored is None:
        return [], False
    problems = lock_resolution.staleness(stored, journey, runner.workflow)
    entries_by_cell = stored.entries_for(runner.workflow, journey.id)
    scenarios = [
        with_journey_preconditions(
            with_values_filled(entries_by_cell[cell.key].scenario(), journey.values),
            journey,
        )
        for cell in sort_cells_by_priority(list(journey.coverage))
        if cell.key not in problems and cell.key in entries_by_cell
    ]
    return scenarios, not problems


def prepare_execution(
    runner: RunnerBoundary,
    journey,
    device_agent,
    session,
    ensure_target_ready,
    build_perception,
    policy_checks: list[PolicyCheckReport] | None = None,
):
    """Resolve known metrics, then open a device only when work remains."""
    if not policy_checks_succeeded(policy_checks or []):
        return PreparedExecution(
            suite=lock_resolution.LockResolution(),
            scenarios=[],
            refused_reports=[],
            device_control=None,
            perception=None,
            device_identity=identity.DeviceIdentity(),
            trace_store=None,
        )
    known_scenarios, has_complete_lock = _known_locked_scenarios(runner, journey)
    runnable_known, refused_reports = preflight_metrics_before_device(
        known_scenarios,
        runner.platform,
    )
    suite = None
    scenarios = []
    if runner.locked or has_complete_lock:
        suite = runner._suite(journey, device_agent.llm)
        scenarios = runnable_known

    device_control = None
    perception = None
    device_identity = identity.DeviceIdentity()
    trace_store = None
    if suite is None or scenarios:
        ensure_target_ready(runner.platform, runner.device_name)
        logger.info("Initializing emulator and starting remote webdriver session...")
        device_control = session.start(is_headless=runner.headless)
        perception = build_perception(runner.grounding, session)
        device_identity = identity.device_identity(device_control)
        runner._device = device_identity
        trace_store = runner._trace_store_for(journey)

    if suite is None:
        suite = runner._suite(journey, device_agent.llm)
        refused_ids = {report.scenario_uuid for report in refused_reports}
        scenarios = [
            scenario
            for scenario in suite.scenarios
            if scenario.scenario_uuid not in refused_ids
        ]
    return PreparedExecution(
        suite=suite,
        scenarios=scenarios,
        refused_reports=refused_reports,
        device_control=device_control,
        perception=perception,
        device_identity=device_identity,
        trace_store=trace_store,
    )
