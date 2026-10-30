"""Acceptance proofs for run startup, interruption, and ownership."""

from collections.abc import Callable
from pathlib import Path

import pytest

from yewo.applications import ApplicationInspectionError
from yewo.automation import AppiumServerError
from yewo.devices.errors import DeviceLifecycleError
from yewo.runs import RunCoordinator, RunStage, RunTerminalCategory
from .coordinator_support import (
    SignalStub,
    WorkflowStub,
    build_coordinator,
    run_request,
)


@pytest.mark.parametrize(
    ("interrupt_after", "expected_stage", "expected_events"),
    [
        (
            "inventory",
            RunStage.DEVICE_PREPARATION,
            ["inventory"],
        ),
        (
            "device.prepare",
            RunStage.APPLICATION_INSPECTION,
            ["inventory", "device.prepare", "device.close"],
        ),
        (
            "application.resolve",
            RunStage.APPIUM_STARTUP,
            ["inventory", "device.prepare", "application.resolve", "device.close"],
        ),
        (
            "appium.acquire",
            RunStage.AUTOMATION_STARTUP,
            [
                "inventory",
                "device.prepare",
                "application.resolve",
                "appium.acquire",
                "appium.close",
                "device.close",
            ],
        ),
        (
            "session.open",
            RunStage.EXECUTION,
            [
                "inventory",
                "device.prepare",
                "application.resolve",
                "appium.acquire",
                "session.open",
                "session.close",
                "appium.close",
                "device.close",
            ],
        ),
    ],
)
def test_interruption_between_startup_stages_closes_only_acquired_resources(
    tmp_path: Path,
    interrupt_after: str,
    expected_stage: RunStage,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    coordinator = build_coordinator(events, WorkflowStub(events))

    terminal = coordinator.run(
        run_request(tmp_path),
        interruption=SignalStub(events, interrupt_after=interrupt_after),
    )

    assert terminal.category is RunTerminalCategory.INTERRUPTED
    assert terminal.stage is expected_stage
    assert events == expected_events


FailureFactory = Callable[[list[str], WorkflowStub], RunCoordinator]


@pytest.mark.parametrize(
    ("factory", "expected_category", "expected_stage", "expected_events"),
    [
        (
            lambda events, workflow: build_coordinator(
                events,
                workflow,
                inventory_failure=RuntimeError("inventory failed"),
            ),
            RunTerminalCategory.INFRASTRUCTURE_FAILURE,
            RunStage.DEVICE_SELECTION,
            ["inventory"],
        ),
        (
            lambda events, workflow: build_coordinator(
                events,
                workflow,
                lifecycle_failure=DeviceLifecycleError("device failed"),
            ),
            RunTerminalCategory.BLOCKED_BY_SETUP,
            RunStage.DEVICE_PREPARATION,
            ["inventory", "device.prepare"],
        ),
        (
            lambda events, workflow: build_coordinator(
                events,
                workflow,
                application_failure=ApplicationInspectionError("inspection failed"),
            ),
            RunTerminalCategory.BLOCKED_BY_SETUP,
            RunStage.APPLICATION_INSPECTION,
            ["inventory", "device.prepare", "application.resolve", "device.close"],
        ),
        (
            lambda events, workflow: build_coordinator(
                events,
                workflow,
                server_failure=AppiumServerError("server failed"),
            ),
            RunTerminalCategory.BLOCKED_BY_SETUP,
            RunStage.APPIUM_STARTUP,
            [
                "inventory",
                "device.prepare",
                "application.resolve",
                "appium.acquire",
                "device.close",
            ],
        ),
        (
            lambda events, workflow: build_coordinator(
                events,
                workflow,
                session_failure=RuntimeError("session failed"),
            ),
            RunTerminalCategory.INFRASTRUCTURE_FAILURE,
            RunStage.AUTOMATION_STARTUP,
            [
                "inventory",
                "device.prepare",
                "application.resolve",
                "appium.acquire",
                "session.open",
                "appium.close",
                "device.close",
            ],
        ),
    ],
)
def test_startup_failure_closes_only_resources_that_were_acquired(
    tmp_path: Path,
    factory: FailureFactory,
    expected_category: RunTerminalCategory,
    expected_stage: RunStage,
    expected_events: list[str],
) -> None:
    events: list[str] = []
    coordinator = factory(events, WorkflowStub(events))

    terminal = coordinator.run(run_request(tmp_path))

    assert terminal.category is expected_category
    assert terminal.stage is expected_stage
    assert events == expected_events
    assert (tmp_path / "runs/run_1/run/terminal.json").is_file()
