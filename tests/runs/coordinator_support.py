"""Reusable deterministic boundaries for run-coordinator tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallMode,
    ApplicationRequest,
)
from yewo.automation import AppiumServerConfig, AppiumServerLease, AutomationSession
from yewo.devices import (
    DeviceDescriptor,
    DeviceKind,
    DeviceLease,
    DevicePlatform,
    DeviceQuery,
    DeviceReadiness,
    ResourceOwnership,
)
from yewo.devices.logs import DeviceLogRedactionResult
from yewo.observations import RedactionRegion
from yewo.runs import (
    RunCoordinator,
    RunExecutionContext,
    RunMode,
    RunRequest,
    RunSessionPolicy,
    RunStore,
    RunTerminalCategory,
    RunWorkflowOutcome,
    RunWorkspaceFactory,
)


STARTED_AT = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)
DEVICE = DeviceDescriptor(
    platform=DevicePlatform.ANDROID,
    target_id="emulator-5554",
    connection_id="emulator-5554",
    display_name="Pixel",
    kind=DeviceKind.VIRTUAL,
    readiness=DeviceReadiness.READY,
    native_state="device",
)
APPLICATION = ApplicationDescriptor(
    platform=DevicePlatform.ANDROID,
    application_id="com.example.app",
)


class InventoryStub:
    def __init__(self, events: list[str], failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure

    def list_devices(self) -> Sequence[DeviceDescriptor]:
        self.events.append("inventory")
        if self.failure is not None:
            raise self.failure
        return (DEVICE,)


class LifecycleStub:
    def __init__(
        self,
        events: list[str],
        failure: Exception | None = None,
        ownership: ResourceOwnership = ResourceOwnership.RUN,
    ) -> None:
        self.events = events
        self.failure = failure
        self.ownership = ownership

    def prepare(self, device: DeviceDescriptor) -> DeviceLease:
        assert device is DEVICE
        self.events.append("device.prepare")
        if self.failure is not None:
            raise self.failure
        return DeviceLease(
            device,
            self.ownership,
            lambda: self.events.append("device.close"),
        )


class ApplicationResolverStub:
    def __init__(self, events: list[str], failure: Exception | None = None) -> None:
        self.events = events
        self.failure = failure

    def resolve(self, request: ApplicationRequest) -> ApplicationDescriptor:
        del request
        self.events.append("application.resolve")
        if self.failure is not None:
            raise self.failure
        return APPLICATION


class ServerManagerStub:
    def __init__(
        self,
        events: list[str],
        failure: Exception | None = None,
        ownership: ResourceOwnership = ResourceOwnership.RUN,
    ) -> None:
        self.events = events
        self.failure = failure
        self.ownership = ownership

    def acquire(self, config: AppiumServerConfig) -> AppiumServerLease:
        assert config.log_path is not None
        self.events.append("appium.acquire")
        if self.failure is not None:
            raise self.failure
        return AppiumServerLease(
            config.server_url,
            self.ownership,
            release=lambda: self.events.append("appium.close"),
        )


class SessionStub:
    def __init__(
        self,
        events: list[str],
        device: DeviceLease,
        server: AppiumServerLease,
        *,
        fail_close: bool = False,
        interrupt_close: bool = False,
    ) -> None:
        self.events = events
        self.device = device
        self.server = server
        self.fail_close = fail_close
        self.interrupt_close = interrupt_close

    def close(self) -> None:
        self.events.append("session.close")
        self.server.close()
        self.device.close()
        if self.interrupt_close:
            raise KeyboardInterrupt
        if self.fail_close:
            raise RuntimeError("cleanup failed")


class SessionManagerStub:
    def __init__(
        self,
        events: list[str],
        *,
        failure: Exception | None = None,
        fail_close: bool = False,
        interrupt_close: bool = False,
    ) -> None:
        self.events = events
        self.failure = failure
        self.fail_close = fail_close
        self.interrupt_close = interrupt_close
        self.recording_redaction_regions: tuple[RedactionRegion, ...] = ()

    def open(
        self,
        device,
        server,
        request,
        application,
        *,
        artifact_root: Path,
        scrub_device_log_text,
        recording_redaction_regions,
    ) -> AutomationSession:
        del request, application, scrub_device_log_text
        self.recording_redaction_regions = tuple(recording_redaction_regions)
        if self.recording_redaction_regions:
            self.events.append(f"session.redaction:{len(self.recording_redaction_regions)}")
        assert artifact_root.name == "run_1"
        self.events.append("session.open")
        if self.failure is not None:
            raise self.failure
        return cast(
            AutomationSession,
            SessionStub(
                self.events,
                device,
                server,
                fail_close=self.fail_close,
                interrupt_close=self.interrupt_close,
            ),
        )


class WorkflowStub:
    def __init__(
        self,
        events: list[str],
        *,
        outcome: RunWorkflowOutcome | None = None,
        failure: Exception | None = None,
        write_evidence: bool = False,
        recording_redaction_regions: tuple[RedactionRegion, ...] = (),
    ) -> None:
        self.events = events
        self.outcome = outcome or RunWorkflowOutcome(
            RunTerminalCategory.COMPLETED,
            "The workflow completed.",
        )
        self.failure = failure
        self.write_evidence = write_evidence
        self.recording_redaction_regions = recording_redaction_regions

    def session_policy(self) -> RunSessionPolicy:
        return RunSessionPolicy(
            recording_redaction_regions=self.recording_redaction_regions,
        )

    def execute(self, context: RunExecutionContext, interruption) -> RunWorkflowOutcome:
        assert not interruption.is_set()
        self.events.append("workflow.execute")
        if self.write_evidence:
            evidence = context.workspace.root / "assets/evidence.txt"
            evidence.parent.mkdir(parents=True)
            evidence.write_text("partial")
        if self.failure is not None:
            raise self.failure
        return self.outcome


class SignalStub:
    def __init__(
        self,
        events: list[str],
        *,
        interrupt_after: str | None = None,
        initially_interrupted: bool = False,
    ) -> None:
        self.events = events
        self.interrupt_after = interrupt_after
        self.initially_interrupted = initially_interrupted

    def is_set(self) -> bool:
        return self.initially_interrupted or (
            self.interrupt_after is not None and self.interrupt_after in self.events
        )


def run_request(tmp_path: Path, *, target_id: str = "emulator-5554") -> RunRequest:
    return RunRequest(
        mode=RunMode.NORMAL,
        output_root=tmp_path / "runs",
        device=DeviceQuery(platform=DevicePlatform.ANDROID, target_id=target_id),
        application=ApplicationRequest(
            platform=DevicePlatform.ANDROID,
            install_mode=ApplicationInstallMode.REUSE,
            application_id="com.example.app",
        ),
    )


def build_coordinator(
    events: list[str],
    workflow: WorkflowStub,
    *,
    inventory_failure: Exception | None = None,
    lifecycle_failure: Exception | None = None,
    application_failure: Exception | None = None,
    server_failure: Exception | None = None,
    session_failure: Exception | None = None,
    fail_close: bool = False,
    interrupt_close: bool = False,
    device_ownership: ResourceOwnership = ResourceOwnership.RUN,
    server_ownership: ResourceOwnership = ResourceOwnership.RUN,
    include_workflow: bool = True,
    store_factory: Callable[[Path], RunStore] = RunStore,
) -> RunCoordinator:
    return RunCoordinator(
        RunWorkspaceFactory(
            identifier_factory=lambda: "run_1",
            time_factory=lambda: STARTED_AT,
        ),
        {DevicePlatform.ANDROID: InventoryStub(events, inventory_failure)},
        {
            DevicePlatform.ANDROID: LifecycleStub(
                events,
                lifecycle_failure,
                device_ownership,
            )
        },
        ApplicationResolverStub(events, application_failure),
        ServerManagerStub(events, server_failure, server_ownership),
        SessionManagerStub(
            events,
            failure=session_failure,
            fail_close=fail_close,
            interrupt_close=interrupt_close,
        ),
        {RunMode.NORMAL: workflow} if include_workflow else {},
        lambda workspace: AppiumServerConfig(log_path=workspace.root / "runtime/appium.log"),
        lambda text: DeviceLogRedactionResult(text, "test-policy", 0),
        store_factory=store_factory,
        time_factory=lambda: STARTED_AT + timedelta(seconds=1),
    )
