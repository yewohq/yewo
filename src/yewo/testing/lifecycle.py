"""Deterministic state around a scenario's visible user journey."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from yewo.agent.actuation.actions import Capability
from yewo.agent.device.capability_controller import CapabilityController
from yewo.agent.device.launch import (
    LaunchTiming,
    SetupMode,
    clear_app_data,
    measure_android_launch_ttid,
    restart_app,
)
from yewo.agent.device.platform import Platform
from yewo.agent.device.receipts import Receipt
from yewo.agent.device.restoration import RestorationEvent
from yewo.errors import DeviceUnavailable, MeasurementUnavailable


class SetupOrigin(str, Enum):
    """Where a scenario's initial app state came from."""

    DEFAULT = "default"
    DECLARED = "declared"
    LOCKED = "locked"


class LifecyclePhase(str, Enum):
    """When an operation occurred around the visible user journey."""

    SETUP = "setup"
    RESTORATION = "restoration"


class LifecycleStatus(str, Enum):
    """The evidence produced by one lifecycle operation."""

    CONFIRMED = "confirmed"
    RESTORED = "restored"
    FAILED = "failed"


class LifecycleOperation(BaseModel):
    """One deterministic operation outside the user's UI actions."""

    operation_id: str
    phase: LifecyclePhase
    kind: Literal["app_state", "capability", "device_state"]
    requested: str
    status: LifecycleStatus
    source: str
    observed: Optional[str] = None
    failure_reason: Optional[str] = None


class ScenarioPreconditions(BaseModel):
    """Optional state declared for every scenario in one journey."""

    app_state: Optional[SetupMode] = Field(
        default=None,
        description="Initial app state. Independent scenarios default to clear_data.",
    )
    capabilities: list[Capability] = Field(
        default_factory=list,
        description="Device states to apply and confirm before visible UI execution.",
    )


@dataclass(frozen=True)
class AppSetupEvidence:
    """The lifecycle operation and any telemetry its launch produced."""

    operation: LifecycleOperation
    launch_timing: Optional[LaunchTiming] = None
    measurement_failure: Optional[str] = None


def resolve_setup(preconditions: ScenarioPreconditions) -> tuple[SetupMode, SetupOrigin]:
    """Return the declared app state, or the independent-scenario default."""
    if preconditions.app_state is not None:
        return preconditions.app_state, SetupOrigin.DECLARED
    return SetupMode.CLEAR_DATA, SetupOrigin.DEFAULT


def apply_app_precondition(
    device_control,
    mode: SetupMode,
    origin: SetupOrigin,
    collect_launch_ttid: bool = False,
) -> AppSetupEvidence:
    """Establish the requested app state before the planner sees a screen."""
    source = "none"
    if mode is SetupMode.CLEAR_DATA:
        if not clear_app_data(device_control):
            raise DeviceUnavailable("The requested clean app state was not applied.")
        source = (
            "adb shell pm clear"
            if device_control.platform is Platform.ANDROID
            else "fresh app installation"
        )
    launch_timing = None
    measurement_failure = None
    if mode is not SetupMode.PRESERVE:
        if collect_launch_ttid and device_control.platform is Platform.ANDROID:
            try:
                launch_timing = measure_android_launch_ttid(device_control)
            except MeasurementUnavailable as exc:
                measurement_failure = str(exc)
            source = (
                f"{source}; measured app restart"
                if source != "none"
                else "measured app restart"
            )
        else:
            restart_app(device_control)
            source = f"{source}; app restart" if source != "none" else "app restart"
    return AppSetupEvidence(
        operation=LifecycleOperation(
            operation_id="app_state.setup",
            phase=LifecyclePhase.SETUP,
            kind="app_state",
            requested=mode.value,
            status=LifecycleStatus.CONFIRMED,
            source=source,
            observed=f"{mode.value} ({origin.value})",
        ),
        launch_timing=launch_timing,
        measurement_failure=measurement_failure,
    )


def restore_app_precondition(device_control, mode: SetupMode) -> Optional[LifecycleOperation]:
    """Return a clean-state scenario to its initial clean boundary.

    Clearing app data is intentionally destructive, so it cannot recreate the
    data that existed before the scenario. It can still guarantee that a
    scenario which began clean does not leave app data or permissions behind
    for the next scenario.
    """
    if mode is not SetupMode.CLEAR_DATA:
        return None
    source = (
        "adb shell pm clear"
        if device_control.platform is Platform.ANDROID
        else "fresh app installation"
    )
    if not clear_app_data(device_control):
        return failed_operation(
            "app_state.restore",
            LifecyclePhase.RESTORATION,
            "app_state",
            "clear_data",
            source,
            "The clean app-state boundary could not be restored.",
        )
    return LifecycleOperation(
        operation_id="app_state.restore",
        phase=LifecyclePhase.RESTORATION,
        kind="app_state",
        requested="clear_data",
        status=LifecycleStatus.RESTORED,
        source=source,
        observed="clean app state",
    )


def _capability_name(capability: Capability) -> str:
    variant = capability.support_variant
    return f"{capability.kind.value}:{variant}" if variant else capability.kind.value


def apply_capability_precondition(
    controller: CapabilityController, capability: Capability
) -> tuple[LifecycleOperation, Receipt]:
    """Apply and confirm a declared device precondition."""
    receipt = controller.execute(capability)
    if receipt is None or not receipt.confirmed:
        raise DeviceUnavailable(
            f"The {capability.kind.value} precondition has no confirmed read-back."
        )
    capability_name = _capability_name(capability)
    return (
        LifecycleOperation(
            operation_id=f"capability.{capability_name}.setup",
            phase=LifecyclePhase.SETUP,
            kind="capability",
            requested=capability_name,
            status=LifecycleStatus.CONFIRMED,
            source=receipt.source,
            observed="confirmed",
        ),
        receipt,
    )


def failed_operation(
    operation_id: str,
    phase: LifecyclePhase,
    kind: Literal["app_state", "capability", "device_state"],
    requested: str,
    source: str,
    reason: str,
) -> LifecycleOperation:
    """Build a scrub-ready record for an operation that did not finish."""
    return LifecycleOperation(
        operation_id=operation_id,
        phase=phase,
        kind=kind,
        requested=requested,
        status=LifecycleStatus.FAILED,
        source=source,
        failure_reason=reason,
    )


def restoration_operations(
    events: list[RestorationEvent],
) -> list[LifecycleOperation]:
    """Convert device restoration results into canonical run evidence."""
    if not events:
        return [
            LifecycleOperation(
                operation_id="device_state.restore",
                phase=LifecyclePhase.RESTORATION,
                kind="device_state",
                requested="original state",
                status=LifecycleStatus.RESTORED,
                source="state snapshot",
                observed="no changed state",
            )
        ]
    operations = []
    for index, event in enumerate(events, start=1):
        operations.append(
            LifecycleOperation(
                operation_id=f"device_state.restore.{index}",
                phase=LifecyclePhase.RESTORATION,
                kind="device_state",
                requested=event.state,
                status=(LifecycleStatus.RESTORED if event.restored else LifecycleStatus.FAILED),
                source="state snapshot",
                observed="restored" if event.restored else None,
                failure_reason=event.reason,
            )
        )
    return operations
