"""Verified lifecycle interruptions and adverse network conditions."""

from typing import Callable, Optional, TypeVar

from yewo.agent.actuation.actions import (
    AppStateCapability,
    Capability,
    DeviceLockCapability,
    NetworkCapability,
)
from yewo.agent.device.capability_maps import (
    ANDROID_NETWORK_PROFILES,
    active_ios_condition,
    ios_network_profile,
)
from yewo.agent.device.driver_commands import DriverCommands
from yewo.agent.device.emulator_network import network_shape, set_network_shape
from yewo.agent.device.receipts import Receipt
from yewo.errors import ActionRejected

State = TypeVar("State")


def _wait_for(
    read: Callable[[], State],
    accepted: Callable[[State], bool],
    sleep: Callable[[float], None],
    attempts: int = 20,
) -> State:
    """Read an asynchronous device transition for at most two seconds."""
    observed = read()
    for _ in range(attempts - 1):
        if accepted(observed):
            break
        sleep(0.1)
        observed = read()
    return observed


def _receipt(
    target: str,
    capability: Capability,
    requested: str,
    observed: Optional[str],
    source: str,
) -> Receipt:
    return Receipt(
        capability=capability.kind.value,
        requested=requested,
        observed=observed,
        target=target,
        source=source,
    )


def _network_shape_text(shape: dict) -> str:
    return (
        f"up={float(shape['upload_kbps']):g}kbps,"
        f"down={float(shape['download_kbps']):g}kbps,"
        f"latency={int(shape['min_latency_ms'])}:{int(shape['max_latency_ms'])}ms"
    )


def apply_android_network_profile(
    target: str, capability: NetworkCapability, commands: DriverCommands
) -> Receipt:
    profile = ANDROID_NETWORK_PROFILES[capability.mode]
    speed = f"{profile['upload_kbps']}:{profile['download_kbps']}"
    delay = f"{profile['min_latency_ms']}:{profile['max_latency_ms']}"
    set_network_shape(target, speed, delay)
    observed = network_shape(target, commands)
    return _receipt(
        target,
        capability,
        _network_shape_text(profile),
        _network_shape_text(vars(observed)) if observed is not None else None,
        "adb emu network status",
    )


def apply_ios_network_profile(
    target: str, capability: NetworkCapability, commands: DriverCommands
) -> Receipt:
    conditions = commands.condition_inducers()
    if conditions is None:
        raise ActionRejected(
            "network cannot be applied on ios: the physical-device condition "
            "profiles could not be read"
        )
    selected = ios_network_profile(conditions, capability.mode)
    if selected is None:
        available = sorted(
            str(profile.get("name"))
            for condition in conditions
            for profile in condition.get("profiles", ())
            if profile.get("name")
        )
        raise ActionRejected(
            "network cannot be applied on ios: no installed condition profile "
            "matches it. Available profiles: "
            + (", ".join(available) or "none")
        )
    active = active_ios_condition(conditions)
    if active != selected:
        if active is not None:
            commands.disable_condition_inducer()
        commands.enable_condition_inducer(*selected)
    after = commands.condition_inducers()
    observed = active_ios_condition(after) if after is not None else None
    return _receipt(
        target,
        capability,
        f"{selected[0]}/{selected[1]}",
        f"{observed[0]}/{observed[1]}" if observed else "inactive",
        "mobile: listConditionInducers",
    )


def relaunch(
    target: str,
    capability: AppStateCapability,
    app_id: str,
    commands: DriverCommands,
    sleep: Callable[[float], None],
) -> Receipt:
    commands.terminate_app(app_id)
    stopped = _wait_for(
        lambda: commands.app_state(app_id),
        lambda state: state is not None and state <= 1,
        sleep,
    )
    if stopped is None or stopped > 1:
        raise ActionRejected(
            "The app relaunch was not started: queryAppState did not confirm "
            "that the app stopped"
        )
    commands.activate_app(app_id)
    foreground = _wait_for(
        lambda: commands.app_state(app_id), lambda state: state == 4, sleep
    )
    observed = "stopped->foreground" if foreground == 4 else "stopped->not foreground"
    return _receipt(
        target,
        capability,
        "stopped->foreground",
        observed,
        "mobile: queryAppState (before and after activateApp)",
    )


def background_resume(
    target: str,
    capability: AppStateCapability,
    app_id: str,
    commands: DriverCommands,
    sleep: Callable[[float], None],
) -> Receipt:
    background_command_sent = False
    try:
        commands.background_app()
        background_command_sent = True
        state = _wait_for(
            lambda: commands.app_state(app_id),
            lambda observed: observed in (2, 3),
            sleep,
        )
        if state not in (2, 3):
            raise ActionRejected(
                "The background/resume cycle was not confirmed: queryAppState "
                f"reported {state!r} instead of a background state"
            )
        sleep(capability.duration_seconds or 0)
    finally:
        if background_command_sent:
            commands.activate_app(app_id)
    foreground = _wait_for(
        lambda: commands.app_state(app_id), lambda state: state == 4, sleep
    )
    observed = "background->foreground" if foreground == 4 else "background->not foreground"
    return _receipt(
        target,
        capability,
        "background->foreground",
        observed,
        "mobile: queryAppState (before and after activateApp)",
    )


def lock_cycle(
    target: str,
    capability: DeviceLockCapability,
    commands: DriverCommands,
    sleep: Callable[[float], None],
) -> Receipt:
    before = commands.is_locked()
    if before is None:
        raise ActionRejected(
            "The lock cycle cannot run because the device lock state is unreadable"
        )
    if before:
        raise ActionRejected("The lock cycle cannot run because the device was already locked")
    lock_command_sent = False
    try:
        commands.lock()
        lock_command_sent = True
        locked = _wait_for(commands.is_locked, lambda state: state is True, sleep)
        if locked is not True:
            raise ActionRejected(
                "The lock cycle was not confirmed: isLocked did not report locked"
            )
        sleep(capability.duration_seconds)
    finally:
        if lock_command_sent:
            commands.unlock()
    recovered = _wait_for(commands.is_locked, lambda state: state is False, sleep) is False
    return _receipt(
        target,
        capability,
        "locked->unlocked",
        "locked->unlocked" if recovered else "locked->still locked",
        "mobile: isLocked (before and after unlock)",
    )
