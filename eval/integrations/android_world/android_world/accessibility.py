"""Keep AndroidWorld's accessibility forwarder bound between episodes."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Sequence

from absl import logging

FORWARDER_PACKAGE = "com.google.androidenv.accessibilityforwarder"
FORWARDER_COMPONENT = (
    f"{FORWARDER_PACKAGE}/{FORWARDER_PACKAGE}.AccessibilityForwarder"
)
FLAGS_RECEIVER = f"{FORWARDER_PACKAGE}/{FORWARDER_PACKAGE}.FlagsBroadcastReceiver"
TREE_LOG_ACTION = "accessibility_forwarder.intent.action.ENABLE_ACCESSIBILITY_TREE_LOGS"
GRPC_ACTION = "accessibility_forwarder.intent.action.SET_GRPC"

AdbCommand = Callable[[Sequence[str]], str]


def parse_enabled_services(output: str) -> list[str]:
    """Return real component names from Android's colon-delimited setting."""
    return [value for value in output.strip().split(":") if value and value != "null"]


def is_forwarder_running(pidof_output: str) -> bool:
    """Return whether Android reports a live forwarder process."""
    pids = pidof_output.split()
    return bool(pids) and all(pid.isdigit() for pid in pids)


def grpc_port(no_proxy: str) -> str | None:
    """Read the host gRPC port that AndroidWorld placed in ``no_proxy``."""
    match = re.search(r"(?:^|,)10\.0\.2\.2:(\d+)(?:,|$)", no_proxy.strip())
    return match.group(1) if match else None


def restart_forwarder(
    run_adb: AdbCommand,
    *,
    timeout: float = 10.0,
    poll_interval: float = 0.25,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Toggle the forwarder and wait until Android reports it as bound."""
    logging.warning("Accessibility forwarder is unbound; restarting it.")
    enabled = parse_enabled_services(
        run_adb(["shell", "settings", "get", "secure", "enabled_accessibility_services"])
    )
    port = grpc_port(run_adb(["shell", "settings", "get", "global", "no_proxy"]))
    preserved = [service for service in enabled if service != FORWARDER_COMPONENT]

    run_adb(["shell", "settings", "delete", "secure", "enabled_accessibility_services"])
    # Settings observers run asynchronously. If the component is re-added
    # immediately, Android can coalesce both writes and never unbind/rebind the
    # service. Give AccessibilityManager time to observe the disabled state.
    sleep(max(0.5, poll_interval))
    run_adb(["shell", "settings", "put", "secure", "accessibility_enabled", "1"])
    run_adb(
        [
            "shell",
            "settings",
            "put",
            "secure",
            "enabled_accessibility_services",
            ":".join([*preserved, FORWARDER_COMPONENT]),
        ]
    )

    attempts = max(1, math.ceil(timeout / poll_interval))
    for attempt in range(attempts):
        process = run_adb(["shell", "pidof", FORWARDER_PACKAGE])
        if is_forwarder_running(process):
            break
        if attempt + 1 < attempts:
            sleep(poll_interval)
    else:
        logging.error(
            "AndroidWorld accessibility forwarder did not start after recovery; "
            "continuing so AndroidWorld can use or diagnose its own connection."
        )
        return False

    logging.info("Accessibility forwarder is bound.")

    run_adb(["shell", "am", "broadcast", "-a", TREE_LOG_ACTION, "-n", FLAGS_RECEIVER])
    if port is not None:
        run_adb(
            [
                "shell",
                "am",
                "broadcast",
                "-a",
                GRPC_ACTION,
                "--ei",
                "port",
                port,
                "-n",
                FLAGS_RECEIVER,
            ]
        )
    return True


def ensure_forwarder_bound(run_adb: AdbCommand) -> bool:
    """Recover a forwarder that Android killed during an episode."""
    process = run_adb(["shell", "pidof", FORWARDER_PACKAGE])
    if not is_forwarder_running(process):
        return restart_forwarder(run_adb)
    logging.info("Accessibility forwarder is already bound.")
    return True
