"""Read, write, and preserve Android emulator WAN shaping."""

import logging
import re
import subprocess
from dataclasses import dataclass
from typing import Optional

from yewo.agent.device.adb_control import AdbControl
from yewo.agent.device.driver_commands import DriverCommands
from yewo.utils.subprocess_runner import SubprocessRunError, run_command

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmulatorNetworkShape:
    upload_kbps: float
    download_kbps: float
    min_latency_ms: int
    max_latency_ms: int

    def speed_argument(self) -> str:
        return f"{self.upload_kbps}:{self.download_kbps}"

    def delay_argument(self) -> str:
        return f"{self.min_latency_ms}:{self.max_latency_ms}"


def set_network_shape(serial: str, speed: str, delay: str) -> None:
    """Set both halves of an emulator's WAN condition."""
    AdbControl.adb(
        serial,
        "emu",
        "network",
        "speed",
        speed,
        doing=f"set emulator network speed to {speed}",
    )
    AdbControl.adb(
        serial,
        "emu",
        "network",
        "delay",
        delay,
        doing=f"set emulator network delay to {delay}",
    )


def network_status(
    serial: str, commands: Optional[DriverCommands] = None
) -> Optional[str]:
    """A console answer, which adb may write to either output stream."""
    if commands is not None:
        answer = commands.emulator_console("network status")
        if answer is not None:
            return answer
    try:
        result = run_command(
            ["adb", "-s", serial, "emu", "network", "status"],
            timeout=20,
            log_command=False,
        )
    except (SubprocessRunError, subprocess.TimeoutExpired, OSError):
        return None
    output = (result.stdout or "").strip() or (result.stderr or "").strip()
    return output or None


def network_shape(
    serial: str, commands: Optional[DriverCommands] = None
) -> Optional[EmulatorNetworkShape]:
    """Parse the emulator console's read-back of speed and latency."""
    status = network_status(serial, commands)
    if not status:
        return None

    def rate(label: str) -> Optional[float]:
        found = re.search(
            rf"{label}\s+speed\s*:\s*([\d.]+)\s*(k?bits/s)",
            status,
            re.IGNORECASE,
        )
        if not found:
            return None
        value = float(found.group(1))
        return value if found.group(2).lower().startswith("k") else value / 1000

    def latency(label: str) -> Optional[int]:
        found = re.search(
            rf"{label}\s+latency\s*:\s*([\d.]+)\s*ms",
            status,
            re.IGNORECASE,
        )
        return int(float(found.group(1))) if found else None

    upload = rate("upload")
    download = rate("download")
    minimum = latency("minimum")
    maximum = latency("maximum")
    if upload is None or download is None or minimum is None or maximum is None:
        logger.warning("Could not parse emulator network status: %s", status)
        return None
    return EmulatorNetworkShape(upload, download, minimum, maximum)
