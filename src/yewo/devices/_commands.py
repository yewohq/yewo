"""Shared subprocess boundary for read-only device discovery."""

from __future__ import annotations

from collections.abc import Sequence
import subprocess
from pathlib import Path
from typing import Protocol

from yewo.devices.errors import DeviceCommandError, DeviceInventoryError
from yewo.run_narration import current_narrator


_NARRATION_SOURCE = "device-command"


class CommandRunner(Protocol):
    """Execute one command and capture its result."""

    def __call__(
        self, command: Sequence[str], *, timeout_seconds: float
    ) -> subprocess.CompletedProcess[str]: ...


def run_command(
    command: Sequence[str], *, timeout_seconds: float
) -> subprocess.CompletedProcess[str]:
    """Execute a discovery command without a shell."""
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def capture_command(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    timeout_seconds: float = 30,
) -> str:
    """Return stdout or raise an explicit inventory failure."""
    try:
        result = execute_command(
            runner,
            command,
            timeout_seconds=timeout_seconds,
            purpose="discover devices",
        )
    except DeviceCommandError as error:
        raise DeviceInventoryError(str(error)) from error
    return result.stdout or ""


def execute_command(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    timeout_seconds: float = 30,
    purpose: str = "run a device command",
) -> subprocess.CompletedProcess[str]:
    """Execute one required platform command."""
    narrator = current_narrator()
    if narrator is not None:
        executable = Path(command[0]).name
        narrator.trace(
            _NARRATION_SOURCE,
            f"{executable}: {len(command) - 1} arguments omitted state=starting",
        )
    try:
        result = runner(command, timeout_seconds=timeout_seconds)
    except (OSError, subprocess.SubprocessError) as error:
        raise DeviceCommandError(f"Could not {purpose} with {command[0]}.") from error
    if result.returncode:
        if narrator is not None:
            narrator.trace(
                _NARRATION_SOURCE,
                f"{Path(command[0]).name}: state=failed exit_code={result.returncode}",
            )
        detail = (result.stderr or result.stdout).strip() or "No command output."
        raise DeviceCommandError(
            f"Could not {purpose} with {command[0]}. The exit code was "
            f"{result.returncode}: {detail}"
        )
    if narrator is not None:
        narrator.trace(_NARRATION_SOURCE, f"{Path(command[0]).name}: state=completed exit_code=0")
    return result


def capture_optional(
    runner: CommandRunner,
    command: Sequence[str],
    *,
    timeout_seconds: float = 20,
) -> str | None:
    """Return trimmed stdout when an optional platform read succeeds."""
    narrator = current_narrator()
    executable = Path(command[0]).name
    if narrator is not None:
        narrator.trace(
            _NARRATION_SOURCE,
            f"{executable}: {len(command) - 1} arguments omitted state=starting optional=true",
        )
    try:
        result = runner(command, timeout_seconds=timeout_seconds)
    except (OSError, subprocess.SubprocessError):
        if narrator is not None:
            narrator.trace(_NARRATION_SOURCE, f"{executable}: state=unavailable optional=true")
        return None
    if result.returncode:
        if narrator is not None:
            narrator.trace(
                _NARRATION_SOURCE,
                f"{executable}: state=failed exit_code={result.returncode} optional=true",
            )
        return None
    output = (result.stdout or "").strip()
    if narrator is not None:
        narrator.trace(_NARRATION_SOURCE, f"{executable}: state=completed optional=true")
    return output or None
