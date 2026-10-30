"""Run the Yewo command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from importlib.metadata import version
import signal
import sys
from threading import Event
from types import FrameType

from yewo.cli.run_request import add_run_arguments, run_request_from_namespace
from yewo.devices.logs import DeclaredValueDeviceLogScrubber
from yewo.runs import (
    RunMode,
    RunRequest,
    RunTerminal,
    RunTerminalCategory,
    InterruptionSignal,
    build_default_run_coordinator,
)
from yewo.runs.device_foundation import (
    DeviceFoundationScript,
    DeviceFoundationScriptError,
    DeviceFoundationWorkflow,
    load_device_foundation_script,
)
from yewo.runs.errors import RunCoordinationError, RunTerminalPersistenceError


RunExecutor = Callable[
    [RunRequest, DeviceFoundationScript, InterruptionSignal],
    RunTerminal,
]


class _TerminationSignal:
    def __init__(self) -> None:
        self._requested = Event()

    def request(self) -> None:
        self._requested.set()

    def is_set(self) -> bool:
        return self._requested.is_set()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yewo", description="Test a mobile application.")
    parser.add_argument("--version", action="version", version=version("yewo"))
    commands = parser.add_subparsers(dest="command", required=True)
    add_run_arguments(commands.add_parser("run", help="Run a deterministic device script."))
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    executor: RunExecutor | None = None,
) -> int:
    """Parse command-line arguments."""
    parser = _parser()
    values = parser.parse_args(argv)
    try:
        request = run_request_from_namespace(values)
        if request.workflow_input_path is None:
            raise DeviceFoundationScriptError("A device-foundation script is required.")
        script = load_device_foundation_script(request.workflow_input_path)
        if script.platform is not request.application.platform:
            raise DeviceFoundationScriptError(
                "The device-foundation script and application platforms must match."
            )
        interruption = _TerminationSignal()
        with _capture_termination(interruption):
            terminal = (executor or _execute_run)(request, script, interruption)
    except (argparse.ArgumentTypeError, DeviceFoundationScriptError) as error:
        parser.error(str(error))
    except RunTerminalPersistenceError as error:
        _print_terminal(error.terminal)
        print("Yewo could not persist the run terminal.", file=sys.stderr)
        return 1
    except RunCoordinationError:
        print("Yewo could not preserve the run record.", file=sys.stderr)
        return 1
    _print_terminal(terminal)
    if terminal.category is RunTerminalCategory.COMPLETED:
        return 0
    if terminal.category is RunTerminalCategory.INTERRUPTED:
        return 130
    return 1


def _execute_run(
    request: RunRequest,
    script: DeviceFoundationScript,
    interruption: InterruptionSignal,
) -> RunTerminal:
    scrubber = DeclaredValueDeviceLogScrubber(script.device_log_sensitive_values)
    workflow = DeviceFoundationWorkflow(script)
    coordinator = build_default_run_coordinator(
        {RunMode.DEVICE_FOUNDATION: workflow},
        scrubber,
    )
    return coordinator.run(request, interruption=interruption)


def _print_terminal(terminal: RunTerminal) -> None:
    print(f"{terminal.run_id} {terminal.category.value}")


@contextmanager
def _capture_termination(
    interruption: _TerminationSignal,
) -> Generator[None, None, None]:
    def request_termination(signum: int, frame: FrameType | None) -> None:
        del signum, frame
        interruption.request()

    try:
        previous = signal.signal(signal.SIGTERM, request_termination)
    except ValueError:
        previous = None
    try:
        yield
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
