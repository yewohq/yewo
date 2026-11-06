"""Run the Yewo command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version

from dotenv import find_dotenv, load_dotenv

from yewo.agent import (
    YewoAgent,
    YewoRunConfiguration,
)
from yewo.cli import doctor as doctor_command
from yewo.cli import run as run_command
from yewo.cli import setup as setup_command
from yewo.devices.logs import DeclaredValueDeviceLogScrubber
from yewo.models import ModelInputError, build_model_selections, validate_model_credentials
from yewo.project_config import YewoProjectCompileError, YewoProjectError
from yewo.runs import RunMode, RunTerminalCategory, build_default_run_coordinator
from yewo.telemetry import TelemetryEvent, record_event, record_new_installation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yewo", description="Test a mobile application.")
    parser.add_argument("--version", action="version", version=version("yewo"))
    commands = parser.add_subparsers(dest="command", required=True)
    run_command.register(commands)
    doctor_command.register(commands)
    setup_command.register(commands)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and dispatch one explicit product command."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    record_new_installation()
    if arguments.command == "doctor":
        return doctor_command.run(arguments)
    if arguments.command == "setup":
        return setup_command.run(arguments)
    try:
        configuration = run_command.configuration(arguments)
        validate_model_credentials(configuration.model_profiles)
    except (YewoProjectError, YewoProjectCompileError, ModelInputError) as error:
        parser.error(str(error))
    try:
        return _run_yewo(configuration)
    except KeyboardInterrupt:
        return 130


def _run_yewo(configuration: YewoRunConfiguration) -> int:
    record_event(
        TelemetryEvent.RUN_STARTED,
        {
            "run_mode": configuration.run.mode.value,
            "device_platform": configuration.run.application.platform.value,
            "is_memory_enabled": configuration.memory_root is not None,
        },
    )
    scrubber = DeclaredValueDeviceLogScrubber(())

    def scrub_text(text: str) -> str:
        return scrubber(text).text

    workflow = YewoAgent(
        configuration.task,
        configuration.mode_policy,
        configuration.model_profiles,
        configuration.grounding_profile,
        configuration.planner_profile_id,
        configuration.auditor_profile_id,
        configuration.recovery_profile_id,
        configuration.action_verifier_profile_id,
        configuration.replay_preparation_profile_id,
        scrub_text,
        screen_analyzer_profile_id=configuration.screen_analyzer_profile_id,
        memory_root=configuration.memory_root,
        redaction_regions=configuration.redaction_regions,
    )
    coordinator = build_default_run_coordinator(
        {RunMode.NORMAL: workflow},
        scrubber,
    )
    terminal = coordinator.run(configuration.run)
    record_event(
        TelemetryEvent.RUN_FINISHED,
        {
            "terminal_category": terminal.category.value,
            "terminal_stage": terminal.stage.value,
            "is_partial_evidence_available": terminal.partial_evidence_available,
            **{
                f"model_{role}": selection
                for role, selection in build_model_selections(configuration.model_profiles).items()
            },
        },
    )
    print(f"terminal: {terminal.category.value} — {terminal.reason}")
    return 0 if terminal.category is RunTerminalCategory.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
