"""Run the Yewo command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from yewo.agent import (
    NormalAgentWorkflow,
    NormalRunConfiguration,
    NormalRunConfigurationError,
    load_normal_run_configuration,
)
from yewo.devices.logs import DeclaredValueDeviceLogScrubber
from yewo.models import ModelInputError, validate_model_credentials
from yewo.runs import RunMode, RunTerminalCategory, build_default_run_coordinator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yewo", description="Test a mobile application.")
    parser.add_argument("--version", action="version", version=version("yewo"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run one configured normal device task.")
    run.add_argument("configuration", type=Path, help="Path to a normal-run JSON file.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and dispatch one explicit product command."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        configuration = load_normal_run_configuration(arguments.configuration)
        validate_model_credentials(configuration.model_profiles)
    except (NormalRunConfigurationError, ModelInputError) as error:
        parser.error(str(error))
    return _run_normal(configuration)


def _run_normal(configuration: NormalRunConfiguration) -> int:
    scrubber = DeclaredValueDeviceLogScrubber(())
    workflow = NormalAgentWorkflow(
        configuration.task,
        configuration.mode_policy,
        configuration.model_profiles,
        configuration.grounding_profile,
        configuration.planner_profile_id,
        _preserve_text,
        redaction_regions=configuration.redaction_regions,
    )
    coordinator = build_default_run_coordinator(
        {RunMode.NORMAL: workflow},
        scrubber,
    )
    terminal = coordinator.run(configuration.run)
    print(f"terminal: {terminal.category.value} — {terminal.reason}")
    return 0 if terminal.category is RunTerminalCategory.COMPLETED else 1


def _preserve_text(text: str) -> str:
    return text


if __name__ == "__main__":
    raise SystemExit(main())
