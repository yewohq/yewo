"""Run the Yewo command-line interface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

from yewo.agent import (
    YewoAgent,
    YewoRunConfiguration,
    YewoRunConfigurationError,
    load_yewo_run_configuration,
)
from yewo.devices.logs import DeclaredValueDeviceLogScrubber
from yewo.models import ModelInputError, validate_model_credentials
from yewo.runs import RunMode, RunTerminalCategory, build_default_run_coordinator


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="yewo", description="Test a mobile application.")
    parser.add_argument("--version", action="version", version=version("yewo"))
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Ask Yewo to perform one mobile task.")
    run.add_argument(
        "configuration",
        type=Path,
        help="Path to a Yewo run JSON file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate and dispatch one explicit product command."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        configuration = load_yewo_run_configuration(arguments.configuration)
        validate_model_credentials(configuration.model_profiles)
    except (YewoRunConfigurationError, ModelInputError) as error:
        parser.error(str(error))
    return _run_yewo(configuration)


def _run_yewo(configuration: YewoRunConfiguration) -> int:
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
        scrub_text,
        memory_root=configuration.memory_root,
        redaction_regions=configuration.redaction_regions,
    )
    coordinator = build_default_run_coordinator(
        {RunMode.NORMAL: workflow},
        scrubber,
    )
    terminal = coordinator.run(configuration.run)
    print(f"terminal: {terminal.category.value} — {terminal.reason}")
    return 0 if terminal.category is RunTerminalCategory.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
