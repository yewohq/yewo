"""Public ``yewo setup`` command for explicit project and tool mutations."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from yewo.cli import doctor as doctor_command
from yewo.setup import SetupContext, SetupReport, run_setup


def register(commands: argparse._SubParsersAction) -> None:
    """Register the previewable setup command."""
    parser = commands.add_parser(
        "setup",
        help="Create missing project files and prepare automation tools.",
        description=(
            "Prepare Yewo, Appium, and the requested platform. "
            "Existing project files are never overwritten."
        ),
    )
    parser.add_argument(
        "--platform",
        choices=("android", "ios", "all"),
        help="Platform to prepare. The configured platform is the default.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show every proposed action without changing files or tools.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        dest="assume_yes",
        help="Approve every proposed setup action.",
    )


def run(arguments: argparse.Namespace, start: Path | None = None) -> int:
    """Execute or preview Setup and verify completed work with Doctor."""
    root = (start or Path.cwd()).resolve()
    report = run_setup(
        SetupContext(
            root=root,
            requested_platform=arguments.platform,
            dry_run=arguments.dry_run,
            assume_yes=arguments.assume_yes,
            confirm=_confirm,
        )
    )
    _render(report, arguments.dry_run)
    if arguments.dry_run:
        print("\nPLANNED  verification: would run yewo doctor")
        return 1 if report.failures else 0
    print("\nVerification")
    doctor_exit = doctor_command.run_options(
        doctor_command.DoctorCommandOptions(platform=arguments.platform, as_json=False),
        start=root,
    )
    return 1 if report.failures or doctor_exit else 0


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


def _render(report: SetupReport, dry_run: bool) -> None:
    print("Yewo setup" + (" (dry run)" if dry_run else ""))
    print(f"Platforms: {', '.join(report.platforms)}")
    for step in report.steps:
        label = step.status.value.upper().ljust(7)
        print(f"  {label}  {step.name}: {step.detail}")
        if step.command:
            print(f"           Command: {' '.join(step.command)}")
        if step.hint:
            print(f"           Hint: {step.hint}")
