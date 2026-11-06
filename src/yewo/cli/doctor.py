"""Public ``yewo doctor`` command and report rendering."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from yewo.doctor import DoctorReport, DoctorStatus, run_doctor


@dataclass(frozen=True)
class DoctorCommandOptions:
    """Typed inputs for one Doctor invocation."""

    platform: str | None
    as_json: bool


def register(commands: argparse._SubParsersAction) -> None:
    """Register the read-only environment diagnosis command."""
    parser = commands.add_parser(
        "doctor",
        help="Check whether this machine and project can run Yewo.",
        description="Report each Yewo prerequisite and how to fix every blocking gap.",
    )
    parser.add_argument(
        "--platform",
        choices=("android", "ios", "all"),
        help="Platform to inspect. The configured platform is the default.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Emit machine-readable output for CI.",
    )


def run(arguments: argparse.Namespace, start: Path | None = None) -> int:
    """Run Doctor, render one report, and return its blocking status."""
    return run_options(
        DoctorCommandOptions(platform=arguments.platform, as_json=arguments.as_json),
        start=start,
    )


def run_options(options: DoctorCommandOptions, start: Path | None = None) -> int:
    """Run Doctor from typed command inputs."""
    report = run_doctor(start, options.platform)
    if options.as_json:
        print(json.dumps(report.json_payload(), indent=2))
    else:
        _render(report)
    return 1 if report.blocking else 0


def _render(report: DoctorReport) -> None:
    print("Yewo doctor")
    for section in report.sections:
        print(f"\n{section.title}")
        for check in section.checks:
            label = check.status.value.upper().ljust(4)
            print(f"  {label}  {check.name}: {check.detail}")
            if check.hint and check.status in {DoctorStatus.WARN, DoctorStatus.FAIL}:
                print(f"        Fix: {check.hint}")
    print()
    if report.blocking:
        print(f"{report.blocking} blocking issue(s). {report.warnings} warning(s).")
        print("Fix the blocking issues, then run yewo doctor again.")
    elif report.warnings:
        print(f"Ready with {report.warnings} warning(s).")
    else:
        print("No issues found. Yewo is ready.")
