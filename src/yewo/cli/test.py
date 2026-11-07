"""Public ``yewo test`` command backed by journeys in the nearest YAML project."""

from __future__ import annotations

import argparse
from pathlib import Path

from yewo.project_config import (
    compile_yewo_test_configurations,
    load_nearest_yewo_project,
)
from yewo.test_config import YewoTestConfiguration


def register(commands: argparse._SubParsersAction) -> None:
    """Register journey testing with optional workflow and target selection."""
    parser = commands.add_parser(
        "test",
        help="Discover and test journeys declared in yewo.yaml.",
        description=(
            "Discover each selected journey, generate evidence-grounded scenarios, "
            "and execute them on each selected target."
        ),
    )
    parser.add_argument(
        "--workflow",
        help="Comma-separated workflow names. By default every workflow is tested.",
    )
    parser.add_argument(
        "--target",
        help="Comma-separated target IDs. By default every selected target is tested.",
    )


def configurations(
    arguments: argparse.Namespace,
    invocation_root: Path | None = None,
) -> tuple[YewoTestConfiguration, ...]:
    """Load and compile selected journey-target runs without touching a device."""
    root = (invocation_root or Path.cwd()).resolve()
    project = load_nearest_yewo_project(root)
    return compile_yewo_test_configurations(
        project,
        workflow_selector=arguments.workflow,
        target_selector=arguments.target,
    )
