"""Public ``yewo run`` command backed by the nearest YAML project."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from yewo.agent import YewoRunConfiguration
from yewo.project_config import (
    RunConfigurationOverrides,
    compile_yewo_run_configuration,
    load_nearest_yewo_project,
    YewoProjectCompileError,
)


def register(commands: argparse._SubParsersAction) -> None:
    """Register the standalone plain-English goal command."""
    parser = commands.add_parser(
        "run",
        help="Drive the app towards one plain-English goal.",
        description=(
            "Drive the configured app towards one plain-English goal. "
            "Settings come from the nearest yewo.yaml."
        ),
    )
    parser.add_argument("goal", help="What Yewo should accomplish, in plain English.")
    parser.add_argument("--platform", choices=("android", "ios"), help="Target platform.")
    parser.add_argument("--device", help="Emulator, simulator, or device name.")
    parser.add_argument("--app-path", type=Path, help="Application artifact path.")
    parser.add_argument("--model", help="Default semantic model in provider/model form.")
    parser.add_argument("--output-dir", type=Path, help="Directory for run artifacts.")
    parser.add_argument(
        "--max-steps",
        type=_bounded_integer(1, 500),
        help="Maximum Device Agent steps.",
    )
    parser.add_argument(
        "--settle-time",
        type=_bounded_float(0, 5),
        help="Seconds between settlement observations.",
    )


def configuration(
    arguments: argparse.Namespace,
    invocation_root: Path | None = None,
) -> YewoRunConfiguration:
    """Load and compile one standalone run without touching a device."""
    root = (invocation_root or Path.cwd()).resolve()
    project = load_nearest_yewo_project(root)
    try:
        overrides = RunConfigurationOverrides(
            platform=arguments.platform,
            device_name=arguments.device,
            app_path=_absolute(arguments.app_path, root),
            model=arguments.model,
            output_dir=_absolute(arguments.output_dir, root),
            max_steps=arguments.max_steps,
            settle_time=arguments.settle_time,
        )
    except ValidationError as error:
        raise YewoProjectCompileError("The command-line run overrides are invalid.") from error
    return compile_yewo_run_configuration(project, arguments.goal, overrides=overrides)


def _absolute(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _bounded_integer(minimum: int, maximum: int) -> Callable[[str], int]:
    def parse(raw: str) -> int:
        try:
            value = int(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be an integer") from error
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return value

    return parse


def _bounded_float(minimum: float, maximum: float) -> Callable[[str], float]:
    def parse(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError("must be a number") from error
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return value

    return parse
