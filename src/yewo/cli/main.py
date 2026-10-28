"""Yewo command-line entrypoint.

Each command owns its own parser and handler. This module only assembles them,
and sets up what every command needs: logging, .env loading and configuration.
"""

import argparse
import sys
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv

from yewo import __version__
from yewo.cli import config as config_command
from yewo.cli import devices, doctor, evaluation, lock, run, setup, test
from yewo.cli.console import Console, configure_logging
from yewo.config import load_config
from yewo.defaults import LOG_LEVELS
from yewo.errors import UsageError

COMMANDS = (doctor, setup, devices, config_command, evaluation, test, lock, run)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yewo",
        description="AI-driven mobile end-to-end testing agent.",
    )
    parser.add_argument("--version", action="version", version=f"yewo {__version__}")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=LOG_LEVELS,
        help="Logging verbosity (default: INFO)",
    )

    subcommands = parser.add_subparsers(dest="command", metavar="<command>")
    for command in COMMANDS:
        command.register(subcommands)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parses the command line and dispatches to the requested command."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    args, extra = parser.parse_known_args(argv)
    configure_logging(args.log_level)
    # usecwd: without it find_dotenv walks up from this module's directory,
    # so an installed Yewo would never see the .env next to your app.
    load_dotenv(find_dotenv(usecwd=True))

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 2

    if not getattr(args, "passthrough", False):
        # Re-parse strictly, so an unknown flag is an error rather than ignored.
        args = parser.parse_args(argv)
    args.extra = extra

    try:
        return handler(args, load_config())
    except UsageError as exc:
        Console(stream=sys.stderr).write(str(exc))
        return 2


if __name__ == "__main__":
    sys.exit(main())
