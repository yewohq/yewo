"""`yewo lock` — refresh a suite from already verified app knowledge.

Cold generation belongs to `yewo test`, because only that command can run
bounded discovery. This command can refresh a lock without a device when the
lock already identifies verified knowledge.

Two jobs an ordinary run cannot do: lock with no device attached, and refresh
deliberately rather than as a side effect of testing.
"""

import argparse
import logging
from functools import partial
from pathlib import Path
from uuid import uuid4

from yewo.app_discovery import resolve_journey_file
from yewo.cli.console import Console
from yewo.config import Config
from yewo.errors import UsageError
from yewo.preflight import require_lock

logger = logging.getLogger(__name__)


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "lock",
        help="Refresh yewo.lock from verified app knowledge, with no device",
        description=(
            "Refresh scenarios from verified app knowledge and write yewo.lock "
            "beside the journey file. Run 'yewo test' first when discovery is required."
        ),
    )
    parser.add_argument(
        "journey_file",
        nargs="?",
        help="Path to the journey file. Found in this directory when omitted.",
    )
    parser.add_argument("--model", help="LiteLLM model to generate with")
    parser.add_argument(
        "--workflow",
        help="Which workflow(s) to lock, comma separated. Every one by default",
    )
    parser.add_argument(
        "--allow-secret-exposure",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow declared secrets in the model input that generation sends. "
        "Off by default, exactly as in 'yewo test'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="refresh",
        help="Ask again for every cell, rather than only the missing and stale ones. "
        "Use it to take a new model's suite deliberately",
    )
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    """Writes the lock for every journey the file declares.

    Returns:
        0 when every journey was locked, 1 when a cell could not be covered,
        and 2 when the file itself could not be read.
    """
    from yewo.llm.llm_client import LLMClient
    from yewo.testing import lock_file
    from yewo.testing.generation_startup import GenerationStartupCoordinator
    from yewo.testing.runner_knowledge import RunnerStartupKnowledgeSource
    from yewo.testing.sensitivity import require_safe_secret_policy
    from yewo.testing.suite import load_workflows, select

    console = Console()
    model = config.resolve("model", args.model)
    require_lock(model)
    allow_secret_exposure = bool(
        config.resolve("allow_secret_exposure", args.allow_secret_exposure)
    )

    try:
        path = resolve_journey_file(args.journey_file)
        workflows = select(load_workflows(path), args.workflow)
    except UsageError as exc:
        console.write(str(exc))
        return 2
    except Exception as exc:  # noqa: BLE001 - any unreadable file is one outcome
        console.write(f"Could not read {args.journey_file or 'the journey file'}: {exc}")
        return 2

    target = lock_file.path_beside(Path(path))
    llm_client = LLMClient(model=model, cache_affinity=f"lock-{uuid4()}")
    uncovered = 0

    for workflow in workflows:
        for journey in workflow.journeys:
            coordinator = GenerationStartupCoordinator(
                partial(require_safe_secret_policy, opted_in=allow_secret_exposure),
                RunnerStartupKnowledgeSource(
                    Path(path),
                    workflow.name,
                    Path(path).parent / ".yewo" / "knowledge",
                ),
            )
            startup = coordinator.resolve_suite(
                journey,
                llm_client,
                lock_path=target,
                workflow=workflow.name,
                refresh=args.refresh,
            )
            label = f"{workflow.name}/{journey.id}" if workflow.name else journey.id
            if startup.suite is None:
                uncovered += len(journey.coverage)
                console.write(
                    f"{label}: {startup.readiness.state.value}: "
                    f"{startup.readiness.reason}"
                )
                continue
            resolution = startup.suite
            console.write(f"{label}: {resolution.note}")
            for gap in resolution.uncovered:
                uncovered += 1
                console.write(f"  {gap.cell.key} not covered: {gap.reason}")

    console.write(f"Lock: {target}")
    return 1 if uncovered else 0
