"""`yewo run` — drive the app towards a single goal."""

import argparse
import logging
from pathlib import Path

from yewo.app_discovery import resolve_app_path
from yewo.config import Config
from yewo.preflight import require_run
from yewo.schema import PLATFORMS

logger = logging.getLogger(__name__)


def add_device_arguments(parser: argparse.ArgumentParser) -> None:
    """Device and model flags.

    Defaults stay None so an explicit flag is distinguishable from an absent
    one; yewo.config supplies the value when the flag is omitted.
    """
    parser.add_argument("--platform", choices=list(PLATFORMS), help="Target platform")
    parser.add_argument("--device", help="Name of emulator/simulator/device")
    parser.add_argument("--model", help="LiteLLM model to use")
    parser.add_argument(
        "--perception-model",
        help="Pinned detector package: hf://owner/repository@commit",
    )
    parser.add_argument("--fast-model", help="Cheaper/faster LiteLLM model for per-step planning")
    parser.add_argument("--output-dir", help="Directory to save outputs and screenshots")
    parser.add_argument("--max-steps", type=int, help="Max steps per run")
    parser.add_argument("--settle-time", type=float, help="Seconds to wait after each action")
    parser.add_argument(
        "--headless", action="store_true", default=None, help="Run the emulator headlessly"
    )


def register(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser(
        "run",
        help="Drive the app towards a single goal, with no QA reporting",
        description="Drive the app towards a single plain-English goal.",
    )
    parser.add_argument("goal", help="What the agent should accomplish, in plain English")
    parser.add_argument(
        "--app-path",
        help="Path to the app binary (.apk, .app or .ipa). Discovered from the "
        "current directory when omitted.",
    )
    parser.add_argument(
        "--keep-device",
        action="store_true",
        help="Leave the emulator running and the app installed, so the next run starts warm",
    )
    parser.add_argument(
        "--reuse-app",
        action="store_true",
        help="Skip reinstalling the app. Only use when the build has not changed — "
        "a rebuilt debug APK keeps its version and would be silently skipped",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Skip the screen recording",
    )
    parser.add_argument(
        "--no-persist-inputs",
        action="store_true",
        help="Record the hash of every prompt and not its text",
    )
    add_device_arguments(parser)
    parser.set_defaults(handler=run)


def run(args: argparse.Namespace, config: Config) -> int:
    from yewo.agent.device.platform import Platform
    from yewo.agent.run.agent import DeviceAgent
    from yewo.agent.run.standalone import run_goal

    model = config.resolve("model", args.model)
    fast_model = config.resolve("fast_model", args.fast_model)
    perception_model = config.resolve(
        "perception_model", getattr(args, "perception_model", None)
    )
    platform = config.resolve("platform", args.platform)

    require_run(model, fast_model)

    app_path = resolve_app_path(
        config.resolve("app_path", args.app_path), platform, Path.cwd()
    )

    agent = DeviceAgent(
        model=model,
        fast_model=fast_model,
        max_steps=config.resolve("max_steps", args.max_steps),
        settle_time=config.resolve("settle_time", args.settle_time),
        # A single goal is done when it is done; journeys keep auditing past it.
        stop_when_satisfied=True,
    )

    try:
        result = run_goal(
            agent,
            goal=args.goal,
            device_name=config.resolve("device", args.device),
            app_path=str(app_path),
            platform=Platform.named(platform),
            assets_dir=Path(config.resolve("output_dir", args.output_dir)),
            headless=config.resolve("headless", args.headless),
            record_session=not args.no_record,
            keep_device=args.keep_device,
            reuse_app=args.reuse_app,
            persist_inputs=not args.no_persist_inputs,
            perception_model=perception_model,
        )
    except KeyboardInterrupt:
        logger.info("Agent execution interrupted by user")
        return 130

    if result.status == "goal_reached":
        logger.info("Goal reached in %d step(s).", result.steps_executed)
        return 0
    logger.error(
        "Goal not reached (%s) after %d step(s): %s",
        result.status,
        result.steps_executed,
        result.failure_reason or "no reason reported",
    )
    return 1
