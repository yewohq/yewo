"""Building the runner one workflow needs, from the settings that apply to it.

A setting can come from four places, and the order matters. The flag the user
typed wins, then the workflow block, then yewo.yaml and the YEWO_* variables,
then the built-in default. Resolving that in one place keeps every caller from
re-deciding it.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from yewo.agent.device.platform import Platform
from yewo.app_discovery import resolve_app_path
from yewo.config import SETTINGS, Config
from yewo.defaults import DEFAULT_GROUNDING, DEFAULT_PERCEPTION_MODEL

from .cold_start import require_clean_room
from .discovery_wiring import NO_SETUP_CONTRACT_SHA256, LiveDiscoveryProvider
from .runner import YewoTestRunner
from .suite import Target, Workflow

logger = logging.getLogger(__name__)


def as_bool(raw: Any, unset: bool = False) -> bool:
    """A setting read as a flag, whichever spelling a file or a variable used.

    `unset` is what a setting nobody wrote means. Most default to off. One
    that is on until somebody says otherwise passes True.
    """
    if raw is None:
        return unset
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def runner_for(
    workflow: Workflow,
    args: Any,
    filepath: Path,
    config: Config,
    target: Optional[Target] = None,
    assets_dir: Optional[Path] = None,
    run_id: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> YewoTestRunner:
    """The runner for one workflow, with every setting resolved in precedence order."""

    def setting(name: str, flag: Any) -> Any:
        if flag is not None:
            return flag
        if target is not None and name in target.settings:
            return target.settings[name]
        if name in workflow.settings:
            return workflow.settings[name]
        # grounding is a flag and a file key only, so the caller supplies the
        # default rather than yewo.config.
        return config.get(name) if name in SETTINGS else None

    platform_str = setting("platform", args.platform)
    app_path = resolve_app_path(
        setting("app_path", args.app_path),
        platform_str,
        Path.cwd(),
        filepath.resolve().parent,
    )
    output_root = assets_dir or Path(setting("output_dir", args.output_dir))
    # A saved path is on unless somebody says no, so a cold run turning it off
    # is not overriding a request. Asking for reuse and for a cold run is.
    durable_replay = args.durable_replay and as_bool(setting("durable_replay", None), True)
    knowledge_root: Optional[Path] = None
    cold_start = None
    if bool(getattr(args, "cold", False)):
        output_root = output_root / "cold"
        knowledge_root = output_root / "knowledge"
        durable_replay = False
        cold_start = require_clean_room(
            filepath,
            knowledge_root,
            output_root,
            replay=bool(args.replay),
            durable_replay=durable_replay,
            reuse_app=bool(args.reuse_app),
            locked=bool(args.locked),
        )
    return YewoTestRunner(
        platform=Platform.named(platform_str),
        device_name=setting("device", args.device),
        app_path=str(app_path),
        model=setting("model", args.model),
        perception_model=setting(
            "perception_model", getattr(args, "perception_model", None)
        ),
        assets_dir=output_root,
        max_steps=int(setting("max_steps", args.max_steps)),
        settle_time=float(setting("settle_time", args.settle_time)),
        headless=as_bool(setting("headless", args.headless)),
        fast_model=setting("fast_model", args.fast_model),
        record_session=(False if args.no_record else as_bool(setting("record", None))),
        replay=bool(args.replay),
        durable_replay=durable_replay,
        update_paths=bool(args.update_paths),
        reuse_app=bool(args.reuse_app),
        keep_app=bool(args.keep_app),
        grounding=setting("grounding", args.grounding) or DEFAULT_GROUNDING,
        journey_file=filepath,
        workflow=workflow.name or "",
        locked=bool(args.locked),
        persist_inputs=not bool(args.no_persist_inputs),
        allow_secret_exposure=as_bool(
            setting("allow_secret_exposure", args.allow_secret_exposure)
        ),
        policy_waivers=workflow.policy_waivers,
        run_id=run_id,
        execution_id=execution_id,
        target_id=target.id if target is not None else None,
        knowledge_root=knowledge_root,
        cold_start=cold_start,
        discovery_factory=_discovery_factory(
            knowledge_root,
            app_path=str(app_path),
            platform=platform_str,
            model=setting("model", args.model),
            perception_model=setting(
                "perception_model", getattr(args, "perception_model", None)
            )
            or DEFAULT_PERCEPTION_MODEL,
        ),
    )


def output_dir_for(workflow: Workflow, args: Any, config: Config) -> Path:
    """Resolve the artifact root without constructing a device runner."""
    if args.output_dir is not None:
        return Path(args.output_dir)
    if "output_dir" in workflow.settings:
        return Path(workflow.settings["output_dir"])
    return Path(config.get("output_dir"))


def _discovery_factory(
    knowledge_root: Optional[Path],
    *,
    app_path: str,
    platform: str,
    model: str,
    perception_model: str,
):
    """How this run builds bounded discovery, or nothing when it may not.

    Discovery writes an AppMap and RouteMemory, so it needs somewhere of its
    own to write them. A run with no knowledge root reuses a reviewed lock
    instead, and asks for no discovery at all.
    """
    if knowledge_root is None:
        return None

    def build(
        device_control,
        perception,
        llm_client,
        scrubber,
        artifact_writer=None,
        diagnostic_dir=None,
    ) -> LiveDiscoveryProvider:
        return LiveDiscoveryProvider(
            device_control,
            perception,
            model=model,
            perception_model=perception_model,
            app_path=app_path,
            platform=platform,
            setup_contract_sha256=NO_SETUP_CONTRACT_SHA256,
            knowledge_root=knowledge_root,
            scrubber=scrubber,
            usage_callback=llm_client.usage_callback,
            # The run's store, so discovery's prompts land beside every other
            # lane's rather than going unrecorded.
            prompt_store=llm_client.prompt_store,
            artifact_writer=artifact_writer,
            diagnostic_dir=diagnostic_dir,
        )

    return build
