"""One goal, against a device session this module opens and closes."""

import hashlib
import logging
from pathlib import Path

from yewo import __version__, identity
from yewo.agent.device.platform import Platform
from yewo.agent.device.session import DeviceSession
from yewo.agent.perception.blind import for_grounding
from yewo.agent.run.outcomes import DeviceRunResult
from yewo.llm.prompts import PromptStore
from yewo.testing import sensitivity
from yewo.testing.report.usage_tracker import LLMUsageTracker

logger = logging.getLogger(__name__)

AGENT = "agent"
"""Every call a goal run makes. One client answers for all of them."""


def run_goal(
    agent,
    goal: str,
    device_name: str,
    app_path: str,
    platform: Platform = Platform.ANDROID,
    assets_dir: Path = Path("assets"),
    headless: bool = True,
    record_session: bool = True,
    keep_device: bool = False,
    reuse_app: bool = False,
    persist_inputs: bool = True,
) -> DeviceRunResult:
    """Opens a session, runs the goal against it, and closes it either way.

    The run leaves a manifest, as a journey run does. It needs no report: a
    goal reached is the whole verdict. Without a manifest it could not be
    read back at all, and nothing could say which build it walked.
    """
    session = DeviceSession(
        device_name=device_name,
        app_path=app_path,
        platform=platform,
        record_session=record_session,
        session_dir=assets_dir,
        keep_device=keep_device,
        reuse_app=reuse_app,
    )
    agent.record_prompts(
        PromptStore(
            session.prompts_dir,
            scrub=agent.scrubber.scrub,
            keep_bodies=persist_inputs,
        )
    )
    # A goal run costs the same per call as a journey run, and left
    # unattributed its prompts would be files nothing points at.
    usage = LLMUsageTracker(
        output_dir=session.telemetry_dir, scrubber=agent.scrubber
    )
    agent.llm.usage_callback = lambda call: usage.add_call(AGENT, call)
    if agent.fast_llm is not agent.llm:
        agent.fast_llm.usage_callback = lambda call: usage.add_call(AGENT, call)
    agent.step_opened = usage.mark

    manifest = None
    try:
        device_control = session.start(is_headless=headless)
        manifest = _manifest(session, agent, goal, platform, device_control)
        result = agent.run(
            goal=goal,
            device_control=device_control,
            perception=for_grounding(agent.grounding, session),
            session_dir=session.screenshots_dir,
        )
        manifest.record_unattached_calls(usage.calls_outside_steps(AGENT))
        manifest.record_goal(result, usage=usage.summary())
        return result
    except BaseException as interruption:
        if manifest is not None:
            manifest.abort(f"Aborted: {type(interruption).__name__}")
        raise
    finally:
        try:
            usage.save_report()
        except Exception as exc:  # noqa: BLE001 - telemetry must not fail a run
            logger.warning("Could not serialize the usage of this run: %s", exc)
        session.close()


def _manifest(session, agent, goal: str, platform: Platform, device_control):
    """The record for this run, opened as soon as the device answers."""
    from yewo.testing.models import RunEnvironment
    from yewo.testing.report.manifest import RunInputs, RunManifest, RunSubject

    return RunManifest(
        session.manifest_path,
        RunSubject(
            kind="goal",
            id=hashlib.sha256(goal.encode("utf-8")).hexdigest()[:16],
            name=agent.scrubber.scrub(goal),
            description=agent.scrubber.scrub(goal),
        ),
        scenarios=(),
        environment=RunEnvironment(
            yewo_version=__version__,
            model=agent.llm.model,
            platform=platform.value,
            grounding=agent.grounding,
            app=identity.app_identity(
            session.app_path, installed=not session.reuse_app
        ),
            device=identity.device_identity(device_control),
            git=identity.git_identity(),
            ci=identity.ci_identity(),
        ),
        inputs=RunInputs(
            subject="goal",
            subject_sha256=hashlib.sha256(goal.encode("utf-8")).hexdigest(),
        ),
        scrubber=agent.scrubber,
        mode="run",
        artifacts=sensitivity.declared_for(_goal_journey(goal)),
    )


def _goal_journey(goal: str):
    """A goal in the shape the artifact classes read.

    A goal is prose a person typed, so it declares no secret and nothing in
    the run can mask one. The classes are the same for every run, and this
    keeps one owner for them rather than a second list here.
    """
    from yewo.testing.models import UserJourney

    return UserJourney(id="goal", name=goal, description=goal)
