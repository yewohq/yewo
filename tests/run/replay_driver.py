"""Driving the saved-path service the way the run loop drives it.

The loop asks for one cached step per turn, on the screen it already read.
A refusal costs that turn only. Tests that reimplemented the asking would
pass while the loop did something else, so they ask through the real
`saved_path.try_cached_step` instead.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

from yewo.agent.memory.history import AgentHistory
from yewo.agent.memory.loop_detector import RollingLoopDetector
from yewo.agent.run import saved_path
from yewo.agent.run.record import StepLog


class Artifacts:
    """Writes nothing. A screenshot is not what these tests are about."""

    @staticmethod
    def save_step(session_dir, step, snapshot) -> dict:
        return {}


def context() -> SimpleNamespace:
    """What the loop carries between its turns."""
    return SimpleNamespace(
        steps=StepLog(),
        history=AgentHistory(),
        loop_detector=RollingLoopDetector(),
        steps_executed=0,
        prev_phash=None,
    )


def agent_for(replay, settle_time: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        durable_replay=replay,
        settle_time=settle_time,
        scrubber=None,
        artifacts=Artifacts(),
    )


class Settling:
    """Stands in for the settle module, which needs a driver."""

    def __init__(self, phash: Optional[str] = "p"):
        self.phash = phash

    def settle(self, device_control, settle_time):
        return self.phash


def drive(
    replay,
    device_control,
    screens,
    settle_phash: Optional[str] = "p",
    ctx: Optional[SimpleNamespace] = None,
) -> Tuple[List[str], SimpleNamespace]:
    """One turn per screen, and what each turn came to.

    Returns a source per turn: "cache" where the saved path supplied the
    step, "model" where the planner would have had to. A screen is handed
    over once, as the loop hands over the reading it already took.
    """
    ctx = ctx or context()
    agent = agent_for(replay)
    settling = Settling(settle_phash)
    sources: List[str] = []
    for step, screen in enumerate(screens):
        ctx.steps.open(step + 1)
        took = saved_path.try_cached_step(
            agent, screen, device_control, ctx, settling, Path("unused"), step
        )
        sources.append("cache" if took else "model")
    return sources, ctx
