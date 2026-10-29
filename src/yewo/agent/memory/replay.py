"""Replaying transitions the agent has already worked out.

Every scenario opens after a fresh app launch, so the same opening flow is
planned again and again. Successful transitions are remembered and replayed
directly. The taps are real. Only the deliberation is skipped.
"""

import logging
from dataclasses import dataclass
from typing import Callable, Optional

from yewo.agent.actuation.actions import (
    Action,
    ClearAction,
    DoubleTapAction,
    KeyEventAction,
    LongPressAction,
    TapAction,
    TypeAction,
)
from yewo.agent.actuation.safety import (
    ActionSafetyPolicy,
    ActionSource,
    REPLAYABLE_ACTION_TYPES,
    REPLAYABLE_KEYCODES,
    execute_authorized,
)
from yewo.agent.memory.history import ActionEntry, ActionResult
from yewo.agent.memory.screen_identity import (
    SCREEN_PHASH_DISTANCE_THRESHOLD,
    find_matching_screen,
    is_same_screen,
)
from yewo.identity import new_uuid
from yewo.agent.run.instruction_controller import InstructionController

logger = logging.getLogger(__name__)

PHASH_THRESHOLD = SCREEN_PHASH_DISTANCE_THRESHOLD
MAX_REPLAY_ACTIONS = 30

# Replay re-executes without planning or auditing, so only actions that move
# between screens qualify. Gestures and capability changes are what destructive
# scenarios are built from, and must never be inherited by another scenario.
REPLAYABLE_ACTIONS = REPLAYABLE_ACTION_TYPES

PLACED_ACTIONS = (
    TypeAction,
    TapAction,
    ClearAction,
    DoubleTapAction,
    LongPressAction,
)
"""The actions that name a point on the screen.

Listed rather than asked for, because the union is closed and a member
that grew coordinates should be added here on purpose.
"""

# Back, which is how the agent dismisses the keyboard. Other key events reach
# system surfaces that have nothing to do with navigating the app.
ACTION_SAFETY_POLICY = ActionSafetyPolicy()


@dataclass
class ReplayStep:
    """An action that is known to lead from one screen to another."""

    action: Action
    result_phash: str
    description: str
    target_label: str = ""


class ReplayLedger:
    """Remembers which action moved the app on from each screen."""

    def __init__(self, threshold: int = PHASH_THRESHOLD):
        self._steps: dict[str, ReplayStep] = {}
        self._threshold = threshold

    def __len__(self) -> int:
        return len(self._steps)

    def _match(self, phash: Optional[str]) -> Optional[str]:
        """The known screen closest to this one, within the pHash threshold."""
        if not phash:
            return None
        return find_matching_screen(phash, self._steps, self._threshold)

    def record(
        self,
        from_phash: Optional[str],
        action: Action,
        to_phash: Optional[str],
        description: str,
        target_label: str = "",
    ) -> None:
        """Remembers an action that demonstrably moved the app to a new screen."""
        if not from_phash or not to_phash or from_phash == to_phash:
            return
        if action.type not in REPLAYABLE_ACTIONS:
            return
        if self._match(from_phash) is not None:
            return
        self._steps[from_phash] = ReplayStep(action, to_phash, description, target_label)

    def lookup(self, phash: Optional[str]) -> Optional[ReplayStep]:
        known = self._match(phash)
        return self._steps.get(known) if known else None

    def leads_to(self, step: ReplayStep, phash: Optional[str]) -> bool:
        """Whether the screen reached matches the one this step produced before."""
        return phash is not None and is_same_screen(
            phash, step.result_phash, self._threshold
        )


def is_replayable(step: ReplayStep, goal_text: str) -> bool:
    """Whether a remembered step is one this scenario actually asked for.

    A tap is not safe by virtue of being a tap. The same gesture can submit a
    change, delete data, or grant a permission. A step only replays when this
    scenario's own instructions name what it touches. They must name the text
    it types or the label of the control it taps. Planning takes over for
    anything the scenario never mentions.
    """
    action = step.action
    if action.type not in REPLAYABLE_ACTIONS:
        return False

    goal = goal_text.lower()

    if isinstance(action, TypeAction):
        return (action.text or "").lower() in goal

    if isinstance(action, KeyEventAction):
        return action.key in REPLAYABLE_KEYCODES

    label = (step.target_label or "").strip().lower()
    return bool(label) and label in goal


def target_label(action: Action, elements) -> str:
    """The visible text of the element an action targets, if it has one."""
    if not isinstance(action, PLACED_ACTIONS) or not elements:
        return ""
    coordinates = action.coordinates
    for element in elements:
        box = element.bounding_box
        if (
            box.x_min <= coordinates.x <= box.x_max
            and box.y_min <= coordinates.y <= box.y_max
            and (element.text or "").strip()
        ):
            return element.text.strip()
    return ""


def replay_known_path(
    ledger: Optional[ReplayLedger],
    device_control,
    goal,
    history,
    quick_phash: Callable[[object], Optional[str]],
    settle: Callable[[object], Optional[str]],
    scrubber=None,
    instruction_controller: Optional[InstructionController] = None,
    context_id: Optional[str] = None,
) -> int:
    """Re-executes transitions already worked out, without consulting the planner.

    Each replayed action is verified against the screen it produced before.
    The first disagreement hands control back to planning. A changed app then
    costs one wasted tap rather than a wrong result.
    """
    if not ledger or not len(ledger):
        return 0

    goal_text = "\n".join(goal) if isinstance(goal, list) else str(goal)
    replayed = 0

    while replayed < MAX_REPLAY_ACTIONS:
        current_phash = quick_phash(device_control)
        step = ledger.lookup(current_phash)
        if step is None:
            break
        if not is_replayable(step, goal_text):
            logger.info("Replay stops here: %s is this scenario's own input.", step.description)
            break

        try:
            authorization = ACTION_SAFETY_POLICY.authorize(
                step.action,
                source=ActionSource.REPLAY,
                observation_id=new_uuid(),
                screen_phash=current_phash,
                screen_size=None,
                context_id=context_id,
                instruction_controller=instruction_controller,
            )
            execute_authorized(authorization, device_control, scrubber)
        except Exception as exc:
            logger.info("Replay stopped — %s failed: %s", step.description, exc)
            break

        if not ledger.leads_to(step, settle(device_control)):
            logger.info("Replay stopped — %s did not lead where it used to.", step.description)
            break

        replayed += 1
        history.append(
            ActionEntry(
                step_num=0,
                action_desc=f"{step.description} (replayed)",
                result=ActionResult(success=True, observation="Replayed a known transition"),
            )
        )

    if replayed:
        logger.info("Replayed %d known step(s) before planning.", replayed)
    return replayed
