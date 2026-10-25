"""
Rolling-window loop and stagnation detector.
Tracks action repetition and page stagnation independently, emitting
escalating nudge messages before hard-terminating.
"""

import hashlib
from collections import deque
from typing import Optional

_NUDGE_SOFT = 4
_NUDGE_MEDIUM = 7
_NUDGE_STRONG = 10
_TERMINATE_AT = 13
_STAGNANT_THRESHOLD = 7

_NUDGE_MESSAGES = {
    _NUDGE_SOFT: (
        "Note: You appear to be repeating the same action. "
        "If intentional, continue. Otherwise, consider a different approach."
    ),
    _NUDGE_MEDIUM: (
        "Warning: You have repeated the same action several times without progress. "
        "Re-evaluate whether this action is working. Try a different element, "
        "action type, or strategy."
    ),
    _NUDGE_STRONG: (
        "Critical: You are stuck in a repetition loop. The current approach is not working. "
        "You MUST try a completely different strategy — dismiss the keyboard (key_event 4), "
        "scroll to find elements, or reassess the goal entirely."
    ),
}

_STAGNANT_MESSAGE = (
    "Warning: The screen has not changed for several consecutive steps. "
    "Your actions are having no visible effect. "
    "Check for keyboards, overlays, or permission dialogs blocking interaction. "
    "Try dismissing the keyboard (key_event 4) before attempting your next action."
)


class RollingLoopDetector:
    """Tracks action repetition and page stagnation across a rolling window of steps."""

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._action_window: deque[str] = deque(maxlen=window_size)
        self._last_page_hash: Optional[str] = None
        self._stagnant_count: int = 0
        self._action_repetition_terminated: bool = False
        self._page_stagnation_terminated: bool = False

    def check(
        self, action_hash: Optional[str], page_hash: Optional[str]
    ) -> Optional[str]:
        """Record this step's action and page state. Returns a nudge message or None."""
        if page_hash is not None:
            if self._last_page_hash is not None and page_hash == self._last_page_hash:
                self._stagnant_count += 1
            else:
                self._stagnant_count = 0
                self._page_stagnation_terminated = False
            self._last_page_hash = page_hash

        if self._stagnant_count >= _STAGNANT_THRESHOLD:
            self._page_stagnation_terminated = True
            return None

        if action_hash is not None:
            self._action_window.append(action_hash)
            repeat_count = sum(1 for h in self._action_window if h == action_hash)

            if repeat_count >= _TERMINATE_AT:
                self._action_repetition_terminated = True
                return None

            for threshold in (_NUDGE_STRONG, _NUDGE_MEDIUM, _NUDGE_SOFT):
                if repeat_count >= threshold:
                    return _NUDGE_MESSAGES[threshold]

        if self._stagnant_count > 0 and self._stagnant_count % 3 == 0:
            return _STAGNANT_MESSAGE

        return None

    def terminated(self) -> bool:
        return self._action_repetition_terminated or self._page_stagnation_terminated

    @staticmethod
    def hash_action(action) -> str:
        """Normalized hash for an action — collapses minor coordinate variations into the same bucket."""
        from yewo.agent.actuation.actions import ActionType

        t = action.type.value if hasattr(action.type, "value") else str(action.type)

        # A target named by number has no point until it is resolved, and an
        # unresolvable one never gets there — repeats of it still have to be
        # counted, so bucket by the number instead.
        coordinates = getattr(action, "coordinates", None)
        if coordinates is None:
            spot = f"mark:{getattr(action, 'mark', None)}"
        else:
            spot = f"{coordinates.x // 40}:{coordinates.y // 40}"

        if t == ActionType.TYPE.value:
            key = f"type:{spot}"
        elif t in (
            ActionType.TAP.value,
            ActionType.DOUBLE_TAP.value,
            ActionType.LONG_PRESS.value,
            ActionType.CLEAR.value,
        ):
            key = f"{t}:{spot}"
        elif t == ActionType.KEY_EVENT.value:
            key = f"key:{action.key}"
        elif t == ActionType.SWIPE.value:
            key = f"swipe:{action.start_x // 40}:{action.start_y // 40}:{action.end_x // 40}:{action.end_y // 40}"
        elif t == ActionType.DRAG_AND_DROP.value:
            key = f"drag:{action.source_x // 40}:{action.source_y // 40}:{action.destination_x // 40}:{action.destination_y // 40}"
        elif t == ActionType.CAPABILITY.value:
            key = f"cap:{action.capability}"
        else:
            key = t

        return hashlib.sha256(key.encode()).hexdigest()[:12]
