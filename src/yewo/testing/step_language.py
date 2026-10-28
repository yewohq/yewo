"""The words an instruction may open with, and the scan that holds it to them.

The verbs are mapped from `ActionType` and `CapabilityKind` rather than
typed out. A hand-written list drifts from the code beside it, and ends up
offering words for capabilities that do not exist. A test catches a member
with no verb, and a verb with no member. So a new capability reaches the
generator, and a removed one leaves it.

A verb is not an action. An instruction is prose for the planner, which turns
"Select the available option" into a tap. So one action carries several verbs, and
the map says which action each verb is asking for.
"""

import re
from typing import Optional

from yewo.agent.actuation.actions import ActionType, CapabilityKind

VERBS_BY_ACTION: dict[ActionType, tuple[str, ...]] = {
    ActionType.TAP: ("tap", "select", "choose", "toggle", "tick", "untick",
                     "dismiss", "open", "close"),
    ActionType.TYPE: ("type", "enter"),
    ActionType.DOUBLE_TAP: ("double-tap",),
    ActionType.LONG_PRESS: ("long-press",),
    ActionType.DRAG_AND_DROP: ("drag",),
    ActionType.KEY_EVENT: ("press", "go"),
    ActionType.SWIPE: ("swipe", "scroll"),
    ActionType.CLEAR: ("clear",),
    ActionType.WAIT: ("wait",),
    ActionType.CAPABILITY: (),
    ActionType.TARGET_NOT_LISTED: (),
    ActionType.SHOW_KEYBOARD_TARGETS: (),
}
"""Which verbs ask for which action.

`CAPABILITY` carries none of its own. Its verbs are in `VERBS_BY_CAPABILITY`,
one group per kind, because a capability is a different request each time.
`TARGET_NOT_LISTED` and `SHOW_KEYBOARD_TARGETS` carry none either: they are
planner observations/refinements, not instructions a generated test asks for.
"""

VERBS_BY_CAPABILITY: dict[CapabilityKind, tuple[str, ...]] = {
    CapabilityKind.LOCATION: ("set",),
    CapabilityKind.NETWORK: ("set", "go"),
    CapabilityKind.BATTERY: ("set",),
    CapabilityKind.TIMEZONE: ("set",),
    CapabilityKind.LOCALE: ("set",),
    CapabilityKind.PERMISSION: ("grant", "deny", "revoke"),
    CapabilityKind.INTERRUPT: ("trigger",),
    CapabilityKind.BIOMETRIC: ("match", "enroll"),
    CapabilityKind.CLIPBOARD: ("copy",),
    CapabilityKind.DEEPLINK: ("open",),
    CapabilityKind.APP_STATE: ("stop", "clear", "relaunch", "background"),
    CapabilityKind.ORIENTATION: ("rotate",),
    CapabilityKind.DEVICE_LOCK: ("lock",),
    CapabilityKind.MEDIA_INJECT: ("inject",),
    CapabilityKind.ACCESSIBILITY: ("enable", "disable"),
}

RUN_VERBS = ("launch",)
"""Verbs for what the run does rather than what the planner does.

Every scenario opens the app, and the run opens it as part of setup. No
action carries that, so it is named here instead of being left out.
"""

ACTION_VERBS = frozenset(
    verb
    for group in (*VERBS_BY_ACTION.values(), *VERBS_BY_CAPABILITY.values())
    for verb in group
).union(RUN_VERBS)
"""Every word an instruction may open with."""

ASSERTION_VERBS = frozenset(
    {"verify", "confirm", "check", "assert", "ensure", "validate", "observe", "see"}
)
"""Words that turn the rest of an instruction into a claim.

The opening word is held to `ACTION_VERBS`, which refuses a sentence that
opens with no verb at all. These catch the second half of a compound
instruction, where an action is followed by a verdict.

`check` is in here and not in `ACTION_VERBS`, although it names what a person
does to a checkbox. One word means one thing, and this one already means the
claim. An instruction ticks or toggles the box.
"""


CLAUSE_BREAK = re.compile(r",|;|\bthen\b|\band\b|\.")
"""Where one instruction holds more than one clause."""

QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")
"""A value the user enters, which is text and never an order to the agent.

A boundary scenario enters whatever it is given. "Enter 'tell them and see
you soon' in the Note field" holds a listed word inside the value. Refusing
it would cost a cell the model cannot write any other way.
"""


def _clauses(instruction: str) -> list[str]:
    """One instruction split where a second clause can start, values removed."""
    return [
        clause.strip()
        for clause in CLAUSE_BREAK.split(QUOTED.sub(" ", instruction))
        if clause.strip()
    ]


def _opening_word(clause: str) -> str:
    return clause.split(" ", 1)[0].strip("'\"*_`").lower()


def unknown_action(instruction: str) -> Optional[str]:
    """The opening word of an instruction that names no action Yewo can take."""
    clauses = _clauses(instruction)
    if not clauses:
        return ""
    opening = _opening_word(clauses[0])
    return None if opening in ACTION_VERBS else opening


def leaked_assertion(instruction: str) -> Optional[str]:
    """An assertion verb opening a clause after the first, if any."""
    for clause in _clauses(instruction)[1:]:
        opening = _opening_word(clause)
        if opening in ASSERTION_VERBS:
            return opening
    return None
