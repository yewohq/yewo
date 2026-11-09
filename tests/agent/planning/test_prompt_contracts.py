"""Prompt contracts for fields the model must fill but cannot infer."""

import re

from yewo.agent.actuation import ACTION_DEFINITIONS, ActionHandling
from yewo.agent.planning.prompt import TASK_PLANNER_SYSTEM_PROMPT


def _keyboard_rule() -> str:
    paragraphs = TASK_PLANNER_SYSTEM_PROMPT.split("\n\n")
    return next(paragraph for paragraph in paragraphs if "Executor constraint" in paragraph)


def _action_names(handling: ActionHandling, *, keyboard_blocked: bool | None = None) -> set[str]:
    return {
        action.value
        for action, definition in ACTION_DEFINITIONS.items()
        if definition.handling is handling
        and keyboard_blocked in (None, definition.blocked_when_keyboard_visible)
    }


def test_keyboard_prompt_actions_match_catalog() -> None:
    # The prompt names actions as bare words, so only whole tokens count:
    # "untappable" must not read as a mention of tap.
    named = set(re.findall(r"[a-z_]+", _keyboard_rule()))
    selectable = _action_names(ActionHandling.EXECUTE, keyboard_blocked=False) | _action_names(
        ActionHandling.TERMINAL
    )
    blocked = _action_names(ActionHandling.EXECUTE, keyboard_blocked=True)

    assert selectable <= named
    assert not blocked & named
