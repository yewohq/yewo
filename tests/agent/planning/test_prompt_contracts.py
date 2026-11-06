"""Prompt contracts for fields the model must fill but cannot infer."""

import re

from yewo.agent.actuation import ACTION_DEFINITIONS, ActionHandling
from yewo.agent.planning.prompt import TASK_PLANNER_SYSTEM_PROMPT


def _keyboard_rule() -> str:
    paragraphs = TASK_PLANNER_SYSTEM_PROMPT.split("\n\n")
    return next(paragraph for paragraph in paragraphs if "A visible keyboard blocks" in paragraph)


def test_keyboard_prompt_actions_match_catalog() -> None:
    available = {
        action.value
        for action, definition in ACTION_DEFINITIONS.items()
        if definition.handling is ActionHandling.EXECUTE
        and not definition.blocked_when_keyboard_visible
    }

    assert set(re.findall(r"`(\w+)`", _keyboard_rule())) == available
