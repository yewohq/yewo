"""Machine-readable planner action contracts."""

import re
from typing import cast

from yewo.agent.actuation import ACTION_DEFINITIONS, ActionHandling
from yewo.agent.planning.prompt import TASK_PLANNER_SYSTEM_PROMPT, planner_action_catalog


def test_planner_catalog_is_derived_from_runtime_action_definitions() -> None:
    catalog = planner_action_catalog(ACTION_DEFINITIONS)
    definitions_by_name = {
        action.value: definition for action, definition in ACTION_DEFINITIONS.items()
    }

    assert {entry["action"] for entry in catalog} == set(definitions_by_name)
    for entry in catalog:
        definition = definitions_by_name[str(entry["action"])]
        assert entry["blocked_while_keyboard_visible"] is definition.blocked_when_keyboard_visible
        required_fields = cast(list[str], entry["required_fields"])
        assert set(required_fields) == definition.required_fields


def test_keyboard_prompt_actions_match_catalog() -> None:
    paragraph = next(
        item for item in TASK_PLANNER_SYSTEM_PROMPT.split("\n\n") if "Executor constraint" in item
    )
    named = set(re.findall(r"[a-z_]+", paragraph))
    selectable = {
        action.value
        for action, definition in ACTION_DEFINITIONS.items()
        if definition.handling is ActionHandling.TERMINAL
        or (
            definition.handling is ActionHandling.EXECUTE
            and not definition.blocked_when_keyboard_visible
        )
    }
    blocked = {
        action.value
        for action, definition in ACTION_DEFINITIONS.items()
        if definition.handling is ActionHandling.EXECUTE
        and definition.blocked_when_keyboard_visible
    }

    assert selectable <= named
    assert not blocked & named


def test_planner_prompt_keeps_non_structural_safety_rules() -> None:
    required_rules = (
        "Do not use untargeted type to change focus",
        "A refinement cannot name an element_ref",
        "Choose done only when current evidence",
        "Neither a grounding correction nor an interaction refinement is available for scroll or drag",
    )

    assert all(rule in TASK_PLANNER_SYSTEM_PROMPT for rule in required_rules)
