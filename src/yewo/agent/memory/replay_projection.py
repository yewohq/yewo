"""Project raw perception into the controls a saved path may use."""

from collections.abc import Sequence

from yewo.agent.grounding import GROUNDING_MARKS
from yewo.agent.grounding.marks import project_targets
from yewo.agent.perception.elements import GUIElement, ScanInfo


def project_actionable_elements(
    elements: Sequence[GUIElement],
    scan: ScanInfo,
    grounding: str,
) -> tuple[GUIElement, ...]:
    """Return exactly the controls available to the active grounding mode."""
    if grounding != GROUNDING_MARKS:
        return tuple(elements)
    projection = project_targets(
        elements,
        screen_width=scan.screen_width,
        screen_height=scan.screen_height,
        keyboard_open=scan.keyboard_effective_visible,
        include_keyboard_targets=scan.keyboard_targets_expanded,
    )
    return projection.actionable_elements
