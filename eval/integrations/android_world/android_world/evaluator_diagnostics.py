"""Read-only diagnostics around strict AndroidWorld evaluator predicates."""

from __future__ import annotations

import logging
from functools import wraps

logger = logging.getLogger(__name__)


def install_clock_diagnostics() -> None:
    """Log the exact inputs used by the stopwatch evaluator."""
    from android_world.task_evals.single import clock

    original = clock._is_stopwatch_running
    if getattr(original, "_yewo_diagnostic", False):
        return

    @wraps(original)
    def diagnosed(ui_elements, current_activity):
        result = original(ui_elements, current_activity)
        descriptions = [element.content_description for element in ui_elements]
        logger.info(
            "Clock evaluator input: activity_matches=%s pause_count=%d "
            "lap_count=%d result=%s",
            "DeskClock" in current_activity,
            descriptions.count("Pause"),
            descriptions.count("Lap"),
            result,
        )
        return result

    diagnosed._yewo_diagnostic = True
    clock._is_stopwatch_running = diagnosed
