"""Compact model-facing projections of current planning evidence."""

from yewo.observations import Observation


def planning_observation_context(observation: Observation) -> dict[str, object]:
    """Project one observation without copying screenshot bytes or device internals."""
    return {
        "platform": observation.device.platform.value,
        "application_id": observation.application.application_id,
        "application_state": observation.application_state.name.lower(),
        "foreground_application_id": observation.foreground_application_id,
        "platform_context": observation.platform_context,
        "keyboard_visible": observation.keyboard_visible,
        "orientation": (
            observation.orientation.value if observation.orientation is not None else None
        ),
        "screen_fingerprint": observation.screen_fingerprint.value,
        "screenshot_sha256": observation.screenshot.sha256,
    }
