"""Refusing a run that cannot have what its settings ask for.

A missing key, a missing extra and a grounding mode that needs one are all
answerable before a device is opened. Asked here, together, so that two
missing things cost one message rather than two boot cycles.
"""

from typing import Optional

from yewo.agent.grounding import (
    GROUNDING_DESCRIBE,
    GROUNDING_MARKS,
)
from yewo.defaults import DEFAULT_GROUNDING
from yewo.errors import UsageError
from yewo.llm.credentials import missing_credentials
from yewo.llm.jev_client import DEFAULT_DECISION_MODEL
from yewo.utils.jev_deps import INSTALL_HINT as JEV_HINT
from yewo.utils.jev_deps import missing_jev_packages
from yewo.utils.vision_deps import INSTALL_HINT as VISION_HINT
from yewo.utils.vision_deps import missing_vision_packages

DETECTED_GROUNDING = frozenset({GROUNDING_MARKS})
"""Modes that choose a target from detected elements, and so need a detector."""

CREDENTIAL_HINT = "Add them to a .env file in this directory, or export them."


def models_for(grounding: str) -> tuple[Optional[str], ...]:
    """Models a grounding mode calls for beyond the planner's own."""
    if grounding == GROUNDING_DESCRIBE:
        return (DEFAULT_DECISION_MODEL,)
    return ()


def packages_for(grounding: str) -> list[str]:
    """The extra a grounding mode needs and does not have, named with its fix."""
    if grounding in DETECTED_GROUNDING:
        absent, hint, why = (
            missing_vision_packages(),
            VISION_HINT,
            f"grounding={grounding} reads detected elements",
        )
    elif grounding == GROUNDING_DESCRIBE:
        absent, hint, why = (
            missing_jev_packages(),
            JEV_HINT,
            "grounding=describe places targets with a decision model",
        )
    else:
        return []

    if not absent:
        return []
    return [f"{why}, which needs {', '.join(absent)}.\n" + hint.replace("\n", "\n  ")]


def missing_for_models(*models: Optional[str]) -> list[str]:
    """The credentials these models need and do not have, each named once."""
    wanted: dict[str, str] = {}
    for model in models:
        for name in missing_credentials(model):
            wanted.setdefault(name, model or "")
    if not wanted:
        return []
    named = ", ".join(f"{name} (for {model})" for name, model in wanted.items())
    return [f"Missing credentials: {named}.\n  {CREDENTIAL_HINT}"]


def require_run(
    model: Optional[str],
    fast_model: Optional[str] = None,
    grounding: str = DEFAULT_GROUNDING,
) -> None:
    """Refuses to start when a run cannot have what its settings ask for.

    Every missing thing is named at once. Fixing them one error at a time
    costs a device boot for each.
    """
    missing = missing_for_models(
        model, fast_model, *models_for(grounding)
    ) + packages_for(grounding)
    if missing:
        raise UsageError("This run cannot start:\n\n" + "\n\n".join(missing))


def require_lock(model: Optional[str]) -> None:
    """Refuses to generate a suite when the generator has no credentials.

    Narrower than `require_run`: locking needs one model and no device, so a
    missing grounding package or step model is not a reason to refuse.

    Raises:
        UsageError: If the generator's model has no credentials.
    """
    missing = missing_for_models(model)
    if missing:
        raise UsageError("Cannot generate a suite:\n\n" + "\n\n".join(missing))
