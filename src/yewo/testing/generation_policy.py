"""Deterministic authority and duplication policy for generated scenarios."""

from yewo.testing import steps
from yewo.testing.models import ScenarioDraft, UserJourney
from yewo.testing.taxonomy import CoverageCell, ScenarioType


def semantic_signature(
    draft: ScenarioDraft,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Identify duplicate generated flows without translating their language."""
    return (
        tuple(_normalize(value) for value in steps.rendered(draft.steps)),
        tuple(_normalize(outcome.statement) for outcome in draft.expected_outcomes),
    )


def unsupported_reason(journey: UserJourney, cell: CoverageCell) -> str | None:
    """Return the missing product authority that prevents generation."""
    if cell.scenario_type is ScenarioType.ROLE and not journey.actors:
        return "A role scenario needs another declared actor. Add one under 'actors'."
    return None


def _normalize(value: str) -> str:
    """Normalize presentation whitespace and case without translating text."""
    return " ".join(value.split()).casefold()
