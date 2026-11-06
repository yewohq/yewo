"""Match platform-owned screens against compiled replay expectations."""

from collections.abc import Callable

from yewo.agent.replay.reconciliation.records import (
    EvidenceDocument,
    ProcedureEffect,
    ReplayProcedureStep,
)
from yewo.agent.replay.state_matching import state_matches_observation
from yewo.observations import Observation, ScreenshotReference


def expected_system_surface_matches(
    observation: Observation,
    pending_step: ReplayProcedureStep | None,
    next_step: ReplayProcedureStep | None,
    evidence: EvidenceDocument,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
) -> bool:
    """Return whether the current system screen is an expected compiled state."""
    expected_state_ids: list[str] = []
    if pending_step is not None and pending_step.expects.destination_state_id is not None:
        expected_state_ids.append(pending_step.expects.destination_state_id)
    if next_step is not None and next_step.procedure_effect is ProcedureEffect.ENVIRONMENT:
        expected_state_ids.append(next_step.source_state_id)
    return any(
        state_matches_observation(
            observation,
            evidence.states[state_id],
            evidence.state_grouping_maximum_phash_distance,
            screenshot_loader,
        )
        for state_id in dict.fromkeys(expected_state_ids)
    )
