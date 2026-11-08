"""Select a guarded entry into one compiled replay procedure."""

from collections.abc import Callable

from yewo.agent.replay.reconciliation.records import EvidenceDocument, ReplayProcedure
from yewo.agent.replay.state_matching import match_evidence_state
from yewo.observations import Observation, ScreenshotReference


def select_procedure_entry_index(
    observation: Observation,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
) -> int | None:
    """Return the earliest matching entry that skips no milestone evidence."""
    candidates = tuple(
        step.procedure_index
        for step in procedure.steps
        if match_evidence_state(
            observation,
            step.source_state_id,
            evidence,
            screenshot_loader,
        ).is_accepted
        and not any(
            skipped.captures_completion_conditions
            for skipped in procedure.steps[: step.procedure_index]
        )
    )
    return min(candidates) if candidates else None
