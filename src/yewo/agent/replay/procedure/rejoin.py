"""Forward-only state selection for compiled-procedure repair."""

from collections.abc import Callable

from yewo.agent.replay.procedure.records import ProcedureRepairTarget
from yewo.agent.replay.reconciliation.records import EvidenceDocument, ReplayProcedure
from yewo.agent.replay.state_matching import (
    AMBIGUOUS_SOURCE_STATE,
    match_evidence_state,
    repair_state_matches_observation,
    trace_repair_target_match,
)
from yewo.observations import Observation, ScreenshotReference


def forward_rejoin_index(
    observation: Observation,
    target: ProcedureRepairTarget | None,
    procedure: ReplayProcedure,
    evidence: EvidenceDocument,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
    target_matches: Callable[[Observation], bool],
) -> int | None:
    """Return the furthest safely proven procedure frontier index."""
    if target is None:
        return None
    if target_matches(observation):
        return target.resume_procedure_index
    candidates: dict[str, int] = {}
    for step in procedure.steps[target.resume_procedure_index :]:
        state_id = step.expects.destination_state_id
        if state_id is None or state_id == target.expected_state_id:
            continue
        candidates.setdefault(state_id, step.procedure_index + 1)
    matched_indexes = tuple(
        resume_index
        for state_id, resume_index in candidates.items()
        if match_evidence_state(
            observation,
            state_id,
            evidence,
            screenshot_loader,
        ).is_accepted
    )
    if not matched_indexes:
        return None
    resume_index = max(matched_indexes)
    trace_repair_target_match(True, f"forward_state_{resume_index}_matched")
    return resume_index


def repair_target_matches(
    observation: Observation,
    target: ProcedureRepairTarget | None,
    evidence: EvidenceDocument,
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
    requires_exact_content_match: bool,
) -> bool:
    """Match the immediate target with its compiler-selected strictness."""
    if target is None:
        return False
    source_match = match_evidence_state(
        observation,
        target.expected_state_id,
        evidence,
        screenshot_loader,
    )
    if not requires_exact_content_match or (
        not source_match.is_accepted and source_match.reason != AMBIGUOUS_SOURCE_STATE
    ):
        return trace_repair_target_match(source_match.is_accepted, source_match.reason)
    is_accepted = repair_state_matches_observation(
        observation,
        evidence.states[target.expected_state_id],
        evidence.state_grouping_maximum_phash_distance,
        screenshot_loader,
    )
    return trace_repair_target_match(is_accepted, "exact_content_match")
