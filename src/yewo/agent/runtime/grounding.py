"""Build grounding requests for one planner proposal."""

from __future__ import annotations

from collections.abc import Callable

from yewo.agent.actuation import ACTION_DEFINITIONS, ActionProposal
from yewo.agent.planning.history import (
    PlanningHistoryEntry,
    grounding_correction_for_history,
)
from yewo.agent.runtime.records import build_runtime_grounding_request
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    screen_analysis_grounding_candidate,
)
from yewo.artifacts import RecordIdentifier
from yewo.grounding import (
    GroundingDisposition,
    GroundingEvidence,
    GroundingRequest,
    VisualGrounder,
    resolve_grounding_request,
)
from yewo.observations import Observation


def build_proposal_grounding_requests(
    identifier_factory: Callable[[], RecordIdentifier],
    proposal: ActionProposal,
    observation: Observation,
    history: list[PlanningHistoryEntry],
    screen_analysis: CurrentScreenAnalysis | None,
) -> tuple[GroundingRequest | None, GroundingRequest | None]:
    """Build target and destination requests from one resolved proposal."""
    definition = ACTION_DEFINITIONS[proposal.action]
    target_correction = grounding_correction_for_history(proposal, history)
    target = build_runtime_grounding_request(
        identifier_factory(),
        proposal.target_expression,
        definition.target_operation,
        observation,
        target_correction,
        screen_analysis_grounding_candidate(
            screen_analysis,
            observation,
            proposal.target_element_ref,
        ),
    )
    destination = build_runtime_grounding_request(
        identifier_factory(),
        proposal.destination_expression,
        definition.destination_operation,
        observation,
        None,
        screen_analysis_grounding_candidate(
            screen_analysis,
            observation,
            proposal.destination_element_ref,
        ),
    )
    return target, destination


def consumed_model_requests(*evidence: GroundingEvidence | None) -> int:
    """Count logical model requests actually used by resolved groundings."""
    return sum(item.result.model_request_count for item in evidence if item is not None)


def resolve_proposal_grounding(
    grounder: VisualGrounder,
    target_request: GroundingRequest | None,
    destination_request: GroundingRequest | None,
) -> tuple[GroundingEvidence | None, GroundingEvidence | None]:
    """Resolve both endpoints without spending another request after provider exhaustion."""
    target = resolve_grounding_request(grounder, target_request)
    if _provider_is_unavailable(target):
        return target, None
    destination = resolve_grounding_request(grounder, destination_request)
    return target, destination


def _provider_is_unavailable(evidence: GroundingEvidence | None) -> bool:
    return (
        evidence is not None
        and evidence.result.disposition is GroundingDisposition.PROVIDER_UNAVAILABLE
    )
