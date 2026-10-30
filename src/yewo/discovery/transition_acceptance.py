"""Validate one executed discovery transition into stored evidence."""

from yewo.agent.actuation.actions import ActionType
from yewo.agent.actuation.safety import ActionAuthorization
from yewo.agent.memory.screen_identity import is_same_screen
from yewo.agent.run.settlement import SettlementStatus
from yewo.discovery.contracts import DiscoveryTransitionEvidence
from yewo.discovery.runtime import (
    DiscoveryActionCandidate,
    DiscoveryExecutionReceipt,
    transition_identity_for,
)


def build_accepted_transition(
    candidate: DiscoveryActionCandidate,
    authorization: ActionAuthorization,
    execution: DiscoveryExecutionReceipt,
) -> DiscoveryTransitionEvidence | None:
    dispatch = execution.dispatch
    settlement = execution.settlement
    destination = execution.destination.observation
    if dispatch.authorization_id != authorization.authorization_id or not dispatch.success:
        return None
    if settlement.status is not SettlementStatus.DURABLE_PROGRESS:
        return None
    if not settlement.control_identity.matches(authorization.control_identity):
        return None
    if not {
        candidate.source_observation_id,
        destination.observation_id,
    }.issubset(set(settlement.observation_ids)):
        return None
    if not is_same_screen(settlement.destination_phash, destination.phash):
        return None
    transition_identity = transition_identity_for(
        candidate.source_observation_id,
        candidate.control_identity,
        destination.observation_id,
    )
    return DiscoveryTransitionEvidence(
        transition_identity=transition_identity,
        source_observation_id=candidate.source_observation_id,
        destination_observation_id=destination.observation_id,
        control_identity=candidate.control_identity,
        action_kind=(
            candidate.action.type.value
            if isinstance(candidate.action.type, ActionType)
            else candidate.action.type
        ),
        authorization_receipt_id=authorization.authorization_id,
        dispatch_receipt_id=dispatch.dispatch_id,
        settlement_receipt_id=settlement.settlement_id,
        side_effect=candidate.side_effect,
        reversible=candidate.reversible,
    )
