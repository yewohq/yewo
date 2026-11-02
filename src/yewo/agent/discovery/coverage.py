"""Truthful discovery coverage from verified route evidence."""

from __future__ import annotations

from yewo.agent.app_memory import (
    CapabilityEvidence,
    VerifiedRoute,
    derived_memory_identifier,
)
from yewo.agent.runtime_records import AgentResult, AgentTerminalReason
from yewo.agent.discovery.records import (
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
)


DISCOVERY_CONTROL_COVERAGE = (
    "During discovery, inspect every visible interactive control and setting that can affect the "
    "requested work. Exercise each relevant control when policy permits. Observe its result "
    "before advancing or declaring completion. Do not skip a control because another action can "
    "advance. A visible default does not prove coverage. Open each relevant dropdown or spinner "
    "and observe its available choices without changing it unless the requested work requires a "
    "change. No relevant visible field or control is optional during discovery. Exercise it "
    "before leaving its workflow, or record trusted evidence that it is disabled, irrelevant to "
    "the objective, unsafe, inaccessible, or already exercised."
)


def build_objective_evidence(
    objective: DiscoveryObjective,
    agent_result: AgentResult,
    shared_route: VerifiedRoute | None,
    independently_verified: bool,
) -> tuple[DiscoveryCoverageEntry, VerifiedRoute | None, CapabilityEvidence | None]:
    """Build one coverage entry without upgrading incomplete evidence."""
    if shared_route is not None and independently_verified:
        route = _objective_route(objective, shared_route)
        capability = CapabilityEvidence(
            capability_id=derived_memory_identifier(
                "capability", objective.objective_id, route.route_id
            ),
            objective_id=objective.objective_id,
            statement=f"The application can reach: {objective.statement}",
            route_id=route.route_id,
            evidence_observation_ids=route.evidence_observation_ids,
        )
        return (
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.SUPPORTED,
                capability_id=capability.capability_id,
                route_id=route.route_id,
                evidence_observation_ids=route.evidence_observation_ids,
                reason="A contiguous accepted route reached the requested state.",
            ),
            route,
            capability,
        )
    if shared_route is not None:
        return (
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.UNRESOLVED,
                reason=(
                    "A reusable route was observed, but independent final-state verification "
                    "did not support the objective."
                ),
            ),
            _objective_route(objective, shared_route),
            None,
        )
    state = (
        DiscoveryObjectiveState.BLOCKED
        if agent_result.terminal_reason
        in {
            AgentTerminalReason.APPLICATION_EXITED,
            AgentTerminalReason.APPLICATION_INACTIVE,
            AgentTerminalReason.DEVICE_REFUSED,
        }
        else DiscoveryObjectiveState.UNRESOLVED
    )
    return (
        DiscoveryCoverageEntry(
            objective_id=objective.objective_id,
            state=state,
            reason=_coverage_reason(agent_result),
        ),
        None,
        None,
    )


def _objective_route(
    objective: DiscoveryObjective,
    shared_route: VerifiedRoute,
) -> VerifiedRoute:
    return shared_route.model_copy(
        update={
            "route_id": derived_memory_identifier(
                "route", objective.objective_id, shared_route.route_id
            ),
            "objective_id": objective.objective_id,
            "objective": objective.statement,
        }
    )


def _coverage_reason(agent_result: AgentResult) -> str:
    return {
        AgentTerminalReason.DONE_REQUESTED: (
            "The planner requested completion without a contiguous reusable route."
        ),
        AgentTerminalReason.CANNOT_PROGRESS: (
            "The agent found no further permitted useful action. The objective remains unresolved."
        ),
        AgentTerminalReason.BUDGET_EXHAUSTED: (
            "The discovery budget ended. Evidence does not support the objective."
        ),
        AgentTerminalReason.APPLICATION_EXITED: "The application exited during discovery.",
        AgentTerminalReason.APPLICATION_INACTIVE: "The application became inactive during discovery.",
        AgentTerminalReason.DEVICE_REFUSED: "The device refused a discovery action.",
    }[agent_result.terminal_reason]
