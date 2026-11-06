"""Truthful discovery coverage from compiled-procedure evidence."""

from __future__ import annotations

from yewo.agent.discovery.records import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
)
from yewo.agent.replay.procedure.records import (
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.reconciliation.records import ReplayProcedure
from yewo.agent.runtime.records import AgentResult, AgentTerminalReason


DISCOVERY_CONTROL_COVERAGE = (
    "During discovery, inspect every visible interactive control and setting that can affect the "
    "requested work. Exercise each relevant control when policy permits. Observe its result "
    "before advancing or declaring completion. Do not skip a control because another action can "
    "advance. A visible default does not prove coverage. Open each relevant dropdown or spinner "
    "and observe its available choices without changing it unless the requested work requires a "
    "change. Visibility alone does not make a field or control mandatory. Exercise controls "
    "before leaving its workflow, or record trusted evidence that it is disabled, irrelevant to "
    "the objective, unsafe, inaccessible, or already exercised."
)


def build_objective_evidence(
    objective: DiscoveryObjective,
    agent_result: AgentResult,
    procedure: ReplayProcedure | None,
    evidence_observation_ids: tuple[str, ...],
    independently_verified: bool,
) -> DiscoveryCoverageEntry:
    """Build one coverage entry without upgrading incomplete evidence."""
    if procedure is not None and independently_verified and evidence_observation_ids:
        return DiscoveryCoverageEntry(
            objective_id=objective.objective_id,
            state=DiscoveryObjectiveState.SUPPORTED,
            procedure_id=procedure.procedure_id,
            evidence_observation_ids=evidence_observation_ids,
            reason="A compiled replay procedure reached the audited requested state.",
        )
    if procedure is not None:
        return DiscoveryCoverageEntry(
            objective_id=objective.objective_id,
            state=DiscoveryObjectiveState.UNRESOLVED,
            reason=(
                "A reusable procedure was compiled, but independent final-state verification "
                "did not support the objective."
            ),
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
    return DiscoveryCoverageEntry(
        objective_id=objective.objective_id,
        state=state,
        reason=_coverage_reason(agent_result),
    )


def _coverage_reason(agent_result: AgentResult) -> str:
    return {
        AgentTerminalReason.DONE_REQUESTED: (
            "The planner requested completion without a reusable compiled procedure."
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


def budget_coverage(request: DiscoveryRequest) -> DiscoveryCoverage:
    """Report unresolved coverage after the shared run budget is consumed."""
    objective = request.objectives[0]
    return DiscoveryCoverage(
        entries=(
            DiscoveryCoverageEntry(
                objective_id=objective.objective_id,
                state=DiscoveryObjectiveState.UNRESOLVED,
                reason="The run exhausted the shared replay-and-repair discovery budget.",
            ),
        )
    )


def procedure_replay_coverage(
    request: DiscoveryRequest,
    replay: ProcedureReplayResult,
) -> DiscoveryCoverage:
    """Report task proof separately from reusable discovery coverage."""
    state = (
        DiscoveryObjectiveState.BLOCKED
        if replay.terminal
        in {
            ProcedureReplayTerminal.APPLICATION_FAILURE,
            ProcedureReplayTerminal.DEVICE_REFUSED,
        }
        else DiscoveryObjectiveState.UNRESOLVED
    )
    return DiscoveryCoverage(
        entries=(
            DiscoveryCoverageEntry(
                objective_id=request.objectives[0].objective_id,
                state=state,
                reason=(
                    "Compiled procedure replay completed the task, but it did not compile "
                    "a new reusable procedure."
                    if replay.terminal is ProcedureReplayTerminal.COMPLETED
                    else f"Compiled procedure replay stopped: {replay.reason}"
                ),
            ),
        )
    )
