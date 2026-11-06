"""Compose discovery task briefs from retained evidence."""

from yewo.agent.memory.records import AppMemory
from yewo.agent.discovery.coverage import (
    DISCOVERY_CONTROL_COVERAGE,
)
from yewo.agent.discovery.records import DiscoveryRequest
from yewo.agent.runtime.records import AgentTask


def build_discovery_agent_task(
    request: DiscoveryRequest,
    memory: AppMemory | None,
) -> AgentTask:
    """Build one planner brief without treating retained memory as current state."""
    objective_statements = tuple(objective.statement for objective in request.objectives)
    requested_work = request.priorities or objective_statements
    return AgentTask(
        objective="Discover evidence for: " + objective_statements[0],
        known_facts=request.known_facts,
        open_work=(*requested_work, DISCOVERY_CONTROL_COVERAGE),
        completion_conditions=request.objectives[0].completion_conditions,
        declared_value_names=request.declared_value_names,
        targeted_type_available=(
            any(transition.action.input_contract is not None for transition in memory.transitions)
            if memory is not None
            else False
        ),
        discovery_mode=True,
    )
