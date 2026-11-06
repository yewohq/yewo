"""Discovery planner-brief scope tests."""

from yewo.agent.discovery.records import DiscoveryObjective, DiscoveryRequest
from yewo.agent.discovery.task import build_discovery_agent_task


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="objective_1",
                statement="Create and save one record.",
                completion_conditions=("The saved record is visible.",),
            ),
        ),
    )


def test_discovery_always_requires_relevant_visible_control_coverage() -> None:
    task = build_discovery_agent_task(_request(), None)

    assert "Exercise each relevant control when policy permits" in task.open_work[-1]
    assert "Visibility alone does not make a field or control mandatory" in task.open_work[-1]
