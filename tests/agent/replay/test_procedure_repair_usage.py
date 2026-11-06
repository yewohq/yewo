"""Procedure-repair model reservation and consumption accounting."""

from yewo.agent.planning.records import PlanningRequest

from agent.replay.support import observations as _observations
from agent.replay.test_procedure_repair import RepairPlanner, _repair_source, _request


class RetryingRepairPlanner(RepairPlanner):
    """Reserve a retry while consuming only the first successful attempt."""

    def required_model_requests(self, request: PlanningRequest) -> int:
        del request
        return 2

    def last_planning_model_request_count(self) -> int:
        return 1


def test_repair_evidence_counts_consumed_calls_instead_of_reserved_retries() -> None:
    source, unused_planner = _repair_source(planner=RetryingRepairPlanner())
    wrong = _observations()[-1]
    request = _request(wrong)
    source.plan(request)

    assert source.required_model_requests(request) == 2
    source.plan(request)

    assert source.last_planning_model_request_count() == 1
    assert source.repair_evidence is not None
    assert source.repair_evidence.model_call_count == 1
