"""Verdict ownership for deterministic control-discovery failures."""

from types import SimpleNamespace

from yewo.agent.memory.history import AgentHistory
from yewo.agent.run.outcomes import DeviceRunResult


def test_unresolved_control_is_agent_failure_not_an_app_defect():
    from yewo.testing.qa_agent import QAAgent

    result = DeviceRunResult(
        status="control_unresolved",
        steps_executed=1,
        failure_reason="Bounded control discovery was exhausted.",
        history=AgentHistory(),
    )
    agent = QAAgent.__new__(QAAgent)
    status, reason, cause = agent._determine_status(
        run_result=result,
        stop_reason=None,
        all_verifications=[],
        all_issues=[],
        qa_outputs=[
            SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)
        ],
        assessments=[SimpleNamespace(rule_verifications=[], detected_issues=[])],
        required_outcomes=[],
        security_rules=[],
    )

    assert status == "UNVERIFIED"
    assert "exhausted" in reason
    assert cause.value == "agent"
