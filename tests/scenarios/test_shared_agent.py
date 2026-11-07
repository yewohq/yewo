"""Tests for scenario delegation to the shared Device Agent composition."""

from typing import cast

from yewo.agent import AgentModePolicy, AgentTask
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient
from yewo.runs import NeverInterrupted, RunExecutionContext
from yewo.scenarios import SharedDeviceAgentTaskExecutor

from agent.discovery.support import agent_result
from agent.replay.support import observations


def test_executor_delegates_one_task_to_the_existing_device_agent(monkeypatch) -> None:
    before, *_middle, after = observations()
    expected = agent_result(before, after)
    task = AgentTask(objective="Inspect the state after interruption.")
    mode_policy = cast(AgentModePolicy, object())
    interruption = NeverInterrupted()
    calls = []

    class _Agent:
        def run(self, selected_task, selected_policy, selected_interruption):
            calls.append((selected_task, selected_policy, selected_interruption))
            return expected

    def build_agent(*args, **kwargs):
        calls.append((args, kwargs))
        return _Agent()

    monkeypatch.setattr("yewo.scenarios.shared_agent.build_device_agent", build_agent)
    executor = SharedDeviceAgentTaskExecutor(
        cast(RunExecutionContext, object()),
        cast(ModelClient, object()),
        cast(GroundingProfile, object()),
        "planner_profile",
        mode_policy,
        interruption,
    )

    result = executor.execute(task)

    assert result is expected
    build_args, build_kwargs = calls[0]
    assert build_args[3] == "planner_profile"
    assert build_kwargs["recover_application"] is False
    assert calls[1] == (task, mode_policy, interruption)
