"""Normal-workflow terminal classification tests."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import (
    AgentModePolicy,
    AgentResult,
    AgentTask,
    AgentTerminalReason,
    AgentUsage,
)
from yewo.agent.actuation import ActionKind, ActionPolicyLimits, SideEffectClass
from yewo.agent.workflow import NormalAgentWorkflow
from yewo.grounding import (
    GroundingCoordinateConvention,
    GroundingOutputShape,
    GroundingProfile,
    GroundingTargetOperation,
)
from yewo.models import ModelClient, ModelProfileRegistry
from yewo.runs import (
    InterruptionSignal,
    NeverInterrupted,
    RunExecutionContext,
    RunTerminalCategory,
)


class StubAgent:
    def __init__(self, terminal: AgentTerminalReason) -> None:
        self._terminal = terminal

    def run(
        self,
        task: AgentTask,
        mode_policy: AgentModePolicy,
        interruption: InterruptionSignal,
    ) -> AgentResult:
        del task, mode_policy, interruption
        return AgentResult(
            terminal_reason=self._terminal,
            reason="The configured terminal occurred.",
            final_observation_id="observation-1",
            usage=AgentUsage(
                planning_requests=0,
                grounding_requests=0,
                dispatched_actions=0,
            ),
            steps=(),
        )


def _workflow() -> NormalAgentWorkflow:
    limits = ActionPolicyLimits(
        allowed_actions=frozenset({ActionKind.WAIT}),
        supported_actions=frozenset({ActionKind.WAIT}),
        allowed_side_effects=frozenset({SideEffectClass.NONE}),
        maximum_actions=1,
    )
    return NormalAgentWorkflow(
        AgentTask(
            objective="Wait for the application.",
            completion_conditions=("The application is ready.",),
        ),
        AgentModePolicy(
            action_limits=limits,
            maximum_steps=1,
            maximum_grounding_requests=0,
        ),
        cast(ModelProfileRegistry, object()),
        GroundingProfile(
            profile_id="grounding-primary",
            model_profile_id="grounder-primary",
            coordinate_convention=GroundingCoordinateConvention.SOURCE_PIXELS,
            supported_operations=frozenset({GroundingTargetOperation.TAP}),
            supported_shapes=frozenset({GroundingOutputShape.POINT}),
        ),
        "planner-primary",
        lambda text: text,
    )


@pytest.mark.parametrize(
    ("terminal", "expected"),
    [
        (AgentTerminalReason.APPLICATION_INACTIVE, RunTerminalCategory.INCOMPLETE),
        (AgentTerminalReason.DEVICE_REFUSED, RunTerminalCategory.INCOMPLETE),
        (
            AgentTerminalReason.APPLICATION_EXITED,
            RunTerminalCategory.APPLICATION_FAILURE,
        ),
    ],
)
def test_execute_classifies_agent_terminals(
    monkeypatch: pytest.MonkeyPatch,
    terminal: AgentTerminalReason,
    expected: RunTerminalCategory,
) -> None:
    agent = StubAgent(terminal)
    monkeypatch.setattr(
        "yewo.agent.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.workflow._build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )

    outcome = _workflow().execute(
        cast(
            RunExecutionContext,
            SimpleNamespace(
                workspace=SimpleNamespace(
                    root=Path("run-root"),
                    start=SimpleNamespace(run_id="run-1"),
                )
            ),
        ),
        NeverInterrupted(),
    )

    assert outcome.category is expected
