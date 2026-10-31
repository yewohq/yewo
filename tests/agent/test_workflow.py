"""YewoAgent terminal classification tests."""

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
    AppMemoryStore,
    GoalVerification,
)
from yewo.agent.actuation import ActionKind, ActionPolicyLimits, SideEffectClass
from yewo.agent.discovery import DiscoveryRequest
from yewo.agent.workflow import YewoAgent
from yewo.grounding import (
    GroundingCoordinateConvention,
    GroundingOutputShape,
    GroundingProfile,
    GroundingTargetOperation,
)
from yewo.models import ModelClient, ModelProfileRegistry
from yewo.observations import Observation
from yewo.run_narration import RunNarrator, narration_scope
from yewo.runs import (
    InterruptionSignal,
    NeverInterrupted,
    RunExecutionContext,
    RunTerminalCategory,
)


class StubAgent:
    def __init__(
        self,
        terminal: AgentTerminalReason,
        *,
        planning_requests: int = 0,
    ) -> None:
        self._terminal = terminal
        self._planning_requests = planning_requests

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
                planning_requests=self._planning_requests,
                grounding_requests=0,
                dispatched_actions=0,
            ),
            steps=(),
        )


class StubMemoryStore:
    def __init__(self, memory: object) -> None:
        self.memory = memory
        self.compatibilities: list[object] = []

    def load(self, compatibility: object) -> object:
        self.compatibilities.append(compatibility)
        return self.memory


def _workflow(
    *,
    memory_store: object | None = None,
    task: AgentTask | None = None,
    declared_value_resolver=None,
    maximum_model_requests: int | None = None,
) -> YewoAgent:
    limits = ActionPolicyLimits(
        allowed_actions=frozenset({ActionKind.WAIT}),
        supported_actions=frozenset({ActionKind.WAIT}),
        allowed_side_effects=frozenset({SideEffectClass.NONE}),
        maximum_actions=1,
    )
    return YewoAgent(
        task
        or AgentTask(
            objective="Wait for the application.",
            completion_conditions=("The application is ready.",),
        ),
        AgentModePolicy(
            action_limits=limits,
            maximum_steps=1,
            maximum_grounding_requests=0,
            maximum_model_requests=maximum_model_requests,
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
        "auditor-primary",
        lambda text: text,
        memory_store=cast(AppMemoryStore, memory_store) if memory_store else None,
        declared_value_resolver=declared_value_resolver,
    )


def _context(root: Path = Path("run-root")) -> RunExecutionContext:
    return cast(
        RunExecutionContext,
        SimpleNamespace(
            session=SimpleNamespace(application=object()),
            workspace=SimpleNamespace(
                root=root,
                start=SimpleNamespace(
                    run_id="run-1",
                    request=SimpleNamespace(application=object()),
                ),
            ),
        ),
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
        "yewo.agent.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )

    outcome = _workflow().execute(_context(), NeverInterrupted())

    assert outcome.category is expected


@pytest.mark.parametrize(
    ("satisfied", "expected"),
    [
        (True, RunTerminalCategory.COMPLETED),
        (False, RunTerminalCategory.INCOMPLETE),
    ],
)
def test_direct_done_requires_independent_goal_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    satisfied: bool,
    expected: RunTerminalCategory,
) -> None:
    agent = StubAgent(AgentTerminalReason.DONE_REQUESTED)

    class StubGoalVerifier:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def verify(
            self,
            objective_id: str,
            objective: str,
            completion_conditions: tuple[str, ...],
            observation: Observation,
        ) -> GoalVerification:
            del objective, completion_conditions
            return GoalVerification(
                verification_id="verification_1",
                objective_id=objective_id,
                final_observation_id=observation.observation_id,
                satisfied=satisfied,
                reason="The configured audit conclusion.",
            )

    monkeypatch.setattr(
        "yewo.agent.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr("yewo.agent.workflow.ModelGoalVerifier", StubGoalVerifier)
    monkeypatch.setattr(
        "yewo.agent.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (SimpleNamespace(observation_id="observation-1"),),
    )

    with narration_scope(RunNarrator(tmp_path)):
        outcome = _workflow().execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is expected
    assert (tmp_path / "assets/agent/goal-verification.json").is_file()
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    verdict = "satisfied" if satisfied else "not satisfied"
    assert f"Independent goal verification {verdict}" in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "verification=verification_1 final_observation=observation-1" in trace


def test_direct_done_fails_closed_when_auditor_budget_is_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = StubAgent(AgentTerminalReason.DONE_REQUESTED, planning_requests=1)

    monkeypatch.setattr(
        "yewo.agent.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.workflow.ModelGoalVerifier",
        lambda *args, **kwargs: pytest.fail("auditor budget is exhausted"),
    )
    monkeypatch.setattr(
        "yewo.agent.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (SimpleNamespace(observation_id="observation-1"),),
    )

    outcome = _workflow(maximum_model_requests=1).execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.INCOMPLETE
    assert outcome.reason == "The model-request budget ended before final verification."
    assert (tmp_path / "assets/agent/goal-verification.json").is_file()


def test_yewo_agent_routes_memory_backed_tasks_through_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StubMemoryStore(None)
    captured: dict[str, object] = {}

    class StubDiscoveryWorkflow:
        def __init__(self, request, *args, **kwargs) -> None:
            captured["request"] = request
            captured["memory_store"] = kwargs["memory_store"]

        def execute(self, context, interruption):
            captured["context"] = context
            captured["interruption"] = interruption
            return SimpleNamespace(
                category=RunTerminalCategory.COMPLETED,
                reason="Discovery completed the task and updated memory.",
            )

    monkeypatch.setattr(
        "yewo.agent.discovery.DiscoveryWorkflow",
        StubDiscoveryWorkflow,
    )

    workflow = _workflow(memory_store=store)
    outcome = workflow.execute(_context(), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.COMPLETED
    assert outcome.reason == "Discovery completed the task and updated memory."
    request = cast(DiscoveryRequest, captured["request"])
    assert request.objectives[0].statement == "Wait for the application."
    assert request.objectives[0].completion_conditions == ("The application is ready.",)
    assert captured["memory_store"] is store


def test_declared_value_task_enters_discovery_by_name_without_exposing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = StubMemoryStore(None)
    captured: dict[str, object] = {}

    class StubDiscoveryWorkflow:
        def __init__(self, request, *args, **kwargs) -> None:
            captured["request"] = request
            captured["resolver"] = kwargs["declared_value_resolver"]

        def execute(self, context, interruption):
            del context, interruption
            return SimpleNamespace(
                category=RunTerminalCategory.COMPLETED,
                reason="The declared-value task completed.",
            )

    monkeypatch.setattr("yewo.agent.discovery.DiscoveryWorkflow", StubDiscoveryWorkflow)
    task = AgentTask(
        objective="Enter the disposable account value.",
        completion_conditions=("The value is accepted.",),
        declared_value_names=("account_value",),
    )

    workflow = _workflow(
        memory_store=store,
        task=task,
        declared_value_resolver=lambda unused: "actual-secret-value",
    )
    outcome = workflow.execute(_context(), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.COMPLETED
    request = cast(DiscoveryRequest, captured["request"])
    assert request.declared_value_names == ("account_value",)
    assert "actual-secret-value" not in request.model_dump_json()
    resolver = cast(object, captured["resolver"])
    assert callable(resolver)
