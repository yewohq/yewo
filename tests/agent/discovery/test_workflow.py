"""YewoAgent journey-directed discovery workflow tests."""

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import (
    AgentModePolicy,
    AgentResult,
    AgentStep,
    AgentTask,
    AgentTerminalReason,
    AgentUsage,
    AppMemory,
    AppMemoryStore,
    MemoryCompatibility,
    ReplayPathKind,
    RouteReplayResult, RouteReplayTerminal,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionPolicyLimits,
    SideEffectClass,
)
from yewo.applications import ApplicationDescriptor, ApplicationInstallMode
from yewo.devices.inventory import DevicePlatform
from yewo.agent.discovery import (
    AppMemoryBuilder,
    DiscoveryObjective,
    DiscoveryRequest,
    DiscoveryWorkflow,
    GoalVerification,
    GoalVerifier,
)
from yewo.observations import Observation
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient, ModelProfileRegistry
from yewo.runs import NeverInterrupted, RunExecutionContext, RunTerminalCategory

from agent.discovery.support import agent_result, observation
from agent.actuation.support import proposal


class StubAgent:
    def __init__(self, result) -> None:
        self._result = result
        self.task: AgentTask | None = None

    def run(self, task, mode_policy, interruption):
        del mode_policy, interruption
        self.task = task
        return self._result


class StubMemoryStore:
    def __init__(self, existing: AppMemory | None = None) -> None:
        self.existing = existing
        self.saved: AppMemory | None = None

    def load(self, compatibility):
        del compatibility
        return self.existing

    def save(self, memory, source_run_root):
        del source_run_root
        self.saved = memory
        return memory


class StubGoalVerifier:
    def __init__(self, satisfied: bool) -> None:
        self._satisfied = satisfied

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
            satisfied=self._satisfied,
            reason=(
                "The final observation supports the objective."
                if self._satisfied
                else "The final screen does not establish the requested state."
            ),
        )


def _mode_policy() -> AgentModePolicy:
    return AgentModePolicy(
        action_limits=ActionPolicyLimits(
            allowed_actions=frozenset({ActionKind.TAP, ActionKind.DISMISS_KEYBOARD}),
            supported_actions=frozenset({ActionKind.TAP, ActionKind.DISMISS_KEYBOARD}),
            allowed_side_effects=frozenset(
                {
                    SideEffectClass.UNCLASSIFIED_INTERACTION,
                    SideEffectClass.NAVIGATION,
                }
            ),
            maximum_actions=2,
        ),
        maximum_steps=3,
        maximum_grounding_requests=2,
    )


def _discovery_request() -> DiscoveryRequest:
    return DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="account_form",
                statement="The account form is reachable.",
            ),
        ),
        known_facts=("The account list is visible.",),
        priorities=("Open the account form.", "Establish that it can be reached again."),
    )


def _context(tmp_path: Path) -> RunExecutionContext:
    return cast(
        RunExecutionContext,
        SimpleNamespace(
            workspace=SimpleNamespace(
                root=tmp_path / "run",
                start=SimpleNamespace(
                    run_id="run_1",
                    started_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
                    request=SimpleNamespace(
                        application=SimpleNamespace(
                            install_mode=ApplicationInstallMode.REUSE,
                            retain_installation=False,
                        )
                    ),
                ),
            ),
            session=SimpleNamespace(
                application=ApplicationDescriptor(
                    platform=DevicePlatform.ANDROID,
                    application_id="com.example.app",
                    version="2.1.0",
                    build="42",
                    artifact_sha256="b" * 64,
                )
            ),
        ),
    )


def _workflow(
    tmp_path: Path,
    memory_store: StubMemoryStore,
    goal_verifier: GoalVerifier | None = None,
    *,
    require_reusable_evidence: bool = True,
) -> DiscoveryWorkflow:
    return DiscoveryWorkflow(
        _discovery_request(),
        _mode_policy(),
        cast(ModelProfileRegistry, object()),
        cast(GroundingProfile, object()),
        "planner-primary", "auditor-primary", "recovery-primary", "action-verifier-primary",
        tmp_path / "memory",
        lambda text: text,
        memory_store=cast(AppMemoryStore, memory_store),
        goal_verifier=goal_verifier or StubGoalVerifier(True),
        require_reusable_evidence=require_reusable_evidence,
    )


def _existing_memory(before, after) -> AppMemory:
    promotion = AppMemoryBuilder().promote(
        _discovery_request(),
        agent_result(before, after),
        {before.observation_id: before, after.observation_id: after},
        MemoryCompatibility(
            application=before.application,
            setup_contract_sha256=sha256(
                b'{"install_mode":"reuse","retain_installation":false}'
            ).hexdigest(),
        ),
        verified_objective_ids=frozenset({"account_form"}),
        source_run_id="cold_run",
        updated_at=datetime(2026, 10, 30, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    return promotion.memory


def _diverged_agent_result(current) -> AgentResult:
    terminal_proposal = proposal(
        ActionKind.CANNOT_PROGRESS,
        proposal_id="replay_terminal",
        source_observation_id=current.observation_id,
    )
    return AgentResult(
        terminal_reason=AgentTerminalReason.CANNOT_PROGRESS,
        reason="The route source guard diverged.",
        final_observation_id=current.observation_id,
        usage=AgentUsage(
            planning_requests=1,
            grounding_requests=0,
            dispatched_actions=0,
        ),
        steps=(
            AgentStep(
                step_id="replay_step",
                step_index=1,
                source_observation_id=current.observation_id,
                proposal=terminal_proposal,
                terminal_reason=AgentTerminalReason.CANNOT_PROGRESS,
            ),
        ),
    )


def test_workflow_uses_shared_agent_and_promotes_supported_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    agent = StubAgent(agent_result(before, after))
    memory_store = StubMemoryStore()
    workflow = _workflow(tmp_path, memory_store)
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )
    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())
    assert outcome.category is RunTerminalCategory.COMPLETED
    assert agent.task is not None
    assert agent.task.completion_conditions == ("The account form is reachable.",)
    assert agent.task.known_facts == _discovery_request().known_facts
    assert agent.task.open_work[:-1] == _discovery_request().priorities
    assert "visible default does not prove coverage" in agent.task.open_work[-1]
    assert memory_store.saved is not None
    assert len(memory_store.saved.routes) == 1
    assert (tmp_path / "run/assets/agent/discovery/result.json").is_file()


@pytest.mark.parametrize("require_reusable_evidence", [True, False])
def test_unsatisfied_verification_never_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    require_reusable_evidence: bool,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    agent = StubAgent(agent_result(before, after))
    memory_store = StubMemoryStore()

    workflow = _workflow(
        tmp_path,
        memory_store,
        StubGoalVerifier(False),
        require_reusable_evidence=require_reusable_evidence,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )
    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())
    assert outcome.category is RunTerminalCategory.INCOMPLETE
    assert memory_store.saved is not None
    assert len(memory_store.saved.routes) == 1
    assert memory_store.saved.capabilities == ()


def test_satisfied_verification_completes_without_reusable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    final_observation = observation("obs_1", "0000000000000000")
    agent = StubAgent(
        AgentResult(
            terminal_reason=AgentTerminalReason.DONE_REQUESTED,
            reason="The planner requested completion.",
            final_observation_id=final_observation.observation_id,
            usage=AgentUsage(
                planning_requests=1,
                grounding_requests=0,
                dispatched_actions=0,
            ),
            steps=(),
        )
    )
    workflow = _workflow(
        tmp_path,
        StubMemoryStore(),
        require_reusable_evidence=False,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (final_observation,),
    )
    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.COMPLETED


@pytest.mark.parametrize(
    ("terminal", "expected_category"),
    [
        (AgentTerminalReason.CANNOT_PROGRESS, RunTerminalCategory.INCOMPLETE),
        (AgentTerminalReason.BUDGET_EXHAUSTED, RunTerminalCategory.BUDGET_EXHAUSTED),
        (AgentTerminalReason.APPLICATION_EXITED, RunTerminalCategory.APPLICATION_FAILURE),
        (AgentTerminalReason.APPLICATION_INACTIVE, RunTerminalCategory.INCOMPLETE),
        (AgentTerminalReason.DEVICE_REFUSED, RunTerminalCategory.INCOMPLETE),
    ],
)
def test_workflow_preserves_each_non_success_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal: AgentTerminalReason,
    expected_category: RunTerminalCategory,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    agent = StubAgent(agent_result(before, after, terminal=terminal))
    workflow = _workflow(tmp_path, StubMemoryStore())
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is expected_category
    assert (tmp_path / "run/assets/agent/discovery/result.json").is_file()


def test_completed_replay_branch_skips_task_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    memory = _existing_memory(before, after)
    replay_agent_result = agent_result(before, after)
    replay = RouteReplayResult(
        path_id=memory.routes[0].route_id, path_kind=ReplayPathKind.VERIFIED_ROUTE,
        terminal=RouteReplayTerminal.COMPLETED,
        completed_segment_count=1,
        reason="The route completed.",
        agent_result=replay_agent_result,
    )
    memory_store = StubMemoryStore(memory)
    workflow = _workflow(tmp_path, memory_store)
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.replay_route",
        lambda *args, **kwargs: replay,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: pytest.fail("task planner should not run"),
    )
    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())
    assert outcome.category is RunTerminalCategory.COMPLETED
    assert memory_store.saved is None
    assert (tmp_path / "run/assets/agent/replay/evidence.json").is_file()
    assert (tmp_path / "run/assets/agent/replay/report.json").is_file()


def test_diverged_replay_runs_bounded_discovery_and_keeps_old_route(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_start = observation("old_start", "0000000000000000")
    destination = observation(
        "destination",
        "1111111111111111",
        predecessor_observation_id=old_start.observation_id,
        offset_seconds=1,
    )
    memory = _existing_memory(old_start, destination)
    changed_start = observation("changed_start", "2222222222222222")
    repaired_destination = observation(
        "repaired_destination",
        "1111111111111111",
        predecessor_observation_id=changed_start.observation_id,
        offset_seconds=1,
    )
    replay_agent_result = _diverged_agent_result(changed_start)
    replay = RouteReplayResult(
        path_id=memory.routes[0].route_id, path_kind=ReplayPathKind.VERIFIED_ROUTE,
        terminal=RouteReplayTerminal.DIVERGED,
        completed_segment_count=0,
        failed_segment_index=0,
        reason="The route source guard diverged.",
        agent_result=replay_agent_result,
    )
    repair_result = agent_result(changed_start, repaired_destination)
    repair_action = repair_result.steps[0]
    assert repair_action.outcome is not None
    repair_result = repair_result.model_copy(
        update={
            "steps": (
                repair_action.model_copy(
                    update={
                        "step_id": "repair_step",
                        "outcome": repair_action.outcome.model_copy(
                            update={"outcome_id": "repair_outcome"}
                        ),
                    }
                ),
                repair_result.steps[1].model_copy(update={"step_id": "repair_terminal"}),
            )
        }
    )
    repair_agent = StubAgent(repair_result)
    memory_store = StubMemoryStore(memory)
    workflow = _workflow(tmp_path, memory_store)
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.replay_route",
        lambda *args, **kwargs: replay,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: repair_agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (changed_start, repaired_destination),
    )

    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.COMPLETED
    assert repair_agent.task is not None
    assert "diverged at segment 0" in repair_agent.task.known_facts[-1]
    assert memory_store.saved is not None
    assert len(memory_store.saved.routes) == 2
    assert memory_store.saved.routes[0] == memory.routes[0]


def test_replay_and_repair_share_one_step_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    memory = _existing_memory(before, after)
    replay_usage = AgentUsage(
        steps=_mode_policy().maximum_steps,
        planning_requests=_mode_policy().maximum_steps,
        grounding_requests=0,
        dispatched_actions=0,
    )
    replay_result = _diverged_agent_result(before).model_copy(update={"usage": replay_usage})
    replay = RouteReplayResult(
        path_id=memory.routes[0].route_id, path_kind=ReplayPathKind.VERIFIED_ROUTE,
        terminal=RouteReplayTerminal.DIVERGED,
        completed_segment_count=0,
        failed_segment_index=0,
        reason="The route diverged after consuming the step budget.",
        agent_result=replay_result,
    )
    workflow = _workflow(tmp_path, StubMemoryStore(memory))
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.replay_route",
        lambda *args, **kwargs: replay,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: pytest.fail("repair must not receive a second budget"),
    )
    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())
    assert outcome.category is RunTerminalCategory.BUDGET_EXHAUSTED
