"""YewoAgent journey-directed discovery workflow tests."""

from datetime import datetime, timezone
import json
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
    AppMemory,
    AppMemoryStore,
)
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.actuation import (
    ActionKind,
    ActionPolicyLimits,
    SideEffectClass,
)
from yewo.applications import (
    ApplicationDescriptor,
    ApplicationInstallationChange,
    ApplicationInstallMode,
)
from yewo.devices.inventory import DevicePlatform
from yewo.agent.discovery import (
    DiscoveryObjective,
    DiscoveryRequest,
    GoalVerification,
    GoalVerifier,
)
from yewo.agent.planning.goal_verifier import GoalVerificationMilestone
from yewo.observations import Observation
from yewo.grounding import GroundingProfile
from yewo.models import (
    ModelCallDisposition,
    ModelClient,
    ModelImageConstraints,
    ModelProfileRegistry,
    ModelRole,
)
from yewo.runs import (
    NeverInterrupted,
    RunExecutionContext,
    RunTerminalCategory,
)

from yewo.agent.discovery.compilation_milestones import DiscoveryCompilationMilestones
from yewo.agent.discovery.workflow import DiscoveryWorkflow
from yewo.agent.replay.reconciliation.records import (
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
)

from agent.discovery.support import agent_result, observation
from agent.replay.test_procedure_budget import _procedure as compiled_procedure_fixture
from models.support import build_model_profile


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
        self.milestone_evidence: tuple[GoalVerificationMilestone, ...] = ()

    def verify(
        self,
        objective_id: str,
        objective: str,
        completion_conditions: tuple[str, ...],
        observation: Observation,
        *,
        milestone_evidence: tuple[GoalVerificationMilestone, ...] = (),
    ) -> GoalVerification:
        del objective, completion_conditions
        self.milestone_evidence = milestone_evidence
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
                installation_change=ApplicationInstallationChange.NONE,
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
    procedure_store: CompiledProcedureStore | None = None,
) -> DiscoveryWorkflow:
    auditor_profile = build_model_profile(
        profile_id="auditor-primary",
        role=ModelRole.AUDITOR,
    ).model_copy(
        update={
            "images": ModelImageConstraints(
                max_images=4,
                max_image_bytes=4_000_000,
                max_width_px=2_048,
                max_height_px=4_096,
            )
        }
    )
    return DiscoveryWorkflow(
        _discovery_request(),
        _mode_policy(),
        ModelProfileRegistry(profiles=(auditor_profile,)),
        cast(GroundingProfile, object()),
        "planner-primary",
        "auditor-primary",
        "recovery-primary",
        "action-verifier-primary",
        "reconciler-primary",
        tmp_path / "memory",
        lambda text: text,
        memory_store=cast(AppMemoryStore, memory_store),
        procedure_store=(
            procedure_store
            if procedure_store is not None
            else CompiledProcedureStore(tmp_path / "memory")
        ),
        goal_verifier=goal_verifier or StubGoalVerifier(True),
        require_reusable_evidence=require_reusable_evidence,
    )


def _stub_procedure() -> ReplayProcedure:
    return compiled_procedure_fixture()


def _stub_compilation() -> DiscoveryCompilationMilestones:
    procedure = _stub_procedure()
    return DiscoveryCompilationMilestones(
        report=ProcedureCompilationReport(
            source_run_id="stub_run",
            evidence_id="stub_evidence",
            request_id="stub_request",
            status=ProcedureCompilationStatus.COMPILED,
            model_disposition=ModelCallDisposition.COMPLETED,
            procedure_id=procedure.procedure_id,
            reason="A stub compilation produced a reusable procedure.",
        ),
        procedure=procedure,
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
        "yewo.agent.discovery.workflow.compile_completed_discovery",
        lambda *args, **kwargs: _stub_compilation(),
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
    assert memory_store.saved.transitions
    result = json.loads((tmp_path / "run/assets/agent/discovery/result.json").read_bytes())
    assert result["coverage"]["entries"][0]["state"] == "supported"
    assert result["coverage"]["entries"][0]["procedure_id"] == result["compiled_procedure_id"]


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
        "yewo.agent.discovery.workflow.compile_completed_discovery",
        lambda *args, **kwargs: _stub_compilation(),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )
    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())
    assert outcome.category is RunTerminalCategory.INCOMPLETE
    assert memory_store.saved is not None
    assert memory_store.saved.transitions
    result = json.loads((tmp_path / "run/assets/agent/discovery/result.json").read_bytes())
    assert result["coverage"]["entries"][0]["state"] == "unresolved"


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
        "yewo.agent.discovery.workflow.compile_completed_discovery",
        lambda *args, **kwargs: _stub_compilation(),
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
        "yewo.agent.discovery.workflow.compile_completed_discovery",
        lambda *args, **kwargs: _stub_compilation(),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is expected_category
    assert (tmp_path / "run/assets/agent/discovery/result.json").is_file()
