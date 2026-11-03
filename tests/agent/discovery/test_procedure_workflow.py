"""Compiled-procedure selection and discovery handoff tests."""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import AgentTerminalReason, MemoryCompatibility
from yewo.agent.discovery.procedure_handoff import (
    procedure_terminal_discovery_result,
    replay_compiled_procedure_if_available,
)
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.replay import ReplayProcedure, compile_replay_procedure
from yewo.agent.replay.procedure_records import (
    MilestoneEvidence,
    ProcedureDivergence,
    ProcedureRepairEvidence,
    ProcedureRepairTarget,
    ProcedureReplayResult,
    ProcedureReplayTerminal,
)
from yewo.agent.replay.procedure_store import CompiledProcedureStore
from yewo.applications import ApplicationInstallationChange, ApplicationInstallMode
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient
from yewo.observations import Observation
from yewo.runs import NeverInterrupted, RunExecutionContext, RunTerminalCategory
from yewo.runs import RunPreparationContext

from agent.discovery.support import agent_result, observation
from agent.discovery.test_workflow import (
    StubAgent,
    StubGoalVerifier,
    StubMemoryStore,
    _context,
    _discovery_request,
    _mode_policy,
    _workflow,
)
from agent.replay.test_hazard_gating import memory_with_hazard
from agent.replay.test_procedure_compiler import _evidence, _response


def _procedure_replay(
    before: Observation,
    after: Observation,
    *,
    satisfied: bool,
) -> ProcedureReplayResult:
    result = agent_result(before, after)
    verification = GoalVerification(
        verification_id="procedure_verification",
        objective_id="account_form",
        final_observation_id=after.observation_id,
        satisfied=satisfied,
        reason="The configured procedure audit conclusion.",
    )
    return ProcedureReplayResult(
        procedure_id="procedure_1",
        evidence_id="evidence_1",
        source_run_id="source_run",
        setup_id="setup_1",
        application=before.application,
        terminal=(
            ProcedureReplayTerminal.COMPLETED
            if satisfied
            else ProcedureReplayTerminal.REPLAY_COMPLETE
        ),
        procedure_step_count=1,
        completed_step_count=1,
        milestones=(
            MilestoneEvidence(
                completion_condition_index=0,
                procedure_index=0,
                source_step=1,
                observation_id=after.observation_id,
                screenshot_artifact_id=after.redacted_screenshot.artifact_id,
                screen_fingerprint=after.screen_fingerprint,
            ),
        ),
        verification=verification,
        reason=verification.reason,
        agent_result=result,
    )


def test_compiled_procedure_never_runs_without_a_fresh_install() -> None:
    class UnexpectedStore:
        def load(self, *args, **kwargs):
            raise AssertionError("non-fresh sessions must not load a procedure")

    context = cast(
        RunExecutionContext,
        SimpleNamespace(
            session=SimpleNamespace(installation_change=ApplicationInstallationChange.NONE)
        ),
    )

    replay = replay_compiled_procedure_if_available(
        context,
        cast(MemoryCompatibility, object()),
        "objective_1",
        "Complete the task.",
        _mode_policy(),
        cast(ModelClient, object()),
        cast(GroundingProfile, object()),
        "auditor-primary",
        NeverInterrupted(),
        cast(CompiledProcedureStore, UnexpectedStore()),
        memory=None,
        redaction_regions=(),
        declared_value_resolver=None,
    )

    assert replay is None


def test_compiled_procedure_does_not_run_after_memory_learns_a_hazard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(
        evidence,
        _response(),
        identifier_factory=lambda: "procedure_1",
    )
    assert isinstance(procedure, ReplayProcedure)
    memory = memory_with_hazard(
        tmp_path,
        target=evidence.steps[0].proposal.target_expression,
    )

    class StoredProcedure:
        def load(self, compatibility, objective_id, objective):
            del compatibility, objective_id, objective
            return SimpleNamespace(evidence=evidence, procedure=procedure)

    monkeypatch.setattr(
        "yewo.agent.discovery.procedure_handoff.execute_compiled_procedure",
        lambda *args, **kwargs: pytest.fail("known-hazard procedure must not execute"),
    )
    context = cast(
        RunExecutionContext,
        SimpleNamespace(
            session=SimpleNamespace(
                installation_change=ApplicationInstallationChange.REINSTALLED
            )
        ),
    )

    replay = replay_compiled_procedure_if_available(
        context,
        memory.compatibility,
        "objective_1",
        "Complete the task.",
        _mode_policy(),
        cast(ModelClient, object()),
        cast(GroundingProfile, object()),
        "auditor-primary",
        NeverInterrupted(),
        cast(CompiledProcedureStore, StoredProcedure()),
        memory=memory,
        redaction_regions=(),
        declared_value_resolver=None,
    )

    assert replay is None


def test_preparation_selects_an_available_install_procedure_without_falling_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = _discovery_request().objectives[0]
    evidence = _evidence().model_copy(
        update={
            "task": _evidence().task.model_copy(
                update={
                    "objective_id": requested.objective_id,
                    "objective": requested.statement,
                    "completion_conditions": (requested.statement,),
                }
            )
        }
    )
    procedure = compile_replay_procedure(
        evidence,
        _response(),
        identifier_factory=lambda: "procedure_1",
    )
    assert isinstance(procedure, ReplayProcedure)

    class AvailableProcedureStore:
        def load(self, compatibility, objective_id, objective):
            del compatibility, objective_id, objective
            return SimpleNamespace(evidence=evidence, procedure=procedure)

    workflow = _workflow(
        tmp_path,
        StubMemoryStore(),
        procedure_store=cast(CompiledProcedureStore, AvailableProcedureStore()),
    )
    execution_context = _context(tmp_path)
    preparation_context = cast(
        RunPreparationContext,
        SimpleNamespace(
            application=execution_context.session.application,
            workspace=SimpleNamespace(
                start=SimpleNamespace(
                    request=SimpleNamespace(
                        application=SimpleNamespace(
                            install_mode=ApplicationInstallMode.INSTALL,
                            retain_installation=False,
                        )
                    )
                )
            ),
        ),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.preparation.prepare_memory_identity",
        lambda *args, **kwargs: pytest.fail("procedure selection should not fall through"),
    )

    preparation = workflow.prepare(preparation_context, NeverInterrupted())

    assert preparation.procedure_selection is not None
    assert preparation.procedure_selection.procedure_id == "procedure_1"


def test_satisfied_compiled_procedure_finishes_before_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = observation("procedure_before", "0000000000000000")
    after = observation(
        "procedure_after",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    replay = _procedure_replay(before, after, satisfied=True)
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
        "yewo.agent.discovery.procedure_handoff.replay_compiled_procedure_if_available",
        lambda *args, **kwargs: replay,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: pytest.fail("discovery should not run"),
    )

    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.COMPLETED
    result = json.loads((tmp_path / "run/assets/agent/discovery/result.json").read_bytes())
    assert result["procedure_replay"]["terminal"] == "completed"


def test_replay_complete_continues_with_remaining_discovery_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = observation("procedure_before", "0000000000000000")
    after = observation(
        "procedure_after",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    replay = _procedure_replay(before, after, satisfied=False)
    discovery_agent = StubAgent(agent_result(before, after))
    verifier = StubGoalVerifier(True)
    workflow = _workflow(
        tmp_path,
        StubMemoryStore(),
        require_reusable_evidence=False,
        goal_verifier=verifier,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_model_client",
        lambda *args, **kwargs: cast(ModelClient, object()),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.procedure_handoff.replay_compiled_procedure_if_available",
        lambda *args, **kwargs: replay,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.build_device_agent",
        lambda *args, **kwargs: discovery_agent,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.workflow.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    outcome = workflow.execute(_context(tmp_path), NeverInterrupted())

    assert outcome.category is RunTerminalCategory.COMPLETED
    assert discovery_agent.task is not None
    assert tuple(item.completion_condition_index for item in verifier.milestone_evidence) == (0,)
    result = json.loads((tmp_path / "run/assets/agent/discovery/result.json").read_bytes())
    assert result["procedure_replay"]["terminal"] == "replay_complete"


def test_repaired_completed_replay_stops_before_general_discovery() -> None:
    before = observation("procedure_before", "0000000000000000")
    after = observation("procedure_after", "1111111111111111")
    completed_agent = agent_result(before, after).model_copy(
        update={"terminal_reason": AgentTerminalReason.DONE_REQUESTED}
    )
    verification = GoalVerification(
        verification_id="repaired_verification",
        objective_id=_discovery_request().objectives[0].objective_id,
        final_observation_id=after.observation_id,
        satisfied=True,
        reason="The repaired replay satisfied the objective.",
    )
    divergence = ProcedureDivergence(
        procedure_id="procedure_1",
        procedure_index=0,
        source_step=1,
        guard="source",
        expected_state_id="state_1",
        actual_observation_id=before.observation_id,
        actual_fingerprint=before.screen_fingerprint,
        reason="The procedure source guard failed.",
    )
    replay = ProcedureReplayResult(
        procedure_id="procedure_1",
        evidence_id="evidence_1",
        source_run_id="source_run",
        setup_id="setup_1",
        application=before.application,
        terminal=ProcedureReplayTerminal.COMPLETED,
        procedure_step_count=1,
        completed_step_count=1,
        failed_procedure_index=0,
        divergence=divergence,
        repair=ProcedureRepairEvidence(
            target=ProcedureRepairTarget(
                expected_state_id="state_1",
                resume_procedure_index=0,
                reason=divergence.reason,
            ),
            maximum_actions=8,
            action_count=2,
            model_call_count=1,
            succeeded=True,
            rejoined_observation_id=after.observation_id,
            reason="The compiled rejoin guard accepted the live observation.",
        ),
        verification=verification,
        reason=verification.reason,
        agent_result=completed_agent,
    )

    conclusion = procedure_terminal_discovery_result(_discovery_request(), replay, memory_id=None)

    assert conclusion is not None
    assert conclusion.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert conclusion.procedure_replay == replay
    assert conclusion.verification == verification
