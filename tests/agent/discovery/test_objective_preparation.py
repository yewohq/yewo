"""Semantic objective selection during replay preparation."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from yewo.agent import MemoryCompatibility
from yewo.agent.discovery.preparation import DiscoveryPreparation
from yewo.agent.discovery.procedure_handoff import replay_compiled_procedure_if_available
from yewo.agent.replay import ReplayProcedure, compile_replay_procedure
from yewo.agent.replay.procedure.objective_matching import (
    ProcedureSelectionMethod,
    ReplayProcedureSelection,
)
from yewo.agent.replay.procedure.records import ProcedureReplayResult
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.reconciliation.records import EvidenceTask
from yewo.applications import ApplicationInstallationChange, ApplicationInstallMode
from yewo.grounding import GroundingProfile
from yewo.models import ModelClient
from yewo.runs import NeverInterrupted, RunExecutionContext, RunPreparationContext

from agent.discovery.test_workflow import (
    StubMemoryStore,
    _context,
    _discovery_request,
    _mode_policy,
    _workflow,
)
from agent.planning.support import StubPlannerModel
from agent.replay.support import evidence as _evidence, response as _response


def test_semantic_selection_loads_stored_identity_and_audits_requested_objective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence()
    procedure = compile_replay_procedure(
        evidence,
        _response(completion_conditions=evidence.task.completion_conditions),
        identifier_factory=lambda: "procedure_1",
    )
    assert isinstance(procedure, ReplayProcedure)
    requested = _discovery_request().objectives[0]
    replay = cast(ProcedureReplayResult, object())
    loaded: list[tuple[str, str]] = []
    captured: dict[str, object] = {}

    class StoredProcedure:
        def load(self, compatibility, objective_id, objective):
            del compatibility
            loaded.append((objective_id, objective))
            return SimpleNamespace(evidence=evidence, procedure=procedure)

    def execute(*args, **kwargs):
        captured.update(kwargs)
        return replay

    monkeypatch.setattr(
        "yewo.agent.discovery.procedure_handoff.execute_compiled_procedure",
        execute,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.procedure_handoff._policy_can_execute",
        lambda *args: True,
    )
    context = cast(
        RunExecutionContext,
        SimpleNamespace(
            session=SimpleNamespace(
                installation_change=ApplicationInstallationChange.REINSTALLED
            )
        ),
    )
    preparation = DiscoveryPreparation(
        memory_id=None,
        progress_context=None,
        procedure_selection=ReplayProcedureSelection(
            procedure_id="procedure_1",
            stored_objective_id=evidence.task.objective_id,
            stored_objective=evidence.task.objective,
            method=ProcedureSelectionMethod.LLM_EQUIVALENT,
            preparation_model_requests=1,
        ),
    )

    result = replay_compiled_procedure_if_available(
        context,
        MemoryCompatibility(application=evidence.run.application),
        requested.objective_id,
        requested.statement,
        _mode_policy(),
        cast(ModelClient, object()),
        cast(GroundingProfile, object()),
        "auditor-primary",
        NeverInterrupted(),
        cast(CompiledProcedureStore, StoredProcedure()),
        planner_profile_id="planner-primary",
        memory=None,
        redaction_regions=(),
        declared_value_resolver=None,
        preparation=preparation,
        verification_task=evidence.task.model_copy(
            update={
                "objective_id": requested.objective_id,
                "objective": requested.statement,
            }
        ),
    )

    assert result is replay
    assert loaded == [(evidence.task.objective_id, evidence.task.objective)]
    assert captured["initial_model_requests"] == 1
    verification_task = cast(EvidenceTask, captured["verification_task"])
    assert verification_task.objective_id == requested.objective_id
    assert verification_task.objective == requested.statement


def test_preparation_uses_llm_equivalent_procedure_after_exact_lookup_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = _evidence().model_copy(
        update={
            "task": _evidence().task.model_copy(
                update={
                    "objective_id": "stored_objective",
                    "objective": "Open the form used to add a record.",
                    "completion_conditions": (
                        "The form used to add a record is visible.",
                    ),
                }
            )
        }
    )
    procedure = compile_replay_procedure(
        evidence,
        _response(completion_conditions=evidence.task.completion_conditions),
        identifier_factory=lambda: "procedure_1",
    )
    assert isinstance(procedure, ReplayProcedure)
    bundle = SimpleNamespace(evidence=evidence, procedure=procedure)

    class CandidateProcedureStore:
        def load(self, compatibility, objective_id, objective):
            del compatibility, objective
            return bundle if objective_id == "stored_objective" else None

        def compatible_candidates(self, compatibility):
            del compatibility
            return (bundle,)

    class EquivalentObjectiveModel(StubPlannerModel):
        def complete(self, request, response_model=None, validation_context=None):
            self.output = {
                "procedure_id": "procedure_1",
                "relationship": "equivalent",
                "reason": "Both objectives require the record form to be visible.",
            }
            return super().complete(request, response_model, validation_context)

    model = EquivalentObjectiveModel(None)
    workflow = _workflow(
        tmp_path,
        StubMemoryStore(),
        procedure_store=cast(CompiledProcedureStore, CandidateProcedureStore()),
    )
    execution_context = _context(tmp_path)
    preparation_context = cast(
        RunPreparationContext,
        SimpleNamespace(
            application=execution_context.session.application,
            workspace=SimpleNamespace(
                root=tmp_path / "run",
                start=SimpleNamespace(
                    run_id="run_1",
                    request=SimpleNamespace(
                        application=SimpleNamespace(
                            install_mode=ApplicationInstallMode.INSTALL,
                            retain_installation=False,
                        )
                    ),
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.preparation.build_model_client",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "yewo.agent.discovery.preparation.prepare_memory_identity",
        lambda *args, **kwargs: pytest.fail("route review should not run after a match"),
    )

    preparation = workflow.prepare(preparation_context, NeverInterrupted())

    assert preparation.procedure_selection is not None
    assert preparation.procedure_selection.procedure_id == "procedure_1"
    assert preparation.procedure_selection.stored_objective_id == "stored_objective"
    assert preparation.procedure_selection.preparation_model_requests == 1
    assert len(model.requests) == 1
