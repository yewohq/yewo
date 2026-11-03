"""Post-discovery replay-procedure compilation integration tests."""

from decimal import Decimal
import json
from pathlib import Path
from collections.abc import Callable

from pydantic import BaseModel
import pytest

from yewo.agent import AgentTerminalReason, ScreenComparisonPolicy
from yewo.agent.actuation import ActionKind
from yewo.agent.discovery import (
    DiscoveryCoverage,
    DiscoveryCoverageEntry,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
    DiscoveryResult,
)
from yewo.agent.replay import (
    COMPILER_REFUSAL_PATH,
    EVIDENCE_DOCUMENT_PATH,
    RECONCILER_RESPONSE_PATH,
    REPLAY_PROCEDURE_PATH,
    PROCEDURE_COMPILATION_REPORT_PATH,
    ProcedureCompilationStatus,
)
from yewo.agent.replay.preparation import compile_completed_run
from yewo.agent.replay.procedure_store import CompiledProcedureStore
from yewo.agent.memory.knowledge import (
    AppKnowledge,
    HazardKind,
    HazardKnowledge,
    KnowledgeEvidenceReference,
    KnowledgeRule,
)
from yewo.agent.memory.records import MemoryCompatibility
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity
from yewo.models import (
    ModelArtifactKind,
    ModelArtifactReference,
    ModelCallDisposition,
    ModelCallUsage,
    ModelRequest,
    ModelResponse,
    TextModelContent,
)
from yewo.observations import Observation

from agent.discovery.support import agent_result, observation
from agent.planning.support import StubPlannerModel
from agent.memory.test_store import _memory


class MeasuredReconciler(StubPlannerModel):
    """Return one measured response with a durable call reference."""

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        response = super().complete(request, response_model, validation_context)
        call = response.calls[0].model_copy(
            update={
                "latency_ms": 125.5,
                "usage": ModelCallUsage(
                    prompt_tokens=1_000,
                    completion_tokens=200,
                    reasoning_tokens=150,
                    total_tokens=1_200,
                    cost_usd=Decimal("0.0125"),
                ),
                "call_artifact": ModelArtifactReference(
                    artifact_id="model_call_1",
                    kind=ModelArtifactKind.CALL,
                    sha256="a" * 64,
                    body_sha256="b" * 64,
                    content_sha256="c" * 64,
                    path=Path("assets/models/calls.jsonl"),
                    body_retained=True,
                    sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
                    retention=ArtifactRetention.RUN,
                ),
            }
        )
        return response.model_copy(update={"calls": (call,)})


class SequentialReconciler(StubPlannerModel):
    """Return request-bound reconciler outputs in call order."""

    def __init__(self, procedures: tuple[list[int], ...]) -> None:
        super().__init__(None)
        self._procedures = iter(procedures)

    def complete(
        self,
        request: ModelRequest,
        response_model: type[BaseModel] | None = None,
        validation_context: dict[str, object] | None = None,
    ) -> ModelResponse:
        procedure = next(self._procedures)
        self.output = {
            "schema_version": 1,
            "request_id": request.request_id,
            "procedure": procedure,
            "milestones": {"0": 1} if procedure else {},
            "decisions": [],
            "findings": [],
        }
        return super().complete(request, response_model, validation_context)


def _result() -> tuple[DiscoveryResult, Observation, Observation]:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "ffffffffffffffff",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    request = DiscoveryRequest(
        request_id="discovery_1",
        objectives=(
            DiscoveryObjective(
                objective_id="objective_1",
                statement="Save one reusable schedule.",
                completion_conditions=("The schedule is saved.",),
            ),
        ),
        priorities=("Create the schedule.",),
    )
    result = DiscoveryResult(
        request=request,
        terminal_reason=AgentTerminalReason.DONE_REQUESTED,
        coverage=DiscoveryCoverage(
            entries=(
                DiscoveryCoverageEntry(
                    objective_id="objective_1",
                    state=DiscoveryObjectiveState.SUPPORTED,
                    procedure_id="procedure_1",
                    evidence_observation_ids=(after.observation_id,),
                    reason="The compiled procedure supports the objective.",
                ),
            )
        ),
        agent_result=agent_result(before, after),
        compiled_procedure_id="procedure_1",
        compiled_procedure_step_count=1,
        memory_id="memory_1",
    )
    return result, before, after


def _identifiers(*values: str) -> Callable[[], str]:
    identifiers = iter(values)
    return lambda: next(identifiers)


def test_completed_result_builds_calls_compiles_and_measures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, before, after = _result()
    model = MeasuredReconciler(
        {
            "schema_version": 1,
            "request_id": "model_request_1",
            "procedure": [1],
            "milestones": {"0": 1},
            "decisions": [],
            "findings": [],
        }
    )
    monkeypatch.setattr(
        "yewo.agent.replay.procedure_compilation.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )
    compatibility = MemoryCompatibility(application=before.application)
    procedure_store = CompiledProcedureStore(tmp_path / "durable")

    report = compile_completed_run(
        root=tmp_path,
        run_id="run_1",
        result=result,
        model_client=model,
        model_profile_id="reconciler_primary",
        comparison=ScreenComparisonPolicy(),
        procedure_store=procedure_store,
        compatibility=compatibility,
        identifier_factory=_identifiers("evidence_1", "model_request_1", "procedure_1"),
    )

    assert report.status is ProcedureCompilationStatus.COMPILED
    assert report.procedure_id == "procedure_1"
    assert report.model_call_artifact_ids == ("model_call_1",)
    assert report.latency_ms == 125.5
    assert report.prompt_tokens == 1_000
    assert report.completion_tokens == 200
    assert report.reasoning_tokens == 150
    assert report.cost_usd == Decimal("0.0125")
    request = model.requests[0]
    assert request.reasoning_effort is None
    assert request.max_completion_tokens == 8_192
    assert request.response_contract is not None
    assert request.response_contract.name == "replay-reconciliation-v1"
    content = request.messages[1].content[0]
    assert isinstance(content, TextModelContent)
    payload = json.loads(content.text.split("Request and evidence JSON follow:\n", 1)[1])
    projected = payload["evidence"]
    assert projected["projection_version"] == 1
    assert "timeline" not in projected
    assert "fingerprint" not in next(iter(projected["states"].values()))
    assert "target_grounding" not in projected["steps"][0]
    assert "target_visual_descriptor" not in projected["steps"][0]
    for path in (
        EVIDENCE_DOCUMENT_PATH,
        RECONCILER_RESPONSE_PATH,
        REPLAY_PROCEDURE_PATH,
        PROCEDURE_COMPILATION_REPORT_PATH,
    ):
        assert (tmp_path / path).is_file()
    assert not (tmp_path / COMPILER_REFUSAL_PATH).exists()
    stored = procedure_store.load(
        compatibility,
        "objective_1",
        "Save one reusable schedule.",
    )
    assert stored is not None
    assert stored.procedure.procedure_id == "procedure_1"


def test_reconciler_receives_hazard_facts_and_advisory_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, before, after = _result()
    unused_compatibility, memory = _memory(tmp_path / "memory-source")
    source = KnowledgeEvidenceReference(run_id="run_1", observation_id="obs_1")
    hazard = HazardKnowledge(
        hazard_id="hazard_1",
        screen_id=memory.screens[0].screen_id,
        action=ActionKind.BACK,
        kind=HazardKind.CRASH,
        seen_count=1,
        sources=(source,),
    )
    rule = KnowledgeRule(
        rule_id="rule_1",
        text="Complete required fields before submission.",
        sources=(source,),
    )
    memory = memory.model_copy(
        update={"app_knowledge": AppKnowledge(hazards=(hazard,), rules=(rule,))}
    )
    model = StubPlannerModel(
        {
            "schema_version": 1,
            "request_id": "model_request_1",
            "procedure": [1],
            "milestones": {"0": 1},
            "decisions": [],
            "findings": [],
        }
    )
    monkeypatch.setattr(
        "yewo.agent.replay.procedure_compilation.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    report = compile_completed_run(
        root=tmp_path,
        run_id="run_1",
        result=result,
        model_client=model,
        model_profile_id="reconciler_primary",
        comparison=ScreenComparisonPolicy(),
        memory=memory,
        identifier_factory=_identifiers("evidence_1", "model_request_1", "procedure_1"),
    )

    assert report.status is ProcedureCompilationStatus.COMPILED
    content = model.requests[0].messages[1].content[0]
    assert isinstance(content, TextModelContent)
    assert '"memory_hazards"' in content.text
    assert '"hazard_id":"hazard_1"' in content.text
    assert '"memory_claims"' in content.text
    assert '"claim_id":"rule_1"' in content.text


def test_compiler_refusal_stores_no_procedure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, before, after = _result()
    model = StubPlannerModel(
        {
            "schema_version": 1,
            "request_id": "model_request_1",
            "procedure": [],
            "milestones": {},
            "decisions": [],
            "findings": [],
        }
    )
    monkeypatch.setattr(
        "yewo.agent.replay.procedure_compilation.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    report = compile_completed_run(
        root=tmp_path,
        run_id="run_1",
        result=result,
        model_client=model,
        model_profile_id="reconciler_primary",
        comparison=ScreenComparisonPolicy(),
        identifier_factory=_identifiers(
            "evidence_1",
            "model_request_1",
            "model_request_2",
        ),
    )

    assert report.status is ProcedureCompilationStatus.COMPILER_REFUSED
    assert [request.reasoning_effort for request in model.requests] == [None, None]
    assert report.compiler_refusal_reason_count > 0
    assert (tmp_path / COMPILER_REFUSAL_PATH).is_file()
    assert not (tmp_path / REPLAY_PROCEDURE_PATH).exists()


def test_compiler_feedback_fallback_can_repair_a_refusal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, before, after = _result()
    model = SequentialReconciler(([], [1]))
    monkeypatch.setattr(
        "yewo.agent.replay.procedure_compilation.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    report = compile_completed_run(
        root=tmp_path,
        run_id="run_1",
        result=result,
        model_client=model,
        model_profile_id="reconciler_primary",
        comparison=ScreenComparisonPolicy(),
        identifier_factory=_identifiers(
            "evidence_1",
            "model_request_1",
            "model_request_2",
            "procedure_1",
        ),
    )

    assert report.status is ProcedureCompilationStatus.COMPILED
    assert report.request_id == "model_request_2"
    assert [request.reasoning_effort for request in model.requests] == [None, None]
    fallback_context = model.requests[1].messages[1].content[0]
    assert isinstance(fallback_context, TextModelContent)
    assert "compiler_refusal" in fallback_context.text


def test_model_failure_stores_report_without_compilation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, before, after = _result()
    model = StubPlannerModel(None, disposition=ModelCallDisposition.PROVIDER_ERROR)
    monkeypatch.setattr(
        "yewo.agent.replay.procedure_compilation.ObservationStore.load_selected",
        lambda unused, observation_ids: (before, after),
    )

    report = compile_completed_run(
        root=tmp_path,
        run_id="run_1",
        result=result,
        model_client=model,
        model_profile_id="reconciler_primary",
        comparison=ScreenComparisonPolicy(),
        identifier_factory=_identifiers("evidence_1", "model_request_1"),
    )

    assert report.status is ProcedureCompilationStatus.MODEL_FAILED
    assert (tmp_path / EVIDENCE_DOCUMENT_PATH).is_file()
    assert (tmp_path / PROCEDURE_COMPILATION_REPORT_PATH).is_file()
    assert not (tmp_path / RECONCILER_RESPONSE_PATH).exists()
    assert not (tmp_path / COMPILER_REFUSAL_PATH).exists()
    assert not (tmp_path / REPLAY_PROCEDURE_PATH).exists()
