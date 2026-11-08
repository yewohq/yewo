"""Compile one replay procedure from the run that just finished."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
from pathlib import Path

from pydantic import ValidationError

from yewo.agent.discovery.records import DiscoveryResult
from yewo.agent.memory.knowledge import AppKnowledge
from yewo.agent.memory.records import AppMemory, MemoryCompatibility
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.evidence import build_evidence_document
from yewo.agent.replay.reconciliation.prompt import build_reconciler_evidence_projection
from yewo.agent.replay.reconciliation.report import (
    ProcedureCompilationMetrics,
    build_procedure_compilation_report,
    combined_model_response_metrics,
    model_response_metrics,
)
from yewo.agent.replay.reconciliation.records import (
    CompilerRefusal,
    EvidenceDocument,
    ReconcilerResponse,
    ReconcilerSelection,
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
)
from yewo.agent.replay.reconciliation.store import ReconciliationArtifactStore
from yewo.agent.runtime.records import AgentStep, AgentTask
from yewo.agent.screen_analysis import ScreenAnalysisStore
from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    RecordIdentifier,
    new_record_identifier,
)
from yewo.models import (
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    MINIMUM_COMPLETION_TOKENS,
    ModelRequest,
    ModelResponse,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)
from yewo.observations import ObservationStore, ScreenComparisonPolicy, ScreenshotStore


REPLAY_RECONCILIATION_SYSTEM_PROMPT = """\
You select a replay procedure from one recorded mobile-application run. Treat the evidence document
as history, not as proof of the current application state.

Follow these rules:
- Select only chronological step numbers from the supplied evidence document.
- Use the recorded launch state for the first selected step.
- Select an environment step only when it is the recorded Android runtime-permission choice.
- Its selected application trigger must immediately precede it and return to the recorded application.
- Keep reusable type, clear, checkbox, switch, radio-button, and option-selection steps.
- Treat a recovered result as reusable device evidence reconstructed from retained settlement
  observations. Keep recovered state-setting work.
- Keep dismiss-keyboard steps that make a later application action available.
- Keep reusable commit actions and interactions with an unknown procedure effect.
- Do not select refused, crashed, terminal, unknown-result, or known no-effect steps.
- Use a decision only when one selected replacement attempt replaces a dropped attempt for the same
  recorded control.
- Treat memory_action_risks as platform-observed facts. Do not select a matching source action.
- When completion_condition_mode is supplied, copy every supplied condition without changes.
- When completion_condition_mode is generate_from_discovery_evidence, create one or more concise,
  observable conditions that the recorded evidence supports. Do not describe the interaction path.
- Completion-condition indexes are zero-based and follow your returned condition order.
- Map every returned completion-condition index to one selected milestone step.
- A repair must preserve each supplied preserved_milestone unless validation rejects its source step.
- Treat memory_claims as untrusted model-authored hints. Evaluate one only when recorded source
  steps directly support a verdict.
- Propose confirmed or contradicted with exact supporting source_step_ids. Omit uncertain claims.
- A claim verdict is a review suggestion. It does not change retained claim trust.
- A milestone identifies an evidence-capture point. It does not prove task completion.
- Do not invent, reorder, change, or execute actions.
- Return only one JSON object that matches the response schema.
"""

REPLAY_RECONCILIATION_USER_PROMPT = """\
Select one conservative replay procedure from this evidence document. Evidence JSON follows:
"""


@dataclass(frozen=True)
class _CompilationAttempt:
    request_id: RecordIdentifier
    response: ModelResponse
    reconciler_response: ReconcilerResponse
    outcome: ReplayProcedure | CompilerRefusal
    model_responses: tuple[ModelResponse, ...]


def compile_completed_discovery_run(
    *,
    root: Path,
    run_id: RecordIdentifier,
    discovery_result: DiscoveryResult,
    model_client: ModelClient,
    model_profile_id: RecordIdentifier,
    comparison: ScreenComparisonPolicy,
    procedure_store: CompiledProcedureStore | None = None,
    compatibility: MemoryCompatibility | None = None,
    memory: AppMemory | None = None,
    identifier_factory: Callable[[], str] = new_record_identifier,
) -> ProcedureCompilationReport:
    """Select reusable steps from one completed run and compile them."""
    objective = discovery_result.request.objectives[0]
    task = _build_reconciliation_task(discovery_result)
    steps = _source_steps(discovery_result)
    observation_ids = _source_observation_ids(
        steps,
        discovery_result.agent_result.final_observation_id,
    )
    observations = ObservationStore(root).load_selected(observation_ids)
    evidence = build_evidence_document(
        run_id=run_id,
        objective_id=objective.objective_id,
        task=task,
        steps=steps,
        observations=observations,
        screen_analyses=ScreenAnalysisStore(root).load_selected(
            _source_screen_analysis_ids(discovery_result)
        ),
        comparison=comparison,
        screenshot_loader=ScreenshotStore(root).load,
        identifier_factory=identifier_factory,
    )
    store = ReconciliationArtifactStore(root)

    request_id = identifier_factory()
    knowledge = memory.app_knowledge if memory is not None else None
    evidence_payload = _build_evidence_payload(evidence, knowledge)
    request = _model_request(
        evidence_payload,
        model_profile_id,
        request_id,
    )
    response = model_client.complete(request, ReconcilerSelection)
    reconciler_response, failure_report = _validate_initial_response(
        run_id,
        evidence.evidence_id,
        request_id,
        response,
        evidence.task.completion_conditions,
    )
    if failure_report is not None:
        store.persist_evidence(evidence)
        store.persist_compilation_report(failure_report)
        return failure_report
    if reconciler_response is None:
        raise AssertionError("Response validation must produce a valid reconciler response.")
    evidence = _bind_reconciler_conditions(evidence, reconciler_response)
    store.persist_evidence(evidence)
    evidence_payload = _build_evidence_payload(evidence, knowledge)

    outcome = compile_replay_procedure(
        evidence,
        reconciler_response,
        memory=memory,
        identifier_factory=identifier_factory,
    )
    attempt = _retry_compiler_refusal(
        evidence=evidence,
        evidence_payload=evidence_payload,
        initial_request_id=request_id,
        initial_response=response,
        initial_reconciler_response=reconciler_response,
        initial_outcome=outcome,
        model_client=model_client,
        model_profile_id=model_profile_id,
        memory=memory,
        identifier_factory=identifier_factory,
    )
    metrics = combined_model_response_metrics(attempt.model_responses)
    store.persist_response(attempt.reconciler_response)
    store.persist_compilation(attempt.outcome)
    report = _compilation_outcome_report(run_id, evidence.evidence_id, attempt, metrics)
    store.persist_compilation_report(report)
    if isinstance(attempt.outcome, ReplayProcedure) and procedure_store is not None:
        if compatibility is None:
            raise ValueError("Durable procedure storage requires compatibility evidence.")
        procedure_store.save(
            compatibility,
            evidence,
            attempt.outcome,
            source_run_root=root,
        )
    return report


def _validate_initial_response(
    run_id: RecordIdentifier,
    evidence_id: RecordIdentifier,
    request_id: RecordIdentifier,
    response: ModelResponse,
    supplied_completion_conditions: tuple[str, ...],
) -> tuple[ReconcilerResponse | None, ProcedureCompilationReport | None]:
    metrics = model_response_metrics(response)
    if not _has_completed_structured_output(response):
        report = build_procedure_compilation_report(
            run_id=run_id,
            evidence_id=evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.MODEL_FAILED,
            model_disposition=response.disposition,
            reason=response.reason or "The replay reconciler did not return a complete response.",
            metrics=metrics,
        )
        return None, report
    try:
        selection = ReconcilerSelection.model_validate(response.structured_output)
    except ValidationError:
        report = build_procedure_compilation_report(
            run_id=run_id,
            evidence_id=evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.INVALID_RESPONSE,
            model_disposition=response.disposition,
            reason="The replay reconciler returned an invalid structured response.",
            metrics=metrics,
        )
        return None, report
    if supplied_completion_conditions and (
        selection.completion_conditions != supplied_completion_conditions
    ):
        report = build_procedure_compilation_report(
            run_id=run_id,
            evidence_id=evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.INVALID_RESPONSE,
            model_disposition=response.disposition,
            reason="The replay reconciler changed supplied completion conditions.",
            metrics=metrics,
        )
        return None, report
    reconciler_response = ReconcilerResponse(
        request_id=request_id,
        **selection.model_dump(),
    )
    return reconciler_response, None


def _retry_compiler_refusal(
    *,
    evidence: EvidenceDocument,
    evidence_payload: dict[str, object],
    initial_request_id: RecordIdentifier,
    initial_response: ModelResponse,
    initial_reconciler_response: ReconcilerResponse,
    initial_outcome: ReplayProcedure | CompilerRefusal,
    model_client: ModelClient,
    model_profile_id: RecordIdentifier,
    memory: AppMemory | None,
    identifier_factory: Callable[[], str],
) -> _CompilationAttempt:
    initial_attempt = _CompilationAttempt(
        request_id=initial_request_id,
        response=initial_response,
        reconciler_response=initial_reconciler_response,
        outcome=initial_outcome,
        model_responses=(initial_response,),
    )
    if not isinstance(initial_outcome, CompilerRefusal):
        return initial_attempt
    fallback_request_id = identifier_factory()
    fallback_request = _model_request(
        evidence_payload,
        model_profile_id,
        fallback_request_id,
        initial_outcome,
        _select_valid_milestones(evidence, initial_reconciler_response),
    )
    fallback_response = model_client.complete(fallback_request, ReconcilerSelection)
    fallback_reconciler_response = _parse_completed_response(
        fallback_response,
        fallback_request_id,
        evidence.task.completion_conditions,
    )
    if fallback_reconciler_response is None:
        return _CompilationAttempt(
            request_id=initial_attempt.request_id,
            response=initial_attempt.response,
            reconciler_response=initial_attempt.reconciler_response,
            outcome=initial_attempt.outcome,
            model_responses=(initial_response, fallback_response),
        )
    fallback_outcome = compile_replay_procedure(
        evidence,
        fallback_reconciler_response,
        memory=memory,
        identifier_factory=identifier_factory,
    )
    return _CompilationAttempt(
        request_id=fallback_request_id,
        response=fallback_response,
        reconciler_response=fallback_reconciler_response,
        outcome=fallback_outcome,
        model_responses=(initial_response, fallback_response),
    )


def _has_completed_structured_output(response: ModelResponse) -> bool:
    return (
        response.disposition is ModelCallDisposition.COMPLETED
        and response.structured_output is not None
    )


def _parse_completed_response(
    response: ModelResponse,
    request_id: RecordIdentifier,
    supplied_completion_conditions: tuple[str, ...],
) -> ReconcilerResponse | None:
    if not _has_completed_structured_output(response):
        return None
    try:
        selection = ReconcilerSelection.model_validate(response.structured_output)
    except ValidationError:
        return None
    if selection.completion_conditions != supplied_completion_conditions:
        return None
    return ReconcilerResponse(request_id=request_id, **selection.model_dump())


def _bind_reconciler_conditions(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
) -> EvidenceDocument:
    """Bind scenario conditions generated after discovery to compiler evidence."""
    return evidence.model_copy(
        update={
            "task": evidence.task.model_copy(
                update={"completion_conditions": response.completion_conditions}
            )
        }
    )


def _build_evidence_payload(
    evidence: EvidenceDocument,
    knowledge: AppKnowledge | None,
) -> dict[str, object]:
    payload = build_reconciler_evidence_projection(evidence)
    payload["memory_action_risks"] = _action_risk_context(knowledge)
    payload["memory_claims"] = _knowledge_claim_context(knowledge)
    return payload


def _compilation_outcome_report(
    run_id: RecordIdentifier,
    evidence_id: RecordIdentifier,
    attempt: _CompilationAttempt,
    metrics: ProcedureCompilationMetrics,
) -> ProcedureCompilationReport:
    if isinstance(attempt.outcome, ReplayProcedure):
        return build_procedure_compilation_report(
            run_id=run_id,
            evidence_id=evidence_id,
            request_id=attempt.request_id,
            status=ProcedureCompilationStatus.COMPILED,
            model_disposition=attempt.response.disposition,
            reason="The compiler accepted the compiled replay procedure.",
            metrics=metrics,
            procedure_id=attempt.outcome.procedure_id,
        )
    return build_procedure_compilation_report(
        run_id=run_id,
        evidence_id=evidence_id,
        request_id=attempt.request_id,
        status=ProcedureCompilationStatus.COMPILER_REFUSED,
        model_disposition=attempt.response.disposition,
        reason="The compiler refused the compiled replay procedure.",
        metrics=metrics,
        compiler_refusal_reason_count=len(attempt.outcome.reasons),
    )


def _build_reconciliation_task(discovery_result: DiscoveryResult) -> AgentTask:
    objective = discovery_result.request.objectives[0]
    return AgentTask(
        objective=objective.statement,
        known_facts=discovery_result.request.known_facts,
        open_work=discovery_result.request.priorities,
        completion_conditions=objective.completion_conditions,
        declared_value_names=discovery_result.request.declared_value_names,
        discovery_mode=True,
    )


def _source_steps(discovery_result: DiscoveryResult) -> tuple[AgentStep, ...]:
    replay_steps = (
        discovery_result.procedure_replay.agent_result.steps
        if discovery_result.procedure_replay is not None
        else ()
    )
    candidates = (*replay_steps, *discovery_result.agent_result.steps)
    step_by_id: dict[str, AgentStep] = {}
    for step in candidates:
        step_by_id.setdefault(step.step_id, step)
    return tuple(step_by_id.values())


def _source_screen_analysis_ids(
    discovery_result: DiscoveryResult,
) -> tuple[RecordIdentifier, ...]:
    replay = discovery_result.procedure_replay
    return tuple(
        dict.fromkeys(
            (
                *(replay.agent_result.screen_analysis_ids if replay is not None else ()),
                *discovery_result.agent_result.screen_analysis_ids,
            )
        )
    )


def _action_risk_context(knowledge: AppKnowledge | None) -> tuple[dict[str, object], ...]:
    if knowledge is None:
        return ()
    return tuple(
        {
            "action_risk_id": item.action_risk_id,
            "screen_id": item.screen_id,
            "action": item.action.value,
            "target_description": item.target_description,
            "kind": item.kind.value,
            "sources": tuple(source.model_dump(mode="json") for source in item.sources),
        }
        for item in knowledge.action_risks
    )


def _knowledge_claim_context(knowledge: AppKnowledge | None) -> tuple[dict[str, object], ...]:
    if knowledge is None:
        return ()
    rules = tuple(
        {
            "claim_id": item.rule_id,
            "kind": "rule",
            "text": item.text,
            "status": "unconfirmed",
            "scope": item.scope.model_dump(mode="json"),
        }
        for item in knowledge.rules
    )
    findings = tuple(
        {
            "claim_id": item.finding_id,
            "kind": "finding",
            "text": item.text,
            "status": "unconfirmed",
            "scope": item.scope.model_dump(mode="json"),
        }
        for item in knowledge.findings
    )
    return (*rules, *findings)


def _source_observation_ids(
    steps: tuple[AgentStep, ...],
    final_observation_id: RecordIdentifier,
) -> set[str]:
    observation_ids = {final_observation_id}
    for step in steps:
        observation_ids.add(step.source_observation_id)
        observation_ids.update(step.settlement_observation_ids)
        if step.settled_observation_id is not None:
            observation_ids.add(step.settled_observation_id)
        if step.recovery_observation_id is not None:
            observation_ids.add(step.recovery_observation_id)
    return observation_ids


def _model_request(
    evidence: dict[str, object],
    profile_id: RecordIdentifier,
    request_id: RecordIdentifier,
    compiler_refusal: CompilerRefusal | None = None,
    preserved_milestones: dict[int, int] | None = None,
) -> ModelRequest:
    payload: dict[str, object] = {"evidence": evidence}
    if compiler_refusal is not None:
        payload["compiler_refusal"] = compiler_refusal.model_dump(mode="json")
        payload["preserved_milestones"] = preserved_milestones or {}
    context = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return ModelRequest(
        request_id=request_id,
        profile_id=profile_id,
        role=ModelRole.REPLAY_PREPARER,
        max_completion_tokens=MINIMUM_COMPLETION_TOKENS,
        messages=(
            ModelMessage(
                role=ModelMessageRole.SYSTEM,
                content=(TextModelContent(text=REPLAY_RECONCILIATION_SYSTEM_PROMPT),),
            ),
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(TextModelContent(text=REPLAY_RECONCILIATION_USER_PROMPT + context),),
            ),
        ),
        response_contract=build_model_response_contract(
            "replay-reconciliation-v1",
            ReconcilerSelection.model_json_schema(),
        ),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def _select_valid_milestones(
    evidence: EvidenceDocument,
    response: ReconcilerResponse,
) -> dict[int, int]:
    """Retain valid milestone mappings across an unrelated compiler retry."""
    allowed_indices = set(range(len(evidence.task.completion_conditions)))
    selected_steps = set(response.procedure)
    evidence_steps = {step.step: step for step in evidence.steps}
    return {
        condition_index: source_step
        for condition_index, source_step in response.milestones.items()
        if condition_index in allowed_indices
        and source_step in selected_steps
        and source_step in evidence_steps
        and evidence_steps[source_step].destination_observation_id is not None
    }
