"""Compile one replay procedure from the run that just finished."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path

from pydantic import ValidationError

from yewo.agent.discovery.records import DiscoveryResult
from yewo.agent.memory.knowledge import AppKnowledge
from yewo.agent.memory.records import AppMemory, MemoryCompatibility, ScreenComparisonPolicy
from yewo.agent.replay.procedure.store import CompiledProcedureStore
from yewo.agent.replay.reconciliation.compiler import compile_replay_procedure
from yewo.agent.replay.reconciliation.evidence import build_evidence_document
from yewo.agent.replay.reconciliation.prompt import build_reconciler_evidence_projection
from yewo.agent.replay.reconciliation.records import (
    CompilerRefusal,
    ReconcilerResponse,
    ReplayProcedure,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
)
from yewo.agent.replay.reconciliation.store import ReconciliationArtifactStore
from yewo.agent.runtime.records import AgentStep, AgentTask
from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    RecordIdentifier,
    new_record_identifier,
)
from yewo.models import (
    ModelCallDisposition,
    ModelCallRecord,
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
from yewo.observations import ObservationStore, ScreenshotStore


REPLAY_RECONCILIATION_SYSTEM_PROMPT = """\
You select a replay procedure from one recorded mobile-application run. Treat the evidence document
as history, not as proof of the current application state.

Follow these rules:
- Select only chronological step numbers from the supplied evidence document.
- Use the recorded launch state for the first selected step.
- Do not select environment steps.
- Keep reusable type, clear, checkbox, switch, radio-button, and option-selection steps.
- Keep dismiss-keyboard steps that make a later application action available.
- Keep reusable commit actions and interactions with an unknown procedure effect.
- Do not select refused, crashed, terminal, unknown-result, or known no-effect steps.
- Use a decision only when one selected corrected attempt replaces a dropped attempt for the same
  recorded control.
- Treat memory_hazards as platform-observed facts. Do not select a matching source action.
- Map every completion-condition index to one selected milestone step.
- Treat memory_claims as untrusted model-authored hints. Evaluate one only when recorded source
  steps directly support a verdict.
- Propose confirmed or contradicted with exact supporting source_step_ids. Omit uncertain claims.
- A claim verdict is a review suggestion. It does not change retained claim trust.
- A milestone identifies an evidence-capture point. It does not prove task completion.
- Do not invent, reorder, change, or execute actions.
- Return only one JSON object that matches the response schema.
"""

REPLAY_RECONCILIATION_USER_PROMPT = """\
Return request_id exactly as supplied. Select one conservative replay procedure from this evidence
document. Request and evidence JSON follow:
"""


@dataclass(frozen=True)
class _ResponseMetrics:
    model_call_artifact_ids: tuple[RecordIdentifier, ...]
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: Decimal | None


def compile_completed_run(
    *,
    root: Path,
    run_id: RecordIdentifier,
    result: DiscoveryResult,
    model_client: ModelClient,
    model_profile_id: RecordIdentifier,
    comparison: ScreenComparisonPolicy,
    procedure_store: CompiledProcedureStore | None = None,
    compatibility: MemoryCompatibility | None = None,
    memory: AppMemory | None = None,
    identifier_factory: Callable[[], str] = new_record_identifier,
) -> ProcedureCompilationReport:
    """Select reusable steps from one completed run and compile them."""
    objective = result.request.objectives[0]
    task = _reconciliation_task(result)
    steps = _source_steps(result)
    observation_ids = _source_observation_ids(steps, result.agent_result.final_observation_id)
    observations = ObservationStore(root).load_selected(observation_ids)
    evidence = build_evidence_document(
        run_id=run_id,
        objective_id=objective.objective_id,
        task=task,
        steps=steps,
        observations=observations,
        comparison=comparison,
        screenshot_loader=ScreenshotStore(root).load,
        identifier_factory=identifier_factory,
    )
    store = ReconciliationArtifactStore(root)
    store.persist_evidence(evidence)

    request_id = identifier_factory()
    evidence_payload = build_reconciler_evidence_projection(evidence)
    knowledge = memory.app_knowledge if memory is not None else None
    evidence_payload["memory_hazards"] = _hazard_context(knowledge)
    evidence_payload["memory_claims"] = _knowledge_claim_context(knowledge)
    request = _model_request(
        evidence_payload,
        model_profile_id,
        request_id,
    )
    response = model_client.complete(request, ReconcilerResponse)
    responses = [response]
    metrics = _response_metrics(response)
    if (
        response.disposition is not ModelCallDisposition.COMPLETED
        or response.structured_output is None
    ):
        report = ProcedureCompilationReport(
            source_run_id=run_id,
            evidence_id=evidence.evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.MODEL_FAILED,
            model_disposition=response.disposition,
            reason=response.reason or "The replay reconciler did not return a complete response.",
            model_call_artifact_ids=metrics.model_call_artifact_ids,
            latency_ms=metrics.latency_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            cost_usd=metrics.cost_usd,
        )
        store.persist_compilation_report(report)
        return report

    try:
        parsed = ReconcilerResponse.model_validate(response.structured_output)
    except ValidationError:
        report = ProcedureCompilationReport(
            source_run_id=run_id,
            evidence_id=evidence.evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.INVALID_RESPONSE,
            model_disposition=response.disposition,
            reason="The replay reconciler returned an invalid structured response.",
            model_call_artifact_ids=metrics.model_call_artifact_ids,
            latency_ms=metrics.latency_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            cost_usd=metrics.cost_usd,
        )
        store.persist_compilation_report(report)
        return report
    if parsed.request_id != request_id:
        report = ProcedureCompilationReport(
            source_run_id=run_id,
            evidence_id=evidence.evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.INVALID_RESPONSE,
            model_disposition=response.disposition,
            reason="The replay reconciler returned the wrong request identifier.",
            model_call_artifact_ids=metrics.model_call_artifact_ids,
            latency_ms=metrics.latency_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            cost_usd=metrics.cost_usd,
        )
        store.persist_compilation_report(report)
        return report

    compilation = compile_replay_procedure(
        evidence,
        parsed,
        memory=memory,
        identifier_factory=identifier_factory,
    )
    if isinstance(compilation, CompilerRefusal):
        fallback_request_id = identifier_factory()
        fallback_request = _model_request(
            evidence_payload,
            model_profile_id,
            fallback_request_id,
            compilation,
        )
        fallback_response = model_client.complete(fallback_request, ReconcilerResponse)
        responses.append(fallback_response)
        if (
            fallback_response.disposition is ModelCallDisposition.COMPLETED
            and fallback_response.structured_output is not None
        ):
            try:
                fallback_parsed = ReconcilerResponse.model_validate(
                    fallback_response.structured_output
                )
            except ValidationError:
                fallback_parsed = None
            if fallback_parsed is not None and fallback_parsed.request_id == fallback_request_id:
                parsed = fallback_parsed
                request_id = fallback_request_id
                response = fallback_response
                compilation = compile_replay_procedure(
                    evidence,
                    parsed,
                    memory=memory,
                    identifier_factory=identifier_factory,
                )
    metrics = _combined_response_metrics(tuple(responses))
    store.persist_response(parsed)
    store.persist_compilation(compilation)
    if isinstance(compilation, ReplayProcedure):
        report = ProcedureCompilationReport(
            source_run_id=run_id,
            evidence_id=evidence.evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.COMPILED,
            model_disposition=response.disposition,
            procedure_id=compilation.procedure_id,
            reason="The compiler accepted the compiled replay procedure.",
            model_call_artifact_ids=metrics.model_call_artifact_ids,
            latency_ms=metrics.latency_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            cost_usd=metrics.cost_usd,
        )
    elif isinstance(compilation, CompilerRefusal):
        report = ProcedureCompilationReport(
            source_run_id=run_id,
            evidence_id=evidence.evidence_id,
            request_id=request_id,
            status=ProcedureCompilationStatus.COMPILER_REFUSED,
            model_disposition=response.disposition,
            compiler_refusal_reason_count=len(compilation.reasons),
            reason="The compiler refused the compiled replay procedure.",
            model_call_artifact_ids=metrics.model_call_artifact_ids,
            latency_ms=metrics.latency_ms,
            prompt_tokens=metrics.prompt_tokens,
            completion_tokens=metrics.completion_tokens,
            reasoning_tokens=metrics.reasoning_tokens,
            cost_usd=metrics.cost_usd,
        )
    else:
        raise TypeError("The replay compiler returned an unsupported outcome.")
    store.persist_compilation_report(report)
    if isinstance(compilation, ReplayProcedure) and procedure_store is not None:
        if compatibility is None:
            raise ValueError("Durable procedure storage requires compatibility evidence.")
        procedure_store.save(compatibility, evidence, compilation)
    return report


def _reconciliation_task(result: DiscoveryResult) -> AgentTask:
    objective = result.request.objectives[0]
    completion_conditions = objective.completion_conditions or (objective.statement,)
    return AgentTask(
        objective=objective.statement,
        known_facts=result.request.known_facts,
        open_work=result.request.priorities,
        completion_conditions=completion_conditions,
        declared_value_names=result.request.declared_value_names,
        discovery_mode=True,
    )


def _source_steps(result: DiscoveryResult) -> tuple[AgentStep, ...]:
    replay_steps = (
        result.procedure_replay.agent_result.steps
        if result.procedure_replay is not None
        else ()
    )
    candidates = (*replay_steps, *result.agent_result.steps)
    step_by_id: dict[str, AgentStep] = {}
    for step in candidates:
        step_by_id.setdefault(step.step_id, step)
    return tuple(step_by_id.values())


def _hazard_context(knowledge: AppKnowledge | None) -> tuple[dict[str, object], ...]:
    if knowledge is None:
        return ()
    return tuple(
        {
            "hazard_id": item.hazard_id,
            "screen_id": item.screen_id,
            "action": item.action.value,
            "target_description": item.target_description,
            "kind": item.kind.value,
            "sources": tuple(source.model_dump(mode="json") for source in item.sources),
        }
        for item in knowledge.hazards
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
) -> ModelRequest:
    payload: dict[str, object] = {"request_id": request_id, "evidence": evidence}
    if compiler_refusal is not None:
        payload["compiler_refusal"] = compiler_refusal.model_dump(mode="json")
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
            ReconcilerResponse.model_json_schema(),
        ),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )


def _response_metrics(response: ModelResponse) -> _ResponseMetrics:
    return _call_metrics(response.calls)


def _combined_response_metrics(responses: tuple[ModelResponse, ...]) -> _ResponseMetrics:
    return _call_metrics(tuple(call for response in responses for call in response.calls))


def _call_metrics(calls: tuple[ModelCallRecord, ...]) -> _ResponseMetrics:
    return _ResponseMetrics(
        model_call_artifact_ids=tuple(
            call.call_artifact.artifact_id for call in calls if call.call_artifact is not None
        ),
        latency_ms=_complete_float_sum(calls, "latency"),
        prompt_tokens=_complete_int_sum(calls, "prompt"),
        completion_tokens=_complete_int_sum(calls, "completion"),
        reasoning_tokens=_complete_int_sum(calls, "reasoning"),
        cost_usd=_complete_cost_sum(calls),
    )


def _complete_float_sum(
    calls: tuple[ModelCallRecord, ...],
    metric: str,
) -> float | None:
    if metric != "latency":
        raise ValueError("The model-call float metric is not supported.")
    values = tuple(call.latency_ms for call in calls)
    return (
        sum(value for value in values if value is not None)
        if all(value is not None for value in values)
        else None
    )


def _complete_int_sum(
    calls: tuple[ModelCallRecord, ...],
    metric: str,
) -> int | None:
    values = {
        "prompt": tuple(call.usage.prompt_tokens for call in calls),
        "completion": tuple(call.usage.completion_tokens for call in calls),
        "reasoning": tuple(call.usage.reasoning_tokens for call in calls),
    }.get(metric)
    if values is None:
        raise ValueError("The model-call token metric is not supported.")
    return (
        sum(value for value in values if value is not None)
        if all(value is not None for value in values)
        else None
    )


def _complete_cost_sum(calls: tuple[ModelCallRecord, ...]) -> Decimal | None:
    values = tuple(call.usage.cost_usd for call in calls)
    return (
        sum(
            (value for value in values if value is not None),
            start=Decimal(0),
        )
        if all(value is not None for value in values)
        else None
    )
