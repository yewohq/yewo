"""Capture typed app-memory knowledge from canonical run evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from yewo.agent.actuation import ActionOutcomeDisposition
from yewo.agent.memory.knowledge import (
    ClaimReview,
    GroundingAssessment,
    ActionRiskKind,
    ActionRiskKnowledge,
    KnowledgeEvidenceReference,
    KnowledgeFinding,
    KnowledgeRule,
    KnowledgeScope,
    MemoryWritebackReceipt,
    RunOutcomeKnowledge,
    ScreenSummaryKnowledge,
    TransitionTimingKnowledge,
    TransitionTimingSample,
)
from yewo.agent.memory.records import AcceptedTransition, AppMemory, derived_memory_identifier
from yewo.agent.memory.screens import observation_matches_screen
from yewo.agent.memory.writeback import MemoryDelta, MemoryWritebackSkip
from yewo.agent.memory.control_capture import capture_controls
from yewo.agent.recovery.advisor import RecoveryAssessment
from yewo.agent.replay.reconciliation.records import ReconcilerResponse
from yewo.agent.replay.procedure.records import ProcedureReplayResult
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.runtime.records import AgentResult, AgentStep
from yewo.agent.screen_analysis import CurrentScreenAnalysis
from yewo.artifacts import RecordIdentifier
from yewo.devices.diagnostics import ApplicationFailureKind
from yewo.observations import Observation


def build_memory_delta(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    agent_result: AgentResult,
    observations: Mapping[RecordIdentifier, Observation],
    applied_at: datetime,
    *,
    screen_analyses: tuple[CurrentScreenAnalysis, ...] = (),
    procedure_attempt_id: RecordIdentifier | None = None,
    grounding_assessments: tuple[GroundingAssessment, ...] = (),
    recovery_assessments: Sequence[RecoveryAssessment] = (),
    reconciler_response: ReconcilerResponse | None = None,
    verification: GoalVerification | None = None,
    procedure_replay: ProcedureReplayResult | None = None,
) -> MemoryDelta:
    """Build one deterministic batch from canonical run evidence."""
    skipped_records: list[MemoryWritebackSkip] = []
    action_risks = _action_risks(
        memory,
        source_run_id,
        agent_result.steps,
        observations,
        procedure_attempt_id,
        skipped_records,
    )
    claim_reviews = _claim_reviews(
        memory,
        source_run_id,
        agent_result.steps,
        reconciler_response,
        procedure_attempt_id,
        skipped_records,
    )
    return MemoryDelta(
        receipt=MemoryWritebackReceipt(
            writeback_id=derived_memory_identifier(
                "memory-writeback", source_run_id, procedure_attempt_id or "ordinary-run"
            ),
            source_run_id=source_run_id,
            procedure_attempt_id=procedure_attempt_id,
            applied_at=applied_at,
        ),
        grounding_assessments=grounding_assessments,
        action_risks=action_risks,
        rules=_rules(
            memory,
            source_run_id,
            agent_result.steps,
            observations,
            recovery_assessments,
            action_risks,
            procedure_attempt_id,
            skipped_records,
        ),
        findings=(
            *_findings(source_run_id, reconciler_response, procedure_attempt_id),
            *_procedure_findings(source_run_id, procedure_replay, verification),
        ),
        claim_reviews=claim_reviews,
        timing=_timing(
            memory,
            source_run_id,
            agent_result.steps,
            observations,
            procedure_attempt_id,
            skipped_records,
        ),
        run_outcomes=_run_outcomes(
            memory,
            source_run_id,
            verification,
            procedure_attempt_id,
            skipped_records,
        ),
        screen_summaries=_screen_summaries(
            memory,
            source_run_id,
            screen_analyses,
            observations,
            procedure_attempt_id,
        ),
        controls=capture_controls(
            memory,
            source_run_id,
            agent_result,
            screen_analyses,
            observations,
            procedure_attempt_id,
            skipped_records,
        ),
        skipped_records=tuple(skipped_records),
    )


def _action_risks(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    steps: tuple[AgentStep, ...],
    observations: Mapping[RecordIdentifier, Observation],
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[ActionRiskKnowledge, ...]:
    action_risks: list[ActionRiskKnowledge] = []
    for step in steps:
        failure = step.application_failure
        if failure is None:
            continue
        observation = observations.get(step.source_observation_id)
        screen_id = _screen_id(memory, observation) if observation is not None else None
        if screen_id is None:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="action_risk",
                    source_step_id=step.step_id,
                    reason="No unique retained source screen matched the failure evidence.",
                )
            )
            continue
        kind = _action_risk_kind(failure.failure_kinds)
        action_risks.append(
            ActionRiskKnowledge(
                action_risk_id=derived_memory_identifier(
                    "action_risk",
                    screen_id,
                    step.proposal.action.value,
                    step.proposal.target_expression or "none",
                    kind.value,
                ),
                screen_id=screen_id,
                action=step.proposal.action,
                target_description=step.proposal.target_expression,
                kind=kind,
                seen_count=1,
                recovered_count=int(failure.recovery_succeeded),
                sources=(_reference(source_run_id, step, procedure_attempt_id),),
            )
        )
    return tuple(action_risks)


def _rules(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    steps: tuple[AgentStep, ...],
    observations: Mapping[RecordIdentifier, Observation],
    assessments: Sequence[RecoveryAssessment],
    action_risks: tuple[ActionRiskKnowledge, ...],
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[KnowledgeRule, ...]:
    steps_by_proposal_id = {step.proposal.proposal_id: step for step in steps}
    action_risks_by_scope = {
        (item.screen_id, item.action, item.target_description): item for item in action_risks
    }
    rules: list[KnowledgeRule] = []
    for assessment in assessments:
        step = steps_by_proposal_id.get(assessment.failed_proposal_id)
        if step is None:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="rule",
                    reason="Recovery guidance did not match a retained run step.",
                )
            )
            continue
        observation = observations.get(step.source_observation_id)
        screen_id = _screen_id(memory, observation) if observation is not None else None
        if screen_id is None:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="rule",
                    source_step_id=step.step_id,
                    reason="No unique retained source screen matched the recovery guidance.",
                )
            )
            continue
        scope = KnowledgeScope(
            screen_id=screen_id,
            action=step.proposal.action,
            target_description=step.proposal.target_expression,
        )
        action_risk = action_risks_by_scope.get(
            (screen_id, step.proposal.action, step.proposal.target_expression)
        )
        for guidance in assessment.planner_guidance:
            rules.append(
                KnowledgeRule(
                    rule_id=derived_memory_identifier(
                        "knowledge-rule", _normalized_text(guidance), scope.model_dump_json()
                    ),
                    text=guidance,
                    scope=scope,
                    linked_action_risk_id=action_risk.action_risk_id
                    if action_risk is not None
                    else None,
                    sources=(_reference(source_run_id, step, procedure_attempt_id),),
                )
            )
    return tuple(rules)


def _findings(
    source_run_id: RecordIdentifier,
    response: ReconcilerResponse | None,
    procedure_attempt_id: RecordIdentifier | None,
) -> tuple[KnowledgeFinding, ...]:
    if response is None:
        return ()
    source = KnowledgeEvidenceReference(
        run_id=source_run_id, procedure_attempt_id=procedure_attempt_id
    )
    texts = tuple(decision.why for decision in response.decisions) + response.findings
    return tuple(
        KnowledgeFinding(
            finding_id=derived_memory_identifier("knowledge-finding", _normalized_text(text)),
            text=text,
            sources=(source,),
        )
        for text in dict.fromkeys(texts)
    )


def _claim_reviews(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    steps: tuple[AgentStep, ...],
    response: ReconcilerResponse | None,
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[ClaimReview, ...]:
    if response is None:
        return ()
    claim_ids = {
        *(item.rule_id for item in memory.app_knowledge.rules),
        *(item.finding_id for item in memory.app_knowledge.findings),
    }
    steps_by_id = {item.step_id: item for item in steps}
    reviews: list[ClaimReview] = []
    for verdict in response.claim_verdicts:
        selected_steps = tuple(steps_by_id.get(step_id) for step_id in verdict.source_step_ids)
        if any(step is None for step in selected_steps):
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="claim",
                    reason="A claim verdict referenced an unavailable source step.",
                )
            )
            continue
        references = tuple(
            _reference(source_run_id, step, procedure_attempt_id)
            for step in selected_steps
            if step is not None
        )
        if verdict.claim_id not in claim_ids:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="claim",
                    reason="A claim verdict referenced an unavailable retained claim.",
                )
            )
            continue
        reviews.append(
            ClaimReview(
                review_id=derived_memory_identifier(
                    "claim-review", source_run_id, verdict.claim_id, verdict.status
                ),
                claim_id=verdict.claim_id,
                proposed_status=verdict.status,
                reason=verdict.reason,
                sources=references,
            )
        )
    return tuple(reviews)


def _procedure_findings(
    source_run_id: RecordIdentifier,
    replay: ProcedureReplayResult | None,
    verification: GoalVerification | None,
) -> tuple[KnowledgeFinding, ...]:
    if (
        replay is None
        or verification is None
        or not verification.satisfied
        or replay.completed_step_count >= replay.procedure_step_count
    ):
        return ()
    text = (
        "The task was verified after fewer compiled expectations than the retained procedure "
        "contains. Review whether the procedure can be shortened safely."
    )
    return (
        KnowledgeFinding(
            finding_id=derived_memory_identifier(
                "knowledge-finding",
                "procedure-shortfall",
                replay.procedure_id,
            ),
            text=text,
            sources=(
                KnowledgeEvidenceReference(
                    run_id=source_run_id,
                    procedure_attempt_id=replay.setup_id,
                ),
            ),
        ),
    )


def _run_outcomes(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    verification: GoalVerification | None,
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[RunOutcomeKnowledge, ...]:
    if verification is None or verification.observed_app_state is None:
        return ()
    outcome = RunOutcomeKnowledge(
        run_id=source_run_id,
        satisfied=verification.satisfied,
        observed_app_state=verification.observed_app_state,
        source=KnowledgeEvidenceReference(
            run_id=source_run_id,
            observation_id=verification.final_observation_id,
            procedure_attempt_id=procedure_attempt_id,
        ),
    )
    retained = next(
        (item for item in memory.app_knowledge.run_outcomes if item.run_id == source_run_id),
        None,
    )
    if retained is None:
        return (outcome,)
    if retained != outcome:
        skipped_records.append(
            MemoryWritebackSkip(
                record_kind="outcome",
                reason="The run outcome conflicts with the retained outcome for this run.",
            )
        )
    return ()


def _screen_summaries(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    screen_analyses: tuple[CurrentScreenAnalysis, ...],
    observations: Mapping[RecordIdentifier, Observation],
    procedure_attempt_id: RecordIdentifier | None,
) -> tuple[ScreenSummaryKnowledge, ...]:
    summaries: dict[RecordIdentifier, ScreenSummaryKnowledge] = {}
    for analysis in screen_analyses:
        observation = observations.get(analysis.observation_id)
        if observation is None:
            continue
        screen_id = _screen_id(memory, observation)
        if screen_id is None:
            continue
        summaries[screen_id] = ScreenSummaryKnowledge(
            screen_id=screen_id,
            text=analysis.screen_summary,
            sources=(
                KnowledgeEvidenceReference(
                    run_id=source_run_id,
                    observation_id=analysis.observation_id,
                    procedure_attempt_id=procedure_attempt_id,
                ),
            ),
        )
    return tuple(summaries.values())


def _timing(
    memory: AppMemory,
    source_run_id: RecordIdentifier,
    steps: tuple[AgentStep, ...],
    observations: Mapping[RecordIdentifier, Observation],
    procedure_attempt_id: RecordIdentifier | None,
    skipped_records: list[MemoryWritebackSkip],
) -> tuple[TransitionTimingKnowledge, ...]:
    timing: list[TransitionTimingKnowledge] = []
    for step in steps:
        if not _is_successful_settled_step(step):
            continue
        settled_observation_id = step.settled_observation_id
        if settled_observation_id is None:
            raise AssertionError("A successful settled step must name its observation.")
        source = observations.get(step.source_observation_id)
        destination = observations.get(settled_observation_id)
        if source is None or destination is None:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="timing",
                    source_step_id=step.step_id,
                    reason="Transition timing observations were unavailable.",
                )
            )
            continue
        transition = _transition(memory, step, source, destination)
        if transition is None:
            continue
        receipt = step.receipt
        if receipt is None:
            raise AssertionError("A successful settled step must contain a dispatch receipt.")
        measurement = "dispatch_to_settled_v1"
        duration_ms = (destination.captured_at - receipt.dispatched_at).total_seconds() * 1_000
        if duration_ms < 0:
            skipped_records.append(
                MemoryWritebackSkip(
                    record_kind="timing",
                    source_step_id=step.step_id,
                    reason="Transition timing was negative and was not retained.",
                )
            )
            continue
        timing.append(
            TransitionTimingKnowledge(
                transition_id=transition.transition_id,
                samples=(
                    TransitionTimingSample(
                        sample_id=derived_memory_identifier(
                            "transition-timing",
                            source_run_id,
                            step.step_id,
                            transition.transition_id,
                        ),
                        measurement=measurement,
                        duration_ms=duration_ms,
                        source=_reference(source_run_id, step, procedure_attempt_id),
                    ),
                ),
            )
        )
    return tuple(timing)


def _is_successful_settled_step(step: AgentStep) -> bool:
    return (
        step.outcome is not None
        and step.outcome.disposition is ActionOutcomeDisposition.OBSERVED
        and step.application_failure is None
        and step.settled_observation_id is not None
    )


def _transition(
    memory: AppMemory,
    step: AgentStep,
    source: Observation,
    destination: Observation,
) -> AcceptedTransition | None:
    direct = next(
        (item for item in memory.transitions if item.source_step_id == step.step_id), None
    )
    if direct is not None:
        return direct
    source_screen_id = _screen_id(memory, source)
    destination_screen_id = _screen_id(memory, destination)
    matches = tuple(
        item
        for item in memory.transitions
        if item.source_screen_id == source_screen_id
        and item.destination_screen_id == destination_screen_id
        and item.action.action is step.proposal.action
        and item.action.target_description == step.proposal.target_expression
        and item.action.destination_description == step.proposal.destination_expression
    )
    return matches[0] if len(matches) == 1 else None


def _screen_id(memory: AppMemory, observation: Observation) -> RecordIdentifier | None:
    matches = tuple(
        screen.screen_id
        for screen in memory.screens
        if observation_matches_screen(observation, screen, memory.compatibility.screen_comparison)
    )
    return matches[0] if len(matches) == 1 else None


def _reference(
    source_run_id: RecordIdentifier,
    step: AgentStep,
    procedure_attempt_id: RecordIdentifier | None,
) -> KnowledgeEvidenceReference:
    return KnowledgeEvidenceReference(
        run_id=source_run_id,
        step_id=step.step_id,
        observation_id=step.source_observation_id,
        procedure_attempt_id=procedure_attempt_id,
    )


def _action_risk_kind(failure_kinds: tuple[ApplicationFailureKind, ...]) -> ActionRiskKind:
    kinds = frozenset(failure_kinds)
    if ApplicationFailureKind.ANR in kinds:
        return ActionRiskKind.ANR
    if ApplicationFailureKind.CRASH in kinds:
        return ActionRiskKind.CRASH
    return ActionRiskKind.APPLICATION_EXIT


def _normalized_text(text: str) -> str:
    return " ".join(text.casefold().split())
