"""App-memory knowledge contracts and idempotent write-back tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from yewo.agent import AppMemoryStore
from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.actuation import ActionKind, ActionOutcomeDisposition
from yewo.agent.memory.knowledge_capture import build_memory_delta
from yewo.agent.memory.knowledge import (
    AppKnowledge,
    GroundingAssessment,
    GroundingAssessmentStatus,
    GroundingRole,
    HazardKind,
    HazardKnowledge,
    KnowledgeEvidenceReference,
    KnowledgeRule,
    MemoryWritebackReceipt,
    TransitionTimingKnowledge,
    TransitionTimingSample,
)
from yewo.agent.memory.records import AppMemory
from yewo.agent.memory.writeback import (
    MemoryDelta,
    apply_memory_delta,
)
from yewo.agent.recovery.advisor import RecoveryAssessment, RecoveryDisposition
from yewo.agent.planning.goal_verifier import GoalVerification
from yewo.agent.replay.reconciliation_records import (
    ReconcilerClaimVerdict,
    ReconcilerResponse,
)
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory


_NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


def _reference(run_id: str = "run_2") -> KnowledgeEvidenceReference:
    return KnowledgeEvidenceReference(
        run_id=run_id,
        step_id="replay_step_1",
        observation_id="obs_1",
        procedure_attempt_id="attempt_1",
    )


def _receipt(writeback_id: str = "writeback_1") -> MemoryWritebackReceipt:
    return MemoryWritebackReceipt(
        writeback_id=writeback_id,
        source_run_id="run_2",
        procedure_attempt_id="attempt_1",
        applied_at=_NOW,
    )


def _grounding(memory: AppMemory) -> GroundingAssessment:
    transition = memory.transitions[0]
    return GroundingAssessment(
        assessment_id="grounding_1",
        transition_id=transition.transition_id,
        role=GroundingRole.TARGET,
        source_observation_id=transition.source_observation_id,
        successful_use_count=1,
        last_success_at=_NOW,
        sources=(_reference(),),
    )


def test_memory_delta_is_idempotent(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    delta = MemoryDelta(
        receipt=_receipt(),
        grounding_assessments=(_grounding(memory),),
    )

    updated = apply_memory_delta(memory, delta)
    repeated = apply_memory_delta(updated, delta)

    assert repeated == updated
    assert updated.source_run_ids == ("run_1", "run_2")
    assert updated.app_knowledge.grounding_assessments[0].successful_use_count == 1
    assert updated.app_knowledge.writebacks == (delta.receipt,)


def test_grounding_invalidation_wins_without_erasing_success_evidence(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    reinforced = apply_memory_delta(
        memory,
        MemoryDelta(receipt=_receipt(), grounding_assessments=(_grounding(memory),)),
    )
    invalidation_source = _reference("run_3")
    invalidated = _grounding(memory).model_copy(
        update={
            "status": GroundingAssessmentStatus.INVALIDATED,
            "successful_use_count": 0,
            "last_success_at": None,
            "invalidation_reason": "A corrected grounding succeeded at a different point.",
            "sources": (invalidation_source,),
        }
    )
    delta = MemoryDelta(
        receipt=MemoryWritebackReceipt(
            writeback_id="writeback_2",
            source_run_id="run_3",
            procedure_attempt_id="attempt_2",
            applied_at=_NOW + timedelta(seconds=1),
        ),
        grounding_assessments=(invalidated,),
    )

    updated = apply_memory_delta(reinforced, delta)
    assessment = updated.app_knowledge.grounding_assessments[0]

    assert assessment.status is GroundingAssessmentStatus.INVALIDATED
    assert assessment.successful_use_count == 1
    assert assessment.last_success_at == _NOW
    assert assessment.sources == (_reference(), invalidation_source)


def test_knowledge_rejects_references_outside_the_memory_graph(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    hazard = HazardKnowledge(
        hazard_id="hazard_1",
        screen_id="unknown_screen",
        action=ActionKind.TAP,
        kind=HazardKind.CRASH,
        seen_count=1,
        sources=(_reference(),),
    )
    payload = memory.model_dump()
    payload.update(
        {
            "source_run_ids": ("run_1", "run_2"),
            "app_knowledge": AppKnowledge(hazards=(hazard,)),
        }
    )

    with pytest.raises(ValueError, match="hazard.*retained screen"):
        AppMemory.model_validate(payload)


def test_reconciler_verdict_is_retained_without_promoting_claim_trust(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    rule = KnowledgeRule(
        rule_id="rule_1",
        text="The submit action succeeds after every required field is filled.",
        sources=(_reference("run_1"),),
    )
    memory = memory.model_copy(update={"app_knowledge": AppKnowledge(rules=(rule,))})
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    evidence = agent_result(before, after)
    source_step = evidence.steps[0]
    response = ReconcilerResponse(
        request_id="request_verdict",
        procedure=(1,),
        milestones={},
        claim_verdicts=(
            ReconcilerClaimVerdict(
                claim_id=rule.rule_id,
                status="confirmed",
                source_step_ids=(source_step.step_id,),
                reason="The recorded transition directly demonstrated the predicted effect.",
            ),
        ),
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        evidence,
        {before.observation_id: before, after.observation_id: after},
        _NOW,
        reconciler_response=response,
    )
    updated = apply_memory_delta(memory, delta)

    retained = updated.app_knowledge.rules[0]
    assert retained == rule
    assert updated.app_knowledge.claim_reviews[0].claim_id == rule.rule_id
    assert updated.app_knowledge.claim_reviews[0].reason == (
        "The recorded transition directly demonstrated the predicted effect."
    )
    assert updated.app_knowledge.claim_reviews[0].sources[0].step_id == source_step.step_id


def test_store_update_applies_delta_under_one_lock(tmp_path: Path) -> None:
    source_root = tmp_path / "run"
    compatibility, memory = _memory(source_root)
    store = AppMemoryStore(tmp_path / "shared")
    store.save(memory, source_root)
    transition = memory.transitions[0]
    timing = TransitionTimingKnowledge(
        transition_id=transition.transition_id,
        samples=(
            TransitionTimingSample(
                sample_id="timing_1",
                duration_ms=840,
                source=_reference(),
            ),
        ),
    )
    delta = MemoryDelta(receipt=_receipt(), timing=(timing,))

    persisted = store.update(
        compatibility,
        source_root,
        lambda current: apply_memory_delta(current, delta),
    )
    loaded = store.load(compatibility)

    assert loaded == persisted
    assert loaded is not None
    assert loaded.app_knowledge.timing == (timing,)


def test_memory_rejects_claim_sources_from_an_unretained_run(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    rule = KnowledgeRule(
        rule_id="rule_1",
        text="Use the visible editor row.",
        sources=(_reference("run_missing"),),
    )
    payload = memory.model_dump()
    payload["app_knowledge"] = AppKnowledge(rules=(rule,))

    with pytest.raises(ValueError, match="source.*retained memory run"):
        AppMemory.model_validate(payload)


def test_capture_records_only_observed_settled_transition_timing(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )

    captured = build_memory_delta(
        memory,
        "run_2",
        agent_result(before, after),
        {"obs_1": before, "obs_2": after},
        _NOW,
    )
    unavailable = build_memory_delta(
        memory,
        "run_3",
        agent_result(
            before,
            after,
            disposition=ActionOutcomeDisposition.UNAVAILABLE,
        ),
        {"obs_1": before, "obs_2": after},
        _NOW,
    )

    assert captured.timing[0].transition_id == memory.transitions[0].transition_id
    assert captured.timing[0].samples[0].duration_ms == 1_000
    assert captured.timing[0].samples[0].measurement == "source_to_settled_v1"
    assert unavailable.timing == ()


def test_capture_measures_settlement_after_dispatch_when_timestamp_exists(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    result = agent_result(before, after)
    step = result.steps[0]
    assert step.receipt is not None
    timed_step = step.model_copy(
        update={
            "receipt": step.receipt.model_copy(
                update={"dispatched_at": before.captured_at + timedelta(milliseconds=600)}
            )
        }
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        result.model_copy(update={"steps": (timed_step, *result.steps[1:])}),
        {"obs_1": before, "obs_2": after},
        _NOW,
    )

    sample = delta.timing[0].samples[0]
    assert sample.measurement == "dispatch_to_settled_v2"
    assert sample.duration_ms == 400


def test_capture_labels_reconciler_output_as_unconfirmed_findings(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    response = ReconcilerResponse(
        request_id="request_2",
        procedure=(1,),
        milestones={},
        findings=("The confirmation screen follows the submit action.",),
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        agent_result(before, after),
        {"obs_1": before, "obs_2": after},
        _NOW,
        reconciler_response=response,
    )

    assert len(delta.findings) == 1
    assert delta.findings[0].sources[0].run_id == "run_2"


def test_capture_records_auditor_observed_application_state(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    verification = GoalVerification(
        verification_id="verification_1",
        objective_id="objective_1",
        final_observation_id=after.observation_id,
        satisfied=True,
        reason="The final state proves completion.",
        observed_app_state="The saved account appears in the account list.",
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        agent_result(before, after),
        {"obs_1": before, "obs_2": after},
        _NOW,
        verification=verification,
    )

    assert len(delta.run_outcomes) == 1
    assert delta.run_outcomes[0].observed_app_state.startswith("The saved account")


def test_capture_records_crash_hazard_and_unconfirmed_recovery_rule(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )
    evidence = agent_result(before, after)
    failed_step = evidence.steps[0].model_copy(
        update={
            "application_failure": ActionApplicationFailure(
                classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
                failure_kinds=(ApplicationFailureKind.CRASH,),
                summary="The application crashed after submission.",
                recovery_succeeded=True,
            )
        }
    )
    failed_result = evidence.model_copy(update={"steps": (failed_step, *evidence.steps[1:])})
    recovery = RecoveryAssessment(
        assessment_id="recovery_1",
        failed_proposal_id=failed_step.proposal.proposal_id,
        pre_failure_observation_id=before.observation_id,
        recovery_observation_id=after.observation_id,
        disposition=RecoveryDisposition.STOP,
        reason="The unchanged action must not be repeated.",
        planner_guidance=("Complete the required fields before submission.",),
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        failed_result,
        {"obs_1": before, "obs_2": after},
        _NOW,
        recovery_assessments=(recovery,),
    )

    assert delta.hazards[0].kind is HazardKind.CRASH
    assert delta.hazards[0].recovered_count == 1
    assert delta.rules[0].linked_hazard_id == delta.hazards[0].hazard_id
    assert delta.timing == ()


def test_unmatched_hazard_is_skipped_without_discarding_other_records(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("unknown_source", "2222222222222222")
    after = observation(
        "unknown_destination",
        "3333333333333333",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    evidence = agent_result(before, after)
    step = evidence.steps[0]
    failed_step = step.model_copy(
        update={
            "application_failure": ActionApplicationFailure(
                classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
                failure_kinds=(ApplicationFailureKind.CRASH,),
                summary="The application crashed.",
            )
        }
    )
    response = ReconcilerResponse(
        request_id="request_skip",
        procedure=(),
        milestones={},
        findings=("Retain this independent finding.",),
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        evidence.model_copy(update={"steps": (failed_step, *evidence.steps[1:])}),
        {before.observation_id: before, after.observation_id: after},
        _NOW,
        reconciler_response=response,
    )

    assert delta.hazards == ()
    assert len(delta.findings) == 1
    assert len(delta.skipped_records) == 1
    assert delta.skipped_records[0].record_kind == "hazard"
