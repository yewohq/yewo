"""Bounded AppMemory knowledge-retention tests."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from yewo.agent.memory.knowledge import (
    AppKnowledge,
    ClaimReview,
    GroundingAssessment,
    GroundingRole,
    KnowledgeEvidenceReference,
    KnowledgeRule,
    MemoryWritebackReceipt,
    RunOutcomeKnowledge,
)
from yewo.agent.memory.records import AppMemory
from yewo.agent.memory.writeback import (
    MemoryDelta,
    MemoryDeltaDisposition,
    apply_memory_delta,
    apply_memory_delta_result,
)

from agent.memory.test_store import _memory


_NOW = datetime(2026, 11, 3, tzinfo=timezone.utc)


def _reference(run_id: str) -> KnowledgeEvidenceReference:
    return KnowledgeEvidenceReference(
        run_id=run_id,
        step_id="replay_step_1",
        observation_id="obs_1",
        procedure_attempt_id="attempt_1",
    )


def _grounding(memory: AppMemory, run_id: str) -> GroundingAssessment:
    transition = memory.transitions[0]
    return GroundingAssessment(
        assessment_id="grounding_1",
        transition_id=transition.transition_id,
        role=GroundingRole.TARGET,
        source_observation_id=transition.source_observation_id,
        successful_use_count=1,
        last_success_at=_NOW,
        sources=(_reference(run_id),),
    )


def test_grounding_reinforcement_retains_bounded_recent_provenance(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    updated = memory
    for index in range(2, 203):
        run_id = f"run_{index}"
        receipt = MemoryWritebackReceipt(
            writeback_id=f"writeback_{index}",
            source_run_id=run_id,
            procedure_attempt_id=f"attempt_{index}",
            applied_at=_NOW + timedelta(seconds=index),
        )
        updated = apply_memory_delta(
            updated,
            MemoryDelta(receipt=receipt, grounding_assessments=(_grounding(memory, run_id),)),
        )

    assessment = updated.app_knowledge.grounding_assessments[0]
    assert assessment.successful_use_count == 201
    assert len(assessment.sources) == 200
    assert assessment.sources[0].run_id == "run_3"
    assert assessment.sources[-1].run_id == "run_202"


def test_writeback_retention_floor_uses_evicted_timestamp_not_list_position(
    tmp_path: Path,
) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    receipts = tuple(
        MemoryWritebackReceipt(
            writeback_id=f"writeback_{index}",
            source_run_id=f"run_{index}",
            applied_at=_NOW + timedelta(seconds=(1_000 if index == 0 else index)),
        )
        for index in range(1_000)
    )
    memory = memory.model_copy(
        update={
            "source_run_ids": (
                *memory.source_run_ids,
                *(receipt.source_run_id for receipt in receipts),
            ),
            "app_knowledge": AppKnowledge(writebacks=receipts),
        }
    )
    retained = apply_memory_delta_result(
        memory,
        MemoryDelta(
            receipt=MemoryWritebackReceipt(
                writeback_id="writeback_new",
                source_run_id="run_new",
                applied_at=_NOW + timedelta(seconds=1_001),
            )
        ),
    ).memory
    stale = MemoryDelta(
        receipt=MemoryWritebackReceipt(
            writeback_id="writeback_stale",
            source_run_id="run_stale",
            applied_at=_NOW + timedelta(seconds=500),
        )
    )

    application = apply_memory_delta_result(retained, stale)

    assert retained.app_knowledge.writeback_floor == _NOW + timedelta(seconds=1_000)
    assert application.disposition is MemoryDeltaDisposition.STALE
    assert application.memory == retained


def test_writeback_prunes_unreferenced_source_run_identifiers(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "source_run_ids": (*memory.source_run_ids, "orphan_run"),
        }
    )

    updated = apply_memory_delta(
        memory,
        MemoryDelta(
            receipt=MemoryWritebackReceipt(
                writeback_id="writeback_2",
                source_run_id="run_2",
                applied_at=_NOW,
            )
        ),
    )

    assert updated.source_run_ids == ("run_1", "run_2")


def test_run_outcome_retention_evicts_the_oldest_run_record(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    outcomes = tuple(
        RunOutcomeKnowledge(
            run_id=f"run_{index}",
            satisfied=True,
            observed_app_state="The controlled application reached its final state.",
            source=_reference(f"run_{index}"),
        )
        for index in range(2, 1_002)
    )
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "source_run_ids": (
                *memory.source_run_ids,
                *(outcome.run_id for outcome in outcomes),
            ),
            "app_knowledge": AppKnowledge(run_outcomes=outcomes),
        }
    )
    newest = RunOutcomeKnowledge(
        run_id="run_new",
        satisfied=True,
        observed_app_state="The newest controlled run reached its final state.",
        source=_reference("run_new"),
    )

    updated = apply_memory_delta(
        memory,
        MemoryDelta(
            receipt=MemoryWritebackReceipt(
                writeback_id="writeback_new",
                source_run_id="run_new",
                applied_at=_NOW,
            ),
            run_outcomes=(newest,),
        ),
    )

    assert len(updated.app_knowledge.run_outcomes) == 1_000
    assert updated.app_knowledge.run_outcomes[0].run_id == "run_3"
    assert updated.app_knowledge.run_outcomes[-1] == newest
    assert "run_2" not in updated.source_run_ids


def test_claim_review_retention_evicts_the_oldest_review(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    rule = KnowledgeRule(
        rule_id="rule_1",
        text="The submit action succeeds after every required field is filled.",
        sources=(_reference("run_1"),),
    )
    reviews = tuple(
        ClaimReview(
            review_id=f"review_{index}",
            claim_id=rule.rule_id,
            proposed_status="confirmed",
            reason="The current run supplied evidence for this review proposal.",
            sources=(_reference(f"run_{index}"),),
        )
        for index in range(2, 5_002)
    )
    memory = AppMemory.model_validate(
        {
            **memory.model_dump(mode="python"),
            "source_run_ids": (
                *memory.source_run_ids,
                *(review.sources[0].run_id for review in reviews),
            ),
            "app_knowledge": AppKnowledge(rules=(rule,), claim_reviews=reviews),
        }
    )
    newest = ClaimReview(
        review_id="review_new",
        claim_id=rule.rule_id,
        proposed_status="contradicted",
        reason="The newest run supplied contrary evidence for review.",
        sources=(_reference("run_new"),),
    )

    updated = apply_memory_delta(
        memory,
        MemoryDelta(
            receipt=MemoryWritebackReceipt(
                writeback_id="writeback_new",
                source_run_id="run_new",
                applied_at=_NOW,
            ),
            claim_reviews=(newest,),
        ),
    )

    assert len(updated.app_knowledge.claim_reviews) == 5_000
    assert updated.app_knowledge.claim_reviews[0].review_id == "review_3"
    assert updated.app_knowledge.claim_reviews[-1] == newest
    assert "run_2" not in updated.source_run_ids
