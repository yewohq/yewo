"""Procedure replay record invariants."""

from yewo.agent.replay.procedure.records import (
    ProcedureRepairEvidence,
    ProcedureRepairTarget,
)


def test_repair_evidence_tracks_model_calls_independently_from_actions() -> None:
    evidence = ProcedureRepairEvidence(
        target=ProcedureRepairTarget(
            expected_state_id="state_expected",
            resume_procedure_index=3,
            reason="Restore the expected replay state.",
        ),
        maximum_actions=8,
        action_count=1,
        model_call_count=2,
        reason="One model decision was refused before dispatch.",
    )

    assert evidence.action_count == 1
    assert evidence.model_call_count == 2
