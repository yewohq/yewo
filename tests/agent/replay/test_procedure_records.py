"""Procedure replay record invariants."""

import pytest
from pydantic import ValidationError

from yewo.agent.replay.procedure.records import (
    ProcedureRepairEvidence,
    ProcedureRepairTarget,
)
from yewo.agent.replay.reconciliation.records import EvidenceState

from agent.replay.support import observations


def test_evidence_state_representative_must_belong_to_the_state() -> None:
    presentation = observations()[0].presentation_key

    with pytest.raises(ValidationError, match="representative must belong"):
        EvidenceState(
            state_id="state-1",
            representative_observation_id="observation-missing",
            representative_screenshot=observations()[0].redacted_screenshot,
            observation_ids=("observation-1",),
            presentations=(presentation,),
        )


def test_repair_evidence_tracks_model_calls_independently_from_actions() -> None:
    evidence = ProcedureRepairEvidence(
        target=ProcedureRepairTarget(
            expected_state_id="state_expected",
            resume_procedure_index=3,
            presentation=observations()[0].presentation_key,
            reason="Restore the expected replay state.",
        ),
        maximum_actions=8,
        action_count=1,
        model_call_count=2,
        reason="One model decision was refused before dispatch.",
    )

    assert evidence.action_count == 1
    assert evidence.model_call_count == 2
