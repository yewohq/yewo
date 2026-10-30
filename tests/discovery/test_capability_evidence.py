"""Who judged that a goal was satisfied, and what that judgement may claim.

V0 lets a model judge the semantics: whether this screen satisfies the
journey's goal. The mechanical half stays locally checked whoever judged.
The owner travels with the receipt, so a later reader separates a judged
capability from a measured one.
"""

import pytest
from pydantic import ValidationError

from yewo.discovery.contracts import DiscoveryBlocker
from yewo.discovery.controller import DiscoveryController
from yewo.discovery.runtime import CapabilityEvidenceReceipt

from discovery_runtime_fakes import (
    FirstCandidateProposer,
    SuccessfulRuntime,
    _run,
    _source_batch,
)


def test_capability_evidence_owner_cannot_be_changed_by_a_proposer():
    with pytest.raises(ValidationError):
        CapabilityEvidenceReceipt(
            evidence_owner="model",
            goal_id="goal-1",
            observation_id="observation-1",
            kind="observed",
            predicate_id="predicate-1",
        )


def test_a_model_judgement_still_cannot_invent_the_transition_it_claims(header, budgets):
    """The mechanical half stays locally checked whoever judged the semantics."""
    run = _run(header, budgets)
    goal_id = run.goals[0].goal_id
    runtime = SuccessfulRuntime(
        run.header.compatibility.setup_contract_sha256,
        _source_batch(goal_id),
        goal_id,
        evidence_owner="model_proposal",
        transition_identity="a-transition-the-controller-never-accepted",
    )

    result = DiscoveryController(run).run_to_terminal(
        runtime, FirstCandidateProposer()
    )

    assert result.run.terminal.blocker is DiscoveryBlocker.INFRASTRUCTURE
