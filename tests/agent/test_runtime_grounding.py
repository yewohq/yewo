"""Grounding failures at the Device Agent runtime boundary."""

from pathlib import Path

import pytest

from yewo.grounding import GroundingInputError, GroundingRequest, GroundingResult

from agent.actuation.support import proposal
from agent.test_runtime import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _stored_steps,
    _task,
)
from grounding.support import FailingGrounder


class InvalidRequestGrounder:
    """Reject a request while the runtime calculates its model-call cost."""

    def required_model_requests(self, request: GroundingRequest) -> int:
        del request
        raise GroundingInputError("The configured profile rejects this operation.")

    def ground(self, request: GroundingRequest) -> GroundingResult:
        del request
        raise AssertionError("An invalid grounding request must not reach grounding.")


def test_grounding_failure_preserves_the_proposal_as_a_partial_step(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal()])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    with pytest.raises(RuntimeError, match="grounding failed"):
        _runtime(
            tmp_path,
            planner,
            observations,
            RuntimeDevice(),
            grounder=FailingGrounder(),
        ).run(_task(), _mode_policy(), NeverInterrupted())
    stored = _stored_steps(tmp_path)
    stored_proposal = stored[0]["proposal"]
    assert isinstance(stored_proposal, dict)
    assert stored_proposal["action"] == "tap"
    assert stored[0]["terminal_reason"] == "runtime_failure"
    assert stored[0]["authorization"] is None


def test_grounding_request_validation_failure_preserves_the_partial_step(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner([proposal()])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    with pytest.raises(GroundingInputError, match="rejects this operation"):
        _runtime(
            tmp_path,
            planner,
            observations,
            RuntimeDevice(),
            grounder=InvalidRequestGrounder(),
        ).run(_task(), _mode_policy(), NeverInterrupted())

    stored = _stored_steps(tmp_path)
    assert stored[0]["terminal_reason"] == "runtime_failure"
    assert stored[0]["target_grounding"] is None
