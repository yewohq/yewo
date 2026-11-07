"""Infrastructure handling for exhausted visual-grounding providers."""

from pathlib import Path

import pytest

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionRefusal, ActionRefusalCode
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.runtime.no_progress import MAXIMUM_CONSECUTIVE_REFUSALS
from yewo.agent.runtime.no_progress import ActionNoProgressTracker
from yewo.agent.runtime.failure import AgentFailureSummary, ModelServiceUnavailableError
from yewo.grounding import GroundingService
from yewo.models import ModelCallDisposition

from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)
from grounding.support import StubModelClient, build_grounding_profile


def test_mixed_refusal_streak_is_not_classified_as_transient() -> None:
    screen = _observation("obs_1", "0000000000000000")
    attempted = proposal(ActionKind.WAIT, source_observation_id=screen.observation_id)
    ordinary = ActionRefusal(
        proposal_id=attempted.proposal_id,
        code=ActionRefusalCode.INVALID_INPUT,
        reason="The action input is invalid.",
    )
    transient = ActionRefusal(
        proposal_id=attempted.proposal_id,
        code=ActionRefusalCode.PROVIDER_UNAVAILABLE,
        reason="RateLimitError (HTTP 429)",
    )

    for refusals in ((ordinary,) * 4 + (transient,), (transient,) * 4 + (ordinary,)):
        tracker = ActionNoProgressTracker(ScreenComparisonPolicy())
        tracker.start(screen)
        for refusal in refusals[:-1]:
            assert tracker.record_refusal(attempted, screen, refusal) is None
        assert tracker.record_refusal(attempted, screen, refusals[-1]) is not None
        assert not tracker.refusal_streak_is_entirely_transient


def test_exhausted_grounding_provider_reaches_planner_as_a_refusal(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.TAP), proposal(ActionKind.DONE)])
    model = StubModelClient(
        None,
        ModelCallDisposition.PROVIDER_ERROR,
        "RateLimitError (HTTP 429)",
    )
    grounder = GroundingService(model, build_grounding_profile())
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        grounder=grounder,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert len(planner.requests) == 2
    assert len(model.requests) == 1
    refusal = result.steps[0].refusal
    assert refusal is not None
    assert refusal.code is ActionRefusalCode.PROVIDER_UNAVAILABLE
    history_refusal = planner.requests[1].history[-1].refusal
    assert history_refusal is not None
    assert history_refusal.code is ActionRefusalCode.PROVIDER_UNAVAILABLE


def test_repeated_grounding_provider_outage_is_an_infrastructure_terminal(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [proposal(ActionKind.TAP) for _ in range(MAXIMUM_CONSECUTIVE_REFUSALS)]
    )
    model = StubModelClient(
        None,
        ModelCallDisposition.PROVIDER_ERROR,
        "RateLimitError (HTTP 429)",
    )

    runtime = _runtime(
        tmp_path,
        planner,
        ObservationSequence([_observation("obs_1", "0000000000000000")]),
        RuntimeDevice(),
        grounder=GroundingService(model, build_grounding_profile()),
    )

    with pytest.raises(
        ModelServiceUnavailableError,
        match="model service remained unavailable",
    ) as raised:
        runtime.run(_task(), _mode_policy(), NeverInterrupted())

    assert len(planner.requests) == MAXIMUM_CONSECUTIVE_REFUSALS
    assert len(model.requests) == MAXIMUM_CONSECUTIVE_REFUSALS
    summary = raised.value.summary
    assert summary.usage.planning_requests == MAXIMUM_CONSECUTIVE_REFUSALS
    assert summary.usage.grounding_requests == MAXIMUM_CONSECUTIVE_REFUSALS
    assert summary.usage.dispatched_actions == 0
    assert summary.usage.steps == MAXIMUM_CONSECUTIVE_REFUSALS
    persisted = AgentFailureSummary.model_validate_json(
        (tmp_path / "agent/failure-summary.json").read_text()
    )
    assert persisted == summary


def test_repeated_invalid_grounding_concludes_as_application_deadlock(tmp_path: Path) -> None:
    planner = SequencePlanner(
        [proposal(ActionKind.TAP) for _ in range(MAXIMUM_CONSECUTIVE_REFUSALS)]
    )
    model = StubModelClient("not a coordinate")
    runtime = _runtime(
        tmp_path,
        planner,
        ObservationSequence([_observation("obs_1", "0000000000000000")]),
        RuntimeDevice(),
        grounder=GroundingService(model, build_grounding_profile()),
    )

    result = runtime.run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert len(result.steps) == MAXIMUM_CONSECUTIVE_REFUSALS
    assert all(
        step.refusal is not None and step.refusal.code is ActionRefusalCode.INVALID_GROUNDING
        for step in result.steps
    )
