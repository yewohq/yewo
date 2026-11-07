"""Typed and cross-phase screen-analysis capacity containment."""

import pytest

from agent.planning.support import StubPlannerModel
from observations.factories import build_observation
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysisManager,
    ModelScreenAnalyzer,
    ScreenAnalysisCapacityError,
    ScreenAnalysisFailureLedger,
)
from yewo.models import ModelCallDisposition


def test_reasoning_exhaustion_is_typed_and_shared_across_managers() -> None:
    model = StubPlannerModel(
        None,
        disposition=ModelCallDisposition.REASONING_EXHAUSTED,
    )
    source = ModelScreenAnalyzer(model, "screen-analyzer-primary")
    observation = build_observation()

    with pytest.raises(ScreenAnalysisCapacityError):
        source.analyze(observation)

    ledger = ScreenAnalysisFailureLedger()
    replay_manager = CurrentScreenAnalysisManager(source, failure_ledger=ledger)
    assert replay_manager.analyze(observation) is None
    assert replay_manager.capacity_failure_reason(observation) is not None

    fallback_manager = CurrentScreenAnalysisManager(source, failure_ledger=ledger)
    assert fallback_manager.required_model_requests(observation) == 0
    assert fallback_manager.analyze(observation) is None
    reason = fallback_manager.capacity_failure_reason(observation)
    assert reason is not None and reason.startswith("screen_analysis_capacity_exhausted:")
    assert len(model.requests) == 2
