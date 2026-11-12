"""Runtime integration for bounded current-screen analysis."""

from pathlib import Path
from typing import cast

from yewo.agent.actuation import ActionKind, ActionOutcome, DispatchDisposition
from yewo.observations import ScreenComparisonPolicy
from yewo.agent.recovery.runtime import RecoveryRunState
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    CurrentScreenAnalysisManager,
    ScreenAnalysisCapacityError,
)
from yewo.agent.screen_analysis.runtime import (
    ScreenAnalysisBudget,
    ScreenAnalysisRunState,
)
from yewo.observations import Observation

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


class _Analyzer:
    def __init__(self) -> None:
        self.calls = 0

    def required_model_requests(self, observation: Observation) -> int:
        del observation
        return 1

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        self.calls += 1
        return CurrentScreenAnalysis(
            analysis_id=f"analysis-{self.calls}",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible application screen.",
            elements=(),
        )


class _AnalysisOptionalPlanner(SequencePlanner):
    def requires_screen_analysis(self, observation: Observation) -> bool:
        del observation
        return False


class _ObservationAwareAnalysisSource:
    def requires_screen_analysis(self, observation: Observation) -> bool:
        return observation.observation_id != "obs_rejoined"


class _CapacityExhaustedAnalyzer(_Analyzer):
    def capacity_identity(self) -> tuple[str, str]:
        return "screen-analyzer-primary", "screen-analysis-hypothesis-v1"

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        del observation
        self.calls += 1
        raise ScreenAnalysisCapacityError(*self.capacity_identity())


class _RecordingManager(CurrentScreenAnalysisManager):
    def __init__(self, source: _Analyzer) -> None:
        super().__init__(source)
        self.outcome_calls = 0
        self.invalidation_calls = 0

    def observe_outcome(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        self.outcome_calls += 1

    def invalidate(self, observation: Observation, reason: str) -> None:
        del observation, reason
        self.invalidation_calls += 1


def test_runtime_counts_and_retains_screen_analysis(tmp_path: Path) -> None:
    source = _Analyzer()
    observation = _observation("obs_1", "0000000000000000")
    result = _runtime(
        tmp_path,
        SequencePlanner([proposal(ActionKind.CANNOT_PROGRESS)]),
        ObservationSequence([observation]),
        RuntimeDevice(),
        screen_analysis=CurrentScreenAnalysisManager(source),
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert source.calls == 1
    assert result.usage.planning_requests == 2
    assert result.screen_analysis_ids == ("analysis-1",)


def test_runtime_skips_analysis_when_the_action_source_does_not_require_it(
    tmp_path: Path,
) -> None:
    source = _Analyzer()
    observation = _observation("obs_1", "0000000000000000")
    result = _runtime(
        tmp_path,
        _AnalysisOptionalPlanner([proposal(ActionKind.CANNOT_PROGRESS)]),
        ObservationSequence([observation]),
        RuntimeDevice(),
        screen_analysis=CurrentScreenAnalysisManager(source),
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert source.calls == 0
    assert result.usage.planning_requests == 1
    assert result.screen_analysis_ids == ()


def test_runtime_stops_before_planning_after_analyzer_capacity_exhaustion(
    tmp_path: Path,
) -> None:
    source = _CapacityExhaustedAnalyzer()
    observation = _observation("obs_1", "0000000000000000")
    result = _runtime(
        tmp_path,
        SequencePlanner([]),
        ObservationSequence([observation]),
        RuntimeDevice(),
        screen_analysis=CurrentScreenAnalysisManager(source),
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert source.calls == 1
    assert result.reason.startswith("screen_analysis_capacity_exhausted:")
    assert result.usage.planning_requests == 1


def test_deterministic_rejoin_skips_analyzer_capacity_failure() -> None:
    analyzer = _CapacityExhaustedAnalyzer()
    state = ScreenAnalysisRunState(
        manager=CurrentScreenAnalysisManager(analyzer),
        recovery=RecoveryRunState(None, ScreenComparisonPolicy()),
        policy=_mode_policy(),
        started_at=0.0,
        requirement_source=_ObservationAwareAnalysisSource(),
    )
    observation = _observation("obs_rejoined", "0000000000000000")

    attempt = state.analyze(
        observation,
        ScreenAnalysisBudget(
            completed_steps=0,
            planning_requests=0,
            grounding_requests=0,
            budget_time=0.0,
        ),
    )

    assert analyzer.calls == 0
    assert attempt.budget_reason is None


def test_optional_analysis_source_skips_manager_callbacks() -> None:
    manager = _RecordingManager(_Analyzer())
    source = _AnalysisOptionalPlanner([])
    state = ScreenAnalysisRunState(
        manager=manager,
        recovery=RecoveryRunState(None, ScreenComparisonPolicy()),
        policy=_mode_policy(),
        started_at=0.0,
        requirement_source=source,
    )
    before = _observation("obs_1", "0000000000000000")
    after = _observation("obs_2", "1111111111111111", offset_seconds=1)

    state.observe_outcome(
        proposal(),
        before,
        after,
        cast(ActionOutcome, object()),
    )
    state.invalidate(before, "contradiction")
    state.observe_dispatch_rejection(DispatchDisposition.TARGET_REJECTED, before)

    assert manager.outcome_calls == 0
    assert manager.invalidation_calls == 0
