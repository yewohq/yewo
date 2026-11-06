"""Persistence tests for observation-bound screen analyses."""

from pathlib import Path

import pytest

from observations.factories import build_observation
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    CurrentScreenAnalysisManager,
    ScreenAnalysisStore,
    ScreenAnalysisStoreError,
)
from yewo.agent.screen_analysis.store import SCREEN_ANALYSES_PATH
from yewo.observations import Observation


class _Analyzer:
    def required_model_requests(self, observation: Observation) -> int:
        del observation
        return 1

    def analyze(self, observation: Observation) -> CurrentScreenAnalysis:
        return CurrentScreenAnalysis(
            analysis_id="analysis-source",
            observation_id=observation.observation_id,
            screenshot_sha256=observation.redacted_screenshot.sha256,
            presentation=observation.presentation_key,
            screen_summary="Visible application screen.",
        )


def test_manager_persists_source_and_rebound_analyses(tmp_path: Path) -> None:
    store = ScreenAnalysisStore(tmp_path)
    manager = CurrentScreenAnalysisManager(
        _Analyzer(),
        identifier_factory=lambda: "analysis-rebound",
        store=store,
    )
    source = build_observation(observation_id="obs-source")
    rebound = source.model_copy(update={"observation_id": "obs-rebound"})

    first = manager.analyze(source)
    second = manager.analyze(rebound)

    assert first is not None and second is not None
    loaded = store.load_selected((second.analysis_id, first.analysis_id))
    assert tuple(item.observation_id for item in loaded) == ("obs-rebound", "obs-source")


def test_store_rejects_missing_analysis(tmp_path: Path) -> None:
    with pytest.raises(ScreenAnalysisStoreError, match="unavailable"):
        ScreenAnalysisStore(tmp_path).load_selected(("analysis-missing",))


def test_store_rejects_missing_requested_evidence(tmp_path: Path) -> None:
    store = ScreenAnalysisStore(tmp_path)
    store.persist(_Analyzer().analyze(build_observation()))

    with pytest.raises(ScreenAnalysisStoreError, match="missing requested evidence"):
        store.load_selected(("analysis-missing",))


def test_store_rejects_an_incomplete_final_line(tmp_path: Path) -> None:
    path = tmp_path / SCREEN_ANALYSES_PATH
    path.parent.mkdir(parents=True)
    path.write_bytes(b'{"analysis_id":"analysis-source"}')

    with pytest.raises(ScreenAnalysisStoreError, match="incomplete final line"):
        ScreenAnalysisStore(tmp_path).load_selected(("analysis-source",))


def test_store_rejects_a_repeated_analysis_identity(tmp_path: Path) -> None:
    analysis = _Analyzer().analyze(build_observation())
    path = tmp_path / SCREEN_ANALYSES_PATH
    path.parent.mkdir(parents=True)
    row = analysis.model_dump_json().encode()
    path.write_bytes(row + b"\n" + row + b"\n")

    with pytest.raises(ScreenAnalysisStoreError, match="repeats an analysis identity"):
        ScreenAnalysisStore(tmp_path).load_selected((analysis.analysis_id,))
