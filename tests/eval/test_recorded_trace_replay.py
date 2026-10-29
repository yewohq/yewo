"""Recorded trainer traces reproduce reviewed deterministic readings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from yewo.evaluation.recorded_trace_replay import (
    DEFAULT_CORPUS,
    SourceArtifact,
    corpus_obligation_readiness,
    load_corpus,
    replay_case,
    replay_corpus,
)


def test_every_recorded_trace_matches_the_reviewed_phase_one_baseline() -> None:
    corpus = load_corpus()

    drifted = [result for result in replay_corpus(corpus) if not result.matches_baseline]

    assert drifted == []


def test_corpus_records_both_closed_contracts_and_known_architecture_gaps() -> None:
    corpus = load_corpus()

    gaps = {case.id for case in corpus.cases if case.known_gap}
    closed = {case.id for case in corpus.cases if not case.known_gap}

    assert gaps == set()
    assert closed == {
        "index-error-remains-separate-from-earlier-false-progress",
        "final-outcome-is-not-sent-on-an-intermediate-screen",
        "native-and-visual-keyboard-disagreement-is-preserved",
        "planner-index-cannot-close-route-recovery",
        "planner-self-assessment-is-not-evidence",
        "live-snackbar-jitter-does-not-settle-or-pollute-navigation",
        "renamed-control-keeps-one-identity",
        "snackbar-reversion-keeps-control-evidence",
        "text-entry-requires-value-and-retained-form",
    }


def test_every_trace_names_a_reviewed_run_baseline() -> None:
    corpus = load_corpus()
    run_ids = {baseline.run_id for baseline in corpus.run_baselines}

    assert run_ids == {case.source_run_id for case in corpus.cases}
    assert all(baseline.model_calls > 0 for baseline in corpus.run_baselines)
    assert all(baseline.total_tokens > 0 for baseline in corpus.run_baselines)
    assert all(
        baseline.time_to_first_finding_ms is not None or baseline.issues_found == 0
        for baseline in corpus.run_baselines
    )


def test_obligation_readiness_is_measured_across_semantic_trace_cases() -> None:
    readiness = corpus_obligation_readiness(load_corpus())

    assert readiness.total == 1
    assert readiness.inactive == 1
    assert readiness.unknown == 0
    assert readiness.strict_ready


def test_improved_behavior_is_reported_as_baseline_drift_until_reviewed() -> None:
    corpus = load_corpus()
    case = next(
        case
        for case in corpus.cases
        if case.id == "final-outcome-is-not-sent-on-an-intermediate-screen"
    )
    case.baseline.scenario_rule_sent = True

    result = replay_case(case)

    assert not result.matches_baseline


@pytest.mark.parametrize("unsafe_path", ["/tmp/run.json", "../run.json"])
def test_source_artifact_paths_are_repository_relative(unsafe_path: str) -> None:
    with pytest.raises(ValidationError):
        SourceArtifact(path=unsafe_path, sha256="a" * 64)


def test_default_corpus_is_a_tracked_evaluation_input() -> None:
    assert Path("eval/recorded_traces/trainer_app_1645_buggy.json").resolve() == DEFAULT_CORPUS
