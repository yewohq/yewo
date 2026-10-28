"""Saved whole-screen OCR observations and corpus-level shadow metrics."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from tools.perception_eval.corpus import PerceptionCorpus
from yewo.agent.perception.ocr import OCRResult


class OCRCaseObservation(BaseModel):
    """Independent OCR output for one frozen screenshot."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    fixture_digest: str
    result: OCRResult
    peak_rss_mib: float = Field(ge=0)


class OCREvaluationRun(BaseModel):
    """One OCR engine evaluated over the complete frozen corpus."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ocr: str
    observations: list[OCRCaseObservation]


class OCREvaluationMetrics(BaseModel):
    """OCR-only quality and resource measurements."""

    cases: int
    expected_text_count: int
    exact_line_recall: float
    contained_text_recall: float
    mean_line_count: float
    cold_wall_ms: float
    mean_warm_wall_ms: float
    mean_engine_ms: float
    peak_rss_mib: float


def score_ocr(corpus: PerceptionCorpus, run: OCREvaluationRun) -> OCREvaluationMetrics:
    """Score OCR text independently from detector boxes and classes."""
    expected_case_ids = [case.id for case in corpus.cases]
    observed_case_ids = [observation.case_id for observation in run.observations]
    if observed_case_ids != expected_case_ids:
        raise ValueError("OCR observations must match corpus cases and order exactly")

    expected_count = exact_count = contained_count = 0
    line_counts: list[int] = []
    wall_durations: list[float] = []
    engine_durations: list[float] = []
    peak_rss_values: list[float] = []
    for case, observation in zip(corpus.cases, run.observations, strict=True):
        if observation.fixture_digest != case.digest:
            raise ValueError(f"{case.id}: OCR observation covers a different fixture version")
        if observation.result.engine_identity != run.ocr:
            raise ValueError(f"{case.id}: OCR result identity differs from the run identity")
        observed_lines = [_normalize(line.text) for line in observation.result.lines]
        for expected_text in case.expected_text:
            expected_count += 1
            normalized_expected = _normalize(expected_text)
            exact_count += normalized_expected in observed_lines
            contained_count += any(
                _contains_text(observed_text, normalized_expected)
                for observed_text in observed_lines
            )
        line_counts.append(len(observation.result.lines))
        wall_durations.append(observation.result.timing.wall_ms)
        engine_durations.append(observation.result.timing.engine_total_ms)
        peak_rss_values.append(observation.peak_rss_mib)

    warm_durations = wall_durations[1:] or wall_durations
    return OCREvaluationMetrics(
        cases=len(corpus.cases),
        expected_text_count=expected_count,
        exact_line_recall=_ratio(exact_count, expected_count),
        contained_text_recall=_ratio(contained_count, expected_count),
        mean_line_count=_mean(line_counts),
        cold_wall_ms=round(wall_durations[0], 3),
        mean_warm_wall_ms=_mean(warm_durations),
        mean_engine_ms=_mean(engine_durations),
        peak_rss_mib=max(peak_rss_values),
    )


def _contains_text(observed: str, expected: str) -> bool:
    if not observed or not expected:
        return False
    if _uses_word_boundaries(expected):
        return re.search(rf"(?<!\w){re.escape(expected)}(?!\w)", observed) is not None
    return expected in observed


def _uses_word_boundaries(text: str) -> bool:
    return all(character.isascii() for character in text)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _mean(values: list[float] | list[int]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
