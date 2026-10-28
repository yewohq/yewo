"""Score one detector/OCR observation against the frozen corpus."""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

from tools.perception_eval.corpus import PerceptionCase, PerceptionCorpus, TargetPoint

STRUCTURAL_ELEMENTS = {"Keyboard", "Toolbar", "Keys", "StatusBar", "ToolbarItem"}


class Box(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def contains(self, point: TargetPoint) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max


class ObservedElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element: str
    text: str = ""
    confidence: float = Field(ge=0.0, le=1.0)
    bounding_box: Box
    semantic_roles: set[str] = Field(default_factory=set)


class CaseObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    fixture_digest: str
    visual_keyboard_visible: bool
    mark_count: int = Field(ge=0)
    latency_ms: float = Field(ge=0)
    peak_rss_mib: float = Field(ge=0)
    elements: list[ObservedElement]
    association_count: int = Field(default=0, ge=0)
    out_of_bounds_count: int = Field(default=0, ge=0)
    normal_keyboard_target_count: int = Field(default=0, ge=0)
    expanded_keyboard_target_count: int = Field(default=0, ge=0)


class EvaluationRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    detector: str
    inference_image_size: int | None = Field(default=None, gt=0)
    ocr: str
    observations: list[CaseObservation]


class EvaluationMetrics(BaseModel):
    cases: int
    keyboard_true_positives: int
    keyboard_false_positives: int
    keyboard_false_negatives: int
    keyboard_precision: float
    keyboard_recall: float
    normalized_text_recall: float
    target_coverage: float
    mean_mark_count: float
    mean_latency_ms: float
    peak_rss_mib: float


class CaseQuality(BaseModel):
    """Stable human-reviewed expectations satisfied by one observation."""

    keyboard_correct: bool
    matched_text_indices: tuple[int, ...]
    covered_target_names: tuple[str, ...]


def case_quality(case: PerceptionCase, observation: CaseObservation) -> CaseQuality:
    """Return per-case evidence used by both aggregate scoring and regression gates."""
    observed_text = Counter(
        normalized
        for element in observation.elements
        if (normalized := _normalize(element.text))
    )
    matched_text_indices = []
    for index, expected in enumerate(case.expected_text):
        normalized = _normalize(expected)
        if observed_text[normalized]:
            matched_text_indices.append(index)
            observed_text[normalized] -= 1

    covered_target_names = tuple(
        point.name
        for point in case.target_points
        if any(
            element.element not in STRUCTURAL_ELEMENTS and element.bounding_box.contains(point)
            for element in observation.elements
        )
    )
    return CaseQuality(
        keyboard_correct=observation.visual_keyboard_visible == case.keyboard_visible,
        matched_text_indices=tuple(matched_text_indices),
        covered_target_names=covered_target_names,
    )


def score(corpus: PerceptionCorpus, run: EvaluationRun) -> EvaluationMetrics:
    """Calculate stable quality metrics and descriptive resource measurements."""
    by_id = {observation.case_id: observation for observation in run.observations}
    expected_ids = {case.id for case in corpus.cases}
    if set(by_id) != expected_ids:
        raise ValueError("observations must contain every corpus case exactly once")

    true_positive = false_positive = false_negative = 0
    expected_text_count = matched_text_count = 0
    target_count = covered_target_count = 0
    mark_counts: list[int] = []
    latencies: list[float] = []
    peak_rss_values: list[float] = []

    for case in corpus.cases:
        observation = by_id[case.id]
        if observation.fixture_digest != case.digest:
            raise ValueError(f"{case.id}: observation covers a different fixture version")
        if observation.visual_keyboard_visible and case.keyboard_visible:
            true_positive += 1
        elif observation.visual_keyboard_visible:
            false_positive += 1
        elif case.keyboard_visible:
            false_negative += 1

        quality = case_quality(case, observation)
        expected_text_count += len(case.expected_text)
        matched_text_count += len(quality.matched_text_indices)
        target_count += len(case.target_points)
        covered_target_count += len(quality.covered_target_names)

        mark_counts.append(observation.mark_count)
        latencies.append(observation.latency_ms)
        peak_rss_values.append(observation.peak_rss_mib)

    return EvaluationMetrics(
        cases=len(corpus.cases),
        keyboard_true_positives=true_positive,
        keyboard_false_positives=false_positive,
        keyboard_false_negatives=false_negative,
        keyboard_precision=_ratio(true_positive, true_positive + false_positive),
        keyboard_recall=_ratio(true_positive, true_positive + false_negative),
        normalized_text_recall=_ratio(matched_text_count, expected_text_count),
        target_coverage=_ratio(covered_target_count, target_count),
        mean_mark_count=round(sum(mark_counts) / len(mark_counts), 3),
        mean_latency_ms=round(sum(latencies) / len(latencies), 3),
        peak_rss_mib=max(peak_rss_values),
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0
