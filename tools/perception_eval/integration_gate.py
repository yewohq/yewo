"""Run the production perception stack through its cross-module contracts.

This is an explicit, model-backed integration gate. Unit tests exercise its
comparison rules without loading vision models; CI runs the command only when
the private model token and the opt-in repository variable are available.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import BaseModel, ConfigDict

from yewo.agent.actuation.actions import Coordinates, TapAction
from yewo.agent.grounding import GROUNDING_MARKS, marks
from yewo.agent.memory import replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo
from yewo.agent.perception.keyboard import KEYBOARD_KEY_ROLE, KEYBOARD_REGION_ROLE
from yewo.defaults import DEFAULT_PERCEPTION_MODEL

from tools.perception_eval.benchmark_incremental import benchmark
from tools.perception_eval.capture import capture
from tools.perception_eval.corpus import MANIFEST_PATH, PerceptionCorpus, read_corpus
from tools.perception_eval.metrics import (
    CaseQuality,
    EvaluationMetrics,
    EvaluationRun,
    case_quality,
    score,
)

DEFAULT_BASELINE = MANIFEST_PATH.parent / "integration_baseline.json"


class BaselineCase(BaseModel):
    """Stable evidence a later production run must continue to satisfy."""

    model_config = ConfigDict(extra="forbid")
    fixture_digest: str
    quality: CaseQuality


class IntegrationBaseline(BaseModel):
    """Reviewed evidence, excluding volatile boxes and confidences."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1
    detector: str
    ocr: str
    cases: dict[str, BaselineCase]


class CaseRegression(BaseModel):
    case_id: str
    failures: tuple[str, ...]


class IntegrationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    passed: bool
    detector: str
    ocr: str
    metrics: EvaluationMetrics
    regressions: tuple[CaseRegression, ...]
    contract_failures: tuple[str, ...]
    replay_cases: tuple[str, ...]
    incremental: dict[str, object] | None = None


def build_baseline(corpus: PerceptionCorpus, run: EvaluationRun) -> IntegrationBaseline:
    """Reduce a full capture to stable, reviewable regression evidence."""
    score(corpus, run)
    by_id = {observation.case_id: observation for observation in run.observations}
    return IntegrationBaseline(
        detector=run.detector,
        ocr=run.ocr,
        cases={
            case.id: BaselineCase(
                fixture_digest=case.digest,
                quality=case_quality(case, by_id[case.id]),
            )
            for case in corpus.cases
        },
    )


def evaluate_run(
    corpus: PerceptionCorpus,
    baseline: IntegrationBaseline,
    run: EvaluationRun,
    *,
    validate_replay: bool = True,
) -> IntegrationReport:
    """Compare one real capture with its baseline and runtime contracts."""
    metrics = score(corpus, run)
    regressions: list[CaseRegression] = []
    failures: list[str] = []
    replay_cases: list[str] = []
    by_id = {observation.case_id: observation for observation in run.observations}
    corpus_ids = {case.id for case in corpus.cases}
    if set(baseline.cases) != corpus_ids:
        failures.append("baseline does not cover exactly the current corpus")
    if baseline.detector != run.detector:
        failures.append("detector identity differs from the reviewed baseline")
    if baseline.ocr != run.ocr:
        failures.append("OCR identity differs from the reviewed baseline")
    if corpus.missing_platforms or corpus.missing_traits:
        failures.append("corpus does not meet its declared platform and trait coverage")

    for case in corpus.cases:
        observation = by_id[case.id]
        expected = baseline.cases.get(case.id)
        case_failures: list[str] = []
        if expected is None:
            continue
        if expected.fixture_digest != case.digest:
            case_failures.append("fixture changed without accepting a new baseline")
        actual = case_quality(case, observation)
        missing_text = set(expected.quality.matched_text_indices) - set(
            actual.matched_text_indices
        )
        if missing_text:
            values = ", ".join(repr(case.expected_text[index]) for index in sorted(missing_text))
            case_failures.append(f"lost expected text: {values}")
        missing_targets = set(expected.quality.covered_target_names) - set(
            actual.covered_target_names
        )
        if missing_targets:
            case_failures.append(
                "lost target coverage: " + ", ".join(sorted(missing_targets))
            )
        if expected.quality.keyboard_correct and not actual.keyboard_correct:
            case_failures.append("keyboard visibility no longer matches the fixture")
        if observation.out_of_bounds_count:
            case_failures.append("one or more detections are outside the screenshot")
        if observation.normal_keyboard_target_count:
            case_failures.append("keyboard-owned detections leaked into normal app targets")
        has_keys = any(KEYBOARD_KEY_ROLE in element.semantic_roles for element in observation.elements)
        if has_keys and not observation.expanded_keyboard_target_count:
            case_failures.append("explicit keyboard expansion produced no key targets")
        if case_failures:
            regressions.append(CaseRegression(case_id=case.id, failures=tuple(case_failures)))

    if validate_replay:
        try:
            replay_cases.extend(_validate_replay_round_trip(corpus, run))
        except ValueError as exc:
            failures.append(f"perception-to-replay contract failed: {exc}")
    return IntegrationReport(
        passed=not regressions and not failures,
        detector=run.detector,
        ocr=run.ocr,
        metrics=metrics,
        regressions=tuple(regressions),
        contract_failures=tuple(failures),
        replay_cases=tuple(replay_cases),
    )


@dataclass(frozen=True)
class _Reading:
    gui_elements: tuple[GUIElement, ...]
    scan_info: ScanInfo
    current_phash: str
    raw_img_bytes: bytes = b""


def _validate_replay_round_trip(
    corpus: PerceptionCorpus, run: EvaluationRun
) -> tuple[str, ...]:
    """Prove real detections survive projection, trace learning, and matching."""
    by_id = {observation.case_id: observation for observation in run.observations}
    validated: list[str] = []
    validated_keyboard = False
    for case in corpus.cases:
        observation = by_id[case.id]
        elements = tuple(_element(item) for item in observation.elements)
        with Image.open(case.screenshot) as image:
            width, height = image.size
        expanded = not validated_keyboard and any(
            KEYBOARD_KEY_ROLE in element.semantic_roles for element in elements
        )
        projection = marks.project_targets(
            elements,
            width,
            height,
            case.keyboard_visible,
            include_keyboard_targets=expanded,
        )
        target = next(
            (
                mark
                for mark in projection.marks
                if mark.element.text.strip()
                and (
                    KEYBOARD_KEY_ROLE in mark.element.semantic_roles
                    if expanded
                    else KEYBOARD_KEY_ROLE not in mark.element.semantic_roles
                )
            ),
            None,
        )
        if target is None:
            continue
        scan = ScanInfo(
            mode="integration",
            reason="first_pass",
            phash=case.digest,
            screen_width=width,
            screen_height=height,
            element_count=len(elements),
            keyboard_effective_visible=case.keyboard_visible,
            keyboard_targets_expanded=expanded,
        )
        identity = replay_store.TraceIdentity(
            scenario_uuid=f"integration-{case.id}",
            scenario_digest=case.digest,
            platform=case.platform,
            target_profile="perception-corpus",
            detector=run.detector,
            perception_pipeline=f"detector={run.detector}|ocr={run.ocr}",
            grounding=GROUNDING_MARKS,
        )
        action = TapAction(
            coordinates=Coordinates(x=target.center[0], y=target.center[1]),
            target=target.element.text,
            reason="integration round trip",
        )
        trace = replay_walk.build_trace(
            identity,
            [action],
            [((elements, scan), (elements, scan))],
            replay_store.ValueNames(),
        )
        if trace is None:
            continue
        reading = _Reading(elements, scan, case.digest)
        if replay_walk.select_transition(trace, 0, reading, replay_store.ValueNames()) != 0:
            raise ValueError(f"{case.id} learned a trace that its source cannot match")
        validated.append(case.id)
        validated_keyboard = validated_keyboard or expanded
        if len(validated) >= 2 and validated_keyboard:
            break
    if not validated:
        raise ValueError("no real corpus detection could be learned and matched")
    if not validated_keyboard:
        raise ValueError("no detected keyboard key could be learned in expanded mode")
    return tuple(validated)


def _element(item) -> GUIElement:
    return GUIElement(
        element=item.element,
        bounding_box=BoundingBox(**item.bounding_box.model_dump()),
        confidence=item.confidence,
        type="yolo_detection",
        text=item.text,
        semantic_roles=frozenset(item.semantic_roles),
    )


def incremental_failures(result: dict) -> list[str]:
    """Return machine-relative incremental contract failures."""
    failures = []
    if not result.get("detector_reused"):
        failures.append("incremental refresh did not reuse detector geometry")
    if KEYBOARD_REGION_ROLE not in result.get("surface_roles", []):
        failures.append("incremental refresh did not identify the keyboard surface")
    if not result.get("unique_app_text_equal"):
        failures.append("incremental refresh changed app-visible fused text")
    if result.get("incremental_app_class_counts") != result.get("full_app_class_counts"):
        failures.append("incremental refresh changed app-visible detector classes")
    if float(result.get("speedup", 0)) <= 1:
        failures.append("incremental refresh was not faster than a same-process full pass")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the production perception integration gate.")
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--model", default=DEFAULT_PERCEPTION_MODEL)
    parser.add_argument("--accept-baseline", action="store_true")
    parser.add_argument("--skip-incremental", action="store_true")
    arguments = parser.parse_args()
    arguments.output.mkdir(parents=True, exist_ok=True)
    corpus = read_corpus(arguments.manifest)
    run = capture(arguments.output / "capture.json", arguments.manifest, arguments.model)
    if arguments.accept_baseline:
        accepted = build_baseline(corpus, run)
        arguments.baseline.write_text(accepted.model_dump_json(indent=2) + "\n", encoding="utf-8")
        print(f"Accepted integration baseline at {arguments.baseline}")
        return 0
    baseline = IntegrationBaseline.model_validate_json(arguments.baseline.read_bytes())
    report = evaluate_run(corpus, baseline, run)
    if not arguments.skip_incremental:
        keyboard_case = next(case for case in corpus.cases if case.keyboard_visible)
        incremental = benchmark(
            keyboard_case.screenshot,
            arguments.model,
            arguments.output / "incremental_progression.png",
        )
        failures = incremental_failures(incremental)
        report.incremental = incremental
        if failures:
            report.contract_failures += tuple(failures)
            report.passed = False
        (arguments.output / "incremental.json").write_text(
            json.dumps(incremental, indent=2) + "\n", encoding="utf-8"
        )
    (arguments.output / "report.json").write_text(
        report.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    print(f"Perception integration gate: {'passed' if report.passed else 'failed'}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
