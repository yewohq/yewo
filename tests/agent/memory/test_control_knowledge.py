"""Screen-analysis AppMemory capture tests."""

from datetime import datetime, timezone
from pathlib import Path

from yewo.agent.memory.knowledge_capture import build_memory_delta
from yewo.agent.memory.writeback import apply_memory_delta
from yewo.agent.screen_analysis import (
    CurrentScreenAnalysis,
    ScreenElement,
    ScreenElementConfidence,
    ScreenElementKind,
    ScreenSemanticRole,
    ScreenVisualKind,
)
from yewo.observations import Observation

from agent.discovery.support import agent_result, observation
from agent.memory.test_store import _memory


def _analysis(
    observed: Observation,
    analysis_id: str,
    elements: tuple[ScreenElement, ...],
) -> CurrentScreenAnalysis:
    return CurrentScreenAnalysis(
        analysis_id=analysis_id,
        observation_id=observed.observation_id,
        screen_fingerprint=observed.screen_fingerprint.value,
        screenshot_sha256=observed.redacted_screenshot.sha256,
        platform_context=observed.platform_context,
        keyboard_visible=observed.keyboard_visible,
        orientation=observed.orientation,
        elements=elements,
    )


def _element(
    element_ref: str,
    name: str,
    semantic: ScreenSemanticRole,
    state: str,
    *,
    options: tuple[str, ...] = (),
) -> ScreenElement:
    return ScreenElement(
        element_ref=element_ref,
        name=name,
        kind=ScreenElementKind(
            visual=(
                ScreenVisualKind.TEXT
                if semantic is ScreenSemanticRole.LABEL
                else ScreenVisualKind.INPUT_SURFACE
            ),
            semantic=semantic,
        ),
        state=state,
        options=options,
        confidence=ScreenElementConfidence.HIGH,
        evidence="The element is visibly established.",
    )


def test_analyzed_elements_are_retained_as_screen_knowledge(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2", "1111111111111111", predecessor_observation_id="obs_1", offset_seconds=1
    )
    result = agent_result(before, after)
    exercised_step = result.steps[0].model_copy(
        update={
            "proposal": result.steps[0].proposal.model_copy(
                update={"target_element_ref": "primary-field"}
            )
        }
    )
    first_analysis = _analysis(
        before,
        "analysis-1",
        (
            _element("primary-field", "Primary field", ScreenSemanticRole.TEXT_FIELD, "empty"),
            _element("section-heading", "Section heading", ScreenSemanticRole.LABEL, "visible"),
            _element(
                "mode-selector",
                "Mode selector",
                ScreenSemanticRole.DROPDOWN,
                "standard",
                options=("standard", "advanced"),
            ),
        ),
    )
    later_analysis = _analysis(
        before,
        "analysis-2",
        (_element("primary-field-later", "Primary field", ScreenSemanticRole.TEXT_FIELD, "filled"),),
    )
    result = result.model_copy(
        update={"steps": (exercised_step,), "screen_analyses": (first_analysis, later_analysis)}
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        result,
        {before.observation_id: before, after.observation_id: after},
        datetime(2026, 11, 3, tzinfo=timezone.utc),
    )
    updated = apply_memory_delta(memory, delta)

    controls = {control.name: control for control in updated.app_knowledge.controls}
    assert controls["Primary field"].exercised is True
    assert controls["Primary field"].state == "filled"
    assert controls["Mode selector"].options == ("standard", "advanced")
    assert controls["Mode selector"].state == "standard"
    assert "Section heading" not in controls


def test_missing_optional_analysis_records_a_control_capture_skip(tmp_path: Path) -> None:
    unused_compatibility, memory = _memory(tmp_path)
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id="obs_1",
        offset_seconds=1,
    )

    delta = build_memory_delta(
        memory,
        "run_2",
        agent_result(before, after),
        {before.observation_id: before, after.observation_id: after},
        datetime(2026, 11, 3, tzinfo=timezone.utc),
    )

    assert any(
        skipped.record_kind == "control"
        and "No screen-analysis evidence" in skipped.reason
        for skipped in delta.skipped_records
    )
