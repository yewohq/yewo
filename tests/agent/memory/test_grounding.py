"""Exact-screen grounding reuse tests."""

from datetime import datetime, timezone
from pathlib import Path

import pytest

from yewo.agent.discovery import AppMemoryBuilder, DiscoveryObjective, DiscoveryRequest
from yewo.agent.memory.grounding import AppMemoryGrounder
from yewo.agent.memory.knowledge import (
    AppKnowledge,
    GroundingAssessment,
    GroundingAssessmentStatus,
    GroundingRole,
    KnowledgeEvidenceReference,
)
from yewo.agent import AgentTerminalReason, AppMemory, MemoryCompatibility
from yewo.grounding import (
    GroundingCoordinateConvention,
    GroundingCandidate,
    GroundingOutputShape,
    GroundingProfile,
    GroundingInputError,
    GroundingRequest,
    GroundingResolutionSource,
    GroundingTargetOperation,
    MemoryGroundingResolution,
    ModelGroundingResolution,
)
from yewo.run_narration import RunNarrator, narration_scope

from agent.discovery.support import agent_result, observation
from grounding.support import DeterministicGrounder


class CountingGrounder:
    def __init__(self) -> None:
        self.calls = 0

    def required_model_requests(self, request: GroundingRequest) -> int:
        del request
        return 1

    def ground(self, request: GroundingRequest):
        self.calls += 1
        return DeterministicGrounder(40, 50).ground(request)


def _memory_and_source():
    source = observation("obs_1", "0000000000000000")
    destination = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=source.observation_id,
        offset_seconds=1,
    )
    promotion = AppMemoryBuilder().promote(
        DiscoveryRequest(
            request_id="request_1",
            objectives=(
                DiscoveryObjective(
                    objective_id="record_form",
                    statement="The record form is reachable.",
                ),
            ),
        ),
        agent_result(source, destination),
        {source.observation_id: source, destination.observation_id: destination},
        MemoryCompatibility(application=source.application),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    return promotion.memory, source, destination


def _profile() -> GroundingProfile:
    return GroundingProfile(
        profile_id="grounder",
        model_profile_id="grounder-model",
        coordinate_convention=GroundingCoordinateConvention.SOURCE_PIXELS,
        supported_operations=frozenset(GroundingTargetOperation),
        supported_shapes=frozenset({GroundingOutputShape.POINT}),
    )


def _request(source, **updates):
    values = {
        "grounding_id": "grounding_replay",
        "screenshot": source.redacted_screenshot,
        "source_presentation": source.presentation_key,
        "expression": "the add button",
        "operation": GroundingTargetOperation.TAP,
        "output_shape": GroundingOutputShape.POINT,
        "excluded_regions": source.system_regions,
    }
    values.update(updates)
    return GroundingRequest.model_validate(values)


def test_exact_source_pixels_reuse_accepted_grounding_without_a_model_call(
    tmp_path: Path,
) -> None:
    memory, source, _ = _memory_and_source()
    delegate = CountingGrounder()
    grounder = AppMemoryGrounder(delegate, memory, _profile())
    request = _request(source)

    assert grounder.required_model_requests(request) == 0
    with narration_scope(RunNarrator(tmp_path)):
        result = grounder.ground(request)

    assert delegate.calls == 0
    assert result.resolution_source is GroundingResolutionSource.MEMORY
    assert result.resolution == MemoryGroundingResolution(
        transition_id=memory.transitions[0].transition_id
    )
    assert memory.transitions[0].target_grounding is not None
    assert result.location == memory.transitions[0].target_grounding.location
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Reusing an exact grounding location from app memory." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "lookup=hit reason=exact_source_match" in trace


def test_fallback_model_grounding_remains_reusable(tmp_path: Path) -> None:
    memory, source, _ = _memory_and_source()
    transition = memory.transitions[0]
    assert transition.target_grounding is not None
    fallback_grounding = transition.target_grounding.model_copy(
        update={
            "resolution": ModelGroundingResolution(
                grounding_profile_id="grounder-fallback",
                model_request_ids=("fallback-request",),
            )
        }
    )
    memory = memory.model_copy(
        update={
            "transitions": (transition.model_copy(update={"target_grounding": fallback_grounding}),)
        }
    )
    profile = _profile().model_copy(
        update={
            "fallbacks": (
                GroundingCandidate(
                    model_profile_id="grounder-fallback",
                    coordinate_convention=GroundingCoordinateConvention.SOURCE_PIXELS,
                ),
            )
        }
    )
    delegate = CountingGrounder()
    grounder = AppMemoryGrounder(delegate, memory, profile)

    with narration_scope(RunNarrator(tmp_path)):
        result = grounder.ground(_request(source))

    assert result.resolution_source is GroundingResolutionSource.MEMORY
    assert delegate.calls == 0


def test_changed_exact_pixels_fall_back_to_the_visual_grounder(tmp_path: Path) -> None:
    memory, source, _ = _memory_and_source()
    changed_fingerprint = source.screen_fingerprint.model_copy(
        update={"masked_pixels_sha256": "f" * 64}
    )
    delegate = CountingGrounder()
    grounder = AppMemoryGrounder(delegate, memory, _profile())
    request = _request(
        source,
        source_presentation=source.presentation_key.model_copy(
            update={"fingerprint": changed_fingerprint}
        ),
    )

    assert grounder.required_model_requests(request) == 1
    with narration_scope(RunNarrator(tmp_path)):
        result = grounder.ground(request)

    assert delegate.calls == 1
    assert result.resolution_source is GroundingResolutionSource.MODEL
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Grounding memory was invalidated. Requesting fresh grounding." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "lookup=invalidated reason=source_pixels_changed" in trace


def test_writeback_invalidation_forces_fresh_grounding(tmp_path: Path) -> None:
    memory, source, _ = _memory_and_source()
    transition = memory.transitions[0]
    assessment = GroundingAssessment(
        assessment_id="assessment_1",
        transition_id=transition.transition_id,
        role=GroundingRole.TARGET,
        source_observation_id=transition.source_observation_id,
        status=GroundingAssessmentStatus.INVALIDATED,
        invalidation_reason="Corrective grounding succeeded at a different point.",
        sources=(
            KnowledgeEvidenceReference(
                run_id="run_1",
                step_id=transition.source_step_id,
                observation_id=transition.source_observation_id,
            ),
        ),
    )
    memory = memory.model_copy(
        update={"app_knowledge": AppKnowledge(grounding_assessments=(assessment,))}
    )
    delegate = CountingGrounder()
    grounder = AppMemoryGrounder(delegate, memory, _profile())
    request = _request(source)

    assert grounder.required_model_requests(request) == 1
    with narration_scope(RunNarrator(tmp_path)):
        result = grounder.ground(request)

    assert result.resolution_source is GroundingResolutionSource.MODEL
    assert delegate.calls == 1
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "lookup=invalidated reason=writeback_invalidated" in trace


def test_reobserved_transition_enriches_legacy_memory_with_exact_grounding() -> None:
    memory, source, destination = _memory_and_source()
    legacy_transition = memory.transitions[0].model_copy(update={"target_grounding": None})
    legacy_memory = memory.model_copy(update={"transitions": (legacy_transition,)})
    reobserved_source = observation("obs_3", "0000000000000000", offset_seconds=2)
    reobserved_source = reobserved_source.model_copy(
        update={
            "screen_fingerprint": reobserved_source.screen_fingerprint.model_copy(
                update={"masked_pixels_sha256": "f" * 64}
            )
        }
    )
    reobserved_destination = observation(
        "obs_4",
        "1111111111111111",
        predecessor_observation_id=reobserved_source.observation_id,
        offset_seconds=3,
    )

    promotion = AppMemoryBuilder().promote(
        DiscoveryRequest(
            request_id="request_1",
            objectives=(
                DiscoveryObjective(
                    objective_id="record_form",
                    statement="The record form is reachable.",
                ),
            ),
        ),
        agent_result(
            reobserved_source,
            reobserved_destination,
            terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
        ),
        {
            reobserved_source.observation_id: reobserved_source,
            reobserved_destination.observation_id: reobserved_destination,
        },
        memory.compatibility,
        verified_objective_ids=frozenset(),
        source_run_id="run_2",
        updated_at=datetime(2026, 10, 31, 1, tzinfo=timezone.utc),
        existing=legacy_memory,
    )

    assert promotion.memory is not None
    enriched = promotion.memory.transitions[0]
    assert enriched.source_observation_id == source.observation_id
    assert enriched.target_grounding is not None
    assert enriched.target_grounding.source_observation_id == reobserved_source.observation_id
    cached = AppMemoryGrounder(CountingGrounder(), promotion.memory, _profile()).ground(
        _request(reobserved_source)
    )
    assert cached.resolution_source is GroundingResolutionSource.MEMORY


def test_memory_hit_still_requires_a_supported_grounding_operation() -> None:
    memory, source, _ = _memory_and_source()
    profile = _profile().model_copy(
        update={"supported_operations": frozenset({GroundingTargetOperation.TEXT_FIELD})}
    )
    delegate = CountingGrounder()
    grounder = AppMemoryGrounder(delegate, memory, profile)

    with pytest.raises(GroundingInputError, match="does not support tap"):
        grounder.required_model_requests(_request(source))

    assert delegate.calls == 0


def test_legacy_cached_grounding_uses_the_transition_source_observation() -> None:
    memory, source, _ = _memory_and_source()
    payload = memory.model_dump(mode="python")
    payload["transitions"][0]["target_grounding"].pop("source_observation_id")

    loaded = AppMemory.model_validate(payload)
    grounding = loaded.transitions[0].target_grounding

    assert grounding is not None
    assert grounding.source_observation_id is None
    result = AppMemoryGrounder(CountingGrounder(), loaded, _profile()).ground(_request(source))
    assert result.resolution_source is GroundingResolutionSource.MEMORY
