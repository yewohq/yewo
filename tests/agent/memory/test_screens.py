"""Current-screen reuse of verified and unsuccessful app-memory evidence."""

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import cast
from pathlib import Path

from yewo.agent import (
    AppMemory,
    MemoryCompatibility,
    UnsuccessfulAttempt,
    UnsuccessfulAttemptDisposition,
)
from yewo.agent.actuation import ActionKind, ActionProposal
from yewo.agent.discovery import AppMemoryBuilder, DiscoveryObjective, DiscoveryRequest
from yewo.agent.planning.records import (
    PlanningRequest,
)
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.memory.screens import AppMemoryActionSource, memory_planning_facts
from yewo.grounding import (
    GroundingCoordinateConvention,
    GroundingOutputShape,
    GroundingProfile,
    GroundingTargetOperation,
    PixelPoint,
)
from yewo.run_narration import RunNarrator, narration_scope
from yewo.observations import ScreenMatcher

from agent.actuation.support import proposal
from agent.discovery.support import agent_result, observation


def test_exact_screen_match_rejects_a_fingerprint_algorithm_change() -> None:
    first = observation("obs-1", "0000000000000000")
    second = first.model_copy(
        update={
            "observation_id": "obs-2",
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"algorithm": "other-fingerprint-algorithm"}
            ),
        }
    )

    assert not ScreenMatcher.exact(first.presentation_key, second.presentation_key)


class RecordingActionSource:
    """Record the enriched request supplied by app memory."""

    def __init__(self) -> None:
        self.request: PlanningRequest | None = None

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Count no model call for this recording double."""
        del request
        return 0

    def decision_metadata(self) -> PlannerDecisionMetadata:
        """Return retained-path evidence through transparent planner decorators."""
        return PlannerDecisionMetadata(
            source=PlannerDecisionSource.COMPILED_PROCEDURE,
            reused_flow_id="procedure_1",
            reused_procedure_step_ids=("procedure_step_1",),
            reused_segment_index=0,
        )

    def plan(self, request: PlanningRequest) -> ActionProposal:
        self.request = request
        return proposal(
            ActionKind.DONE,
            proposal_id="done",
            source_observation_id=request.observation.observation_id,
        )


def _grounding_profile(
    profile_id: str = "grounder",
    model_profile_id: str = "grounder-model",
) -> GroundingProfile:
    return GroundingProfile(
        profile_id=profile_id,
        model_profile_id=model_profile_id,
        coordinate_convention=GroundingCoordinateConvention.SOURCE_PIXELS,
        supported_operations=frozenset({GroundingTargetOperation.TAP}),
        supported_shapes=frozenset({GroundingOutputShape.POINT}),
    )


def test_action_source_supplies_only_matching_screen_and_grounder_evidence(
    tmp_path: Path,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
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
        agent_result(before, after),
        {before.observation_id: before, after.observation_id: after},
        MemoryCompatibility(application=before.application),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert promotion.memory is not None
    memory = promotion.memory
    transition = memory.transitions[0]
    unrelated_transition = transition.model_copy(
        update={
            "transition_id": "unrelated_transition",
            "source_screen_id": transition.destination_screen_id,
            "source_observation_id": transition.destination_observation_id,
            "action": transition.action.model_copy(
                update={"target_description": "the unrelated settings menu"}
            ),
        }
    )
    attempt = UnsuccessfulAttempt(
        attempt_id="attempt_1",
        screen_id=transition.source_screen_id,
        source_observation_id=before.observation_id,
        action=ActionKind.TAP,
        target_description="the add button",
        disposition=UnsuccessfulAttemptDisposition.TARGET_REJECTED,
        reason="The grounded point selected the label.",
        grounded_point=PixelPoint(x=10, y=20),
        screenshot_width_px=before.redacted_screenshot.width_px,
        screenshot_height_px=before.redacted_screenshot.height_px,
        source_step_id="step_1",
        source_run_id="run_1",
    )
    memory = memory.model_copy(
        update={
            "transitions": (*memory.transitions, unrelated_transition),
            "unsuccessful_attempts": (attempt,),
        }
    )
    source = RecordingActionSource()
    wrapped = AppMemoryActionSource(source, memory)

    with narration_scope(RunNarrator(tmp_path)):
        wrapped.plan(
            PlanningRequest(
                observation=before,
                objective="Open the record form.",
                completion_conditions=("The record form is visible.",),
                available_actions=frozenset({ActionKind.DONE}),
            )
        )

    assert source.request is not None
    assert wrapped.decision_metadata() == source.decision_metadata()
    assert source.request.progress_context is not None
    historical_facts = source.request.progress_context.historical_memory_facts
    assert any("Previous unsuccessful attempt" in fact for fact in historical_facts)
    assert any("Previously grounded target" in fact for fact in historical_facts)
    assert all("unrelated settings menu" not in fact for fact in historical_facts)
    assert source.request.known_facts == ()
    narration = (tmp_path / "runtime/yewo-session.log").read_text()
    assert "Reusing 2 app-memory fact(s) for the planner." in narration
    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "lookup=hit reason=compatible_screen_evidence facts=2" in trace
    transient_before = before.model_copy(
        update={
            "screen_fingerprint": before.screen_fingerprint.model_copy(
                update={"value": "0000000000000003"}
            )
        }
    )
    assert memory_planning_facts(memory, transient_before) == historical_facts
    destination_facts = memory_planning_facts(memory, after)
    assert all("unrelated settings menu" in fact for fact in destination_facts)


def test_memory_wrapper_does_not_hide_delegated_source_capabilities() -> None:
    from yewo.agent.planning.planner import (
        ActionVerificationContext,
        source_accepts_planning_focus,
        resolve_action_verification_context,
    )
    from yewo.agent.planning.evidence import resolve_planning_model_request_count

    class CapableSource(RecordingActionSource):
        def declines_planning_focus(self) -> bool:
            return True

        def last_planning_model_request_count(self) -> int:
            return 0

        def action_verification_context(
            self,
            default_request: PlanningRequest,
        ) -> ActionVerificationContext:
            return ActionVerificationContext(request=default_request, assess_task_progress=False)

    screen = observation("obs_1", "0000000000000000")
    wrapped = AppMemoryActionSource(
        CapableSource(),
        cast(AppMemory, SimpleNamespace(screens=(), transitions=(), unsuccessful_attempts=())),
    )
    request = PlanningRequest(
        observation=screen,
        objective="Open the record form.",
        completion_conditions=("The record form is visible.",),
        available_actions=frozenset({ActionKind.DONE}),
    )

    # Capability protocols are method-presence checks on the outermost object, so a
    # wrapper that cannot forward them silently downgrades the inner source.
    assert source_accepts_planning_focus(wrapped) is False
    assert resolve_planning_model_request_count(wrapped, 3) == 0
    assert resolve_action_verification_context(wrapped, request).assess_task_progress is False
