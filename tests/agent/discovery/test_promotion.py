from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from yewo.agent import (
    AgentResult,
    AgentTerminalReason,
    AgentUsage,
    InputValueShape,
    MemoryCompatibility,
    MemoryRetentionPolicy,
    ScreenComparisonPolicy,
)
from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    DispatchDisposition,
)
from yewo.artifacts import EvidenceApplicationIdentity
from yewo.devices.inventory import DevicePlatform
from yewo.grounding import GroundedPoint, ModelGroundingResolution
from yewo.agent.discovery import (
    AppMemoryBuilder,
    DiscoveryObjective,
    DiscoveryObjectiveState,
    DiscoveryRequest,
)

from agent.discovery.support import agent_result, observation
from agent.replay.test_procedure_budget import _procedure as compiled_procedure_fixture


def _request() -> DiscoveryRequest:
    return DiscoveryRequest(
        request_id="request_1",
        objectives=(
            DiscoveryObjective(
                objective_id="record_form",
                statement="The record form is reachable.",
            ),
        ),
    )


def _compatibility() -> MemoryCompatibility:
    return MemoryCompatibility(
        application=EvidenceApplicationIdentity(
            platform=DevicePlatform.ANDROID,
            application_id="com.example.app",
            version="2.1.0",
            build="42",
            artifact_sha256="b" * 64,
        )
    )


def _promote(
    *,
    disposition=ActionOutcomeDisposition.OBSERVED,
    terminal=None,
    action=None,
    compiled_procedure=None,
    evidence_observation_ids=(),
):
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(
        before,
        after,
        disposition=disposition,
        terminal=terminal or AgentTerminalReason.DONE_REQUESTED,
        action=action or ActionKind.TAP,
    )
    return AppMemoryBuilder().promote(
        _request(),
        result,
        {before.observation_id: before, after.observation_id: after},
        _compatibility(),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        compiled_procedure=compiled_procedure,
        evidence_observation_ids=evidence_observation_ids,
    )


def test_memory_requires_a_versioned_application_build() -> None:
    with pytest.raises(ValidationError, match="artifact or versioned build"):
        MemoryCompatibility(
            application=EvidenceApplicationIdentity(
                platform=DevicePlatform.ANDROID,
                application_id="com.example.app",
            )
        )


def test_done_promotes_one_transition_and_exact_source_grounding() -> None:
    promotion = _promote(
        compiled_procedure=compiled_procedure_fixture(),
        evidence_observation_ids=("obs_1", "obs_2"),
    )

    assert promotion.memory is not None
    assert len(promotion.memory.transitions) == 1
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.SUPPORTED
    serialized = promotion.memory.model_dump_json()
    assert '"target_description":"the add button"' in serialized
    assert '"target"' not in serialized
    transition = promotion.memory.transitions[0]
    assert transition.target_grounding is not None
    assert isinstance(transition.target_grounding.location, GroundedPoint)
    assert transition.target_grounding.location.pixel.x == 100
    assert transition.target_grounding.location.pixel.y == 200
    assert (
        transition.target_grounding.source_masked_pixels_sha256
        == promotion.memory.screens[0].variants[0].presentation.fingerprint.masked_pixels_sha256
    )
    assert transition.outcome_signals
    assert isinstance(transition.target_grounding.resolution, ModelGroundingResolution)
    assert transition.target_grounding.resolution.grounding_profile_id == "grounder"


@pytest.mark.parametrize(
    "disposition",
    [
        ActionOutcomeDisposition.AMBIGUOUS,
        ActionOutcomeDisposition.UNAVAILABLE,
    ],
)
def test_unaccepted_outcome_cannot_become_memory(
    disposition: ActionOutcomeDisposition,
) -> None:
    promotion = _promote(disposition=disposition)

    assert promotion.memory is None
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED


def test_no_observable_change_becomes_bounded_attempt_history() -> None:
    promotion = _promote(disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE)

    assert promotion.memory is not None
    assert promotion.memory.transitions == ()
    attempt = promotion.memory.unsuccessful_attempts[0]
    assert attempt.disposition.value == "no_observable_change"
    assert attempt.target_description == "the add button"
    assert attempt.grounded_point is not None
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED


def test_target_rejection_retains_device_reason_and_grounded_point() -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(before, after)
    action_step = result.steps[0]
    assert action_step.receipt is not None
    rejected_receipt = action_step.receipt.model_copy(
        update={
            "disposition": DispatchDisposition.TARGET_REJECTED,
            "reason": "The grounded target did not accept focus.",
        }
    )
    rejected_step = action_step.model_copy(
        update={
            "receipt": rejected_receipt,
            "settlement_observation_ids": (),
            "settled_observation_id": None,
            "outcome": None,
        }
    )
    result = result.model_copy(
        update={
            "final_observation_id": before.observation_id,
            "usage": result.usage.model_copy(update={"dispatched_actions": 0}),
            "steps": (rejected_step,),
        }
    )

    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {before.observation_id: before},
        _compatibility(),
        verified_objective_ids=frozenset(),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is not None
    attempt = promotion.memory.unsuccessful_attempts[0]
    assert attempt.disposition.value == "target_rejected"
    assert attempt.reason == "The grounded target did not accept focus."
    assert attempt.grounded_point is not None
    assert (attempt.grounded_point.x, attempt.grounded_point.y) == (100, 200)
    assert attempt.screenshot_width_px == before.screenshot.width_px
    assert attempt.screenshot_height_px == before.screenshot.height_px


def test_unsuccessful_attempt_retention_keeps_only_the_latest_evidence() -> None:
    first = observation("first", "0000000000000000")
    first_after = observation("first_after", "1111111111111111")
    compatibility = _compatibility().model_copy(
        update={"retention": MemoryRetentionPolicy(maximum_unsuccessful_attempts=1)}
    )
    first_promotion = AppMemoryBuilder().promote(
        _request(),
        agent_result(
            first,
            first_after,
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        ),
        {first.observation_id: first, first_after.observation_id: first_after},
        compatibility,
        verified_objective_ids=frozenset(),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )
    assert first_promotion.memory is not None
    second = observation("second", "2222222222222222")
    second_after = observation("second_after", "3333333333333333")

    second_promotion = AppMemoryBuilder().promote(
        _request(),
        agent_result(
            second,
            second_after,
            disposition=ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE,
        ),
        {second.observation_id: second, second_after.observation_id: second_after},
        compatibility,
        verified_objective_ids=frozenset(),
        source_run_id="run_2",
        updated_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        existing_memory=first_promotion.memory,
    )

    assert second_promotion.memory is not None
    assert len(second_promotion.memory.unsuccessful_attempts) == 1
    assert second_promotion.memory.unsuccessful_attempts[0].source_run_id == "run_2"


def test_application_exit_cannot_become_a_reusable_transition() -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(before, after)
    action_step = result.steps[0]
    assert action_step.outcome is not None
    action_step = action_step.model_copy(
        update={
            "outcome": action_step.outcome.model_copy(
                update={"signals": frozenset({ActionOutcomeSignal.APP_EXITED})}
            )
        }
    )
    result = result.model_copy(update={"steps": (action_step, result.steps[1])})

    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {before.observation_id: before, after.observation_id: after},
        _compatibility(),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is None
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED


def test_budget_terminal_preserves_transition_without_claiming_coverage() -> None:
    promotion = _promote(terminal=AgentTerminalReason.BUDGET_EXHAUSTED)

    assert promotion.memory is not None
    assert len(promotion.memory.transitions) == 1
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED


def test_nearby_fingerprints_become_variants_of_one_logical_screen() -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "0000000000000001",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(before, after)
    action_step = result.steps[0]
    assert action_step.outcome is not None
    result = result.model_copy(
        update={
            "steps": (
                action_step.model_copy(
                    update={
                        "outcome": action_step.outcome.model_copy(
                            update={
                                "signals": frozenset(
                                    {
                                        ActionOutcomeSignal.PIXELS_CHANGED,
                                        ActionOutcomeSignal.FOCUS_CHANGED,
                                    }
                                )
                            }
                        )
                    }
                ),
                result.steps[1],
            )
        }
    )

    compatibility = _compatibility().model_copy(
        update={"screen_comparison": ScreenComparisonPolicy(maximum_phash_distance=1)}
    )
    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {before.observation_id: before, after.observation_id: after},
        compatibility,
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is not None
    assert len(promotion.memory.screens) == 1
    assert len(promotion.memory.screens[0].variants) == 2


def test_incompatible_masks_cannot_be_grouped() -> None:
    before = observation("obs_1", "0000000000000000", mask_sha256="a" * 64)
    after = observation(
        "obs_2",
        "0000000000000000",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
        mask_sha256="c" * 64,
    )
    result = agent_result(before, after)

    with pytest.raises(ValidationError, match="canonical effective mask"):
        AppMemoryBuilder().promote(
            _request(),
            result,
            {before.observation_id: before, after.observation_id: after},
            _compatibility(),
            verified_objective_ids=frozenset({"record_form"}),
            source_run_id="run_1",
            updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
        )


def test_disposable_literal_text_retains_only_its_non_secret_input_shape() -> None:
    promotion = _promote(action=ActionKind.TYPE)

    assert promotion.memory is not None
    action = promotion.memory.transitions[0].action
    assert action.declared_value_name is None
    assert action.input_contract is not None
    assert action.input_contract.value_shape is InputValueShape.TEXT
    assert "hello" not in promotion.memory.model_dump_json()
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED


def test_focus_then_literal_type_learns_a_targeted_input_contract() -> None:
    first = observation("obs_1", "0000000000000000")
    focused = observation(
        "obs_2",
        "0101010101010101",
        predecessor_observation_id=first.observation_id,
        offset_seconds=1,
    )
    typed = observation(
        "obs_3",
        "1111111111111111",
        predecessor_observation_id=focused.observation_id,
        offset_seconds=2,
    )
    focus_result = agent_result(first, focused, action=ActionKind.FOCUS)
    type_result = agent_result(focused, typed, action=ActionKind.TYPE)
    type_step = type_result.steps[0]
    assert type_step.outcome is not None
    type_step = type_step.model_copy(
        update={
            "step_id": "type_step",
            "step_index": 2,
            "outcome": type_step.outcome.model_copy(update={"outcome_id": "type_outcome"}),
        }
    )
    terminal_step = type_result.steps[1].model_copy(
        update={"step_id": "terminal_step", "step_index": 3}
    )
    result = AgentResult(
        terminal_reason=AgentTerminalReason.DONE_REQUESTED,
        reason="The planner requested completion.",
        final_observation_id=typed.observation_id,
        usage=AgentUsage(steps=3, planning_requests=3, grounding_requests=1, dispatched_actions=2),
        steps=(focus_result.steps[0], type_step, terminal_step),
    )

    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {item.observation_id: item for item in (first, focused, typed)},
        _compatibility(),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is not None
    type_action = promotion.memory.transitions[1].action
    assert type_action.target_description == "the name field"
    assert type_action.input_contract is not None
    assert type_action.input_contract.value_shape is InputValueShape.TEXT


@pytest.mark.parametrize("action", [ActionKind.TYPE, ActionKind.CLEAR])
def test_declared_type_and_clear_actions_can_be_reused_without_a_value(
    action: ActionKind,
) -> None:
    before = observation("obs_1", "0000000000000000")
    after = observation(
        "obs_2",
        "1111111111111111",
        predecessor_observation_id=before.observation_id,
        offset_seconds=1,
    )
    result = agent_result(
        before,
        after,
        action=action,
        declared_value_name="declared_name" if action is ActionKind.TYPE else None,
    )

    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {before.observation_id: before, after.observation_id: after},
        _compatibility(),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is not None
    assert len(promotion.memory.transitions) == 1
    serialized = promotion.memory.model_dump_json()
    assert "hello" not in serialized
    if action is ActionKind.TYPE:
        assert "declared_name" in serialized


def test_literal_text_transition_retains_only_its_input_shape() -> None:
    first = observation("obs_1", "0000000000000000")
    typed = observation(
        "obs_2",
        "0101010101010101",
        predecessor_observation_id=first.observation_id,
        offset_seconds=1,
    )
    destination = observation(
        "obs_3",
        "1111111111111111",
        predecessor_observation_id=typed.observation_id,
        offset_seconds=2,
    )
    type_result = agent_result(first, typed, action=ActionKind.TYPE)
    tap_result = agent_result(typed, destination)
    tap_step = tap_result.steps[0]
    assert tap_step.outcome is not None
    tap_step = tap_step.model_copy(
        update={
            "step_id": "tap_step",
            "step_index": 2,
            "outcome": tap_step.outcome.model_copy(update={"outcome_id": "tap_outcome"}),
        }
    )
    terminal_step = tap_result.steps[1].model_copy(
        update={"step_id": "terminal_step", "step_index": 3}
    )
    result = AgentResult(
        terminal_reason=AgentTerminalReason.DONE_REQUESTED,
        reason="The planner requested completion.",
        final_observation_id=destination.observation_id,
        usage=AgentUsage(
            steps=3,
            planning_requests=3,
            grounding_requests=2,
            dispatched_actions=2,
        ),
        steps=(type_result.steps[0], tap_step, terminal_step),
    )

    promotion = AppMemoryBuilder().promote(
        _request(),
        result,
        {
            first.observation_id: first,
            typed.observation_id: typed,
            destination.observation_id: destination,
        },
        _compatibility(),
        verified_objective_ids=frozenset({"record_form"}),
        source_run_id="run_1",
        updated_at=datetime(2026, 10, 31, tzinfo=timezone.utc),
    )

    assert promotion.memory is not None
    assert len(promotion.memory.transitions) == 2
    assert all(
        transition.action.declared_value_name is None for transition in promotion.memory.transitions
    )
    assert promotion.coverage.entries[0].state is DiscoveryObjectiveState.UNRESOLVED
