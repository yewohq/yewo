"""Fail-closed action-policy tests with no Appium dependency."""

import pytest
from pydantic import ValidationError

from yewo.agent.actuation import (
    ActionKind,
    ActionPolicy,
    ActionProposal,
    ActionRefusal,
    ActionRefusalCode,
    ActionSource,
    AuthorizedAction,
    ScrollDistance,
    SideEffectClass,
    TapRepetition,
)
from yewo.devices.application_lifecycle import ApplicationState
from yewo.devices.inventory import DevicePlatform
from yewo.devices.screen import SystemRegionKind
from yewo.grounding import GroundingDisposition, GroundingTargetOperation
from yewo.models import ModelCallDisposition
from yewo.observations import ScreenRegion

from agent.actuation.support import budget, limits, point_grounding, proposal, region_grounding
from observations.factories import build_observation


def test_authorizes_current_grounded_tap_and_binds_evidence() -> None:
    result = ActionPolicy(limits(), lambda: "authorization-1").authorize(
        proposal(), build_observation(), budget(), target_grounding=point_grounding()
    )

    assert isinstance(result, AuthorizedAction)
    assert result.authorization_id == "authorization-1"
    assert result.source_observation_id == "obs_1"
    assert result.source_screenshot_sha256 == "a" * 64
    assert result.expected_outcome == "The editor opens."
    assert result.target is not None
    assert (result.target.x, result.target.y) == (100, 200)


@pytest.mark.parametrize(
    ("candidate", "grounding", "expected"),
    [
        (
            proposal(source_observation_id="obs_old"),
            point_grounding(),
            ActionRefusalCode.STALE_OBSERVATION,
        ),
        (
            proposal(),
            point_grounding(screenshot_sha256="f" * 64),
            ActionRefusalCode.STALE_GROUNDING,
        ),
    ],
)
def test_rejects_stale_evidence(
    candidate: ActionProposal, grounding: object, expected: object
) -> None:
    result = ActionPolicy(limits()).authorize(
        candidate,
        build_observation(),
        budget(),
        target_grounding=grounding,  # type: ignore[arg-type]
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is expected


@pytest.mark.parametrize(
    "grounding",
    [
        point_grounding(expression="a different button"),
        point_grounding(operation=GroundingTargetOperation.TEXT_FIELD),
        point_grounding(disposition=GroundingDisposition.REFUSED),
    ],
)
def test_rejects_grounding_that_does_not_match_the_proposal(grounding: object) -> None:
    result = ActionPolicy(limits()).authorize(
        proposal(),
        build_observation(),
        budget(),
        target_grounding=grounding,  # type: ignore[arg-type]
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.INVALID_GROUNDING


def test_reports_provider_unavailability_without_rejecting_the_target() -> None:
    grounding = point_grounding(
        disposition=GroundingDisposition.PROVIDER_UNAVAILABLE,
        transport_disposition=ModelCallDisposition.PROVIDER_ERROR,
        reason="RateLimitError (HTTP 429)",
    )

    result = ActionPolicy(limits()).authorize(
        proposal(), build_observation(), budget(), target_grounding=grounding
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.PROVIDER_UNAVAILABLE
    assert result.reason == "RateLimitError (HTTP 429)"


def test_rejects_grounding_in_system_owned_region() -> None:
    observation = build_observation(
        system_regions=(
            ScreenRegion(
                kind=SystemRegionKind.KEYBOARD,
                left=0,
                top=150,
                right=200,
                bottom=400,
            ),
        )
    )

    result = ActionPolicy(limits()).authorize(
        proposal(), observation, budget(), target_grounding=point_grounding()
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.INVALID_GROUNDING


def test_rejects_action_budget_exhaustion() -> None:
    policy = ActionPolicy(limits(maximum_actions=1))

    action_result = policy.authorize(
        proposal(),
        build_observation(),
        budget(dispatched_actions=1),
        target_grounding=point_grounding(),
    )
    assert isinstance(action_result, ActionRefusal)
    assert action_result.code is ActionRefusalCode.ACTION_BUDGET_EXHAUSTED


@pytest.mark.parametrize(
    "action",
    [
        ActionKind.TAP,
        ActionKind.DOUBLE_TAP,
        ActionKind.LONG_PRESS,
        ActionKind.DRAG,
        ActionKind.SCROLL,
        ActionKind.BACK,
    ],
)
def test_visible_keyboard_blocks_pointer_and_navigation_actions(action: ActionKind) -> None:
    candidate = proposal(action)
    result = ActionPolicy(limits(allow_exploratory_back=True)).authorize(
        candidate,
        build_observation(keyboard_visible=True),
        budget(),
        target_grounding=(point_grounding() if candidate.target_expression else None),
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.KEYBOARD_RULE


def test_keyboard_rule_can_refuse_before_grounding() -> None:
    result = ActionPolicy(limits()).refusal_before_grounding(
        proposal(),
        build_observation(keyboard_visible=True),
        budget(),
    )

    assert result is not None
    assert result.code is ActionRefusalCode.KEYBOARD_RULE


@pytest.mark.parametrize("action", [ActionKind.TYPE, ActionKind.CLEAR])
def test_text_editing_requires_a_focused_field(action: ActionKind) -> None:
    result = ActionPolicy(limits()).refusal_before_grounding(
        proposal(action),
        build_observation(keyboard_visible=False),
        budget(),
    )

    assert result is not None
    assert result.code is ActionRefusalCode.KEYBOARD_RULE


def test_targeted_type_can_focus_a_known_field_before_typing() -> None:
    candidate = proposal(ActionKind.TYPE, target_expression="the temperature field")
    result = ActionPolicy(limits()).authorize(
        candidate,
        build_observation(keyboard_visible=False),
        budget(),
        target_grounding=point_grounding(
            expression="the temperature field",
            operation=GroundingTargetOperation.TEXT_FIELD,
        ),
    )

    assert isinstance(result, AuthorizedAction)
    assert result.target is not None


def test_focus_remains_available_while_the_keyboard_is_visible() -> None:
    candidate = proposal(ActionKind.FOCUS)

    result = ActionPolicy(limits()).authorize(
        candidate,
        build_observation(keyboard_visible=True),
        budget(),
        target_grounding=point_grounding(
            expression=candidate.target_expression or "the name field",
            operation=GroundingTargetOperation.TEXT_FIELD,
        ),
    )

    assert isinstance(result, AuthorizedAction)


def test_enforces_application_scope() -> None:
    app_result = ActionPolicy(limits()).authorize(
        proposal(),
        build_observation(
            application_state=ApplicationState.BACKGROUND,
            foreground_application_id="com.example.other",
        ),
        budget(),
        target_grounding=point_grounding(),
    )

    assert isinstance(app_result, ActionRefusal)
    assert app_result.code is ActionRefusalCode.APPLICATION_SCOPE


def test_authorizes_a_grounded_action_on_android_permission_controller() -> None:
    result = ActionPolicy(limits()).authorize(
        proposal(),
        build_observation(
            application_state=ApplicationState.BACKGROUND,
            foreground_application_id="com.google.android.permissioncontroller",
        ),
        budget(),
        target_grounding=point_grounding(),
    )

    assert isinstance(result, AuthorizedAction)


def test_back_requires_a_destination_guard_unless_mode_allows_exploration() -> None:
    guarded = ActionPolicy(limits()).authorize(
        proposal(ActionKind.BACK),
        build_observation(),
        budget(),
        expected_back_destination_observation_id="obs_previous",
    )
    unguarded = ActionPolicy(limits()).authorize(
        proposal(ActionKind.BACK), build_observation(), budget()
    )

    assert isinstance(guarded, AuthorizedAction)
    assert guarded.expected_back_destination_observation_id == "obs_previous"
    assert isinstance(unguarded, ActionRefusal)
    assert unguarded.code is ActionRefusalCode.INVALID_CONSTRAINT


def test_back_is_not_supported_on_ios() -> None:
    android_observation = build_observation()
    observation = build_observation(
        device=android_observation.device.model_copy(update={"platform": DevicePlatform.IOS}),
        application=android_observation.application.model_copy(
            update={"platform": DevicePlatform.IOS}
        ),
    )

    result = ActionPolicy(limits(allow_exploratory_back=True)).authorize(
        proposal(ActionKind.BACK), observation, budget()
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.UNSUPPORTED_ACTION


def test_targeted_scroll_requires_matching_region_grounding() -> None:
    candidate = proposal(
        ActionKind.SCROLL,
        target_expression="the results list",
        direction="down",
        distance=ScrollDistance.SHORT,
    )

    result = ActionPolicy(limits()).authorize(
        candidate, build_observation(), budget(), target_grounding=region_grounding()
    )

    assert isinstance(result, AuthorizedAction)
    assert result.region is not None


def test_mode_with_keyboard_blocked_actions_requires_keyboard_dismissal() -> None:
    with pytest.raises(ValidationError, match="keyboard dismissal"):
        limits(
            allowed_actions=frozenset({ActionKind.TAP}),
            supported_actions=frozenset({ActionKind.TAP}),
        )


def test_drag_requires_separate_source_and_destination_groundings() -> None:
    candidate = proposal(ActionKind.DRAG)

    result = ActionPolicy(limits()).authorize(
        candidate,
        build_observation(),
        budget(),
        target_grounding=point_grounding(
            expression="the first item",
            operation=GroundingTargetOperation.DRAG_SOURCE,
            x=50,
        ),
        destination_grounding=point_grounding(
            grounding_id="grounding-2",
            expression="the second position",
            operation=GroundingTargetOperation.DRAG_DESTINATION,
            x=150,
        ),
    )

    assert isinstance(result, AuthorizedAction)
    assert result.target is not None
    assert result.destination is not None


@pytest.mark.parametrize(
    ("candidate", "platform"),
    [
        (proposal(ActionKind.WAIT, duration_seconds=0), DevicePlatform.ANDROID),
        (proposal(ActionKind.LONG_PRESS, duration_seconds=0.05), DevicePlatform.ANDROID),
        (proposal(ActionKind.DRAG, duration_seconds=0.05), DevicePlatform.ANDROID),
        (proposal(ActionKind.DRAG, duration_seconds=0.4), DevicePlatform.IOS),
    ],
)
def test_rejects_duration_below_the_platform_minimum(
    candidate: ActionProposal, platform: DevicePlatform
) -> None:
    base = build_observation()
    observation = build_observation(
        device=base.device.model_copy(update={"platform": platform}),
        application=base.application.model_copy(update={"platform": platform}),
    )

    result = ActionPolicy(limits()).authorize(candidate, observation, budget())

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.INVALID_CONSTRAINT


def test_repeated_tap_requires_a_deterministic_test_source() -> None:
    repetition = TapRepetition(count=4, interval_seconds=0.1)
    planner_result = ActionPolicy(limits()).authorize(
        proposal(),
        build_observation(),
        budget(),
        target_grounding=point_grounding(),
        tap_repetition=repetition,
    )
    deterministic_result = ActionPolicy(limits()).authorize(
        proposal(),
        build_observation(),
        budget(),
        target_grounding=point_grounding(),
        tap_repetition=repetition,
        source=ActionSource.DETERMINISTIC_TEST,
    )

    assert isinstance(planner_result, ActionRefusal)
    assert planner_result.code is ActionRefusalCode.ACTION_NOT_ALLOWED
    assert isinstance(deterministic_result, AuthorizedAction)
    assert deterministic_result.tap_count == 4
    assert deterministic_result.tap_interval_seconds == 0.1


def test_side_effect_policy_rejects_unclassified_tap() -> None:
    restricted = limits(
        allowed_side_effects=frozenset({SideEffectClass.NONE, SideEffectClass.NAVIGATION})
    )

    result = ActionPolicy(restricted).authorize(
        proposal(), build_observation(), budget(), target_grounding=point_grounding()
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.SIDE_EFFECT_NOT_ALLOWED


def test_capability_action_is_explicitly_deferred() -> None:
    result = ActionPolicy(limits()).authorize(
        proposal(ActionKind.SET_DEVICE_CAPABILITY), build_observation(), budget()
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.UNSUPPORTED_ACTION


def test_terminal_signals_are_typed_but_never_dispatched() -> None:
    result = ActionPolicy(limits()).authorize(
        proposal(ActionKind.DONE), build_observation(), budget()
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.NON_DISPATCH_ACTION


def test_malformed_proposal_fails_validation() -> None:
    with pytest.raises(ValidationError):
        ActionProposal(
            proposal_id="proposal-1",
            source_observation_id="obs_1",
            action=ActionKind.TYPE,
            target_expression="the name field",
        )


@pytest.mark.parametrize(
    ("disposition", "expected"),
    [
        (GroundingDisposition.REFUSED, "could not locate the described target"),
        (GroundingDisposition.AMBIGUOUS, "more than one match"),
        (GroundingDisposition.MALFORMED, "no usable coordinate"),
        (GroundingDisposition.OUT_OF_BOUNDS, "platform-owned region"),
    ],
)
def test_a_grounding_refusal_reports_which_outcome_blocked_the_action(
    disposition: GroundingDisposition,
    expected: str,
) -> None:
    # The planner's next move differs per outcome, so one shared reason string hides it.
    result = ActionPolicy(limits()).authorize(
        proposal(),
        build_observation(),
        budget(),
        target_grounding=point_grounding(disposition=disposition),
    )

    assert isinstance(result, ActionRefusal)
    assert result.code is ActionRefusalCode.INVALID_GROUNDING
    assert disposition.value in result.reason
    assert expected in result.reason
