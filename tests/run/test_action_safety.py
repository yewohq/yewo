"""Every device action source crosses one authorization boundary."""

from unittest.mock import Mock, patch

import pytest

from yewo.agent.actuation.actions import (
    Coordinates,
    KeyEventAction,
    ShowKeyboardTargetsAction,
    SwipeAction,
    TapAction,
)
from yewo.agent.actuation.safety import (
    ActionSafetyPolicy,
    ActionSource,
    execute_authorized,
)
from yewo.agent.run.instruction_controller import InstructionController


def _tap() -> TapAction:
    return TapAction(coordinates=Coordinates(x=100, y=200), reason="Open the editor")


def test_authorization_binds_action_to_observation_context_and_attempt() -> None:
    controller = InstructionController.from_goal("Open editor")

    authorization = ActionSafetyPolicy().authorize(
        _tap(),
        source=ActionSource.PLANNER,
        observation_id="observation-1",
        screen_phash="a" * 64,
        screen_size=(1000, 2000),
        context_id="context-1",
        instruction_controller=controller,
    )

    assert authorization.observation_id == "observation-1"
    assert authorization.context_id == "context-1"
    assert authorization.instruction_id == controller.instructions[0].instruction_id
    assert authorization.attempt_id is not None
    assert authorization.control_identity.points == ((0.1, 0.1),)


def test_replay_rejects_unsafe_actions_and_non_back_key_events() -> None:
    policy = ActionSafetyPolicy()
    swipe = SwipeAction(
        start_x=0,
        start_y=100,
        end_x=0,
        end_y=0,
        duration=100,
        reason="Scroll",
    )
    unsupported_key = KeyEventAction(key=3, reason="Leave the app")

    for action in (swipe, unsupported_key):
        with pytest.raises(ValueError, match="Replay cannot"):
            policy.authorize(
                action,
                source=ActionSource.REPLAY,
                observation_id="observation-1",
                screen_phash="a" * 64,
                screen_size=(1000, 2000),
            )


def test_replay_allows_taps_but_not_unverified_back_navigation() -> None:
    policy = ActionSafetyPolicy()
    authorization = policy.authorize(
        _tap(),
        source=ActionSource.REPLAY,
        observation_id="observation-1",
        screen_phash="a" * 64,
        screen_size=(1000, 2000),
    )
    assert authorization.action == _tap()
    with pytest.raises(ValueError, match="Replay cannot"):
        policy.authorize(
            KeyEventAction(key=4, reason="Navigate back"),
            source=ActionSource.REPLAY,
            observation_id="observation-2",
            screen_phash="a" * 64,
            screen_size=(1000, 2000),
        )


def test_back_cannot_be_used_as_keyboard_dismissal() -> None:
    with pytest.raises(ValueError, match="cannot dismiss the keyboard"):
        ActionSafetyPolicy().authorize(
            KeyEventAction(key=4, reason="Hide keyboard"),
            source=ActionSource.PLANNER,
            observation_id="observation-1",
            screen_phash="a" * 64,
            screen_size=(1000, 2000),
            keyboard_visible=True,
        )


def test_back_requires_a_previous_observed_destination() -> None:
    policy = ActionSafetyPolicy()
    action = KeyEventAction(key=4, reason="Return to the schedule list")

    with pytest.raises(ValueError, match="destination postcondition"):
        policy.authorize(
            action,
            source=ActionSource.PLANNER,
            observation_id="observation-1",
            screen_phash="b" * 64,
            screen_size=(1000, 2000),
        )

    authorization = policy.authorize(
        action,
        source=ActionSource.PLANNER,
        observation_id="observation-1",
        screen_phash="b" * 64,
        screen_size=(1000, 2000),
        expected_destination_phash="a" * 64,
    )
    assert authorization.expected_destination_phash == "a" * 64


def test_non_device_and_unowned_actions_are_rejected() -> None:
    policy = ActionSafetyPolicy()
    reveal = ShowKeyboardTargetsAction(reason="Inspect keyboard controls")

    with pytest.raises(ValueError, match="exact source observation"):
        policy.authorize(
            _tap(),
            source=ActionSource.PLANNER,
            observation_id="",
            screen_phash=None,
            screen_size=None,
        )
    with pytest.raises(ValueError, match="non-device"):
        policy.authorize(
            reveal,
            source=ActionSource.PLANNER,
            observation_id="observation-1",
            screen_phash=None,
            screen_size=None,
        )


@patch("yewo.agent.actuation.safety.ActionOrchestrator")
def test_dispatch_accepts_only_an_authorization(orchestrator: Mock) -> None:
    authorization = ActionSafetyPolicy().authorize(
        _tap(),
        source=ActionSource.PLANNER,
        observation_id="observation-1",
        screen_phash="a" * 64,
        screen_size=(1000, 2000),
    )

    execute_authorized(authorization, object())

    orchestrator.return_value.execute.assert_called_once_with(authorization.action)
