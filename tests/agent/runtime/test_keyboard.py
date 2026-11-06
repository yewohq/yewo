"""Keyboard-state planning tests for the shared Device Agent runtime."""

from pathlib import Path

from yewo.agent.actuation import ActionKind, DispatchDisposition, LiteralText
from yewo.devices.errors import DeviceInputRejectedError, DeviceTargetRejectedError
from yewo.runs import NeverInterrupted

from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)
from grounding.support import FailingGrounder


def test_keyboard_state_filters_planner_actions_and_keeps_direct_type(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner([proposal(ActionKind.TYPE), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", keyboard_visible=True),
            _observation(
                "obs_2",
                "1111111111111111",
                keyboard_visible=True,
                offset_seconds=1,
            ),
            _observation(
                "obs_3",
                "1111111111111111",
                keyboard_visible=True,
                offset_seconds=2,
            ),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    available = planner.requests[0].available_actions
    assert ActionKind.TYPE in available
    assert ActionKind.CLEAR in available
    assert ActionKind.FOCUS in available
    assert ActionKind.DISMISS_KEYBOARD in available
    assert ActionKind.TAP not in available
    assert ActionKind.SCROLL not in available
    assert result.usage.grounding_requests == 0


def test_hidden_keyboard_offers_focus_but_not_text_editing(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.CANNOT_PROGRESS)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", keyboard_visible=False)]
    )

    _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    available = planner.requests[0].available_actions
    assert ActionKind.FOCUS in available
    assert ActionKind.TYPE not in available
    assert ActionKind.CLEAR not in available


def test_known_input_contract_offers_targeted_type_with_hidden_keyboard(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.CANNOT_PROGRESS)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", keyboard_visible=False)]
    )
    task = _task().model_copy(update={"targeted_type_available": True})

    _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        task, _mode_policy(), NeverInterrupted()
    )

    available = planner.requests[0].available_actions
    assert ActionKind.TYPE in available
    assert ActionKind.CLEAR not in available


def test_state_refusal_happens_before_grounding(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(), proposal(ActionKind.CANNOT_PROGRESS)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", keyboard_visible=True)]
    )

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        grounder=FailingGrounder(),
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.steps[0].refusal is not None
    assert result.steps[0].refusal.code.value == "keyboard_rule"
    assert result.usage.grounding_requests == 0


class _RejectFirstTextTarget(RuntimeDevice):
    def __init__(self) -> None:
        super().__init__()
        self._rejected = False

    def type_focused_text(self, text: str) -> bool | None:
        if not self._rejected:
            self._rejected = True
            raise DeviceTargetRejectedError("The active field rejected text entry.")
        return super().type_focused_text(text)


class _RejectIncompatibleInput(RuntimeDevice):
    def type_focused_text(self, text: str) -> bool | None:
        if text == "medium":
            raise DeviceInputRejectedError(
                "The focused field accepts numeric input. Choose a compatible numeric value."
            )
        return super().type_focused_text(text)


def test_pre_dispatch_text_rejection_is_not_dispatched(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [
            proposal(ActionKind.TYPE),
            proposal(ActionKind.DONE),
        ]
    )
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", keyboard_visible=True),
            _observation("obs_2", "1111111111111111", keyboard_visible=False),
            _observation(
                "obs_3",
                "1111111111111111",
                keyboard_visible=False,
                offset_seconds=1,
            ),
        ]
    )
    device = _RejectFirstTextTarget()

    result = _runtime(tmp_path, planner, observations, device).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    receipt = result.steps[0].receipt
    assert receipt is not None
    assert receipt.disposition is DispatchDisposition.TARGET_REJECTED
    assert receipt.reason == "The active field rejected text entry."
    assert result.steps[0].outcome is None
    assert result.steps[0].settled_observation_id is None
    assert planner.requests[1].observation.observation_id == "obs_1"
    assert planner.requests[1].history[0].receipt == result.steps[0].receipt
    assert planner.requests[1].history[0].destination_screen is None
    assert [name for name, unused in device.calls] == []
    assert result.usage.dispatched_actions == 0


def test_rejected_input_returns_to_planner_and_allows_one_corrected_value(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [
            proposal(
                ActionKind.TYPE,
                text=LiteralText(value="medium"),
                target_expression="the focused numeric field",
            ),
            proposal(ActionKind.TYPE, text=LiteralText(value="5")),
            proposal(ActionKind.DONE),
        ]
    )
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", keyboard_visible=True),
            _observation(
                "obs_2", "1111111111111111", keyboard_visible=True, offset_seconds=1
            ),
            _observation(
                "obs_3", "1111111111111111", keyboard_visible=True, offset_seconds=2
            ),
        ]
    )
    device = _RejectIncompatibleInput()

    result = _runtime(tmp_path, planner, observations, device).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    rejected = result.steps[0]
    assert rejected.receipt is not None
    assert rejected.receipt.disposition is DispatchDisposition.INPUT_REJECTED
    assert planner.requests[1].history[0].receipt == rejected.receipt
    assert planner.requests[1].history[0].grounded_target_point is None
    assert [name for name, unused in device.calls] == ["focus", "type_focused_text"]
    assert device.active_text == "5"
    assert result.usage.dispatched_actions == 1
