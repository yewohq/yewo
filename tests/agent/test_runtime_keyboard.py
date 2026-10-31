"""Keyboard-state planning tests for the shared Device Agent runtime."""

from pathlib import Path

from yewo.agent.actuation import ActionKind, DispatchDisposition
from yewo.devices.errors import DeviceTargetRejectedError
from yewo.runs import NeverInterrupted

from agent.actuation.support import proposal
from agent.test_runtime import (
    FailingGrounder,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


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


def test_rejected_text_target_returns_to_planner_for_keyboard_dismissal(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner(
        [
            proposal(ActionKind.TYPE),
            proposal(ActionKind.DISMISS_KEYBOARD),
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

    assert result.steps[0].receipt is not None
    assert result.steps[0].receipt.disposition is DispatchDisposition.TARGET_REJECTED
    assert planner.requests[1].history[0].receipt == result.steps[0].receipt
    assert ActionKind.DISMISS_KEYBOARD in planner.requests[1].available_actions
    assert [name for name, unused in device.calls] == ["dismiss_keyboard"]
    assert result.usage.dispatched_actions == 1
