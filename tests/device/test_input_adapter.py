from dataclasses import dataclass

import pytest

from yewo.agent.device.focus import FocusedFieldReading
from yewo.agent.device.input_adapter import DeviceInputAdapter


@dataclass
class _InputDevice:
    readings: list[FocusedFieldReading]
    keyboard_states: list[bool | None]
    observed: str | None

    def __post_init__(self) -> None:
        self.sent: list[tuple[str, int, int]] = []

    def type(self, text: str, x: int, y: int):
        self.sent.append((text, x, y))
        return self.observed

    def read_focused_field(self) -> FocusedFieldReading:
        return self.readings.pop(0)

    def is_keyboard_open(self) -> bool | None:
        return self.keyboard_states.pop(0)


@pytest.mark.parametrize("text", ["جدول التدريب", "训练计划", "🏋️‍♀️ plan"])
def test_adapter_preserves_text_and_collects_secret_free_focus_evidence(text: str) -> None:
    device = _InputDevice(
        readings=[
            FocusedFieldReading(None, (0, 0, 10, 10)),
            FocusedFieldReading(text, (50, 100, 500, 180)),
        ],
        keyboard_states=[False, True],
        observed=text,
    )

    receipt = DeviceInputAdapter(device).enter_text(text, 100, 140)

    assert device.sent == [(text, 100, 140)]
    assert receipt.observed == text
    assert receipt.focus.before_bounds == (0, 0, 10, 10)
    assert receipt.focus.after_bounds == (50, 100, 500, 180)
    assert receipt.focus.keyboard_before is False
    assert receipt.focus.keyboard_after is True
    assert text not in repr(receipt.focus)


def test_missing_auxiliary_evidence_does_not_duplicate_or_block_dispatch() -> None:
    class EvidenceUnavailable:
        calls = 0

        def read_focused_field(self):
            raise RuntimeError("driver cannot report focus")

        def is_keyboard_open(self):
            raise RuntimeError("driver cannot report keyboard")

        def type(self, text: str, x: int, y: int):
            self.calls += 1
            return None

    device = EvidenceUnavailable()

    receipt = DeviceInputAdapter(device).enter_text("value", 10, 20)

    assert device.calls == 1
    assert receipt.observed is None
    assert receipt.focus.before_bounds is None
    assert receipt.focus.after_bounds is None
    assert receipt.focus.keyboard_before is None
    assert receipt.focus.keyboard_after is None
