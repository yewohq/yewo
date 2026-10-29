"""Platform-neutral text entry around platform-owned device mechanics."""

from __future__ import annotations

from typing import Protocol

from yewo.agent.device.focus import FocusedFieldReading
from yewo.agent.run.input_transaction import FocusReceipt
from yewo.agent.run.postconditions import TextEntryReceipt


class InputDevice(Protocol):
    def type(self, text: str, x: int, y: int) -> str | None: ...

    def read_focused_field(self) -> FocusedFieldReading: ...

    def is_keyboard_open(self) -> bool | None: ...


class DeviceInputAdapter:
    """Collect portable evidence while DeviceControl owns Android/iOS details."""

    def __init__(self, device: InputDevice):
        self._device = device

    def enter_text(self, text: str, x: int, y: int) -> TextEntryReceipt:
        before = self._focused_field()
        keyboard_before = self._keyboard_state()
        observed = self._device.type(text, x, y)
        after = self._focused_field()
        keyboard_after = self._keyboard_state()
        return TextEntryReceipt(
            observed=observed if isinstance(observed, str) else None,
            focus=FocusReceipt(
                before_bounds=before.bounds if before is not None else None,
                after_bounds=after.bounds if after is not None else None,
                keyboard_before=keyboard_before,
                keyboard_after=keyboard_after,
            ),
        )

    def _focused_field(self) -> FocusedFieldReading | None:
        try:
            return self._device.read_focused_field()
        except Exception:  # noqa: BLE001 - optional device evidence
            return None

    def _keyboard_state(self) -> bool | None:
        try:
            return self._device.is_keyboard_open()
        except Exception:  # noqa: BLE001 - optional device evidence
            return None
