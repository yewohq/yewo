"""Keyboard dismissal safety tests."""

from devices._interaction_stubs import DriverStub, application
from yewo.devices import DevicePlatform
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.interaction import AndroidKey, DeviceInteraction


class _InputReader:
    def __init__(self, output: str) -> None:
        self.output = output

    def ask(self, connection_id: str, *args: str, timeout_seconds: float = 20) -> str:
        del connection_id, args, timeout_seconds
        return self.output


_CLOSABLE_KEYBOARD = "mInputShown=true mIsInputViewShown=true"


def _interaction(
    driver: DriverStub, input_output: str | None = _CLOSABLE_KEYBOARD
) -> DeviceInteraction:
    reader = _InputReader(input_output) if input_output is not None else None
    return DeviceInteraction(
        driver,
        DevicePlatform.ANDROID,
        application(DevicePlatform.ANDROID),
        AndroidInteractionBackend(reader, "emulator-5554"),
    )


def test_confirmed_closable_keyboard_uses_guarded_back() -> None:
    driver = DriverStub()
    driver.keyboard = True

    assert _interaction(driver).dismiss_keyboard()
    assert driver.keycodes == [AndroidKey.BACK.value]
    assert driver.back_calls == 0
    assert driver.hide_keyboard_calls == 0


def test_semantic_keyboard_fallback_runs_when_guarded_back_has_no_effect() -> None:
    driver = DriverStub()
    driver.keyboard = True
    driver.back_closes_keyboard = False

    assert _interaction(driver).dismiss_keyboard()
    assert driver.keycodes == [AndroidKey.BACK.value]
    assert driver.back_calls == 0
    assert driver.hide_keyboard_calls == 1


def test_missing_ime_evidence_uses_semantic_keyboard_command() -> None:
    driver = DriverStub()
    driver.keyboard = True

    assert _interaction(driver, input_output=None).dismiss_keyboard()
    assert driver.keycodes == []
    assert driver.hide_keyboard_calls == 1


def test_non_closable_ime_evidence_uses_semantic_keyboard_command() -> None:
    driver = DriverStub()
    driver.keyboard = True

    assert _interaction(driver, "mInputShown=true mIsInputViewShown=false").dismiss_keyboard()
    assert driver.keycodes == []
    assert driver.hide_keyboard_calls == 1


class _ContextChangingDriver(DriverStub):
    def press_keycode(self, keycode: int, metastate: int | None = None) -> object:
        answer = super().press_keycode(keycode, metastate)
        self.current_activity = "com.example.OtherActivity"
        return answer


def test_context_change_blocks_semantic_keyboard_fallback() -> None:
    driver = _ContextChangingDriver()
    driver.keyboard = True
    driver.back_closes_keyboard = False

    assert not _interaction(driver).dismiss_keyboard()
    assert driver.keycodes == [AndroidKey.BACK.value]
    assert driver.hide_keyboard_calls == 0


def test_unknown_context_blocks_semantic_keyboard_fallback() -> None:
    driver = DriverStub()
    driver.current_activity = None
    driver.keyboard = True
    driver.back_closes_keyboard = False

    assert not _interaction(driver).dismiss_keyboard()
    assert driver.hide_keyboard_calls == 0


class _FallbackContextChangingDriver(DriverStub):
    def hide_keyboard(self) -> object:
        answer = super().hide_keyboard()
        self.current_activity = "com.example.OtherActivity"
        return answer


def test_context_change_after_semantic_fallback_reports_failure() -> None:
    driver = _FallbackContextChangingDriver()
    driver.keyboard = True
    driver.back_closes_keyboard = False

    assert not _interaction(driver).dismiss_keyboard()
    assert driver.hide_keyboard_calls == 1


def test_hidden_keyboard_is_a_noop() -> None:
    driver = DriverStub()

    assert _interaction(driver).dismiss_keyboard()
    assert driver.keycodes == []
    assert driver.hide_keyboard_calls == 0
