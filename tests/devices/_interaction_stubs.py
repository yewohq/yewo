"""Shared driver fakes for device-interaction tests."""

from typing import Any

from selenium.common.exceptions import NoAlertPresentException

from yewo.applications import ApplicationDescriptor
from yewo.devices import DevicePlatform


def png(width: int, height: int) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x06\x00\x00\x00"
    )


class ElementStub:
    def __init__(self) -> None:
        self.value = ""
        self.cleared = False
        self.focused = True
        self.requested_attributes: list[str] = []

    def send_keys(self, value: str) -> None:
        self.value += value

    def clear(self) -> None:
        self.cleared = True
        self.value = ""

    def get_attribute(self, name: str) -> str | None:
        self.requested_attributes.append(name)
        if name == "focused":
            return str(self.focused).lower()
        return self.value if name in {"value", "text"} else None


class SwitchToStub:
    def __init__(self, element: ElementStub) -> None:
        self.active_element = element
        self.has_alert = False

    @property
    def alert(self) -> object:
        if not self.has_alert:
            raise NoAlertPresentException()
        return object()


class DriverStub:
    def __init__(self) -> None:
        self.element = ElementStub()
        self.switch_to = SwitchToStub(self.element)
        self.scripts: list[tuple[str, dict[str, Any]]] = []
        self.keycodes: list[int] = []
        self.back_calls = 0
        self.keyboard = False
        self.installed = True
        self.state = 4
        self.lifecycle: list[str] = []
        self.current_package: str | None = "com.example.app"
        self.current_activity = "com.example.MainActivity"
        self.orientation = "PORTRAIT"
        self.system_bars: dict[str, object] = {
            "statusBar": {
                "visible": True,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 20,
            },
        }
        self.viewport_rect: dict[str, object] = {
            "left": 0,
            "top": 40,
            "width": 200,
            "height": 320,
        }

    def execute_script(self, script: str, params: dict[str, Any]) -> Any:
        self.scripts.append((script, params))
        if script == "mobile: activeAppInfo":
            return {"bundleId": self.current_package}
        if script == "mobile: getPressableButtons":
            return {"xctest": ["home", "lock"]}
        if script == "mobile: viewportRect":
            return self.viewport_rect
        return None

    def get_system_bars(self) -> dict[str, object]:
        return self.system_bars

    def get_screenshot_as_png(self) -> bytes:
        return png(200, 400)

    def get_window_size(self) -> dict[str, int]:
        return {"width": 100, "height": 200}

    def is_keyboard_shown(self) -> bool:
        return self.keyboard

    def hide_keyboard(self) -> object:
        self.keyboard = False
        return self

    def press_keycode(self, keycode: int, metastate: int | None = None) -> object:
        del metastate
        self.keycodes.append(keycode)
        return self

    def back(self) -> None:
        self.back_calls += 1

    def is_app_installed(self, application_id: str) -> bool:
        del application_id
        return self.installed

    def query_app_state(self, application_id: str) -> int:
        del application_id
        return self.state

    def activate_app(self, application_id: str) -> object:
        self.lifecycle.append(f"activate:{application_id}")
        self.state = 4
        self.current_package = application_id
        return self

    def terminate_app(self, application_id: str) -> bool:
        self.lifecycle.append(f"terminate:{application_id}")
        self.state = 1
        return True

    def background_app(self, seconds: int) -> object:
        self.lifecycle.append(f"background:{seconds}")
        return self


def application(platform: DevicePlatform) -> ApplicationDescriptor:
    return ApplicationDescriptor(
        platform=platform,
        application_id="com.example.app",
    )
