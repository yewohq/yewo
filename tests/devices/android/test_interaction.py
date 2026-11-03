"""Tests for Android-specific interaction evidence."""

from typing import Any, cast

from yewo.devices.application_lifecycle import ApplicationLifecycleDriver
from yewo.devices.android.interaction import AndroidInteractionBackend
from yewo.devices.interaction import TextInputConstraint
from yewo.devices.screen import ScreenCalibration, ScreenCoordinateSpace


class _SystemBarDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {
            "statusBar": {
                "visible": True,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 20,
            },
        }


class _PackageDriver:
    current_package = "com.example.testing_app"


def test_backend_reads_system_regions_from_the_driver() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _SystemBarDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is not None
    assert regions[0].coordinate_space is ScreenCoordinateSpace.DRIVER
    assert [region.kind.value for region in regions] == ["status_bar"]
    assert (regions[0].left, regions[0].top, regions[0].right, regions[0].bottom) == (
        0,
        0,
        100,
        20,
    )


class _KeyboardRegionDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {
            "ime": {
                "visible": True,
                "x": 0,
                "y": 120,
                "width": 100,
                "height": 80,
            },
        }


def test_backend_classifies_driver_ime_geometry_as_keyboard() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _KeyboardRegionDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is not None
    assert [region.kind.value for region in regions] == ["keyboard"]
    assert (regions[0].left, regions[0].top, regions[0].right, regions[0].bottom) == (
        0,
        120,
        100,
        200,
    )


class _HiddenSystemBarDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {
            "statusBar": {
                "visible": False,
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 20,
            },
        }


def test_backend_excludes_hidden_system_regions() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _HiddenSystemBarDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions == ()


class _MalformedSystemBarDriver:
    def get_system_bars(self) -> dict[str, object]:
        return {"statusBar": {"visible": True, "height": 20}}


def test_backend_preserves_unknown_for_unparsable_system_regions() -> None:
    driver = cast(ApplicationLifecycleDriver, cast(Any, _MalformedSystemBarDriver()))

    regions = AndroidInteractionBackend().system_regions(
        driver,
        ScreenCalibration(200, 400, 100, 200),
    )

    assert regions is None


class _InputReader:
    def __init__(self, output: str | None) -> None:
        self.output = output
        self.calls: list[tuple[str, tuple[str, ...], float]] = []

    def ask(self, connection_id: str, *args: str, timeout_seconds: float = 20) -> str | None:
        self.calls.append((connection_id, args, timeout_seconds))
        return self.output


def test_backend_reads_numeric_focused_field_constraint() -> None:
    reader = _InputReader(
        "mCurAttribute=EditorInfo{packageName=com.example, inputType=0x00003002, imeOptions=5}"
    )
    backend = AndroidInteractionBackend(reader, "emulator-5554")

    constraint = backend.focused_text_constraint(cast(Any, object()))

    assert constraint == TextInputConstraint(
        is_numeric=True,
        allows_sign=True,
        allows_decimal=True,
    )
    assert reader.calls == [
        ("emulator-5554", ("dumpsys", "input_method"), 3),
    ]


def test_backend_uses_latest_android_start_input_history_entry() -> None:
    reader = _InputReader(
        """Current Input Method Manager state:
 mStartInputHistory:
  StartInput #4:
   targetWin=window [com.example]
   inputType=0x1 imeOptions=0x5
  StartInput #5:
   targetWin=window [com.example]
   inputType=0x2 imeOptions=0x5
 mSoftInputShowHideHistory:
  SoftInputShowHide #1:
   reason=SHOW_SOFT_INPUT
 Input method service state:
  inputType=0x1
"""
    )
    backend = AndroidInteractionBackend(reader, "emulator-5554")

    constraint = backend.focused_text_constraint(cast(Any, object()))

    assert constraint == TextInputConstraint(is_numeric=True)


def test_backend_preserves_unknown_when_android_does_not_report_current_editor() -> None:
    backend = AndroidInteractionBackend(_InputReader("mInputShown=true"), "emulator-5554")

    assert backend.focused_text_constraint(cast(Any, object())) is None


def test_backend_prefers_android_focused_window_package() -> None:
    reader = _InputReader(
        "mCurrentFocus=Window{8127b5a u0 com.android.systemui/com.android.systemui.statusbar."
        "phone.CentralSurfaces}"
    )
    backend = AndroidInteractionBackend(reader, "emulator-5554")

    application_id = backend.active_application_id(cast(Any, _PackageDriver()))

    assert application_id == "com.android.systemui"
    assert reader.calls == [
        ("emulator-5554", ("dumpsys", "window"), 3),
    ]


def test_backend_maps_bare_notification_shade_window_to_system_ui() -> None:
    reader = _InputReader("mCurrentFocus=Window{8127b5a u0 NotificationShade}")
    backend = AndroidInteractionBackend(reader, "emulator-5554")

    application_id = backend.active_application_id(cast(Any, _PackageDriver()))

    assert application_id == "com.android.systemui"


def test_backend_falls_back_to_appium_package_without_focused_window() -> None:
    backend = AndroidInteractionBackend(_InputReader("WINDOW MANAGER WINDOWS"), "emulator-5554")

    application_id = backend.active_application_id(cast(Any, _PackageDriver()))

    assert application_id == "com.example.testing_app"
