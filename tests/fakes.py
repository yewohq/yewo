"""Builders for the values Yewo's boundaries hand to its logic.

Each one calls the real constructor. Rename or remove a field and the tests
fail to build, rather than keep passing against a shape that production no
longer uses.

The device harness is here for the same reason. Two test files ask what a
capability does, and one driver that records what it was asked for keeps them
asking it the same way.
"""

import io
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from yewo.agent.actuation.actions import (
    AccessibilityCapability,
    AppStateCapability,
    BatteryCapability,
    BiometricCapability,
    CapabilityKind,
    ClipboardCapability,
    DeeplinkCapability,
    InterruptCapability,
    LocaleCapability,
    LocationCapability,
    MediaInjectCapability,
    NetworkCapability,
    PermissionCapability,
    TimezoneCapability,
)
from yewo.agent.device import capability_receipts
from yewo.agent.device.capability_controller import CapabilityController
from yewo.agent.device.platform import Platform
from yewo.agent.memory.history import AgentHistory
from yewo.agent.run.outcomes import DeviceRunResult
from yewo.agent.step.observation import PerceptionSnapshot
from yewo.agent.perception.elements import BoundingBox, GUIElement
from yewo.agent.grounding.marks import Mark

STAMP = "2026-01-01T00:00:00"


def device_result(
    status="goal_reached",
    failure_reason=None,
    cause=None,
    capability_receipts=(),
    postcondition_verifications=(),
) -> DeviceRunResult:
    """Build the real result object consumed by the verdict boundary."""
    return DeviceRunResult(
        status=status,
        steps_executed=0,
        failure_reason=failure_reason,
        history=AgentHistory(),
        cause=cause,
        capability_receipts=list(capability_receipts),
        postcondition_verifications=list(postcondition_verifications),
    )


class FakeClock:
    """A monotonic clock that advances only when production asks it to sleep."""

    def __init__(self) -> None:
        self.elapsed = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def sleep(self, seconds: float) -> None:
        self.elapsed += seconds


def solid_png(fill: str = "white", size: tuple[int, int] = (200, 400)) -> bytes:
    """A screenshot of one colour, for comparing a screen against itself."""
    buffer = io.BytesIO()
    Image.new("RGB", size, fill).save(buffer, format="PNG")
    return buffer.getvalue()


def element(
    box: tuple[int, int, int, int] = (10, 10, 60, 60),
    name: str = "Button",
    text: str = "Continue",
    confidence: float = 0.9,
    kind: str = "yolo_detection",
) -> GUIElement:
    """One element, as the detector reports it."""
    return GUIElement(
        element=name,
        type=kind,
        confidence=confidence,
        text=text,
        bounding_box=BoundingBox(
            x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
        ),
    )


def numbered(*elements: GUIElement) -> list[Mark]:
    """The numbered targets offered to the planner, numbered in the order given."""
    return [Mark(index=index, element=item) for index, item in enumerate(elements, 1)]


def snapshot(marks=(), gui_elements=(), png=None, phash="p") -> PerceptionSnapshot:
    """What perception hands the step loop."""
    screen = solid_png() if png is None else png
    return PerceptionSnapshot(
        gui_elements=list(gui_elements),
        marks=list(marks),
        raw_img_bytes=screen,
        planner_img_bytes=screen,
        annotated_img=Image.new("RGB", (1, 1)),
        current_phash=phash,
        timestamp=STAMP,
    )


def qa_prompt(goal_lines=("Sign in",), setup=None):
    """The prompt a real QAPlanner builds, through its real constructor.

    Through the constructor, so a field it stopped taking cannot be assigned
    back in and go on looking present.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock

    from yewo.agent.device.launch import SetupMode
    from yewo.testing.qa_planner import QAPlanner

    llm = SimpleNamespace(generate_response=Mock())
    planner = QAPlanner(llm_client=llm, setup=setup or SetupMode.CLEAR_DATA)
    return planner._build_qa_prompt(
        goal_lines=list(goal_lines),
        gui_elements=None,
        history="",
        trail=None,
        drift_warning=None,
        loop_nudge=None,
    )


def walk(*instructions: str):
    """Prose actions as the typed steps a scenario carries.

    Built through the real step models, so an instruction a step would refuse
    is refused here too rather than reaching a test as a bare string.
    """
    from yewo.testing.steps import ActionStep

    return [ActionStep(instruction=text) for text in instructions]


SERIAL = "emulator-5554"
PACKAGE = "com.example.app"
BUNDLE = "com.example.ExampleApp"

# Every read-back the command-line fallback can ask for. Silenced by default,
# so a dispatch case is about the command that was sent and nothing else.
READERS = (
    "airplane_mode",
    "timezone",
    "app_locale",
    "battery_level",
    "battery_state",
    "permission_granted",
    "accessibility_enabled",
    "app_running",
)

# One instance of every capability, so a capability added to the union without
# a case in the dispatch fails here rather than at run time on a device.
EVERY_CAPABILITY = {
    CapabilityKind.LOCATION: LocationCapability(latitude=1.0, longitude=2.0),
    CapabilityKind.NETWORK: NetworkCapability(mode="offline"),
    CapabilityKind.BATTERY: BatteryCapability(level=50),
    CapabilityKind.TIMEZONE: TimezoneCapability(timezone="Europe/London"),
    CapabilityKind.LOCALE: LocaleCapability(language="fr", country="FR"),
    CapabilityKind.PERMISSION: PermissionCapability(permission="camera", grant=True),
    CapabilityKind.INTERRUPT: InterruptCapability(type="low_memory"),
    CapabilityKind.BIOMETRIC: BiometricCapability(action="match"),
    CapabilityKind.CLIPBOARD: ClipboardCapability(text="hello"),
    CapabilityKind.DEEPLINK: DeeplinkCapability(url="app://cart"),
    CapabilityKind.APP_STATE: AppStateCapability(action="force_stop"),
    CapabilityKind.MEDIA_INJECT: MediaInjectCapability(local_path=__file__),
    CapabilityKind.ACCESSIBILITY: AccessibilityCapability(enabled=True),
}


class FakeDriver:
    """A driver that records the `mobile:` methods it was asked for.

    `answers` is what a read returns. `without` names methods this driver does
    not have, which is how a real driver behaves on a platform or version that
    never implemented one, and is what sends a capability to the command line.
    """

    def __init__(
        self, answers=None, without=(), package=PACKAGE, platform=Platform.ANDROID
    ):
        # Each platform names the app its own way, and a driver that answered
        # with Android's key on iOS hid every capability that asks for one.
        named = "bundleId" if platform == Platform.IOS else "appPackage"
        self.capabilities = {named: package, "udid": SERIAL}
        self.calls = []
        self.answers = answers or {}
        self.without = set(without)
        self.set_network_connection = mock.Mock()
        self.finger_print = mock.Mock()

    def execute_script(self, script, params=None):
        method = script.removeprefix("mobile: ")
        self.calls.append((method, params or {}))
        if method in self.without:
            raise RuntimeError(f"Unknown mobile command '{method}'")
        return self.answers.get(method)

    def asked(self, method):
        """The params one method was called with, or None if it never was."""
        for name, params in self.calls:
            if name == method:
                return params
        return None


def controller(platform=Platform.ANDROID, package=None, driver=None):
    """A controller over a session that resolved its app the way a run does.

    `package_id` is Android's alone, as it is in a real session, so a path
    that reaches for it on iOS fails here rather than on a device.
    """
    ios = platform == Platform.IOS
    app_id = package if package is not None else (BUNDLE if ios else PACKAGE)
    driver = driver or FakeDriver(package=app_id, platform=platform)
    device = SimpleNamespace(
        driver=driver,
        platform=platform,
        device_name=SERIAL,
        serial=SERIAL,
        app_id=app_id,
        package_id=None if ios else app_id,
        adb_control=SimpleNamespace(resolve_emulator_serial=lambda name: SERIAL),
    )
    return CapabilityController(device)


def silent_reads():
    """A device that answers no command-line state question."""
    patched = mock.patch.object(capability_receipts, "receipts")
    reads = patched.start()
    for name in READERS:
        getattr(reads, name).return_value = None
    return patched, reads


UDID = "00008140-00096DC43A52801C"


def session(platform=Platform.ANDROID, serial=SERIAL, driver=None):
    """A session as the restorer reads one, with every field it asks for."""
    return SimpleNamespace(
        platform=platform,
        serial=serial,
        package_id=PACKAGE,
        app_id=PACKAGE,
        driver=driver,
    )


class IosDriver:
    """A driver answering the reads an iOS snapshot takes.

    Each read answers in the shape the real driver answers in, so a reader
    that expects the wrong one fails here. `reads` counts them, because each
    one is a call to the device. `state` is read on every call rather than
    queued, so a test moves a state inside the scenario the way a capability
    would. `refuse` names the methods this driver fails, which is how a
    restore that cannot land is exercised.
    """

    def __init__(self, voice_over=None, location=None, granted=None, refuse=()):
        self.capabilities = {"udid": UDID}
        self.state = {
            "voice_over": voice_over,
            "location": location,
            "granted": dict(granted or {}),
        }
        self.sent = []
        self.reads = 0
        self.refuse = set(refuse)

    def execute_script(self, script, params=None):
        method = script.removeprefix("mobile: ")
        if method == "isVoiceOverEnabled":
            self.reads += 1
            return {"enabled": self.state["voice_over"]}
        if method == "getSimulatedLocation":
            self.reads += 1
            if "getSimulatedLocation" in self.refuse:
                return None
            placed = self.state["location"]
            return {"latitude": placed[0], "longitude": placed[1]} if placed else {
                "latitude": None,
                "longitude": None,
            }
        if method == "getPermission":
            self.reads += 1
            return self.state["granted"].get(params["service"])
        self.sent.append((method, params or {}))
        if method in self.refuse:
            raise RuntimeError(f"'{method}' is not available here")
        return None

    def asked(self, method):
        """The params one method was sent with, or None if it never was."""
        for name, params in self.sent:
            if name == method:
                return params
        return None


def ios_device(driver):
    found = session(platform=Platform.IOS, serial=UDID, driver=driver)
    found.app_id = BUNDLE
    return found


def rate_limited():
    """A provider failure that one pause could clear."""
    from litellm import exceptions

    return exceptions.RateLimitError("slow down", "m", "openrouter")


def refused():
    """A provider failure that every later call meets again."""
    from litellm import exceptions

    return exceptions.AuthenticationError("bad key", "m", "openrouter")


def suite_run(
    scenarios,
    directory,
    journey=None,
    report_for=None,
    uncovered=None,
    session_calls=None,
    **runner,
):
    """One journey run end to end, with the device and the models faked.

    Every real part of the runner takes part: the lock, the manifest, the
    reports and the usage telemetry. What is replaced is the session, the
    generator and the scenario agent, which are the three things that need a
    device or a paid call.

    Args:
        scenarios: The suite the generator hands back.
        directory: Where artifacts are written.
        journey: What is run. A one-cell journey by default.
        report_for: Called with each scenario, returning its report. A
            passing report by default.
        uncovered: Cells the generator could produce nothing for.
        session_calls: A list the fake session appends each method name to,
            so a caller can assert what the runner asked of it and when.
        runner: Anything else to pass to `YewoTestRunner`.

    Returns:
        The journey report, and the manifest as it was written.
    """
    import json
    from pathlib import Path
    from unittest.mock import patch

    from yewo.testing.generator import GeneratedSuite
    from yewo.testing.models import ScenarioReport, UserJourney
    from yewo.testing.runner import YewoTestRunner

    root = Path(directory)

    def passing(scenario):
        return ScenarioReport(
            scenario_uuid=scenario.scenario_uuid,
            test_id=scenario.test_id,
            scenario_name=scenario.scenario_name,
            cell=scenario.cell,
            status="PASSED",
        )

    class Session:
        def __init__(self, *_, **__):
            self.telemetry_dir = root / "telemetry"
            self.screenshots_dir = root / "screenshots"
            self.telemetry_dir.mkdir(parents=True, exist_ok=True)
            self.session_id = "s1"
            self.installed_by_run = False
            self.root_log_file = root / "run.log"
            self.appium_log_path = root / "appium.log"
            self.manifest_path = root / "run.json"
            self.prompts_dir = root / "prompts"
            self.asked = session_calls if session_calls is not None else []

        def start(self, **_):
            # A real control reports its size once the driver answers, and
            # a saved path is only valid for the size it was learned on.
            return SimpleNamespace(
                platform=None,
                serial=None,
                os_version=None,
                model="Pixel 8",
                screen_width=1080,
                screen_height=2400,
            )

        def checkpoint_recording(self):
            self.asked.append("checkpoint_recording")

        def checkpoint_device_logs(self):
            self.asked.append("checkpoint_device_logs")

        def close(self, **_):
            self.asked.append("close")

    class Generator:
        def __init__(self, *_, **__):
            pass

        def generate_suite(self, _journey, cells=None):
            return GeneratedSuite(
                scenarios=list(scenarios),
                locked=list(scenarios),
                uncovered=list(uncovered or []),
            )

    class Agent:
        def __init__(self, *_, **__):
            from yewo.utils.secrets import SecretScrubber

            llm = SimpleNamespace(usage_callback=None)
            self._device_agent = SimpleNamespace(
                llm=llm,
                fast_llm=llm,
                scrubber=SecretScrubber(),
                record_prompts=lambda _store: None,
            )

        def execute_scenario(self, test_case, **_):
            return (report_for or passing)(test_case)

    with (
        patch("yewo.testing.runner.DeviceSession", Session),
        patch("yewo.testing.runner.ScenarioGenerator", Generator),
        patch("yewo.testing.runner.QAAgent", Agent),
        patch("yewo.testing.runner.ensure_device_ready"),
        patch("yewo.testing.runner.for_grounding"),
        patch("yewo.testing.runner.is_session_responsive", return_value=True),
    ):
        report = YewoTestRunner(
            device_name="Pixel_8", app_path="app.apk", **runner
        ).run_journey(
            journey
            or UserJourney(id="j", name="J", description="A user signs in.")
        )

    written = json.loads((root / "run.json").read_text())
    return report, written


LOG_EPOCH_MS = 1761000000000
"""A fixed receipt time, in the milliseconds since the epoch Selenium uses."""


def log_entries(*messages):
    """Log entries in the shape Selenium's `get_log` returns them."""
    return [
        {"timestamp": LOG_EPOCH_MS + n * 1000, "level": "INFO", "message": text}
        for n, text in enumerate(messages)
    ]


class LoggingDriver:
    """A driver answering `get_log`, as each kind of target answers it.

    Reads drain, the way Appium's own buffer does: each `get_log` returns
    only what arrived since the last one. A fake that returned everything
    every time would let a capture that duplicates its lines pass.

    `offers` is what `log_types` lists, and `logs` is what is buffered to
    begin with. A log listed with nothing behind it is the case the capture
    has to explain rather than write an empty file for. `failing` names
    logs this driver lists and then refuses, which is a different fault.
    """

    def __init__(self, offers=(), logs=None, failing=(), list_fails=False):
        self._offers = tuple(offers)
        self._buffered = {name: list(entries) for name, entries in (logs or {}).items()}
        self._failing = set(failing)
        self._list_fails = list_fails
        self.asked = []
        self.type_reads = 0
        self.quit_called = False
        self._failing_next = False
        self._failing_next_loses = False

    @property
    def log_types(self):
        self.type_reads += 1
        if self._list_fails:
            raise RuntimeError("the session is gone")
        return list(self._offers)

    def arrive(self, log_type, *messages):
        """More lines reach the buffer, as they do while a scenario runs."""
        self._buffered.setdefault(log_type, []).extend(log_entries(*messages))

    def fail_next(self, *, losing=False):
        """The next read raises. `losing` drains the buffer first, as the
        server does when it answered and the response never arrived."""
        self._failing_next = True
        self._failing_next_loses = losing

    def get_log(self, log_type):
        self.asked.append(log_type)
        if self.quit_called:
            raise RuntimeError("a quit driver reads nothing")
        if log_type in self._failing:
            raise RuntimeError(f"no such log '{log_type}'")
        if self._failing_next:
            self._failing_next = False
            if self._failing_next_loses:
                self._buffered[log_type] = []
            raise RuntimeError("the server dropped the response")
        drained = self._buffered.get(log_type, [])
        self._buffered[log_type] = []
        return drained

    def quit(self):
        self.quit_called = True
