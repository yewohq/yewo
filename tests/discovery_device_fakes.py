"""Reusable device and perception doubles for discovery runtime tests."""


from PIL import Image
from fakes import solid_png
from perception_fakes import detected, frame

from yewo.agent.device.platform import Platform
from yewo.agent.perception.elements import ScanInfo
from yewo.discovery.contracts import JourneyGoal
from yewo.discovery.device_runtime import DeviceDiscoveryRuntime
from yewo.utils.secrets import SecretScrubber

SCREEN_SIZE = (400, 800)
JOURNEY_GOAL = JourneyGoal(
    goal_id="goal-1", statement="A user places an order."
)


class DiscoveryDriver:
    """A driver answering the `mobile:` methods discovery sends it.

    A driver, not a command surface. `DeviceControl` exposes `driver` and
    nothing else, so a fake offering a ready-made command object would let the
    runtime reach for something no real device has.
    """

    def __init__(self, app_state=4, refuse=(), system_bars=None):
        self.app_state = app_state
        self.refuse = frozenset(refuse)
        self.orientation = "PORTRAIT"
        self.calls = []
        self.system_bars = system_bars or {}

    def get_system_bars(self):
        return self.system_bars

    def execute_script(self, script, params=None):
        del params
        method = script.removeprefix("mobile: ")
        self.calls.append(method)
        if method in self.refuse:
            raise RuntimeError(f"the device refused {method}")
        if method == "queryAppState":
            return self.app_state
        if method == "terminateApp":
            self.app_state = 1
            return None
        if method == "activateApp":
            self.app_state = 4
            return None
        raise AssertionError(f"the fake driver was asked for {method!r}")


class DiscoveryDevice:
    """A started device session, answering what discovery reads off one.

    Every attribute here exists on `DeviceControl`. The guard test holds it to
    that, because a fake with a convenience attribute hides a runtime that
    cannot run.
    """

    def __init__(self, driver=None):
        self.driver = driver or DiscoveryDriver()
        self.platform = Platform.ANDROID
        self.device_name = "emulator-5554"
        self.app_id = "com.example.app"
        self.screen_width, self.screen_height = SCREEN_SIZE
        self.dispatched = []

    @staticmethod
    def is_keyboard_open():
        return False

    @staticmethod
    def get_screenshot():
        return solid_png(size=SCREEN_SIZE)

    def tap(self, x, y, **_kwargs):
        self.dispatched.append((x, y))
        return True


class PerceptionSequence:
    def __init__(self, *frames, phashes=None):
        self._frames = list(frames)
        self._phashes = list(phashes or ["0" * 16] * len(frames))
        self.index = 0

    def _advance(self):
        position = min(self.index, len(self._frames) - 1)
        self.index += 1
        return position

    def perceive(self):
        position = self._advance()
        self.current_frame = self._frames[position]
        self.current_hash = self._phashes[position]
        self.last_scan_info = ScanInfo(
            mode="full",
            reason="first_pass",
            phash=self.current_hash,
            screen_width=SCREEN_SIZE[0],
            screen_height=SCREEN_SIZE[1],
        )
        image = Image.new("RGB", SCREEN_SIZE, "white")
        return list(self.current_frame.elements), image, image


class NoSatisfiedGoals:
    @staticmethod
    def satisfied_goal_ids(goals, frame, chrome):
        del goals, frame, chrome
        return ()


class AllGoalsSatisfied:
    @staticmethod
    def satisfied_goal_ids(goals, frame, chrome):
        del frame, chrome
        return tuple(goal.goal_id for goal in goals)


def discovery_screen(*elements, text_lines=()):
    return frame(elements=tuple(elements), text_lines=tuple(text_lines))


def discovery_button(y=100, name="View", text="Open"):
    return detected((20, y, 180, y + 60), name=name, text=text)


def build_device_runtime(
    perception,
    judge=None,
    driver=None,
    goals=(JOURNEY_GOAL,),
    scrubber=None,
    artifact_writer=None,
    diagnostic_dir=None,
):
    return DeviceDiscoveryRuntime(
        DiscoveryDevice(driver),
        perception,
        app_id="com.example.app",
        setup_contract_sha256="b" * 64,
        goals=goals,
        goal_judge=judge or NoSatisfiedGoals(),
        scrubber=scrubber or SecretScrubber(),
        artifact_writer=artifact_writer,
        diagnostic_dir=diagnostic_dir,
        settle_time=0.0,
        sleep=lambda _seconds: None,
    )
