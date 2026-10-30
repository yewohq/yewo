"""What a run reached, named from what the screen itself reads as.

A run that misses its goal still learned which screens exist. These cases
hold that record to being observed rather than predicted, written once per
screen, and scrubbed before it reaches shared knowledge.
"""

from perception_fakes import detected

from yewo.discovery.contracts import CapabilityState
from yewo.discovery.device_runtime import DeviceDiscoveryRuntime
from yewo.utils.secrets import SecretScrubber

from discovery_device_fakes import (
    DiscoveryDevice as _Device,
    DiscoveryDriver as _Driver,
    JOURNEY_GOAL as GOAL,
    NoSatisfiedGoals as _NothingSatisfied,
    PerceptionSequence as _Perception,
    build_device_runtime as _runtime,
    discovery_screen as _screen,
)


def _titled_screen(title, *extra):
    """A real Android screen: the platform's status bar above the app's title."""
    return _screen(
        detected((0, 0, 400, 24), name="StatusBar", text="7:51"),
        detected((20, 30, 380, 70), name="Text", text=title),
        detected((20, 100, 180, 160), text="Open"),
        *extra,
    )


def test_a_reached_screen_is_recorded_by_what_it_reads_as():
    """A run that misses its goal still learned which screens exist."""
    titled = _titled_screen("Your order")

    candidates = _runtime(_Perception(titled)).observe().capability_candidates

    assert [item.capability for item in candidates] == [
        "reach the screen headed Your order"
    ]
    assert candidates[0].goal_ids == ("goal-1",)


def test_one_screen_reached_twice_is_recorded_once():
    titled = _titled_screen("Your order")
    runtime = _runtime(_Perception(titled, titled))

    first = runtime.observe().capability_candidates[0]
    second = runtime.observe().capability_candidates[0]

    assert first.candidate_id == second.candidate_id


def test_a_screen_that_reads_as_nothing_is_not_named():
    batch = _runtime(_Perception(_screen(detected((20, 100, 180, 160))))).observe()

    assert batch.capability_candidates == ()

def test_a_declared_secret_in_a_heading_does_not_reach_shared_knowledge():
    scrubber = SecretScrubber()
    scrubber.learn_secrets({"card": "4242424242424242"})
    exposed = _titled_screen("Card 4242424242424242")
    runtime = DeviceDiscoveryRuntime(
        _Device(),
        _Perception(exposed),
        app_id="com.example.app",
        setup_contract_sha256="b" * 64,
        goals=(GOAL,),
        goal_judge=_NothingSatisfied(),
        scrubber=scrubber,
        settle_time=0.0,
        sleep=lambda _seconds: None,
    )

    capability = runtime.observe().capability_candidates[0].capability

    assert "4242424242424242" not in capability
    assert "{{card}}" in capability


def test_a_settled_goal_leaves_no_reached_screen_to_record():
    settled = GOAL.activate().close(CapabilityState.REACHED, ("observation-1",))
    titled = _titled_screen("Your order")

    batch = _runtime(_Perception(titled), goals=(settled,)).observe()

    assert batch.capability_candidates == ()


def test_the_platform_clock_is_never_what_a_screen_is_called():
    """Fused elements distinguish platform text from the app's own heading."""
    batch = _runtime(_Perception(_titled_screen("Trainings Schedule"))).observe()

    assert [item.capability for item in batch.capability_candidates] == [
        "reach the screen headed Trainings Schedule"
    ]


def test_the_clock_is_not_the_heading_when_the_detector_misses_the_bar():
    """The detector can report the status band as an ordinary view.

    It did, on every frame of one live run. Keyed on the role alone, the time
    became the screen's name, and that name changed every minute, so one
    screen was recorded as a new screen on every observation.
    """
    missed_bar = _screen(
        detected((0, 0, 400, 24), name="View", text="7:51"),
        detected((20, 30, 380, 70), name="Text", text="Trainings Schedule"),
    )
    driver = _Driver(
        system_bars={"statusBar": {"x": 0, "y": 0, "width": 400, "height": 24}}
    )

    batch = _runtime(_Perception(missed_bar), driver=driver).observe()

    assert [item.capability for item in batch.capability_candidates] == [
        "reach the screen headed Trainings Schedule"
    ]


def test_a_keyboard_key_is_never_what_a_screen_is_called():
    typing = _screen(
        detected((0, 0, 400, 24), name="StatusBar", text="7:51"),
        detected((0, 500, 400, 800), name="Keyboard", roles={"keyboard_region"}, text="q w e"),
        detected((20, 30, 380, 70), name="Text", text="New schedule"),
    )

    batch = _runtime(_Perception(typing)).observe()

    assert [item.capability for item in batch.capability_candidates] == [
        "reach the screen headed New schedule"
    ]


def test_a_screen_showing_only_platform_chrome_is_not_named():
    chrome_only = _screen(detected((0, 0, 400, 24), name="StatusBar", text="7:51"))

    assert _runtime(_Perception(chrome_only)).observe().capability_candidates == ()
