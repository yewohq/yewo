"""The device half of discovery: what one screen offers, and where a tap landed.

Every boundary here is faked. Nothing starts a device, loads weights, or calls
a model. What is under test is the projection from a perceived frame to offered
controls, the settlement record one dispatch leaves, the identity a screen
keeps across a transient change, and what a relaunch returns to.
"""

from unittest.mock import patch

import pytest
from perception_fakes import detected

from yewo.agent.actuation.action_models import PlannerTapAction
from yewo.agent.actuation.safety import ActionSafetyPolicy, ActionSource
from yewo.agent.perception.elements import ScanInfo
from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.artifacts.steps import StepArtifacts
from yewo.agent.run.settlement import SettlementStatus
from yewo.discovery.contracts import SideEffectClass
from yewo.discovery.control_identity import discovery_control_identity
from yewo.utils.secrets import SecretScrubber
from yewo.agent.memory.replay_store import RelativeBounds
from yewo.discovery.device_runtime import (
    EXPLORATION_REASON,
    DeviceDiscoveryRuntime,
    _relative_bounds,
)

from discovery_device_fakes import (
    AllGoalsSatisfied as _EverythingSatisfied,
    DiscoveryDevice,
    DiscoveryDriver as _Driver,
    JOURNEY_GOAL as GOAL,
    NoSatisfiedGoals as _NothingSatisfied,
    PerceptionSequence as _Perception,
    build_device_runtime as _runtime,
    discovery_button as _button,
    discovery_screen as _screen,
)


def test_every_perceived_control_is_offered_against_the_open_goal():
    runtime = _runtime(_Perception(_screen(_button(100), _button(300))))

    batch = runtime.observe()

    assert len(batch.candidates) == 2
    assert {candidate.relevant_goal_ids for candidate in batch.candidates} == {("goal-1",)}


def test_a_text_only_drawer_row_is_offered_at_its_observed_bounds():
    """A detector may report a tappable drawer row only as bounded text."""
    drawer = _screen(
        detected((0, 0, 400, 24), name="StatusBar", text="7:51"),
        detected((20, 160, 260, 220), name="Text", text="Edit Schedule"),
    )

    batch = _runtime(_Perception(drawer)).observe()

    assert len(batch.candidates) == 1
    assert batch.candidates[0].action.coordinates.x == 140
    assert batch.candidates[0].action.coordinates.y == 190


def test_platform_bar_bounds_exclude_chrome_when_detection_misses_the_bar():
    driver = _Driver(
        system_bars={
            "statusBar": {"x": 0, "y": 0, "width": 400, "height": 24}
        }
    )
    screen = _screen(
        detected((330, 4, 380, 20), name="Text", text="7:00"),
        detected((20, 100, 180, 160), name="View", text="Open"),
    )

    batch = _runtime(_Perception(screen), driver=driver).observe()

    assert len(batch.candidates) == 1
    assert batch.candidates[0].action.coordinates.y == 130


def test_each_observation_persists_scrubbed_images_and_fused_scan(tmp_path):
    scrubber = SecretScrubber()
    scrubber.learn_secrets({"account": "private-account"})
    diagnostic_dir = tmp_path / "screenshots"
    writer = StepArtifacts(RedactedImages(scrubber), scrubber)
    observed = _screen(
        detected((20, 100, 300, 160), name="Text", text="private-account")
    )
    runtime = _runtime(
        _Perception(observed),
        scrubber=scrubber,
        artifact_writer=writer,
        diagnostic_dir=diagnostic_dir,
    )

    observation = runtime.observe().observation

    written_names = {path.name for path in diagnostic_dir.iterdir()}
    assert len(observation.evidence) == 3
    assert all(reference.path is not None for reference in observation.evidence)
    assert {
        reference.path.removeprefix("screenshots/")
        for reference in observation.evidence
        if reference.path is not None
    } == written_names
    scan = next(diagnostic_dir.glob("*_scan.json")).read_text()
    assert "private-account" not in scan
    assert "{{account}}" in scan


def test_an_offered_control_matches_the_identity_the_controller_will_authorize():
    """The controller refuses a candidate whose identity its authorization misses."""
    runtime = _runtime(_Perception(_screen(_button())))

    batch = runtime.observe()
    candidate = batch.candidates[0]
    assert (
        discovery_control_identity(
            candidate.source_screen_state_id,
            candidate.action,
            candidate.screen_size,
        )
        == candidate.control_identity
    )
    assert batch.observation.controls[0].control_identity == candidate.control_identity


def test_one_screen_names_its_controls_once_however_its_fingerprint_moves():
    """A clock ticking must not rename every control on an unchanged screen.

    It did. The frontier then re-offered the whole screen under new names,
    and the offers under the old names stood as work no later frame could
    ever present, so the run could neither reach them nor retire them.
    """
    screen = _screen(_button())
    perception = _Perception(screen, screen, phashes=["0" * 16, "0" * 15 + "1"])
    runtime = _runtime(perception)

    first = runtime.observe()
    second = runtime.observe()

    assert first.observation.phash != second.observation.phash
    assert {candidate.control_identity for candidate in first.candidates} == {
        candidate.control_identity for candidate in second.candidates
    }
    assert {candidate.frontier_id for candidate in first.candidates} == {
        candidate.frontier_id for candidate in second.candidates
    }


def test_a_candidate_carries_the_V0_side_effect_policy():
    """REVERSIBLE because reset can return home, and no reset after each tap."""
    runtime = _runtime(_Perception(_screen(_button())))

    candidate = runtime.observe().candidates[0]

    assert candidate.side_effect is SideEffectClass.REVERSIBLE
    assert candidate.reversible
    assert not candidate.reset_after


def test_two_controls_sharing_a_centre_are_offered_once():
    """An identity is what an action addresses, so one tap is one target."""
    stacked = _screen(
        detected((20, 100, 180, 160), name="View", text="Open"),
        detected((20, 100, 180, 160), name="ImageView"),
    )

    batch = _runtime(_Perception(stacked)).observe()

    assert len(batch.candidates) == 1
    assert len(batch.observation.controls) == 1


def test_bounds_are_recorded_as_thousandths_of_the_screen():
    runtime = _runtime(_Perception(_screen(detected((0, 0, 200, 400), text="Half"))))

    bounds = runtime.observe().observation.controls[0].bounds

    assert (bounds.left, bounds.top, bounds.right, bounds.bottom) == (0, 0, 500, 500)


def test_the_tap_reason_is_fixed_so_no_app_vocabulary_reaches_an_artifact():
    candidate = _runtime(_Perception(_screen(_button()))).observe().candidates[0]

    assert isinstance(candidate.action, PlannerTapAction)
    assert candidate.action.reason == EXPLORATION_REASON
    assert candidate.action.target is None


def test_a_transient_change_does_not_make_the_same_screen_a_new_one():
    """Otherwise the controller never recognizes that it is back at the start."""
    screen = _screen(_button())
    perception = _Perception(screen, screen, phashes=["0" * 16, "0" * 15 + "1"])
    runtime = _runtime(perception)

    first = runtime.observe().observation.screen_state_id
    second = runtime.observe().observation.screen_state_id

    assert first == second


def test_a_different_screen_gets_a_different_identity():
    perception = _Perception(
        _screen(_button()),
        _screen(_button()),
        phashes=["0" * 16, "f" * 16],
    )
    runtime = _runtime(perception)

    assert runtime.observe().observation.screen_state_id != (
        runtime.observe().observation.screen_state_id
    )


def test_a_reading_with_no_fused_frame_is_refused():
    class _Frameless(_Perception):
        def perceive(self):
            result = super().perceive()
            self.current_frame = None
            return result

    runtime = _runtime(_Frameless(_screen(_button())))

    with pytest.raises(RuntimeError, match="fused perception frame"):
        runtime.observe()


def test_a_reading_with_no_fingerprint_is_refused():
    class _Unfingerprinted(_Perception):
        def perceive(self):
            result = super().perceive()
            self.current_hash = None
            return result

    runtime = _runtime(_Unfingerprinted(_screen(_button())))

    with pytest.raises(RuntimeError, match="fingerprinted observation"):
        runtime.observe()


def test_prepare_refuses_a_setup_contract_it_was_not_built_for():
    receipt = _runtime(_Perception(_screen(_button()))).prepare("f" * 64)

    assert not receipt.success
    assert "not the prepared one" in receipt.reason


def test_prepare_refuses_an_app_that_is_not_in_the_foreground():
    receipt = _runtime(
        _Perception(_screen(_button())),
        driver=_Driver(app_state=1),
    ).prepare("b" * 64)

    assert not receipt.success
    assert "foreground" in receipt.reason


def test_prepare_accepts_its_own_setup_with_the_app_running():
    receipt = _runtime(_Perception(_screen(_button()))).prepare("b" * 64)

    assert receipt.success
    assert receipt.setup_contract_sha256 == "b" * 64


def test_reset_relaunches_the_app_and_returns_the_start_screen():
    driver = _Driver()
    runtime = _runtime(_Perception(_screen(_button())), driver=driver)

    receipt = runtime.reset()

    assert receipt.success
    assert receipt.observation is not None
    assert driver.calls == ["terminateApp", "queryAppState", "activateApp", "queryAppState"]


def test_a_failed_relaunch_is_reported_rather_than_raised():
    runtime = _runtime(
        _Perception(_screen(_button())), driver=_Driver(refuse=("terminateApp",))
    )

    receipt = runtime.reset()

    assert not receipt.success
    assert "terminateApp" in receipt.reason
    assert receipt.observation is None


@pytest.fixture(autouse=True)
def _no_settle_poll():
    """Settling waits on a real clock. Nothing here needs to wait."""
    with patch("yewo.agent.perception.settle.time.sleep"):
        yield


def _dispatched_through(runtime, batch):
    """The device, and what it was actually asked to do."""
    candidate, execution = _dispatched(runtime, batch)
    return runtime.device_control.dispatched, candidate, execution


def _dispatched(runtime, batch):
    candidate = batch.candidates[0]
    authorization = ActionSafetyPolicy().authorize(
        candidate.action,
        source=ActionSource.DISCOVERY,
        observation_id=candidate.source_observation_id,
        screen_phash=candidate.source_phash,
        screen_size=candidate.screen_size,
    )
    return candidate, runtime.execute(authorization, candidate)


def _two_screens(judge=None):
    perception = _Perception(
        _screen(_button(text="Open")),
        _screen(_button(text="Confirmed")),
        phashes=["0" * 16, "f" * 16],
    )
    return _runtime(perception, judge=judge)


def test_a_dispatched_action_settles_on_the_screen_it_reached():
    runtime = _two_screens()

    _, execution = _dispatched(runtime, runtime.observe())

    assert execution.dispatch.success
    assert execution.settlement.status is SettlementStatus.DURABLE_PROGRESS
    assert execution.destination.observation.phash == "f" * 16


def test_the_settlement_names_both_observations_the_controller_will_check():
    runtime = _two_screens()
    batch = runtime.observe()

    candidate, execution = _dispatched(runtime, batch)

    assert set(execution.settlement.observation_ids) >= {
        candidate.source_observation_id,
        execution.destination.observation.observation_id,
    }


def test_an_action_that_changed_nothing_is_reported_unverified_not_absent():
    """The controller refuses an unverified settlement, so the entry closes."""
    screen = _screen(_button())
    runtime = _runtime(_Perception(screen, screen, phashes=["0" * 16, "0" * 16]))

    _, execution = _dispatched(runtime, runtime.observe())

    assert execution.settlement.status is not SettlementStatus.DURABLE_PROGRESS


def test_depth_grows_with_each_transition_and_returns_to_zero_on_reset():
    runtime = _two_screens()

    assert runtime.observe().state_depth == 0
    _, execution = _dispatched(runtime, runtime.observe())
    assert execution.destination.state_depth == 1
    assert runtime.reset().observation.state_depth == 0


def test_no_capability_evidence_is_emitted_when_the_judge_finds_none():
    runtime = _two_screens()

    _, execution = _dispatched(runtime, runtime.observe())

    assert execution.capability_evidence == ()


def test_a_judged_goal_is_recorded_as_a_model_proposal_naming_its_transition():
    runtime = _two_screens(judge=_EverythingSatisfied())
    batch = runtime.observe()

    candidate, execution = _dispatched(runtime, batch)

    from yewo.discovery.runtime import transition_identity_for

    receipt = execution.capability_evidence[0]
    assert receipt.evidence_owner == "model_proposal"
    assert receipt.goal_id == "goal-1"
    assert receipt.observation_id == execution.destination.observation.observation_id
    assert receipt.transition_identity == transition_identity_for(
        candidate.source_observation_id,
        candidate.control_identity,
        execution.destination.observation.observation_id,
    )


def test_a_judge_naming_a_goal_this_run_does_not_hold_is_ignored():
    class _Inventing:
        @staticmethod
        def satisfied_goal_ids(goals, frame, chrome):
            del goals, frame, chrome
            return ("a-goal-nobody-declared",)

    runtime = _two_screens(judge=_Inventing())

    _, execution = _dispatched(runtime, runtime.observe())

    assert execution.capability_evidence == ()


def test_a_settled_goal_is_not_judged_again():
    settled = GOAL.activate().close(
        __import__("yewo.discovery.contracts", fromlist=["CapabilityState"]).CapabilityState.REACHED,
        ("observation-1",),
    )
    perception = _Perception(
        _screen(_button()),
        _screen(_button()),
        phashes=["0" * 16, "f" * 16],
    )
    runtime = _runtime(perception, judge=_EverythingSatisfied(), goals=(settled,))

    _, execution = _dispatched(runtime, runtime.observe())

    assert execution.capability_evidence == ()


def test_a_session_that_never_started_is_refused_at_construction():
    """`DeviceControl.driver` is None until the session starts."""

    class _Unstarted(DiscoveryDevice):
        def __init__(self):
            super().__init__()
            self.driver = None

    with pytest.raises(ValueError, match="started device session"):
        DeviceDiscoveryRuntime(
            _Unstarted(),
            _Perception(_screen(_button())),
            app_id="com.example.app",
            setup_contract_sha256="b" * 64,
            goals=(GOAL,),
            goal_judge=_NothingSatisfied(),
            scrubber=SecretScrubber(),
            sleep=lambda _seconds: None,
        )


class _Sizeless(_Perception):
    """A reading whose scan reports no measured viewport."""

    def perceive(self):
        result = super().perceive()
        self.last_scan_info = ScanInfo(
            mode="full",
            reason="first_pass",
            phash=self.current_hash,
            screen_width=0,
            screen_height=0,
        )
        return result


def test_a_screen_reporting_no_size_offers_no_control():
    """Geometry that cannot be scaled is not trusted. The surface drops it all."""
    batch = _runtime(_Sizeless(_screen(_button()))).observe()

    assert batch.observation.controls == ()
    assert batch.candidates == ()


def test_relative_bounds_never_divides_by_a_missing_extent():
    assert _relative_bounds((10, 20, 30, 40), (0, 0)) == RelativeBounds(
        left=0, top=0, right=0, bottom=0
    )


def test_the_control_centre_reaches_the_device_unchanged():
    """Dispatch runs through the real orchestrator, so the point is not re-derived."""
    runtime = _two_screens()
    batch = runtime.observe()

    taps, candidate, _ = _dispatched_through(runtime, batch)

    assert taps == [(candidate.action.coordinates.x, candidate.action.coordinates.y)]
    assert taps == [(100, 130)]
