"""Offline integration tests for the complete cold-start discovery path."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from discovery_device_fakes import (
    DiscoveryDevice as _Device,
    PerceptionSequence,
)
from perception_fakes import detected, frame, read

from yewo.discovery.model_proposer import ControlChoice, GoalJudgement
from yewo.llm.prompts import PromptStore
from yewo.testing.discovery_wiring import V0_DISCOVERY_BUDGETS
from yewo.testing.generation_readiness import (
    GenerationReadinessState,
    assess_generation_readiness,
)
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType
from yewo.testing.runner_startup import build_runner_startup_authority
from yewo.testing.wiring import _discovery_factory

def _journey():
    from yewo.testing.models import UserJourney

    return UserJourney(
        id="checkout",
        name="Checkout",
        description="A user places an order.",
        values={},
    )


def _cells():
    return [CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)]


def _screen(title, control):
    """One Android screen: platform status bar, the app's title, one control."""
    return frame(
        elements=(
            detected((0, 0, 400, 24), name="StatusBar", text="7:51"),
            detected((20, 30, 380, 70), name="Text", text=title),
            detected((20, 100, 180, 160), text=control),
        ),
        text_lines=(read((20, 30, 380, 70), title),),
    )


def _perception():
    return PerceptionSequence(
        _screen("Menu", "Place order"),
        _screen("Order placed", "Done"),
        phashes=["0" * 16, "f" * 16],
    )


def _text_drawer_perception():
    root = frame(
        elements=(
            detected((0, 0, 400, 24), name="StatusBar", text="7:51"),
            detected((20, 30, 380, 70), name="Text", text="Menu"),
            detected((20, 100, 180, 160), name="ImageView"),
        ),
        text_lines=(read((20, 30, 380, 70), "Menu"),),
    )
    drawer = frame(
        elements=(
            detected((0, 0, 400, 24), name="StatusBar", text="7:51"),
            detected((20, 80, 180, 140), name="Text", text="Navigation"),
            detected((20, 160, 180, 220), name="Text", text="Place order"),
        ),
        text_lines=(
            read((20, 80, 180, 140), "Navigation"),
            read((20, 160, 180, 220), "Place order"),
        ),
    )
    return PerceptionSequence(
        root,
        drawer,
        _screen("Order placed", "Done"),
        phashes=["0" * 16, "a" * 16, "f" * 16],
    )


@pytest.fixture(autouse=True)
def _no_settle_poll():
    with patch("yewo.agent.perception.settle.time.sleep"):
        yield


@pytest.fixture
def app_file(tmp_path):
    path = tmp_path / "app.apk"
    path.write_bytes(b"an application artifact")
    return path


@contextmanager
def _cold_start(tmp_path, app_file, answers):
    """A startup authority wired exactly as `wiring.runner_for` wires one.

    The discovery lane builds its own client, the way `yewo lock` does, so the
    fake is installed where that client is constructed. Without this the lane
    would reach a real provider.
    """
    factory = _discovery_factory(
        tmp_path / "knowledge",
        app_path=str(app_file),
        platform="android",
        model="openrouter/m",
        perception_model="yewo-ui@v1",
    )
    assert factory is not None
    authority = build_runner_startup_authority(
        lambda _journey: None,
        None,
        "default",
        False,
        tmp_path / "knowledge",
        None,
        None,
        None,
        factory,
    )
    lane = SimpleNamespace(
        generate_response=None,
        usage_callback=None,
        prompt_store=PromptStore(tmp_path / "prompts"),
    )

    def build_lane(*, model, cache_affinity, usage_callback, prompt_store):
        """The lane's client, recording and reporting the way a real one does."""
        del model, cache_affinity
        remaining = list(answers)

        def answer(*, messages, **_kwargs):
            prompt_store.store_messages(messages)
            usage_callback(SimpleNamespace(cost=0.0005))
            if not remaining:
                raise AssertionError("the lane asked more than the test answered")
            model_answer = remaining.pop(0)
            if isinstance(model_answer, Exception):
                raise model_answer
            return model_answer

        lane.generate_response = Mock(side_effect=answer)
        return lane

    with patch("yewo.testing.discovery_wiring.LLMClient", side_effect=build_lane):
        yield authority, lane


def test_a_cold_run_discovers_an_unseen_app_and_becomes_ready(tmp_path, app_file):
    """DISCOVERY_REQUIRED, then a map, then READY. The join nothing else covers."""
    journey = _journey()
    assert (
        assess_generation_readiness(journey, None).state
        is GenerationReadinessState.DISCOVERY_REQUIRED
    )
    device = _Device()
    answers = [ControlChoice(control=2), GoalJudgement(satisfied_goals=[1])]

    with _cold_start(tmp_path, app_file, answers) as (authority, lane):
        authority.attach_discovery(device, _perception(), lane, _scrubber())
        knowledge = authority.knowledge_source.discover(journey)

    assert knowledge is not None
    evidence = knowledge.generation_evidence
    assert evidence is not None
    readiness = assess_generation_readiness(
        journey.model_copy(update={"coverage": _cells()}),
        evidence,
        discovery_completed=True,
    )

    assert readiness.state is GenerationReadinessState.READY
    assert device.dispatched, "discovery dispatched no action"
    assert (tmp_path / "knowledge" / "app_maps").is_dir()
    assert tuple((tmp_path / "knowledge" / "discovery_runs").glob("*.json"))


def test_a_cold_run_can_follow_a_text_only_drawer_row(tmp_path, app_file):
    journey = _journey()
    device = _Device()
    answers = [
        ControlChoice(control=2),
        GoalJudgement(satisfied_goals=[]),
        ControlChoice(control=2),
        GoalJudgement(satisfied_goals=[1]),
    ]

    with _cold_start(tmp_path, app_file, answers) as (authority, lane):
        authority.attach_discovery(
            device, _text_drawer_perception(), lane, _scrubber()
        )
        knowledge = authority.knowledge_source.discover(journey)

    assert knowledge is not None
    assert knowledge.generation_evidence is not None
    assert device.dispatched == [(100, 130), (100, 190)]


def test_a_cold_run_that_misses_its_goal_reports_what_it_reached(tmp_path, app_file):
    journey = _journey()
    answers = [
        ControlChoice(control=2),
        GoalJudgement(satisfied_goals=[]),
        ControlChoice(control=None),
    ]

    with _cold_start(tmp_path, app_file, answers) as (authority, lane):
        authority.attach_discovery(_Device(), _perception(), lane, _scrubber())
        knowledge = authority.knowledge_source.discover(journey)

    assert knowledge is not None
    evidence = knowledge.generation_evidence
    readiness = assess_generation_readiness(
        journey.model_copy(update={"coverage": _cells()}),
        evidence,
        discovery_completed=True,
    )

    assert readiness.state is GenerationReadinessState.CAPABILITY_UNOBSERVED
    assert "The run did reach:" in readiness.reason
    assert "Menu" in readiness.reason


def test_the_run_record_charges_every_model_call_discovery_made(tmp_path, app_file):
    """Including the judgement that ended the run, which no proposal carried."""
    journey = _journey()
    answers = [ControlChoice(control=2), GoalJudgement(satisfied_goals=[1])]

    with _cold_start(tmp_path, app_file, answers) as (authority, lane):
        authority.attach_discovery(_Device(), _perception(), lane, _scrubber())
        authority.knowledge_source.discover(journey)
        calls = lane.generate_response.call_count

    runs = tuple((tmp_path / "knowledge" / "discovery_runs").glob("*.json"))
    assert len(runs) == 1
    import json

    usage = json.loads(runs[0].read_text())["usage"]
    assert usage["model_proposals"] == calls
    assert usage["model_proposals"] <= V0_DISCOVERY_BUDGETS.model_proposals


def test_a_declared_value_on_screen_never_reaches_the_persisted_run(
    tmp_path, app_file
):
    """The guard is wired, not only constructible."""
    journey = _journey().model_copy(update={"values": {"card": "Menu"}})
    answers = [ControlChoice(control=2), GoalJudgement(satisfied_goals=[1])]

    with _cold_start(tmp_path, app_file, answers) as (authority, lane):
        authority.attach_discovery(_Device(), _perception(), lane, _scrubber())
        authority.knowledge_source.discover(journey)

    for run in (tmp_path / "knowledge" / "discovery_runs").glob("*.json"):
        assert "Menu" not in run.read_text()


def _scrubber():
    from yewo.utils.secrets import SecretScrubber

    return SecretScrubber()


def test_discovery_records_what_it_asked_the_model(tmp_path, app_file):
    """The discovery-owned client writes prompts to the run's prompt store."""
    journey = _journey()
    answers = [ControlChoice(control=2), GoalJudgement(satisfied_goals=[1])]

    with _cold_start(tmp_path, app_file, answers) as (authority, lane):
        authority.attach_discovery(
            _Device(), _perception(), lane, _scrubber()
        )
        authority.knowledge_source.discover(journey)

    recorded = sorted((tmp_path / "prompts").glob("*.json"))
    assert recorded, "discovery recorded no prompt"
    asked = "".join(path.read_text() for path in recorded)
    assert "These goals are open" in asked
    assert "A user places an order." in asked


def test_discovery_calls_are_filed_under_their_own_lane():
    """They borrow the generation callback, so the shape has to name them."""
    from yewo.testing.run_usage import DISCOVERY, GENERATION, component_for

    assert component_for(GENERATION, ControlChoice.__name__) == DISCOVERY
    assert component_for(GENERATION, GoalJudgement.__name__) == DISCOVERY
    assert component_for(GENERATION, "ScenarioDraft") == GENERATION
