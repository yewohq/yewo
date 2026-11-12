"""Deterministic tests for post-action settlement timing."""

from pathlib import Path
from typing import cast

from yewo.agent.actuation import ActionKind
from yewo.agent.planning.evidence import PlannerDecisionMetadata, PlannerDecisionSource
from yewo.agent.runtime.observation_settlement import (
    ActionSettlementPolicy,
    capture_settled_observation,
    default_action_settlement_policy,
    observations_are_compatible,
    observations_are_identical,
    resolve_action_settlement_policy,
)
from yewo.agent.runtime.limits import AgentModePolicy
from yewo.observations import Observation, ObservationService, VisualSettlementProbe
from yewo.run_narration import RunNarrator, narration_scope

from agent.actuation.support import limits
from agent.runtime.test_agent import NeverInterrupted, _mode_policy, _observation


class OrderedObservations:
    def __init__(
        self,
        events: list[str],
        observations: list[Observation],
        *,
        materialized: Observation | None = None,
    ) -> None:
        self._events = events
        self._observations = iter(observations)
        self._latest: Observation | None = None
        self._materialized = materialized

    def capture_settlement_probe(self) -> VisualSettlementProbe:
        self._events.append("probe")
        self._latest = next(self._observations)
        return VisualSettlementProbe.from_observation(self._latest)

    def capture(self, *, predecessor_observation_id: str | None = None) -> Observation:
        del predecessor_observation_id
        self._events.append("materialize")
        if self._latest is None:
            raise AssertionError("A settlement probe must precede materialization.")
        return self._materialized or self._latest


def test_settlement_captures_immediately_then_waits_for_stability() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    observations = OrderedObservations(
        events,
        [
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ],
    )

    capture = capture_settled_observation(
        cast(ObservationService, observations),
        source,
        default_action_settlement_policy(_mode_policy(settlement_interval_seconds=0.35)),
        NeverInterrupted(),
        lambda unused: events.append("sleep"),
    )

    assert events == ["probe", "sleep", "probe", "materialize"]
    assert capture.is_settled


def test_default_settlement_adds_the_immediate_capture_to_the_wait_window() -> None:
    policy = default_action_settlement_policy(
        AgentModePolicy(action_limits=limits(), maximum_grounding_requests=0)
    )

    assert policy.initial_delay_seconds == 0
    assert policy.maximum_observations == 7
    assert round(policy.interval_seconds * (policy.maximum_observations - 1), 2) == 2.1


def test_focus_actions_capture_a_third_frame_in_discovery_and_replay() -> None:
    policy = resolve_action_settlement_policy(
        _mode_policy(maximum_settlement_observations=6),
        object(),
        PlannerDecisionMetadata(source=PlannerDecisionSource.MODEL),
        ActionKind.FOCUS,
    )

    assert policy.minimum_observations == 3
    assert policy.maximum_observations == 6


def test_settlement_detects_exact_change_when_phash_is_unchanged() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    first = _observation("obs_2", "0000000000000000", offset_seconds=1)
    first = first.model_copy(
        update={
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"masked_pixels_sha256": "f" * 64}
            )
        }
    )
    second = first.model_copy(
        update={
            "observation_id": "obs_3",
            "screenshot": first.screenshot.model_copy(update={"artifact_id": "shot-obs_3"}),
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"source_artifact_id": "shot-obs_3"}
            ),
        }
    )

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, [first, second])),
        source,
        default_action_settlement_policy(_mode_policy(maximum_settlement_observations=3)),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.is_settled
    assert capture.observation_ids == ("obs_3",)


def test_settlement_accepts_minor_perceptual_variation_between_captures() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    first = _observation("obs_2", "1000000000000000", offset_seconds=1)
    second = _observation("obs_3", "1000000000000001", offset_seconds=2)

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, [first, second])),
        source,
        default_action_settlement_policy(_mode_policy(maximum_settlement_observations=3)),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.is_settled
    assert capture.observation_ids == ("obs_3",)


def test_observation_comparison_rejects_different_dimensions_without_a_mask_digest() -> None:
    first = _observation("obs_1", "1111111111111111").model_copy(update={"system_regions": None})
    first = first.model_copy(
        update={
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"system_region_mask_sha256": None}
            )
        }
    )
    second = first.model_copy(
        update={
            "observation_id": "obs_2",
            "screenshot": first.screenshot.model_copy(
                update={"artifact_id": "shot-obs_2", "width_px": first.screenshot.width_px + 1}
            ),
            "screen_fingerprint": first.screen_fingerprint.model_copy(
                update={"source_artifact_id": "shot-obs_2"}
            ),
        }
    )

    assert not observations_are_compatible(first, second)
    assert not observations_are_identical(first, second)


def test_materialized_observation_must_match_the_confirming_probe() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    stable = [
        _observation("probe_1", "1111111111111111", offset_seconds=1),
        _observation("probe_2", "1111111111111111", offset_seconds=2),
    ]
    moved = _observation("obs_2", "7777777777777777", offset_seconds=3)

    capture = capture_settled_observation(
        cast(
            ObservationService,
            OrderedObservations(events, stable, materialized=moved),
        ),
        source,
        default_action_settlement_policy(_mode_policy(maximum_settlement_observations=3)),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert not capture.is_settled
    assert capture.observation is moved


def test_settlement_accepts_an_exact_recurring_cursor_state() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    first = _observation("obs_2", "1000000000000000", offset_seconds=1)
    alternate = _observation("obs_3", "7000000000000000", offset_seconds=2)
    recurring = _observation("obs_4", "1000000000000000", offset_seconds=3)

    capture = capture_settled_observation(
        cast(
            ObservationService,
            OrderedObservations(events, [first, alternate, recurring]),
        ),
        source,
        default_action_settlement_policy(_mode_policy(maximum_settlement_observations=4)),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.is_settled
    assert capture.observation_ids == ("obs_4",)


def test_focus_settlement_captures_past_two_identical_caret_frames() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    caret_hidden = _observation("obs_2", "1000000000000000", offset_seconds=1)
    still_hidden = caret_hidden.model_copy(update={"observation_id": "obs_3"})
    caret_visible = _observation("obs_4", "1000000000000003", offset_seconds=3)
    hidden_again = caret_hidden.model_copy(update={"observation_id": "obs_5"})

    capture = capture_settled_observation(
        cast(
            ObservationService,
            OrderedObservations(
                events,
                [caret_hidden, still_hidden, caret_visible, hidden_again],
            ),
        ),
        source,
        ActionSettlementPolicy(
            maximum_observations=4,
            initial_delay_seconds=0,
            interval_seconds=0.2,
            minimum_observations=3,
        ),
        NeverInterrupted(),
        lambda unused: None,
    )

    assert capture.is_settled
    assert capture.observation_ids == ("obs_5",)


def test_action_policy_can_capture_immediately_then_use_its_interval() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    observations = OrderedObservations(
        events,
        [
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ],
    )

    capture = capture_settled_observation(
        cast(ObservationService, observations),
        source,
        ActionSettlementPolicy(
            maximum_observations=6,
            initial_delay_seconds=0,
            interval_seconds=0.2,
        ),
        NeverInterrupted(),
        lambda duration: events.append(f"sleep:{duration}"),
    )

    assert events == ["probe", "sleep:0.2", "probe", "materialize"]
    assert capture.is_settled


def test_settlement_records_requested_and_actual_sleep_time(tmp_path: Path) -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    observations = OrderedObservations(
        events,
        [
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ],
    )

    with narration_scope(RunNarrator(tmp_path)):
        capture_settled_observation(
            cast(ObservationService, observations),
            source,
            ActionSettlementPolicy(
                maximum_observations=2,
                initial_delay_seconds=0,
                interval_seconds=0.2,
            ),
            NeverInterrupted(),
            lambda unused: None,
        )

    trace = (tmp_path / "runtime/trace.log").read_text()
    assert "settlement_sleep_attempt=2 requested_ms=200.0 duration_ms=" in trace


def test_continuous_animation_stops_at_the_observation_limit() -> None:
    """A changing screen must preserve evidence and return without waiting forever."""
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    frames = [
        _observation("obs_2", "1111111111111111", offset_seconds=1),
        _observation("obs_3", "3333333333333333", offset_seconds=2),
        _observation("obs_4", "7777777777777777", offset_seconds=3),
        _observation("obs_5", "ffffffffffffffff", offset_seconds=4),
    ]

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, frames)),
        source,
        ActionSettlementPolicy(
            maximum_observations=4,
            initial_delay_seconds=0,
            interval_seconds=0.2,
        ),
        NeverInterrupted(),
        lambda duration: events.append(f"sleep:{duration}"),
    )

    assert not capture.is_settled
    assert capture.observation.observation_id == "obs_5"
    assert capture.observation_ids == ("obs_5",)
    assert events.count("probe") == 4
    assert events.count("materialize") == 1
    assert events.count("sleep:0.2") == 3


def test_replay_policy_accepts_its_first_changed_observation() -> None:
    events: list[str] = []
    source = _observation("obs_1", "0000000000000000")
    changed = _observation("obs_2", "1111111111111111", offset_seconds=1)

    capture = capture_settled_observation(
        cast(ObservationService, OrderedObservations(events, [changed])),
        source,
        ActionSettlementPolicy(
            maximum_observations=6,
            initial_delay_seconds=0,
            interval_seconds=0.2,
            required_stable_observations=1,
        ),
        NeverInterrupted(),
        lambda duration: events.append(f"sleep:{duration}"),
    )

    assert events == ["probe", "materialize"]
    assert capture.observation_ids == ("obs_2",)
    assert capture.is_settled
