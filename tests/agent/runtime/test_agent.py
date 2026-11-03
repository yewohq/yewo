"""Deterministic integration tests for the shared Device Agent loop."""
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, cast
import pytest

from yewo.agent import (
    AgentModePolicy,
    AgentStepStore,
    AgentTask,
    AgentTerminalReason,
    DeviceAgent,
    PlannerDecisionSource,
)
from yewo.agent.actuation import (
    ActionExecutor,
    ActionKind,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    ActionOutcomeVerifier,
    ActionProposal,
    DeclaredValueReference,
    SideEffectClass,
)
from yewo.agent.runtime.application_state import ApplicationRecovery
from yewo.agent.planning.planner import ActionProposalSource, PlanningRequest
from yewo.devices.focus import FocusedFieldReading
from yewo.devices.interaction import ScrollDirection
from yewo.grounding import VisualGrounder
from yewo.observations import Observation, ObservationService
from yewo.runs.errors import RunWorkflowInterrupted

from agent.actuation.support import limits, proposal
from grounding.support import DeterministicGrounder, ExpressionGrounder
from observations.factories import build_observation, build_screenshot


class SequencePlanner:
    """Return configured proposal templates against the current observation."""

    def __init__(
        self,
        proposals: list[ActionProposal],
        *,
        model_requests_per_plan: int = 1,
    ) -> None:
        self._proposals = iter(proposals)
        self._model_requests_per_plan = model_requests_per_plan
        self.requests: list[PlanningRequest] = []

    def required_model_requests(self, request: PlanningRequest) -> int:
        """Count one deterministic stand-in planner request."""
        del request
        return self._model_requests_per_plan

    def plan(self, request: PlanningRequest) -> ActionProposal:
        self.requests.append(request)
        selected = next(self._proposals)
        return selected.model_copy(
            update={
                "proposal_id": f"proposal-{len(self.requests)}",
                "source_observation_id": request.observation.observation_id,
            }
        )


class ObservationSequence:
    """Return configured observations while preserving requested lineage."""

    def __init__(self, observations: list[Observation]) -> None:
        self._observations = iter(observations)
        self.capture_count = 0

    def capture(self, *, predecessor_observation_id: str | None = None) -> Observation:
        self.capture_count += 1
        return next(self._observations).model_copy(
            update={"predecessor_observation_id": predecessor_observation_id}
        )


class RuntimeDevice:
    """Record dispatches and expose only secret-free post-action checks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.active_text = ""
        self.reject = False

    def _record(self, operation: str, *arguments: Any) -> None:
        if self.reject:
            from yewo.devices.errors import DeviceActionRejectedError

            raise DeviceActionRejectedError("Configured refusal with private detail.")
        self.calls.append((operation, arguments))

    def tap(self, x: int, y: int, *, count: int = 1, interval_seconds: float = 0.1) -> None:
        self._record("tap", x, y, count, interval_seconds)

    def focus(self, x: int, y: int) -> object:
        self._record("focus", x, y)
        return object()

    def type_focused_text(self, text: str) -> bool | None:
        self._record("type_focused_text", text)
        self.active_text = text
        return True

    def clear_focused_text(self) -> bool | None:
        self._record("clear_focused_text")
        self.active_text = ""
        return True

    def double_tap(self, x: int, y: int) -> None:
        self._record("double_tap", x, y)

    def long_press(self, x: int, y: int, duration_seconds: float) -> None:
        self._record("long_press", x, y, duration_seconds)

    def drag(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        *,
        duration_seconds: float = 1,
    ) -> None:
        self._record("drag", start_x, start_y, end_x, end_y, duration_seconds)

    def scroll(
        self,
        direction: ScrollDirection,
        *,
        left: int = 0,
        top: int = 0,
        right: int | None = None,
        bottom: int | None = None,
        distance_fraction: float = 0.6,
    ) -> None:
        self._record(
            "scroll",
            direction,
            left,
            top,
            right,
            bottom,
            distance_fraction,
        )

    def back(self) -> None:
        self._record("back")

    def dismiss_keyboard(self) -> bool:
        self._record("dismiss_keyboard")
        return True

    def wait(self, duration_seconds: float) -> None:
        self._record("wait", duration_seconds)

    def focused_field_reading(self) -> FocusedFieldReading:
        return FocusedFieldReading(True, (0, 0, 100, 50), True)


class NeverInterrupted:
    @staticmethod
    def is_set() -> bool:
        return False


class InterruptAfterFirstPostActionCapture:
    def __init__(self, observations: ObservationSequence) -> None:
        self._observations = observations

    def is_set(self) -> bool:
        return self._observations.capture_count >= 2


def _observation(
    observation_id: str,
    fingerprint: str,
    *,
    application_state: int = 4,
    keyboard_visible: bool | None = False,
    offset_seconds: int = 0,
) -> Observation:
    screenshot = build_screenshot(artifact_id=f"shot-{observation_id}")
    base = build_observation(
        observation_id=observation_id,
        redacted_screenshot=screenshot,
        application_state=application_state,
        keyboard_visible=keyboard_visible,
        captured_at=datetime(2026, 10, 30, tzinfo=timezone.utc) + timedelta(seconds=offset_seconds),
    )
    return base.model_copy(
        update={
            "screen_fingerprint": base.screen_fingerprint.model_copy(
                update={
                    "value": fingerprint,
                    "masked_pixels_sha256": sha256(fingerprint.encode()).hexdigest(),
                    "source_artifact_id": screenshot.artifact_id,
                }
            )
        }
    )


def _task() -> AgentTask:
    return AgentTask(
        objective="Open the account form.",
        completion_conditions=("The account form is visible.",),
        declared_value_names=("account_password",),
    )


def _mode_policy(**updates: object) -> AgentModePolicy:
    values: dict[str, object] = {
        "action_limits": limits(),
        "maximum_steps": 5,
        "maximum_grounding_requests": 10,
        "maximum_settlement_observations": 2,
        "settlement_interval_seconds": 0,
    }
    values.update(updates)
    return AgentModePolicy.model_validate(values)


def _runtime(
    tmp_path: Path,
    planner: ActionProposalSource,
    observations: ObservationSequence,
    device: RuntimeDevice,
    *,
    declared_value_resolver=None,
    grounder: VisualGrounder | None = None,
    application_recovery=None,
    application_failure_recovery=None,
    action_diagnostic_source=None,
    screen_comparison=None,
    monotonic=None,
    planner_decision_source: PlannerDecisionSource | None = None,
    reused_flow_id: str | None = None,
    recovery_advisor=None,
    action_verifier=None,
    initial_planning_requests: int = 0,
) -> DeviceAgent:
    return DeviceAgent(
        planner,
        grounder or DeterministicGrounder(100, 200),
        cast(ObservationService, observations),
        ActionExecutor(device, declared_value_resolver=declared_value_resolver),
        ActionOutcomeVerifier(),
        AgentStepStore(tmp_path),
        sleep=lambda unused: None,
        application_recovery=ApplicationRecovery(
            application_recovery, application_failure_recovery, action_diagnostic_source,
            cast(ObservationService, observations),
        ),
        screen_comparison=screen_comparison,
        **(
            {"planner_decision_source": planner_decision_source}
            if planner_decision_source is not None
            else {}
        ),
        reused_flow_id=reused_flow_id,
        recovery_advisor=recovery_advisor,
        action_verifier=action_verifier,
        initial_planning_requests=initial_planning_requests,
        **({"monotonic": monotonic} if monotonic is not None else {}),
    )


def _stored_steps(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "assets/agent/steps.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize(
    "task",
    [
        _task(),
        AgentTask(
            objective="Learn the route to the account form.",
            completion_conditions=("The account form route is observed.",),
        ),
        AgentTask(
            objective="Exercise the invalid-password scenario.",
            completion_conditions=("The validation message is visible.",),
        ),
    ],
    ids=("normal", "discovery", "scenario"),
)
def test_runtime_connects_all_components_and_records_canonical_steps(
    tmp_path: Path,
    task: AgentTask,
) -> None:
    planner = SequencePlanner([proposal(), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )
    device = RuntimeDevice()
    result = _runtime(tmp_path, planner, observations, device).run(
        task, _mode_policy(), NeverInterrupted()
    )
    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert result.usage.planning_requests == 2
    assert result.usage.grounding_requests == 1
    assert result.usage.dispatched_actions == 1
    assert result.final_observation_id == "obs_3"
    assert result.steps[0].outcome is not None
    assert result.steps[0].outcome.disposition is ActionOutcomeDisposition.OBSERVED
    assert ActionOutcomeSignal.PIXELS_CHANGED in result.steps[0].outcome.signals
    assert planner.requests[1].history[0].outcome == result.steps[0].outcome
    assert {ActionKind.BACK, ActionKind.DISMISS_KEYBOARD}.isdisjoint(
        planner.requests[0].available_actions
    )
    assert [step["terminal_reason"] for step in _stored_steps(tmp_path)] == [
        None,
        "done_requested",
    ]
    assert [step["schema_version"] for step in _stored_steps(tmp_path)] == [2, 2]
    assert [set(step) for step in _stored_steps(tmp_path)] == [
        {
            "schema_version",
            "step_id",
            "step_index",
            "source_observation_id",
            "proposal",
            "decision",
            "target_grounding",
            "destination_grounding",
            "authorization",
            "refusal",
            "receipt",
            "settlement_observation_ids",
            "settled_observation_id",
            "recovery_observation_id",
            "outcome",
            "application_failure",
            "terminal_reason",
        },
    ] * 2


def test_policy_refusal_returns_to_the_same_planner_history(tmp_path: Path) -> None:
    restricted_limits = limits(
        allowed_actions=frozenset({ActionKind.WAIT}),
        supported_actions=frozenset({ActionKind.WAIT}),
        allowed_side_effects=frozenset({SideEffectClass.NONE}),
    )
    planner = SequencePlanner(
        [proposal(), proposal(ActionKind.WAIT), proposal(ActionKind.CANNOT_PROGRESS)]
    )
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "0000000000000000", offset_seconds=1),
            _observation("obs_3", "0000000000000000", offset_seconds=2),
        ]
    )
    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
    ).run(
        _task(),
        _mode_policy(action_limits=restricted_limits),
        NeverInterrupted(),
    )
    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.steps[0].refusal is not None
    assert planner.requests[1].history[0].refusal == result.steps[0].refusal
    second_outcome = planner.requests[2].history[1].outcome
    assert second_outcome is not None
    assert second_outcome.disposition is ActionOutcomeDisposition.NO_OBSERVABLE_CHANGE
    assert result.usage.dispatched_actions == 1
    assert result.usage.grounding_requests == 0


def test_unsettled_application_returns_unavailable_feedback_to_the_planner(
    tmp_path: Path,
) -> None:
    planner = SequencePlanner([proposal(ActionKind.TYPE), proposal(ActionKind.CANNOT_PROGRESS)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", keyboard_visible=True),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "2222222222222222", offset_seconds=2),
            _observation("obs_4", "3333333333333333", offset_seconds=3),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.CANNOT_PROGRESS
    assert result.steps[0].outcome is not None
    assert result.steps[0].outcome.disposition is ActionOutcomeDisposition.UNAVAILABLE
    assert result.steps[0].settled_observation_id is None
    assert planner.requests[1].history[0].outcome == result.steps[0].outcome
    assert result.final_observation_id == "obs_3"


def test_interruption_after_dispatch_preserves_partial_step(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
        ]
    )
    interruption = InterruptAfterFirstPostActionCapture(observations)

    with pytest.raises(RunWorkflowInterrupted):
        _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
            _task(), _mode_policy(), interruption
        )

    stored = _stored_steps(tmp_path)
    assert stored[0]["terminal_reason"] == "interrupted"
    receipt = stored[0]["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["disposition"] == "dispatched"
    assert stored[0]["outcome"] is None
    assert stored[0]["settlement_observation_ids"] == ["obs_2"]


def test_declared_text_is_verified_without_entering_step_artifacts(tmp_path: Path) -> None:
    secret = "correct horse battery staple"
    typed = proposal(
        ActionKind.TYPE,
        text=DeclaredValueReference(value_name="account_password"),
    )
    planner = SequencePlanner([typed, proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", keyboard_visible=True),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )
    runtime = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        declared_value_resolver=lambda name: secret if name == "account_password" else "",
    )

    result = runtime.run(_task(), _mode_policy(), NeverInterrupted())

    assert result.steps[0].outcome is not None
    assert ActionOutcomeSignal.TEXT_LANDED in result.steps[0].outcome.signals
    assert secret not in (tmp_path / "assets/agent/steps.jsonl").read_text()


def test_device_refusal_is_a_distinct_terminal_without_settlement(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])
    device = RuntimeDevice()
    device.reject = True

    result = _runtime(tmp_path, planner, observations, device).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.DEVICE_REFUSED
    assert result.usage.dispatched_actions == 0
    assert observations.capture_count == 1
    assert "private detail" not in (tmp_path / "assets/agent/steps.jsonl").read_text()


def test_grounding_budget_stops_before_an_unrecorded_model_request(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DRAG)])
    observations = ObservationSequence([_observation("obs_1", "0000000000000000")])

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(maximum_grounding_requests=1),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.BUDGET_EXHAUSTED
    assert result.usage.grounding_requests == 0
    assert result.steps[0].target_grounding is None
    assert result.steps[0].destination_grounding is None


def test_drag_grounds_and_dispatches_two_distinct_endpoints(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DRAG), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", offset_seconds=1),
            _observation("obs_3", "1111111111111111", offset_seconds=2),
        ]
    )
    device = RuntimeDevice()
    result = _runtime(
        tmp_path,
        planner,
        observations,
        device,
        grounder=ExpressionGrounder(),
    ).run(_task(), _mode_policy(), NeverInterrupted())
    assert result.usage.grounding_requests == 2
    assert device.calls == [("drag", (20, 200, 150, 200, 1.0))]


def test_step_budget_stops_the_loop_without_an_extra_planner_request(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "0000000000000000", offset_seconds=1),
            _observation("obs_3", "0000000000000000", offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(maximum_steps=1), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.BUDGET_EXHAUSTED
    assert result.usage.planning_requests == 1
    assert len(planner.requests) == 1
