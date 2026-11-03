"""Application-state and settlement tests for the shared runtime."""

from pathlib import Path

from yewo.agent import AgentTerminalReason
from yewo.agent.actuation.diagnostics import ActionApplicationFailure
from yewo.agent.runtime.application_state import (
    observation_terminal,
    restart_and_rebaseline_application,
)
from yewo.agent.actuation import ActionKind, ActionOutcomeSignal
from yewo.agent.planning.planner import PlanningRequest
from yewo.agent.recovery.advisor import RecoveryAssessment, RecoveryDisposition
from yewo.agent.runtime.records import AgentStep
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
)
from yewo.observations import Observation, ObservationCaptureError

from agent.actuation.support import proposal
from agent.runtime.test_agent import (
    NeverInterrupted,
    ObservationSequence,
    RuntimeDevice,
    SequencePlanner,
    _mode_policy,
    _observation,
    _runtime,
    _task,
)


def test_application_exit_before_planning_is_a_distinct_terminal(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", application_state=1)]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.usage.planning_requests == 0
    assert result.steps == ()
    assert planner.requests == []
    assert not (tmp_path / "assets/agent/steps.jsonl").exists()


def test_background_application_stops_before_planning(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [_observation("obs_1", "0000000000000000", application_state=3)]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_INACTIVE
    assert result.usage.planning_requests == 0
    assert planner.requests == []


def test_android_permission_controller_is_not_an_inactive_application() -> None:
    observation = _observation("obs_1", "0000000000000000", application_state=3).model_copy(
        update={"foreground_application_id": "com.google.android.permissioncontroller"}
    )

    assert observation_terminal(observation) is None


def test_discovery_can_recover_one_initial_application_exit(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", application_state=1),
            _observation("obs_2", "1111111111111111"),
        ]
    )
    recovery_calls = 0

    def recover_application() -> bool:
        nonlocal recovery_calls
        recovery_calls += 1
        return True

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        application_recovery=recover_application,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert recovery_calls == 1
    assert planner.requests[0].observation.observation_id == "obs_2"


def test_relaunch_consumes_recovery_owned_diagnostics_before_returning() -> None:
    events: list[str] = []

    recovered = restart_and_rebaseline_application(
        lambda: True,
        lambda: events.append("relaunch"),
        lambda recovery_id: events.append(f"drain:{recovery_id}"),
        lambda: "recovery_1",
    )

    assert recovered
    assert events == ["relaunch", "drain:recovery_1"]


def test_unchanged_screen_settles_only_on_the_final_attempt(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "0000000000000000", offset_seconds=1),
            _observation("obs_3", "0000000000000000", offset_seconds=2),
            _observation("obs_4", "0000000000000000", offset_seconds=3),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(),
        _mode_policy(maximum_settlement_observations=3),
        NeverInterrupted(),
    )

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert observations.capture_count == 4
    assert result.steps[0].settlement_observation_ids == ("obs_2", "obs_3", "obs_4")
    assert result.steps[0].settled_observation_id == "obs_4"


def test_application_exit_after_action_is_recorded_as_an_observed_outcome(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", application_state=1, offset_seconds=1),
            _observation("obs_3", "1111111111111111", application_state=1, offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.steps[0].outcome is not None
    assert ActionOutcomeSignal.APP_EXITED in result.steps[0].outcome.signals
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_EXITED


def test_discovery_can_recover_one_post_action_application_exit(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", application_state=1),
            _observation("obs_3", "1111111111111111", application_state=1),
            _observation("obs_4", "2222222222222222"),
        ]
    )
    recovery_calls = 0

    def recover_application() -> bool:
        nonlocal recovery_calls
        recovery_calls += 1
        return True

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        application_recovery=recover_application,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert recovery_calls == 1
    assert result.steps[0].terminal_reason is None
    assert result.steps[0].settled_observation_id == "obs_3"
    assert result.steps[0].recovery_observation_id == "obs_4"
    assert planner.requests[1].observation.observation_id == "obs_4"


def test_action_diagnostic_restarts_and_informs_the_planner(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111"),
            _observation("obs_3", "1111111111111111"),
            _observation("obs_4", "2222222222222222"),
        ]
    )
    recovery_calls = 0
    diagnostic_action_ids: list[str] = []

    def restart_application() -> bool:
        nonlocal recovery_calls
        recovery_calls += 1
        return True

    def read_action_diagnostic(action_id: str) -> ActionApplicationFailure:
        diagnostic_action_ids.append(action_id)
        return ActionApplicationFailure(
            classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
            failure_kinds=(ApplicationFailureKind.CRASH,),
            summary="The application crashed after Save.",
            source_artifact_ids=("diagnostic_1",),
        )

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        application_failure_recovery=restart_application,
        action_diagnostic_source=read_action_diagnostic,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    failure = result.steps[0].application_failure
    assert result.terminal_reason is AgentTerminalReason.DONE_REQUESTED
    assert recovery_calls == 1
    assert len(diagnostic_action_ids) == 1
    assert failure is not None
    assert failure.recovery_succeeded
    assert result.steps[0].terminal_reason is None
    assert result.steps[0].recovery_observation_id == "obs_4"
    assert result.steps[0].outcome is not None
    assert ActionOutcomeSignal.APP_EXITED in result.steps[0].outcome.signals
    assert planner.requests[1].observation.observation_id == "obs_4"
    assert planner.requests[1].history[0].application_failure == failure


def test_recovery_advisor_can_stop_after_one_relaunch(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT), proposal(ActionKind.DONE)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111"),
            _observation("obs_3", "1111111111111111"),
            _observation("obs_4", "2222222222222222"),
        ]
    )

    class StopAdvisor:
        def assess(
            self,
            planning_request: PlanningRequest,
            failed_step: AgentStep,
            recovery_observation: Observation,
        ) -> RecoveryAssessment:
            return RecoveryAssessment(
                assessment_id="assessment_1",
                failed_proposal_id=failed_step.proposal.proposal_id,
                pre_failure_observation_id=planning_request.observation.observation_id,
                recovery_observation_id=recovery_observation.observation_id,
                disposition=RecoveryDisposition.STOP,
                reason="The same submission cannot recover safely.",
                model_request_id="request_1",
                model_call_count=1,
            )

    def read_action_diagnostic(unused_action_id: str) -> ActionApplicationFailure:
        return ActionApplicationFailure(
            classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
            failure_kinds=(ApplicationFailureKind.CRASH,),
            summary="The application crashed after Save.",
        )

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        application_failure_recovery=lambda: True,
        action_diagnostic_source=read_action_diagnostic,
        recovery_advisor=StopAdvisor(),
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert len(planner.requests) == 1
    assert result.usage.planning_requests == 2
    assert result.recovery_budget_credit.planning_requests == 1
    assert result.recovery_budget_credit.assessment_id == "assessment_1"


def test_action_diagnostic_overrides_a_foreground_visual_transition(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111"),
            _observation("obs_3", "1111111111111111"),
        ]
    )

    def read_action_diagnostic(unused_action_id: str) -> ActionApplicationFailure:
        return ActionApplicationFailure(
            classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
            failure_kinds=(ApplicationFailureKind.CRASH,),
            summary="The application crashed after Save.",
            source_artifact_ids=("diagnostic_1",),
        )

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        action_diagnostic_source=read_action_diagnostic,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.reason == "The application crashed after Save."
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.steps[0].application_failure is not None


def test_capture_failure_with_crash_diagnostics_is_an_application_exit(
    tmp_path: Path,
) -> None:
    class CaptureFailureAfterDispatch(ObservationSequence):
        def capture(self, *, predecessor_observation_id: str | None = None) -> Observation:
            if self.capture_count > 0:
                raise ObservationCaptureError("The application process disappeared.")
            return super().capture(predecessor_observation_id=predecessor_observation_id)

    observations = CaptureFailureAfterDispatch([_observation("obs_1", "0000000000000000")])

    def read_action_diagnostic(unused_action_id: str) -> ActionApplicationFailure:
        return ActionApplicationFailure(
            classification=ApplicationDiagnosticClassification.APPLICATION_CRASH,
            failure_kinds=(ApplicationFailureKind.CRASH,),
            summary="The application crashed during outcome capture.",
            source_artifact_ids=("diagnostic_1",),
        )

    result = _runtime(
        tmp_path,
        SequencePlanner([proposal(ActionKind.WAIT)]),
        observations,
        RuntimeDevice(),
        action_diagnostic_source=read_action_diagnostic,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.reason == "The application crashed during outcome capture."
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert result.steps[0].outcome is None
    assert result.steps[0].settled_observation_id is None
    assert result.steps[0].application_failure is not None


def test_discovery_recovery_is_attempted_only_once(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000", application_state=1),
            _observation("obs_2", "1111111111111111"),
            _observation("obs_3", "2222222222222222", application_state=1),
            _observation("obs_4", "2222222222222222", application_state=1),
        ]
    )
    recovery_calls = 0

    def recover_application() -> bool:
        nonlocal recovery_calls
        recovery_calls += 1
        return True

    result = _runtime(
        tmp_path,
        planner,
        observations,
        RuntimeDevice(),
        application_recovery=recover_application,
    ).run(_task(), _mode_policy(), NeverInterrupted())

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_EXITED
    assert recovery_calls == 1
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_EXITED


def test_background_application_after_action_is_terminal(tmp_path: Path) -> None:
    planner = SequencePlanner([proposal(ActionKind.WAIT)])
    observations = ObservationSequence(
        [
            _observation("obs_1", "0000000000000000"),
            _observation("obs_2", "1111111111111111", application_state=3, offset_seconds=1),
            _observation("obs_3", "1111111111111111", application_state=3, offset_seconds=2),
        ]
    )

    result = _runtime(tmp_path, planner, observations, RuntimeDevice()).run(
        _task(), _mode_policy(), NeverInterrupted()
    )

    assert result.terminal_reason is AgentTerminalReason.APPLICATION_INACTIVE
    assert result.steps[0].terminal_reason is AgentTerminalReason.APPLICATION_INACTIVE
