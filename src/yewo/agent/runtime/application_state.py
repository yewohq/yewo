"""Device Agent application-state terminals and bounded recovery."""

from collections.abc import Callable

from yewo.agent.actuation.diagnostics import (
    ActionApplicationFailure,
    outcome_with_application_failure,
)
from yewo.agent.actuation import ActionOutcome, AuthorizedAction
from yewo.agent.runtime.records import AgentTerminalReason
from yewo.devices.application_lifecycle import ApplicationState, trusted_system_surface
from yewo.observations import Observation, ObservationService
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowInterrupted
from yewo.artifacts import new_record_identifier


class ApplicationRecovery:
    """Attempt one configured recovery without creating another agent loop."""

    def __init__(
        self,
        recover: Callable[[], bool] | None,
        restart_after_failure: Callable[[], bool] | None,
        action_diagnostic_source: Callable[[str], ActionApplicationFailure | None] | None,
        observations: ObservationService,
    ) -> None:
        self._recover = recover
        self._restart_after_failure = restart_after_failure
        self._action_diagnostic_source = action_diagnostic_source
        self._observations = observations
        self._used = False

    def apply(
        self,
        observation: Observation,
        terminal: AgentTerminalReason | None,
        interruption: InterruptionSignal,
    ) -> tuple[Observation, AgentTerminalReason | None, str | None]:
        """Return a fresh active observation after at most one recovery attempt."""
        if terminal is None or self._recover is None or self._used:
            return observation, terminal, None
        self._used = True
        _require_continuation(interruption)
        if not self._recover():
            return observation, terminal, None
        _require_continuation(interruption)
        recovered = self._observations.capture(
            predecessor_observation_id=observation.observation_id
        )
        return recovered, observation_terminal(recovered), recovered.observation_id

    def apply_failure(
        self,
        observation: Observation,
        interruption: InterruptionSignal,
    ) -> tuple[Observation, AgentTerminalReason | None, str | None, bool]:
        """Restart once after a platform-established application failure."""
        terminal = AgentTerminalReason.APPLICATION_EXITED
        if self._restart_after_failure is None or self._used:
            return observation, terminal, None, False
        self._used = True
        _require_continuation(interruption)
        if not self._restart_after_failure():
            return observation, terminal, None, False
        _require_continuation(interruption)
        recovered = self._observations.capture(
            predecessor_observation_id=observation.observation_id
        )
        recovered_terminal = observation_terminal(recovered)
        return recovered, recovered_terminal, recovered.observation_id, recovered_terminal is None

    def apply_action(
        self,
        authorization: AuthorizedAction,
        outcome: ActionOutcome,
        observation: Observation,
        interruption: InterruptionSignal,
    ) -> tuple[
        Observation,
        AgentTerminalReason | None,
        str | None,
        ActionOutcome,
        ActionApplicationFailure | None,
    ]:
        """Override visual success and recover after a diagnosed failure."""
        failure = (
            self._action_diagnostic_source(authorization.authorization_id)
            if self._action_diagnostic_source is not None
            else None
        )
        if failure is None:
            terminal = observation_terminal(observation)
            recovered, terminal, recovery_id = self.apply(
                observation,
                terminal,
                interruption,
            )
            return recovered, terminal, recovery_id, outcome, None
        failed_outcome = outcome_with_application_failure(outcome, failure)
        recovered, terminal, recovery_id, succeeded = self.apply_failure(
            observation,
            interruption,
        )
        failure = failure.model_copy(update={"recovery_succeeded": succeeded})
        return recovered, terminal, recovery_id, failed_outcome, failure

    def diagnose_action(self, authorization: AuthorizedAction) -> ActionApplicationFailure | None:
        """Return platform-established failure evidence without requiring an observation."""
        if self._action_diagnostic_source is None:
            return None
        return self._action_diagnostic_source(authorization.authorization_id)


def restart_and_rebaseline_application(
    is_installed: Callable[[], bool],
    relaunch: Callable[[], None],
    drain_recovery_logs: Callable[[str], object],
    identifier_factory: Callable[[], str] = new_record_identifier,
) -> bool:
    """Relaunch once and consume diagnostics caused by the recovery itself."""
    if not is_installed():
        return False
    relaunch()
    drain_recovery_logs(identifier_factory())
    return True


def observation_terminal(observation: Observation | None) -> AgentTerminalReason | None:
    """Return the terminal implied by the observed application state."""
    if observation is not None and observation.application_state in {
        ApplicationState.NOT_INSTALLED,
        ApplicationState.NOT_RUNNING,
    }:
        return AgentTerminalReason.APPLICATION_EXITED
    if (
        observation is not None
        and trusted_system_surface(
            observation.device.platform,
            observation.foreground_application_id,
        )
        is not None
    ):
        return None
    if observation is not None and (
        observation.application_state is not ApplicationState.FOREGROUND
        or (
            observation.foreground_application_id is not None
            and observation.foreground_application_id != observation.application.application_id
        )
    ):
        return AgentTerminalReason.APPLICATION_INACTIVE
    return None


def application_terminal_reason(terminal: AgentTerminalReason, *, initial: bool) -> str:
    """Describe an application-state terminal without platform-specific wording."""
    state = (
        "was not running"
        if terminal is AgentTerminalReason.APPLICATION_EXITED
        else "did not own the foreground"
    )
    suffix = " when the Device Agent started" if initial else ""
    return f"The application {state}{suffix}."


def _require_continuation(interruption: InterruptionSignal) -> None:
    if interruption.is_set():
        raise RunWorkflowInterrupted
