"""Device Agent application-state terminals and bounded recovery."""

from collections.abc import Callable

from yewo.agent.runtime_records import AgentTerminalReason
from yewo.devices.application_lifecycle import ApplicationState
from yewo.observations import Observation, ObservationService
from yewo.runs import InterruptionSignal
from yewo.runs.errors import RunWorkflowInterrupted


class ApplicationRecovery:
    """Attempt one configured recovery without creating another agent loop."""

    def __init__(
        self,
        recover: Callable[[], bool] | None,
        observations: ObservationService,
    ) -> None:
        self._recover = recover
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


def observation_terminal(observation: Observation | None) -> AgentTerminalReason | None:
    """Return the terminal implied by the observed application state."""
    if observation is not None and observation.application_state in {
        ApplicationState.NOT_INSTALLED,
        ApplicationState.NOT_RUNNING,
    }:
        return AgentTerminalReason.APPLICATION_EXITED
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
