"""Device Agent application-state terminals."""

from yewo.agent.runtime_records import AgentTerminalReason
from yewo.devices.application_lifecycle import ApplicationState
from yewo.observations import Observation


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
