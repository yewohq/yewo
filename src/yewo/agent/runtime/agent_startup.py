"""Capture and recover the initial application state for an agent run."""

from yewo.agent.narration import narrate_observation
from yewo.agent.runtime import result as runtime_result
from yewo.agent.runtime import settlement
from yewo.agent.runtime.application_state import ApplicationRecovery, observation_terminal
from yewo.agent.runtime.records import AgentResult, AgentTerminalReason
from yewo.observations import Observation, ObservationService
from yewo.runs import InterruptionSignal


def capture_initial_application_state(
    observations: ObservationService,
    recovery: ApplicationRecovery,
    interruption: InterruptionSignal,
) -> tuple[Observation, AgentTerminalReason | None]:
    """Capture the launch state and apply bounded application recovery."""
    settlement.require_agent_continuation(interruption)
    observation = observations.capture()
    narrate_observation(observation, label="Initial")
    terminal = observation_terminal(observation)
    recovered, terminal, unused_recovery_id = recovery.apply(
        observation,
        terminal,
        interruption,
    )
    del unused_recovery_id
    return recovered, terminal


def build_initial_result(
    terminal: AgentTerminalReason,
    observation: Observation,
    planning_requests: int,
) -> AgentResult:
    """Build the result for an application that is terminal at launch."""
    return runtime_result.build_initial_agent_result(
        terminal,
        settlement.application_terminal_reason(terminal, initial=True),
        observation,
        planning_requests,
    )
