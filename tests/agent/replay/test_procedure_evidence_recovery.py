"""Recovery of compiled evidence from retained settlement observations."""

from yewo.agent import AgentTask, AgentTerminalReason
from yewo.agent.actuation import ActionKind, ActionOutcomeDisposition
from yewo.agent.replay import EvidenceResult, build_evidence_document

from agent.discovery.support import agent_result, observation


def test_evidence_recovers_a_recurring_unsettled_destination() -> None:
    source = observation("source", "0000000000000000")
    first = observation("cursor_on", "1000000000000000", offset_seconds=1)
    alternate = observation("cursor_off", "7000000000000000", offset_seconds=2)
    recurring = observation("cursor_on_again", "1000000000000000", offset_seconds=3)
    step = agent_result(
        source,
        recurring,
        action=ActionKind.TYPE,
        terminal=AgentTerminalReason.BUDGET_EXHAUSTED,
    ).steps[0]
    assert step.outcome is not None
    unavailable = step.outcome.model_copy(
        update={
            "disposition": ActionOutcomeDisposition.UNAVAILABLE,
            "settled_observation_id": None,
            "signals": frozenset(),
            "reason": "No adjacent captures settled.",
        }
    )
    step = step.model_copy(
        update={
            "settled_observation_id": None,
            "settlement_observation_ids": (
                first.observation_id,
                alternate.observation_id,
                recurring.observation_id,
            ),
            "outcome": unavailable,
        }
    )

    evidence = build_evidence_document(
        run_id="run_1",
        objective_id="objective_1",
        task=AgentTask(
            objective="Enter one value.",
            completion_conditions=("The value is entered.",),
            discovery_mode=True,
        ),
        steps=(step,),
        observations=(source, first, alternate, recurring),
    )

    assert evidence.steps[0].destination_observation_id == recurring.observation_id
    assert evidence.steps[0].destination_state_id is not None
    assert evidence.steps[0].result is EvidenceResult.RECOVERED
