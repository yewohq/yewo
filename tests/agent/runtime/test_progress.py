"""Evidence-linked same-run workflow progress tests."""

import json
from pathlib import Path

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
)
from yewo.agent.runtime.progress import RunProgressLedger
from yewo.artifacts import ArtifactRetention, ArtifactSensitivity

from agent.actuation.support import proposal
from agent.runtime.test_agent import _observation


CREATE = "Create and save one reusable training schedule."
REOPEN = "Return to the saved schedule and establish that it can be repeated or updated."


def test_progress_ledger_retains_completed_work_and_routes(tmp_path: Path) -> None:
    before = _observation("obs_1", "0000000000000000").model_copy(
        update={"platform_context": "ScheduleList"}
    )
    after = _observation("obs_2", "1111111111111111").model_copy(
        update={"platform_context": "ScheduleEditor"}
    )
    candidate = proposal(
        ActionKind.TAP,
        target_expression="the floating add button",
        source_observation_id=before.observation_id,
    )
    ledger = RunProgressLedger(
        tmp_path,
        identifier_factory=iter(("event_1", "event_2", "event_3")).__next__,
    )
    ledger.start((CREATE, REOPEN), before, None)
    ledger.record_assessment(
        before,
        candidate,
        (CREATE,),
        REOPEN,
        None,
    )
    ledger.record_route(
        1,
        candidate,
        CREATE,
        before,
        after,
        ActionOutcome(
            outcome_id="outcome_1",
            authorization_id="authorization_1",
            receipt_id="receipt_1",
            action=ActionKind.TAP,
            source_observation_id=before.observation_id,
            settled_observation_id=after.observation_id,
            disposition=ActionOutcomeDisposition.OBSERVED,
            signals=frozenset({ActionOutcomeSignal.CONTEXT_CHANGED}),
            reason="The editor opened.",
            sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
            retention=ArtifactRetention.RUN,
        ),
    )

    context = ledger.progress_context(None)

    assert context.completed_work_items == (CREATE,)
    assert context.remaining_work == (REOPEN,)
    assert len(context.route_attempts) == 1
    route = context.route_attempts[0]
    assert route.work_item == CREATE
    assert route.target == "the floating add button"
    assert route.source_context == "ScheduleList"
    assert route.destination_context == "ScheduleEditor"
    events = [
        json.loads(line)
        for line in (tmp_path / "assets/agent/progress-ledger.jsonl").read_text().splitlines()
    ]
    assert [event["kind"] for event in events] == ["initialized", "assessment", "route"]
