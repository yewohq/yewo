"""Deciding a stored run's verdicts again, with no device and no model.

The record is the evidence a run leaves. It is only evidence if the verdict
in it can be reached again from the rest of it. That was not true until the
judge's inputs were kept. A record held the verdict and the auditor's
answers alone. Nothing in it could tell a changed rule from a changed app.

So this takes one scenario out of a `run.json` and asks the judge what it
would answer now. The judge is the same function a live run calls. It is
given the same inputs, out of the file rather than out of a walk.

**This reads a record. It never writes one.** A verdict reached again is an
answer about the code. It is not a correction to what happened. A record
that could be rewritten after the fact is not evidence.

What reads this today is the test that proves a record is sufficient. The
report that shows a reader which verdicts moved is a later phase, and it
brings its own presentation.
"""

import json
from pathlib import Path
from typing import List, Tuple

from yewo.agent.memory.history import AgentHistory
from yewo.agent.run.outcomes import DeviceRunResult
from yewo.errors import Cause, UsageError
from yewo.testing.judge import decide
from yewo.testing.models import QAIssue, RuleVerification, VerdictInputs

FIRST_WITH_INPUTS = 2
"""The schema version from which a record holds the judge's inputs."""


class UnreadableRunRecord(Exception):
    """A scenario that kept too little of what its verdict was decided from.

    Raised rather than answered from what is there. Half the inputs reach a
    verdict that looks like a judgement and is not one.
    """


def scenarios_in(path: Path) -> Tuple[int, List[dict]]:
    """The schema a record was written under, and the scenarios in it.

    Raises:
        UsageError: The file is absent, or is not a run record. A file that
            cannot be read is never reported as a run with no scenarios.
    """
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise UsageError(f"Could not read {path}: {exc}") from exc
    except ValueError as exc:
        raise UsageError(f"{path} is not readable as JSON: {exc}") from exc
    if not isinstance(record, dict) or "scenarios" not in record:
        raise UsageError(f"{path} does not hold a run record.")
    return record.get("schema", 0), list(record["scenarios"])


def verdict_of(entry: dict, schema: int) -> decide.Verdict:
    """What the judge answers now about one scenario in a record.

    Raises:
        UnreadableRunRecord: The record kept no account of what the judge was given.
    """
    if schema < FIRST_WITH_INPUTS or not entry.get("verdict_inputs"):
        raise UnreadableRunRecord(
            f"written under schema {schema}, which kept no record of what "
            "the judge was given"
        )
    inputs = VerdictInputs.model_validate(entry["verdict_inputs"])
    return decide.determine_status(
        run_result=DeviceRunResult(
            status=inputs.run_status,
            steps_executed=entry.get("steps_executed") or 0,
            failure_reason=entry.get("failure_reason"),
            history=AgentHistory(),
            capability_receipts=inputs.receipts,
            cause=Cause(entry["cause"]) if entry.get("cause") else None,
        ),
        stop_reason=inputs.stop_reason,
        all_verifications=[
            RuleVerification.model_validate(one)
            for one in entry.get("rule_verifications", [])
        ],
        all_issues=[
            QAIssue.model_validate(one) for one in entry.get("detected_issues", [])
        ],
        qa_outputs=[inputs],
        assessments=list(range(inputs.screens_judged)),
        required_outcomes=inputs.expected_outcomes,
        security_rules=inputs.security_rules,
        all_findings=inputs.sensitive_findings,
        required_capabilities=inputs.required_capabilities,
    )
