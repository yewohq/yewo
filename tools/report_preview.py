#!/usr/bin/env python3
"""Writes a sample run report, so the page can be looked at without a device.

The run it describes is invented. It carries one scenario that failed on the
app, one that ended on the provider, a replayed walk, masked values, findings
and a rule nothing evidenced. That is the page at its busiest, which is the
state worth checking a change against.

    uv run python tools/report_preview.py build/preview

A real run's record renders the same page, with no run and no device:

    uv run python tools/report_preview.py build/preview --from path/to/run.json

Images are not written by the invented run. The report links the files a real
run would leave beside it, so a preview shows the frames empty.
"""
from pathlib import Path
import sys

from yewo.agent.run.record import (
    ActionRecord, DecisionRecord, GroundingRecord, OutcomeRecord, StepRecord,
)
from yewo.identity import AppIdentity, CiIdentity, DeviceIdentity, GitIdentity
from yewo.testing.models import (
    JourneyReport, QAIssue, RunEnvironment, RuleVerification, ScenarioReport,
)
from yewo.testing.report.page import now, page_of_record, save_html_report, save_page


def step(n, title, ok=True, screen="Basket", why="", observed="", error=None,
         shots=True, replay=False, refusal=None):
    return StepRecord(
        n=n, source="replay" if replay else "model", screen_name=screen,
        decision=DecisionRecord(
            current_screen=screen,
            evaluation_previous_goal="The previous tap landed on the right control.",
            observation=observed or "The basket lists three items and a Pay button.",
            reasoning=why or "The next thing the scenario asks for is on this screen.",
            goal_satisfied=ok,
            goal_evidence="An order number is shown." if ok and n > 4 else "",
        ),
        action=ActionRecord(type="tap", described=title, target_label="Pay £24.99"),
        grounding=GroundingRecord(mode="marks", resolved=refusal is None, answer=7,
                                  refusal=refusal, unmatched=bool(refusal)),
        outcome=OutcomeRecord(
            success=ok,
            observation=observed or ("The screen changed." if ok else
                                     "The app showed “Something went wrong. Please try again.”"),
            error=error,
        ),
        screenshots={"raw": f"step{n:03d}_041912_raw.png",
                     "annotated": f"step{n:03d}_041912.png",
                     "scan": f"step{n:03d}_041912_scan.json"} if shots else {},
        perceive_ms=820.0, plan_ms=1180.0, act_ms=140.0,
    )


report = JourneyReport(
    journey_id="checkout",
    journey_name="Checkout",
    description=(
        "Log in with {{TEST_USER_EMAIL}} and {{TEST_USER_PASSWORD}}. Add two items to "
        "the basket, go to checkout and pay with the saved card {{TEST_CARD}}. "
        "Success means an order confirmation screen shows an order number."
    ),
    status="FAILED",
    total_duration=372.4,
    passed_count=3,
    failed_count=2,
    scenarios=[
        ScenarioReport(
            test_id="TS_1", scenario_name="Happy path checkout",
            cell="functional.happy_path.0", status="PASSED",
            duration=94.2, steps_executed=6,
            steps=[step(n, t) for n, t in enumerate(
                ["Open the basket", "Tap “Checkout”", "Choose the saved card {{TEST_CARD}}",
                 "Tap “Pay £24.99”", "Check: an order confirmation is shown",
                 "Check: the order number is legible"], start=1)],
            rule_verifications=[
                RuleVerification(rule="An order confirmation screen shows an order number.",
                                 rule_id="expected:0", status="passed",
                                 reasoning="Order #41822 is shown under the heading “Thank you”.",
                                 screen_name="Order confirmation"),
            ],
        ),
        ScenarioReport(
            test_id="TS_2", scenario_name="Apply discount code",
            cell="functional.happy_path.1", status="PASSED",
            duration=71.0, steps_executed=7,
            steps=[step(n, t, replay=n < 3) for n, t in enumerate(
                ["Open the basket", "Tap “Add a code”", "Type SAVE10",
                 "Tap “Apply”", "Check: the total fell by 10%"], start=1)],
        ),
        ScenarioReport(
            test_id="TS_3", scenario_name="Pay with saved card",
            cell="functional.happy_path.2", status="FAILED",
            duration=31.4, steps_executed=5,
            failure_reason=(
                "After tapping Pay £24.99 the app showed “Something went wrong” and stayed "
                "on the basket. Yewo waited 10 seconds and checked twice. The Pay button was "
                "found and tapped correctly, so this looks like a defect in the app."
            ),
            steps=[
                step(1, "Open the basket"),
                step(2, "Tap “Checkout”", screen="Checkout"),
                step(3, "Choose the saved card {{TEST_CARD}}", screen="Payment",
                     why="The card list was still sliding up, so this waited and tried again.",
                     refusal=None),
                step(4, "Tap “Pay £24.99”", screen="Checkout"),
                step(5, "Check: an order confirmation is shown", ok=False, screen="Basket",
                     error="No order confirmation after 10 s",
                     observed="Still on the basket, with “Something went wrong. Please try again.”"),
            ],
            rule_verifications=[
                RuleVerification(rule="An order confirmation screen shows an order number.",
                                 rule_id="expected:0", status="failed",
                                 reasoning="No order number anywhere on screen after two checks.",
                                 screen_name="Basket"),
                RuleVerification(rule="The card number is never shown in full.",
                                 rule_id="security:0", status="passed",
                                 reasoning="The saved card reads “Visa ····4242”.",
                                 screen_name="Payment"),
            ],
            detected_issues=[
                QAIssue(issue_type="sensitive_data_exposure", severity="high",
                        description=("The full card number is legible for about two seconds "
                                     "before the field masks it."),
                        current_screen="Payment"),
            ],
        ),
        ScenarioReport(
            test_id="TS_4", scenario_name="Empty basket",
            cell="functional.negative.0", status="PASSED", duration=38.0, steps_executed=4,
            steps=[step(n, t) for n, t in enumerate(
                ["Open the basket", "Remove every item", "Check: the basket says it is empty"], start=1)],
            detected_issues=[
                QAIssue(issue_type="accessibility_violation", severity="medium",
                        description="The “Remove item” button is 32 × 32 px; 44 × 44 px is the minimum.",
                        current_screen="Basket"),
            ],
        ),
        ScenarioReport(
            test_id="TS_5", scenario_name="Card number is masked",
            cell="security.negative.0", status="ERROR", cause="provider",
            duration=12.9, steps_executed=2,
            failure_reason="The model provider returned 503 twice. Nothing was learned about the app.",
            steps=[step(1, "Open the basket"),
                   step(2, "Tap “Checkout”", ok=False, error="provider unavailable", shots=False)],
        ),
    ],
    unevidenced_security_rules=["A signed-out user must not reach another account's orders."],
    environment=RunEnvironment(
        yewo_version="0.1.0", model="openrouter/google/gemini-2.5-flash",
        fast_model="openrouter/google/gemini-2.5-flash-lite", platform="android",
        grounding="marks", detector="orasul/deki-yolo@287f12d1",
        app=AppIdentity(package="shop.basket.android", version="1.4.0", build="212"),
        device=DeviceIdentity(model="Pixel 8", os_version="14", emulator=True, serial="5554"),
        git=GitIdentity(sha="a3f9c1e4d2b", branch="main"),
        ci=CiIdentity(provider="github"),
    ),
)

usage = {"total_calls": 118, "total_estimated_cost_usd": 0.1943, "cost_unknown_calls": 0}

def from_record(path: Path, out: Path) -> Path:
    """The page a reader gets from `run.json` alone, with nothing else open."""
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    view = page_of_record(payload, generated=now())
    return save_page(view, out, label="record")


arguments = sys.argv[1:]
record = None
if "--from" in arguments:
    where = arguments.index("--from")
    record = Path(arguments[where + 1])
    arguments = arguments[:where] + arguments[where + 2:]

out = Path(arguments[0] if arguments else "build/preview")
out.mkdir(parents=True, exist_ok=True)

if record is not None:
    print(from_record(record, out))
else:
    path = save_html_report(report, out, screenshots=out / "screenshots", usage=usage,
                            label="sample", artifacts=("screenshots/", "logs/", "run.json"))
    print(path)
