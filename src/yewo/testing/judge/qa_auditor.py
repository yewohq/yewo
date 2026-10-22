"""Audits one screen against the rules and reports what it shows.

The auditor receives the screenshot and the rules, and nothing else. Not the
planned action, not the history, not what the scenario expects next. A verdict
counts only if the thing that gave it could disagree with the planner. Withhold
the plan and it can.

It reads the clean screenshot, not the numbered one. Marks cover the interface,
and a drawn box hides the defects the auditor looks for: clipped text,
overlapping controls, and overflow stripes.

One assessment covers every rule on one screen. The auditor reads a given
screen once.

It judges rules and reports defects. It does not judge whether the scenario
finished: that needs the goal, and withholding the goal is the point.
"""

import logging
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from yewo.agent.perception import elements
from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import MultiPartMessage
from yewo.testing.judge import rules
from yewo.testing.models import IssueKind, QAIssue, RuleVerification, SensitiveFinding
from functools import partial

from yewo.utils.redactor import safe_text

logger = logging.getLogger(__name__)


class Assessment(BaseModel):
    """What one look at one screen says about the rules."""

    screen_name: str = Field(
        ..., description="Descriptive name of the screen, e.g. 'Login Screen'."
    )
    rule_verifications: List[RuleVerification] = Field(
        default_factory=list,
        description="One entry per rule listed, carrying that rule's bracketed id.",
    )
    detected_issues: List[QAIssue] = Field(
        default_factory=list,
        description="Visual defects visible on this screen.",
    )
    sensitive_findings: List[SensitiveFinding] = Field(
        default_factory=list,
        description=(
            "One entry for each place a value the listed rules treat as "
            "sensitive could appear, saying what is actually there."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _normalize_common_provider_shapes(cls, value):
        """Normalize equivalent shapes produced despite advisory JSON schemas."""
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        if "screen_name" not in normalized and "current_screen" in normalized:
            normalized["screen_name"] = normalized["current_screen"]
        verifications = normalized.get("rule_verifications")
        if isinstance(verifications, dict):
            normalized["rule_verifications"] = [
                {
                    "rule": rule,
                    "status": status,
                    "reasoning": "Provider returned a compact rule-status mapping.",
                }
                for rule, status in verifications.items()
            ]
        for verification in normalized.get("rule_verifications") or []:
            if isinstance(verification, dict) and "reasoning" not in verification:
                verification["reasoning"] = verification.pop(
                    "details", "No reasoning supplied by the provider."
                )
        for issue in normalized.get("detected_issues") or []:
            if not isinstance(issue, dict):
                continue
            if "description" not in issue and "issue" in issue:
                issue["description"] = issue.pop("issue")
            issue.setdefault("issue_type", IssueKind.OTHER.value)
        return normalized



PROMPT = """You are auditing one screenshot of a mobile app.

You are not driving the app and you do not know what it was asked to do. Judge
only what this screen shows.

### Rules to audit
{rules}

### How to answer
1. `screen_name`: what this screen is, from what is on it.
2. `rule_verifications`: one entry per rule above, copying its bracketed id into
   `rule_id`.
   - "passed" means you can see this screen satisfying the rule, now.
   - "failed" means you can see this screen breaking it, now. A rule is not
     failed because a field could show something once it is filled, because a
     label suggests a value might appear beside it, or because the layout
     could expose something under other conditions. Report what is rendered,
     not what the design permits. If the violating content is not on the
     screen in front of you, the rule is not failed.
   - "untested" means this screen does not show you either way. A rule about a
     screen you are not looking at is "untested", never "failed".
   - `applicable` is false when this screen holds nothing the rule could apply
     to, such as a rule about a field this screen does not have. Pair it with
     "untested". Never mark a rule passed because its subject is absent, and
     never from what you expect a field to do once it is filled. Judge the
     pixels in front of you.
   - `reasoning`: what you saw, not what you infer.
3. `detected_issues`: visual defects on this screen — clipped or overlapping
   text, layout overflow stripes, or a blank or crashed screen. Each needs a
   `severity`.
4. `sensitive_findings`: one entry for every place a value the rules above
   treat as sensitive could be on this screen. Say what is there, not what
   could be there:
   - "value" only when you can read the real characters. Copy them into
     `quoted`.
   - "masked" when dots, bullets or asterisks stand in for them.
   - "label_only" when there is a caption and nothing beside it.
   - "absent" when the screen holds no such field at all.
   A caption is not a value. An empty field is not a value.
"""


ENTERING = elements.ARRIVALS
"""Scan reasons that mean a different screen, rather than the same one working.

A layout shift is fewer than half the previous elements still matching, which
is how a crash screen or a modal arrives when it happens to resemble what it
replaced. Typing reports stable or ocr_refresh: the fingerprint moves, the
screen does not.
"""


class ScreenWatch:
    """Decides when a screen is worth auditing, rather than auditing each step.

    Twice per screen: once on arrival, and once on the state it is left in.
    Arrival is what catches a crash or a blank screen while there is still a
    scenario to stop. The parting look is what catches a value that appeared
    while the screen was being filled in, which is where a secret leaks.

    The steps between are the same screen being worked. A cache keyed on the
    fingerprint reuses nothing there, because every keystroke moves it.

    A defect that appears and is gone again before the screen is left goes
    unseen. That is the trade: a transient is worth less than the six-fold
    saving, and the states that matter are both still covered.
    """

    def __init__(self, auditor: "QAAuditor"):
        self._auditor = auditor
        self._pending: Optional[tuple] = None
        self._visit: set = set()
        self._forced = False

    def look_again(self) -> None:
        """Audits the next step whatever the screen reports.

        A verdict withdrawn as too early is promised a second look. The screen
        it was withdrawn on usually reports stable next step, so without this
        the promise is not kept and the verdict is simply lost.
        """
        self._forced = True

    def look(self, snapshot) -> List[Assessment]:
        """Assessments to act on now. Empty while a screen is merely worked."""
        reason = snapshot.scan_info.reason if snapshot.scan_info else "navigation"
        fresh: List[Assessment] = []

        if self._forced:
            self._forced = False
            self._visit.discard(snapshot.current_phash)
            fresh.extend(self._once(snapshot.raw_img_bytes, snapshot.current_phash))
        elif reason in ENTERING:
            if self._pending is not None:
                fresh.extend(self._once(*self._pending))
            # A visit ends here. Coming back to this screen later is a new one,
            # and gets judged again rather than being answered from memory.
            self._visit = set()
            fresh.extend(self._once(snapshot.raw_img_bytes, snapshot.current_phash))

        self._pending = (snapshot.raw_img_bytes, snapshot.current_phash)
        return fresh

    def close(self) -> List[Assessment]:
        """The screen the scenario ended on, which nothing else will leave."""
        if self._pending is None:
            return []
        last, self._pending = self._pending, None
        return self._once(*last)

    def _once(self, screenshot: bytes, screen_id: Optional[str]) -> List[Assessment]:
        """One assessment per screen state, per visit to that screen.

        A screen left exactly as it was found is read twice, on arrival and on
        departure. The auditor's cache spares the second model call but still
        answers, and reporting that answer would count every rule on that
        screen twice.

        Per visit, not per run. A scenario that comes back to a screen is
        asking about it again, and a verdict withheld because the same screen
        was judged earlier leaves the rule looking untested.
        """
        if screen_id is not None and screen_id in self._visit:
            return []
        if screen_id is not None:
            self._visit.add(screen_id)
        return [self._auditor.audit(screenshot, screen_id)]


class QAAuditor:
    """Asks a model what one screen shows, with no knowledge of the plan."""

    def __init__(self, llm_client: LLMClient, expected_outcomes, security_rules, scrubber=None):
        self._llm = llm_client
        # Outcomes keep the index they were given, so a verdict still names the
        # rule it answers even though only some of them are asked about.
        on_screen = rules.on_screen_only(expected_outcomes)
        self._rules = rules.every_rule(on_screen, security_rules)
        self._asked = rules.listed_ids(on_screen, rules.EXPECTED) | rules.listed_ids(
            security_rules, rules.SECURITY
        )
        self._by_statement = rules.by_statement(on_screen, security_rules)
        self._scrubber = scrubber
        self._by_screen: dict[str, Assessment] = {}

    def audit(self, screenshot: bytes, screen_id: Optional[str]) -> Assessment:
        """What this screen shows. Asks once per distinct screen.

        A screen that did not change cannot give a different answer, so an
        identical fingerprint reuses the answer. The match is exact on purpose.
        A looser one would reuse an assessment across the small change the
        auditor exists to catch — a value appearing in a field that was empty.

        Expect few hits. Typing changes the fingerprint, so a form pays per
        step.
        """
        if screen_id and screen_id in self._by_screen:
            logger.debug("Screen %s was audited already. Reusing it.", screen_id[:12])
            return self._kept(self._by_screen[screen_id])

        message = MultiPartMessage()
        message.add_text("The screen to audit:")
        message.add_image(screenshot)
        message.add_text(PROMPT.format(rules=self._rules))

        logger.info("Auditing the current screen against the rules...")
        assessment = self._answered(message)
        if self._asked and not self._any_listed(assessment):
            # Every rule on this screen would otherwise report unevidenced,
            # and the scenario would fail on evidence the app had given.
            logger.warning("The audit named none of the rules it was given. Asking again.")
            assessment = self._answered(message)
        self._clean(assessment)

        if screen_id:
            self._by_screen[screen_id] = assessment
        return self._kept(assessment)

    def _answered(self, message: MultiPartMessage) -> Assessment:
        """One answer, with each verdict matched to the rule it is about."""
        assessment: Assessment = self._llm.generate_response([message], Assessment)
        self._name_the_rule_answered(assessment)
        self._only_what_was_asked(assessment)
        return assessment

    def _name_the_rule_answered(self, assessment: Assessment) -> None:
        """Fills the id for a verdict that quoted its rule and left the id out.

        Observed on a working model: every rule answered correctly, each one
        quoted word for word, and `rule_id` null on all of them. Nothing
        downstream reads such a verdict, because the gate matches a verdict to
        its rule by id. The scenario then reports every outcome unevidenced,
        which blames the run for an answer it was given.
        """
        for verification in assessment.rule_verifications:
            if verification.rule_id:
                continue
            found = self._by_statement.get((verification.rule or "").strip())
            if found:
                verification.rule_id = found

    def _any_listed(self, assessment: Assessment) -> bool:
        """Whether any verdict names a rule this screen was shown."""
        return any(
            verification.rule_id in self._asked
            for verification in assessment.rule_verifications
        )

    def _only_what_was_asked(self, assessment: Assessment) -> None:
        """Drops verdicts naming a rule this screen was never shown.

        A verdict carrying an id that was not listed is not evidence about
        that rule. It would settle an outcome nothing looked at, or fail the
        scenario on a security rule the auditor was never asked.

        A verdict with no id is kept. The auditor may raise a rule of its own,
        and one that names nothing settles nothing either way.
        """
        kept = [
            verify
            for verify in assessment.rule_verifications
            if verify.rule_id is None or verify.rule_id in self._asked
        ]
        dropped = len(assessment.rule_verifications) - len(kept)
        if dropped:
            logger.warning("Ignored %d verdict(s) naming a rule that was not listed.", dropped)
        assessment.rule_verifications = kept

    @staticmethod
    def _kept(assessment: Assessment) -> Assessment:
        """A copy, because a caller withdraws findings from what it receives.

        The step loop drops a crash reported on the first look, then checks the
        screen again. Returning the stored object would carry that withdrawal
        into every later look at the same screen, and a crash that persists
        would stay invisible.
        """
        return assessment.model_copy(deep=True)

    def _clean(self, assessment: Assessment) -> None:
        """Masks declared secrets in what the auditor wrote, before anything keeps it."""
        if self._scrubber is None:
            return
        scrub = partial(safe_text, scrubber=self._scrubber)
        assessment.screen_name = scrub(assessment.screen_name)
        for verification in assessment.rule_verifications:
            verification.rule = scrub(verification.rule)
            verification.reasoning = scrub(verification.reasoning)
        for issue in assessment.detected_issues:
            issue.description = scrub(issue.description)
