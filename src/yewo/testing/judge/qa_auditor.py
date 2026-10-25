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
from functools import partial
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from yewo.agent.artifacts.redaction import RedactedImages
from yewo.agent.perception import elements
from yewo.llm.llm_client import LLMClient
from yewo.llm.messages import Message, MultiPartMessage, SystemMessage
from yewo.testing.judge import rules
from yewo.testing.models import (
    IssueKind,
    QAIssue,
    Rendering,
    RuleVerification,
    SensitiveFinding,
)
from yewo.utils.redactor import safe_text
from yewo.utils.secrets import (
    AppValueRendering,
    SECRET_MASK_GUIDANCE,
    SecretScrubber,
)

logger = logging.getLogger(__name__)


class ScreenAssessment(BaseModel):
    """What one look at one screen says about the rules."""

    screen_name: str = Field(
        ..., description="Descriptive name of the screen from its visible content."
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
        issues = []
        for raw_issue in normalized.get("detected_issues") or []:
            if isinstance(raw_issue, QAIssue):
                issues.append(raw_issue)
                continue
            if isinstance(raw_issue, str):
                issue = {"description": raw_issue}
            elif isinstance(raw_issue, dict):
                issue = raw_issue
            else:
                continue
            if "description" not in issue:
                for alias in ("issue", "reason", "details"):
                    if alias in issue:
                        issue["description"] = issue.pop(alias)
                        break
            if "issue_type" not in issue and "type" in issue:
                issue["issue_type"] = issue.pop("type")
            issue.setdefault("issue_type", IssueKind.OTHER.value)
            # Keep a provider's visible defect when it omitted only the
            # ranking. Medium is deliberately non-critical: the finding is
            # reported without allowing an invented severity to abort a run.
            issue.setdefault("severity", "medium")
            issues.append(issue)
        normalized["detected_issues"] = issues
        return normalized


PROMPT = """You audit one screenshot of a mobile app.

You do not drive the app. You do not know the plan, action history, or expected
next screen. Judge only this screenshot and local device evidence supplied with
it.

### Rules to audit
{rules}

### How to answer
1. Set `screen_name` from visible content.
2. Return one `rule_verifications` entry for every listed rule. Copy its
   bracketed id into `rule_id`.
   - Use "passed" only when visible evidence satisfies the rule now.
   - Use "failed" only when visible evidence contradicts the rule now.
   - Use "untested" when this screenshot cannot decide the rule.
   - Set `applicable` false when the rule's subject is absent from this screen.
     Pair false with "untested".
   - An empty subject can be applicable and untested.
   - Never turn an absent subject into a pass or failure.
   - State visible evidence in `reasoning`. Do not infer earlier or later state.
3. Put visible app defects in `detected_issues`.
   - Report clipping only when app content is visibly cut off.
   - Report overlap only when app-rendered items visibly cover each other.
   - Report overflow only when content or an overflow indicator exceeds bounds.
   - Report a crash only when visible failure UI identifies one.
   - A splash, progress indicator, skeleton, or loading message is loading UI.
     It is not a crash.
   - A plain screen can be blank. Pixels alone do not prove that the app hung.
4. Use `sensitive_findings` only for subjects the listed rules treat as
   sensitive.
   - Use "value" for readable characters and copy only visible characters into
     `quoted`.
   - Use "masked" when replacement glyphs stand in for a value.
   - Use "label_only" when the location is visible but no value is rendered.
   - Use "absent" when the subject does not appear on this screen.
   - Never reconstruct or guess hidden characters.

{secret_mask_guidance}
If a `yewo_secret_mask` covers evidence needed for a rule and no local device
evidence resolves it, mark that rule "untested".

Judge only the rules listed above. Do not invent requirements for a particular
kind of app. Do not infer behavior from a control's name, an icon, a disabled
state, or a possible future action.
"""


ENTERING = elements.ARRIVALS
"""Scan reasons that mean a different screen, rather than the same one working.

A layout shift has fewer than half the previous elements still matching. This
is how a crash screen or modal can arrive when it resembles the prior screen.
Typing reports stable or ocr_refresh. The fingerprint moves, but the screen
does not.
"""


class ScreenAuditSession:
    """Audit every observed screen state once during its current visit.

    A reading is the post-action state of the preceding turn. Auditing it when
    it is observed keeps the verdict in front of every later action. Exact
    duplicate fingerprints are assessed once per visit, while a changed field
    or keyboard state gets its own assessment.
    """

    def __init__(self, auditor: "QAAuditor"):
        self._auditor = auditor
        self._visit: set = set()
        self._forced = False

    def request_repeat_audit(self) -> None:
        """Audits the next step whatever the screen reports.

        A verdict withdrawn as too early is promised a second look. The screen
        it was withdrawn on usually reports stable next step, so without this
        the promise is not kept and the verdict is simply lost.
        """
        self._forced = True

    def audit_snapshot(
        self, snapshot, prepared_model_image: Optional[bytes] = None
    ) -> List[ScreenAssessment]:
        """Assess this exact post-action state before another action runs."""
        reason = snapshot.scan_info.reason if snapshot.scan_info else "navigation"
        if prepared_model_image is None and hasattr(self._auditor, "prepare_model_image"):
            prepared_model_image = self._auditor.prepare_model_image(
                snapshot.raw_img_bytes, snapshot.gui_elements
            )

        if self._forced:
            self._forced = False
            self._visit.discard(snapshot.current_phash)
        elif reason in ENTERING:
            self._visit = set()
        return self._audit_once(
            snapshot.raw_img_bytes,
            snapshot.current_phash,
            snapshot.gui_elements,
            prepared_model_image,
        )

    def finalize_audit(self) -> List[ScreenAssessment]:
        """Every observed state is already audited."""
        return []

    def audit_pending(self) -> List[ScreenAssessment]:
        """Compatibility hook. Observation already crossed the audit barrier."""
        return []

    def _audit_once(
        self,
        screenshot: bytes,
        screen_id: Optional[str],
        gui_elements,
        prepared_model_image: Optional[bytes] = None,
    ) -> List[ScreenAssessment]:
        """One assessment per screen state, per visit to that screen.

        A screen left exactly as it was found is read twice, on arrival and on
        departure. The auditor's cache spares the second model call but still
        answers, and reporting that answer would count every rule on that
        screen twice.

        This scope is per visit, not per run. A scenario that returns to a
        screen asks about it again. Withholding the new verdict would leave
        the rule looking untested.
        """
        if screen_id is not None and screen_id in self._visit:
            return []
        if screen_id is not None:
            self._visit.add(screen_id)
        if prepared_model_image is None:
            return [self._auditor.audit(screenshot, screen_id, gui_elements)]
        return [
            self._auditor.audit(
                screenshot,
                screen_id,
                gui_elements,
                prepared_model_image=prepared_model_image,
            )
        ]


class QAAuditor:
    """Asks a model what one screen shows, with no knowledge of the plan."""

    def __init__(self, llm_client: LLMClient, expected_outcomes, security_rules, scrubber=None):
        self._llm = llm_client
        # Outcomes keep the index they were given, so a verdict still names the
        # rule it answers even though only some of them are asked about.
        on_screen = rules.on_screen_only(expected_outcomes)
        self._rules = rules.every_rule(on_screen, security_rules)
        self._audited_rule_ids = rules.listed_ids(on_screen, rules.EXPECTED) | rules.listed_ids(
            security_rules, rules.SECURITY
        )
        self._rule_ids_by_statement = rules.map_statements_to_rule_ids(on_screen, security_rules)
        self._security_rules_by_id = {
            rules.rule_id(rules.SECURITY, index): str(rule)
            for index, rule in enumerate(security_rules)
        }
        self._scrubber = scrubber or SecretScrubber()
        self._model_images = RedactedImages(self._scrubber)
        self._by_screen: dict[str, ScreenAssessment] = {}

    def prepare_model_image(self, screenshot: bytes, gui_elements=()) -> bytes:
        """Return the secret-safe image for this exact screen reading."""
        return self._model_images.redact_model_image(screenshot, gui_elements)

    def audit(
        self,
        screenshot: bytes,
        screen_id: Optional[str],
        gui_elements=(),
        prepared_model_image: Optional[bytes] = None,
    ) -> ScreenAssessment:
        """What this screen shows. Asks once per distinct screen.

        An unchanged screen cannot give a different answer. An identical
        fingerprint therefore reuses the answer. The match is exact. A looser
        match could miss a value appearing in a field that was empty.

        Expect few hits. Typing changes the fingerprint, so a form pays per
        step.
        """
        if screen_id and screen_id in self._by_screen:
            logger.debug("Screen %s was audited already. Reusing it.", screen_id[:12])
            return self._copy_assessment(self._by_screen[screen_id])

        prompt = PROMPT.format(
            rules=self._rules,
            secret_mask_guidance=SECRET_MASK_GUIDANCE,
        )
        stable = SystemMessage(content=self._scrubber.for_model(prompt))
        current = MultiPartMessage()
        current.add_text(self._current_evidence_text())
        current.add_image(
            prepared_model_image
            if prepared_model_image is not None
            else self.prepare_model_image(screenshot, gui_elements)
        )
        messages: list[Message] = [stable, current]

        logger.info("Auditing the current screen against the rules...")
        assessment = self._audit_screen_and_normalize(messages)
        if self._audited_rule_ids and not self._every_rule_is_answered(assessment):
            # A partial answer silently loses evidence for whichever rule was
            # omitted. The final gate can then fail a screen the model actually
            # saw, so give one fresh answer a chance to satisfy the contract.
            logger.warning("The audit did not answer every rule it was given. Asking again.")
            assessment = self._audit_screen_and_normalize(messages)
        assessment.sensitive_findings.extend(self._local_secret_exposures(gui_elements, screen_id))
        self._discard_unsupported_secret_passes(assessment)
        self._scrub_assessment(assessment)

        if screen_id:
            self._by_screen[screen_id] = assessment
        return self._copy_assessment(assessment)

    def _current_evidence_text(self) -> str:
        """Describe secret-free local facts beside the changing screenshot."""
        if not isinstance(self._scrubber, SecretScrubber):
            return "The screen to audit:"
        renderings = self._scrubber.app_renderings()
        measured = [
            (label, rendering)
            for label, rendering in renderings.items()
            if rendering is not AppValueRendering.UNKNOWN
        ]
        if not measured:
            return "The screen to audit:"
        lines = [
            "Local device evidence for this screenshot:",
            "Yewo measured these app-rendered fields before adding transport masks.",
        ]
        lines.extend(
            f"- {label}: app_rendering={rendering.value}"
            for label, rendering in measured
        )
        lines.append("The screen to audit:")
        return "\n".join(lines)

    def _discard_unsupported_secret_passes(self, assessment: ScreenAssessment) -> None:
        """Do not let an empty field or transport mask prove a secret rule."""
        if not isinstance(self._scrubber, SecretScrubber):
            return
        local_renderings = self._scrubber.app_renderings()
        for verification in assessment.rule_verifications:
            if verification.status != "passed":
                continue
            statement = self._security_rules_by_id.get(verification.rule_id or "")
            declared_names = self._scrubber.declared_names_in(statement)
            if not declared_names:
                continue
            if self._has_masking_evidence(
                declared_names,
                local_renderings,
                assessment.sensitive_findings,
            ):
                continue
            verification.status = "untested"
            verification.reasoning = (
                "The rule concerns a declared value, but this screen contains no "
                "app-rendered masking evidence for it. An empty field or a Yewo "
                "transport mask does not prove the rule."
            )

    def _has_masking_evidence(
        self,
        declared_names: tuple[str, ...],
        local_renderings: dict[str, AppValueRendering],
        findings: list[SensitiveFinding],
    ) -> bool:
        """Whether local or unobscured visual evidence shows app masking."""
        for name in declared_names:
            label = self._scrubber.model_label_for_name(name)
            if label and local_renderings.get(label) is AppValueRendering.MASKED:
                return True
            if label and label in local_renderings:
                continue
            for finding in findings:
                finding_names = self._scrubber.declared_names_in(
                    f"{finding.field} {finding.quoted}"
                )
                if name in finding_names and finding.rendering is Rendering.MASKED:
                    return True
        return False

    def _audit_screen_and_normalize(self, messages: list[Message]) -> ScreenAssessment:
        """One answer, with each verdict matched to the rule it is about."""
        assessment: ScreenAssessment = self._llm.generate_response(messages, ScreenAssessment)
        self._resolve_rule_ids_from_statements(assessment)
        self._filter_unlisted_verifications(assessment)
        return assessment

    def _local_secret_exposures(self, gui_elements, screen_id) -> list[SensitiveFinding]:
        """Exact declared values OCR found before the model receives a masked image."""
        findings = []
        known: set[tuple[str, Rendering]] = set()
        for element in gui_elements or ():
            for label in self._scrubber.declared_labels_in(element.text):
                identity = (label, Rendering.VALUE)
                if identity in known:
                    continue
                known.add(identity)
                findings.append(
                    SensitiveFinding(
                        field=label,
                        rendering=Rendering.VALUE,
                        quoted=label,
                        screen_name=screen_id,
                    )
                )
        return findings

    def _resolve_rule_ids_from_statements(self, assessment: ScreenAssessment) -> None:
        """Fills the id for a verdict that quoted its rule and left the id out.

        Observed on a working model: every rule answered correctly, each one
        quoted word for word, and `rule_id` null on all of them. Nothing
        downstream reads such a verdict, because the gate matches a verdict to
        its rule by id. The scenario then reports every outcome unevidenced,
        which blames the run for an answer it was given.
        """
        for verification in assessment.rule_verifications:
            if verification.rule_id in self._audited_rule_ids:
                continue
            rule_id = self._rule_ids_by_statement.get((verification.rule or "").strip())
            if rule_id:
                verification.rule_id = rule_id

    def _every_rule_is_answered(self, assessment: ScreenAssessment) -> bool:
        """Whether the answer contains a verdict for every listed rule."""
        answered = {
            verification.rule_id
            for verification in assessment.rule_verifications
            if verification.rule_id in self._audited_rule_ids
        }
        return self._audited_rule_ids <= answered

    def _filter_unlisted_verifications(self, assessment: ScreenAssessment) -> None:
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
            if verify.rule_id is None or verify.rule_id in self._audited_rule_ids
        ]
        dropped = len(assessment.rule_verifications) - len(kept)
        if dropped:
            logger.warning("Ignored %d verdict(s) naming a rule that was not listed.", dropped)
        assessment.rule_verifications = kept

    @staticmethod
    def _copy_assessment(assessment: ScreenAssessment) -> ScreenAssessment:
        """A copy, because a caller withdraws findings from what it receives.

        The step loop drops a crash reported on the first look, then checks the
        screen again. Returning the stored object would carry that withdrawal
        into every later look at the same screen, and a crash that persists
        would stay invisible.
        """
        return assessment.model_copy(deep=True)

    def _scrub_assessment(self, assessment: ScreenAssessment) -> None:
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
