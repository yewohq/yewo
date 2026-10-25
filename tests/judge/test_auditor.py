"""The audit is only worth its call if it could have disagreed."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock


def _assessment(**overrides):
    from yewo.testing.judge.qa_auditor import ScreenAssessment

    values = {"screen_name": "Login Screen"}
    values.update(overrides)
    return ScreenAssessment(**values)


class AuditorInputTests(unittest.TestCase):
    def _auditor(self, llm):
        from yewo.testing.judge.qa_auditor import QAAuditor

        return QAAuditor(llm, ["Order confirmed"], ["Card masked"])

    def _llm(self):
        return SimpleNamespace(generate_response=Mock(return_value=_assessment()))

    def _sent(self, llm):
        (messages, _model), _ = llm.generate_response.call_args
        return [message.to_dict() for message in messages]

    def _sent_text(self, llm):
        text = []
        for message in self._sent(llm):
            content = message["content"]
            if isinstance(content, str):
                text.append(content)
            else:
                text.extend(part.get("text", "") for part in content)
        return " ".join(text)

    def test_the_screenshot_is_sent(self):
        llm = self._llm()
        self._auditor(llm).audit(b"png-bytes", screen_id="abc")

        kinds = [part["type"] for part in self._sent(llm)[1]["content"]]
        self.assertIn("image_url", kinds)

    def test_the_rules_are_sent(self):
        llm = self._llm()
        self._auditor(llm).audit(b"png", screen_id="abc")

        self.assertIn("[security:0] Card masked", self._sent_text(llm))

    def test_reusable_rules_precede_the_changing_screenshot(self):
        llm = self._llm()
        self._auditor(llm).audit(b"png", screen_id="abc")

        messages = self._sent(llm)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("[security:0] Card masked", messages[0]["content"])
        self.assertIn("Judge only the rules listed above", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"][-1]["type"], "image_url")

    def test_the_plan_is_not_sent(self):
        """A verdict from something that knows the intent is not independent."""
        llm = self._llm()
        self._auditor(llm).audit(b"png", screen_id="abc")

        text = self._sent_text(llm).lower()
        for leak in (
            "tap the place order button",
            "typed the delivery address",
            "next, submit the order",
        ):
            self.assertNotIn(leak, text, f"the auditor was told about {leak!r}")


class ScreenCachingTests(unittest.TestCase):
    def _auditor(self, llm):
        from yewo.testing.judge.qa_auditor import QAAuditor

        return QAAuditor(llm, [], [])

    def test_the_same_screen_is_audited_once(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        auditor = self._auditor(llm)

        auditor.audit(b"png", screen_id="same")
        auditor.audit(b"png", screen_id="same")

        self.assertEqual(llm.generate_response.call_count, 1)

    def test_a_new_screen_is_audited_again(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        auditor = self._auditor(llm)

        auditor.audit(b"png", screen_id="one")
        auditor.audit(b"png", screen_id="two")

        self.assertEqual(llm.generate_response.call_count, 2)

    def test_an_unidentified_screen_is_never_cached(self):
        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        auditor = self._auditor(llm)

        auditor.audit(b"png", screen_id=None)
        auditor.audit(b"png", screen_id=None)

        self.assertEqual(llm.generate_response.call_count, 2)


class AuditorPromptTests(unittest.TestCase):
    def _prompt(self):
        from unittest.mock import Mock

        from yewo.testing.judge.qa_auditor import QAAuditor

        llm = SimpleNamespace(generate_response=Mock(return_value=_assessment()))
        QAAuditor(llm, ["Order is confirmed"], ["Card is masked"]).audit(b"png", "s")
        messages = llm.generate_response.call_args[0][0]
        return "".join(
            message.to_dict()["content"]
            for message in messages
            if isinstance(message.to_dict()["content"], str)
        )

    def test_a_rule_is_not_passed_because_its_subject_is_absent(self):
        self.assertIn("Never turn an absent subject into a pass or failure", self._prompt())

    def test_a_rule_is_not_failed_on_what_a_screen_might_show_later(self):
        """It failed rules on hypotheticals: a label that could expose a value."""
        prompt = self._prompt()

        self.assertIn("visible evidence contradicts the rule now", prompt)
        self.assertIn("Do not infer behavior", prompt)

    def test_the_auditor_is_never_asked_whether_the_plan_finished(self):
        """It is told it does not know the plan, so it cannot answer that."""
        prompt = self._prompt()

        self.assertIn("You do not know the plan", prompt)
        self.assertNotIn("goal_satisfied", prompt)

class AssessmentScrubbingTests(unittest.TestCase):
    def test_declared_secrets_are_masked_in_what_it_wrote(self):
        from yewo.testing.judge.qa_auditor import QAAuditor
        from yewo.testing.models import QAIssue, RuleVerification
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        written = _assessment(
            screen_name="Screen showing hunter2",
            rule_verifications=[
                RuleVerification(
                    rule="Password masked",
                    rule_id="security:0",
                    status="failed",
                    reasoning="hunter2 was visible",
                )
            ],
            detected_issues=[
                QAIssue(
                    issue_type="sensitive_data_exposure",
                    severity="critical",
                    description="hunter2 shown in plain text",
                )
            ],
        )
        llm = SimpleNamespace(generate_response=Mock(return_value=written))

        assessed = QAAuditor(llm, [], ["Password masked"], scrubber=scrubber).audit(b"p", "s")

        self.assertNotIn("hunter2", assessed.screen_name)
        self.assertNotIn("hunter2", assessed.rule_verifications[0].reasoning)
        self.assertNotIn("hunter2", assessed.detected_issues[0].description)
        self.assertIn("{{password}}", assessed.screen_name)


class CacheAliasingTests(unittest.TestCase):
    """Callers withdraw findings from what they are given, so give them a copy."""

    def _auditor(self):
        from yewo.testing.models import QAIssue
        from yewo.testing.judge.qa_auditor import QAAuditor

        crash = QAIssue(issue_type="crash_screen", severity="critical", description="Blank screen")
        llm = SimpleNamespace(
            generate_response=Mock(return_value=_assessment(detected_issues=[crash]))
        )
        return QAAuditor(llm, [], []), llm

    def test_withdrawing_a_finding_does_not_erase_it_from_the_cache(self):
        """A crash dropped on the first look must still be there on the second."""
        auditor, _ = self._auditor()

        first = auditor.audit(b"png", screen_id="same")
        first.detected_issues = []
        second = auditor.audit(b"png", screen_id="same")

        self.assertEqual(len(second.detected_issues), 1)

    def test_the_screen_is_still_only_audited_once(self):
        auditor, llm = self._auditor()

        auditor.audit(b"png", screen_id="same")
        auditor.audit(b"png", screen_id="same")

        self.assertEqual(llm.generate_response.call_count, 1)


class ScreenWatchTests(unittest.TestCase):
    """Every distinct post-action state is audited before the next action."""

    def _watch(self):
        from yewo.testing.judge.qa_auditor import ScreenAuditSession

        looked = []

        class Recording:
            def audit(self, screenshot, screen_id, gui_elements=()):
                looked.append(screen_id)
                return _assessment()

        return ScreenAuditSession(Recording()), looked

    def _snapshot(self, phash, reason):
        return SimpleNamespace(
            raw_img_bytes=b"png",
            current_phash=phash,
            scan_info=SimpleNamespace(reason=reason),
            gui_elements=[],
        )

    def test_a_layout_shift_counts_as_arriving(self):
        """It means under half the elements still match: a crash or a modal.

        Treating it as the same screen leaves that unaudited until departure,
        and nothing ever departs a crash.
        """
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))

        fresh = watch.audit_snapshot(self._snapshot("B", "layout_shift"))

        self.assertTrue(fresh)
        self.assertIn("B", looked)

    def test_arriving_on_a_screen_is_audited(self):
        """A crash or a blank screen has to be caught while it can be acted on."""
        watch, looked = self._watch()

        fresh = watch.audit_snapshot(self._snapshot("A", "first_pass"))

        self.assertEqual(len(fresh), 1)
        self.assertEqual(looked, ["A"])

    def test_each_changed_form_state_is_audited(self):
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))

        self.assertTrue(watch.audit_snapshot(self._snapshot("A2", "stable")))
        self.assertTrue(watch.audit_snapshot(self._snapshot("A3", "stable")))
        self.assertEqual(looked, ["A", "A2", "A3"])

    def test_navigation_does_not_repeat_the_state_already_audited(self):
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))
        watch.audit_snapshot(self._snapshot("A-filled", "stable"))

        watch.audit_snapshot(self._snapshot("B", "navigation"))

        self.assertEqual(looked, ["A", "A-filled", "B"])

    def test_a_screen_left_as_it_was_found_is_reported_once(self):
        """The cache spares the call, but a second answer double-counts the rules."""
        watch, looked = self._watch()
        first = watch.audit_snapshot(self._snapshot("A", "first_pass"))

        leaving = watch.audit_snapshot(self._snapshot("B", "navigation"))

        self.assertEqual(len(first), 1)
        self.assertEqual(len(leaving), 1, "only the arrival on B")
        self.assertEqual(looked, ["A", "B"])

    def test_the_last_screen_was_read_when_it_was_observed(self):
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))
        watch.audit_snapshot(self._snapshot("A-filled", "stable"))

        closing = watch.finalize_audit()

        self.assertEqual(closing, [])
        self.assertEqual(looked, ["A", "A-filled"])

    def test_closing_twice_reads_nothing_again(self):
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))

        self.assertEqual(watch.finalize_audit(), [], "the screen was not changed since arrival")
        self.assertEqual(watch.finalize_audit(), [])
        self.assertEqual(looked, ["A"])

    def test_a_snapshot_with_no_scan_is_treated_as_arrival(self):
        """Not knowing why the screen changed is no reason to skip looking."""
        watch, looked = self._watch()

        fresh = watch.audit_snapshot(
            SimpleNamespace(
                raw_img_bytes=b"png",
                current_phash="A",
                scan_info=None,
                gui_elements=[],
            )
        )

        self.assertEqual(len(fresh), 1)

    def test_navigation_reports_only_the_new_arrival(self):
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))
        watch.audit_snapshot(self._snapshot("A-filled", "stable"))

        leaving = watch.audit_snapshot(self._snapshot("B", "navigation"))

        self.assertEqual(len(leaving), 1)
        self.assertEqual(looked, ["A", "A-filled", "B"])

    def test_returning_to_a_screen_is_judged_again(self):
        """Deduping across the run leaves a rule looking untested.

        A scenario that comes back to a screen is asking about it again. An
        answer withheld because the same pixels were judged earlier never
        reaches the report, and the outcome it evidenced goes missing.
        """
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))
        watch.audit_snapshot(self._snapshot("B", "navigation"))

        returning = watch.audit_snapshot(self._snapshot("A", "navigation"))

        self.assertTrue(returning)
        self.assertEqual(looked, ["A", "B", "A"])

    def test_a_withdrawn_verdict_gets_the_second_look_it_was_promised(self):
        """A crash dialog reports stable next step, so nothing would re-read it.

        The first look withdraws a crash in case the app is still drawing. If
        the second look never comes, the crash is not deferred — it is lost.
        """
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))

        watch.request_repeat_audit()
        fresh = watch.audit_snapshot(self._snapshot("A", "stable"))

        self.assertEqual(len(fresh), 1)
        self.assertEqual(looked, ["A", "A"])

    def test_the_second_look_is_only_taken_once(self):
        watch, looked = self._watch()
        watch.audit_snapshot(self._snapshot("A", "first_pass"))
        watch.request_repeat_audit()
        watch.audit_snapshot(self._snapshot("A", "stable"))

        self.assertEqual(watch.audit_snapshot(self._snapshot("A", "stable")), [])


class ScanVocabularyTests(unittest.TestCase):
    """The audit fires on a reason, so the producers have to speak it.

    A mode naming its passes something the trigger does not know is silently
    never audited, and every outcome then reports as unevidenced.
    """

    def _perception(self):
        from yewo.agent.perception.blind import BlindPerception

        return BlindPerception(SimpleNamespace())

    def _frame(self, perception, image):
        import io

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        perception.device_session = SimpleNamespace(
            device_control=SimpleNamespace(get_screenshot=lambda: buffer.getvalue())
        )
        perception.perceive()
        return perception.last_scan_info.reason

    def _image(self, seed):
        from PIL import Image

        image = Image.new("RGB", (120, 240), "white")
        for y in range(0, 240, 8):
            for x in range(0, 120, 8):
                shade = (x * seed + y * (seed + 3)) % 256
                image.putpixel((x, y), (shade, shade, shade))
        return image

    def test_every_reason_describe_mode_reports_is_one_the_trigger_knows(self):
        from yewo.agent.perception import elements

        known = {
            elements.FIRST_PASS,
            elements.NAVIGATION,
            elements.LAYOUT_SHIFT,
            elements.OCR_REFRESH,
            elements.STABLE,
        }
        perception = self._perception()

        reasons = {self._frame(perception, self._image(seed)) for seed in (1, 1, 29, 29)}

        self.assertTrue(reasons <= known, f"unknown reasons: {reasons - known}")

    def test_the_first_look_is_an_arrival(self):
        from yewo.agent.perception import elements

        reason = self._frame(self._perception(), self._image(1))

        self.assertEqual(reason, elements.FIRST_PASS)
        self.assertIn(reason, elements.ARRIVALS)

    def test_the_same_screen_twice_is_not_an_arrival(self):
        from yewo.agent.perception import elements

        perception = self._perception()
        self._frame(perception, self._image(1))

        reason = self._frame(perception, self._image(1))

        self.assertEqual(reason, elements.STABLE)
        self.assertNotIn(reason, elements.ARRIVALS)

    def test_a_distance_past_the_threshold_is_an_arrival(self):
        from yewo.agent.perception import elements

        self.assertEqual(elements.reason_for(elements.NAVIGATION_THRESHOLD), elements.NAVIGATION)
        self.assertIn(elements.NAVIGATION, elements.ARRIVALS)
        self.assertEqual(elements.reason_for(elements.NAVIGATION_THRESHOLD - 1), elements.STABLE)

    def test_each_describe_mode_arrival_is_audited(self):
        """One audit per screen arrived at, which is what marks mode gets."""
        from yewo.agent.perception import elements
        from yewo.testing.judge.qa_auditor import ScreenAuditSession

        audited = []
        watch = ScreenAuditSession(
            SimpleNamespace(audit=lambda png, sid, found: audited.append(sid) or _assessment())
        )

        watch.audit_snapshot(
            SimpleNamespace(
                raw_img_bytes=b"png",
                current_phash="A",
                scan_info=SimpleNamespace(reason=elements.FIRST_PASS),
                gui_elements=[],
            )
        )
        watch.audit_snapshot(
            SimpleNamespace(
                raw_img_bytes=b"png",
                current_phash="B",
                scan_info=SimpleNamespace(reason=elements.NAVIGATION),
                gui_elements=[],
            )
        )

        self.assertEqual(audited, ["A", "B"])


class AnsweredRulesOnlyTests(unittest.TestCase):
    """The auditor may answer the rules it was shown, and no others.

    A verdict naming an unlisted rule would settle an outcome nothing looked
    at, or fail the scenario on a security rule never put to it.
    """

    def _audited(self, *verdicts, expected=None, security=None):
        from yewo.testing.models import RuleVerification
        from yewo.testing.judge.qa_auditor import QAAuditor

        answer = _assessment(
            rule_verifications=[
                RuleVerification(rule=rid or "a rule", rule_id=rid, status=status, reasoning="r")
                for rid, status in verdicts
            ]
        )
        llm = SimpleNamespace(generate_response=Mock(return_value=answer))
        auditor = QAAuditor(
            llm,
            expected if expected is not None else ["Order confirmed"],
            security if security is not None else ["Card masked"],
        )
        return auditor.audit(b"png", screen_id="abc").rule_verifications

    def test_the_rules_it_was_shown_are_kept(self):
        kept = self._audited(("expected:0", "passed"), ("security:0", "passed"))

        self.assertEqual([v.rule_id for v in kept], ["expected:0", "security:0"])

    def test_an_outcome_it_was_not_shown_is_dropped(self):
        """A transition outcome is blanked before the prompt, so it is not its to answer."""
        from yewo.testing.models import AssertionScope, ExpectedOutcome

        kept = self._audited(
            ("expected:0", "passed"),
            expected=[
                ExpectedOutcome(statement="Login reaches Home", scope=AssertionScope.TRANSITION)
            ],
        )

        self.assertEqual(kept, [])

    def test_a_rule_it_was_shown_is_kept_however_the_id_was_written(self):
        """A bracketed id is the same rule, so dropping it loses real evidence."""
        kept = self._audited(("[expected:0]", "passed"), ("[security:0]", "passed"))

        self.assertEqual([v.rule_id for v in kept], ["expected:0", "security:0"])

    def test_a_security_rule_that_does_not_exist_is_dropped(self):
        kept = self._audited(("security:4", "failed"), security=["Card masked"])

        self.assertEqual(kept, [])

    def test_a_rule_it_raised_itself_is_kept(self):
        """Dynamic assertions carry no id, and settle nothing either way."""
        kept = self._audited((None, "failed"))

        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0].rule_id)


class BareIndexIsAmbiguousTests(unittest.TestCase):
    """The auditor is shown both kinds of rule, so a bare index names neither."""

    def test_a_bare_index_is_not_read_as_an_expected_outcome(self):
        from yewo.testing.models import RuleVerification
        from yewo.testing.judge.qa_auditor import QAAuditor

        answer = _assessment(
            rule_verifications=[RuleVerification(rule="0", status="passed", reasoning="r")]
        )
        llm = SimpleNamespace(generate_response=Mock(return_value=answer))
        auditor = QAAuditor(llm, ["Order confirmed"], ["Card masked"])

        kept = auditor.audit(b"png", screen_id="abc").rule_verifications

        self.assertEqual([v.rule_id for v in kept], [None])
