"""Reports, progress files and screenshots, and what must never reach them."""

import tempfile
import unittest

from fakes import walk
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from yewo.agent.device.platform import Platform
from yewo.utils.secrets import SecretScrubber


class ArtifactSafetyTests(unittest.TestCase):
    def _scenario(self):
        from yewo.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Checkout",
            cell="functional.happy_path.0",
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=walk("Enter 'password123' in Password"),
            expected_outcomes=["Do not display '4111222233334444'"],
        )

    def test_junit_distinguishes_errors_from_assertion_failures(self):
        from yewo.testing.models import JourneyReport, ScenarioReport
        from yewo.testing.report.reports import save_junit_report

        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="FAILED",
            scenarios=[
                ScenarioReport(
                    test_id="one",
                    scenario_name="Harness",
                    cell="functional.happy_path.0",
                    status="ERROR",
                    failure_reason="connection refused",
                ),
                ScenarioReport(
                    test_id="two",
                    scenario_name="Evidence",
                    cell="functional.happy_path.0",
                    status="UNVERIFIED",
                    failure_reason="missing evidence",
                ),
            ],
            failed_count=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            save_junit_report(report, Path(directory))
            path = next(Path(directory).glob("junit_*.xml"))
            root = ET.parse(path).getroot()

            self.assertEqual(root.attrib["errors"], "1")
            self.assertEqual(root.attrib["failures"], "1")
            self.assertEqual(len(root.findall(".//error")), 1)
            self.assertEqual(len(root.findall(".//failure")), 1)

    def test_the_report_records_what_produced_it(self):
        """A run that behaves differently must be attributable to what changed."""
        from yewo.testing.models import JourneyReport, RunEnvironment
        from yewo.testing.report.reports import save_markdown_report

        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="PASSED",
            passed_count=1,
            environment=RunEnvironment(
                yewo_version="0.1.0",
                model="openrouter/some/model",
                fast_model="openrouter/some/fast",
                platform="android",
                grounding="marks",
                detector="orasul/deki-yolo@287f12d1",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = save_markdown_report(report, Path(directory))
            payload = path.read_text()

        self.assertIn("openrouter/some/model", payload)
        self.assertIn("openrouter/some/fast", payload)
        self.assertIn("orasul/deki-yolo@287f12d1", payload)
        self.assertIn("marks", payload)

    def test_a_report_without_an_environment_still_writes(self):
        from yewo.testing.models import JourneyReport
        from yewo.testing.report.reports import save_markdown_report

        report = JourneyReport(
            journey_id="j", journey_name="Journey", description="d", status="PASSED"
        )

        with tempfile.TemporaryDirectory() as directory:
            payload = save_markdown_report(report, Path(directory)).read_text()

        self.assertIn("Journey", payload)
        self.assertNotIn("**Detector**", payload)

    def test_the_runner_records_the_pinned_detector(self):
        from yewo.agent.perception.detector import YOLO_REVISION
        from yewo.testing.runner import YewoTestRunner

        runner = YewoTestRunner(
            device_name="emulator-5554",
            app_path="app.apk",
            platform=Platform.ANDROID,
            model="openrouter/some/model",
            grounding="marks",
        )

        environment = runner._environment()

        self.assertEqual(environment.platform, "android")
        self.assertIn(YOLO_REVISION, environment.detector)

    def test_report_write_failure_makes_the_run_fail(self):
        from yewo.testing.models import JourneyReport
        from yewo.testing.report import infrastructure

        report = JourneyReport(
            journey_id="j",
            journey_name="Journey",
            description="d",
            status="PASSED",
            passed_count=1,
        )

        infrastructure.write_failure(
            report, "junit_report", OSError("disk full")
        )

        self.assertEqual(report.status, "FAILED")
        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.scenarios[-1].status, "ERROR")

    def _element(self, text, box=(2, 3, 8, 9)):
        return SimpleNamespace(
            text=text,
            bounding_box=SimpleNamespace(
                x_min=box[0], y_min=box[1], x_max=box[2], y_max=box[3]
            ),
        )

    def test_declared_secret_regions_are_covered_before_saving(self):
        from PIL import Image

        from yewo.agent.artifacts.redaction import RedactedImages
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"card_number": "4111222233334444"})
        image = Image.new("RGB", (20, 20), "white")

        safe = RedactedImages(scrubber).redact_artifact_image(
            image, [self._element("4111 2222 3333 4444")]
        )
        self.assertEqual(safe.getpixel((4, 5)), (255, 255, 255))

        safe = RedactedImages(scrubber).redact_artifact_image(
            image, [self._element("4111222233334444")]
        )
        self.assertEqual(safe.getpixel((4, 5)), (0, 0, 0))
        self.assertEqual(image.getpixel((4, 5)), (255, 255, 255))

    def test_ordinary_labels_are_never_covered(self):
        from PIL import Image

        from yewo.agent.artifacts.redaction import RedactedImages
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "secret-password123"})
        image = Image.new("RGB", (20, 20), "white")

        safe = RedactedImages(scrubber).redact_artifact_image(
            image, [self._element("Continue")]
        )

        self.assertEqual(safe.getpixel((4, 5)), (255, 255, 255))

class ScanDumpSafetyTests(unittest.TestCase):
    """A scan dump is a file, so detected text in it outlives the run."""

    def _element(self, text):
        from yewo.agent.perception.elements import BoundingBox, GUIElement

        return GUIElement(
            element="Text",
            type="text",
            confidence=0.9,
            text=text,
            bounding_box=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
        )

    def test_declared_secrets_are_masked_in_the_dump(self):
        from yewo.agent.artifacts.scan import element_to_dict
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "secret-password123"})

        dumped = element_to_dict(self._element("secret-password123"), 0, scrubber)

        self.assertEqual(dumped["text"], "{{password}}")

    def test_ordinary_detected_text_survives(self):
        from yewo.agent.artifacts.scan import element_to_dict
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "secret-password123"})

        dumped = element_to_dict(self._element("Continue"), 0, scrubber)

        self.assertEqual(dumped["text"], "Continue")


class SecretScrubbingTests(unittest.TestCase):
    def _scrubber(self):
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"email": "test@example.com", "password": "password123"})
        return scrubber

    def test_values_are_replaced_by_the_name_that_declared_them(self):
        scrubbed = self._scrubber().scrub("Type 'password123' into the Password field")

        self.assertNotIn("password123", scrubbed)
        self.assertIn("{{password}}", scrubbed)
        self.assertIn("Password field", scrubbed)

    def test_a_value_the_journey_did_not_declare_is_written_as_it_is(self):
        """The `env` block is the whole of what Yewo hides.

        Masking a value nobody declared made reports unreadable: an assertion
        about an ordinary title came back as "the month view shows
        '[REDACTED]'", and a reader could not tell what the test checked.
        """
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})

        self.assertEqual(
            scrubber.scrub("The month view shows 'Dentist appointment' today."),
            "The month view shows 'Dentist appointment' today.",
        )

    def test_text_shaped_like_a_credential_is_still_written_as_it_is(self):
        """Patterns guessed at secrets, and a guess is not a declaration."""
        from yewo.utils.secrets import SecretScrubber

        text = "Password: hunter2 and card 4111 1111 1111 1111"

        self.assertEqual(SecretScrubber().scrub(text), text)

    def test_ordinary_text_is_untouched(self):
        self.assertEqual(
            self._scrubber().scrub("Tap the Login button"), "Tap the Login button"
        )


class PlannerViewArtifactTests(unittest.TestCase):
    """The numbered screenshot is the only image the planner is shown.

    Saving it needs a snapshot whose numbering is done, which is why this runs
    after planning rather than beside the raw screen and detector overlay.
    """

    def _artifacts(self, saved):
        from yewo.agent.artifacts.steps import StepArtifacts

        media = SimpleNamespace(
            save_redacted_bytes=lambda data, elements, path: saved.append((path, data))
        )
        return StepArtifacts.__new__(StepArtifacts).__class__(media, scrubber=None)

    def _snapshot(self, planner_bytes, marks=(1,)):
        return SimpleNamespace(
            planner_img_bytes=planner_bytes,
            raw_img_bytes=b"raw-png",
            marks=list(marks),
            gui_elements=[],
            timestamp="20260101_000000",
        )

    def test_the_numbered_screenshot_is_written(self):
        import tempfile
        from pathlib import Path

        saved = []
        with tempfile.TemporaryDirectory() as directory:
            self._artifacts(saved).save_planner_view(
                Path(directory), 0, self._snapshot(b"numbered-png")
            )

        self.assertEqual(len(saved), 1)
        path, data = saved[0]
        self.assertTrue(path.name.endswith("_marks.png"))
        self.assertEqual(data, b"numbered-png")

    def test_nothing_is_written_when_the_screen_was_never_numbered(self):
        """Coordinates mode numbers nothing, so there is no planner view."""
        import tempfile
        from pathlib import Path

        saved = []
        with tempfile.TemporaryDirectory() as directory:
            self._artifacts(saved).save_planner_view(
                Path(directory), 0, self._snapshot(b"raw-png", marks=())
            )

        self.assertEqual(saved, [])


class StepFileNameTests(unittest.TestCase):
    """One step's files sit together, and the steps sit in order."""

    def _names(self, steps):
        from yewo.agent.artifacts.steps import StepArtifacts

        written = []

        class Recorder:
            def save_redacted_pair(self, annotated, raw, elements, a, b):
                written.extend([a.name, b.name])

            def save_redacted_bytes(self, data, elements, path):
                written.append(path.name)

            def save_redacted_action_annotation(self, annotated, elements, action, path):
                written.append(path.name)
                return True

        artifacts = StepArtifacts(Recorder(), SecretScrubber())
        for step in steps:
            snapshot = SimpleNamespace(
                timestamp="20261017_041552", annotated_img=None, raw_img_bytes=b"",
                gui_elements=[], scan_info=None, marks=[1], planner_img_bytes=b"x",
                prev_phash=None,
            )
            with tempfile.TemporaryDirectory() as directory:
                artifacts.save_step(Path(directory), step, snapshot)
                artifacts.save_planner_view(Path(directory), step, snapshot)
                artifacts.save_action(Path(directory), step, snapshot, object())
        return written

    def test_every_file_of_one_step_shares_its_prefix(self):
        names = self._names([0])

        self.assertTrue(
            all(name.startswith("step001_20261017_041552") for name in names), names
        )

    def test_an_action_with_nothing_to_draw_names_no_file(self):
        """A wait and a key event have no point on screen to annotate.

        Naming a file that was never written puts a broken image in every
        report that lists what a step left.
        """
        from yewo.agent.artifacts.steps import StepArtifacts

        class DrewNothing:
            def save_redacted_action_annotation(self, annotated, elements, action, path):
                return False

        artifacts = StepArtifacts(DrewNothing(), SecretScrubber())
        snapshot = SimpleNamespace(
            timestamp="20261017_041552", annotated_img=None, gui_elements=[]
        )
        with tempfile.TemporaryDirectory() as directory:
            named = artifacts.save_action(Path(directory), 0, snapshot, object())

        self.assertIsNone(named)

    def test_an_action_that_was_drawn_names_its_file(self):
        from yewo.agent.artifacts.steps import StepArtifacts

        class DrewIt:
            def save_redacted_action_annotation(self, annotated, elements, action, path):
                return True

        artifacts = StepArtifacts(DrewIt(), SecretScrubber())
        snapshot = SimpleNamespace(
            timestamp="20261017_041552", annotated_img=None, gui_elements=[]
        )
        with tempfile.TemporaryDirectory() as directory:
            named = artifacts.save_action(Path(directory), 0, snapshot, object())

        self.assertEqual(named, "step001_20261017_041552_action.png")

    def test_a_step_never_records_a_file_that_is_not_there(self):
        """The record is what the page reads, so the filter belongs in it too."""
        from yewo.agent.run.record import StepLog

        log = StepLog()
        log.open(1)
        log.files({"action": None, "raw": "step001_raw.png"})

        self.assertIsNotNone(log.current)
        self.assertEqual(log.current.screenshots, {"raw": "step001_raw.png"})

    def test_a_listing_puts_step_two_before_step_ten(self):
        names = self._names([1, 9])

        self.assertLess(
            sorted(names)[0].split("_")[0], sorted(names)[-1].split("_")[0]
        )
        self.assertIn("step002_20261017_041552.png", names)
        self.assertIn("step010_20261017_041552.png", names)
