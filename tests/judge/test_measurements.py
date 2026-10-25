"""Metric contracts and deterministic measured verdicts."""

import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from fakes import device_result, walk

from yewo.agent.device.platform import Platform
from yewo.metrics import MeasurementRelation, Metric, MetricSource, MetricUnit
from yewo.testing.models import ExpectedOutcome, Measurement, UserJourney


def outcome(metric=Metric.APP_LAUNCH_TTID_MS, threshold=800) -> ExpectedOutcome:
    return ExpectedOutcome(
        statement="The app launches within its declared limit",
        scope="measured",
        metric=metric,
        threshold=threshold,
    )


def sample(metric=Metric.APP_LAUNCH_TTID_MS, value=600) -> Measurement:
    unit = MetricUnit.MILLISECONDS
    source = MetricSource.DEVICE
    if metric is Metric.MIN_TOUCH_TARGET:
        unit = MetricUnit.POINTS
        source = MetricSource.ACCESSIBILITY_AUDIT
    if metric is Metric.UNLABELLED_ACTIONABLE_COUNT:
        unit = MetricUnit.COUNT
        source = MetricSource.ACCESSIBILITY_AUDIT
    return Measurement(
        metric=metric,
        value=value,
        unit=unit,
        source=source,
        platform=Platform.IOS,
        target_kind="emulator",
        collected_at="2026-10-25T12:00:00Z",
    )


class MeasuredOutcomeContractTests(unittest.TestCase):
    def test_a_named_metric_derives_the_measured_scope(self):
        measured = ExpectedOutcome(
            statement="Launch is quick",
            metric=Metric.APP_LAUNCH_TTID_MS,
            threshold=800,
        )

        self.assertEqual(measured.scope, "measured")

    def test_a_measured_outcome_requires_a_metric(self):
        with self.assertRaisesRegex(ValueError, "must name its metric"):
            ExpectedOutcome(statement="Launch is quick", scope="measured", threshold=800)

    def test_a_measured_outcome_requires_a_threshold(self):
        with self.assertRaisesRegex(ValueError, "must name its threshold"):
            ExpectedOutcome(
                statement="Launch is quick",
                scope="measured",
                metric=Metric.APP_LAUNCH_TTID_MS,
            )

    def test_required_metrics_are_derived_without_duplicates(self):
        from yewo.testing.models import GeneratedTestCase

        case = GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Launch",
            cell="performance.happy_path.0",
            category="performance",
            scenario_type="happy_path",
            priority="P1",
            steps=walk("Open the app."),
            expected_outcomes=[outcome(), outcome(threshold=900)],
        )

        self.assertEqual(case.required_metrics, [Metric.APP_LAUNCH_TTID_MS])

    def test_jank_is_diagnostic_and_cannot_become_a_gate(self):
        with self.assertRaisesRegex(ValueError, "diagnostic telemetry"):
            outcome(Metric.JANKY_FRAME_PERCENT, threshold=5)

    def test_a_sample_must_use_the_catalogue_unit(self):
        with self.assertRaisesRegex(ValueError, "samples use ms"):
            Measurement(
                metric=Metric.APP_LAUNCH_TTID_MS,
                value=600,
                unit=MetricUnit.COUNT,
                source=MetricSource.DEVICE,
                platform=Platform.IOS,
                collected_at="2026-10-25T12:00:00Z",
            )

    def test_non_finite_samples_are_refused(self):
        with self.assertRaises(ValueError):
            sample(value=math.nan)

    def test_a_sample_from_an_ineligible_platform_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not collectable on android"):
            Measurement(
                metric=Metric.MIN_TOUCH_TARGET,
                value=44,
                unit=MetricUnit.POINTS,
                source=MetricSource.ACCESSIBILITY_AUDIT,
                platform=Platform.ANDROID,
                collected_at="2026-10-25T12:00:00Z",
            )

    def test_launch_ttid_accepts_android_and_ios_samples(self):
        for platform in (Platform.ANDROID, Platform.IOS):
            with self.subTest(platform=platform):
                measured = sample()
                measured.platform = platform

                self.assertEqual(
                    Measurement.model_validate(measured.model_dump()).platform,
                    platform,
                )


class MeasuredVerdictTests(unittest.TestCase):
    def _verdict(self, required_outcome, samples, verifications=None):
        from yewo.testing.judge import decide

        verifications = list(verifications or [])
        verdict = decide.determine_status(
            run_result=device_result(),
            stop_reason=None,
            all_verifications=verifications,
            all_issues=[],
            qa_outputs=[SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)],
            assessments=[object()],
            required_outcomes=[required_outcome],
            security_rules=[],
            measurement_samples=samples,
        )
        return verdict, verifications

    def test_lower_is_better_uses_the_worst_sample(self):
        verdict, verifications = self._verdict(outcome(threshold=800), [sample(value=700), sample(value=900)])

        self.assertEqual(verdict.status, "FAILED")
        self.assertEqual(verifications[0].status, "failed")
        self.assertIn("900 ms", verifications[0].reasoning)

    def test_lower_is_better_reports_the_worst_of_multiple_failures(self):
        verdict, verifications = self._verdict(
            outcome(threshold=800),
            [sample(value=900), sample(value=1000)],
        )

        self.assertEqual(verdict.status, "FAILED")
        self.assertIn("1000 ms", verifications[0].reasoning)

    def test_higher_is_better_uses_the_worst_sample(self):
        verdict, _ = self._verdict(
            outcome(Metric.MIN_TOUCH_TARGET, threshold=44),
            [sample(Metric.MIN_TOUCH_TARGET, 50), sample(Metric.MIN_TOUCH_TARGET, 40)],
        )

        self.assertEqual(verdict.status, "FAILED")

    def test_a_lower_bound_can_prove_the_touch_target_floor(self):
        bounded = sample(Metric.MIN_TOUCH_TARGET, 44).model_copy(
            update={"relation": MeasurementRelation.AT_LEAST}
        )

        verdict, verifications = self._verdict(
            outcome(Metric.MIN_TOUCH_TARGET, threshold=44),
            [bounded],
        )

        self.assertEqual(verdict.status, "PASSED")
        self.assertIn("at least 44 pt", verifications[0].reasoning)

    def test_a_platform_floor_cannot_prove_a_stricter_requirement(self):
        bounded = sample(Metric.MIN_TOUCH_TARGET, 44).model_copy(
            update={"relation": MeasurementRelation.AT_LEAST}
        )

        verdict, verifications = self._verdict(
            outcome(Metric.MIN_TOUCH_TARGET, threshold=48),
            [bounded],
        )

        self.assertEqual(verdict.status, "UNVERIFIED")
        self.assertEqual(verifications, [])

    def test_no_sample_produces_no_verdict_and_unverified_evidence(self):
        verdict, verifications = self._verdict(outcome(), [])

        self.assertEqual(verdict.status, "UNVERIFIED")
        self.assertEqual(verdict.cause.value, "evidence")
        self.assertEqual(verifications, [])
        self.assertIn("expected:0", verdict.reason)

    def test_recorded_model_evidence_cannot_settle_a_measured_outcome(self):
        from yewo.testing.models import RuleVerification

        recorded = RuleVerification(
            rule="The app launches within its declared limit",
            rule_id="expected:0",
            status="passed",
            reasoning="A stored verdict claimed it passed.",
        )

        verdict, verifications = self._verdict(outcome(), [], [recorded])

        self.assertEqual(verdict.status, "UNVERIFIED")
        self.assertEqual(verifications, [])


class MetricPreflightTests(unittest.TestCase):
    def _case(self, metric):
        from yewo.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="Measured",
            cell="performance.happy_path.0",
            category="performance",
            scenario_type="happy_path",
            priority="P1",
            steps=walk("Open the app."),
            expected_outcomes=[outcome(metric)],
        )

    def test_an_unsupported_metric_refuses_before_device_work(self):
        from yewo.testing.judge.capabilities import CapabilityTarget, unsupported_reason

        reason = unsupported_reason(
            self._case(Metric.MIN_TOUCH_TARGET),
            CapabilityTarget(platform=Platform.ANDROID, emulator=True),
        )

        self.assertIn("min_touch_target", reason)
        self.assertIn("android", reason)

    def test_platform_preflight_keeps_ios_only_metrics_on_ios(self):
        from yewo.testing.metric_preflight import preflight_metrics_before_device

        case = self._case(Metric.MIN_TOUCH_TARGET)

        runnable, refused = preflight_metrics_before_device([case], Platform.IOS)

        self.assertEqual(runnable, [case])
        self.assertEqual(refused, [])

    def test_platform_preflight_refuses_ios_only_metrics_on_android(self):
        from yewo.testing.metric_preflight import preflight_metrics_before_device

        case = self._case(Metric.MIN_TOUCH_TARGET)

        runnable, refused = preflight_metrics_before_device([case], Platform.ANDROID)

        self.assertEqual(runnable, [])
        self.assertEqual(refused[0].status, "UNSUPPORTED")
        self.assertIn("not collectable on android", refused[0].failure_reason)

    def test_ios_launch_without_an_accepted_collector_is_a_scenario_problem(self):
        from yewo.testing.judge.capabilities import (
            CapabilityTarget,
            unavailable_metric_reason,
        )

        reason = unavailable_metric_reason(
            self._case(Metric.APP_LAUNCH_TTID_MS),
            CapabilityTarget(platform=Platform.IOS, emulator=True, release=(17,)),
        )

        self.assertIn("No live collector", reason)
        self.assertIn("app_launch_ttid_ms", reason)

    def test_live_execution_stops_before_device_work_without_a_collector(self):
        from unittest.mock import Mock

        from yewo.testing.qa_agent import QAAgent
        from yewo.utils.secrets import SecretScrubber

        agent = QAAgent.__new__(QAAgent)
        agent._fast_llm = None
        agent._trace_store = None
        agent._device_agent = SimpleNamespace(scrubber=SecretScrubber())
        perception = SimpleNamespace(reset=Mock())
        device_control = SimpleNamespace(
            platform=Platform.IOS,
            serial=None,
            os_version=None,
        )

        report = agent.execute_scenario(
            test_case=self._case(Metric.APP_LAUNCH_TTID_MS),
            security_rules=[],
            device_control=device_control,
            perception=perception,
            session_dir=Path("unused"),
        )

        self.assertEqual(report.status, "UNVERIFIED")
        self.assertEqual(report.cause, "scenario")
        self.assertIn("No live collector", report.failure_reason)
        perception.reset.assert_not_called()

    def test_a_locked_unsupported_metric_does_not_start_a_device(self):
        from unittest.mock import Mock, patch

        from yewo.testing.lock_resolution import LockResolution
        from yewo.testing.runner import YewoTestRunner
        from yewo.utils.secrets import SecretScrubber

        case = self._case(Metric.MIN_TOUCH_TARGET)
        started = Mock()

        class Session:
            def __init__(self, *_, session_dir, **__):
                root = Path(session_dir) / "run"
                root.mkdir(parents=True)
                self.telemetry_dir = root
                self.screenshots_dir = root / "screenshots"
                self.screenshots_dir.mkdir()
                self.prompts_dir = root / "prompts"
                self.prompts_dir.mkdir()
                self.manifest_path = root / "run.json"
                self.root_log_file = root / "run.log"
                self.root_log_file.write_text("", encoding="utf-8")
                self.appium_log_path = root / "appium.log"
                self.session_id = "run"

            def start(self, **_):
                started()

            def close(self):
                pass

        class Agent:
            def __init__(self, *_, **__):
                client = SimpleNamespace(usage_callback=None)
                self._device_agent = SimpleNamespace(
                    llm=client,
                    fast_llm=client,
                    scrubber=SecretScrubber(),
                    record_prompts=lambda _store: None,
                )

        with TemporaryDirectory() as directory:
            runner = YewoTestRunner(
                device_name="Pixel_8",
                app_path="app.apk",
                assets_dir=Path(directory),
                platform=Platform.ANDROID,
                locked=True,
            )
            with (
                patch("yewo.testing.runner.DeviceSession", Session),
                patch("yewo.testing.runner.QAAgent", Agent),
                patch("yewo.testing.runner.lock_resolution.require_lock"),
                patch(
                    "yewo.testing.metric_preflight._known_locked_scenarios",
                    return_value=([case], True),
                ),
                patch.object(
                    YewoTestRunner,
                    "_suite",
                    return_value=LockResolution(scenarios=[case], reused_count=1),
                ),
                patch("yewo.testing.runner.ensure_device_ready") as ready,
                patch("yewo.testing.runner.evaluate_package_policy", return_value=[]),
            ):
                report = runner.run_journey(
                    UserJourney(id="j", name="J", description="Measure the app.")
                )

        self.assertEqual(report.scenarios[0].status, "UNSUPPORTED")
        ready.assert_not_called()
        started.assert_not_called()


class MetricGenerationTests(unittest.TestCase):
    def test_a_globally_collectable_metric_is_kept_for_target_preflight(self):
        from unittest.mock import Mock

        from yewo.testing.generator import ScenarioGenerator
        from yewo.testing.models import ScenarioDraft, UserJourney

        measured = ScenarioDraft(
            scenario_name="Measure launch",
            steps=walk("Launch the app."),
            expected_outcomes=[outcome()],
        )
        client = SimpleNamespace(generate_response=Mock(return_value=measured))
        journey = UserJourney(id="j", name="J", description="Open the app.")

        suite = ScenarioGenerator(client).generate_suite(journey)

        self.assertEqual(len(suite.scenarios), 1)
        self.assertEqual(
            suite.scenarios[0].required_metrics,
            [Metric.APP_LAUNCH_TTID_MS],
        )
        self.assertEqual(suite.uncovered, [])


class MeasurementLogTests(unittest.TestCase):
    def test_a_sample_is_written_as_jsonl(self):
        from yewo.testing.report.measurement_log import MeasurementLog

        with TemporaryDirectory() as directory:
            log = MeasurementLog(Path(directory))
            self.assertTrue(log.append(sample()))
            written = json.loads(log.path.read_text(encoding="utf-8"))

        self.assertEqual(written["metric"], "app_launch_ttid_ms")
        self.assertEqual(written["unit"], "ms")

    def test_a_failed_write_warns_and_does_not_raise(self):
        from yewo.testing.report.measurement_log import MeasurementLog

        with self.assertLogs("yewo.testing.report.measurement_log", level="WARNING"):
            wrote = MeasurementLog(Path("/dev/null/measurements")).append(sample())

        self.assertFalse(wrote)
