"""Where a provider failure goes, once the client has judged it.

Three layers between the client and the suite catch broadly: the device
agent turns an exception into a result, the generator retries, and the
transition judge answers with no verdicts. Each is correct for its own
failures and wrong for a provider's, so each one lets a provider through.

Where it lands then depends on which kind it is. A spent rate limit ends
the scenario and the suite goes on. A refused request ends the run,
because every later call carries the same request.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from fakes import refused
from litellm import exceptions
from pydantic import ValidationError

from yewo.errors import Cause, ProviderUnavailable


def device_run(failure):
    """Drives `DeviceAgent.run` to its handler, with one step that fails."""
    import tempfile
    from pathlib import Path

    from yewo.agent.device.launch import SetupMode
    from yewo.agent.run.agent import DeviceAgent

    agent = object.__new__(DeviceAgent)
    agent.replay_ledger = None
    agent.scrubber = SimpleNamespace(scrub=str)
    agent.max_steps = 1
    agent.settle_time = 0
    agent._default_fast_planner = SimpleNamespace()

    with tempfile.TemporaryDirectory() as directory, \
         mock.patch.object(DeviceAgent, "_run_step", side_effect=failure), \
         mock.patch("yewo.agent.run.agent.restart_app"), \
         mock.patch("yewo.agent.run.agent.replay_known_path"), \
         mock.patch("yewo.agent.run.outcomes.log_recovery_cost"):
        return agent.run(
            goal="Open the app.",
            device_control=SimpleNamespace(screen_height=800),
            perception=SimpleNamespace(),
            session_dir=Path(directory),
            setup=SetupMode.PRESERVE,
        )


class PropagationTests(unittest.TestCase):
    """A spent provider has to reach the runner through the real handlers.

    Three layers catch broadly between the client and the suite: the device
    agent turns an exception into a result, the generator retries, and the
    transition judge answers with no verdicts. Each one is correct for its
    own failures and wrong for this one, so each one lets this through.
    """

    def _device_run(self, failure):
        return device_run(failure)

    def test_the_device_agent_does_not_turn_a_spent_provider_into_a_result(self):
        """Read as a result it would be judged, and no state was reached."""
        with self.assertRaises(ProviderUnavailable):
            self._device_run(ProviderUnavailable("spent"))

    def test_every_other_failure_is_still_a_result_the_gate_can_read(self):
        """A device that misbehaved mid-scenario is a verdict, not an outage."""
        from yewo.errors import DeviceUnavailable

        result = self._device_run(DeviceUnavailable("the driver went away"))

        self.assertEqual(result.status, "exception")
        self.assertIs(result.cause, Cause.DEVICE)

    def _journey(self):
        from yewo.testing.models import UserJourney

        return UserJourney(id="j", name="J", description="A journey")

    def _cell(self):
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        return CoverageCell(
            category=Category.FUNCTIONAL,
            scenario_type=ScenarioType.HAPPY_PATH,
            priority=Priority.P1,
        )

    def _generator(self, failure):
        from yewo.testing.generator import ScenarioGenerator

        generator = object.__new__(ScenarioGenerator)
        generator.llm = SimpleNamespace(
            generate_response=mock.Mock(side_effect=failure)
        )
        return generator

    def test_the_generator_does_not_retry_what_the_client_already_judged(self):
        """Three generation attempts over three client attempts is nine calls."""
        from yewo.testing.generator import ScenarioGenerator

        generator = self._generator(ProviderUnavailable("spent"))

        with mock.patch.object(
            ScenarioGenerator, "_prompt_sections", return_value=("policy", "input")
        ), \
             self.assertRaises(ProviderUnavailable):
            generator._draft(self._journey(), self._cell())

        self.assertEqual(generator.llm.generate_response.call_count, 1)

    def test_a_schema_mismatch_is_still_retried_by_the_generator(self):
        """That loop is for a model that answered badly, and it stays."""
        from yewo.testing.generator import GENERATION_ATTEMPTS, ScenarioGenerator

        mismatch = ValidationError.from_exception_data("ScenarioDraft", [])
        generator = self._generator(mismatch)

        with mock.patch.object(
            ScenarioGenerator, "_prompt_sections", return_value=("policy", "input")
        ), \
             self.assertRaises(ValidationError):
            generator._draft(self._journey(), self._cell())

        self.assertEqual(
            generator.llm.generate_response.call_count, GENERATION_ATTEMPTS
        )

    def test_the_transition_judge_does_not_report_an_outage_as_thin_evidence(self):
        """Missing evidence blames the run for something the service did."""
        from yewo.testing.qa_agent import QAAgent

        agent = object.__new__(QAAgent)
        agent._fast_llm = SimpleNamespace()
        agent._device_agent = SimpleNamespace(scrubber=SimpleNamespace(scrub=str))

        with mock.patch(
            "yewo.testing.judge.transitions.settle",
            side_effect=ProviderUnavailable("spent"),
        ), self.assertRaises(ProviderUnavailable):
            agent._settled_transitions(SimpleNamespace(expected_outcomes=[]), [])

    def test_the_transition_judge_still_answers_nothing_for_its_own_failures(self):
        from yewo.testing.qa_agent import QAAgent

        agent = object.__new__(QAAgent)
        agent._fast_llm = SimpleNamespace()
        agent._device_agent = SimpleNamespace(scrubber=SimpleNamespace(scrub=str))

        with mock.patch(
            "yewo.testing.judge.transitions.settle",
            side_effect=ValueError("the screen names did not line up"),
        ):
            self.assertEqual(
                agent._settled_transitions(SimpleNamespace(expected_outcomes=[]), []), []
            )


class SuiteSurvivalTests(unittest.TestCase):
    """One scenario ends, and the suite goes on to the next.

    Before this, an exhausted provider left the runner's outermost handler,
    which aborts the run and marks every scenario after it ABORTED. The device
    is already open and already restored, and a rate limit that has passed by
    the next scenario costs nothing.
    """

    def _report(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        from fakes import walk
        from yewo.testing.models import GeneratedTestCase, ScenarioReport, UserJourney
        from yewo.testing.runner import YewoTestRunner

        def scenario(number: int) -> GeneratedTestCase:
            return GeneratedTestCase(
                test_id=f"TS_{number}",
                scenario_name=f"Scenario {number}",
                cell=f"functional.happy_path.{number}",
                category="functional",
                scenario_type="happy_path",
                priority="P1",
                steps=walk("Open the app."),
                expected_outcomes=["a banner appears"],
            )

        scenarios = [scenario(1), scenario(2)]
        asked = []

        class Session:
            def __init__(self, *_, **__):
                self.telemetry_dir = Path(tempfile.mkdtemp())
                self.screenshots_dir = self.telemetry_dir
                self.session_id = "s1"
                self.installed_by_run = False
                self.root_log_file = self.telemetry_dir / "run.log"
                self.appium_log_path = self.telemetry_dir / "appium.log"
                self.manifest_path = self.telemetry_dir / "run.json"
                self.prompts_dir = self.telemetry_dir / "prompts"

            def start(self, **_):
                return SimpleNamespace()

            def checkpoint_recording(self):
                pass

            def close(self, **_):
                pass

        class Generator:
            def __init__(self, *_, **__):
                pass

            def generate_suite(self, _journey, _cells=None):
                from yewo.testing.generator import GeneratedSuite

                return GeneratedSuite(scenarios=list(scenarios))

        class Agent:
            def __init__(self, *_, **__):
                from yewo.utils.secrets import SecretScrubber

                llm = SimpleNamespace(usage_callback=None)
                self._device_agent = SimpleNamespace(
                    llm=llm,
                    fast_llm=llm,
                    scrubber=SecretScrubber(),
                    record_prompts=lambda _store: None,
                )

            def execute_scenario(self, test_case, **_):
                asked.append(test_case.test_id)
                if test_case.test_id == "TS_1":
                    raise ProviderUnavailable(
                        "The provider did not answer after 3 attempts: "
                        "RateLimitError (HTTP 429)"
                    )
                return ScenarioReport(
                    test_id=test_case.test_id,
                    scenario_name=test_case.scenario_name,
                    cell=test_case.cell,
                    status="PASSED",
                )

        with (
            patch("yewo.testing.runner.DeviceSession", Session),
            patch("yewo.testing.runner.ScenarioGenerator", Generator),
            patch("yewo.testing.runner.QAAgent", Agent),
            patch("yewo.testing.runner.ensure_device_ready"),
            patch("yewo.testing.runner.for_grounding"),
            patch("yewo.testing.runner.is_session_responsive", return_value=True),
        ):
            report = YewoTestRunner(
                device_name="Pixel_8", app_path="app.apk"
            ).run_journey(UserJourney(id="j", name="J", description="A journey"))
        return report, asked

    def test_the_scenario_after_an_exhausted_provider_still_runs(self):
        _, asked = self._report()

        self.assertEqual(asked, ["TS_1", "TS_2"])

    def test_the_scenario_that_lost_its_provider_is_an_error_and_not_a_failure(self):
        """Nothing was learned about the app, so FAILED would be a claim."""
        report, _ = self._report()
        first = next(r for r in report.scenarios if r.test_id == "TS_1")

        self.assertEqual(first.status, "ERROR")
        self.assertEqual(first.cause, Cause.PROVIDER.value)
        self.assertIn("429", first.failure_reason)

    def test_no_scenario_is_aborted_by_one_spent_provider(self):
        report, _ = self._report()
        statuses = {r.test_id: r.status for r in report.scenarios}

        self.assertNotIn("ABORTED", statuses.values())
        self.assertEqual(statuses["TS_2"], "PASSED")


class TerminalPropagationTests(unittest.TestCase):
    """A refused request is refused again, so it stops the run rather than the step.

    An exhausted rate limit and a rejected key are both the provider's, and
    they want opposite answers. The first may clear by the next scenario. The
    second will refuse every call after it, so continuing spends a device on
    an answer nobody can change.
    """

    def test_a_rejected_key_leaves_the_device_agent_as_it_came(self):
        with self.assertRaises(exceptions.AuthenticationError):
            device_run(refused())

    def test_a_rejected_key_leaves_the_transition_judge_too(self):
        """Answering with no verdicts would read as a thin audit of a good run."""
        from yewo.testing.qa_agent import QAAgent

        agent = object.__new__(QAAgent)
        agent._fast_llm = SimpleNamespace()
        agent._device_agent = SimpleNamespace(scrubber=SimpleNamespace(scrub=str))

        with mock.patch(
            "yewo.testing.judge.transitions.settle", side_effect=refused()
        ), self.assertRaises(exceptions.AuthenticationError):
            agent._settled_transitions(SimpleNamespace(expected_outcomes=[]), [])
