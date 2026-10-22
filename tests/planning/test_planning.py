"""What Yewo asks the model, and what it records about the answer."""

import unittest

from fakes import qa_prompt, walk

from types import SimpleNamespace
from unittest.mock import Mock
from yewo.agent.run.agent import (
    AgentHistory,
    DeviceAgent,
    RollingLoopDetector,
    RunContext,
)


SENTINEL_OUTCOME = "the sign-in screen remains visible"
SENTINEL_SECURITY = "the card number is masked"


class ProviderNormalizationTests(unittest.TestCase):
    """Schemas are advisory, so equivalent shapes have to be accepted."""

    def test_the_planner_accepts_a_coordinate_pair(self):
        from yewo.testing.qa_planner import QAStepResponse

        response = QAStepResponse.model_validate(
            {
                "current_screen": "Login",
                "observation": "Login form",
                "reasoning": "Focus email",
                "action": {"type": "tap", "coordinates": [10, 20], "reason": "focus"},
            }
        )

        self.assertEqual(response.action.coordinates.x, 10)

    def test_the_auditor_accepts_a_compact_rule_mapping(self):
        from yewo.testing.judge.qa_auditor import ScreenAssessment
        from yewo.testing.models import IssueKind

        assessment = ScreenAssessment.model_validate(
            {
                "screen_name": "Login",
                "rule_verifications": {"Login is visible": "passed"},
                "detected_issues": [{"issue": "Debug banner", "severity": "low"}],
            }
        )

        self.assertEqual(assessment.rule_verifications[0].status, "passed")
        self.assertEqual(assessment.detected_issues[0].issue_type, IssueKind.OTHER)
        self.assertEqual(assessment.detected_issues[0].description, "Debug banner")


class PlannerAgreementTests(unittest.TestCase):
    """QAPlanner subclasses DevicePlanner. Their action rules drifted apart once."""

    def _qa_prompt(self):
        return qa_prompt()

    def _device_prompt(self):
        from yewo.agent.plan.planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        return planner._build_prompt(
            goal_lines=["Sign in"],
            gui_elements=None,
            history="",
            trail=None,
            drift_warning=None,
            loop_nudge=None,
        )

    def test_both_planners_carry_the_same_action_rules(self):
        from yewo.agent.plan.planner import DevicePlanner

        planner = DevicePlanner.__new__(DevicePlanner)
        shared = planner._action_vocabulary() + planner._action_constraints()

        self.assertIn(shared, self._device_prompt())
        self.assertIn(shared, self._qa_prompt())

    def test_neither_planner_asks_for_a_focusing_tap_before_typing(self):
        for prompt in (self._device_prompt(), self._qa_prompt()):
            self.assertIn("do not spend a separate", prompt)
            self.assertNotIn("Step A: tap", prompt)


class RunnerWiringTests(unittest.TestCase):
    """A name used only inside `finally` is never exercised by importing the module."""

    def test_every_name_the_runner_calls_is_bound(self):
        import yewo.testing.runner as runner

        for name in ("unevidenced_security_rules", "aggregate_journey_status", "QAAgent"):
            self.assertTrue(hasattr(runner, name), f"{name} is not importable")


class VisionWeightsTests(unittest.TestCase):
    """The weights decide what the agent can see, so which weights must be fixed."""

    def test_the_detector_is_pinned_to_a_commit(self):
        from yewo.agent.perception.detector import YOLO_REVISION

        self.assertRegex(YOLO_REVISION, r"^[0-9a-f]{40}$")

    def test_setup_prefetches_exactly_what_a_run_loads(self):
        """Otherwise setup warms one set of weights and the run downloads another."""
        from yewo.agent.perception.detector import (
            YOLO_FILENAME,
            YOLO_REPO_ID,
            YOLO_REVISION,
        )
        from yewo.setup.steps import PREFETCH_SCRIPT

        for value in (YOLO_REPO_ID, YOLO_FILENAME, YOLO_REVISION):
            self.assertIn(repr(value), PREFETCH_SCRIPT)

    def test_the_reported_box_keeps_its_model_facing_meaning(self):
        """The model answers in this schema, so its description is prompt text."""
        from yewo.testing.models import QAIssue

        description = QAIssue.model_json_schema()["$defs"]["BoundingBox"]["description"]

        self.assertEqual(
            description, "Represents coordinates highlighting a visual issue."
        )


def _element(x_min, y_min, x_max, y_max, text=""):
    return SimpleNamespace(
        element="View",
        text=text,
        bounding_box=SimpleNamespace(
            x_min=x_min,
            y_min=y_min,
            x_max=x_max,
            y_max=y_max,
        ),
    )


class _RecordingPlanner:
    def plan_step(self, **kwargs):
        self.kwargs = kwargs
        return "planned"


class KeyboardPlanningTests(unittest.TestCase):
    def test_keyboard_open_uses_current_reflowed_elements(self):
        agent = DeviceAgent.__new__(DeviceAgent)
        planner = _RecordingPlanner()
        current_field = _element(60, 816, 1014, 966, "Email Address")
        snapshot = SimpleNamespace(
            gui_elements=[current_field],
            marks=[],
            raw_img_bytes=b"image",
            planner_img_bytes=b"image",
        )
        context = RunContext(
            history=AgentHistory(),
            loop_detector=RollingLoopDetector(),
            keyboard_open=True,
            screen_height=2400,
        )

        result = agent._plan_step(
            planner=planner,
            goal="Enter an email",
            snapshot=snapshot,
            ctx=context,
            session_trail=None,
            loop_nudge=None,
        )

        self.assertEqual(result, "planned")
        self.assertIs(planner.kwargs["gui_elements"][0], current_field)
        box = planner.kwargs["gui_elements"][0].bounding_box
        self.assertEqual(((box.x_min + box.x_max) // 2, (box.y_min + box.y_max) // 2), (537, 891))


class ElementFormattingTests(unittest.TestCase):
    """The restaurant list that caused a one-row mis-tap in a real run."""

    def _planner(self):
        from yewo.agent.plan.planner import DevicePlanner

        return DevicePlanner.__new__(DevicePlanner)

    def _row(self, name, cuisine, y):
        return [
            _element(200, y - 30, 520, y + 2, name),
            _element(200, y + 28, 560, y + 60, cuisine),
            _element(820, y, 910, y + 60, "Menu"),
        ]

    def test_repeated_labels_carry_their_row(self):
        elements = self._row("Sweet Treats", "Desserts 0.5 km", 576) + self._row(
            "Sushi Zen", "Japanese 0.8 km", 805
        )

        lines = self._planner()._format_elements(elements).splitlines()
        menus = [line for line in lines if "'Menu'" in line]

        self.assertEqual(len(menus), 2)
        self.assertIn("Sweet Treats", menus[0])
        self.assertIn("Sushi Zen", menus[1])

    def test_unique_labels_are_left_alone(self):
        elements = [_element(0, 0, 100, 50, "Login")]

        self.assertNotIn("in row with", self._planner()._format_elements(elements))

    def test_duplicate_detections_collapse(self):
        elements = [
            _element(820, 805, 910, 865, "Menu"),
            _element(821, 806, 911, 866, "Menu"),
        ]

        self.assertEqual(len(self._planner()._format_elements(elements).splitlines()), 1)

    def test_elements_read_top_to_bottom(self):
        elements = [
            _element(0, 900, 100, 950, "bottom"),
            _element(0, 100, 100, 150, "top"),
            _element(500, 100, 600, 150, "top right"),
        ]

        lines = self._planner()._format_elements(elements).splitlines()

        self.assertIn("top'", lines[0])
        self.assertIn("top right", lines[1])
        self.assertIn("bottom", lines[2])


class StepBudgetTests(unittest.TestCase):
    def _agent(self, ceiling):
        from yewo.testing.qa_agent import QAAgent

        agent = QAAgent.__new__(QAAgent)
        agent._device_agent = SimpleNamespace(max_steps=ceiling)
        return agent

    def _case(self, instruction_count):
        return SimpleNamespace(
            steps_instructions=["Tap the Next button."] * instruction_count
        )

    def test_short_scenarios_get_the_floor(self):
        self.assertEqual(self._agent(40)._step_budget(self._case(5)), 15)

    def test_long_scenarios_scale_with_their_instructions(self):
        """A long walk needs more steps than its instruction count, not fewer."""
        self.assertEqual(self._agent(40)._step_budget(self._case(19)), 38)

    def test_the_configured_ceiling_always_wins(self):
        self.assertEqual(self._agent(10)._step_budget(self._case(19)), 10)
        self.assertEqual(self._agent(10)._step_budget(self._case(2)), 10)


class ScenarioGenerationTests(unittest.TestCase):
    def _generator(self, responses):
        from yewo.testing.generator import ScenarioGenerator

        client = SimpleNamespace(generate_response=Mock(side_effect=responses))
        return ScenarioGenerator(client), client

    def _journey(self, *cells, **extra):
        from yewo.testing.models import UserJourney
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        coverage = list(cells) or [
            CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)
        ]
        return UserJourney(
            id="j",
            name="J",
            description="A user signs in.",
            security_rules=[],
            coverage=coverage,
            **extra,
        )

    def _draft(self, name="Sign in"):
        from yewo.testing.models import ScenarioDraft

        return ScenarioDraft(
            scenario_name=name,
            steps=walk("Launch the app."),
            expected_outcomes=["the app opens"],
        )

    def test_the_draft_schema_carries_no_identity(self):
        """Yewo knows which cell it asked about, so a model cannot miscategorise.

        A model asked to repeat the category, the type or the id could only
        disagree with the question. The identity is stamped on afterwards.
        """
        from yewo.testing.models import ScenarioDraft

        offered = set(ScenarioDraft.model_json_schema()["properties"])
        self.assertEqual(
            offered & {"test_id", "cell", "category", "scenario_type", "priority"}, set()
        )

    def test_the_cell_decides_the_identity(self):
        from yewo.testing.models import GeneratedTestCase
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        cell = CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P0)
        stamped = GeneratedTestCase.of(cell, self._draft())

        self.assertEqual(stamped.cell, "security.negative.0")
        self.assertEqual(stamped.test_id, "TS_security_negative_0")
        self.assertEqual(stamped.category, Category.SECURITY)
        self.assertEqual(stamped.priority, Priority.P0)

    def test_a_scenario_without_an_outcome_is_refused(self):
        """A scenario with no outcome is a schema mismatch, not a valid scenario.

        The gate judges a scenario by its outcomes, so one that declares none
        can only report as unverified. Refusing it here is what starts the
        retry: `LLMClient.generate_response` sends the validation error back,
        and the generator asks again.
        """
        from pydantic import ValidationError

        from yewo.testing.models import GeneratedTestCase

        with self.assertRaises(ValidationError):
            GeneratedTestCase(
                test_id="TS_1",
                scenario_name="S",
                cell="functional.happy_path.0",
                category="functional",
                scenario_type="happy_path",
                priority="P0",
                steps=walk("Open the app."),
                expected_outcomes=[],
            )

        schema = GeneratedTestCase.model_json_schema()
        self.assertIn("expected_outcomes", schema["required"])
        self.assertEqual(schema["properties"]["expected_outcomes"]["minItems"], 1)

    def test_a_refused_response_is_retried(self):
        from pydantic import ValidationError

        rejected = ValidationError.from_exception_data("ScenarioDraft", [])
        generator, client = self._generator([rejected, self._draft()])

        suite = generator.generate_suite(self._journey())

        self.assertEqual(len(suite.scenarios), 1)
        self.assertEqual(suite.uncovered, [])
        self.assertEqual(client.generate_response.call_count, 2)

    def test_one_dead_cell_does_not_cost_the_others(self):
        """A device is already open, so one unanswered cell is not a reason to test nothing."""
        from pydantic import ValidationError

        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        rejected = ValidationError.from_exception_data("ScenarioDraft", [])
        generator, client = self._generator(
            [rejected] * 3 + [self._draft("Refuse a malformed address")]
        )
        first = CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)
        second = CoverageCell(Category.SECURITY, ScenarioType.NEGATIVE, Priority.P0)

        suite = generator.generate_suite(self._journey(first, second))

        self.assertEqual([s.cell for s in suite.scenarios], ["security.negative.0"])
        self.assertEqual(
            [gap.cell.key for gap in suite.uncovered], ["functional.happy_path.0"]
        )
        self.assertEqual(client.generate_response.call_count, 4)

    def test_a_response_that_never_matches_the_schema_blames_the_provider(self):
        """A model that cannot answer is not a cell that asked for the impossible."""
        from pydantic import ValidationError

        from yewo.errors import Cause

        rejected = ValidationError.from_exception_data("ScenarioDraft", [])
        generator, _ = self._generator([rejected] * 3)

        gap = generator.generate_suite(self._journey()).uncovered[0]

        self.assertEqual(gap.status, "ERROR")
        self.assertEqual(gap.cause, Cause.PROVIDER)

    def test_a_provider_outage_is_not_blamed_on_the_scenario(self):
        """The client retried and gave up, so this cell is lost and the rest are not."""
        from yewo.errors import Cause, ProviderUnavailable

        spent = ProviderUnavailable("the provider did not answer after 3 attempts")
        generator, _ = self._generator([spent] * 3)

        gap = generator.generate_suite(self._journey()).uncovered[0]

        self.assertEqual(gap.status, "ERROR")
        self.assertEqual(gap.cause, Cause.PROVIDER)

    def test_a_refused_request_stops_the_generator_asking_for_more(self):
        """A key the service rejects is rejected for every cell after this one."""
        from litellm import exceptions

        refused = exceptions.AuthenticationError("bad key", "m", "openrouter")
        generator, llm = self._generator([refused] * 9)

        with self.assertRaises(exceptions.AuthenticationError):
            generator.generate_suite(self._journey())

        self.assertEqual(llm.generate_response.call_count, 1)

    def test_yewo_writes_the_outcome_for_a_value_the_file_says_is_refused(self):
        """The file said so, so the words carrying it are not the model's prose."""
        from yewo.testing.models import ScenarioDraft
        from yewo.testing.steps import ActionStep, EnterStep

        draft = ScenarioDraft(
            scenario_name="Refuse a short password",
            steps=[
                ActionStep(instruction="Launch the app."),
                EnterStep(field="Password", value="rejected_password"),
            ],
            expected_outcomes=["the sign-in screen is visible"],
        )
        generator, _ = self._generator([draft])
        journey = self._journey(
            values={"rejected_password": "abc"}, refused=frozenset({"rejected_password"})
        )

        scenario = generator.generate_suite(journey).scenarios[0]

        statements = [outcome.statement for outcome in scenario.expected_outcomes]
        self.assertIn("the sign-in screen is visible", statements)
        self.assertTrue(
            any("refuses the value entered in the Password" in s for s in statements),
            statements,
        )

    def test_a_value_meant_to_work_gets_no_outcome_of_yewos(self):
        """Nothing is owed, so an outcome here would assert the app's own choice."""
        from yewo.testing.models import ScenarioDraft
        from yewo.testing.steps import ActionStep, EnterStep

        draft = ScenarioDraft(
            scenario_name="Sign in",
            steps=[
                ActionStep(instruction="Launch the app."),
                EnterStep(field="Email", value="email"),
            ],
            expected_outcomes=["the home screen is visible"],
        )
        generator, _ = self._generator([draft])
        journey = self._journey(values={"email": "qa@example.com"})

        scenario = generator.generate_suite(journey).scenarios[0]

        self.assertEqual(len(scenario.expected_outcomes), 1)

    def test_the_rendered_walk_comes_from_the_steps(self):
        """One object, so the sentence and the value cannot name different things."""
        from yewo.testing.models import ScenarioDraft
        from yewo.testing.steps import ActionStep, EnterStep

        draft = ScenarioDraft(
            scenario_name="Sign in",
            steps=[
                ActionStep(instruction="Launch the app."),
                EnterStep(field="Email", value="email"),
            ],
            expected_outcomes=["the home screen is visible"],
        )
        generator, _ = self._generator([draft])
        journey = self._journey(values={"email": "qa@example.com"})

        scenario = generator.generate_suite(journey).scenarios[0]

        self.assertEqual(
            scenario.steps_instructions,
            ["Launch the app.", "Enter 'qa@example.com' in the Email field."],
        )

    def test_a_cell_needing_an_undeclared_actor_is_not_asked_for(self):
        """A second account cannot be guessed, so the cell is reported rather than faked."""
        from yewo.errors import Cause
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        generator, client = self._generator([self._draft()])
        role = CoverageCell(Category.SECURITY, ScenarioType.ROLE, Priority.P0)

        suite = generator.generate_suite(self._journey(role))

        self.assertEqual(suite.scenarios, [])
        gap = suite.uncovered[0]
        self.assertEqual(gap.cell.key, "security.role.0")
        self.assertEqual(gap.status, "UNSUPPORTED")
        self.assertEqual(gap.cause, Cause.SCENARIO)
        self.assertIn("'actors'", gap.reason)
        client.generate_response.assert_not_called()

    def test_a_declared_actor_unblocks_the_role_cell(self):
        from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType

        generator, _ = self._generator([self._draft()])
        role = CoverageCell(Category.SECURITY, ScenarioType.ROLE, Priority.P0)
        from yewo.testing.models import Actor

        journey = self._journey(role, actors={"editor": Actor(values={"email": "e@x.com"})})

        suite = generator.generate_suite(journey)

        self.assertEqual([s.cell for s in suite.scenarios], ["security.role.0"])
        self.assertEqual(suite.uncovered, [])


if __name__ == "__main__":
    unittest.main()
