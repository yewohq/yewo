"""`run.json` — the record a run leaves, and what must never reach it.

One writer, listing every scenario before the first one starts and updating
each as it finishes. These cases hold it to being readable while the run is
going, honest about a run that was killed, and free of any declared secret.
"""

import json
import tempfile
import unittest
from pathlib import Path

from fakes import walk

from yewo.agent.run.record import DecisionRecord, StepRecord
from yewo.testing.models import (
    GeneratedTestCase,
    ReportStatus,
    RunEnvironment,
    ScenarioReport,
    UserJourney,
)
from yewo.testing.report.manifest import SCHEMA, RunInputs, RunManifest, RunSubject
from yewo.utils.secrets import SecretScrubber


def environment() -> RunEnvironment:
    return RunEnvironment(
        yewo_version="0.1.0",
        model="openrouter/m",
        platform="android",
        grounding="marks",
    )


def inputs() -> RunInputs:
    return RunInputs(
        subject="j1",
        subject_sha256="0" * 64,
        coverage_declared=["functional.happy_path.0"],
        lock_sha256="1" * 64,
        scenarios_from_lock=1,
        scenarios_generated=1,
    )


def scenario(number: int, **extra) -> GeneratedTestCase:
    fields = {
        "test_id": f"TS_00{number}",
        "scenario_name": f"Scenario {number}",
        "cell": "functional.happy_path.0",
        "category": "functional",
        "scenario_type": "happy_path",
        "priority": "P0",
        "steps": walk("Open the app."),
        "expected_outcomes": ["ok"],
    }
    fields.update(extra)
    return GeneratedTestCase(**fields)


def manifest(directory, scenarios=None, journey=None, scrubber=None) -> RunManifest:
    about = journey or UserJourney(
        id="j1", name="Journey", description="d", security_rules=[]
    )
    return RunManifest(
        Path(directory) / "run.json",
        RunSubject(kind="journey", id=about.id, name=about.name),
        scenarios if scenarios is not None else [scenario(1), scenario(2)],
        environment=environment(),
        inputs=inputs(),
        scrubber=scrubber,
    )


def written(record: RunManifest) -> dict:
    return json.loads(record.path.read_text())


class PlanTests(unittest.TestCase):
    """What the file says before any scenario has run."""

    def test_every_scenario_is_listed_before_anything_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.write()
            data = written(record)

        self.assertEqual(data["total"], 2)
        self.assertEqual(data["counts"], {"PENDING": 2})

    def test_the_schema_is_the_first_thing_a_reader_can_find(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.write()
            data = written(record)

        self.assertEqual(data["schema"], SCHEMA)
        self.assertEqual(list(data)[0], "schema")

    def test_the_run_is_named_by_a_drawn_identifier(self):
        with tempfile.TemporaryDirectory() as directory:
            first = manifest(directory)
            second = manifest(directory)

        self.assertNotEqual(first.run_uuid, second.run_uuid)

    def test_identity_and_inputs_are_recorded_with_the_plan(self):
        """A reader of a failed scenario needs the build it failed on."""
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.write()
            data = written(record)

        self.assertEqual(data["identity"]["platform"], "android")
        self.assertEqual(data["inputs"]["lock_sha256"], "1" * 64)
        self.assertEqual(data["inputs"]["scenarios_from_lock"], 1)
        self.assertEqual(data["inputs"]["coverage_declared"], ["functional.happy_path.0"])

    def test_each_scenario_carries_the_identity_it_was_locked_under(self):
        with tempfile.TemporaryDirectory() as directory:
            one = scenario(1)
            record = manifest(directory, [one])
            record.write()
            data = written(record)

        self.assertEqual(data["scenarios"][0]["scenario_uuid"], one.scenario_uuid)
        self.assertEqual(data["scenarios"][0]["test_id"], "TS_001")

    def test_an_outcome_is_named_once_per_position(self):
        """Derived from the scenario and the position, which is what it is."""
        with tempfile.TemporaryDirectory() as directory:
            one = scenario(1, expected_outcomes=["a", "b"])
            record = manifest(directory, [one])
            record.write()
            named = written(record)["scenarios"][0]["outcome_uuids"]

        self.assertEqual(named, [one.outcome_uuid(0), one.outcome_uuid(1)])
        self.assertEqual(len(set(named)), 2)

    def test_two_scenarios_do_not_share_the_name_of_their_first_outcome(self):
        """Position alone is not an identity. Two tests have a first outcome."""
        first = scenario(1, expected_outcomes=["a"])
        second = scenario(2, expected_outcomes=["a"])

        self.assertNotEqual(first.outcome_uuid(0), second.outcome_uuid(0))

    def test_a_locked_scenario_names_its_outcomes_the_same_way_every_run(self):
        """Which is what lets one outcome be compared between two runs."""
        from yewo.testing.models import GeneratedTestCase

        one = scenario(1, expected_outcomes=["a"])
        again = GeneratedTestCase(
            scenario_uuid=one.scenario_uuid,
            test_id=one.test_id,
            scenario_name=one.scenario_name,
            cell=one.cell,
            category="functional",
            scenario_type="happy_path",
            priority="P0",
            steps=one.steps,
            expected_outcomes=one.expected_outcomes,
        )

        self.assertEqual(one.outcome_uuid(0), again.outcome_uuid(0))

    def test_the_file_is_named_the_same_whatever_the_journey_is_called(self):
        """A journey id reached the filename before, and ids come from a file."""
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(
                directory,
                journey=UserJourney(id="../../escape", name="J", description="d"),
            )
            record.write()

            self.assertEqual(record.path.parent, Path(directory))
            self.assertEqual(record.path.name, "run.json")

    def test_nothing_part_written_is_left_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.write()

            self.assertEqual(
                sorted(p.name for p in Path(directory).iterdir()), ["run.json"]
            )


class ProgressTests(unittest.TestCase):
    """What the file says while the run is going, and after it stops."""

    def _report(self, **extra):
        fields = {
            "test_id": "TS_001",
            "scenario_name": "Scenario 1",
            "cell": "functional.happy_path.0",
            "status": "FAILED",
            "duration": 12.34,
            "steps_executed": 9,
            "failure_reason": "wrong screen",
        }
        fields.update(extra)
        return ScenarioReport(**fields)

    def test_a_status_updates_as_each_scenario_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.start("TS_001")
            record.record("TS_001", self._report())
            data = written(record)

        first = data["scenarios"][0]
        self.assertEqual(first["status"], "FAILED")
        self.assertEqual(first["steps_executed"], 9)
        self.assertEqual(first["duration_seconds"], 12.3)
        self.assertEqual(data["counts"], {"FAILED": 1, "PENDING": 1})

    def test_every_status_in_the_record_is_one_yewo_spells(self):
        """One vocabulary, so a reader of the file needs one list of words.

        The statuses a scenario passes through on its way to a verdict are
        written by the field that later holds the verdict. Spelled apart,
        `counts` gathers one run under two keys, and whoever reads the
        record has to know both lists.

        Read while one scenario is running and another has not started, so
        both of those words are in the file rather than overwritten.
        """
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.start("TS_001")
            data = written(record)

        spelled = {status.value for status in ReportStatus}
        self.assertEqual(
            [row["status"] for row in data["scenarios"]], ["RUNNING", "PENDING"]
        )
        self.assertIn(data["status"], spelled)
        self.assertLessEqual(set(data["counts"]), spelled)
        for row in data["scenarios"]:
            self.assertIn(row["status"], spelled, row["test_id"])

    def test_a_crash_leaves_nothing_marked_running(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.start("TS_001")
            record.abort("interrupted")
            data = written(record)

        self.assertEqual(data["counts"], {"ABORTED": 2})
        self.assertEqual(data["status"], "ABORTED")
        self.assertTrue(
            all(row["failure_reason"] == "interrupted" for row in data["scenarios"])
        )

    def test_a_run_that_was_never_closed_has_no_end(self):
        """So a killed run reads as killed rather than as one that finished."""
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.write()
            running = written(record)
            record.finish("PASSED", usage={"total_calls": 4})
            finished = written(record)

        self.assertIsNone(running.get("ended_at"))
        self.assertEqual(running["status"], "RUNNING")
        self.assertIsNotNone(finished["ended_at"])
        self.assertEqual(finished["status"], "PASSED")
        self.assertEqual(finished["usage"]["total_calls"], 4)

    def test_the_first_failure_is_named_so_a_reader_starts_there(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory)
            record.record("TS_002", self._report(test_id="TS_002", status="PASSED"))
            record.record(
                "TS_001",
                self._report(
                    rule_verifications=[
                        {
                            "rule": "the sign-in screen stays visible",
                            "rule_id": "expected:0",
                            "status": "failed",
                            "reasoning": "it did not",
                        }
                    ]
                ),
            )
            first = written(record)["first_failure"]

        self.assertEqual(first["test_id"], "TS_001")
        self.assertEqual(first["rule_id"], "expected:0")

    def test_a_passing_run_names_no_first_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory, [scenario(1)])
            record.record("TS_001", self._report(status="PASSED"))

            self.assertIsNone(written(record)["first_failure"])

    def test_an_unsupported_cell_is_reported_as_a_gap_as_well(self):
        """A cell nothing covered is not a scenario that passed."""
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory, [scenario(1)])
            record.record(
                "TS_001", self._report(status="UNSUPPORTED", cause="scenario")
            )
            data = written(record)

        self.assertEqual(data["uncovered"][0]["test_id"], "TS_001")
        self.assertEqual(data["scenarios"][0]["cause"], "scenario")


class StepRecordTests(unittest.TestCase):
    """The step rows, which are what every later reader is built on."""

    def test_each_step_lands_under_the_scenario_that_ran_it(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory, [scenario(1)])
            record.record(
                "TS_001",
                ScenarioReport(
                    test_id="TS_001",
                    scenario_name="Scenario 1",
                    cell="functional.happy_path.0",
                    status="PASSED",
                    steps=[
                        StepRecord(
                            n=1,
                            screen_name="Sign in",
                            decision=DecisionRecord(
                                reasoning="the goal is to sign in",
                                evaluation_previous_goal="Success — the app opened",
                            ),
                        )
                    ],
                ),
            )
            steps = written(record)["scenarios"][0]["steps"]

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]["screen_name"], "Sign in")
        self.assertEqual(steps[0]["decision"]["reasoning"], "the goal is to sign in")
        self.assertEqual(
            steps[0]["decision"]["evaluation_previous_goal"],
            "Success — the app opened",
        )


class SecretTests(unittest.TestCase):
    """The record is an artifact: uploaded, attached to a review, and kept."""

    def _scrubber(self) -> SecretScrubber:
        scrubber = SecretScrubber()
        scrubber.learn_secrets(
            {"password": "password123", "card_number": "4111222233334444"}
        )
        return scrubber

    def test_a_declared_value_is_masked_wherever_it_appears(self):
        with tempfile.TemporaryDirectory() as directory:
            record = manifest(
                directory,
                [
                    scenario(
                        1,
                        steps=walk("Enter 'password123' in Password"),
                        expected_outcomes=["Do not display '4111222233334444'"],
                    )
                ],
                scrubber=self._scrubber(),
            )
            record.record(
                "TS_001",
                ScenarioReport(
                    test_id="TS_001",
                    scenario_name="Scenario 1",
                    cell="functional.happy_path.0",
                    status="FAILED",
                    failure_reason="typed password123 and the field showed it",
                ),
            )
            body = record.path.read_text()

        self.assertNotIn("password123", body)
        self.assertNotIn("4111222233334444", body)
        self.assertIn("{{password}}", body)


    def test_a_declared_value_is_masked_in_what_the_judge_was_given(self):
        """The inputs carry a finding's quoted text and the journey's rules.

        Both are written by somebody else and both reach this file, so both
        go through the scrubber like everything else in it.
        """
        from yewo.testing.models import (
            ExpectedOutcome, Rendering, SensitiveFinding, VerdictInputs,
        )

        with tempfile.TemporaryDirectory() as directory:
            record = manifest(
                directory,
                [scenario(1)],
                scrubber=self._scrubber(),
            )
            record.record(
                "TS_001",
                ScenarioReport(
                    test_id="TS_001",
                    scenario_name="Scenario 1",
                    cell="functional.happy_path.0",
                    status="FAILED",
                    verdict_inputs=VerdictInputs(
                        run_status="goal_reached",
                        stop_reason="stopped after typing password123",
                        infeasibility_reason="4111222233334444 was refused",
                        security_rules=["Never show 4111222233334444"],
                        expected_outcomes=[
                            ExpectedOutcome(statement="password123 is masked")
                        ],
                        sensitive_findings=[
                            SensitiveFinding(
                                field="Password",
                                rendering=Rendering.VALUE,
                                quoted="password123",
                            )
                        ],
                    ),
                ),
            )
            body = record.path.read_text()

        self.assertNotIn("password123", body)
        self.assertNotIn("4111222233334444", body)
        self.assertIn("{{password}}", body)
        self.assertIn("{{card_number}}", body)


class GoalRunTests(unittest.TestCase):
    """`yewo run` writes one too, or it cannot be read back at all."""

    def _goal(self, directory) -> RunManifest:
        from yewo.testing.report.manifest import RunSubject

        return RunManifest(
            Path(directory) / "run.json",
            RunSubject(kind="goal", id="abc123", name="reach checkout"),
            scenarios=(),
            environment=environment(),
            inputs=RunInputs(subject="goal", subject_sha256="2" * 64),
            mode="run",
        )

    def test_a_goal_run_records_what_it_was_about(self):
        with tempfile.TemporaryDirectory() as directory:
            record = self._goal(directory)
            record.write()
            data = written(record)

        self.assertEqual(data["mode"], "run")
        self.assertEqual(data["subject"]["kind"], "goal")
        self.assertEqual(data["subject"]["name"], "reach checkout")
        self.assertEqual(data["scenarios"], [])

    def test_the_steps_of_a_goal_run_sit_at_the_top_level(self):
        """It has no scenario to put them under, and a walk with no steps is unreadable."""
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            record = self._goal(directory)
            record.record_goal(
                SimpleNamespace(
                    status="goal_reached",
                    steps=[StepRecord(n=1, screen_name="Checkout")],
                )
            )
            data = written(record)

        self.assertEqual(data["status"], "PASSED")
        self.assertEqual(len(data["steps"]), 1)
        self.assertEqual(data["steps"][0]["screen_name"], "Checkout")
        self.assertIsNotNone(data["ended_at"])

    def test_a_goal_that_was_not_reached_fails(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            record = self._goal(directory)
            record.record_goal(SimpleNamespace(status="step_limit", steps=[]))

            self.assertEqual(written(record)["status"], "FAILED")

    def test_the_artifact_classes_travel_with_the_record(self):
        from yewo.testing.sensitivity import declared_for

        with tempfile.TemporaryDirectory() as directory:
            record = manifest(directory, [scenario(1)])
            record.artifacts = declared_for(
                UserJourney(id="j1", name="J", description="d", secrets={"p": "abc"})
            )
            record.write()
            classes = {row["kind"]: row for row in written(record)["artifacts"]}

        self.assertEqual(classes["recordings"]["sensitivity"], "unmasked")
        self.assertFalse(classes["recordings"]["exportable"])
        self.assertEqual(classes["screenshots"]["sensitivity"], "redacted")
