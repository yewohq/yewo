"""A run with no verified app evidence reports gaps; it never passes with no scenario."""

import tempfile
import unittest

from fakes import suite_run, walk

from yewo.errors import Cause
from yewo.testing.models import GeneratedTestCase, ReportStatus, ScenarioDraft, UserJourney
from yewo.testing.taxonomy import Category, CoverageCell, Priority, ScenarioType


def _journey() -> UserJourney:
    return UserJourney(
        id="trainer",
        name="Trainer",
        description="A coach edits a schedule.",
        coverage=[CoverageCell(Category.FUNCTIONAL, ScenarioType.HAPPY_PATH, Priority.P0)],
    )


def _scenario(journey: UserJourney) -> GeneratedTestCase:
    return GeneratedTestCase.of(
        journey.coverage[0],
        ScenarioDraft(
            scenario_name="Edit the schedule",
            steps=walk("Open the schedule."),
            expected_outcomes=[
                {"statement": "The schedule is visible.", "evidence": "A coach edits a schedule."}
            ],
        ),
    )


class StartupWithoutEvidenceTests(unittest.TestCase):
    def test_every_declared_cell_is_unverified_and_no_scenario_runs(self):
        journey = _journey()
        with tempfile.TemporaryDirectory() as directory:
            report, written = suite_run(
                [_scenario(journey)],
                directory,
                journey=journey,
                generation_evidence=None,
            )

        self.assertEqual(len(report.scenarios), len(journey.coverage))
        gap = report.scenarios[0]
        self.assertEqual(gap.status, ReportStatus.UNVERIFIED)
        self.assertEqual(gap.cause, Cause.SCENARIO.value)
        self.assertEqual(gap.cell, journey.coverage[0].key)
        self.assertEqual(gap.steps_executed, 0)
        self.assertIn("capability_unobserved", gap.failure_reason)
        self.assertNotIn(
            "Edit the schedule",
            [row["name"] for row in written["scenarios"]],
        )
        self.assertEqual(len(written["uncovered"]), len(journey.coverage))

    def test_the_run_is_not_reported_as_passed(self):
        journey = _journey()
        with tempfile.TemporaryDirectory() as directory:
            _, written = suite_run(
                [_scenario(journey)],
                directory,
                journey=journey,
                generation_evidence=None,
            )

        self.assertEqual(written["status"], ReportStatus.FAILED.value)
        self.assertEqual(written["counts"].get("PASSED", 0), 0)
