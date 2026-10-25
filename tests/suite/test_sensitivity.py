"""How sensitive each artifact is, and the one that decides a policy.

Yewo masks declared values across every artifact but one. Nothing masks a
video frame. These cases hold that difference to being declared in the record
rather than remembered by whoever uploads it.
"""

import unittest

from yewo.testing.models import UserJourney
from yewo.testing.sensitivity import (
    ARTIFACTS,
    RECORDING,
    Sensitivity,
    declared_for,
    exposure_policy_refused,
    recording_refused,
)


def journey(**extra) -> UserJourney:
    return UserJourney(id="j", name="J", description="A user signs in.", **extra)


def classed(journey_in) -> dict:
    return {row["kind"]: row for row in declared_for(journey_in)}


class ClassTests(unittest.TestCase):
    def test_every_artifact_a_run_writes_is_classed(self):
        """A class in the record travels with the file. A list does not."""
        kinds = {artifact.kind for artifact in ARTIFACTS}

        self.assertEqual(
            kinds,
            {
                "manifest",
                "reports",
                "prompts",
                "telemetry",
                "screenshots",
                "logs",
                "device_logs",
                "recordings",
            },
        )

    def test_only_what_yewo_never_touched_is_unmasked(self):
        """The video and the device's own log. Yewo writes neither of them."""
        unmasked = [
            artifact.kind
            for artifact in ARTIFACTS
            if artifact.sensitivity is Sensitivity.UNMASKED
        ]

        self.assertEqual(sorted(unmasked), ["device_logs", RECORDING])

    def test_an_artifact_a_later_reader_needs_is_kept(self):
        """Retention is declared here and enforced nowhere. Nothing deletes a file."""
        from yewo.testing.sensitivity import Retention

        kept = {
            artifact.kind
            for artifact in ARTIFACTS
            if artifact.retention is Retention.KEEP
        }

        self.assertIn("manifest", kept)
        self.assertIn("screenshots", kept)
        self.assertNotIn(RECORDING, kept)
        self.assertNotIn("device_logs", kept)

    def test_an_unmasked_artifact_is_not_exportable(self):
        for artifact in ARTIFACTS:
            with self.subTest(kind=artifact.kind):
                if artifact.sensitivity is Sensitivity.UNMASKED:
                    self.assertFalse(artifact.exportable)

    def test_a_video_is_unmasked_whatever_the_journey_declared(self):
        """A declaration says what Yewo masks, not what the app renders.

        A video holds another account's data, a notification that arrived
        mid-run, a token in a debug banner, or a credential the app made up.
        None of those were declared, and none of them is masked.
        """
        for declaring in (journey(), journey(secrets={"password": "password123"})):
            with self.subTest(secrets=bool(declaring.secrets)):
                row = classed(declaring)[RECORDING]
                self.assertEqual(row["sensitivity"], Sensitivity.UNMASKED.value)
                self.assertFalse(row["exportable"])

    def test_what_a_declaration_changes_is_whether_one_is_made(self):
        self.assertIsNone(recording_refused(journey(), opted_in=False))
        self.assertIsNotNone(
            recording_refused(journey(secrets={"password": "p"}), opted_in=False)
        )

    def test_no_declared_value_reaches_the_classes(self):
        rows = declared_for(journey(secrets={"password": "password123"}))

        self.assertNotIn("password123", str(rows))


class RecordingPolicyTests(unittest.TestCase):
    """A journey that types a secret on screen keeps no video by default."""

    def test_a_journey_with_no_secret_is_recorded(self):
        self.assertIsNone(recording_refused(journey(), opted_in=False))

    def test_a_journey_with_a_secret_is_not(self):
        refused = recording_refused(
            journey(secrets={"password": "password123"}), opted_in=False
        )

        self.assertIsNotNone(refused)
        self.assertIn("password", refused)
        self.assertIn("--allow-secret-exposure", refused)

    def test_the_refusal_names_what_was_declared_and_not_its_value(self):
        refused = recording_refused(
            journey(secrets={"password": "password123"}), opted_in=False
        )

        self.assertNotIn("password123", refused)

    def test_asking_for_it_in_so_many_words_is_enough(self):
        self.assertIsNone(
            recording_refused(
                journey(secrets={"password": "password123"}), opted_in=True
            )
        )


class ModelExposurePolicyTests(unittest.TestCase):
    def test_non_hidden_policy_requires_the_explicit_gate(self):
        refused = exposure_policy_refused(
            journey(secret_displays={"password": "show"}), opted_in=False
        )

        self.assertIn("password", refused)
        self.assertIn("--allow-secret-exposure", refused)

    def test_hidden_policy_needs_no_unsafe_opt_in(self):
        self.assertIsNone(
            exposure_policy_refused(
                journey(secret_displays={"password": "hide"}), opted_in=False
            )
        )
