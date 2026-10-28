"""How each persisted artifact declares its masking and retention policy."""

import unittest

from yewo.testing.models import UserJourney
from yewo.testing.sensitivity import (
    ARTIFACTS,
    RECORDING,
    Sensitivity,
    declared_for,
    exposure_policy_refused,
)


def journey(**extra) -> UserJourney:
    return UserJourney(id="j", name="J", description="A user signs in.", **extra)


def classed(journey_in, allow_secret_exposure=False) -> dict:
    return {
        row["kind"]: row
        for row in declared_for(journey_in, allow_secret_exposure)
    }


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

    def test_a_default_secret_recording_is_redacted_and_exportable(self):
        row = classed(journey(secrets={"password": "password123"}))[RECORDING]

        self.assertEqual(row["sensitivity"], Sensitivity.REDACTED.value)
        self.assertTrue(row["exportable"])

    def test_a_recording_without_declared_secrets_remains_unmasked(self):
        row = classed(journey())[RECORDING]

        self.assertEqual(row["sensitivity"], Sensitivity.UNMASKED.value)
        self.assertFalse(row["exportable"])

    def test_the_explicit_exposure_opt_in_keeps_an_unmasked_recording(self):
        row = classed(
            journey(secrets={"password": "password123"}),
            allow_secret_exposure=True,
        )[RECORDING]

        self.assertEqual(row["sensitivity"], Sensitivity.UNMASKED.value)
        self.assertFalse(row["exportable"])

    def test_no_declared_value_reaches_the_classes(self):
        rows = declared_for(journey(secrets={"password": "password123"}))

        self.assertNotIn("password123", str(rows))
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
