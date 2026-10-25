"""Reading one yewo.yaml into the workflows it describes.

A workflow owns an app, a device and an environment. A journey inside it is
only the test, so one file covers several apps.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from yewo.config import ConfigError
from yewo.testing.suite import load_workflows, select

SUITE = {
    "device": {"name": "Pixel_8", "platform": "android"},
    "model": "openrouter/google/gemini-2.5-flash",
    "env": {"city": "London"},
    "workflows": {
        "calendar": {
            "app": "android/calendar.apk",
            "env": {"title": "Dentist"},
            "journeys": [
                {"id": "cal_01", "name": "Create", "journey": "Add {{title}} in {{city}}."}
            ],
        },
        "markor": {
            "app": "android/markor.apk",
            "device": {"platform": "android", "headless": True},
            "journeys": [{"id": "mk_01", "journey": "Grant storage."}],
        },
    },
}


def write(directory, data):
    path = Path(directory) / "yewo.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


class WorkflowTests(unittest.TestCase):
    """What each workflow ends up owning."""

    def _load(self, data=None):
        with TemporaryDirectory() as directory:
            path = write(directory, data if data is not None else SUITE)
            return load_workflows(path), directory

    def test_every_workflow_in_the_file_is_read(self):
        workflows, _ = self._load()

        self.assertEqual([flow.name for flow in workflows], ["calendar", "markor"])

    def test_a_workflow_inherits_the_settings_above_it(self):
        workflows, _ = self._load()

        self.assertEqual(workflows[0].settings["device"], "Pixel_8")
        self.assertEqual(
            workflows[0].settings["model"], "openrouter/google/gemini-2.5-flash"
        )

    def test_a_workflow_overrides_what_it_sets_itself(self):
        """Two workflows in one file may want different devices."""
        workflows, _ = self._load()

        self.assertTrue(workflows[1].settings["headless"])
        self.assertNotIn("headless", workflows[0].settings)

    def test_app_is_the_app_path_the_runner_wants(self):
        workflows, directory = self._load()

        self.assertEqual(
            workflows[0].settings["app_path"],
            str((Path(directory) / "android/calendar.apk").resolve()),
        )

    def test_an_absolute_app_path_is_left_alone(self):
        data = {"workflows": {"a": {"app": "/opt/app.apk", "journeys": [{"journey": "x"}]}}}
        workflows, _ = self._load(data)

        self.assertEqual(workflows[0].settings["app_path"], "/opt/app.apk")

    def test_a_journey_is_only_the_test(self):
        workflows, _ = self._load()
        journey = workflows[0].journeys[0]

        self.assertEqual(journey.id, "cal_01")
        self.assertEqual(journey.name, "Create")

    def test_a_journey_sees_the_environment_of_its_workflow_and_the_file(self):
        workflows, _ = self._load()

        self.assertEqual(workflows[0].journeys[0].description, "Add Dentist in London.")

    def test_a_journey_may_override_a_shared_value(self):
        data = {
            "env": {"city": "London"},
            "workflows": {
                "a": {
                    "journeys": [
                        {"id": "x", "env": {"city": "Leeds"}, "journey": "Go to {{city}}."}
                    ]
                }
            },
        }
        workflows, _ = self._load(data)

        self.assertEqual(workflows[0].journeys[0].description, "Go to Leeds.")

    def test_workflow_policy_waivers_extend_and_override_the_file(self):
        data = {
            "policy": {"waive": {"android.allow_backup": "legacy migration"}},
            "workflows": {
                "a": {
                    "policy": {
                        "waive": {
                            "android.allow_backup": "encrypted export",
                            "android.debuggable": "internal build",
                        }
                    },
                    "journeys": [{"journey": "x"}],
                }
            },
        }

        workflows, _ = self._load(data)

        self.assertEqual(
            workflows[0].policy_waivers,
            {
                "android.allow_backup": "encrypted export",
                "android.debuggable": "internal build",
            },
        )

    def test_a_workflow_reads_targets_in_declared_order(self):
        data = {
            "workflows": {
                "a": {
                    "app": "fallback.apk",
                    "targets": [
                        {
                            "id": "pixel",
                            "platform": "android",
                            "device": "Pixel_8",
                            "app": "android/app.apk",
                        },
                        {
                            "id": "iphone",
                            "platform": "ios",
                            "device": "iPhone 16 Pro",
                            "app": "ios/Runner.app",
                        },
                    ],
                    "journeys": [{"journey": "x"}],
                }
            }
        }

        workflows, directory = self._load(data)

        self.assertEqual([target.id for target in workflows[0].targets], ["pixel", "iphone"])
        self.assertEqual(workflows[0].targets[0].settings["device"], "Pixel_8")
        self.assertEqual(
            workflows[0].targets[1].settings["app_path"],
            str((Path(directory) / "ios/Runner.app").resolve()),
        )

    def test_file_targets_are_inherited_and_a_workflow_may_replace_them(self):
        data = {
            "targets": [
                {"id": "shared", "platform": "android", "device": "Pixel_8"}
            ],
            "workflows": {
                "a": {"journeys": [{"journey": "x"}]},
                "b": {
                    "targets": [
                        {"id": "ios", "platform": "ios", "device": "iPhone 16"}
                    ],
                    "journeys": [{"journey": "y"}],
                },
            },
        }

        workflows, _ = self._load(data)

        self.assertEqual([target.id for target in workflows[0].targets], ["shared"])
        self.assertEqual([target.id for target in workflows[1].targets], ["ios"])


class LegacyFileTests(unittest.TestCase):
    """A journey file written before workflows existed still runs."""

    def _load(self, data):
        with TemporaryDirectory() as directory:
            return load_workflows(write(directory, data))

    def test_a_single_journey_file_is_one_workflow(self):
        workflows = self._load(
            {"id": "solo", "journey": "Open the app.", "device": {"platform": "ios"}}
        )

        self.assertEqual(len(workflows), 1)
        self.assertEqual(workflows[0].name, "")
        self.assertEqual(workflows[0].journeys[0].id, "solo")
        self.assertEqual(workflows[0].settings["platform"], "ios")

    def test_a_journeys_list_is_one_workflow(self):
        workflows = self._load(
            {"journeys": [{"id": "a", "journey": "x"}, {"id": "b", "journey": "y"}]}
        )

        self.assertEqual([j.id for j in workflows[0].journeys], ["a", "b"])

    def test_a_file_with_no_journeys_is_refused(self):
        """Silently running nothing would report a passing suite."""
        with self.assertRaises(ConfigError):
            self._load({"model": "m"})


class SelectTests(unittest.TestCase):
    """Choosing which workflows a run covers."""

    def _flows(self):
        with TemporaryDirectory() as directory:
            return load_workflows(write(directory, SUITE))

    def test_everything_runs_when_none_is_named(self):
        self.assertEqual(len(select(self._flows(), None)), 2)

    def test_one_name_runs_one_workflow(self):
        self.assertEqual([f.name for f in select(self._flows(), "markor")], ["markor"])

    def test_several_names_run_in_the_order_given(self):
        chosen = select(self._flows(), "markor,calendar")

        self.assertEqual([f.name for f in chosen], ["markor", "calendar"])

    def test_a_name_the_file_does_not_have_is_refused(self):
        """Running everything instead would hide the typo."""
        with self.assertRaises(ConfigError) as caught:
            select(self._flows(), "calender")

        self.assertIn("calender", str(caught.exception))


class SchemaTests(unittest.TestCase):
    """A yewo.yaml holds the settings and the tests, so a typo in it can mean
    a setting that does nothing or a test that never runs."""

    def _load(self, data):
        with TemporaryDirectory() as directory:
            return load_workflows(write(directory, data))

    def _refuse(self, data):
        with self.assertRaises(ConfigError) as caught:
            self._load(data)
        return str(caught.exception)

    def test_a_misspelled_setting_is_refused_by_name(self):
        """It used to be accepted, and the run went ahead without it."""
        message = self._refuse(
            {"modle": "gpt", "workflows": {"a": {"journeys": [{"journey": "x"}]}}}
        )

        self.assertIn("modle: unknown key", message)

    def test_a_misspelled_platform_is_refused(self):
        """Anything not 'android' used to select the iOS path."""
        message = self._refuse(
            {"platform": "androdi", "workflows": {"a": {"journeys": [{"journey": "x"}]}}}
        )

        self.assertIn("'androdi'", message)
        self.assertIn("android", message)

    def test_a_key_misspelled_inside_a_workflow_names_the_workflow(self):
        message = self._refuse({"workflows": {"calendar": {"ap": "x.apk", "journeys": []}}})

        self.assertIn("workflows.calendar.ap", message)

    def test_a_key_misspelled_inside_a_journey_names_the_journey(self):
        message = self._refuse(
            {"workflows": {"a": {"journeys": [{"journey": "x", "secrity_rules": []}]}}}
        )

        self.assertIn("secrity_rules", message)

    def test_every_problem_is_reported_at_once(self):
        """Fixing one at a time means one run each."""
        message = self._refuse(
            {"modle": "gpt", "groundng": "marks", "workflows": {"a": {"journeys": []}}}
        )

        self.assertIn("modle", message)
        self.assertIn("groundng", message)

    def test_an_unusable_max_steps_is_refused(self):
        self.assertIn("max_steps", self._refuse({"max_steps": 0, "journeys": [{"journey": "x"}]}))

    def test_a_valid_file_is_accepted(self):
        workflows = self._load(SUITE)

        self.assertEqual(len(workflows), 2)

    def test_a_waiver_without_a_reason_is_refused(self):
        message = self._refuse(
            {
                "policy": {"waive": {"android.debuggable": "  "}},
                "journeys": [{"journey": "x"}],
            }
        )

        self.assertIn("android.debuggable", message)

    def test_duplicate_target_ids_are_refused(self):
        message = self._refuse(
            {
                "targets": [
                    {"id": "phone", "platform": "android", "device": "Pixel"},
                    {"id": "phone", "platform": "ios", "device": "iPhone"},
                ],
                "journeys": [{"journey": "x"}],
            }
        )

        self.assertIn("target IDs must be unique", message)

    def test_a_target_id_cannot_escape_its_artifact_directory(self):
        message = self._refuse(
            {
                "targets": [
                    {"id": "../phone", "platform": "android", "device": "Pixel"}
                ],
                "journeys": [{"journey": "x"}],
            }
        )

        self.assertIn("targets.0.id", message)
