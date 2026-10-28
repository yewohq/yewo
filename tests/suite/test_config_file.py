"""One yewo.yaml holding both the settings and the tests.

The settings say how Yewo runs; the journeys say what it runs. They now live
in the same file, so reading it has to tell them apart, and writing to it has
to keep the half it was not asked about.
"""

import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from yewo.config import (
    ConfigError,
    expand_device,
    load_config,
    read_file,
    set_value,
    unset_value,
)

SUITE = """
device:
  name: Pixel_8
  platform: android
  headless: false
model: openrouter/google/gemini-2.5-flash

workflows:
  calendar:
    app: android/org.fossify.calendar_23.apk
    journeys:
      - id: fossify_calendar_event_01
        journey: A user creates an event.
"""


class DeviceBlockTests(unittest.TestCase):
    """`device` names a target in a block, and the rest of Yewo resolves it flat."""

    def test_a_block_becomes_the_settings_it_stands_for(self):
        expanded = expand_device(
            {"device": {"name": "Pixel_8", "platform": "ios", "headless": True}}, "test"
        )

        self.assertEqual(
            expanded, {"device": "Pixel_8", "platform": "ios", "headless": True}
        )

    def test_a_bare_name_is_still_a_device(self):
        """Every config file written before this change spells it this way."""
        self.assertEqual(expand_device({"device": "Pixel_8"}, "test"), {"device": "Pixel_8"})

    def test_a_partial_block_leaves_the_rest_alone(self):
        expanded = expand_device({"device": {"platform": "ios"}, "model": "m"}, "test")

        self.assertEqual(expanded, {"platform": "ios", "model": "m"})

    def test_a_device_that_is_neither_is_refused(self):
        """A list here would otherwise be coerced into a device name."""
        with self.assertRaises(ConfigError):
            expand_device({"device": ["Pixel_8"]}, "yewo.yaml")


class ReadFileTests(unittest.TestCase):
    """Telling settings apart from the journeys beside them."""

    def _read(self, text):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.yaml"
            path.write_text(text, encoding="utf-8")
            with self.assertLogs("yewo.config", level="WARNING") as captured:
                settings = read_file(path)
                # assertLogs fails an otherwise silent read, so give it one.
                import logging

                logging.getLogger("yewo.config").warning("probe")
            return settings, [line for line in captured.output if "probe" not in line]

    def test_the_settings_are_read_through_the_device_block(self):
        settings, _ = self._read(SUITE)

        self.assertEqual(settings["device"], "Pixel_8")
        self.assertEqual(settings["platform"], "android")
        self.assertEqual(settings["model"], "openrouter/google/gemini-2.5-flash")

    def test_the_journeys_are_not_reported_as_misspelled_settings(self):
        """They are structure. Warning about them would train the eye past it."""
        _, warnings = self._read(SUITE)

        self.assertEqual(warnings, [])

    def test_a_real_misspelling_is_still_named(self):
        _, warnings = self._read("modle: gpt\n")

        self.assertEqual(len(warnings), 1)
        self.assertIn("modle", warnings[0])

    def test_a_setting_the_run_applies_is_not_called_unknown(self):
        """A warning that a value was ignored, when it was not, teaches the
        reader to ignore the warnings that are true."""
        settings, warnings = self._read("grounding: marks\napp: app.apk\nrecord: false\n")

        self.assertEqual(warnings, [])
        self.assertEqual(settings["grounding"], "marks")
        self.assertEqual(settings["app_path"], "app.apk")
        self.assertIs(settings["record"], False)

    def test_a_pinned_private_detector_reference_is_a_setting(self):
        reference = f"hf://example/yewo-ui@{'a' * 40}"

        settings, warnings = self._read(f"perception_model: {reference}\n")

        self.assertEqual(settings["perception_model"], reference)
        self.assertEqual(warnings, [])


class WriteTests(unittest.TestCase):
    """A write keeps the half of the file it was not asked about."""

    def _file(self, directory, text=SUITE):
        path = Path(directory) / "yewo.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_saving_a_setting_keeps_every_journey(self):
        """read_file drops the journeys, so writing that back would delete them."""
        with TemporaryDirectory() as directory:
            path = self._file(directory)

            set_value("model", "anthropic/claude-sonnet-5", start=Path(directory))

            document = yaml.safe_load(path.read_text())
            self.assertIn("calendar", document["workflows"])
            self.assertEqual(
                document["workflows"]["calendar"]["journeys"][0]["id"],
                "fossify_calendar_event_01",
            )

    def test_a_device_stays_in_the_block_the_file_uses(self):
        """Writing it flat would leave the file naming the device twice."""
        with TemporaryDirectory() as directory:
            path = self._file(directory)

            set_value("device", "Pixel_9", start=Path(directory))

            document = yaml.safe_load(path.read_text())
            self.assertEqual(document["device"]["name"], "Pixel_9")
            self.assertNotIn("Pixel_8", path.read_text())

    def test_a_file_without_a_block_gets_a_flat_setting(self):
        with TemporaryDirectory() as directory:
            path = self._file(directory, "model: m\n")

            set_value("device", "Pixel_9", start=Path(directory))

            self.assertEqual(yaml.safe_load(path.read_text())["device"], "Pixel_9")

    def test_removing_a_setting_keeps_every_journey(self):
        with TemporaryDirectory() as directory:
            path = self._file(directory)

            unset_value("model", start=Path(directory))

            document = yaml.safe_load(path.read_text())
            self.assertNotIn("model", document)
            self.assertIn("calendar", document["workflows"])

    def test_a_journey_is_still_readable_after_a_write(self):
        """Prose written as a block must not come back as one long quoted line.

        This file is edited by hand, so `yewo devices use` has to leave it
        in the shape a person wrote it.
        """
        prose = "A user opens the app.\nThey tap the add button.\n"
        with TemporaryDirectory() as directory:
            path = self._file(
                directory,
                yaml.safe_dump(
                    {
                        "device": {"name": "Pixel_8"},
                        "workflows": {"a": {"journeys": [{"id": "x", "journey": prose}]}},
                    }
                ),
            )

            set_value("device", "Pixel_9", start=Path(directory))

            written = path.read_text()
            self.assertIn("journey: |", written)
            self.assertEqual(
                yaml.safe_load(written)["workflows"]["a"]["journeys"][0]["journey"], prose
            )

    def test_the_order_of_the_file_is_kept(self):
        """Sorting the keys would move the journeys above the settings."""
        with TemporaryDirectory() as directory:
            path = self._file(directory, "model: m\ndevice: Pixel_8\nworkflows: {}\n")

            set_value("max_steps", 10, start=Path(directory))

            written = path.read_text()
            self.assertLess(written.index("model"), written.index("device"))

    def test_removing_a_device_empties_the_block(self):
        with TemporaryDirectory() as directory:
            path = self._file(directory)

            unset_value("device", start=Path(directory))

            document = yaml.safe_load(path.read_text())
            self.assertNotIn("name", document["device"])


class LoadConfigTests(unittest.TestCase):
    """What a command sees, now that one file answers for it."""

    def test_the_project_file_supplies_the_settings(self):
        with TemporaryDirectory() as directory:
            Path(directory, "yewo.yaml").write_text(SUITE, encoding="utf-8")

            config = load_config(start=Path(directory))

            self.assertEqual(config.get("device"), "Pixel_8")
            self.assertIn("yewo.yaml", config.origin("device"))

    def test_an_env_variable_still_overrides_the_file(self):
        """The way to run one file against another machine's device."""
        with TemporaryDirectory() as directory:
            Path(directory, "yewo.yaml").write_text(SUITE, encoding="utf-8")

            with unittest.mock.patch.dict("os.environ", {"YEWO_DEVICE": "Pixel_9"}):
                config = load_config(start=Path(directory))

            self.assertEqual(config.get("device"), "Pixel_9")
            self.assertEqual(config.origin("device"), "YEWO_DEVICE")


class SchemaBackedSettingsTests(unittest.TestCase):
    """`yewo config` and the journey loader read the same file.

    Naming the settings twice meant a value could be saved here and refused
    by the run that read it back.
    """

    def test_a_value_outside_the_enum_is_refused(self):
        from yewo.config import coerce

        with self.assertRaises(ConfigError) as caught:
            coerce("platform", "androdi")

        self.assertIn("androdi", str(caught.exception))

    def test_a_value_inside_the_enum_is_kept(self):
        from yewo.config import coerce

        self.assertEqual(coerce("platform", "ios"), "ios")

    def test_a_number_below_its_floor_is_refused(self):
        from yewo.config import coerce

        with self.assertRaises(ConfigError):
            coerce("max_steps", 0)

    def test_values_are_still_coerced_from_their_written_form(self):
        from yewo.config import coerce

        self.assertEqual(coerce("max_steps", "25"), 25)
        self.assertIs(coerce("record", "no"), False)

    def test_a_mutable_detector_revision_is_refused(self):
        from yewo.config import coerce

        with self.assertRaises(ConfigError):
            coerce("perception_model", "hf://example/yewo-ui@main")

    def test_the_settings_are_the_ones_the_schema_declares(self):
        from yewo.config import SETTINGS
        from yewo.schema import Settings

        aliases = {"app", "emulator", "record_session"}
        self.assertEqual(SETTINGS, set(Settings.model_fields) - aliases)


class OneDeclarationTests(unittest.TestCase):
    """The file's shape is declared once, and everything else follows it.

    A second list of the same settings drifts quietly: `yewo config set`
    accepted a platform the loader then refused, and a journey file's own
    keys were reported as misspelled settings while the run used them.
    """

    def test_every_setting_has_a_default(self):
        """config.get() reads BUILT_IN, so a field without one raises KeyError."""
        from yewo.config import BUILT_IN, SETTINGS

        self.assertEqual(SETTINGS, set(BUILT_IN))

    def test_the_keys_that_are_not_settings_are_known_to_be_structure(self):
        """A journey file's own id and name are not misspelled settings."""
        from yewo.config import STRUCTURE_KEYS
        from yewo.schema import YewoFile, Settings

        structural = set(YewoFile.model_fields) - set(Settings.model_fields)
        self.assertEqual(set(STRUCTURE_KEYS), structural)

    def test_the_platform_enum_matches_the_schema(self):
        """Platform.named() and the schema must accept the same words."""
        import typing

        from yewo.agent.device.platform import Platform
        from yewo.schema import PLATFORMS
        from yewo.schema import Platform as PlatformLiteral

        self.assertEqual(set(PLATFORMS), set(typing.get_args(PlatformLiteral)))
        self.assertEqual({p.value for p in Platform}, set(PLATFORMS))

    def test_the_grounding_constants_match_the_schema(self):
        import typing

        from yewo.agent.grounding import (
            GROUNDING_DESCRIBE,
            GROUNDING_MARKS,
        )
        from yewo.schema import GROUNDINGS
        from yewo.schema import Grounding as GroundingLiteral

        self.assertEqual(set(GROUNDINGS), set(typing.get_args(GroundingLiteral)))
        self.assertEqual({GROUNDING_MARKS, GROUNDING_DESCRIBE}, set(GROUNDINGS))
        self.assertNotIn("coordinates", GROUNDINGS)

    def test_the_defaults_are_values_the_schema_allows(self):
        from yewo.config import BUILT_IN, coerce

        for name, value in BUILT_IN.items():
            if value is not None:
                coerce(name, value)

    def test_the_command_line_offers_what_the_schema_allows(self):
        """A flag choice the schema refuses fails after the run starts."""
        from yewo.schema import GROUNDINGS, PLATFORMS
        from yewo.testing.cli import YewoCLI

        parser = YewoCLI.__new__(YewoCLI)._create_parser()
        choices = {
            action.dest: set(action.choices)
            for action in parser._actions
            if action.choices
        }

        self.assertEqual(choices["platform"], set(PLATFORMS))
        self.assertEqual(choices["grounding"], set(GROUNDINGS))
