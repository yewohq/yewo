"""`yewo lock` — what it writes, what it says, and what it returns.

The command exists so generation needs no device. These cases hold it to
that: nothing here starts an emulator, installs an app or opens a driver.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fakes import walk

from yewo.cli import lock as command
from yewo.config import load_config
from yewo.testing.models import ScenarioDraft

YAML = """\
device: Pixel_8
app_path: app.apk
coverage:
  - category: functional
    types: [happy_path]
    priority: P0
journeys:
  - id: checkout
    name: Checkout
    description: A user signs in and places an order.
"""


def draft(name="Sign in"):
    return ScenarioDraft(
        scenario_name=name,
        steps=walk("Launch the app."),
        expected_outcomes=["the app opens"],
    )


class CommandTests(unittest.TestCase):
    def _run(self, directory, responses=(), refresh=False, written=YAML):
        path = Path(directory) / "yewo.yaml"
        path.write_text(written)
        client = SimpleNamespace(generate_response=Mock(side_effect=list(responses)))
        said = []
        args = SimpleNamespace(
            journey_file=str(path), model=None, workflow=None, refresh=refresh
        )

        with (
            patch.object(command, "require_lock"),
            patch("yewo.llm.llm_client.LLMClient", return_value=client),
            patch.object(command.Console, "write", lambda _self, text: said.append(text)),
        ):
            code = command.run(args, load_config())
        return code, said, client, Path(directory) / "yewo.lock"

    def test_locking_a_file_writes_the_lock_beside_it(self):
        with tempfile.TemporaryDirectory() as directory:
            code, said, client, lock_path = self._run(directory, [draft()])

            self.assertEqual(code, 0)
            self.assertTrue(lock_path.is_file())
            entries = json.loads(lock_path.read_text())["entries"]

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["journey"], "checkout")
        self.assertEqual(client.generate_response.call_count, 1)
        self.assertTrue(any("yewo.lock" in line for line in said))

    def test_lock_generation_has_one_cache_affinity_for_the_command(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "yewo.yaml"
            path.write_text(YAML)
            client = SimpleNamespace(generate_response=Mock(return_value=draft()))
            args = SimpleNamespace(
                journey_file=str(path), model=None, workflow=None, refresh=False
            )

            with (
                patch.object(command, "require_lock"),
                patch("yewo.llm.llm_client.LLMClient", return_value=client) as constructor,
                patch.object(command.Console, "write"),
            ):
                command.run(args, load_config())

        affinity = constructor.call_args.kwargs["cache_affinity"]
        self.assertTrue(affinity.startswith("lock-"))

    def test_locking_again_asks_for_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            self._run(directory, [draft()])
            code, said, client, _ = self._run(directory)

        self.assertEqual(code, 0)
        self.assertEqual(client.generate_response.call_count, 0)
        self.assertTrue(any("read from the lock" in line for line in said))

    def test_asking_for_it_all_again_regenerates(self):
        """Which is how a new model's suite is taken deliberately."""
        with tempfile.TemporaryDirectory() as directory:
            self._run(directory, [draft("First")])
            code, _, client, lock_path = self._run(
                directory, [draft("Second")], refresh=True
            )
            entries = json.loads(lock_path.read_text())["entries"]

        self.assertEqual(code, 0)
        self.assertEqual(client.generate_response.call_count, 1)
        self.assertEqual(entries[0]["scenario_name"], "Second")

    def test_a_cell_nothing_could_cover_is_reported_and_fails(self):
        refused = YAML.replace("types: [happy_path]", "types: [role]").replace(
            "category: functional", "category: security"
        )
        with tempfile.TemporaryDirectory() as directory:
            code, said, client, lock_path = self._run(directory, [], written=refused)

        self.assertEqual(code, 1)
        self.assertEqual(client.generate_response.call_count, 0)
        self.assertTrue(any("not covered" in line for line in said))
        self.assertFalse(lock_path.exists())

    def test_a_file_that_cannot_be_read_is_a_usage_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            code, said, _, _ = self._run(directory, written="journeys: not a list\n")

        self.assertEqual(code, 2)
        self.assertTrue(said)

    def test_a_file_that_is_not_there_is_a_usage_failure(self):
        args = SimpleNamespace(
            journey_file="/no/such/yewo.yaml", model=None, workflow=None, refresh=False
        )
        said = []
        with (
            patch.object(command, "require_lock"),
            patch.object(command.Console, "write", lambda _s, text: said.append(text)),
        ):
            self.assertEqual(command.run(args, load_config()), 2)

        self.assertTrue(said)


class NoDeviceTests(unittest.TestCase):
    """The whole reason the command exists.

    A device boot costs 30 to 60 seconds, and a typo in a journey file should
    cost none of it.
    """

    def test_the_lock_is_written_with_every_device_path_booby_trapped(self):
        def refuse(*_args, **_kwargs):
            raise AssertionError("locking opened a device")

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("yewo.agent.device.session.DeviceSession.__init__", refuse),
                patch("yewo.agent.device.readiness.ensure_device_ready", refuse),
            ):
                code, _, _, lock_path = CommandTests()._run(directory, [draft()])

            self.assertEqual(code, 0)
            self.assertTrue(lock_path.is_file())

    def test_it_asks_about_the_generator_model_and_nothing_else(self):
        """`require_run` also asks for a step model and a detector. Locking does not."""
        from yewo import preflight

        with patch.object(preflight, "missing_for_models", return_value=[]) as asked:
            preflight.require_lock("openrouter/m")

        self.assertEqual(asked.call_args.args, ("openrouter/m",))

    def test_a_generator_with_no_credentials_is_refused_before_anything(self):
        from yewo import preflight
        from yewo.errors import UsageError

        with patch.object(
            preflight, "missing_for_models", return_value=["Missing credentials: X"]
        ), self.assertRaises(UsageError) as raised:
            preflight.require_lock("openrouter/m")

        self.assertIn("Missing credentials", str(raised.exception))
