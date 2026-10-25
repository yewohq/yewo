"""Yewo selects a Node runtime that can actually execute Appium."""

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from yewo.agent.device.appium_runtime import resolve_appium_runtime


class AppiumRuntimeTests(unittest.TestCase):
    @patch("yewo.agent.device.session.resolve_appium_runtime")
    @patch("yewo.agent.device.session.urllib.request.urlopen", side_effect=OSError())
    def test_owned_appium_uses_the_resolved_node_runtime(self, _urlopen, resolve_runtime):
        from yewo.agent.device.appium_runtime import AppiumRuntime
        from yewo.agent.device.session import DeviceSession

        resolve_runtime.return_value = AppiumRuntime(
            node_binary=Path("/runtime/node"),
            appium_script=Path("/runtime/appium.js"),
            appium_version="3.5.0",
            environment={"PATH": "/runtime"},
        )
        session = DeviceSession.__new__(DeviceSession)
        session._appium_url = "http://127.0.0.1:4723"
        session._owns_appium = False
        session._appium_service = MagicMock(is_running=False)
        session._appium_service.start.side_effect = lambda **_kwargs: setattr(
            session._appium_service, "is_running", True
        )
        session._appium_args = {"args": []}

        session._start_appium()

        session._appium_service.start.assert_called_once_with(
            args=[],
            node="/runtime/node",
            main_script="/runtime/appium.js",
            env={"PATH": "/runtime"},
        )
        self.assertTrue(session._owns_appium)

    def test_a_compatible_versioned_node_is_selected_after_path_node_fails(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            active_node = root / "node"
            compatible_node = root / "node@24"
            appium = root / "appium"
            for executable in (active_node, compatible_node, appium):
                executable.touch()
            (root / "package.json").write_text("{}", encoding="utf-8")

            def checked(command, **_kwargs):
                if command[1] == "-e" and command[0] == str(active_node.resolve()):
                    return SimpleNamespace(returncode=1, stdout="", stderr="unsupported Node")
                if command[1] == "-e":
                    return SimpleNamespace(returncode=0, stdout="", stderr="")
                return SimpleNamespace(returncode=0, stdout="3.5.0\n", stderr="")

            with (
                patch(
                    "yewo.agent.device.appium_runtime._node_candidates",
                    return_value=[active_node.resolve(), compatible_node.resolve()],
                ),
                patch(
                    "yewo.agent.device.appium_runtime.shutil.which",
                    return_value=str(appium),
                ),
                patch("yewo.agent.device.appium_runtime.subprocess.run", side_effect=checked),
            ):
                runtime = resolve_appium_runtime({"PATH": str(root)})

        self.assertEqual(runtime.node_binary, compatible_node.resolve())
        self.assertEqual(runtime.appium_version, "3.5.0")
        self.assertEqual(
            Path(runtime.environment["PATH"].split(":")[0]), compatible_node.resolve().parent
        )

    def test_an_explicit_incompatible_node_fails_with_remediation(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "node"
            appium = root / "appium"
            node.touch()
            appium.touch()
            (root / "package.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "yewo.agent.device.appium_runtime.shutil.which",
                    return_value=str(appium),
                ),
                patch(
                    "yewo.agent.device.appium_runtime.subprocess.run",
                    return_value=SimpleNamespace(
                        returncode=1, stdout="", stderr="unsupported Node"
                    ),
                ),
                self.assertRaisesRegex(RuntimeError, "YEWO_NODE_BINARY"),
            ):
                resolve_appium_runtime(
                    {"PATH": str(root), "YEWO_NODE_BINARY": str(node)}
                )

    def test_a_runtime_check_timeout_is_reported_as_incompatible(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            node = root / "node"
            appium = root / "appium"
            node.touch()
            appium.touch()
            (root / "package.json").write_text("{}", encoding="utf-8")
            with (
                patch(
                    "yewo.agent.device.appium_runtime.shutil.which",
                    return_value=str(appium),
                ),
                patch(
                    "yewo.agent.device.appium_runtime.subprocess.run",
                    side_effect=subprocess.TimeoutExpired("appium", 10),
                ),
                self.assertRaisesRegex(RuntimeError, "TimeoutExpired"),
            ):
                resolve_appium_runtime(
                    {"PATH": str(root), "YEWO_NODE_BINARY": str(node)}
                )


if __name__ == "__main__":
    unittest.main()
