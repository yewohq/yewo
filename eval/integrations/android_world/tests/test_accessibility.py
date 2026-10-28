"""Accessibility-forwarder lifecycle used by AndroidWorld evaluations."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "android_world"))

from accessibility import (
    FORWARDER_COMPONENT,
    FORWARDER_PACKAGE,
    TREE_LOG_ACTION,
    ensure_forwarder_bound,
    grpc_port,
    is_forwarder_running,
    parse_enabled_services,
    restart_forwarder,
)


class AccessibilityForwarderTests(unittest.TestCase):
    def test_enabled_services_ignore_androids_null_sentinel(self):
        self.assertEqual(parse_enabled_services("null\n"), [])
        self.assertEqual(parse_enabled_services("one:two\n"), ["one", "two"])

    def test_running_state_requires_process_ids(self):
        self.assertTrue(is_forwarder_running("1234\n"))
        self.assertTrue(is_forwarder_running("1234 5678\n"))
        self.assertFalse(is_forwarder_running(""))
        self.assertFalse(is_forwarder_running("service unavailable\n"))

    def test_grpc_port_comes_from_androidworlds_no_proxy_setting(self):
        self.assertEqual(grpc_port("localhost,10.0.2.2:54321\n"), "54321")
        self.assertIsNone(grpc_port("localhost"))

    def test_restart_preserves_other_services_and_waits_for_binding(self):
        commands = []
        sleeps = []
        processes = iter(("", "1234\n"))

        def run_adb(arguments):
            commands.append(list(arguments))
            if arguments[-2:] == ["secure", "enabled_accessibility_services"]:
                return "other/service\n"
            if arguments[-2:] == ["global", "no_proxy"]:
                return "10.0.2.2:54321\n"
            if arguments[-2:] == ["pidof", FORWARDER_PACKAGE]:
                return next(processes)
            return ""

        result = restart_forwarder(
            run_adb, timeout=2, poll_interval=1, sleep=sleeps.append
        )

        self.assertTrue(result)
        self.assertTrue(
            any("other/service:" + FORWARDER_COMPONENT in command for command in commands)
        )
        self.assertEqual(
            commands.count(["shell", "pidof", FORWARDER_PACKAGE]), 2
        )
        self.assertEqual(sleeps, [1, 1])
        self.assertTrue(any(TREE_LOG_ACTION in command for command in commands))
        self.assertTrue(any("54321" in command for command in commands))

    def test_ensure_does_not_toggle_a_bound_service(self):
        commands = []

        def run_adb(arguments):
            commands.append(list(arguments))
            return "1234\n"

        result = ensure_forwarder_bound(run_adb)

        self.assertTrue(result)
        self.assertEqual(commands, [["shell", "pidof", FORWARDER_PACKAGE]])

    def test_failed_recovery_is_non_fatal(self):
        def run_adb(arguments):
            if arguments[-2:] == ["secure", "enabled_accessibility_services"]:
                return f"{FORWARDER_COMPONENT}\n"
            return ""

        result = restart_forwarder(
            run_adb, timeout=1, poll_interval=1, sleep=lambda _: None
        )

        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
