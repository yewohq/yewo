"""The programs on this machine that a run needs, and how it says so.

A device can be perfect and a capability still unusable, because part of it
runs on the host. Reading an iOS privacy decision back is one program away,
and without it a permission scenario is skipped rather than guessed at.

So the tool is checked where a person can act on the answer, and offered
where the rest of the install is offered. A check that reports a tool Yewo
can install must name the command that installs it.
"""

import unittest
from unittest import mock

from yewo.doctor import Status
from yewo.doctor.checks import check_applesimutils


class DoctorTests(unittest.TestCase):
    """What `yewo doctor` says about the host."""

    def _checked(self, present, macos=True, version="applesimutils version 0.9.12"):
        from yewo.doctor import checks

        with mock.patch.object(checks, "is_macos", return_value=macos), \
             mock.patch.object(
                 checks.shutil, "which", return_value="/opt/bin/x" if present else None
             ), \
             mock.patch.object(
                 checks, "_capture", return_value=mock.Mock(stdout=version)
             ):
            return check_applesimutils()

    def test_a_host_that_has_it_reports_the_version(self):
        result = self._checked(present=True)

        self.assertIs(result.status, Status.OK)
        self.assertIn("0.9.12", result.detail)

    def test_a_host_without_it_warns_rather_than_fails(self):
        """Every other iOS capability works without it."""
        result = self._checked(present=False)

        self.assertIs(result.status, Status.WARN)
        self.assertIn("skipped", result.detail)

    def test_the_warning_carries_the_command_that_fixes_it(self):
        result = self._checked(present=False)

        self.assertIn("brew tap wix/brew", result.hint)
        self.assertIn("brew install applesimutils", result.hint)

    def test_a_machine_that_cannot_run_ios_is_not_asked(self):
        result = self._checked(present=False, macos=False)

        self.assertIs(result.status, Status.SKIP)


class SetupTests(unittest.TestCase):
    """What `yewo setup` offers to do about it."""

    def _step(
        self, present, macos=True, brew=True, approved=True, dry_run=False, exit_code=0
    ):
        from yewo.setup import steps

        def which(name):
            if name == "applesimutils":
                return "/opt/bin/applesimutils" if present else None
            return "/opt/bin/brew" if brew else None

        ctx = mock.Mock(dry_run=dry_run)
        ctx.approved.return_value = approved
        with mock.patch.object(steps, "is_macos", return_value=macos), \
             mock.patch("shutil.which", side_effect=which), \
             mock.patch.object(steps, "run_command", return_value=mock.Mock(stdout="0.9.12")), \
             mock.patch.object(steps, "_run_visible", return_value=exit_code) as ran:
            return steps.install_applesimutils(ctx), ran

    def test_a_host_that_has_it_installs_nothing(self):
        result, ran = self._step(present=True)

        ran.assert_not_called()
        self.assertEqual(result.detail, "0.9.12")

    def test_the_tap_is_trusted_before_the_formula_is_installed(self):
        """Homebrew refuses a formula from an untrusted tap."""
        _, ran = self._step(present=False)

        commands = [call.args[0] for call in ran.call_args_list]
        self.assertEqual(
            commands,
            [
                ["brew", "tap", "wix/brew"],
                ["brew", "trust", "--formula", "wix/brew/applesimutils"],
                ["brew", "install", "applesimutils"],
            ],
        )

    def test_one_formula_is_trusted_rather_than_the_whole_tap(self):
        """Narrower: nothing else from that tap becomes installable unasked."""
        _, ran = self._step(present=False)

        trust = [c.args[0] for c in ran.call_args_list if "trust" in c.args[0]][0]

        self.assertIn("--formula", trust)
        self.assertIn("wix/brew/applesimutils", trust)

    def test_nothing_is_installed_without_being_asked(self):
        result, ran = self._step(present=False, approved=False)

        ran.assert_not_called()
        self.assertIn("brew install applesimutils", result.hint)

    def test_a_dry_run_says_what_it_would_do(self):
        result, ran = self._step(present=False, dry_run=True)

        ran.assert_not_called()
        self.assertIn("brew install applesimutils", result.detail)

    def test_a_step_that_fails_names_the_command_to_run_by_hand(self):
        """The tap step is the one that fails on an older Homebrew."""
        result, ran = self._step(present=False, exit_code=1)

        self.assertEqual(result.detail, "install failed")
        self.assertIn("brew tap wix/brew", result.hint)
        ran.assert_called_once()

    def test_a_host_that_cannot_run_ios_installs_nothing(self):
        result, ran = self._step(present=False, macos=False)

        ran.assert_not_called()
        self.assertIn("macOS", result.detail)

    def test_a_host_without_brew_is_told_what_to_run(self):
        result, ran = self._step(present=False, brew=False)

        ran.assert_not_called()
        self.assertIn("brew tap wix/brew", result.hint)
