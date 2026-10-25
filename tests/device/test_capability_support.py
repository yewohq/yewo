"""Deciding whether a target can give a scenario what it declared.

Support is not a property of the platform alone. The emulator console, the API
level and the installed tools all decide it, and a scenario that needs one this
target lacks is skipped before it spends a step finding out.
"""

import unittest

from fakes import device_result, walk
from types import SimpleNamespace
from unittest import mock

from yewo.agent.actuation.actions import (
    AppStateCapability,
    CapabilityKind,
    ClipboardCapability,
    InterruptCapability,
    NetworkCapability,
    PermissionCapability,
)
from yewo.agent.device.platform import Platform

SERIAL = "emulator-5554"


class PreflightTests(unittest.TestCase):
    """What a scenario declares, read against the target in front of it."""

    def _case(self, *required, outcomes=("a banner appears",), steps=None):
        from yewo.testing.models import GeneratedTestCase

        return GeneratedTestCase(
            test_id="TS_1",
            scenario_name="S",
            cell="input.negative.0",
            category="input",
            scenario_type="negative",
            priority="P1",
            steps=steps or walk("Open the app."),
            expected_outcomes=list(outcomes),
            required_capabilities=list(required),
        )

    def _asking_for(self, capability):
        """A scenario whose steps ask for this capability, as a run would.

        A declared kind is the fallback. The step is where the capability
        itself lives, and the variant with it.
        """
        from yewo.testing.steps import CapabilityStep

        return self._case(steps=[CapabilityStep(capability=capability)])

    def _target(self, platform=Platform.ANDROID, **kwargs):
        from yewo.testing.judge.capabilities import CapabilityTarget

        return CapabilityTarget(platform=platform, **{"emulator": True, **kwargs})

    def _reason(self, case, target):
        from yewo.testing.judge.capabilities import unsupported_reason

        return unsupported_reason(case, target)

    def test_a_capability_the_target_can_apply_is_no_obstacle(self):
        self.assertIsNone(self._reason(self._case(CapabilityKind.NETWORK), self._target()))

    def test_an_initial_capability_must_have_a_restoration_path(self):
        case = self._case()
        case.initial_capabilities = [ClipboardCapability(text="temporary")]

        reason = self._reason(case, self._target())

        self.assertIn("initial precondition", reason)
        self.assertIn("cannot be restored", reason)

    def test_android_retained_data_refuses_a_permission_precondition(self):
        case = self._case()
        case.setup = "relaunch"
        case.initial_capabilities = [PermissionCapability(permission="camera", grant=False)]

        reason = self._reason(case, self._target())

        self.assertIn("previous prompt decision", reason)

    def test_a_capability_the_platform_lacks_skips_the_scenario(self):
        reason = self._reason(self._case(CapabilityKind.NETWORK), self._target(Platform.IOS))

        self.assertIn("network", reason)
        self.assertIn("ios", reason)

    def test_a_capability_needing_an_emulator_names_the_target(self):
        """Mock location goes through the emulator console, which a phone lacks."""
        reason = self._reason(self._case(CapabilityKind.LOCATION), self._target(emulator=False))

        self.assertIn("emulator", reason)
        self.assertIn("physical device", reason)

    def test_a_release_below_the_floor_skips_the_scenario(self):
        reason = self._reason(self._case(CapabilityKind.LOCALE), self._target(release=(31,)))

        self.assertIn("API 33", reason)

    def test_a_release_at_the_floor_is_enough(self):
        self.assertIsNone(
            self._reason(self._case(CapabilityKind.LOCALE), self._target(release=(33,)))
        )

    def test_an_unknown_release_refuses_nothing(self):
        """A read that failed must not skip work the target can do."""
        self.assertIsNone(
            self._reason(self._case(CapabilityKind.LOCALE), self._target(release=None))
        )

    def test_a_missing_tool_skips_the_scenario(self):
        """Only switching the screen reader on needs it installed."""
        from yewo.agent.actuation.actions import AccessibilityCapability

        reason = self._reason(
            self._asking_for(AccessibilityCapability(enabled=True)),
            self._target(packages=frozenset({"com.example.app"})),
        )

        self.assertIn("talkback", reason.lower())

    def test_switching_a_missing_tool_off_asks_nothing_of_it(self):
        """A device without the screen reader already has it off."""
        from yewo.agent.actuation.actions import AccessibilityCapability

        self.assertIsNone(
            self._reason(
                self._asking_for(AccessibilityCapability(enabled=False)),
                self._target(packages=frozenset({"com.example.app"})),
            )
        )

    def test_an_unknown_package_list_refuses_nothing(self):
        self.assertIsNone(
            self._reason(self._case(CapabilityKind.ACCESSIBILITY), self._target(packages=None))
        )

    def test_a_variant_a_step_asks_for_is_read_off_the_step(self):
        """A kind cannot say which variant a scenario wants. The step can.

        Placing a call needs an emulator console, and a low-memory warning
        needs nothing. Both are `interrupt`, so reading the kind alone lets
        one of them reach a phone and fail there.
        """
        reason = self._reason(
            self._asking_for(InterruptCapability(type="incoming_call")),
            self._target(emulator=False),
        )

        self.assertIn("incoming_call", reason)
        self.assertIn("emulator", reason)

    def test_the_other_variant_of_that_kind_is_no_obstacle(self):
        self.assertIsNone(
            self._reason(
                self._asking_for(InterruptCapability(type="low_memory")),
                self._target(emulator=False),
            )
        )

    def test_a_capability_a_step_asks_for_is_refused_once(self):
        """The kind is derived from the steps, so it is declared as well.

        Reported twice, the same obstacle would read as two.
        """
        reason = self._reason(
            self._asking_for(NetworkCapability(mode="offline")),
            self._target(Platform.IOS),
        )

        self.assertEqual(reason.count("network cannot be set"), 1)

    def test_a_capability_with_no_row_for_the_platform_is_refused(self):
        """An absent row is how two implemented iOS capabilities were skipped."""
        from yewo.testing.judge.capabilities import (
            CAPABILITY_SUPPORT,
            CapabilitySupport,
            unsupported_capability,
        )

        with mock.patch.dict(
            CAPABILITY_SUPPORT,
            {CapabilityKind.CLIPBOARD: {Platform.ANDROID: CapabilitySupport()}},
        ):
            refusal = unsupported_capability(CapabilityKind.CLIPBOARD, self._target(Platform.IOS))

        self.assertIn("clipboard is not available on ios", refusal)

    def test_declaring_nothing_leaves_the_prose_check_alone(self):
        """A step is typed now, so the prose left to read is the outcome."""
        reason = self._reason(
            self._case(outcomes=("Network inspection shows no plaintext.",)),
            self._target(),
        )

        self.assertIn("network traffic capture", reason)

    def test_a_caller_with_no_target_checks_only_the_prose(self):
        """The default cannot refuse a scenario it knows nothing about."""
        from yewo.testing.judge.capabilities import unsupported_reason

        self.assertIsNone(unsupported_reason(self._case(CapabilityKind.NETWORK)))

    def test_every_capability_states_a_verdict_on_every_platform(self):
        """A platform with no row is refused the capability by that absence.

        One row per platform makes a capability say what it can do on each,
        rather than inherit a refusal from a line nobody wrote. Two iOS
        capabilities were implemented and refused this way.
        """
        from yewo.testing.judge.capabilities import CAPABILITY_SUPPORT

        self.assertEqual(set(CAPABILITY_SUPPORT), set(CapabilityKind))
        for kind in CapabilityKind:
            with self.subTest(kind=kind.value):
                self.assertEqual(set(CAPABILITY_SUPPORT[kind]), set(Platform))


class TargetReadingTests(unittest.TestCase):
    """Identifying the target without a driver session."""

    def _read(self, avd, api="33", packages=frozenset()):
        device = SimpleNamespace(platform=Platform.ANDROID, serial=SERIAL, os_version="14")
        adb = mock.patch("yewo.agent.device.adb_control.AdbControl.avd_name", return_value=avd)
        installed = mock.patch(
            "yewo.agent.device.adb_control.AdbControl.installed_packages",
            return_value=packages,
        )
        sdk = mock.patch("yewo.agent.device.adb_control.AdbControl.ask", return_value=api)
        from yewo.testing.judge.capabilities import target_of

        with adb, installed, sdk:
            return target_of(device)

    def test_a_device_started_from_an_avd_is_an_emulator(self):
        self.assertTrue(self._read(avd="Pixel_8").emulator)

    def test_a_device_with_no_avd_name_is_physical(self):
        self.assertFalse(self._read(avd=None).emulator)

    def test_the_api_level_is_read_as_the_release(self):
        """CapabilitySupport is keyed on the API level, which is the number adb reports."""
        self.assertEqual(self._read(avd="Pixel_8", api="36").release, (36,))

    def test_an_api_level_that_is_not_a_number_is_unknown(self):
        self.assertIsNone(self._read(avd="Pixel_8", api="null").release)

    def test_a_target_with_no_serial_reads_nothing_off_it(self):
        from yewo.testing.judge.capabilities import target_of

        target = target_of(SimpleNamespace(platform=Platform.ANDROID, serial="", os_version=None))

        self.assertIsNone(target.release)
        self.assertIsNone(target.packages)


class PhysicalDeviceTests(unittest.TestCase):
    """A real iPhone has fewer of these than a simulator, and says which.

    Before this, every iOS target was read as a simulator, so capabilities a
    physical device cannot apply preflighted as supported. A scenario
    declaring one was run rather than skipped, and it failed inside the run.
    """

    PHYSICAL = "00008140-00096DC43A52801C"
    SIMULATED = "8B59D40E-33B7-40EC-AB5D-50FFA0AB306E"

    def _target(self, listed, os_version="18.2"):
        from yewo.testing.judge.capabilities import target_of

        with mock.patch(
            "yewo.agent.device.simctl_control.SimctlControl.ask",
            return_value=listed,
        ):
            return target_of(
                SimpleNamespace(
                    platform=Platform.IOS,
                    serial=self.PHYSICAL,
                    os_version=os_version,
                )
            )

    def _allowed(self, target):
        from yewo.testing.judge.capabilities import unsupported_capability

        return {
            kind.value for kind in CapabilityKind if unsupported_capability(kind, target) is None
        }

    def test_a_udid_simctl_does_not_list_is_a_physical_device(self):
        """simctl enumerates simulators and nothing else."""
        target = self._target(f"-- iOS 18.2 --\n  iPhone 15 ({self.SIMULATED}) (Booted)")

        self.assertFalse(target.emulator)
        self.assertIn("physical", target.describe())

    def test_a_udid_simctl_lists_is_a_simulator(self):
        target = self._target(f"-- iOS 18.2 --\n  iPhone 15 ({self.PHYSICAL}) (Booted)")

        self.assertTrue(target.emulator)

    def test_a_physical_iphone_allows_only_what_the_driver_reaches(self):
        """Each of these four is a driver call rather than a simctl one.

        A simctl command needs a simulator. A command the driver proxies to
        the test runner on the device does not, so refusing it here would skip
        work this target can do.
        """
        target = self._target("-- iOS 18.2 --\n  (none)")

        self.assertEqual(
            self._allowed(target),
            {"clipboard", "deeplink", "app_state", "interrupt", "location"},
        )

    def test_the_release_a_capability_needs_is_read_off_the_driver(self):
        """VoiceOver arrived in iOS 27, and this target reports 18.2."""
        from yewo.testing.judge.capabilities import unsupported_capability

        refusal = unsupported_capability(CapabilityKind.ACCESSIBILITY, self._target("  (none)"))

        self.assertIn("iOS 27", refusal)
        self.assertIn("iOS 18", refusal)

    def test_a_release_nothing_reported_refuses_nothing(self):
        """The same rule as an unread API level, for the same reason."""
        from yewo.testing.judge.capabilities import unsupported_capability

        self.assertIsNone(
            unsupported_capability(
                CapabilityKind.ACCESSIBILITY, self._target("  (none)", os_version=None)
            )
        )

    def test_a_simulator_still_allows_what_it_can_do(self):
        target = self._target(f"  iPhone ({self.PHYSICAL}) (Booted)")

        self.assertIn("location", self._allowed(target))
        self.assertIn("biometric", self._allowed(target))

    def test_a_variant_the_target_cannot_apply_is_refused_on_its_own(self):
        """A kind this target supports does not clear every variant of it.

        Clearing app data reads the simulator's data container, and stopping
        the app is a driver call. Both are `app_state`, and only one of them
        works on a phone.
        """
        from yewo.testing.judge.capabilities import unsupported_for

        target = self._target("  (none)")

        self.assertIsNone(unsupported_for(AppStateCapability(action="force_stop"), target))
        refusal = unsupported_for(AppStateCapability(action="clear_data"), target)
        self.assertIn("clear_data", refusal)
        self.assertIn("emulator", refusal)

    def test_a_call_no_ios_target_can_place_is_refused_on_either_kind(self):
        from yewo.testing.judge.capabilities import unsupported_for

        for listed in ("  (none)", f"  iPhone ({self.PHYSICAL}) (Booted)"):
            with self.subTest(listed=listed):
                refusal = unsupported_for(
                    InterruptCapability(type="incoming_call"), self._target(listed)
                )

                self.assertIn("incoming_call", refusal)

    def test_a_low_memory_warning_is_allowed_on_either_kind(self):
        """The driver sends one to a phone and simctl posts one to a simulator."""
        from yewo.testing.judge.capabilities import unsupported_for

        for listed in ("  (none)", f"  iPhone ({self.PHYSICAL}) (Booted)"):
            with self.subTest(listed=listed):
                self.assertIsNone(
                    unsupported_for(InterruptCapability(type="low_memory"), self._target(listed))
                )

    def test_a_simctl_that_will_not_answer_refuses_nothing(self):
        """An unestablished kind refuses nothing, as an unread release does."""
        from yewo.testing.judge.capabilities import unsupported_capability

        target = self._target(None)

        self.assertIsNone(target.emulator)
        self.assertIn("unestablished", target.describe())
        self.assertIsNone(unsupported_capability(CapabilityKind.LOCATION, target))


class DeclaredCapabilityTests(unittest.TestCase):
    """A scenario declares the state it needs, and the run has to reach it.

    Preflight answers whether this target could give a capability. Whether the
    run did is a different question, and without it a scenario can declare
    `network`, stay online for every step, and pass on outcomes that were never
    about the state it asked for.
    """

    def _receipt(self, capability, observed, requested="on"):
        from yewo.agent.device.receipts import Receipt

        return Receipt(
            capability=capability,
            requested=requested,
            observed=observed,
            target=SERIAL,
            source="mobile: getConnectivity",
        )

    def _missing(self, required, receipts):
        from yewo.testing.judge.capabilities import unconfirmed_capabilities

        return unconfirmed_capabilities(required, receipts)

    def test_a_capability_the_run_never_applied_is_named(self):
        missing = self._missing([CapabilityKind.NETWORK], [])

        self.assertEqual(missing, ["network (never applied)"])

    def test_a_capability_applied_and_confirmed_is_not_named(self):
        missing = self._missing([CapabilityKind.NETWORK], [self._receipt("network", "on")])

        self.assertEqual(missing, [])

    def test_a_capability_applied_but_never_confirmed_is_named(self):
        """Unreadable is safe only while something later notices it."""
        missing = self._missing([CapabilityKind.NETWORK], [self._receipt("network", None)])

        self.assertEqual(missing, ["network (applied but never confirmed)"])

    def test_another_capability_s_receipt_does_not_answer_for_this_one(self):
        missing = self._missing([CapabilityKind.NETWORK], [self._receipt("battery", "42", "42")])

        self.assertEqual(missing, ["network (never applied)"])

    def test_declaring_nothing_requires_nothing(self):
        self.assertEqual(self._missing([], []), [])


class DeclaredCapabilityGateTests(unittest.TestCase):
    """The gate refuses a scenario whose declared state was never confirmed."""

    def _status(self, required, receipts):
        from yewo.agent.device.receipts import Receipt
        from yewo.testing.qa_agent import QAAgent

        made = [
            Receipt(
                capability=name,
                requested="on",
                observed=observed,
                target=SERIAL,
                source="s",
            )
            for name, observed in receipts
        ]
        return QAAgent.__new__(QAAgent)._determine_status(
            run_result=device_result(capability_receipts=made),
            stop_reason=None,
            all_verifications=[],
            all_issues=[],
            qa_outputs=[SimpleNamespace(is_scenario_feasible=True, infeasibility_reason=None)],
            assessments=[SimpleNamespace(rule_verifications=[], detected_issues=[])],
            required_outcomes=[],
            security_rules=[],
            required_capabilities=required,
        )

    def test_a_scenario_that_never_went_offline_cannot_pass(self):
        status, reason, cause = self._status([CapabilityKind.NETWORK], [])

        self.assertEqual(status, "UNVERIFIED")
        self.assertIn("network", reason)
        self.assertIn("never applied", reason)

    def test_a_scenario_whose_state_was_confirmed_is_not_held_back_by_it(self):
        """Whatever else the scenario still needs, the capability is settled."""
        _, reason, _ = self._status([CapabilityKind.NETWORK], [("network", "on")])

        self.assertNotIn("network", reason or "")
