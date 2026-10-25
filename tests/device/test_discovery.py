"""Enumerating the targets a run could name.

A plugged-in iPhone was reachable and invisible: a session against one worked
when its udid was typed by hand, and `yewo devices` listed simulators only.
Nothing a person cannot see is a target they can choose.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

from yewo.agent.device.platform import Platform
from yewo.devices import discovery

# `xcrun xctrace list devices`, in the shape it answers. The host Mac sits in
# the same section as the phone and carries no release, which is what tells a
# target apart from the machine driving it.
XCTRACE = """== Devices ==
Studio Mac (7BB293E1-5FE9-564C-8A69-F2EAFFB24CD4)
QA iPhone (18.2) (00008140-00096DC43A52801C)

== Simulators ==
iPhone 17 Pro Max (26.3.1) (2506D933-6214-4860-BFF2-6D3193F0B801)
"""


def answered(stdout: str, code: int = 0):
    return SimpleNamespace(stdout=stdout, returncode=code)


class AttachedDeviceTests(unittest.TestCase):
    """What `xctrace` says, read as targets."""

    def _found(self, *results):
        with mock.patch.object(discovery, "is_macos", return_value=True), \
             mock.patch.object(discovery, "_capture", side_effect=list(results)):
            return discovery.ios_physical_devices()

    def test_an_attached_iphone_is_a_target(self):
        found = self._found(answered(XCTRACE))

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "QA iPhone")
        self.assertEqual(found[0].identifier, "00008140-00096DC43A52801C")
        self.assertEqual(found[0].runtime, "iOS 18.2")
        self.assertEqual(found[0].kind, "physical")
        self.assertEqual(found[0].state, discovery.RUNNING)

    def test_the_machine_running_the_suite_is_not_a_target(self):
        """The host Mac is listed beside the phone, with no release."""
        self.assertNotIn("Studio Mac", [d.name for d in self._found(answered(XCTRACE))])

    def test_the_simulators_are_left_to_simctl(self):
        """simctl reports their boot state, and this section does not."""
        self.assertNotIn(
            "iPhone 17 Pro Max", [d.name for d in self._found(answered(XCTRACE))]
        )

    def test_nothing_attached_is_not_a_failure(self):
        empty = "== Devices ==\nStudio Mac (7BB293E1-5FE9-564C-8A69-F2EAFFB24CD4)\n"
        self.assertEqual(self._found(answered(empty), None), [])

    def test_a_udid_from_the_fallback_is_a_target_without_a_name(self):
        """libimobiledevice reports a udid alone, which is still a target."""
        found = self._found(
            answered("", code=1),
            answered("00008140-00096DC43A52801C\n"),
        )

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].identifier, found[0].name)
        self.assertIsNone(found[0].runtime)

    def test_a_machine_with_neither_tool_reports_nothing(self):
        self.assertEqual(self._found(None, None), [])

    def test_nothing_is_asked_off_macos(self):
        with mock.patch.object(discovery, "is_macos", return_value=False), \
             mock.patch.object(discovery, "_capture") as capture:
            self.assertEqual(discovery.ios_physical_devices(), [])

        capture.assert_not_called()


class IosListingTests(unittest.TestCase):
    """Attached devices and simulators, in one answer."""

    def test_an_attached_device_is_listed_before_the_simulators(self):
        """It is the scarcer target, and a reader scanning the list wants it first."""
        phone = discovery.Device("QA iPhone", "ios", "physical", discovery.RUNNING)
        simulator = discovery.Device("iPhone 17", "ios", "simulator", discovery.SHUTDOWN)

        with mock.patch.object(
            discovery, "ios_physical_devices", return_value=[phone]
        ), mock.patch.object(discovery, "ios_simulators", return_value=[simulator]):
            found = discovery.ios_devices()

        self.assertEqual([d.kind for d in found], ["physical", "simulator"])

    def test_the_ios_listing_reaches_the_command(self):
        """`list_devices` is what `yewo devices` prints, so it has to carry both."""
        phone = discovery.Device("QA iPhone", "ios", "physical", discovery.RUNNING)

        with mock.patch.object(
            discovery, "ios_physical_devices", return_value=[phone]
        ), mock.patch.object(discovery, "ios_simulators", return_value=[]), \
             mock.patch.object(discovery, "android_devices", return_value=[]):
            found = discovery.list_devices()

        self.assertEqual(found, [phone])


class IdentifierResolutionTests(unittest.TestCase):
    """From the name a person typed to the id a session addresses.

    `yewo devices` listed the phone and nothing carried its udid into a
    session. XCUITest selects a real device by udid, and a name picks a
    target only while one target carries that name.
    """

    PHONE = discovery.Device(
        "QA iPhone", "ios", "physical", discovery.RUNNING, "00008140-00096DC43A5280"
    )
    SIMULATOR = discovery.Device(
        "iPhone 17", "ios", "simulator", discovery.SHUTDOWN, "2506D933-6214-4860"
    )

    def _resolved(self, name):
        with mock.patch.object(
            discovery, "ios_devices", return_value=[self.PHONE, self.SIMULATOR]
        ), mock.patch.object(discovery, "android_devices", return_value=[]):
            return discovery.identifier_for(name, "ios")

    def test_a_listed_name_resolves_to_its_own_id(self):
        self.assertEqual(self._resolved("QA iPhone"), self.PHONE.identifier)

    def test_a_simulator_resolves_too(self):
        """A udid is more precise than a name for either kind of target."""
        self.assertEqual(self._resolved("iPhone 17"), self.SIMULATOR.identifier)

    def test_an_id_typed_in_place_of_a_name_answers_with_itself(self):
        """A physical iPhone has no other name, so refusing it refuses the device."""
        self.assertEqual(self._resolved(self.PHONE.identifier), self.PHONE.identifier)

    def test_a_listing_with_no_id_is_passed_over(self):
        """An AVD that is not running has no serial, and cannot be addressed."""
        shutdown = discovery.Device("Pixel_8", "android", "avd", discovery.SHUTDOWN)

        with mock.patch.object(discovery, "ios_devices", return_value=[shutdown]), \
             mock.patch.object(discovery, "android_devices", return_value=[]):
            self.assertIsNone(discovery.identifier_for("Pixel_8", "ios"))

    def test_a_name_nothing_listed_resolves_to_nothing(self):
        self.assertIsNone(self._resolved("Somebody Else's iPhone"))


class SessionTargetingTests(unittest.TestCase):
    """What the session is told to connect to."""

    def _capabilities(self, resolved):
        from yewo.agent.device import control

        with mock.patch.object(control, "_listed_udid", return_value=resolved), \
             mock.patch.object(control.DeviceControl, "_start_session") as started, \
             mock.patch.object(control.DeviceControl, "_init_device_info"):
            control.DeviceControl(
                device_name="QA iPhone", app_path="app.ipa", platform=Platform.IOS
            )
        return started.call_args[0][0].to_capabilities()

    def test_a_resolved_target_is_addressed_by_its_udid(self):
        capabilities = self._capabilities("00008140-00096DC43A5280")

        self.assertEqual(capabilities["appium:udid"], "00008140-00096DC43A5280")
        self.assertIs(capabilities["appium:useNewWDA"], True)

    def test_a_name_that_resolves_to_nothing_still_starts_a_session(self):
        """A simulator is reachable by name, so a failed read must not stop a run."""
        capabilities = self._capabilities(None)

        self.assertNotIn("appium:udid", capabilities)
        self.assertEqual(capabilities["appium:deviceName"], "QA iPhone")


class AmbiguousNameTests(unittest.TestCase):
    """Two targets, one name, and no way to pick for someone else.

    `xctrace` lists a physical device before a simulator, so the first match
    was silently the phone. A suite would then run against hardware when the
    person who typed the name meant the simulator, or the reverse.
    """

    def _resolve(self, name, *devices):
        with mock.patch.object(discovery, "ios_devices", return_value=list(devices)), \
             mock.patch.object(discovery, "android_devices", return_value=[]):
            return discovery.identifier_for(name, "ios")

    def _twins(self):
        return (
            discovery.Device("iPhone 17", "ios", "physical", discovery.RUNNING, "PHONE"),
            discovery.Device("iPhone 17", "ios", "simulator", discovery.SHUTDOWN, "SIM"),
        )

    def test_a_name_two_targets_share_is_refused(self):
        from yewo.errors import UsageError

        with self.assertRaises(UsageError) as raised:
            self._resolve("iPhone 17", *self._twins())

        self.assertIn("names 2 targets", str(raised.exception))

    def test_the_refusal_names_both_candidates(self):
        """A person cannot choose between things they were not shown."""
        from yewo.errors import UsageError

        with self.assertRaises(UsageError) as raised:
            self._resolve("iPhone 17", *self._twins())

        message = str(raised.exception)
        self.assertIn("PHONE", message)
        self.assertIn("SIM", message)
        self.assertIn("physical", message)

    def test_an_id_picks_one_of_the_twins_without_ambiguity(self):
        """An id names one target by definition, so it answers even here."""
        self.assertEqual(self._resolve("SIM", *self._twins()), "SIM")

    def test_an_id_wins_over_a_name_that_matches_it(self):
        """Nothing stops a simulator being named after another's id."""
        confusing = (
            discovery.Device("SIM", "ios", "simulator", discovery.SHUTDOWN, "OTHER"),
            discovery.Device("iPhone 17", "ios", "physical", discovery.RUNNING, "SIM"),
        )

        self.assertEqual(self._resolve("SIM", *confusing), "SIM")


class DefaultTargetTests(unittest.TestCase):
    """What `yewo devices default` saves, and what the listing marks.

    A saved name is read on every later run. A name two iOS targets share
    would pick one of them silently, and a physical iPhone has no name but
    its udid, so the stored value has to be the one that identifies it.
    """

    PHONE = discovery.Device(
        "QA iPhone", "ios", "physical", discovery.RUNNING, "00008140-0009"
    )
    TWIN = discovery.Device(
        "QA iPhone", "ios", "simulator", discovery.SHUTDOWN, "2506D933-6214"
    )
    AVD = discovery.Device("Pixel_8", "android", "avd", discovery.RUNNING, "emulator-5554")

    def _saved(self, name, *devices):
        from yewo.cli import devices as command

        console = mock.Mock()
        with mock.patch.object(command, "list_devices", return_value=list(devices)), \
             mock.patch.object(command, "set_value", return_value="yewo.yaml") as saved:
            code = command.set_default(name, mock.Mock(), console)
        return code, (saved.call_args[0][1] if saved.call_args else None), console

    def test_an_ios_name_is_saved_as_the_udid_it_resolves_to(self):
        code, saved, _ = self._saved("QA iPhone", self.PHONE)

        self.assertEqual(code, 0)
        self.assertEqual(saved, self.PHONE.identifier)

    def test_an_android_name_is_saved_as_typed(self):
        """An AVD is booted by name, and its serial is assigned at boot."""
        code, saved, _ = self._saved("Pixel_8", self.AVD)

        self.assertEqual(code, 0)
        self.assertEqual(saved, "Pixel_8")

    def test_a_name_two_ios_targets_share_is_refused_here_too(self):
        """Refused at session time is too late: the run has already started."""
        code, saved, console = self._saved("QA iPhone", self.PHONE, self.TWIN)

        self.assertEqual(code, 1)
        self.assertIsNone(saved)
        self.assertIn("names 2 targets", str(console.result.call_args))

    def test_a_udid_names_one_target_even_among_twins(self):
        code, saved, _ = self._saved("2506D933-6214", self.PHONE, self.TWIN)

        self.assertEqual(code, 0)
        self.assertEqual(saved, "2506D933-6214")

    def test_a_name_nothing_listed_is_refused(self):
        code, saved, _ = self._saved("Nobody's iPhone", self.PHONE)

        self.assertEqual(code, 1)
        self.assertIsNone(saved)

    def _listing(self, target):
        from yewo.cli import devices as command

        config = mock.Mock()
        config.get.return_value = target
        console = mock.Mock()
        command.show([self.PHONE], config, console)
        return "\n".join(
            str(call.args[0]) for call in console.write.call_args_list if call.args
        )

    def test_a_stored_udid_is_marked_in_the_listing(self):
        """Stored canonically, the target would otherwise lose its own marker."""
        listing = self._listing(self.PHONE.identifier)

        self.assertRegex(listing, rf"\*\s+{self.PHONE.name}")

    def test_a_stored_name_is_still_marked(self):
        """A machine configured by hand, or an Android default, keeps working."""
        listing = self._listing(self.PHONE.name)

        self.assertRegex(listing, rf"\*\s+{self.PHONE.name}")

    def test_a_target_that_is_not_listed_marks_nothing(self):
        listing = self._listing("something else")

        self.assertNotIn("*", listing)


class DurableNameTests(unittest.TestCase):
    """What survives the listing, per kind of target.

    A listing name is not always one a run can use again. An Android handset
    is listed by its model, which nothing selects. A running emulator's
    serial is assigned at boot, so it is gone by the next run. Storing either
    one leaves a default that cannot be reached.
    """

    def _addressable(self, platform, kind, name, identifier):
        return discovery.addressable(
            discovery.Device(name, platform, kind, discovery.RUNNING, identifier)
        )

    def test_an_avd_is_named_by_its_own_name(self):
        """It is launched by name, and its serial is assigned at boot."""
        self.assertEqual(
            self._addressable("android", "avd", "Pixel_8", "emulator-5554"), "Pixel_8"
        )

    def test_an_android_handset_is_named_by_its_serial(self):
        """The listing shows a model name, and nothing can select that."""
        self.assertEqual(
            self._addressable("android", "physical", "Pixel 9 Pro", "38291FDH2003TQ"),
            "38291FDH2003TQ",
        )

    def test_an_iphone_is_named_by_its_udid(self):
        self.assertEqual(
            self._addressable("ios", "physical", "QA iPhone", "00008140-0009"),
            "00008140-0009",
        )

    def test_a_simulator_is_named_by_its_udid(self):
        self.assertEqual(
            self._addressable("ios", "simulator", "iPhone 17", "2506D933"), "2506D933"
        )


class ResolutionAgreementTests(unittest.TestCase):
    """One resolver, so a saved default and a session pick the same target.

    `devices default` had its own matching, which combined name and
    identifier hits. An id that happened to equal another device's name was
    then read as ambiguous and refused, where a session would have taken it.
    """

    PHONE = discovery.Device("Studio phone", "ios", "physical", discovery.RUNNING, "UDID-A")
    DECOY = discovery.Device("UDID-A", "ios", "simulator", discovery.SHUTDOWN, "UDID-B")

    def test_an_exact_id_wins_over_another_device_s_name(self):
        found = discovery.resolve("UDID-A", [self.DECOY, self.PHONE])

        self.assertIs(found, self.PHONE)

    def test_the_saved_default_agrees_with_that(self):
        from yewo.cli import devices as command

        console = mock.Mock()
        with mock.patch.object(
            command, "list_devices", return_value=[self.DECOY, self.PHONE]
        ), mock.patch.object(command, "set_value", return_value="yewo.yaml") as saved:
            code = command.set_default("UDID-A", mock.Mock(), console)

        self.assertEqual(code, 0)
        self.assertEqual(saved.call_args[0][1], "UDID-A")

    def test_a_running_avd_is_saved_by_name_not_by_its_serial(self):
        """`emulator-5554` is gone by the next run, and the name is not."""
        from yewo.cli import devices as command

        avd = discovery.Device(
            "Pixel_8", "android", "avd", discovery.RUNNING, "emulator-5554"
        )
        console = mock.Mock()
        with mock.patch.object(command, "list_devices", return_value=[avd]), \
             mock.patch.object(command, "set_value", return_value="yewo.yaml") as saved:
            command.set_default("emulator-5554", mock.Mock(), console)

        self.assertEqual(saved.call_args[0][1], "Pixel_8")

    def test_an_android_handset_is_saved_by_serial_not_by_model(self):
        from yewo.cli import devices as command

        handset = discovery.Device(
            "Pixel 9 Pro", "android", "physical", discovery.RUNNING, "38291FDH"
        )
        console = mock.Mock()
        with mock.patch.object(command, "list_devices", return_value=[handset]), \
             mock.patch.object(command, "set_value", return_value="yewo.yaml") as saved:
            command.set_default("Pixel 9 Pro", mock.Mock(), console)

        self.assertEqual(saved.call_args[0][1], "38291FDH")

    def test_an_ambiguous_target_marks_no_row(self):
        """The run would refuse it, and marking either row would claim a choice."""
        from yewo.cli import devices as command

        twins = [
            discovery.Device("iPhone 17", "ios", "physical", discovery.RUNNING, "A"),
            discovery.Device("iPhone 17", "ios", "simulator", discovery.SHUTDOWN, "B"),
        ]
        config = mock.Mock()
        config.get.return_value = "iPhone 17"
        console = mock.Mock()
        command.show(twins, config, console)
        listing = "\n".join(
            str(call.args[0]) for call in console.write.call_args_list if call.args
        )

        self.assertNotIn("*", listing)
