"""The support table and the controller, asked the same question.

Two components decide whether a target can apply a capability. The table
decides before the device opens, so a scenario it refuses is skipped. The
controller decides at the device, so a scenario it refuses has already spent
a step.

They have to agree. A table that refuses what the controller can do skips work
the target could have done, and says nothing about it. Two iOS capabilities
were implemented and receipted this way, and reachable on no target at all.
The controller grew a driver path, and the table still described the command
line it replaced.
"""

import unittest
from typing import get_args
from unittest import mock

from fakes import EVERY_CAPABILITY, FakeDriver, controller, silent_reads

from yewo.agent.actuation.actions import (
    AppStateCapability,
    CapabilityKind,
    InterruptCapability,
    PermissionCapability,
)
from yewo.agent.device import capability_controller
from yewo.agent.device.platform import Platform
from yewo.errors import ActionRejected
from yewo.testing.judge.capabilities import (
    CapabilityTarget,
    unsupported_capability,
    unsupported_for,
)

# A variant of a kind whose support differs from the kind's own. The table
# answers these on their own, so the agreement has to be checked on them too.
PERMISSIONS = get_args(PermissionCapability.model_fields["permission"].annotation)
"""Every permission a scenario can name, read off the model that accepts them."""

EVERY_VARIANT = (
    InterruptCapability(type="incoming_call"),
    AppStateCapability(action="clear_data"),
    PermissionCapability(permission="notifications", grant=True),
    PermissionCapability(permission="storage", grant=True),
    PermissionCapability(permission="camera", grant=True),
)


class TableAgreementTests(unittest.TestCase):
    """Every capability, against both platforms, through both deciders.

    One question only: whether a path exists at all. Neither a floor nor an
    emulator rule is under test here, because the controller cannot answer
    either against a fake driver. `ReleaseBoundaryTests` covers those.
    """

    def setUp(self):
        self._reads, self.reads = silent_reads()
        self.addCleanup(self._reads.stop)

    def _controller_refuses(self, capability, platform) -> bool:
        """Whether the controller refuses this capability on this platform.

        The driver answers every method. The app is installed and the file is
        on disk. A refusal that is left is about the platform, and not about
        the scenario.
        """
        control = controller(platform=platform, driver=FakeDriver())
        with mock.patch.object(capability_controller, "AdbControl"), \
             mock.patch.object(capability_controller, "SimctlControl"):
            try:
                control.execute(capability)
            except ActionRejected:
                return True
        return False

    def _table_refuses(self, capability, platform) -> bool:
        """Whether preflight refuses it on the most capable target of this platform.

        An emulator, a release above every floor, and an unread package list.
        The only rule left to fail is whether a path exists at all. That is
        the one question the controller also answers.
        """
        target = CapabilityTarget(platform=platform, emulator=True, release=(99,))
        return unsupported_for(capability, target) is not None

    def test_the_table_refuses_what_the_controller_refuses(self):
        for platform in Platform:
            for kind, capability in EVERY_CAPABILITY.items():
                with self.subTest(platform=platform.value, kind=kind.value):
                    self.assertEqual(
                        self._table_refuses(capability, platform),
                        self._controller_refuses(capability, platform),
                    )

    def test_the_table_refuses_what_the_controller_refuses_per_variant(self):
        for platform in Platform:
            for capability in EVERY_VARIANT:
                with self.subTest(
                    platform=platform.value, variant=capability.support_variant
                ):
                    self.assertEqual(
                        self._table_refuses(capability, platform),
                        self._controller_refuses(capability, platform),
                    )


class ReleaseBoundaryTests(unittest.TestCase):
    """Each floor, read one release below it and one at it.

    A floor is a number somebody typed, and a scenario is skipped or run on
    it. Read one release either side, so an off-by-one or a lost minor
    version fails here rather than on a device.
    """

    def _target(self, release, emulator=True, platform=Platform.IOS):
        return CapabilityTarget(platform=platform, emulator=emulator, release=release)

    def _refused(self, kind, release, emulator=True, platform=Platform.IOS):
        return unsupported_capability(
            kind, self._target(release, emulator, platform)
        )

    def test_the_screen_reader_needs_the_release_that_added_it(self):
        """VoiceOver automation arrived in iOS 27."""
        self.assertIn("iOS 27", self._refused(CapabilityKind.ACCESSIBILITY, (26, 4)))
        self.assertIsNone(self._refused(CapabilityKind.ACCESSIBILITY, (27,)))

    def test_a_deep_link_needs_a_minor_release_an_integer_could_not_say(self):
        """The floor is iOS 16.4, so 16.3 is below it and 16.4 is not."""
        self.assertIn("iOS 16.4", self._refused(CapabilityKind.DEEPLINK, (16, 3)))
        self.assertIsNone(self._refused(CapabilityKind.DEEPLINK, (16, 4)))

    def test_a_memory_warning_has_a_floor_only_a_device_has_to_meet(self):
        """simctl posts one to any simulator. The driver path arrived in iOS 17."""
        self.assertIsNone(self._refused(CapabilityKind.INTERRUPT, (16,)))

        refusal = self._refused(CapabilityKind.INTERRUPT, (16,), emulator=False)
        self.assertIn("iOS 17", refusal)
        self.assertIn("physical device", refusal)

        self.assertIsNone(self._refused(CapabilityKind.INTERRUPT, (17,), emulator=False))

    def test_an_android_floor_is_counted_in_api_levels(self):
        """33 is Android 13, and naming it "iOS 33" would be a different year."""
        refusal = self._refused(
            CapabilityKind.LOCALE, (31,), platform=Platform.ANDROID
        )

        self.assertIn("API 33", refusal)
        self.assertNotIn("iOS", refusal)

    def test_a_target_of_unestablished_kind_meets_a_physical_floor(self):
        """A read that failed must not refuse work the target can do."""
        target = CapabilityTarget(platform=Platform.IOS, emulator=None, release=(16,))

        self.assertIsNone(unsupported_capability(CapabilityKind.INTERRUPT, target))

    def test_an_unread_release_meets_every_floor(self):
        target = CapabilityTarget(platform=Platform.IOS, emulator=False, release=None)

        self.assertIsNone(unsupported_capability(CapabilityKind.INTERRUPT, target))
        self.assertIsNone(unsupported_capability(CapabilityKind.DEEPLINK, target))

    def test_the_locale_no_ios_target_can_apply_is_refused_on_both(self):
        """It sets the simulator, and reaches no app that already started."""
        for emulator in (True, False):
            with self.subTest(emulator=emulator):
                refusal = self._refused(CapabilityKind.LOCALE, (18, 2), emulator)

                self.assertIn("locale", refusal)
                self.assertIn("already running", refusal)


class PermissionSupportTests(unittest.TestCase):
    """Where a permission can be set, and what the host has to have.

    Three separate rules. simctl names four of the six services. The read
    runs a program on the host. And two of the Android permissions became
    runtime permissions only in Android 13.
    """

    EVERY_TOOL = frozenset({"applesimutils"})

    def _refused(self, permission: str, platform: Platform, release=(99,), tools=None):
        target = CapabilityTarget(
            platform=platform,
            emulator=True,
            release=release,
            host_tools=self.EVERY_TOOL if tools is None else tools,
        )
        capability = PermissionCapability(permission=permission, grant=True)
        return unsupported_for(capability, target)

    def test_the_permissions_refused_on_ios_are_the_ones_simctl_lacks(self):
        from yewo.agent.device.capability_maps import IOS_PRIVACY_SERVICES

        for permission in PERMISSIONS:
            with self.subTest(permission=permission):
                self.assertEqual(
                    self._refused(permission, Platform.IOS) is not None,
                    permission not in IOS_PRIVACY_SERVICES,
                )

    def test_a_host_without_the_reader_skips_every_ios_permission(self):
        """Set and never confirmed is the shape of a false pass."""
        refusal = self._refused("camera", Platform.IOS, tools=frozenset())

        self.assertIn("applesimutils", refusal)
        self.assertIn("brew", refusal)

    def test_a_host_nothing_was_read_from_refuses_nothing(self):
        """The same rule as an unread package list, for the same reason."""
        target = CapabilityTarget(
            platform=Platform.IOS, emulator=True, release=(18, 2), host_tools=None
        )
        camera = PermissionCapability(permission="camera", grant=True)

        self.assertIsNone(unsupported_for(camera, target))

    def test_android_asks_nothing_of_the_host(self):
        self.assertIsNone(
            self._refused("camera", Platform.ANDROID, tools=frozenset())
        )

    def test_every_permission_android_names_is_allowed_there(self):
        from yewo.agent.device.capability_maps import ANDROID_PERMISSIONS

        for permission in PERMISSIONS:
            with self.subTest(permission=permission):
                self.assertEqual(
                    self._refused(permission, Platform.ANDROID) is not None,
                    permission not in ANDROID_PERMISSIONS,
                )

    def test_the_two_that_arrived_in_android_13_are_refused_below_it(self):
        """POST_NOTIFICATIONS and READ_MEDIA_IMAGES are not on API 32."""
        for permission in ("notifications", "photos"):
            with self.subTest(permission=permission):
                refusal = self._refused(permission, Platform.ANDROID, (32,))

                self.assertIn("API 33", refusal)
                self.assertIn(permission, refusal)

    def test_a_permission_that_predates_them_is_allowed_on_api_32(self):
        self.assertIsNone(self._refused("camera", Platform.ANDROID, (32,)))
