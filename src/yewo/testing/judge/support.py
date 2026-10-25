"""What each platform can do, capability by capability.

Knowledge rather than logic: two tables and the shape of a row. Support is
not a property of the platform alone. A row can also name an emulator, a
release floor, a package on the device, or a program on the host.

A platform absent from a capability's entry cannot apply it at all. The
controller refuses the same cases at the device, and
`test_the_table_refuses_what_the_controller_refuses` is what holds the two
to each other.
"""

from dataclasses import dataclass
from typing import Optional

from yewo.agent.actuation.actions import CapabilityKind
from yewo.agent.device.capability_maps import TALKBACK_PACKAGE
from yewo.agent.device.platform import Platform

HOST_TOOLS = ("applesimutils",)
"""Every program on this machine that a capability can need.

Named here rather than looked for on demand, so one read answers for every
capability and a tool nothing needs is never searched for.
"""


@dataclass(frozen=True)
class CapabilitySupport:
    """What one capability needs of a target on one platform.

    `available` is false for a capability the platform offers no working path
    to at all. Spelled out rather than left out of the table, so the reason
    travels with the refusal.
    """

    available: bool = True
    emulator_only: bool = False
    physical_only: bool = False
    min_release: Optional[tuple[int, ...]] = None
    needs_package: Optional[str] = None
    needs_host_tool: Optional[str] = None
    """A program this machine has to have, named as it is installed.

    The device can be perfect and the capability still unusable, because the
    read runs on the host. Kept beside `needs_package` because the two ask
    the same question of two different machines.
    """

    detail: str = ""

    physical_min_release: Optional[tuple[int, ...]] = None
    """A floor that only a physical target has to meet.

    One capability can reach both kinds of target by two paths with two
    floors. A low-memory warning goes through simctl on a simulator, which
    has no floor, and through the driver on a device, which arrived in iOS 17.
    """

    physical_detail: str = ""
    """Why the physical floor exists, because it is not the reason above."""


# A platform absent from a capability's entry cannot apply it at all. The
# controller refuses the same cases at the device. This is that knowledge
# early enough to skip a scenario instead of failing a step inside it.
CAPABILITY_SUPPORT: dict[CapabilityKind, dict[Platform, CapabilitySupport]] = {
    CapabilityKind.LOCATION: {
        Platform.ANDROID: CapabilitySupport(
            emulator_only=True, detail="'adb emu geo fix' needs an emulator console"
        ),
        Platform.IOS: CapabilitySupport(
            min_release=(16, 4),
            detail=(
                "simulating a location through the driver arrived in iOS 16.4, "
                "and simctl is the simulator's fallback for it"
            ),
        ),
    },
    CapabilityKind.NETWORK: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(),
    },
    CapabilityKind.BATTERY: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(
            available=False,
            detail=(
                "'simctl status_bar' overrides what the status bar draws and "
                "nothing else. Its clock override leaves the system clock "
                "alone in the same way, so no app reads the level it was given"
            ),
        ),
    },
    CapabilityKind.TIMEZONE: {
        Platform.ANDROID: CapabilitySupport(
            available=False,
            detail=(
                "the system zone is persist.sys.timezone, which the shell user "
                "cannot set, and a write to 'settings global timezone' changes "
                "nothing the device reads"
            ),
        ),
        Platform.IOS: CapabilitySupport(
            available=False, detail="simctl sets no system zone at all"
        ),
    },
    CapabilityKind.LOCALE: {
        Platform.ANDROID: CapabilitySupport(
            min_release=(33,), detail="per-app locales arrived in Android 13"
        ),
        Platform.IOS: CapabilitySupport(
            available=False,
            detail=(
                "'mobile: configureLocalization' needs a booted simulator. It "
                "reaches no app that is already running, so the app under test "
                "keeps the locale it launched with"
            ),
        ),
    },
    CapabilityKind.PERMISSION: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(
            emulator_only=True,
            needs_host_tool="applesimutils",
            detail=(
                "'simctl privacy' needs a simulator, and reading the decision "
                "back needs applesimutils. Install it with: brew tap wix/brew "
                "&& brew install applesimutils. Note that simctl warns a "
                "permission change can terminate a running app"
            ),
        ),
    },
    CapabilityKind.INTERRUPT: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(
            physical_min_release=(17,),
            physical_detail=(
                "'mobile: sendMemoryWarning' is the only path to a device, "
                "and it arrived in iOS 17 with Xcode 15"
            ),
        ),
    },
    CapabilityKind.BIOMETRIC: {
        Platform.ANDROID: CapabilitySupport(
            emulator_only=True, detail="a fingerprint can only be injected into an emulator"
        ),
        Platform.IOS: CapabilitySupport(
            emulator_only=True, detail="biometric enrolment needs a simulator"
        ),
    },
    CapabilityKind.CLIPBOARD: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(),
    },
    CapabilityKind.DEEPLINK: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(
            min_release=(16, 4),
            detail="opening a URL through the driver arrived in iOS 16.4",
        ),
    },
    CapabilityKind.APP_STATE: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(),
    },
    CapabilityKind.ORIENTATION: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(),
    },
    CapabilityKind.DEVICE_LOCK: {
        Platform.ANDROID: CapabilitySupport(
            detail="lock/unlock requires a target without an unknown device credential"
        ),
        Platform.IOS: CapabilitySupport(
            detail="lock/unlock supports a simple lock screen without a passcode"
        ),
    },
    CapabilityKind.MEDIA_INJECT: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(
            emulator_only=True, detail="'simctl addmedia' needs a simulator"
        ),
    },
    CapabilityKind.ACCESSIBILITY: {
        Platform.ANDROID: CapabilitySupport(),
        Platform.IOS: CapabilitySupport(
            min_release=(27,), detail="VoiceOver automation arrived in iOS 27"
        ),
    },
}


# A variant one target answers and another does not. Every capability absent
# from here is decided by its kind alone.
VARIANT_SUPPORT: dict[tuple[CapabilityKind, str], dict[Platform, CapabilitySupport]] = {
    **{
        (CapabilityKind.NETWORK, mode): {
            Platform.IOS: CapabilitySupport(
                available=False,
                detail=(
                    "iOS does not expose radio toggles to automation. Simulator "
                    "networking also shares the host connection"
                ),
            )
        }
        for mode in ("offline", "online", "wifi_only", "data_only", "airplane")
    },
    **{
        (CapabilityKind.NETWORK, mode): {
            Platform.ANDROID: CapabilitySupport(
                emulator_only=True,
                detail="network speed and latency shaping needs an emulator console",
            ),
            Platform.IOS: CapabilitySupport(
                physical_only=True,
                min_release=(13,),
                detail=(
                    "Xcode condition inducers require a physical device running "
                    "iOS 13 or later. iOS simulators share the host network"
                ),
            ),
        }
        for mode in ("slow_2g", "slow_3g", "high_latency")
    },
    (CapabilityKind.INTERRUPT, "incoming_call"): {
        Platform.ANDROID: CapabilitySupport(
            emulator_only=True, detail="placing a call needs an emulator console"
        ),
        Platform.IOS: CapabilitySupport(
            available=False,
            detail="neither simctl nor the driver places a call into a device",
        ),
    },
    (CapabilityKind.PERMISSION, "notifications"): {
        Platform.ANDROID: CapabilitySupport(
            min_release=(33,),
            detail="POST_NOTIFICATIONS is a runtime permission from Android 13",
        ),
        Platform.IOS: CapabilitySupport(
            available=False,
            detail=(
                "simctl has no notification service, and the service that "
                "would cover it grants every other permission too"
            ),
        ),
    },
    # Every other service answers yes, no or unset. Location answers always,
    # inuse, never or unset, across two simctl services, so neither the write
    # nor the way back is expressible as a decision to allow or deny.
    (CapabilityKind.PERMISSION, "location"): {
        Platform.IOS: CapabilitySupport(
            available=False,
            detail=(
                "an iOS location decision is always, inuse, never or unset, "
                "which a grant or a revoke cannot express or put back"
            ),
        ),
    },
    (CapabilityKind.PERMISSION, "storage"): {
        Platform.IOS: CapabilitySupport(
            available=False,
            detail="simctl has no service for file or media storage",
        ),
    },
    (CapabilityKind.PERMISSION, "photos"): {
        Platform.ANDROID: CapabilitySupport(
            min_release=(33,),
            detail="READ_MEDIA_IMAGES replaced the storage permission in Android 13",
        ),
    },
    # Only switching the screen reader on needs it installed. Switching it
    # off is already true of a device that does not have it.
    (CapabilityKind.ACCESSIBILITY, "on"): {
        Platform.ANDROID: CapabilitySupport(
            needs_package=TALKBACK_PACKAGE,
            detail="the screen reader must be installed to be switched on",
        ),
    },
    (CapabilityKind.APP_STATE, "clear_data"): {
        Platform.IOS: CapabilitySupport(
            emulator_only=True,
            detail="a fresh app installation provides a clean simulator container",
        ),
    },
}
