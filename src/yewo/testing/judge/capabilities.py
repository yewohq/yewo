"""What a scenario needs, and whether this target can give it.

Two kinds of refusal live here, and both happen before a scenario spends a
step finding out.

The first reads the scenario's prose for evidence Yewo does not collect. The
second reads the capabilities the scenario declared against the target in
front of it. Support is not a property of the platform alone. `adb emu geo
fix` needs an emulator console, `cmd locale` arrived in API 33, and a
screen-reader toggle needs the screen reader installed.
"""

import logging
import shutil
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

from yewo.agent.actuation.actions import Capability, CapabilityKind
from yewo.agent.device.receipts import Receipt
from yewo.agent.device.platform import Platform
from yewo.testing.judge.support import (
    CAPABILITY_SUPPORT,
    HOST_TOOLS,
    VARIANT_SUPPORT,
    CapabilitySupport,
)
from yewo.testing.models import GeneratedTestCase
from yewo.testing.steps import capabilities_in

logger = logging.getLogger(__name__)




_UNSUPPORTED_REQUIREMENTS = {
    "network inspection": "network traffic capture is not configured",
    "https": "HTTP transport inspection is not configured",
    "http header": "HTTP request inspection is not configured",
    "authorization header": "HTTP request inspection is not configured",
    "accessibility tree": "semantic accessibility-tree inspection is not configured",
    "accessible label": "semantic accessibility-tree inspection is not configured",
    "accessible name": "semantic accessibility-tree inspection is not configured",
    "screen reader": "screen-reader verification is not configured",
    "focus order": "keyboard/screen-reader focus traversal is not configured",
    "contrast ratio": "deterministic contrast measurement is not configured",
    "wcag": "deterministic contrast measurement is not configured",
    "colour contrast": "deterministic contrast measurement is not configured",
    "color contrast": "deterministic contrast measurement is not configured",
}


@dataclass(frozen=True)
class CapabilityTarget:
    """The device a scenario would run on, as far as support is concerned.

    Every field but the platform is optional, because a caller may not know
    it. Unknown never refuses a scenario. A requirement nothing can confirm is
    reported as met, so a missing read fails closed on the refusal and not
    on the run.
    """

    platform: Platform
    emulator: Optional[bool] = None
    """Whether this is an emulator, or None where nothing established which.

    Three states and not two. A read that failed is not a physical device,
    and reporting one as the other hides that the kind was assumed.
    """

    release: Optional[tuple[int, ...]] = None
    """The version this platform gates a feature on, as its own numbers.

    Android counts API levels and Apple counts releases, so this is 34 on one
    and 18.2 on the other. Held as a tuple because a floor can be 16.4.
    Comparing tuples puts 16.0 below that floor and 18.2 above it, with no
    rule about how many parts a version has.
    """

    packages: Optional[frozenset[str]] = None

    host_tools: Optional[frozenset[str]] = None
    """The tools on this machine that a capability can need, or None.

    Support is not only a property of the device. Reading an iOS permission
    back needs a program on the host. A run on a machine without it can set
    the state and never confirm it.
    """

    def describe(self) -> str:
        if self.emulator is None:
            kind = "device of unestablished kind"
        else:
            kind = "emulator" if self.emulator else "physical device"
        version = f" {_version(self.platform, self.release)}" if self.release else ""
        return f"{self.platform.value} {kind}{version}"


def _version(platform: Platform, release: tuple[int, ...]) -> str:
    """How a platform names a version, because the numbers do not compare.

    Android counts API levels and Apple counts releases. 27 is Android 8.1 on
    one and iOS 27 on the other, so the number travels with its label.
    """
    spelled = ".".join(str(part) for part in release)
    return f"API {spelled}" if platform == Platform.ANDROID else f"iOS {spelled}"


def _below(target: CapabilityTarget, floor: tuple[int, ...]) -> bool:
    """Whether this target's release is under a floor.

    A release nothing read is under nothing. A missing read must not refuse
    work the target can do.
    """
    return target.release is not None and target.release < floor


def _refusal(
    name: str, support: CapabilitySupport, target: CapabilityTarget
) -> Optional[str]:
    """Why this target fails one support rule, or None when it meets them all.

    The kind table and the variant table are read through here, so a refusal
    reads the same whichever one produced it.
    """
    if not support.available:
        return f"{name} cannot be set on {target.platform.value}: {support.detail}"
    if support.emulator_only and target.emulator is False:
        return f"{name} needs an emulator: {support.detail}"
    floor = support.min_release
    if floor is not None and _below(target, floor):
        return (
            f"{name} needs {_version(target.platform, floor)} and "
            f"{target.describe()} is below it: {support.detail}"
        )
    physical = support.physical_min_release
    if physical is not None and target.emulator is False and _below(target, physical):
        return (
            f"{name} needs {_version(target.platform, physical)} on a physical "
            f"device, and {target.describe()} is below it: "
            f"{support.physical_detail}"
        )
    if (
        support.needs_package
        and target.packages is not None
        and support.needs_package not in target.packages
    ):
        return f"{name} needs {support.needs_package} installed: {support.detail}"
    if (
        support.needs_host_tool
        and target.host_tools is not None
        and support.needs_host_tool not in target.host_tools
    ):
        return (
            f"{name} needs {support.needs_host_tool} on this machine: "
            f"{support.detail}"
        )
    return None


def unsupported_capability(
    kind: CapabilityKind, target: CapabilityTarget
) -> Optional[str]:
    """Why this target cannot apply this capability, or None when it can.

    Reads the kind and not the variant, so a caller holding the capability
    itself should ask `unsupported_for` instead.
    """
    support = CAPABILITY_SUPPORT.get(kind, {}).get(target.platform)
    if support is None:
        return f"{kind.value} is not available on {target.platform.value}"
    return _refusal(kind.value, support, target)


def _variant_refusal(
    capability: Capability, target: CapabilityTarget
) -> Optional[str]:
    """Why this target cannot apply this variant of the capability.

    A variant with no row is governed by its kind. Absence here refuses
    nothing, unlike absence from `CAPABILITY_SUPPORT`, where a platform with
    no row has no path to the capability at all.
    """
    variant = capability.support_variant
    if variant is None:
        return None
    support = VARIANT_SUPPORT.get((capability.kind, variant), {}).get(target.platform)
    if support is None:
        return None
    return _refusal(f"{capability.kind.value} '{variant}'", support, target)


def unsupported_for(
    capability: Capability, target: CapabilityTarget
) -> Optional[str]:
    """Why this target cannot apply this capability, variant included.

    The kind is read first. A kind this target cannot apply at all refuses
    every variant of it, and naming the kind is the more useful reason.
    """
    return unsupported_capability(capability.kind, target) or _variant_refusal(
        capability, target
    )


def unsupported_reason(
    test_case: GeneratedTestCase, target: Optional[CapabilityTarget] = None
) -> str | None:
    """Why a scenario cannot be evidenced here, or None when it can.

    `target` is optional and defaults to reading nothing about the device. A
    caller that does not know the target then keeps the prose rules it had,
    rather than refusing every scenario that declares a capability.

    The steps are read before the declared kinds, because a step carries the
    capability and a kind does not. `interrupt` is one kind and two variants,
    and only one of them needs an emulator console.
    """
    text = "\n".join(
        [
            *test_case.steps_instructions,
            *(outcome.statement for outcome in test_case.expected_outcomes),
        ]
    ).lower()
    reasons = {
        reason for phrase, reason in _UNSUPPORTED_REQUIREMENTS.items() if phrase in text
    }

    if target is not None:
        asked = set()
        for capability in capabilities_in(test_case.steps):
            asked.add(capability.kind)
            refusal = unsupported_for(capability, target)
            if refusal:
                reasons.add(f"{refusal} (target: {target.describe()})")
        # A kind declared with no step to carry it has no variant to read, so
        # the kind is as much as this can answer.
        for kind in test_case.required_capabilities:
            if kind in asked:
                continue
            refusal = unsupported_capability(kind, target)
            if refusal:
                reasons.add(f"{refusal} (target: {target.describe()})")

    if not reasons:
        return None
    return "Unsupported evidence requirement(s): " + "; ".join(sorted(reasons))


def _release(version: Optional[str]) -> Optional[tuple[int, ...]]:
    """A version string as the numbers it opens with, or None where it has none.

    The driver reports an iOS release as "18.2" and adb reports an API level
    as "34". Both become tuples, so one comparison serves both. A version
    nothing parsed stays None, which refuses nothing.
    """
    parts: list[int] = []
    for piece in str(version or "").split("."):
        piece = piece.strip()
        if not piece.isdigit():
            break
        parts.append(int(piece))
    return tuple(parts) or None


def _host_tools() -> frozenset[str]:
    """Which of the programs a capability can need this machine has."""
    return frozenset(name for name in HOST_TOOLS if shutil.which(name))


class SupportedDevice(Protocol):
    """What this module reads off a session to key support on."""

    platform: Platform
    serial: Optional[str]
    os_version: Optional[str]


def target_of(device_control: SupportedDevice) -> CapabilityTarget:
    """The device in front of us, as far as capability support is concerned.

    Reads are best-effort. A property that cannot be established stays None,
    so every requirement about it is treated as met. A scenario refused on a
    read that failed would skip work the target can do.
    """
    from yewo.agent.device.adb_control import AdbControl

    platform = device_control.platform
    serial = device_control.serial or ""
    release = _release(device_control.os_version)

    if platform != Platform.ANDROID:
        # simctl lists simulators and nothing else, so a udid it does not list
        # is a real iPhone, which can have almost none of these capabilities.
        # A udid it would not answer about stays unestablished rather than
        # reading as either, the same way an unread release does.
        from yewo.agent.device.simctl_control import SimctlControl

        target = CapabilityTarget(
            platform=platform,
            emulator=SimctlControl.is_simulator(serial) if serial else None,
            release=release,
            host_tools=_host_tools(),
        )
        logger.info("Capability target: %s", target.describe())
        return target

    if not serial:
        return CapabilityTarget(platform=platform, release=release)

    # The API level decides support, and the release the user would name does
    # not. Android 13 is API 33, and no capability is keyed on "13".
    api = AdbControl.ask(serial, "getprop", "ro.build.version.sdk")
    packages = AdbControl.installed_packages(serial)
    target = CapabilityTarget(
        platform=platform,
        emulator=bool(AdbControl.avd_name(serial)),
        release=_release(api),
        packages=frozenset(packages) if packages else None,
    )
    logger.info("Capability target: %s", target.describe())
    return target


def unconfirmed_capabilities(
    required: Sequence[CapabilityKind], receipts: Sequence[Receipt]
) -> list[str]:
    """The declared capabilities a run never had confirmed on the device.

    A scenario declares what state it needs. Preflight answers whether this
    target could give it, which is a different question from whether the run
    did. Without this a scenario can declare `network`, stay online for every
    step, and pass on outcomes that were never about the state it asked for.

    A capability no receipt mentions was never applied. One whose receipts are all unreadable was applied and never confirmed. That
    is the same gap from the other side, because nothing establishes the
    state existed.
    """
    confirmed = {receipt.capability for receipt in receipts if receipt.confirmed}
    applied = {receipt.capability for receipt in receipts}
    missing = []
    for kind in required:
        name = kind.value
        if name in confirmed:
            continue
        missing.append(
            f"{name} (applied but never confirmed)" if name in applied
            else f"{name} (never applied)"
        )
    return missing
