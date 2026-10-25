"""The failures Yewo recognises, and the ones that surprise it.

Each of these leads somewhere different, and `except Exception` collapses them
into one:

- a device Yewo cannot drive
- a screen Yewo cannot read
- an action the device refused
- an artifact Yewo could not write
- a command that asked for something absent

An adapter over a driver, a subprocess or the filesystem may still catch
broadly, because an SDK can raise almost anything. It then translates what it
caught into one of these types. Above that line, each handler names what it
expects.
"""

from enum import Enum


class YewoError(Exception):
    """Base for a failure Yewo recognises and can describe."""


class DeviceUnavailable(YewoError):
    """Yewo could not drive the device, the emulator or its tooling."""


class ObservationUnavailable(YewoError):
    """Yewo could not read the screen, so it has nothing to decide from."""


class MeasurementUnavailable(YewoError):
    """The app ran, but its measurement source produced no usable sample."""


class ActionRejected(YewoError):
    """The device did not do what the action asked. Nothing may assume it did."""


class TargetNotInteractive(ActionRejected):
    """The target took no text, so the element behind it is not an input.

    Raised when focus stands where it stood before the tap. The screen alone
    cannot tell a field from a picture of one, and this is the evidence that
    settles it for that element.
    """


class ProviderUnavailable(YewoError):
    """A model service would not answer, and the attempts are spent.

    Raised only where another attempt could have answered: a timeout, a rate
    limit, a connection that dropped, a server error. So the scenario ends
    and the suite goes on, because the same service may answer the next one.

    A provider that refused the request itself is not this. It is raised as
    it came, and it ends the run, because every later call carries the same
    request and meets the same refusal.
    """


class ArtifactWriteFailed(YewoError):
    """Yewo could not write a report, a log or a screenshot."""


class UsageError(YewoError):
    """The command asked for something Yewo cannot find or use.

    The message goes to the person who typed the command. It says what to fix.
    """


DEVICE_MODULES = ("selenium", "appium", "urllib3")
"""Packages that speak to the device. A failure from one is not the app's."""

PROVIDER_MODULES = ("litellm", "openai", "httpx", "httpx2", "typesafe_sdk")
"""Packages that speak to a model service."""


class Cause(str, Enum):
    """Whose problem a scenario that reached no verdict is.

    The status says what happened. The cause says what to do about it, which
    is the question a reader of a red build actually has. A flaky emulator, an
    exhausted API key and a thin audit all ended a scenario, and all three
    want a different answer.

    Nothing here is `app`. An app at fault is a FAILED scenario, and that
    needs no cause.
    """

    DEVICE = "device"
    """The device, the emulator or its tooling could not be driven."""

    PROVIDER = "provider"
    """A model service refused, timed out or ran out of credit."""

    ARTIFACT = "artifact"
    """Yewo could not write what a verdict rests on."""

    AGENT = "agent"
    """Yewo drove the app but never reached a state worth judging."""

    EVIDENCE = "evidence"
    """The run finished, and the marks do not settle what was asked."""

    SCENARIO = "scenario"
    """The scenario asked for something nothing in a run can settle."""

    UNKNOWN = "unknown"
    """Not recognised. Claiming a cause without evidence is how a report lies."""


def cause_of(exc: BaseException) -> Cause:
    """What kind of failure an exception is, read from its type.

    By type and by the package it came from, never by its words: a message is
    written for a person and changes between versions.
    """
    if isinstance(exc, (DeviceUnavailable, ObservationUnavailable)):
        return Cause.DEVICE
    if isinstance(exc, ProviderUnavailable):
        return Cause.PROVIDER
    if isinstance(exc, ArtifactWriteFailed):
        return Cause.ARTIFACT
    package = type(exc).__module__.split(".")[0]
    if package in PROVIDER_MODULES:
        return Cause.PROVIDER
    if package in DEVICE_MODULES:
        return Cause.DEVICE
    return Cause.UNKNOWN
