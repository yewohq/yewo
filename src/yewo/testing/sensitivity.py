"""How sensitive each artifact a run writes is, and what may be done with it.

A run leaves screenshots, prompts, logs, reports and a video. Yewo masks
what a journey declared across most of them, and covers declared values on
screen before an image is written. One artifact is covered by neither: the
recording. Nothing masks a video frame, so a password typed on screen is in
it.

So the class is declared here rather than left to whoever uploads. A list
kept by an uploader is a list somebody has to remember to update when a run
starts writing something new. A class in the record travels with the file.

Four classes, by what has been done to the artifact. Not by how bad it would
be to leak.

- `derived` holds nothing from the app at all.
- `scrubbed` had every declared value masked by name.
- `redacted` had every declared value on screen covered before it was written.
- `unmasked` had nothing done to it.

`unmasked` is the one that decides a policy. A journey that declares no
secret has nothing to mask, so the class says the same thing and the policy
reads it differently.

Retention is declared beside it, and **Yewo enforces none of it**. Nothing
here deletes a file. A run writes into a directory the user chose, and
clearing it is theirs to do. The declaration is for whatever keeps these
afterwards, so a store does not have to invent a policy per file type.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from yewo.testing.models import UserJourney
from yewo.errors import UsageError


class Sensitivity(str, Enum):
    """What was done to an artifact before it was written."""

    DERIVED = "derived"
    SCRUBBED = "scrubbed"
    REDACTED = "redacted"
    UNMASKED = "unmasked"


class Retention(str, Enum):
    """How long whatever keeps this should keep it.

    Two values, because a third would be a number nobody can defend. An
    artifact a later reader needs is kept. An artifact that is only useful
    while somebody is looking at a fresh failure is not.
    """

    KEEP = "keep"
    TRANSIENT = "transient"


@dataclass(frozen=True)
class ArtifactClass:
    """One kind of file a run writes, and what may be done with it."""

    kind: str
    sensitivity: Sensitivity
    retention: Retention
    exportable: bool
    detail: str

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "sensitivity": self.sensitivity.value,
            "retention": self.retention.value,
            "exportable": self.exportable,
            "detail": self.detail,
        }


ARTIFACTS: tuple[ArtifactClass, ...] = (
    ArtifactClass(
        "manifest",
        Sensitivity.SCRUBBED,
        Retention.KEEP,
        True,
        "Every string in it goes through the scrubber before it is written.",
    ),
    ArtifactClass(
        "reports",
        Sensitivity.SCRUBBED,
        Retention.KEEP,
        True,
        "Markdown, JUnit and the page, scrubbed. Read by people who were not there.",
    ),
    ArtifactClass(
        "prompts",
        Sensitivity.SCRUBBED,
        Retention.KEEP,
        True,
        "What each call asked, scrubbed, with screenshots referenced by hash.",
    ),
    ArtifactClass(
        "telemetry",
        Sensitivity.SCRUBBED,
        Retention.KEEP,
        True,
        "Token counts, cost, and every answer a model gave, scrubbed.",
    ),
    ArtifactClass(
        "screenshots",
        Sensitivity.REDACTED,
        Retention.KEEP,
        True,
        "Declared values on screen are covered before the file is written.",
    ),
    ArtifactClass(
        "logs",
        Sensitivity.SCRUBBED,
        Retention.KEEP,
        True,
        "The run's own log, scrubbed. Small, and the first thing a reader opens.",
    ),
    ArtifactClass(
        "device_logs",
        Sensitivity.UNMASKED,
        Retention.TRANSIENT,
        False,
        "The Appium log, and the device's own log beside it. Large, and "
        "masked by nothing. A device log holds other apps' output too.",
    ),
    ArtifactClass(
        "recordings",
        Sensitivity.UNMASKED,
        Retention.TRANSIENT,
        False,
        "Nothing masks a video frame. A value typed on screen is in it.",
    ),
)
"""Every kind of file a run writes, by what was done to it."""

RECORDING = "recordings"


def declared_for(_journey: UserJourney) -> tuple[dict, ...]:
    """The classes for a run's artifacts.

    The same for every journey. An earlier version made a recording
    `derived` where the journey declared no secret, reasoning that nothing
    could be exposed. That reasoning is wrong: a declaration says what Yewo
    was asked to mask, not what the app puts on screen. A video holds
    whatever the app rendered. That can be another account's data, a
    notification that arrived mid-run, a token in a debug banner, or a
    credential the app made up. None of those were declared. None of them is
    masked.

    What a declaration changes is whether a recording is made at all. See
    `recording_refused`.
    """
    return tuple(artifact.as_dict() for artifact in ARTIFACTS)


def recording_refused(journey: UserJourney, opted_in: bool) -> Optional[str]:
    """Why this journey is not recorded, or None where it may be.

    A journey with declared secrets types them on screen, and nothing covers
    a video frame. The run still produces every verdict. It keeps no video
    unless the caller asked for one in so many words.
    """
    if not journey.secrets or opted_in:
        return None
    named = ", ".join(sorted(journey.secrets))
    return (
        f"Journey '{journey.id}' declares secrets ({named}), and a recording "
        "is not masked. No video is kept. Pass --allow-secret-exposure to "
        "permit secrets in the recording and model input."
    )


def exposure_policy_refused(journey: UserJourney, opted_in: bool) -> Optional[str]:
    """Why a model-facing secret policy needs explicit unsafe consent."""
    exposed = sorted(
        name
        for name, display in journey.secret_displays.items()
        if display != "hide"
    )
    if opted_in or not exposed:
        return None
    return (
        "These secrets request model exposure: "
        + ", ".join(exposed)
        + ". Pass --allow-secret-exposure to allow `last4` or `show`, or set "
        "their redaction policy to `hide`."
    )


def require_safe_secret_policy(journey: UserJourney, opted_in: bool) -> None:
    """Reject a model-exposure policy without an explicit unsafe opt-in."""
    refusal = exposure_policy_refused(journey, opted_in)
    if refusal:
        raise UsageError(refusal)
