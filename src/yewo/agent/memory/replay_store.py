"""The safe file contract for a judged path that a later run can walk.

It stores semantic anchors and declared value names. It never stores target
coordinates, typed values, or unprotected screen text.
"""

import hashlib
import hmac
import json
import os
import logging
import tempfile
from pathlib import Path
from enum import Enum
from typing import Dict, List, Literal, Optional, Protocol, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field
from yewo.utils.secrets import model_token

from yewo.agent.actuation.actions import MAX_WAIT_SECONDS, MIN_WAIT_SECONDS
from yewo.agent.run.postconditions import (
    EnteredValuePostcondition,
    ValueVerificationPolicy,
    verification_policy,
)

logger = logging.getLogger(__name__)
SCHEMA = 2
"""The shape of the file. A reader refuses a schema it does not know."""

MATCHER = 2
"""How an anchor is compared. Separate from the schema, because the same file
can be read by a stricter matcher, and then it must not be trusted."""

MAX_TAP_COUNT = 20
MAX_TAP_INTERVAL_MS = 5000
"""The bounds `TapAction` already holds. Repeated here because a file is
read before an action is built, and a cache outside them would otherwise
raise where it should be refused."""

TEXT_DIGEST_LENGTH = 16
"""How much of the digest is kept. Enough that two labels do not collide on
one screen, short enough that a person can read the file."""

KEY_PATH = Path.home() / ".yewo" / "replay-key"
"""Where the digest key lives, which is not beside the traces it protects.

A plain hash of screen text is not safe. A four-digit code has ten thousand
candidates, and a label has far fewer, so anyone holding a trace recovers
what the screen said. The key is what stops that, and a key kept in the
store would travel with the thing it protects.

A trace is therefore readable on the machine that learned it. Somewhere
else it matches nothing, which is a refusal and not a fault.
"""

PATH_KEY_ENV = "YEWO_PATH_KEY"
"""Optional shared key for repository paths used on another machine or CI."""

BOUNDS_SCALE = 1000
"""Bounds are held as thousandths of the screen, not as pixels.

A device reports pixels, and two targets of one profile can differ in them.
The fraction is what survives, and an integer keeps the file stable where a
float would round differently on two machines.
"""


class RelativeBounds(BaseModel):
    """Where an element sat, as thousandths of the screen it sat on."""

    model_config = ConfigDict(frozen=True)

    left: int = Field(ge=0, le=BOUNDS_SCALE)
    top: int = Field(ge=0, le=BOUNDS_SCALE)
    right: int = Field(ge=0, le=BOUNDS_SCALE)
    bottom: int = Field(ge=0, le=BOUNDS_SCALE)

    def is_near(self, other: "RelativeBounds", tolerance: int) -> bool:
        """Whether two bounds are close enough to be the same control.

        A rebuild moves a control by a little. A different control is
        somewhere else. The tolerance is what separates those two, and it
        is the caller's to state rather than this model's to assume.
        """
        return all(
            abs(mine - theirs) <= tolerance
            for mine, theirs in (
                (self.left, other.left),
                (self.top, other.top),
                (self.right, other.right),
                (self.bottom, other.bottom),
            )
        )


class AnchorContext(BaseModel):
    """A named detection that disambiguates an unnamed target beside it."""

    model_config = ConfigDict(frozen=True)

    role: str
    text: str = Field(min_length=1)
    bounds: RelativeBounds
    twins: int = Field(ge=1)


class ElementAnchor(BaseModel):
    """One element, by what it is rather than where it is.

    `text` is a digest, never the words. Naming the journey's own values is
    not enough: a screen carries an order number, a message, somebody's
    address. None of that is Yewo's to keep, and a path needs only to know
    that two readings are the same reading.

    A declared value becomes its name before it is digested, so one field
    reads the same across two runs that typed different values into it.
    """

    model_config = ConfigDict(frozen=True)

    role: str
    text: str = Field(
        description="A digest of the words, or the empty string where there were none."
    )
    bounds: RelativeBounds
    twins: int = Field(
        ge=1,
        description=(
            "Elements on that screen with this role and this text. Two "
            "controls reading alike are two different buttons, and a build "
            "that adds a third has changed what the first one meant."
        ),
    )
    context: Optional[AnchorContext] = Field(
        default=None,
        description=(
            "A uniquely placed named detection beside an unnamed target. "
            "The target is matched by its position relative to this detection."
        ),
    )


class ScreenSignature(BaseModel):
    """Enough of a screen to say a run reached it again.

    `phash` is a pre-filter and never the answer. Two screens of one app sit
    within a few bits of each other. So a pHash that matches proves nothing,
    and one that differs is worth a second look.
    """

    model_config = ConfigDict(frozen=True)

    description: Optional[str] = Field(
        default=None,
        description="A secret-safe semantic name for detector-free matching.",
    )
    anchors: Tuple[ElementAnchor, ...] = ()
    element_count: int = 0
    phash: Optional[str] = None
    keyboard_visible: Optional[bool] = Field(default=None, description="Effective keyboard state.")
    distinguishing: Tuple[Tuple[str, str], ...] = Field(
        default=(),
        description=(
            "Role and reading this screen has and the screen before it did "
            "not. A share of a screen is not proof a step arrived: a failure "
            "keeps the surrounding controls and changes only what matters. "
            "Enough of these changes have to remain to distinguish the "
            "arrival despite detection drift. Role as well as reading, "
            "because a heading and a button that read alike are not the "
            "same arrival."
        ),
    )


class CachedAction(BaseModel):
    """What to do, with no coordinate and no typed value in it."""

    model_config = ConfigDict(frozen=True)

    kind: str
    target: Optional[ElementAnchor] = None
    described: Optional[str] = Field(
        default=None,
        description=(
            "The words the planner used for its target, where the run was "
            "grounded by description. The anchor of a mode that detects "
            "nothing, and already free of declared values."
        ),
    )
    target_bounds: Optional[RelativeBounds] = Field(
        default=None,
        description="The successful target point or box in relative screen units.",
    )
    sensitive: bool = Field(
        default=False,
        description="Whether this action requires semantic verification before replay.",
    )
    value_name: Optional[str] = Field(
        default=None,
        description=(
            "The name the journey declared for the text to type. The value "
            "is resolved when the step runs, so no declared value is here."
        ),
    )
    postconditions: Tuple[EnteredValuePostcondition, ...] = Field(
        default=(),
        description=(
            "Checks the current run must earn after this action. These are "
            "specifications, never evidence or a saved verdict."
        ),
    )
    key: Optional[int] = None
    duration: Optional[float] = Field(default=None, ge=MIN_WAIT_SECONDS, le=MAX_WAIT_SECONDS)
    count: int = Field(
        default=1,
        ge=1,
        le=MAX_TAP_COUNT,
        description=(
            "How many times the step tapped. A double tap is a different "
            "action from a tap, and a concurrency scenario is built from "
            "the difference."
        ),
    )
    interval_ms: int = Field(
        default=0,
        ge=0,
        le=MAX_TAP_INTERVAL_MS,
        description="The gap the step left between taps. Zero taps as fast as it can.",
    )
    start_x: Optional[int] = Field(default=None, ge=0, le=BOUNDS_SCALE)
    start_y: Optional[int] = Field(default=None, ge=0, le=BOUNDS_SCALE)
    end_x: Optional[int] = Field(default=None, ge=0, le=BOUNDS_SCALE)
    end_y: Optional[int] = Field(default=None, ge=0, le=BOUNDS_SCALE)


class CachedTransition(BaseModel):
    """One step of a path: where it started, what it did, where it arrived."""

    model_config = ConfigDict(frozen=True)

    position: int = Field(ge=1)
    starts_segment: bool = Field(
        default=False,
        description="This follows a failed-action gap, so replay must match its source.",
    )
    source: ScreenSignature
    action: CachedAction
    destination: ScreenSignature
    arrival_ms: int = Field(
        ge=0,
        description=(
            "How long the action and immediate destination settlement took. "
            "Replay uses it as a bounded arrival window, never a fixed sleep."
        ),
    )


class CachedVerification(BaseModel):
    """Legacy saved evidence, parsed so older path files remain readable."""

    model_config = ConfigDict(frozen=True)

    rule_id: str = Field(pattern=r"^(expected|security):\d+$")
    status: Literal["passed", "failed", "untested"]
    applicable: bool = True


class ReplayProof(BaseModel):
    """Legacy proof data that current runs deliberately ignore."""

    model_config = ConfigDict(frozen=True)

    verifications: Tuple[CachedVerification, ...] = ()


class TraceIdentity(BaseModel):
    """What a trace is valid for.

    The screen size is not here. Bounds are held as thousandths, so a
    resize moves every anchor and the matcher refuses the first step on
    its own. The profile already separates two different targets.

    Refusing on size would buy one screen reading. It would cost a path
    learned in portrait that a landscape run can still walk wherever the
    layout held.

    The artifact hash is recorded and is not here. A code change gives a new
    artifact. Refusing the trace then would miss on every run after a commit,
    which is the run this exists for. A rebuild is what each step is verified
    against instead.
    """

    model_config = ConfigDict(frozen=True)

    scenario_uuid: str
    scenario_digest: str
    platform: str
    target_profile: str
    detector: str
    perception_pipeline: str = Field(min_length=1, description="Perception pipeline identity.")
    grounding: str


class ReplayTrace(BaseModel):
    """A judged sequence of successful path segments, as the file holds it."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    schema_version: int = Field(default=SCHEMA, alias="schema")
    matcher_version: int = Field(default=MATCHER, alias="matcher")
    identity: TraceIdentity
    learned_from: Optional[str] = Field(
        default=None,
        description="The artifact this path was learned on, as a fact about it.",
    )
    transitions: Tuple[CachedTransition, ...] = ()
    proof: Optional[ReplayProof] = Field(
        default=None,
        description="Legacy evidence retained only to read older path files.",
    )


class ValueNames:
    """Reads a declared name back out of the text a step typed.

    A step can carry either a real declared value or the placeholder a model
    received. The store keeps only the name. A reference matching nothing is
    not cached. One matching two names is not cached either, because resolving
    it later would be a guess.

    Every journey value follows the same rule. An app that declares no values
    caches its taps and not its typing. The only other way to keep a typed
    step would be to write its literal text down.
    """

    def __init__(self, values: Optional[Dict[str, str]] = None):
        self._names_by_value: Dict[str, List[str]] = {}
        self._names_by_reference: Dict[str, List[str]] = {}
        for name, value in (values or {}).items():
            if value:
                self._names_by_value.setdefault(value, []).append(name)
                token = model_token(name)
                self._names_by_reference.setdefault(token, []).append(name)
                last_four = f"{token}[…{value[-4:]}]"
                self._names_by_reference.setdefault(last_four, []).append(name)

    def name_of(self, text: Optional[str]) -> Optional[str]:
        """The one name this text was declared under, or None."""
        exact = text or ""
        declared_names = set(self._names_by_value.get(exact, ()))
        declared_names.update(self._names_by_reference.get(exact, ()))
        return declared_names.pop() if len(declared_names) == 1 else None

    def as_names(self, text: Optional[str]) -> str:
        """The text with every declared value replaced by its name.

        Longest first, so a value holding another value inside it is named
        before the shorter one rewrites part of it.
        """
        rewritten = text or ""
        for value in sorted(self._names_by_value, key=len, reverse=True):
            names = self._names_by_value[value]
            if len(names) == 1:
                rewritten = rewritten.replace(value, "{{" + names[0] + "}}")
        return rewritten


class DigestedScenario(Protocol):
    """What the digest reads off a locked scenario."""

    @property
    def setup(self) -> object: ...

    @property
    def steps(self) -> Sequence[object]: ...

    @property
    def expected_outcomes(self) -> Sequence[object]: ...


def _normalize_declared_values(value: object, names: ValueNames) -> object:
    """The same structure, with every declared value replaced by its name.

    Walked rather than searched for in the serialised text. JSON escapes a
    quote, a backslash and a newline, so a password holding one of those
    never matches its own spelling in the printed form. The digest would
    then change when that password did, and the path would be unreadable
    for every run but the one that wrote it.
    """
    if isinstance(value, str):
        return names.as_names(value)
    if isinstance(value, dict):
        return {key: _normalize_declared_values(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_declared_values(item, names) for item in value]
    return value


def _normalize_scenario_part(value: object, names: ValueNames) -> object:
    """One part of a scenario, as data rather than as its printed form.

    A pydantic model answers in JSON mode, so an enum is its value and a
    path is its text. Anything else is read as a string, with declared
    values named first.
    """
    if isinstance(value, BaseModel):
        return _normalize_declared_values(value.model_dump(mode="json"), names)
    if isinstance(value, Enum):
        return value.value
    return names.as_names(str(value))


def scenario_digest(
    scenario: DigestedScenario,
    values: Optional[Dict[str, str]] = None,
    verification_policies: Optional[Dict[str, ValueVerificationPolicy]] = None,
) -> str:
    """What a trace is valid for, over the scenario rather than the cell.

    `contract_sha256` hashes the journey and the cell, so two generated
    suites for one unchanged cell share it. A path belongs to the steps it
    walked, so this hashes those and the outcomes they were judged on.

    Values are named rather than quoted. A step carries the value it typed,
    and a digest must not change because a password did.

    Serialised rather than printed. `str()` on a model is a display, and a
    library that changes how it prints would invalidate every path without
    anything about the scenario changing.
    """
    names = ValueNames(values)
    digest_input = {
        "setup": _normalize_scenario_part(scenario.setup, names),
        "steps": [_normalize_scenario_part(step, names) for step in scenario.steps],
        "outcomes": [_normalize_scenario_part(one, names) for one in scenario.expected_outcomes],
        "value_verifications": {
            name: verification_policy(policy).model_dump(mode="json")
            for name, policy in sorted((verification_policies or {}).items())
        },
    }
    return hashlib.sha256(json.dumps(digest_input, sort_keys=True).encode("utf-8")).hexdigest()[:32]


def relative_bounds(box, screen_width: int, screen_height: int) -> Optional[RelativeBounds]:
    """One element's box as thousandths, or None when the screen is unknown.

    A screen of no size cannot place anything on it. Guessing a denominator
    would put every anchor in the top left corner, where they would all
    match each other.
    """
    if screen_width <= 0 or screen_height <= 0:
        return None

    def across(value: float) -> int:
        return max(0, min(BOUNDS_SCALE, round(value * BOUNDS_SCALE / screen_width)))

    def down(value: float) -> int:
        return max(0, min(BOUNDS_SCALE, round(value * BOUNDS_SCALE / screen_height)))

    return RelativeBounds(
        left=across(box.x_min),
        top=down(box.y_min),
        right=across(box.x_max),
        bottom=down(box.y_max),
    )


def digest_key(path: Path = KEY_PATH) -> bytes:
    """The key this machine digests screen text with, made once.

    Read-and-write for its owner only. A key anyone can read is a key that
    protects nothing, and the content behind these digests is an app's
    screens rather than Yewo's.
    """
    shared_key = os.environ.get(PATH_KEY_ENV)
    if shared_key:
        return hashlib.sha256(shared_key.encode("utf-8")).digest()
    try:
        try:
            return path.read_bytes()
        except FileNotFoundError:
            pass
        path.parent.mkdir(parents=True, exist_ok=True)
        made_key = os.urandom(32)
        descriptor, temporary_name = tempfile.mkstemp(dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as key_file:
                key_file.write(made_key)
            try:
                os.link(temporary_path, path)
                return made_key
            except FileExistsError:
                return path.read_bytes()
        finally:
            temporary_path.unlink(missing_ok=True)
    except OSError:
        # No key, no durable identity. A caller that cannot read one gets a
        # digest nothing else will match, so the trace is simply not reused.
        logger.warning("No replay digest key could be read. Traces will not match.")
        return os.urandom(32)


def text_identity(text: Optional[str], names: ValueNames, key: Optional[bytes] = None) -> str:
    """What a label is written down as, which is not the label.

    Declared values become their names first, so a field holding a password
    reads the same whatever the password is. What is left is keyed, because
    the rest of a screen is the app's content and not Yewo's to keep.

    Keyed rather than hashed. A plain digest of "4721" is recovered by
    trying ten thousand numbers, and most of a screen holds less than that.
    """
    text_without_values = names.as_names(text).strip()
    if not text_without_values:
        return ""
    return hmac.new(
        key if key is not None else digest_key(),
        text_without_values.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:TEXT_DIGEST_LENGTH]


def anchors_of(
    elements: Sequence, scan, names: ValueNames, key: Optional[bytes] = None
) -> Tuple[ElementAnchor, ...]:
    """Every element of a screen, as an identity rather than as words.

    The twin count is taken over the whole screen before any anchor is
    built. It is a fact about the screen, not about the element.

    The key is read once by the caller and handed down, so one screen does
    not open the key file for every element on it.
    """
    anchor_parts: List[Tuple[str, str, RelativeBounds]] = []
    for element in elements:
        bounds = relative_bounds(element.bounding_box, scan.screen_width, scan.screen_height)
        if bounds is None:
            return ()
        anchor_parts.append((element.type or "", text_identity(element.text, names, key), bounds))

    twins_by_identity: Dict[Tuple[str, str], int] = {}
    for role, text, _ in anchor_parts:
        twins_by_identity[(role, text)] = twins_by_identity.get((role, text), 0) + 1

    return tuple(
        ElementAnchor(role=role, text=text, bounds=bounds, twins=twins_by_identity[(role, text)])
        for role, text, bounds in anchor_parts
    )
