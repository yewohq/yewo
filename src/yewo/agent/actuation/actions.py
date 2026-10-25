from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Optional, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator


class Coordinates(BaseModel):
    x: int = Field(..., description="The x coordinate in pixels.")
    y: int = Field(..., description="The y coordinate in pixels.")

    @model_validator(mode="before")
    @classmethod
    def _accept_pairs(cls, value: Any) -> Any:
        """Accepts [x, y] as well as {"x": .., "y": ..}.

        Models routinely answer with a pair even when the schema asks for an
        object, and the request cannot insist: provider-side schema enforcement
        is off. Rejecting the pair fails the whole step response and kills the
        scenario, so the pair is read for what it plainly is.
        """
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return {"x": value[0], "y": value[1]}
        return value


class ActionType(str, Enum):
    TYPE = "type"
    TAP = "tap"
    DOUBLE_TAP = "double_tap"
    LONG_PRESS = "long_press"
    DRAG_AND_DROP = "drag_and_drop"
    KEY_EVENT = "key_event"  # key events (4=back, 3=home, 66=enter, 67=backspace)
    SWIPE = "swipe"
    CAPABILITY = "capability"
    CLEAR = "clear"
    WAIT = "wait"
    TARGET_NOT_LISTED = "target_not_listed"


class CapabilityKind(str, Enum):
    """The device states a scenario can ask for.

    The value selects the matching model below. Adding a capability means
    adding a model and a member. The planner vocabulary and dispatch then
    follow from the union.
    """

    LOCATION = "location"
    NETWORK = "network"
    BATTERY = "battery"
    TIMEZONE = "timezone"
    LOCALE = "locale"
    PERMISSION = "permission"
    INTERRUPT = "interrupt"
    BIOMETRIC = "biometric"
    CLIPBOARD = "clipboard"
    DEEPLINK = "deeplink"
    APP_STATE = "app_state"
    MEDIA_INJECT = "media_inject"
    ACCESSIBILITY = "accessibility"


class CapabilityModel(BaseModel):
    """A capability, which takes the fields it declares and no others.

    Extra fields are refused rather than dropped. The prompt tells the model
    to send only the listed fields. Without enforcement, a stale `parameters`
    object disappears during validation. The capability then runs on defaults
    that the scenario did not request.
    """

    model_config = ConfigDict(extra="forbid")

    @property
    def support_variant(self) -> str | None:
        """The sub-choice the preflight decides separately, or None.

        A kind alone does not always say whether a target can apply it. An
        incoming call needs an emulator console and a low-memory warning does
        not, and both are one kind. A capability whose variants are all
        supported wherever the kind is answers None.
        """
        return None


class LocationCapability(CapabilityModel):
    kind: Literal[CapabilityKind.LOCATION] = CapabilityKind.LOCATION
    latitude: float
    longitude: float


class NetworkCapability(CapabilityModel):
    kind: Literal[CapabilityKind.NETWORK] = CapabilityKind.NETWORK
    mode: Literal["offline", "online", "wifi_only", "data_only", "airplane"]


class BatteryCapability(CapabilityModel):
    kind: Literal[CapabilityKind.BATTERY] = CapabilityKind.BATTERY
    level: int = Field(..., ge=0, le=100)
    charging: bool = True


class TimezoneCapability(CapabilityModel):
    kind: Literal[CapabilityKind.TIMEZONE] = CapabilityKind.TIMEZONE
    timezone: str = Field(..., min_length=1, description="An IANA name, e.g. 'America/New_York'.")


class LocaleCapability(CapabilityModel):
    kind: Literal[CapabilityKind.LOCALE] = CapabilityKind.LOCALE
    language: str = Field(..., min_length=1, description="An ISO code, e.g. 'fr'.")
    country: str = Field(default="", description="An ISO code, e.g. 'FR'. May be empty.")


class PermissionCapability(CapabilityModel):
    kind: Literal[CapabilityKind.PERMISSION] = CapabilityKind.PERMISSION
    permission: Literal["camera", "location", "microphone", "notifications", "photos", "contacts", "storage"]
    grant: bool

    @property
    def support_variant(self) -> str | None:
        return self.permission


class InterruptCapability(CapabilityModel):
    kind: Literal[CapabilityKind.INTERRUPT] = CapabilityKind.INTERRUPT
    type: Literal["low_memory", "incoming_call"]
    phone: str = Field(default="555-5555", description="Used by 'incoming_call' only.")

    @property
    def support_variant(self) -> str | None:
        return self.type


class BiometricCapability(CapabilityModel):
    kind: Literal[CapabilityKind.BIOMETRIC] = CapabilityKind.BIOMETRIC
    action: Literal["match", "unmatch", "enroll"]
    type: Literal["faceId", "touchId"] = "faceId"
    finger_id: int = Field(default=1, ge=1)


class ClipboardCapability(CapabilityModel):
    kind: Literal[CapabilityKind.CLIPBOARD] = CapabilityKind.CLIPBOARD
    text: str


class DeeplinkCapability(CapabilityModel):
    kind: Literal[CapabilityKind.DEEPLINK] = CapabilityKind.DEEPLINK
    url: str = Field(..., min_length=1)


class AppStateCapability(CapabilityModel):
    kind: Literal[CapabilityKind.APP_STATE] = CapabilityKind.APP_STATE
    action: Literal["force_stop", "clear_data"]

    @property
    def support_variant(self) -> str | None:
        return self.action


class MediaInjectCapability(CapabilityModel):
    kind: Literal[CapabilityKind.MEDIA_INJECT] = CapabilityKind.MEDIA_INJECT
    local_path: str = Field(..., min_length=1)


class AccessibilityCapability(CapabilityModel):
    kind: Literal[CapabilityKind.ACCESSIBILITY] = CapabilityKind.ACCESSIBILITY
    enabled: bool

    @property
    def support_variant(self) -> str | None:
        return "on" if self.enabled else "off"


Capability = Annotated[
    LocationCapability
    | NetworkCapability
    | BatteryCapability
    | TimezoneCapability
    | LocaleCapability
    | PermissionCapability
    | InterruptCapability
    | BiometricCapability
    | ClipboardCapability
    | DeeplinkCapability
    | AppStateCapability
    | MediaInjectCapability
    | AccessibilityCapability,
    Field(discriminator="kind"),
]

CAPABILITY_MODELS: dict[CapabilityKind, type[BaseModel]] = {
    model.model_fields["kind"].default: model
    for model in (
        LocationCapability,
        NetworkCapability,
        BatteryCapability,
        TimezoneCapability,
        LocaleCapability,
        PermissionCapability,
        InterruptCapability,
        BiometricCapability,
        ClipboardCapability,
        DeeplinkCapability,
        AppStateCapability,
        MediaInjectCapability,
        AccessibilityCapability,
    )
}


def _field_spec(field) -> str:
    """One field as the prompt shows it: its choices, or its type."""
    if get_origin(field.annotation) is Literal:
        return " | ".join(repr(choice) for choice in get_args(field.annotation))
    return getattr(field.annotation, "__name__", str(field.annotation))


def capability_vocabulary() -> str:
    """The capability listing for the planner prompt, read off the models.

    Generated rather than written out, so a new capability costs no prompt
    edit and the model stays the one statement of its own fields.

    Each line is the object to send, `kind` included, because that is what the
    schema takes. A field with a default is marked, so the reader can tell what
    it has to decide from what it may leave out.
    """
    lines = []
    for kind, model in CAPABILITY_MODELS.items():
        fields = [f'"kind": "{kind.value}"']
        for name, field in model.model_fields.items():
            if name == "kind":
                continue
            spec = _field_spec(field)
            if not field.is_required():
                spec += f" (optional, default {field.default!r})"
            fields.append(f'"{name}": {spec}')
        lines.append("   - {" + ", ".join(fields) + "}")
    return "\n".join(lines)


class TypeAction(BaseModel):
    type: Literal[ActionType.TYPE] = ActionType.TYPE
    text: str = Field(default="", description="The text of the action.")
    coordinates: Coordinates = Field(..., description="The screen coordinates to move to.")
    target: Optional[str] = Field(
        default=None, description="A semantic description of the target field."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


MAX_TAP_COUNT = 20
"""Taps one action may make. A scenario asking for more wants a different test."""


class TapAction(BaseModel):
    type: Literal[ActionType.TAP] = ActionType.TAP
    coordinates: Coordinates = Field(..., description="The screen coordinates to tap.")
    target: Optional[str] = Field(
        default=None, description="A semantic description of the target control."
    )
    count: int = Field(
        default=1,
        ge=1,
        le=MAX_TAP_COUNT,
        description=(
            "How many times to tap, in one action. Use this when a scenario asks "
            "for repeated or rapid taps. Do not count the taps yourself across "
            "several steps."
        ),
    )
    interval_ms: int = Field(
        default=0,
        ge=0,
        le=5000,
        description="Milliseconds to wait between taps. 0 taps as fast as it can.",
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class ClearAction(BaseModel):
    type: Literal[ActionType.CLEAR] = ActionType.CLEAR
    coordinates: Coordinates = Field(
        ..., description="The screen coordinates of the text field to clear."
    )
    target: Optional[str] = Field(
        default=None, description="A semantic description of the target field."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class DoubleTapAction(BaseModel):
    type: Literal[ActionType.DOUBLE_TAP] = ActionType.DOUBLE_TAP
    coordinates: Coordinates = Field(..., description="The screen coordinates to double tap.")
    target: Optional[str] = Field(
        default=None, description="A semantic description of the target control."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class LongPressAction(BaseModel):
    type: Literal[ActionType.LONG_PRESS] = ActionType.LONG_PRESS
    duration: float = Field(..., description="The duration of the long press in seconds.")
    coordinates: Coordinates = Field(..., description="The screen coordinates to long press.")
    target: Optional[str] = Field(
        default=None, description="A semantic description of the target control."
    )
    reason: str = Field(..., description="How the action helps achieve the goal.")


class DragAndDropAction(BaseModel):
    type: Literal[ActionType.DRAG_AND_DROP] = ActionType.DRAG_AND_DROP
    source_x: int = Field(..., description="The x coordinate to drag from.")
    source_y: int = Field(..., description="The y coordinate to drag from.")
    destination_x: int = Field(..., description="The x coordinate to drag to.")
    destination_y: int = Field(..., description="The y coordinate to drag to.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class KeyEventAction(BaseModel):
    type: Literal[ActionType.KEY_EVENT] = ActionType.KEY_EVENT
    key: int = Field(..., description="The key code to press.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class SwipeAction(BaseModel):
    type: Literal[ActionType.SWIPE] = ActionType.SWIPE
    start_x: int = Field(..., description="The x coordinate to start the swipe.")
    start_y: int = Field(..., description="The y coordinate to start the swipe.")
    end_x: int = Field(..., description="The x coordinate to end the swipe.")
    end_y: int = Field(..., description="The y coordinate to end the swipe.")
    reason: str = Field(..., description="How the action helps achieve the goal.")


class CapabilityAction(BaseModel):
    type: Literal[ActionType.CAPABILITY] = ActionType.CAPABILITY
    capability: Capability = Field(
        ...,
        description="The device state to set, named by its 'kind' and that kind's own fields.",
    )
    reason: str = Field(..., description="How the capability control helps achieve the test goal.")


class TargetNotListedAction(BaseModel):
    """Says the target is not among the numbered elements, rather than guessing.

    For when the element a step needs is absent from the numbered targets.
    Never executed: the step stops and the screen is recorded, so a detection
    gap becomes a counted event instead of a tap into empty space.
    """

    type: Literal[ActionType.TARGET_NOT_LISTED] = ActionType.TARGET_NOT_LISTED
    description: str = Field(..., description="What the target looks like and roughly where it is.")
    reason: str = Field(..., description="What you were trying to achieve.")


MIN_WAIT_SECONDS = 0.1
MAX_WAIT_SECONDS = 10.0


class WaitAction(BaseModel):
    """Waits before the next observation so transient rendering can finish."""

    type: Literal[ActionType.WAIT] = ActionType.WAIT
    duration: float = Field(
        default=1.0, description="Seconds to wait before looking at the screen again."
    )
    reason: str = Field(..., description="What the screen is expected to finish doing.")


class RetargetedAction(BaseModel):
    """An action whose target is a mark or a description, not a coordinate.

    Declared so that `coordinate_form` has one owner. Two groundings read it
    off the class, to turn a placed target back into the action the device
    takes. A class attribute nobody declares cannot be read safely.

    No fields and no docstring reach the schema from here. The variant builds
    its own fields, and a created class carries its own empty docstring.
    """

    coordinate_form: ClassVar[type[BaseModel]]


def _retargeted(
    model: type[BaseModel],
    name: str,
    target: str,
    definition: tuple[type, Any],
) -> type[BaseModel]:
    """One action model with `coordinates` replaced by another way to name a target.

    The field map is typed loosely for one reason. `create_model` takes field
    definitions and reserved keywords through the same keyword arguments, and
    no overload describes a map of the first. The looseness is narrow: the map
    is built here, and the class it returns is a BaseModel subclass.

    `coordinate_form` carries the model this one resolves back into, once the
    mark or the description has been placed on the screen.
    """
    fields: dict[str, Any] = {
        field_name: (info.annotation, info)
        for field_name, info in model.model_fields.items()
        if field_name != "coordinates"
    }
    fields[target] = definition
    if target != "target":
        fields["target"] = (
            str,
            Field(
                ...,
                description=(
                    "What the numbered target is, described by its own visible "
                    "text, symbol, shape, or colour."
                ),
            ),
        )
    variant = create_model(name, __base__=RetargetedAction, **fields)
    variant.coordinate_form = model
    return variant


def _marks_variant(model: type[BaseModel], name: str) -> type[BaseModel]:
    """The same action with coordinates removed and `mark` required.

    Marks mode is enforced by the schema rather than by prose alone
    afterwards. An optional field is one the model may omit — and it does,
    intermittently, which turns every omission into a wasted step. A field it
    cannot omit and a field that does not exist leave nothing to get wrong.
    """
    return _retargeted(
        model,
        name,
        "mark",
        (int, Field(..., description="Number of the target element on the screenshot.")),
    )


def _describe_variant(model: type[BaseModel], name: str) -> type[BaseModel]:
    """The same action with coordinates replaced by a description of the target.

    This uses the same structure as the marks variant. A required field and an
    absent field leave nothing for the model to get wrong.
    """
    return _retargeted(
        model,
        name,
        "target",
        (
            str,
            Field(
                ...,
                description=(
                    "What to act on, described as you would point it out to "
                    "someone looking at the screen."
                ),
            ),
        ),
    )


TypeMarkAction = _marks_variant(TypeAction, "TypeMarkAction")
TapMarkAction = _marks_variant(TapAction, "TapMarkAction")
ClearMarkAction = _marks_variant(ClearAction, "ClearMarkAction")
DoubleTapMarkAction = _marks_variant(DoubleTapAction, "DoubleTapMarkAction")
LongPressMarkAction = _marks_variant(LongPressAction, "LongPressMarkAction")

TypeDescribeAction = _describe_variant(TypeAction, "TypeDescribeAction")
TapDescribeAction = _describe_variant(TapAction, "TapDescribeAction")
ClearDescribeAction = _describe_variant(ClearAction, "ClearDescribeAction")
DoubleTapDescribeAction = _describe_variant(DoubleTapAction, "DoubleTapDescribeAction")
LongPressDescribeAction = _describe_variant(LongPressAction, "LongPressDescribeAction")

# Actions that touch a specific place, and so need a target resolved first.
TARGETED_ACTIONS = frozenset(
    {
        ActionType.TAP,
        ActionType.TYPE,
        ActionType.CLEAR,
        ActionType.DOUBLE_TAP,
        ActionType.LONG_PRESS,
    }
)


# Actions whose target must be a text field, and which therefore prove
# whether it is one.
TEXT_FIELD_ACTIONS = frozenset({ActionType.TYPE, ActionType.CLEAR})


Action = Annotated[
    TypeAction
    | TapAction
    | DoubleTapAction
    | LongPressAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
    | CapabilityAction
    | ClearAction
    | WaitAction
    | TargetNotListedAction,
    Field(discriminator="type"),
]


# The same vocabulary, but targets are named in words only.
DescribeAction = Annotated[
    TypeDescribeAction
    | TapDescribeAction
    | DoubleTapDescribeAction
    | LongPressDescribeAction
    | ClearDescribeAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
    | CapabilityAction
    | WaitAction,
    Field(discriminator="type"),
]


# The same vocabulary, but targets are named by number only.
MarksAction = Annotated[
    TypeMarkAction
    | TapMarkAction
    | DoubleTapMarkAction
    | LongPressMarkAction
    | ClearMarkAction
    | DragAndDropAction
    | KeyEventAction
    | SwipeAction
    | CapabilityAction
    | WaitAction
    | TargetNotListedAction,
    Field(discriminator="type"),
]


class ActionPlan(BaseModel):
    instruction: str = Field(..., description="The normalized instruction to achieve the goal.")
    actions: list[Action] = Field(
        ...,
        description="The sequence of machine-executable actions to achieve the goal.",
    )
    external_navigation: bool = Field(
        default=False,
        description="Set to True if the next step is expected to leave the SUT app temporarily. Examples include a browser, map, document picker, or system dialog.",
    )
