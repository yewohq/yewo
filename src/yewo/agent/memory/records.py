"""Reusable application knowledge owned by YewoAgent."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeSignal,
    ScrollDistance,
    SideEffectClass,
)
from yewo.agent.memory.grounding import AcceptedGrounding, validate_transition_groundings
from yewo.agent.memory.knowledge import AppKnowledge, validate_app_knowledge
from yewo.agent.memory.references import (
    referenced_memory_observation_ids,
    validate_retained_observations,
)
from yewo.applications import ApplicationDescriptor, ApplicationRequest
from yewo.artifacts import EvidenceApplicationIdentity, RecordIdentifier, Sha256Digest
from yewo.devices.interaction import ScrollDirection
from yewo.grounding import PixelPoint
from yewo.observations import (
    DeviceOrientation,
    PresentationKey,
    ScreenComparisonPolicy,
    ScreenshotReference,
)


MemoryText = Annotated[str, Field(min_length=1, max_length=4_000)]


class MemoryRetentionPolicy(BaseModel):
    """Bound retained evidence without weakening graph integrity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_variants_per_screen: int = Field(default=20, ge=1, le=100)
    maximum_transitions: int = Field(default=500, ge=1, le=5_000)
    maximum_unsuccessful_attempts: int = Field(default=100, ge=1, le=1_000)


class MemoryCompatibility(BaseModel):
    """Application and schema identity required to reuse stored memory."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    application: EvidenceApplicationIdentity
    observation_schema_version: Literal[1] = 1
    fingerprint_algorithm: Literal["application-phash-v1"] = "application-phash-v1"
    action_schema_version: Literal[1] = 1
    outcome_schema_version: Literal[1] = 1
    screen_comparison: ScreenComparisonPolicy = ScreenComparisonPolicy()
    retention: MemoryRetentionPolicy = MemoryRetentionPolicy()
    setup_contract_sha256: Sha256Digest | None = None

    @model_validator(mode="after")
    def _validate_build_identity(self) -> Self:
        application = self.application
        if application.artifact_sha256 is None and not (
            application.version is not None and application.build is not None
        ):
            raise ValueError("Reusable memory requires an application artifact or versioned build.")
        return self


class ObservationVariant(BaseModel):
    """One exact observed presentation belonging to a logical screen."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    observation_id: RecordIdentifier
    source_run_id: RecordIdentifier
    presentation: PresentationKey
    screenshot: ScreenshotReference

    @model_validator(mode="after")
    def _validate_screenshot(self) -> Self:
        if (self.presentation.width_px, self.presentation.height_px) != (
            self.screenshot.width_px,
            self.screenshot.height_px,
        ):
            raise ValueError("A memory variant must match its screenshot dimensions.")
        if self.presentation.fingerprint.source_artifact_id != self.screenshot.artifact_id:
            raise ValueError("A memory variant fingerprint must identify its screenshot.")
        return self


class LogicalScreen(BaseModel):
    """Compatible exact variants treated as one reusable application state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    screen_id: RecordIdentifier
    variants: tuple[ObservationVariant, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_variants(self) -> Self:
        observation_ids = [variant.observation_id for variant in self.variants]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("A logical screen cannot repeat an observation variant.")
        return self


class InputValueShape(StrEnum):
    """Reusable shape of a disposable value accepted by one editable field."""

    TEXT = "text"
    INTEGER = "integer"
    DECIMAL = "decimal"
    FORMATTED = "formatted"


class InputContract(BaseModel):
    """Compact successful-input evidence that can avoid a separate focus step."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    value_shape: InputValueShape
    format_pattern: str | None = Field(default=None, min_length=1, max_length=100)

    @model_validator(mode="after")
    def _validate_format_pattern(self) -> Self:
        if (self.value_shape is InputValueShape.FORMATTED) != (self.format_pattern is not None):
            raise ValueError("Only formatted input requires a format pattern.")
        return self


class SemanticRouteAction(BaseModel):
    """Reusable action intent without coordinates or runtime-owned text values."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    action: ActionKind
    target_description: str | None = Field(default=None, min_length=1, max_length=2_000)
    destination_description: str | None = Field(default=None, min_length=1, max_length=2_000)
    expected_outcome: MemoryText = Field(
        description=(
            "Unverified action intent retained for replay authorization. Transition outcome "
            "signals and destination observations provide the reusable evidence."
        )
    )
    direction: ScrollDirection | None = None
    distance: ScrollDistance | None = None
    duration_seconds: float | None = Field(default=None, gt=0)
    declared_value_name: RecordIdentifier | None = None
    input_contract: InputContract | None = None

    @model_validator(mode="after")
    def _validate_reusable_action(self) -> Self:
        self._validate_reusable_action_kind()
        self._validate_type_contract()
        self._validate_action_descriptions()
        return self

    def _validate_reusable_action_kind(self) -> None:
        if self.action in {
            ActionKind.SET_DEVICE_CAPABILITY,
            ActionKind.DONE,
            ActionKind.CANNOT_PROGRESS,
        }:
            raise ValueError("This action cannot be stored as reusable route mechanics.")

    def _validate_type_contract(self) -> None:
        if (
            self.action is ActionKind.TYPE
            and self.declared_value_name is None
            and self.input_contract is None
        ):
            raise ValueError("A retained type action requires a value reference or input shape.")
        if self.action is not ActionKind.TYPE and self.declared_value_name is not None:
            raise ValueError("Only a reusable type action can contain a value reference.")
        if self.action is ActionKind.CLEAR and self.target_description is not None:
            raise ValueError("A focused-field clear action cannot contain a target description.")
        if self.action is not ActionKind.TYPE and self.input_contract is not None:
            raise ValueError("Only a reusable type action can contain an input contract.")

    def _validate_action_descriptions(self) -> None:
        if (
            self.action
            in {
                ActionKind.TAP,
                ActionKind.FOCUS,
                ActionKind.DOUBLE_TAP,
                ActionKind.LONG_PRESS,
                ActionKind.DRAG,
            }
            and self.target_description is None
        ):
            raise ValueError("This reusable action requires a target description.")
        if self.action is ActionKind.DRAG and self.destination_description is None:
            raise ValueError("A reusable drag requires a destination description.")
        if self.action is ActionKind.SCROLL and (self.direction is None or self.distance is None):
            raise ValueError("A reusable scroll requires direction and distance.")


class AcceptedTransition(BaseModel):
    """One dispatched action supported by observed outcome and destination evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    transition_id: RecordIdentifier
    source_screen_id: RecordIdentifier
    destination_screen_id: RecordIdentifier
    source_observation_id: RecordIdentifier
    destination_observation_id: RecordIdentifier
    action: SemanticRouteAction
    side_effect: SideEffectClass
    authorization_id: RecordIdentifier
    receipt_id: RecordIdentifier
    outcome_id: RecordIdentifier
    outcome_signals: frozenset[ActionOutcomeSignal] = Field(min_length=1)
    target_grounding: AcceptedGrounding | None = None
    destination_grounding: AcceptedGrounding | None = None
    source_step_id: RecordIdentifier

    @field_serializer("outcome_signals")
    def _serialize_outcome_signals(
        self,
        signals: frozenset[ActionOutcomeSignal],
    ) -> tuple[str, ...]:
        """Keep the persisted memory digest stable across processes."""
        return tuple(sorted(signal.value for signal in signals))

    @model_validator(mode="after")
    def _validate_grounding_provenance(self) -> Self:
        if self.target_grounding is not None and self.action.target_description is None:
            raise ValueError("Transition target grounding requires a target expression.")
        if self.destination_grounding is not None and self.action.destination_description is None:
            raise ValueError("Transition destination grounding requires a destination expression.")
        return self


class UnsuccessfulAttemptDisposition(StrEnum):
    """Reusable classification of one action that did not make progress."""

    TARGET_REJECTED = "target_rejected"
    NO_OBSERVABLE_CHANGE = "no_observable_change"


class UnsuccessfulAttempt(BaseModel):
    """One prior attempt, without upgrading it to an application absence claim."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    attempt_id: RecordIdentifier
    screen_id: RecordIdentifier
    source_observation_id: RecordIdentifier
    action: ActionKind
    target_description: str | None = Field(default=None, min_length=1, max_length=2_000)
    destination_description: str | None = Field(default=None, min_length=1, max_length=2_000)
    direction: ScrollDirection | None = None
    distance: ScrollDistance | None = None
    disposition: UnsuccessfulAttemptDisposition
    reason: MemoryText
    grounded_point: PixelPoint | None = None
    screenshot_width_px: int | None = Field(default=None, gt=0)
    screenshot_height_px: int | None = Field(default=None, gt=0)
    source_step_id: RecordIdentifier
    source_run_id: RecordIdentifier

    @model_validator(mode="after")
    def _validate_attempt_evidence(self) -> Self:
        point_fields = (
            self.grounded_point,
            self.screenshot_width_px,
            self.screenshot_height_px,
        )
        if any(value is not None for value in point_fields) and not all(
            value is not None for value in point_fields
        ):
            raise ValueError("Grounded attempt evidence requires a point and both dimensions.")
        if self.grounded_point is not None:
            width_px = self.screenshot_width_px
            height_px = self.screenshot_height_px
            if width_px is None or height_px is None:
                raise ValueError("Grounded attempt evidence requires screenshot dimensions.")
            if self.grounded_point.x >= width_px or self.grounded_point.y >= height_px:
                raise ValueError("The unsuccessful-attempt point must be within the screenshot.")
        return self

    def has_same_evidence(self, other: UnsuccessfulAttempt) -> bool:
        """Compare reusable evidence while ignoring run-owned identifiers."""
        ignored = {"attempt_id", "source_observation_id", "source_step_id", "source_run_id"}
        return self.model_dump(exclude=ignored) == other.model_dump(exclude=ignored)


class AppMemory(BaseModel):
    """Reusable evidence for one compatible application build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    memory_id: RecordIdentifier
    compatibility: MemoryCompatibility
    source_run_ids: tuple[RecordIdentifier, ...] = Field(min_length=1)
    screens: tuple[LogicalScreen, ...] = Field(min_length=1)
    transitions: tuple[AcceptedTransition, ...] = ()
    unsuccessful_attempts: tuple[UnsuccessfulAttempt, ...] = ()
    app_knowledge: AppKnowledge = AppKnowledge()
    updated_at: datetime

    @model_validator(mode="after")
    def _validate_graph(self) -> Self:
        if self.updated_at.utcoffset() is None:
            raise ValueError("App memory time must include a timezone.")
        screen_ids, screens_by_id = self._validate_screen_ids()
        transition_ids = self._validate_transitions(screen_ids)
        self._validate_unsuccessful_attempts(screen_ids, screens_by_id)
        observation_ids = self._observation_ids()
        self._validate_knowledge(screen_ids, transition_ids, observation_ids)
        self._validate_screen_masks()
        self._validate_retention_limits()
        return self

    def _validate_screen_ids(
        self,
    ) -> tuple[set[RecordIdentifier], dict[RecordIdentifier, LogicalScreen]]:
        screen_ids = [screen.screen_id for screen in self.screens]
        if len(screen_ids) != len(set(screen_ids)):
            raise ValueError("App memory screen identifiers must be unique.")
        return set(screen_ids), {screen.screen_id: screen for screen in self.screens}

    def _validate_transitions(
        self,
        screen_ids: set[RecordIdentifier],
    ) -> set[RecordIdentifier]:
        transition_ids = [transition.transition_id for transition in self.transitions]
        if len(transition_ids) != len(set(transition_ids)):
            raise ValueError("App memory transition identifiers must be unique.")
        if any(
            transition.source_screen_id not in screen_ids
            or transition.destination_screen_id not in screen_ids
            for transition in self.transitions
        ):
            raise ValueError("Every transition must reference known logical screens.")
        validate_transition_groundings(self.screens, self.transitions)
        return set(transition_ids)

    def _validate_unsuccessful_attempts(
        self,
        screen_ids: set[RecordIdentifier],
        screens_by_id: dict[RecordIdentifier, LogicalScreen],
    ) -> None:
        attempt_ids = [attempt.attempt_id for attempt in self.unsuccessful_attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("App memory unsuccessful-attempt identifiers must be unique.")
        if any(attempt.screen_id not in screen_ids for attempt in self.unsuccessful_attempts):
            raise ValueError("Every unsuccessful attempt must reference a known logical screen.")
        for attempt in self.unsuccessful_attempts:
            self._validate_unsuccessful_attempt(attempt, screens_by_id)

    def _validate_unsuccessful_attempt(
        self,
        attempt: UnsuccessfulAttempt,
        screens_by_id: dict[RecordIdentifier, LogicalScreen],
    ) -> None:
        source_variants = tuple(
            variant
            for variant in screens_by_id[attempt.screen_id].variants
            if variant.observation_id == attempt.source_observation_id
        )
        if len(source_variants) != 1:
            raise ValueError("An unsuccessful attempt must reference its source observation.")
        source_variant = source_variants[0]
        has_wrong_grounding_dimensions = attempt.grounded_point is not None and (
            attempt.screenshot_width_px != source_variant.presentation.width_px
            or attempt.screenshot_height_px != source_variant.presentation.height_px
        )
        if has_wrong_grounding_dimensions:
            raise ValueError("An unsuccessful-attempt point must match its source screenshot.")
        if attempt.source_run_id not in self.source_run_ids:
            raise ValueError("An unsuccessful attempt must reference a source memory run.")

    def _observation_ids(self) -> list[RecordIdentifier]:
        return [variant.observation_id for screen in self.screens for variant in screen.variants]

    def _validate_knowledge(
        self,
        screen_ids: set[RecordIdentifier],
        transition_ids: set[RecordIdentifier],
        observation_ids: list[RecordIdentifier],
    ) -> None:
        validate_app_knowledge(
            self.app_knowledge,
            screen_ids,
            transition_ids,
            set(observation_ids),
            set(self.source_run_ids),
        )
        validate_retained_observations(observation_ids)

    def _validate_screen_masks(self) -> None:
        masks_by_shape: dict[tuple[DeviceOrientation | None, int, int], str | None] = {}
        for screen in self.screens:
            for variant in screen.variants:
                presentation = variant.presentation
                shape = (
                    presentation.orientation,
                    presentation.width_px,
                    presentation.height_px,
                )
                mask = presentation.fingerprint.system_region_mask_sha256
                if shape in masks_by_shape and masks_by_shape[shape] != mask:
                    raise ValueError(
                        "A compatible device shape must use one canonical effective mask."
                    )
                masks_by_shape[shape] = mask

    def _validate_retention_limits(self) -> None:
        referenced_observations = referenced_memory_observation_ids(
            self.transitions,
            self.unsuccessful_attempts,
        )
        maximum_optional = self.compatibility.retention.maximum_variants_per_screen
        if any(
            sum(
                variant.observation_id not in referenced_observations for variant in screen.variants
            )
            > maximum_optional
            for screen in self.screens
        ):
            raise ValueError("A logical screen exceeds the optional-variant retention limit.")
        if len(self.transitions) > self.compatibility.retention.maximum_transitions:
            raise ValueError("App memory exceeds a graph retention limit.")


def derived_memory_identifier(namespace: str, *parts: str) -> str:
    """Return one deterministic identifier for derived memory evidence."""
    content = "\x1f".join((namespace, *parts)).encode("utf-8")
    return sha256(content).hexdigest()[:32]


def build_memory_compatibility(
    application: ApplicationDescriptor,
    setup: ApplicationRequest,
) -> MemoryCompatibility:
    """Build reusable-memory identity from observed app and requested setup."""
    setup_contract = json.dumps(
        {
            "install_mode": setup.install_mode.value,
            "retain_installation": setup.retain_installation,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return MemoryCompatibility(
        application=EvidenceApplicationIdentity(
            platform=application.platform,
            application_id=application.application_id,
            version=application.version,
            build=application.build,
            artifact_sha256=application.artifact_sha256,
        ),
        setup_contract_sha256=sha256(setup_contract).hexdigest(),
    )
