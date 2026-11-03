"""Contracts for deterministic discovery-run reconciliation."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import (
    ActionKind,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
    TextDeliveryEvidence,
    ActionProposal,
    SideEffectClass,
)
from yewo.artifacts import EvidenceApplicationIdentity, EvidenceDeviceIdentity, RecordIdentifier
from yewo.devices.focus import TextFieldContextSource, TextFieldType
from yewo.grounding import GroundingEvidence
from yewo.observations import DeviceOrientation, ScreenFingerprint


class EvidenceStepKind(StrEnum):
    """Whether a recorded step belongs to the application path."""

    PATH = "path"
    ENVIRONMENT = "environment"


class EvidenceResult(StrEnum):
    """Deterministic summary of recorded step evidence."""

    OBSERVED = "observed"
    NO_EFFECT = "no_effect"
    AMBIGUOUS = "ambiguous"
    NOT_CAPTURED = "not_captured"
    REFUSED = "refused"
    CRASHED = "crashed"
    TERMINAL = "terminal"
    UNKNOWN = "unknown"


class ControlRole(StrEnum):
    """A deterministic semantic role inferred from recorded planner text."""

    TEXT_FIELD = "text_field"
    STATIC_TEXT = "static_text"
    BUTTON = "button"
    IMAGE_BUTTON = "image_button"
    CHECKBOX = "checkbox"
    RADIO_BUTTON = "radio_button"
    SWITCH = "switch"
    SLIDER = "slider"
    DROPDOWN = "dropdown"
    OPTION = "option"
    MENU_ITEM = "menu_item"
    TAB = "tab"
    LINK = "link"
    LIST = "list"
    LIST_ITEM = "list_item"
    ROW = "row"
    CARD = "card"
    ICON = "icon"
    DATE_PICKER = "date_picker"
    TIME_PICKER = "time_picker"
    UNKNOWN = "unknown"


class ControlValueType(StrEnum):
    """The value shape implied by a control and its recorded input."""

    NONE = "none"
    TEXT = "text"
    INTEGER = "integer"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    ENUM = "enum"
    DATE = "date"
    TIME = "time"
    RANGE = "range"
    UNKNOWN = "unknown"


ControlInferenceInput = Literal[
    "recorded_action",
    "model_authored_target_expression",
    "recorded_input_contract",
]


class ControlInference(BaseModel):
    """Provenance for a control description inferred without an accessibility tree."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["recorded-control-parser-v1"] = "recorded-control-parser-v1"
    input_sources: tuple[ControlInferenceInput, ...] = Field(min_length=1, max_length=3)
    certainty: Literal["inferred", "unknown"]


class ControlInputTraits(BaseModel):
    """Secret-free platform traits for a text input control."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    secure: bool | None
    semantic_type: TextFieldType
    source: TextFieldContextSource

    @classmethod
    def from_delivery(
        cls,
        delivery: TextDeliveryEvidence | None,
    ) -> Self | None:
        if delivery is None:
            return None
        return cls(
            secure=delivery.is_secure,
            semantic_type=delivery.field_type,
            source=delivery.context_source,
        )


class EvidenceControl(BaseModel):
    """One inferred control description and its explicit provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    role: ControlRole
    label: str | None = Field(default=None, min_length=1, max_length=500)
    value_type: ControlValueType
    identity_key: str | None = Field(default=None, min_length=1, max_length=600)
    input_traits: ControlInputTraits | None = None
    inference: ControlInference


class TargetVisualDescriptor(BaseModel):
    """Compact perceptual evidence for one recorded target neighborhood."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["target-neighborhood-phash-v1"] = "target-neighborhood-phash-v1"
    hash_value: str = Field(pattern=r"^[0-9a-f]{16}$")
    patch_width_px: int = Field(gt=0)
    patch_height_px: int = Field(gt=0)
    maximum_hash_distance: int = Field(default=8, ge=0, le=64)
    search_radius_px: int = Field(default=96, gt=0, le=512)
    search_stride_px: int = Field(default=4, gt=0, le=32)
    minimum_runner_up_margin: int = Field(default=2, ge=0, le=64)


class VisualTileFingerprint(BaseModel):
    """One stable tile retained from a group of equivalent observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tile_index: int = Field(ge=0)
    hash_value: str = Field(pattern=r"^[0-9a-f]{16}$")


class StateVisualSignature(BaseModel):
    """Stable visual regions and an evidence-derived acceptance ratio."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["stable-tile-phash-v1"] = "stable-tile-phash-v1"
    column_count: int = Field(default=6, ge=2, le=20)
    row_count: int = Field(default=10, ge=2, le=30)
    stable_tiles: tuple[VisualTileFingerprint, ...] = Field(min_length=1, max_length=600)
    maximum_tile_distance: int = Field(default=4, ge=0, le=64)
    minimum_match_ratio: float = Field(ge=0, le=1)
    observation_count: int = Field(gt=0)


class EvidenceState(BaseModel):
    """A versioned group of replay-equivalent observations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state_id: RecordIdentifier
    representative_observation_id: RecordIdentifier
    observation_ids: tuple[RecordIdentifier, ...] = Field(min_length=1, max_length=1_000)
    foreground_application_id: str | None = Field(default=None, min_length=1)
    platform_context: str | None = Field(default=None, min_length=1)
    keyboard_visible: bool | None
    orientation: DeviceOrientation | None
    width_px: int = Field(gt=0)
    height_px: int = Field(gt=0)
    fingerprint: ScreenFingerprint
    fingerprints: tuple[ScreenFingerprint, ...] = Field(default=(), max_length=1_000)
    screenshot_artifact_id: RecordIdentifier
    visual_signature: StateVisualSignature | None = None

    @model_validator(mode="after")
    def _validate_fingerprints(self) -> Self:
        fingerprints = self.fingerprints or (self.fingerprint,)
        artifact_ids = [item.source_artifact_id for item in fingerprints]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("An evidence state cannot repeat a screen fingerprint.")
        if self.fingerprint not in fingerprints:
            raise ValueError("The representative fingerprint must belong to the evidence state.")
        return self

    def comparison_fingerprints(self) -> tuple[ScreenFingerprint, ...]:
        """Return all recorded variants, including legacy singleton evidence."""
        return self.fingerprints or (self.fingerprint,)


class EvidenceStep(BaseModel):
    """Lossless references and deterministic annotations for one source step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(gt=0)
    recorded_step_index: int = Field(gt=0)
    source_step_id: RecordIdentifier
    source_observation_id: RecordIdentifier
    destination_observation_id: RecordIdentifier | None = None
    source_state_id: RecordIdentifier
    destination_state_id: RecordIdentifier | None = None
    kind: EvidenceStepKind
    action: ActionKind
    proposal: ActionProposal
    control: EvidenceControl
    result: EvidenceResult
    outcome_disposition: ActionOutcomeDisposition | None = None
    signals: frozenset[ActionOutcomeSignal] = frozenset()
    text_delivery: TextDeliveryEvidence | None = None
    authorization_side_effect: SideEffectClass | None = None
    target_grounding: GroundingEvidence | None = None
    target_visual_descriptor: TargetVisualDescriptor | None = None
    destination_grounding: GroundingEvidence | None = None
    refused: bool = False
    application_failed: bool = False


DiagnosticInput = Literal[
    "recorded_action",
    "recorded_outcome",
    "grouped_state",
    "inferred_control",
]


class DiagnosticDerivation(BaseModel):
    """Versioned provenance for one deterministic diagnostic."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method: str = Field(min_length=1, max_length=100)
    inputs: tuple[DiagnosticInput, ...]
    uses_model_authored_expression: bool = False


class EvidenceDiagnostic(BaseModel):
    """One diagnostic that resolves to source steps."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    step_indexes: tuple[int, ...] = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=1_000)
    derivation: DiagnosticDerivation


class EvidenceDiagnostics(BaseModel):
    """Deterministic diagnostic groups for one complete timeline."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conflicts: tuple[EvidenceDiagnostic, ...] = ()
    no_effect_repeats: tuple[EvidenceDiagnostic, ...] = ()
    returned_to_same_state: tuple[EvidenceDiagnostic, ...] = ()
    data_entry: tuple[EvidenceDiagnostic, ...] = ()
    crashes: tuple[EvidenceDiagnostic, ...] = ()
    unexpected_signals: tuple[EvidenceDiagnostic, ...] = ()


class EvidenceRun(BaseModel):
    """Source-run identity and evidence counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RecordIdentifier
    application: EvidenceApplicationIdentity
    device: EvidenceDeviceIdentity | None = None
    step_count: int = Field(ge=0)
    observation_count: int = Field(gt=0)
    launch_state_id: RecordIdentifier


class EvidenceTask(BaseModel):
    """Task fields exposed to reconciliation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: RecordIdentifier
    objective: str = Field(min_length=1, max_length=4_000)
    open_work: tuple[str, ...] = Field(default=(), max_length=100)
    completion_conditions: tuple[str, ...] = Field(min_length=1, max_length=100)


class TimelineEntry(BaseModel):
    """One chronological link from a source step into grouped states."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: int = Field(gt=0)
    source_state_id: RecordIdentifier
    destination_state_id: RecordIdentifier | None = None


class EvidenceDocument(BaseModel):
    """A lossless code-built view of one saved discovery run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    evidence_id: RecordIdentifier
    run: EvidenceRun
    task: EvidenceTask
    state_grouping_method: Literal["replay-equivalent-state-v1"] = "replay-equivalent-state-v1"
    state_grouping_maximum_phash_distance: int = Field(ge=0, le=64)
    state_matching_minimum_phash_margin: int = Field(default=1, ge=0, le=64)
    states: dict[RecordIdentifier, EvidenceState]
    steps: tuple[EvidenceStep, ...]
    diagnostics: EvidenceDiagnostics
    timeline: tuple[TimelineEntry, ...]

    @model_validator(mode="after")
    def _validate_document_links(self) -> Self:
        expected = tuple(range(1, len(self.steps) + 1))
        if tuple(step.step for step in self.steps) != expected:
            raise ValueError("Evidence step selectors must be unique and chronological.")
        if self.run.step_count != len(self.steps):
            raise ValueError("The evidence step count does not match the timeline.")
        if set(self.states) != {state.state_id for state in self.states.values()}:
            raise ValueError("Evidence state keys must match state identifiers.")
        if self.run.launch_state_id not in self.states:
            raise ValueError("The launch state must exist in the evidence document.")
        if len(self.timeline) != len(self.steps):
            raise ValueError("Every evidence step requires one timeline entry.")
        observation_ids = tuple(
            observation_id
            for state in self.states.values()
            for observation_id in state.observation_ids
        )
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("An observation cannot belong to more than one evidence state.")
        if self.run.observation_count != len(observation_ids):
            raise ValueError("The evidence observation count does not match grouped states.")
        for step, timeline in zip(self.steps, self.timeline, strict=True):
            if step.source_state_id not in self.states:
                raise ValueError("A step source state is missing from the evidence document.")
            if (
                step.destination_state_id is not None
                and step.destination_state_id not in self.states
            ):
                raise ValueError("A step destination state is missing from the evidence document.")
            if (
                timeline.step != step.step
                or timeline.source_state_id != step.source_state_id
                or timeline.destination_state_id != step.destination_state_id
            ):
                raise ValueError("Timeline entries must match evidence steps.")
        return self


from yewo.agent.replay.reconciliation_compilation_records import (  # noqa: E402, F401
    CompilationOutcome,
    CompilationWarning,
    CompilerCheck,
    CompilerReason,
    CompilerRefusal,
    ProcedureCompilationReport,
    ProcedureCompilationStatus,
    ProcedureEffect,
    ProcedureExpectation,
    ReconcilerClaimVerdict,
    ReconcilerDecision,
    ReconcilerResponse,
    ReplayProcedure,
    ReplayProcedureStep,
)


