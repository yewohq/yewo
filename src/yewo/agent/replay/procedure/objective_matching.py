"""Match a requested objective to an audited stored procedure."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yewo.agent.discovery.records import DiscoveryObjective
from yewo.agent.replay.procedure.store import StoredProcedureBundle
from yewo.artifacts import (
    ArtifactRetention,
    ArtifactSensitivity,
    RecordIdentifier,
    new_record_identifier,
)
from yewo.models import (
    ModelCallDisposition,
    ModelClient,
    ModelMessage,
    ModelMessageRole,
    ModelRequest,
    ModelRole,
    TextModelContent,
    build_model_response_contract,
)


MAXIMUM_OBJECTIVE_CANDIDATES = 20

OBJECTIVE_MATCHING_SYSTEM_PROMPT = """\
You compare one requested mobile-application objective with audited stored procedure objectives.
Judge the complete objective and every completion condition. Wording can differ.

Return equivalent only when the requested objective and one candidate require the same observable
outcome. Their completion conditions must align one-to-one in the supplied order. A narrower,
broader, overlapping, unrelated, or uncertain candidate is not equivalent. Do not infer current
application state. Do not claim that a stored procedure completed the requested task in this run.
Select at most one candidate. Return only the JSON response.
"""

OBJECTIVE_MATCHING_USER_PROMPT = """\
Compare the requested objective with the stored candidates. Return request_id exactly as supplied.
Context JSON follows:
"""


class ObjectiveRelationship(StrEnum):
    """How one stored objective relates to the requested objective."""

    EQUIVALENT = "equivalent"
    CURRENT_IS_NARROWER = "current_is_narrower"
    CURRENT_IS_BROADER = "current_is_broader"
    OVERLAPS = "overlaps"
    UNRELATED = "unrelated"
    UNCERTAIN = "uncertain"


class ObjectiveMatchCandidate(BaseModel):
    """One audited stored procedure exposed to objective matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    procedure_id: RecordIdentifier
    objective_id: RecordIdentifier
    objective: str = Field(min_length=1, max_length=4_000)
    completion_conditions: tuple[str, ...] = Field(min_length=1, max_length=100)


class ObjectiveMatchOutput(BaseModel):
    """Untrusted structured decision returned by the objective-matching LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: RecordIdentifier
    procedure_id: RecordIdentifier | None = None
    relationship: ObjectiveRelationship
    reason: str = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def _validate_equivalent_selection(self) -> "ObjectiveMatchOutput":
        if self.relationship is ObjectiveRelationship.EQUIVALENT:
            if self.procedure_id is None:
                raise ValueError("An equivalent objective match must select one procedure.")
        elif self.procedure_id is not None:
            raise ValueError("A non-equivalent objective match cannot select a procedure.")
        return self


class ProcedureSelectionMethod(StrEnum):
    """How replay preparation selected one stored procedure."""

    EXACT = "exact"
    LLM_EQUIVALENT = "llm_equivalent"


class ReplayProcedureSelection(BaseModel):
    """One stored procedure authorized for guarded live replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    procedure_id: RecordIdentifier
    stored_objective_id: RecordIdentifier
    stored_objective: str = Field(min_length=1, max_length=4_000)
    method: ProcedureSelectionMethod
    preparation_model_requests: int = Field(default=0, ge=0, le=1)


class ObjectiveMatcher:
    """Use one bounded LLM call after exact procedure lookup misses."""

    def __init__(
        self,
        model_client: ModelClient,
        model_profile_id: RecordIdentifier,
        *,
        identifier_factory: Callable[[], str] = new_record_identifier,
    ) -> None:
        self._model_client = model_client
        self._model_profile_id = model_profile_id
        self._identifier_factory = identifier_factory
        self._model_request_count = 0

    @property
    def model_request_count(self) -> int:
        """Return the number of objective-matching requests made."""
        return self._model_request_count

    def match(
        self,
        objective: DiscoveryObjective,
        bundles: tuple[StoredProcedureBundle, ...],
    ) -> ReplayProcedureSelection | None:
        """Return one equivalent stored procedure, or fail closed."""
        candidates = _objective_candidates(bundles)
        if not candidates:
            return None
        request_id = self._identifier_factory()
        request = _model_request(
            objective,
            candidates,
            self._model_profile_id,
            request_id,
        )
        self._model_request_count += 1
        response = self._model_client.complete(request, ObjectiveMatchOutput)
        if (
            response.disposition is not ModelCallDisposition.COMPLETED
            or response.structured_output is None
        ):
            return None
        try:
            output = ObjectiveMatchOutput.model_validate(response.structured_output)
        except ValidationError:
            return None
        if (
            output.request_id != request_id
            or output.relationship is not ObjectiveRelationship.EQUIVALENT
            or output.procedure_id is None
        ):
            return None
        candidate_by_procedure_id = {item.procedure_id: item for item in candidates}
        candidate = candidate_by_procedure_id.get(output.procedure_id)
        if candidate is None:
            return None
        current_conditions = objective.completion_conditions or (objective.statement,)
        if len(current_conditions) != len(candidate.completion_conditions):
            return None
        return ReplayProcedureSelection(
            procedure_id=candidate.procedure_id,
            stored_objective_id=candidate.objective_id,
            stored_objective=candidate.objective,
            method=ProcedureSelectionMethod.LLM_EQUIVALENT,
            preparation_model_requests=1,
        )


def exact_procedure_selection(bundle: StoredProcedureBundle) -> ReplayProcedureSelection:
    """Describe one exact stored-procedure match without a model call."""
    return ReplayProcedureSelection(
        procedure_id=bundle.procedure.procedure_id,
        stored_objective_id=bundle.evidence.task.objective_id,
        stored_objective=bundle.evidence.task.objective,
        method=ProcedureSelectionMethod.EXACT,
    )


def _objective_candidates(
    bundles: tuple[StoredProcedureBundle, ...],
) -> tuple[ObjectiveMatchCandidate, ...]:
    unique: dict[str, ObjectiveMatchCandidate] = {}
    for bundle in bundles:
        task = bundle.evidence.task
        unique.setdefault(
            bundle.procedure.procedure_id,
            ObjectiveMatchCandidate(
                procedure_id=bundle.procedure.procedure_id,
                objective_id=task.objective_id,
                objective=task.objective,
                completion_conditions=task.completion_conditions,
            ),
        )
    return tuple(sorted(unique.values(), key=lambda item: item.procedure_id))[
        :MAXIMUM_OBJECTIVE_CANDIDATES
    ]


def _model_request(
    objective: DiscoveryObjective,
    candidates: tuple[ObjectiveMatchCandidate, ...],
    profile_id: RecordIdentifier,
    request_id: RecordIdentifier,
) -> ModelRequest:
    current_conditions = objective.completion_conditions or (objective.statement,)
    context = {
        "request_id": request_id,
        "requested_objective": {
            "objective_id": objective.objective_id,
            "objective": objective.statement,
            "completion_conditions": current_conditions,
        },
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }
    return ModelRequest(
        request_id=request_id,
        profile_id=profile_id,
        role=ModelRole.REPLAY_PREPARER,
        messages=(
            ModelMessage(
                role=ModelMessageRole.SYSTEM,
                content=(TextModelContent(text=OBJECTIVE_MATCHING_SYSTEM_PROMPT),),
            ),
            ModelMessage(
                role=ModelMessageRole.USER,
                content=(
                    TextModelContent(
                        text=OBJECTIVE_MATCHING_USER_PROMPT
                        + json.dumps(context, sort_keys=True, separators=(",", ":"))
                    ),
                ),
            ),
        ),
        response_contract=build_model_response_contract(
            "objective-matching-v1",
            ObjectiveMatchOutput.model_json_schema(),
        ),
        sensitivity=ArtifactSensitivity.DEVICE_CONTENT,
        retention=ArtifactRetention.RUN,
    )
