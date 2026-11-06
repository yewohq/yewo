"""Typed model observations of controls visible on an application screen."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RequiredControlBasis(StrEnum):
    """Trusted evidence that can make one visible control mandatory."""

    OBJECTIVE_OR_OPEN_WORK = "objective_or_open_work"
    APPLICATION_REFUSAL = "application_refusal"
    TRUSTED_INSTRUCTION = "trusted_instruction"
    INPUT_CONTRACT = "input_contract"


class UnresolvedControl(BaseModel):
    """One unresolved condition tied to a reported visible control."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    control_name: str = Field(min_length=1, max_length=200)
    required_by: RequiredControlBasis
    requirement_evidence: str = Field(
        min_length=1,
        max_length=4_000,
        description=(
            "An exact objective, open-work item, trusted instruction, application-feedback "
            "reason, or declared input name copied from the verifier context."
        ),
    )
    reason: str = Field(min_length=1, max_length=500)
