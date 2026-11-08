"""Normalize action-scoped application failures for agent recovery."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from yewo.agent.actuation import (
    ActionOutcome,
    ActionOutcomeDisposition,
    ActionOutcomeSignal,
)
from yewo.artifacts import RecordIdentifier
from yewo.devices.diagnostics import (
    ApplicationDiagnosticClassification,
    ApplicationFailureKind,
    classify_application_failures,
)
from yewo.devices.logs import DeviceLogCheckpointKind, DeviceLogDrainReceipt


class ActionApplicationFailure(BaseModel):
    """A platform-established application failure associated with one action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    classification: ApplicationDiagnosticClassification
    failure_kinds: tuple[ApplicationFailureKind, ...] = Field(min_length=1)
    summary: str = Field(min_length=1, max_length=512)
    source_artifact_ids: tuple[RecordIdentifier, ...] = ()
    diagnostic_hints: tuple[str, ...] = Field(default=(), max_length=20)
    recovery_succeeded: bool = False

    @model_validator(mode="after")
    def _validate_classification(self) -> Self:
        if self.classification is not classify_application_failures(self.failure_kinds):
            raise ValueError("The action failure classification does not match its evidence.")
        return self


def planner_failure_context(
    failure: ActionApplicationFailure,
    *,
    include_summary: bool,
) -> dict[str, object]:
    """Render only the trusted failure facts a planner can act on."""
    context: dict[str, object] = {
        "classification": failure.classification.value,
        "recovery_succeeded": failure.recovery_succeeded,
    }
    if include_summary:
        context["summary"] = failure.summary
        if failure.diagnostic_hints:
            context["diagnostic_hints"] = failure.diagnostic_hints
    return context


def action_application_failure(
    receipt: DeviceLogDrainReceipt,
) -> ActionApplicationFailure | None:
    """Return a concise failure established by one action checkpoint."""
    if receipt.checkpoint_kind is not DeviceLogCheckpointKind.ACTION:
        raise ValueError("Action diagnostics require an action checkpoint receipt.")
    sources = tuple(source for source in receipt.sources if source.observed_failure_kinds)
    failure_kinds = tuple(
        failure_kind for source in sources for failure_kind in source.observed_failure_kinds
    )
    if not failure_kinds:
        return None
    artifacts = tuple(
        artifact
        for source in sources
        for artifact in source.artifacts
        if artifact.failure_kind is not None
    )
    classification = classify_application_failures(failure_kinds)
    details = tuple(
        dict.fromkeys(
            " ".join(
                part
                for part in (artifact.platform_reason, artifact.platform_code)
                if part is not None
            )
            for artifact in artifacts
        )
    )
    diagnostic_hints = tuple(
        dict.fromkeys(hint for source in receipt.sources for hint in source.diagnostic_hints)
    )
    detail_suffix = f" Platform evidence: {'; '.join(details)}." if any(details) else ""
    return ActionApplicationFailure(
        classification=classification,
        failure_kinds=failure_kinds,
        summary=f"{_classification_summary(classification)}{detail_suffix}",
        source_artifact_ids=tuple(artifact.artifact_id for artifact in artifacts),
        diagnostic_hints=diagnostic_hints,
    )


def outcome_with_application_failure(
    outcome: ActionOutcome,
    failure: ActionApplicationFailure,
) -> ActionOutcome:
    """Make trusted process-failure evidence override apparent visual success."""
    return outcome.model_copy(
        update={
            "disposition": ActionOutcomeDisposition.OBSERVED,
            "signals": frozenset((*outcome.signals, ActionOutcomeSignal.APP_EXITED)),
            "reason": failure.summary,
        }
    )


def _classification_summary(classification: ApplicationDiagnosticClassification) -> str:
    return {
        ApplicationDiagnosticClassification.APPLICATION_CRASH: (
            "Platform diagnostics established that the application crashed after this action."
        ),
        ApplicationDiagnosticClassification.APPLICATION_ANR: (
            "Platform diagnostics established that the application stopped responding after "
            "this action."
        ),
        ApplicationDiagnosticClassification.MULTIPLE_APPLICATION_FAILURES: (
            "Platform diagnostics established multiple application failures after this action."
        ),
        ApplicationDiagnosticClassification.OTHER_APPLICATION_TERMINATION: (
            "Platform diagnostics established that the application terminated after this action."
        ),
        ApplicationDiagnosticClassification.UNCLASSIFIED_APPLICATION_TERMINATION: (
            "Platform diagnostics established an unclassified application termination after "
            "this action."
        ),
    }[classification]
