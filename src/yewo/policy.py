"""Run-level policy-check evidence shared by checks and reports."""

from enum import Enum
import re
from typing import Iterable

from pydantic import BaseModel, Field, field_validator, model_validator

from yewo.agent.device.platform import Platform


SEMANTIC_CHECK_ID = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
"""A durable namespace and name, such as ``android.debuggable``."""


class PolicyCheckStatus(str, Enum):
    """Every result a run-level deterministic check can produce."""

    PASSED = "passed"
    FAILED = "failed"
    WAIVED = "waived"
    UNSUPPORTED = "unsupported"
    EVIDENCE_MISSING = "evidence_missing"


class PolicyCheckId(str, Enum):
    """Stable IDs for the package-policy checks Yewo owns."""

    ANDROID_DEBUGGABLE = "android.debuggable"
    ANDROID_BACKUP = "android.allow_backup"
    ANDROID_CLEARTEXT = "android.cleartext_traffic"
    ANDROID_EXPORTED_COMPONENTS = "android.exported_components"
    IOS_GET_TASK_ALLOW = "ios.get_task_allow"
    IOS_FILE_SHARING = "ios.file_sharing"
    IOS_APP_TRANSPORT_SECURITY = "ios.app_transport_security"
    IOS_EXTERNAL_ENTRY_POINTS = "ios.external_entry_points"


POLICY_CHECK_IDS_BY_PLATFORM = {
    Platform.ANDROID: tuple(
        check_id.value for check_id in PolicyCheckId if check_id.value.startswith("android.")
    ),
    Platform.IOS: tuple(
        check_id.value for check_id in PolicyCheckId if check_id.value.startswith("ios.")
    ),
}


def bounded_policy_text(value: str, limit: int) -> str:
    """Limit package-owned evidence before the report contract accepts it."""
    text = value.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


class PolicyCheckReport(BaseModel):
    """One deterministic check that does not belong to a scenario step."""

    check_id: str = Field(
        max_length=128, description="A stable semantic ID. It must not depend on list position."
    )
    status: PolicyCheckStatus
    platform: Platform
    source: str = Field(
        max_length=1024,
        description="The command or file that produced the evidence.",
    )
    evidence: str = Field(
        max_length=4096,
        description="The bounded evidence that settled the check.",
    )
    waiver_reason: str | None = Field(default=None, max_length=2048)

    @field_validator("check_id")
    @classmethod
    def _validate_check_id(cls, value: str) -> str:
        check_id = value.strip()
        if not SEMANTIC_CHECK_ID.fullmatch(check_id):
            raise ValueError(
                "check_id must be a semantic namespace and name, such as 'android.debuggable'."
            )
        return check_id

    @field_validator("source", "evidence")
    @classmethod
    def _validate_evidence_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("policy-check evidence fields must not be empty.")
        return text

    @model_validator(mode="after")
    def _validate_waiver_reason(self) -> "PolicyCheckReport":
        reason = self.waiver_reason.strip() if self.waiver_reason else None
        if self.status is PolicyCheckStatus.WAIVED and not reason:
            raise ValueError("A waived policy check must include a waiver reason.")
        if self.status is not PolicyCheckStatus.WAIVED and reason:
            raise ValueError("Only a waived policy check may include a waiver reason.")
        self.waiver_reason = reason
        return self


def policy_checks_succeeded(checks: Iterable[PolicyCheckReport]) -> bool:
    """Whether every check reached an accepted result."""
    accepted = {PolicyCheckStatus.PASSED, PolicyCheckStatus.WAIVED}
    return all(check.status in accepted for check in checks)
