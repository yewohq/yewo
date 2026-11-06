"""Stable records emitted by the read-only Doctor command."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class DoctorStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class DoctorCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    status: DoctorStatus
    detail: str = Field(min_length=1)
    hint: str | None = Field(default=None, min_length=1)


class DoctorSection(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    title: str = Field(min_length=1)
    checks: tuple[DoctorCheck, ...]


class DoctorReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sections: tuple[DoctorSection, ...]

    @property
    def blocking(self) -> int:
        return sum(
            check.status is DoctorStatus.FAIL
            for section in self.sections
            for check in section.checks
        )

    @property
    def warnings(self) -> int:
        return sum(
            check.status is DoctorStatus.WARN
            for section in self.sections
            for check in section.checks
        )

    def json_payload(self) -> dict[str, object]:
        return {
            **self.model_dump(mode="json"),
            "blocking": self.blocking,
            "warnings": self.warnings,
        }
