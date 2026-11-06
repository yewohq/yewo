"""Records for planned and completed Setup actions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SetupStatus(StrEnum):
    OK = "ok"
    PLANNED = "planned"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


class SetupStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    status: SetupStatus
    detail: str = Field(min_length=1)
    hint: str | None = Field(default=None, min_length=1)
    command: tuple[str, ...] = ()


class SetupReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    platforms: tuple[str, ...]
    steps: tuple[SetupStep, ...]

    @property
    def failures(self) -> int:
        return sum(step.status is SetupStatus.FAIL for step in self.steps)
