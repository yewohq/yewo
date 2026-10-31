"""Immutable records owned by one Yewo invocation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from yewo.applications import ApplicationRequest
from yewo.artifacts import RecordIdentifier
from yewo.devices import DeviceQuery


class RunMode(StrEnum):
    """A product workflow that the coordinator can start."""

    NORMAL = "normal"
    SCENARIO = "scenario"


class RunTerminalCategory(StrEnum):
    """The truthful top-level outcome of one invocation."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    BLOCKED_BY_POLICY = "blocked_by_policy"
    BLOCKED_BY_SETUP = "blocked_by_setup"
    APPLICATION_FAILURE = "application_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    INTERRUPTED = "interrupted"


class RunStage(StrEnum):
    """The coordinator stage active when a run reached its terminal."""

    STARTUP = "startup"
    DEVICE_SELECTION = "device_selection"
    DEVICE_PREPARATION = "device_preparation"
    APPLICATION_INSPECTION = "application_inspection"
    APPIUM_STARTUP = "appium_startup"
    AUTOMATION_STARTUP = "automation_startup"
    EXECUTION = "execution"
    REPORTING = "reporting"
    CLEANUP = "cleanup"
    FINISHED = "finished"


class AutomationServerRequest(BaseModel):
    """Validated caller settings for the Appium server boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    server_url: str = "http://127.0.0.1:4723"
    start_if_unavailable: bool = True
    startup_timeout_seconds: float = Field(default=30, gt=0)
    probe_timeout_seconds: float = Field(default=0.75, gt=0)
    poll_interval_seconds: float = Field(default=0.25, gt=0)

    @field_validator("server_url")
    @classmethod
    def _validate_server_url(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError as error:
            raise ValueError("The automation server URL has an invalid port.") from error
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "The automation server URL must contain only a scheme, host, port, and path."
            )
        return value.rstrip("/")


class RunRequest(BaseModel):
    """Validated caller intent for one isolated invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    mode: RunMode
    output_root: Path
    device: DeviceQuery
    application: ApplicationRequest
    automation_server: AutomationServerRequest = AutomationServerRequest()

    @field_validator("output_root")
    @classmethod
    def _validate_output_root(cls, value: Path) -> Path:
        if not value.parts or value.name in {"", ".", ".."}:
            raise ValueError("A run output root must name a directory.")
        return value

    @model_validator(mode="after")
    def _validate_platform(self) -> Self:
        if (
            self.device.platform is not None
            and self.device.platform is not self.application.platform
        ):
            raise ValueError("The requested device and application platforms must match.")
        return self


class RunStart(BaseModel):
    """Append-only evidence that one invocation owns a run directory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: RecordIdentifier
    request: RunRequest
    started_at: datetime

    @field_validator("started_at")
    @classmethod
    def _validate_started_at(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("A run start time must include a timezone.")
        return value


class RunTerminal(BaseModel):
    """One final top-level outcome preserved after cleanup attempts."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    run_id: RecordIdentifier
    category: RunTerminalCategory
    stage: RunStage
    reason: str = Field(min_length=1, max_length=1024)
    completed_at: datetime
    partial_evidence_available: bool
    preceding_category: RunTerminalCategory | None = None

    @model_validator(mode="after")
    def _validate_terminal(self) -> Self:
        if self.completed_at.utcoffset() is None:
            raise ValueError("A run completion time must include a timezone.")
        if self.category is RunTerminalCategory.COMPLETED and self.stage is not RunStage.FINISHED:
            raise ValueError("A completed run must finish every coordinator stage.")
        if self.category is not RunTerminalCategory.COMPLETED and self.stage is RunStage.FINISHED:
            raise ValueError("A failed or incomplete run must identify the affected stage.")
        if self.preceding_category is not None:
            cleanup_or_reporting_override = (
                self.stage in {RunStage.CLEANUP, RunStage.REPORTING}
                and self.category
                in {
                    RunTerminalCategory.INFRASTRUCTURE_FAILURE,
                    RunTerminalCategory.INTERRUPTED,
                }
                and self.preceding_category is not self.category
            )
            evidence_persistence_override = (
                self.stage is RunStage.EXECUTION
                and self.category is RunTerminalCategory.INFRASTRUCTURE_FAILURE
                and self.preceding_category is RunTerminalCategory.INTERRUPTED
            )
            if not (cleanup_or_reporting_override or evidence_persistence_override):
                raise ValueError(
                    "A preceding category requires a later infrastructure or cleanup override."
                )
        return self
