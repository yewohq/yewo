"""Read the frozen screenshots and their human-reviewed expectations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

EVAL_ROOT = Path(__file__).resolve().parents[2] / "eval" / "perception"
MANIFEST_PATH = EVAL_ROOT / "manifest.json"
PNG_HEADER = b"\x89PNG\r\n\x1a\n"


class CorpusError(Exception):
    """The perception corpus is missing, unsafe, or malformed."""


class TargetPoint(BaseModel):
    """One human-reviewed point that an actionable detection must cover."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    x: int = Field(ge=0)
    y: int = Field(ge=0)


class Provenance(BaseModel):
    """Where a fixture came from and whether it is independent of training."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    training_overlap: Literal["excluded"]


class CaseFile(BaseModel):
    """One screenshot and the observations expected from it."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    screenshot: str
    platform: Literal["android", "ios"]
    traits: set[str] = Field(min_length=1)
    keyboard_visible: bool
    expected_text: list[str] = Field(min_length=1)
    target_points: list[TargetPoint] = Field(min_length=1)
    provenance: Provenance

    @field_validator("screenshot")
    @classmethod
    def _plain_png_name(cls, value: str) -> str:
        if Path(value).name != value or not value.endswith(".png"):
            raise ValueError("must be a plain .png file name")
        return value


class ManifestFile(BaseModel):
    """The versioned set of cases used for detector comparisons."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    required_platforms: set[Literal["android", "ios"]]
    required_traits: set[str]
    cases: list[CaseFile] = Field(min_length=1)

    @field_validator("cases")
    @classmethod
    def _unique_case_ids(cls, cases: list[CaseFile]) -> list[CaseFile]:
        ids = [case.id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case ids must be unique")
        return cases


class PerceptionCase(BaseModel):
    """A validated case with a resolved screenshot and stable digest."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    screenshot: Path
    platform: Literal["android", "ios"]
    traits: set[str]
    keyboard_visible: bool
    expected_text: list[str]
    target_points: list[TargetPoint]
    provenance: Provenance | None = None
    digest: str


class PerceptionCorpus(BaseModel):
    """The validated corpus plus any declared coverage it does not meet."""

    schema_version: int
    cases: list[PerceptionCase]
    missing_platforms: set[str]
    missing_traits: set[str]


def read_corpus(path: Path = MANIFEST_PATH) -> PerceptionCorpus:
    """Read every case and refuse missing or unsafe screenshot inputs."""
    try:
        raw = path.read_bytes()
        manifest = ManifestFile.model_validate_json(raw)
    except (OSError, ValidationError, ValueError) as exc:
        raise CorpusError(f"{path}: {exc}") from exc

    screens = path.parent / "screens"
    cases = [_resolve_case(case, screens) for case in manifest.cases]
    platforms = {case.platform for case in cases}
    traits = set().union(*(case.traits for case in cases))
    return PerceptionCorpus(
        schema_version=manifest.schema_version,
        cases=cases,
        missing_platforms=set(manifest.required_platforms) - platforms,
        missing_traits=set(manifest.required_traits) - traits,
    )


def _resolve_case(case: CaseFile, screens: Path) -> PerceptionCase:
    directory = screens.resolve()
    screenshot = (screens / case.screenshot).resolve()
    if screenshot.parent != directory or not screenshot.is_file() or screenshot.is_symlink():
        raise CorpusError(f"{case.id}: unsafe or missing screenshot {case.screenshot}")
    image_bytes = screenshot.read_bytes()
    if not image_bytes.startswith(PNG_HEADER):
        raise CorpusError(f"{case.id}: {case.screenshot} is not a PNG")
    expectation = case.model_dump(mode="json")
    expectation["traits"] = sorted(case.traits)
    expectation_bytes = json.dumps(expectation, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hashlib.sha256(expectation_bytes + b"\0" + image_bytes).hexdigest()[:16]
    values = case.model_dump(exclude={"screenshot"})
    return PerceptionCase(**values, screenshot=screenshot, digest=digest)
