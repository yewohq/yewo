"""Materialize app packages from a versioned, checksum-bound manifest."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from yewo.errors import UsageError

CHUNK_BYTES = 1024 * 1024


class ArtifactSource(BaseModel):
    """Where one exact package comes from."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["download", "build"]
    project_url: HttpUrl
    download_url: HttpUrl | None = None
    revision: str | None = None
    instructions: str | None = None

    @model_validator(mode="after")
    def _validate_download_location(self) -> "ArtifactSource":
        if self.kind == "download" and self.download_url is None:
            raise ValueError("a download source needs download_url")
        return self


class AppArtifact(BaseModel):
    """Identity and local destination for one deployable app package."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    platform: Literal["android", "ios"]
    version: str
    version_code: str | None = None
    architectures: list[str] = Field(min_length=1)
    path: str
    size_bytes: int | None = Field(default=None, ge=1)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    tree_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    license: str
    source: ArtifactSource

    @model_validator(mode="after")
    def _validate_identity(self) -> "AppArtifact":
        relative_path = PurePosixPath(self.path)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("path must stay beneath the manifest directory")
        if not relative_path.parts or relative_path.parts[0] != self.platform:
            raise ValueError("path must start with the artifact platform")
        if (self.sha256 is None) == (self.tree_sha256 is None):
            raise ValueError("declare exactly one of sha256 or tree_sha256")
        if self.tree_sha256 is not None and self.size_bytes is not None:
            raise ValueError("directory artifacts do not declare size_bytes")
        if self.source.download_url is not None and self.sha256 is None:
            raise ValueError("downloadable artifacts must be checksum-bound files")
        return self


class AppManifest(BaseModel):
    """The complete set of application artifacts in one corpus revision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    artifacts: list[AppArtifact] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_artifacts(self) -> "AppManifest":
        ids = [artifact.id for artifact in self.artifacts]
        paths = [artifact.path for artifact in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ValueError("artifact IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("artifact paths must be unique")
        return self


@dataclass(frozen=True)
class MaterializationItem:
    artifact_id: str
    status: Literal["ready", "downloaded", "unavailable"]
    detail: str


@dataclass(frozen=True)
class MaterializationReport:
    items: tuple[MaterializationItem, ...]

    @property
    def ready_count(self) -> int:
        return sum(item.status == "ready" for item in self.items)

    @property
    def downloaded_count(self) -> int:
        return sum(item.status == "downloaded" for item in self.items)

    @property
    def unavailable_count(self) -> int:
        return sum(item.status == "unavailable" for item in self.items)


def default_manifest_path() -> Path:
    """Resolve the repository manifest or its installed wheel copy."""
    repository_path = Path(__file__).resolve().parents[3] / "eval" / "apps" / "manifest.yaml"
    if repository_path.is_file():
        return repository_path
    packaged = resources.files("yewo.evaluation").joinpath("manifest.yaml")
    if not packaged.is_file():
        raise UsageError("Yewo's evaluation app manifest is not installed.")
    return Path(str(packaged))


def default_app_root(manifest_path: Path) -> Path:
    """Keep repository packages beside their manifest and installed ones in the project."""
    if manifest_path.parent.name == "apps" and manifest_path.parent.parent.name == "eval":
        return manifest_path.parent
    return Path.cwd() / "eval" / "apps"


def load_manifest(path: Path) -> AppManifest:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        return AppManifest.model_validate(document)
    except OSError as exc:
        raise UsageError(f"Could not read evaluation app manifest {path}: {exc}") from exc
    except (TypeError, ValueError, yaml.YAMLError) as exc:
        raise UsageError(f"Invalid evaluation app manifest {path}: {exc}") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_tree_sha256(path: Path) -> str:
    """Hash relative names, lengths, and bytes for a directory without an archive."""
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        size = item.stat().st_size
        digest.update(relative)
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with item.open("rb") as source:
            for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _verification_problem(artifact: AppArtifact, path: Path) -> str | None:
    if artifact.sha256 is not None:
        if not path.is_file():
            return "file is missing"
        if artifact.size_bytes is not None and path.stat().st_size != artifact.size_bytes:
            return "file size does not match the manifest"
        if file_sha256(path) != artifact.sha256:
            return "SHA-256 does not match the manifest"
        return None
    if not path.is_dir():
        return "app bundle is missing"
    if directory_tree_sha256(path) != artifact.tree_sha256:
        return "app bundle tree hash does not match the manifest"
    return None


def _selected_artifacts(
    manifest: AppManifest,
    platform: str | None,
    app_ids: frozenset[str],
) -> tuple[AppArtifact, ...]:
    known_ids = {artifact.id for artifact in manifest.artifacts}
    unknown = app_ids - known_ids
    if unknown:
        raise UsageError(f"Unknown evaluation app ID: {', '.join(sorted(unknown))}")
    selected = tuple(
        artifact
        for artifact in manifest.artifacts
        if (platform is None or artifact.platform == platform)
        and (not app_ids or artifact.id in app_ids)
    )
    if not selected:
        raise UsageError("No evaluation apps matched the requested selection.")
    return selected


def _download(
    artifact: AppArtifact,
    destination: Path,
    opener: Callable = urlopen,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        str(artifact.source.download_url),
        headers={"User-Agent": "yewo-eval-setup/1"},
    )
    temporary_path: Path | None = None
    try:
        with (
            opener(request, timeout=60) as response,
            tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".part", dir=destination.parent,
                delete=False,
            ) as temporary,
        ):
            temporary_path = Path(temporary.name)
            while chunk := response.read(CHUNK_BYTES):
                temporary.write(chunk)
        problem = _verification_problem(artifact, temporary_path)
        if problem is not None:
            raise ValueError(problem)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def materialize_apps(
    manifest: AppManifest,
    app_root: Path,
    *,
    platform: str | None = None,
    app_ids: frozenset[str] = frozenset(),
    verify_only: bool = False,
    opener: Callable = urlopen,
) -> MaterializationReport:
    items = []
    for artifact in _selected_artifacts(manifest, platform, app_ids):
        destination = app_root / artifact.path
        problem = _verification_problem(artifact, destination)
        if problem is None:
            items.append(MaterializationItem(artifact.id, "ready", str(destination)))
            continue
        if verify_only:
            items.append(MaterializationItem(artifact.id, "unavailable", problem))
            continue
        if artifact.source.download_url is None:
            detail = artifact.source.instructions or "build this artifact from its pinned source"
            items.append(MaterializationItem(artifact.id, "unavailable", detail))
            continue
        try:
            _download(artifact, destination, opener)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            items.append(MaterializationItem(artifact.id, "unavailable", str(exc)))
        else:
            items.append(MaterializationItem(artifact.id, "downloaded", str(destination)))
    return MaterializationReport(tuple(items))
