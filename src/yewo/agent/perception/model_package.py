"""Resolve a pinned detector package from the Hugging Face Hub cache."""

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from yewo.agent.perception.model_manifest import (
    PerceptionModelManifest,
    load_perception_model_manifest,
)

MANIFEST_FILENAME = "perception.json"
REFERENCE_PATTERN = re.compile(
    r"^hf://(?P<repo_id>[^/@\s]+/[^@\s]+)@(?P<revision>[0-9a-f]{40})$"
)


class ModelPackageError(ValueError):
    """Raised when a model reference or downloaded package is not trustworthy."""


@dataclass(frozen=True)
class HuggingFaceModelReference:
    repo_id: str
    revision: str

    @property
    def value(self) -> str:
        return f"hf://{self.repo_id}@{self.revision}"


@dataclass(frozen=True)
class ResolvedModelPackage:
    reference: HuggingFaceModelReference
    manifest: PerceptionModelManifest
    weights_path: Path


HubDownload = Callable[..., str]


def parse_model_reference(value: str) -> HuggingFaceModelReference:
    """Parse one immutable Hub reference. Reject branches and tags."""
    match = REFERENCE_PATTERN.fullmatch(value)
    if match is None:
        raise ModelPackageError(
            "perception model must be hf://owner/repository@<40-character commit>"
        )
    return HuggingFaceModelReference(**match.groupdict())


def resolve_model_package(
    value: str,
    download: HubDownload | None = None,
) -> ResolvedModelPackage:
    """Download and validate a manifest and its weights from one Hub commit."""
    reference = parse_model_reference(value)
    if download is None:
        from huggingface_hub import hf_hub_download

        download = hf_hub_download

    manifest_path = Path(
        download(
            repo_id=reference.repo_id,
            filename=MANIFEST_FILENAME,
            revision=reference.revision,
        )
    )
    manifest = load_perception_model_manifest(manifest_path)
    weights_path = Path(
        download(
            repo_id=reference.repo_id,
            filename=manifest.weights_file,
            revision=reference.revision,
        )
    )
    actual_sha256 = _sha256(weights_path)
    if actual_sha256 != manifest.sha256:
        raise ModelPackageError(
            f"model weights checksum is {actual_sha256}. Manifest expects {manifest.sha256}"
        )
    return ResolvedModelPackage(reference, manifest, weights_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as weights_file:
        for chunk in iter(lambda: weights_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
