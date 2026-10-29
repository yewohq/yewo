"""Load held-out screenshots and reproducible identity perturbations."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import imagehash
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from yewo.discovery.identity_calibration import (
    IdentityCalibrationThresholds,
    ScreenFamilyFingerprint,
    ScreenFamilyLabel,
    ScreenFamilyPair,
)

EVAL_ROOT = Path(__file__).resolve().parents[2] / "eval"
MANIFEST_PATH = EVAL_ROOT / "identity" / "manifest.json"


class IdentityCorpusError(Exception):
    pass


class SampleFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    screenshot: str
    screenshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transform: Literal[
        "none",
        "status_indicator",
        "transient_banner",
        "brightness",
        "mirror",
        "viewport_resize",
        "animation_shift",
    ] = "none"
    structure_key: str = Field(min_length=1)
    app_scope: str = Field(min_length=1)
    platform: Literal["android", "ios"]
    viewport_class: str = Field(min_length=1)
    traits: frozenset[str] = Field(min_length=1)


class PairFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    left: str
    right: str
    label: ScreenFamilyLabel


class IdentityManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    provenance: str = Field(min_length=1)
    training_overlap: Literal["excluded"]
    required_traits: frozenset[str] = Field(min_length=1)
    thresholds: IdentityCalibrationThresholds
    samples: tuple[SampleFile, ...] = Field(min_length=1)
    pairs: tuple[PairFile, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_references(self) -> "IdentityManifest":
        sample_ids = [sample.id for sample in self.samples]
        pair_ids = [pair.id for pair in self.pairs]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Identity sample identifiers must be unique.")
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Identity pair identifiers must be unique.")
        known_samples = set(sample_ids)
        if any(
            pair.left not in known_samples or pair.right not in known_samples
            for pair in self.pairs
        ):
            raise ValueError("Identity pairs require known samples.")
        observed_traits = set().union(*(sample.traits for sample in self.samples))
        if not self.required_traits.issubset(observed_traits):
            raise ValueError("The identity corpus is missing a required trait.")
        return self


def read_identity_corpus(
    path: Path = MANIFEST_PATH,
    eval_root: Path = EVAL_ROOT,
) -> tuple[tuple[ScreenFamilyPair, ...], str, IdentityCalibrationThresholds]:
    try:
        content = path.read_bytes()
        manifest = IdentityManifest.model_validate_json(content)
    except (OSError, ValidationError, ValueError) as exc:
        raise IdentityCorpusError(f"{path}: {exc}") from exc
    samples = {
        sample.id: _build_fingerprint(sample, eval_root)
        for sample in manifest.samples
    }
    pairs = tuple(
        ScreenFamilyPair(
            pair_id=pair.id,
            left=samples[pair.left],
            right=samples[pair.right],
            label=pair.label,
        )
        for pair in manifest.pairs
    )
    screenshot_digests = sorted(sample.screenshot_sha256 for sample in manifest.samples)
    corpus_payload = content + b"\0" + "\n".join(screenshot_digests).encode()
    corpus_sha256 = hashlib.sha256(corpus_payload).hexdigest()
    return pairs, corpus_sha256, manifest.thresholds


def _build_fingerprint(sample: SampleFile, eval_root: Path) -> ScreenFamilyFingerprint:
    screenshot = (eval_root / sample.screenshot).resolve()
    if eval_root.resolve() not in screenshot.parents or not screenshot.is_file():
        raise IdentityCorpusError(f"{sample.id}: unsafe or missing screenshot")
    image_bytes = screenshot.read_bytes()
    if hashlib.sha256(image_bytes).hexdigest() != sample.screenshot_sha256:
        raise IdentityCorpusError(f"{sample.id}: screenshot digest mismatch")
    with Image.open(screenshot) as source:
        transformed = _transform(source.convert("RGB"), sample.transform)
        phash = str(imagehash.phash(transformed, hash_size=16))
    structure_sha256 = hashlib.sha256(sample.structure_key.encode()).hexdigest()
    return ScreenFamilyFingerprint(
        sample_id=sample.id,
        app_scope=sample.app_scope,
        platform=sample.platform,
        viewport_class=sample.viewport_class,
        phash=phash,
        structure_sha256=structure_sha256,
        traits=sample.traits,
    )


def _transform(image: Image.Image, transform: str) -> Image.Image:
    transformed = image.copy()
    if transform == "none":
        return transformed
    if transform == "status_indicator":
        ImageDraw.Draw(transformed).rectangle(
            (0, 0, transformed.width, max(12, transformed.height // 80)),
            fill="white",
        )
        return transformed
    if transform == "transient_banner":
        top = transformed.height // 7
        ImageDraw.Draw(transformed).rectangle(
            (transformed.width // 10, top, transformed.width * 9 // 10, top + 120),
            fill="gray",
        )
        return transformed
    if transform == "brightness":
        return ImageEnhance.Brightness(transformed).enhance(0.55)
    if transform == "mirror":
        return ImageOps.mirror(transformed)
    if transform == "viewport_resize":
        return transformed.resize(
            (max(1, transformed.width * 3 // 4), transformed.height),
        )
    if transform == "animation_shift":
        shifted = Image.new("RGB", transformed.size, "white")
        shifted.paste(transformed, (3, 0))
        return shifted
    raise IdentityCorpusError(f"Unknown identity transform: {transform}")
