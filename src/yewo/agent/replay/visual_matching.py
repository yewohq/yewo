"""Visual signatures and bounded target matching for compiled replay."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from io import BytesIO
from math import hypot

import imagehash
from PIL import Image, UnidentifiedImageError

from yewo.agent.replay.reconciliation.records import (
    StateVisualSignature,
    TargetVisualDescriptor,
    VisualTileFingerprint,
)
from yewo.grounding import GroundedPoint, NormalizedPoint, PixelPoint
from yewo.observations import Observation, ObservationCaptureError, ScreenshotReference


_ONE_SAMPLE_MISMATCH_ALLOWANCE = 0.2
_TWO_SAMPLE_MISMATCH_ALLOWANCE = 0.15
_MULTI_SAMPLE_MISMATCH_ALLOWANCE = 0.1


@dataclass(frozen=True)
class TargetMatch:
    """One accepted target point and its visual comparison evidence."""

    point: GroundedPoint
    hash_distance: int
    runner_up_distance: int | None
    relocated: bool


def build_observation_visual_signature(
    observations: Sequence[Observation],
    screenshot_loader: Callable[[ScreenshotReference], bytes] | None,
) -> StateVisualSignature | None:
    """Build a signature from persisted observation screenshots when available."""
    if screenshot_loader is None:
        return None
    screenshots: list[bytes] = []
    try:
        for observation in observations:
            screenshots.append(screenshot_loader(observation.redacted_screenshot))
    except ObservationCaptureError:
        return None
    return build_state_visual_signature(tuple(screenshots))


def build_state_visual_signature(
    screenshots: tuple[bytes, ...],
    *,
    column_count: int = 6,
    row_count: int = 10,
) -> StateVisualSignature | None:
    """Retain tiles that stayed perceptually unchanged across source observations."""
    images = tuple(_decode_image(content) for content in screenshots)
    if not images or any(image is None for image in images):
        return None
    decoded = tuple(image for image in images if image is not None)
    if len({image.size for image in decoded}) != 1:
        return None
    hashes_by_image = tuple(_tile_hashes(image, column_count, row_count) for image in decoded)
    stable_tiles: list[VisualTileFingerprint] = []
    for tile_index, reference_hash in enumerate(hashes_by_image[0]):
        if all(reference_hash - hashes[tile_index] == 0 for hashes in hashes_by_image[1:]):
            stable_tiles.append(
                VisualTileFingerprint(
                    tile_index=tile_index,
                    hash_value=str(reference_hash),
                )
            )
    if not stable_tiles:
        return None
    mismatch_allowance = _stable_tile_mismatch_allowance(len(decoded))
    return StateVisualSignature(
        column_count=column_count,
        row_count=row_count,
        stable_tiles=tuple(stable_tiles),
        minimum_match_ratio=1 - mismatch_allowance,
        observation_count=len(decoded),
    )


def _stable_tile_mismatch_allowance(observation_count: int) -> float:
    if observation_count == 1:
        return _ONE_SAMPLE_MISMATCH_ALLOWANCE
    if observation_count == 2:
        return _TWO_SAMPLE_MISMATCH_ALLOWANCE
    return _MULTI_SAMPLE_MISMATCH_ALLOWANCE


def state_visual_match_ratio(png: bytes, signature: StateVisualSignature) -> float | None:
    """Return the fraction of stable evidence tiles matched by a live screen."""
    image = _decode_image(png)
    if image is None:
        return None
    hashes = _tile_hashes(image, signature.column_count, signature.row_count)
    return _match_ratio(hashes, signature.stable_tiles, signature.maximum_tile_distance)


def state_visual_match_is_exact(png: bytes, signature: StateVisualSignature) -> bool:
    """Require every compiler-stable tile to match within the signature tolerance."""
    image = _decode_image(png)
    if image is None:
        return False
    hashes = _tile_hashes(image, signature.column_count, signature.row_count)
    return _match_ratio(hashes, signature.stable_tiles, signature.maximum_tile_distance) == 1.0


def build_target_visual_descriptor(
    png: bytes,
    point: PixelPoint,
    width_px: int,
    height_px: int,
) -> TargetVisualDescriptor | None:
    """Build a non-reversible target-neighborhood perceptual descriptor."""
    image = _decode_image(png)
    if image is None or image.size != (width_px, height_px):
        return None
    patch_width = min(160, width_px)
    patch_height = min(120, height_px)
    patch = _crop_at(image, point.x, point.y, patch_width, patch_height)
    if patch is None:
        return None
    return TargetVisualDescriptor(
        hash_value=str(imagehash.phash(patch.convert("L"), hash_size=8)),
        patch_width_px=patch_width,
        patch_height_px=patch_height,
    )


def match_target_neighborhood(
    png: bytes,
    descriptor: TargetVisualDescriptor,
    stored_point: GroundedPoint,
    width_px: int,
    height_px: int,
) -> TargetMatch | None:
    """Validate the stored point or find one unique bounded translation."""
    image = _decode_image(png)
    if image is None or image.size != (width_px, height_px):
        return None
    reference_hash = imagehash.hex_to_hash(descriptor.hash_value)
    stored_distance = _patch_distance(image, stored_point.pixel, descriptor, reference_hash)
    if stored_distance is not None and stored_distance <= descriptor.maximum_hash_distance:
        return TargetMatch(stored_point, stored_distance, None, False)
    candidates = _target_candidates(image, stored_point.pixel, descriptor, reference_hash)
    if not candidates:
        return None
    best_distance, best_point = candidates[0]
    separation = max(16, min(descriptor.patch_width_px, descriptor.patch_height_px) // 3)
    runner_up = next(
        (
            distance
            for distance, point in candidates[1:]
            if _point_distance(point, best_point) >= separation
        ),
        None,
    )
    if best_distance > descriptor.maximum_hash_distance:
        return None
    if runner_up is not None and runner_up - best_distance < descriptor.minimum_runner_up_margin:
        return None
    return TargetMatch(
        GroundedPoint(
            normalized=NormalizedPoint(
                x=best_point.x / max(1, width_px - 1),
                y=best_point.y / max(1, height_px - 1),
            ),
            pixel=best_point,
        ),
        best_distance,
        runner_up,
        best_point != stored_point.pixel,
    )


def _target_candidates(
    image: Image.Image,
    stored_point: PixelPoint,
    descriptor: TargetVisualDescriptor,
    reference_hash: imagehash.ImageHash,
) -> list[tuple[int, PixelPoint]]:
    candidates: list[tuple[int, PixelPoint]] = []
    radius = descriptor.search_radius_px
    stride = descriptor.search_stride_px
    for y_offset in range(-radius, radius + 1, stride):
        for x_offset in range(-radius, radius + 1, stride):
            point = PixelPoint(
                x=max(0, min(image.width - 1, stored_point.x + x_offset)),
                y=max(0, min(image.height - 1, stored_point.y + y_offset)),
            )
            distance = _patch_distance(image, point, descriptor, reference_hash)
            if distance is not None:
                candidates.append((distance, point))
    candidates.sort(key=lambda item: (item[0], _point_distance(item[1], stored_point)))
    return candidates


def _match_ratio(
    hashes: tuple[imagehash.ImageHash, ...],
    stable_tiles: tuple[VisualTileFingerprint, ...],
    maximum_distance: int,
) -> float:
    matching_count = sum(
        hashes[tile.tile_index] - imagehash.hex_to_hash(tile.hash_value) <= maximum_distance
        for tile in stable_tiles
    )
    return matching_count / len(stable_tiles)


def _tile_hashes(
    image: Image.Image,
    column_count: int,
    row_count: int,
) -> tuple[imagehash.ImageHash, ...]:
    hashes: list[imagehash.ImageHash] = []
    for row_index in range(row_count):
        top = image.height * row_index // row_count
        bottom = image.height * (row_index + 1) // row_count
        for column_index in range(column_count):
            left = image.width * column_index // column_count
            right = image.width * (column_index + 1) // column_count
            tile = image.crop((left, top, right, bottom)).convert("L")
            hashes.append(imagehash.phash(tile, hash_size=8))
    return tuple(hashes)


def _patch_distance(
    image: Image.Image,
    point: PixelPoint,
    descriptor: TargetVisualDescriptor,
    reference_hash: imagehash.ImageHash,
) -> int | None:
    patch = _crop_at(
        image,
        point.x,
        point.y,
        descriptor.patch_width_px,
        descriptor.patch_height_px,
    )
    if patch is None:
        return None
    return int(imagehash.phash(patch.convert("L"), hash_size=8) - reference_hash)


def _crop_at(
    image: Image.Image,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
) -> Image.Image | None:
    left = center_x - width // 2
    top = center_y - height // 2
    right = left + width
    bottom = top + height
    if left < 0 or top < 0 or right > image.width or bottom > image.height:
        return None
    return image.crop((left, top, right, bottom))


def _decode_image(png: bytes) -> Image.Image | None:
    try:
        with Image.open(BytesIO(png)) as source:
            return source.convert("RGB")
    except (OSError, UnidentifiedImageError):
        return None


def _point_distance(left: PixelPoint, right: PixelPoint) -> float:
    return hypot(left.x - right.x, left.y - right.y)
