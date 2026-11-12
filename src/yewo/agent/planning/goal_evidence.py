"""Bounded visual packaging for independent goal verification."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageDraw

from yewo.models import ModelImageConstraints, ModelInputError
from yewo.observations import (
    ScreenshotReference,
    ScreenshotStore,
)


_MAXIMUM_PANEL_MEMBERS = 10
_MINIMUM_RENDER_SCALE = 0.45
_HEADER_HEIGHT_PX = 64


@dataclass(frozen=True)
class GoalEvidenceSource:
    """One unique observation and every milestone role that references it."""

    observation_id: str
    labels: tuple[str, ...]
    screenshot: ScreenshotReference
    related_observation_ids: tuple[str, ...] = ()

    @property
    def observation_ids(self) -> tuple[str, ...]:
        """Return every observation represented by this screenshot."""
        return (self.observation_id, *self.related_observation_ids)


@dataclass(frozen=True)
class GoalEvidenceImage:
    """One direct screenshot or panel supplied to the final auditor."""

    screenshot: ScreenshotReference
    observation_ids: tuple[str, ...]
    labels: tuple[str, ...]


class GoalEvidencePacker(Protocol):
    """Package unique observations for one bounded auditor request."""

    def pack(self, sources: tuple[GoalEvidenceSource, ...]) -> tuple[GoalEvidenceImage, ...]: ...


def direct_goal_evidence_images(
    sources: tuple[GoalEvidenceSource, ...],
) -> tuple[GoalEvidenceImage, ...]:
    """Return one auditor image per unique observation."""
    return tuple(
        GoalEvidenceImage(
            screenshot=source.screenshot,
            observation_ids=source.observation_ids,
            labels=source.labels,
        )
        for source in sources
    )


@dataclass(frozen=True)
class _PanelLayout:
    columns: int
    rows: int
    cell_width_px: int
    cell_height_px: int
    minimum_scale: float


class AdaptiveGoalEvidencePacker:
    """Persist readable panels when direct screenshots exceed the visual budget."""

    def __init__(
        self,
        root: Path,
        constraints: ModelImageConstraints,
    ) -> None:
        self._store = ScreenshotStore(root)
        self._constraints = constraints

    def pack(self, sources: tuple[GoalEvidenceSource, ...]) -> tuple[GoalEvidenceImage, ...]:
        """Prefer direct evidence until panels reduce pixels or image count."""
        if not sources:
            return ()
        batches = _panel_batches(sources, self._constraints)
        if len(batches) > self._constraints.max_images:
            raise ModelInputError(
                "The goal evidence cannot fit the auditor image budget at a readable scale."
            )
        original_pixels = sum(
            source.screenshot.width_px * source.screenshot.height_px for source in sources
        )
        panel_pixels = len(batches) * (
            self._constraints.max_width_px * self._constraints.max_height_px
        )
        if (
            len(sources) <= self._constraints.max_images
            and original_pixels <= panel_pixels
            and all(_fits_directly(source, self._constraints) for source in sources)
        ):
            return direct_goal_evidence_images(sources)
        packed: list[GoalEvidenceImage] = []
        for batch in batches:
            packed.extend(self._pack_batch(batch))
        if len(packed) > self._constraints.max_images:
            raise ModelInputError(
                "The goal evidence cannot fit the auditor image budget at a readable scale."
            )
        return tuple(packed)

    def fits(self, sources: tuple[GoalEvidenceSource, ...]) -> bool:
        """Return whether sources fit without persisting derived panels."""
        try:
            batches = _panel_batches(sources, self._constraints)
            image_count = sum(self._packed_batch_count(batch) for batch in batches)
        except ModelInputError:
            return False
        return image_count <= self._constraints.max_images

    def _packed_batch_count(self, sources: tuple[GoalEvidenceSource, ...]) -> int:
        if len(sources) == 1 and _fits_directly(sources[0], self._constraints):
            return 1
        png, unused_references = _render_panel(
            sources,
            self._store,
            self._constraints,
        )
        if len(png) <= self._constraints.max_image_bytes:
            return 1
        if len(sources) == 1:
            raise ModelInputError("A goal evidence image exceeds the auditor byte budget.")
        midpoint = len(sources) // 2
        return self._packed_batch_count(sources[:midpoint]) + self._packed_batch_count(
            sources[midpoint:]
        )

    def _pack_batch(
        self,
        sources: tuple[GoalEvidenceSource, ...],
    ) -> tuple[GoalEvidenceImage, ...]:
        if len(sources) == 1 and _fits_directly(sources[0], self._constraints):
            return direct_goal_evidence_images(sources)
        png, source_references = _render_panel(sources, self._store, self._constraints)
        if len(png) > self._constraints.max_image_bytes:
            if len(sources) == 1:
                raise ModelInputError("A goal evidence image exceeds the auditor byte budget.")
            midpoint = len(sources) // 2
            return (
                *self._pack_batch(sources[:midpoint]),
                *self._pack_batch(sources[midpoint:]),
            )
        screenshot = self._store.persist(
            png,
            retention=source_references[0].retention,
        )
        return (
            GoalEvidenceImage(
                screenshot=screenshot,
                observation_ids=tuple(
                    observation_id
                    for source in sources
                    for observation_id in source.observation_ids
                ),
                labels=tuple(label for source in sources for label in source.labels),
            ),
        )


def _fits_directly(source: GoalEvidenceSource, constraints: ModelImageConstraints) -> bool:
    screenshot = source.screenshot
    return (
        screenshot.width_px <= constraints.max_width_px
        and screenshot.height_px <= constraints.max_height_px
    )


def _panel_batches(
    sources: tuple[GoalEvidenceSource, ...],
    constraints: ModelImageConstraints,
) -> tuple[tuple[GoalEvidenceSource, ...], ...]:
    batches: list[tuple[GoalEvidenceSource, ...]] = []
    current: tuple[GoalEvidenceSource, ...] = ()
    for source in sources:
        candidate = (*current, source)
        if len(candidate) <= _MAXIMUM_PANEL_MEMBERS and (
            _best_layout(candidate, constraints).minimum_scale >= _MINIMUM_RENDER_SCALE
        ):
            current = candidate
            continue
        if not current:
            raise ModelInputError("A goal evidence image cannot remain readable in a panel.")
        batches.append(current)
        current = (source,)
    if current:
        batches.append(current)
    return tuple(batches)


def _best_layout(
    sources: Sequence[GoalEvidenceSource],
    constraints: ModelImageConstraints,
) -> _PanelLayout:
    candidates: list[_PanelLayout] = []
    for columns in range(1, min(4, len(sources)) + 1):
        rows = ceil(len(sources) / columns)
        cell_width = constraints.max_width_px // columns
        cell_height = constraints.max_height_px // rows
        image_height = max(1, cell_height - _HEADER_HEIGHT_PX)
        minimum_scale = min(
            min(
                cell_width / source.screenshot.width_px,
                image_height / source.screenshot.height_px,
                1.0,
            )
            for source in sources
        )
        candidates.append(
            _PanelLayout(
                columns=columns,
                rows=rows,
                cell_width_px=cell_width,
                cell_height_px=cell_height,
                minimum_scale=minimum_scale,
            )
        )
    return max(candidates, key=lambda item: (item.minimum_scale, -item.rows, -item.columns))


def _render_panel(
    sources: tuple[GoalEvidenceSource, ...],
    store: ScreenshotStore,
    constraints: ModelImageConstraints,
) -> tuple[bytes, tuple[ScreenshotReference, ...]]:
    layout = _best_layout(sources, constraints)
    width = layout.columns * layout.cell_width_px
    height = layout.rows * layout.cell_height_px
    panel = Image.new("RGB", (width, height), "white")
    references: list[ScreenshotReference] = []
    for index, source in enumerate(sources):
        reference = source.screenshot
        references.append(reference)
        with Image.open(BytesIO(store.load(reference))) as opened:
            image = opened.convert("RGB")
        image.thumbnail((layout.cell_width_px, layout.cell_height_px - _HEADER_HEIGHT_PX))
        tile = Image.new(
            "RGB",
            (layout.cell_width_px, layout.cell_height_px),
            "white",
        )
        offset_x = (layout.cell_width_px - image.width) // 2
        offset_y = (
            _HEADER_HEIGHT_PX + (layout.cell_height_px - _HEADER_HEIGHT_PX - image.height) // 2
        )
        tile.paste(image, (offset_x, offset_y))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 6), f"Observation {source.observation_id[:20]}", fill="black")
        draw.text((8, 30), "; ".join(source.labels)[:120], fill="black")
        panel.paste(
            tile,
            (
                (index % layout.columns) * layout.cell_width_px,
                (index // layout.columns) * layout.cell_height_px,
            ),
        )
    output = BytesIO()
    panel.save(output, format="PNG", optimize=True)
    return output.getvalue(), tuple(references)
