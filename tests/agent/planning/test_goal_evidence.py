"""Adaptive visual packaging for final goal verification."""

from io import BytesIO
from pathlib import Path

from PIL import Image
import pytest

from yewo.agent.planning.goal_evidence import (
    AdaptiveGoalEvidencePacker,
    GoalEvidenceSource,
)
from yewo.artifacts import ArtifactRetention
from yewo.models import ModelImageConstraints, ModelInputError
from yewo.observations import ScreenshotStore


def _sources(
    tmp_path: Path,
    count: int,
    *,
    size: tuple[int, int] = (540, 1_200),
) -> tuple[GoalEvidenceSource, ...]:
    identifiers = iter(f"shot_{index}" for index in range(count))
    store = ScreenshotStore(tmp_path, identifier_factory=identifiers.__next__)
    sources = []
    for index in range(count):
        output = BytesIO()
        Image.new("RGB", size, color=(index * 17 % 255, 40, 80)).save(output, format="PNG")
        screenshot = store.persist(
            output.getvalue(),
            retention=ArtifactRetention.RUN,
        )
        sources.append(
            GoalEvidenceSource(
                observation_id=f"observation_{index}",
                labels=(f"source for completion condition index {index}",),
                screenshot=screenshot,
            )
        )
    return tuple(sources)


def _constraints(*, max_images: int = 4) -> ModelImageConstraints:
    return ModelImageConstraints(
        max_images=max_images,
        max_image_bytes=4_000_000,
        max_width_px=1_024,
        max_height_px=2_048,
    )


def test_packer_merges_six_portrait_observations_into_one_traceable_panel(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, 6)

    packed = AdaptiveGoalEvidencePacker(tmp_path, _constraints()).pack(sources)

    assert len(packed) == 1
    assert packed[0].observation_ids == tuple(source.observation_id for source in sources)
    assert packed[0].screenshot.artifact_id not in {
        source.screenshot.artifact_id for source in sources
    }
    assert packed[0].screenshot.width_px <= 1_024
    assert packed[0].screenshot.height_px <= 2_048
    assert ScreenshotStore(tmp_path).load(packed[0].screenshot)


def test_packer_uses_geometry_to_fit_ten_small_observations_in_one_panel(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, 10, size=(200, 300))

    packed = AdaptiveGoalEvidencePacker(tmp_path, _constraints()).pack(sources)

    assert len(packed) == 1
    assert len(packed[0].observation_ids) == 10


def test_packer_keeps_three_portrait_observations_direct_when_panels_cost_more(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, 3)

    packed = AdaptiveGoalEvidencePacker(tmp_path, _constraints()).pack(sources)

    assert tuple(item.screenshot for item in packed) == tuple(
        source.screenshot for source in sources
    )


def test_packer_fails_closed_when_readable_panels_exceed_profile_capacity(
    tmp_path: Path,
) -> None:
    sources = _sources(tmp_path, 11)

    with pytest.raises(ModelInputError, match="cannot fit"):
        AdaptiveGoalEvidencePacker(tmp_path, _constraints(max_images=1)).pack(sources)


def test_capacity_probe_does_not_persist_a_derived_panel(tmp_path: Path) -> None:
    sources = _sources(tmp_path, 6)
    screenshots = tmp_path / "screenshots"
    before = set(screenshots.iterdir())
    packer = AdaptiveGoalEvidencePacker(tmp_path, _constraints(max_images=1))

    fits = packer.fits(sources)

    assert fits
    assert set(screenshots.iterdir()) == before


def test_capacity_probe_rejects_an_unreadable_source_set(tmp_path: Path) -> None:
    sources = _sources(tmp_path, 11)

    assert not AdaptiveGoalEvidencePacker(
        tmp_path,
        _constraints(max_images=1),
    ).fits(sources)
