"""One owner for exact, replay, and logical-screen comparisons."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from yewo.observations.fingerprint import fingerprint_distance
from yewo.observations.records import PresentationKey


class ScreenComparisonPolicy(BaseModel):
    """Versioned fail-closed thresholds for screen presentation matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    algorithm: Literal["effective-mask-phash-distance-v1"] = "effective-mask-phash-distance-v1"
    maximum_phash_distance: int = Field(default=0, ge=0, le=64)
    maximum_replay_phash_distance: int = Field(default=2, ge=0, le=64)
    require_same_effective_mask: Literal[True] = True


class ScreenMatcher:
    """Compare presentation keys under one explicit policy."""

    def __init__(self, policy: ScreenComparisonPolicy) -> None:
        self.policy = policy

    @staticmethod
    def analysis_cache_key(presentation: PresentationKey) -> tuple[object, ...]:
        """Return the bounded structural-variant cache identity."""
        return (
            presentation.fingerprint.value,
            presentation.foreground_application_id,
            presentation.platform_context,
            presentation.keyboard_visible,
            presentation.orientation,
            presentation.width_px,
            presentation.height_px,
        )

    @staticmethod
    def metadata_matches(
        first: PresentationKey,
        second: PresentationKey,
        *,
        compare_keyboard: bool = True,
    ) -> bool:
        """Compare non-pixel presentation evidence."""
        return (
            first.foreground_application_id == second.foreground_application_id
            and first.platform_context == second.platform_context
            and (not compare_keyboard or first.keyboard_visible is second.keyboard_visible)
            and first.orientation is second.orientation
            and (first.width_px, first.height_px) == (second.width_px, second.height_px)
        )

    @staticmethod
    def distance(first: PresentationKey, second: PresentationKey) -> int | None:
        """Return perceptual distance or reject incompatible fingerprint evidence."""
        try:
            return fingerprint_distance(first.fingerprint, second.fingerprint)
        except ValueError:
            return None

    @staticmethod
    def exact(first: PresentationKey, second: PresentationKey) -> bool:
        """Require identical metadata and application-owned pixels."""
        return ScreenMatcher.metadata_matches(first, second) and (
            first.fingerprint.algorithm == second.fingerprint.algorithm
            and first.fingerprint.masked_pixels_sha256 == second.fingerprint.masked_pixels_sha256
            and first.fingerprint.system_region_mask_sha256
            == second.fingerprint.system_region_mask_sha256
        )

    def replay_equivalent(self, first: PresentationKey, second: PresentationKey) -> bool:
        """Allow the configured replay fingerprint tolerance."""
        return self._within(
            first,
            second,
            self.policy.maximum_replay_phash_distance,
            compare_keyboard=True,
        )

    def logical_screen(self, first: PresentationKey, second: PresentationKey) -> bool:
        """Compare logical screens while allowing keyboard presentation changes."""
        return self._within(
            first,
            second,
            self.policy.maximum_phash_distance,
            compare_keyboard=False,
        )

    def _within(
        self,
        first: PresentationKey,
        second: PresentationKey,
        maximum_distance: int,
        *,
        compare_keyboard: bool,
    ) -> bool:
        if not self.metadata_matches(first, second, compare_keyboard=compare_keyboard):
            return False
        distance = self.distance(first, second)
        return distance is not None and distance <= maximum_distance
