"""Shared comparison for perceptual screen fingerprints."""

from collections.abc import Iterable
from enum import Enum
from typing import Optional

SCREEN_PHASH_DISTANCE_THRESHOLD = 10
"""Maximum Hamming distance for two fingerprints of one visual screen."""


class ScreenComparison(str, Enum):
    """Typed result of comparing two visual fingerprints."""

    SAME = "same"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


def phash_distance(first: Optional[str], second: Optional[str]) -> Optional[int]:
    """Return the Hamming distance between equal-length hexadecimal hashes."""
    if not first or not second or len(first) != len(second):
        return None
    try:
        return (int(first, 16) ^ int(second, 16)).bit_count()
    except ValueError:
        return 0 if first == second else None


def is_same_screen(
    first: Optional[str],
    second: Optional[str],
    threshold: int = SCREEN_PHASH_DISTANCE_THRESHOLD,
) -> bool:
    """Whether two readings represent the same visual screen."""
    return compare_screens(first, second, threshold) is ScreenComparison.SAME


def compare_screens(
    first: Optional[str],
    second: Optional[str],
    threshold: int = SCREEN_PHASH_DISTANCE_THRESHOLD,
) -> ScreenComparison:
    """Classify visual identity without turning missing evidence into equality."""
    if not first or not second:
        return ScreenComparison.UNKNOWN
    if first == second:
        return ScreenComparison.SAME
    distance = phash_distance(first, second)
    if distance is None:
        return ScreenComparison.UNKNOWN
    if distance <= threshold:
        return ScreenComparison.SAME
    return ScreenComparison.DIFFERENT


def find_matching_screen(
    phash: str,
    known_hashes: Iterable[str],
    threshold: int = SCREEN_PHASH_DISTANCE_THRESHOLD,
) -> Optional[str]:
    """Return the first equivalent canonical fingerprint, if one exists."""
    return next(
        (known for known in known_hashes if is_same_screen(phash, known, threshold)),
        None,
    )
