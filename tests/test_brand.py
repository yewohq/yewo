"""The packaged Yewo mark, and how it rasterises."""

from __future__ import annotations

import numpy as np
from PIL import ImageDraw

from yewo.brand import mark_image


def test_the_mark_is_a_transparent_square() -> None:
    mark = mark_image(24)

    assert mark.size == (24, 24)
    assert mark.mode == "RGBA"


def test_the_mark_is_antialiased() -> None:
    """A stroke has no antialiasing, so the mark is drawn large and reduced.

    Rasterised straight at a field's height it came out with stepped edges. Two alpha levels
    is what that looks like: opaque or absent, nothing between.
    """
    alpha = np.asarray(mark_image(24))[..., 3]

    assert len(np.unique(alpha)) > 2


def test_the_mark_scales() -> None:
    assert mark_image(48).size == (48, 48)


def test_each_caller_gets_its_own_image() -> None:
    """One shared image is one caller's stray draw away from corrupting every later paste."""
    first, second = mark_image(24), mark_image(24)

    assert first is not second
    ImageDraw.Draw(first).rectangle((0, 0, 23, 23), fill=(255, 0, 0, 255))

    assert np.array_equal(np.asarray(mark_image(24)), np.asarray(second))
