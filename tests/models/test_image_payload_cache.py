"""One screenshot reaches several roles in one step and is prepared once."""

from io import BytesIO

from PIL import Image

from yewo.models.message_encoding import ModelImagePayloadCache


def _png(width: int, height: int) -> bytes:
    stream = BytesIO()
    Image.new("RGB", (width, height), (40, 60, 40)).save(stream, format="PNG")
    return stream.getvalue()


def test_one_screenshot_is_downscaled_once_for_one_long_edge_bound() -> None:
    # Each encoder lives for one request, so the analyzer, the planner and both verifiers each
    # decoded, resized and re-encoded the same frame again.
    cache = ModelImagePayloadCache()
    png = _png(2000, 1000)

    first = cache.downscaled("digest-1", png, 1568)
    second = cache.downscaled("digest-1", png, 1568)

    assert first is second
    assert len(first) < len(png)


def test_a_different_long_edge_bound_is_a_different_entry() -> None:
    cache = ModelImagePayloadCache()
    png = _png(2000, 1000)

    assert cache.downscaled("digest-1", png, 1568) is not cache.downscaled("digest-1", png, 800)


def test_a_different_screenshot_is_never_served_from_another_digest() -> None:
    # The key is the verified content digest. Serving one screenshot's bytes for another would
    # send a model the wrong screen.
    cache = ModelImagePayloadCache()

    wide = cache.downscaled("digest-1", _png(2000, 1000), None)
    tall = cache.downscaled("digest-2", _png(1000, 2000), None)

    assert wide != tall


def test_the_cache_is_bounded() -> None:
    # A run visits many screens. An unbounded cache would hold every frame of the run.
    cache = ModelImagePayloadCache(maximum_entries=2)
    png = _png(2000, 1000)

    first = cache.downscaled("digest-1", png, 1568)
    cache.downscaled("digest-2", png, 1568)
    cache.downscaled("digest-3", png, 1568)

    assert cache.downscaled("digest-1", png, 1568) is not first
