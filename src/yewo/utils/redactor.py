"""Masking the values a journey declared sensitive, in text Yewo persists.

A journey names its values in `env` and says which are secret. Yewo masks only
that declared content. Guessing from a value's shape hid ordinary text and
made reports unreadable.

This module owns persisted text. Model-bound placeholder substitution and
screenshot masking are separate transport operations owned by SecretScrubber
and RedactedImages.
"""


def safe_text(text, scrubber=None) -> str:
    """Text fit to persist, with the journey's declared secrets masked."""
    if not isinstance(text, str):
        return text if text is not None else ""
    return scrubber.scrub(text) if scrubber else text


def scrub_elements(elements, scrubber):
    """Masks declared secrets in detected element text, for persisted copies only.

    Model-bound elements are masked when their rendered prompt is assembled.
    This mutation remains restricted to persisted copies.
    """
    for element in elements or []:
        if element.text:
            element.text = scrubber.scrub(element.text)
    return elements
