"""Keeping what each model call was asked.

`llm_calls.jsonl` kept every answer and no question. So every claim about
what a prompt said was a reconstruction from the code that builds it. A
prompt that changed between two runs could not be read back at all.

Three rules decide the shape of this.

- **Content-addressed.** A prompt is named by its own hash. Two calls that
  asked the same thing keep one file, and no index has to be written.
- **Screenshots referenced, never embedded.** An image reaches a prompt as a
  base64 data URL. That is about a megabyte of text per step. The stored
  prompt keeps the image's hash instead. The screen itself is already on
  disk, beside the step that read it.
- **Scrubbed, like every other artifact.** A prompt carries the values the
  planner types, so it carries whatever the journey declared.

The hash is written whether or not the body is. A run that must not persist
model inputs can still say that two runs asked the same thing.
"""

import base64
import binascii
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

IMAGE_PART = "image_url"
DATA_URL = "data:"
NAME_LENGTH = 32
"""Characters of the hash a file is named by. Enough that two never collide."""


def _plain(value: Optional[str]) -> str:
    return value or ""


def _digest(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def schema_sha256(response_model) -> Optional[str]:
    """The shape an answer was required to take.

    Kept beside the prompt because it decides the answer as much as the words
    do. A field renamed in a schema changes what a model replies with, and
    nothing else in a run would record that.
    """
    if response_model is None:
        return None
    return _digest(response_model.model_json_schema())


class PromptStore:
    """Writes each distinct prompt once, and names it by its content."""

    def __init__(
        self,
        directory: Path,
        scrub: Callable[[Optional[str]], str] = _plain,
        keep_bodies: bool = True,
    ):
        """Builds the store.

        Args:
            directory: Where prompt bodies are written.
            scrub: Masks declared secrets. Every string written goes through it.
            keep_bodies: False writes no body and still returns the hash,
                which is what a run that must not persist model inputs asks
                for.
        """
        self.directory = directory
        self._scrub = scrub
        self.keep_bodies = keep_bodies

    def store_messages(self, messages: list) -> tuple[Optional[str], Optional[str]]:
        """The prompt's hash, and where its body was written.

        Returns:
            The hash, and the path relative to the run's own directory, or
            None where no body was kept. Both are None if the prompt could
            not be read, because a prompt is a record and not a guess.
        """
        try:
            readable = [self._serialize_message(message) for message in messages]
        except Exception as exc:  # noqa: BLE001 - a record must not fail a run
            logger.warning("Could not read the prompt to record it: %s", exc)
            return None, None
        return self.store_payload({"messages": readable})

    def store_payload(self, payload: dict) -> tuple[Optional[str], Optional[str]]:
        """The same, for a route whose question is not a message list.

        The decision model is asked typed questions about one state rather
        than a conversation. It is still a prompt, and a run that recorded
        one route and not the other would answer half the question.

        Scrubbed through the whole structure. A question names the control it
        is about, in the app's own words, at whatever depth the caller nested
        it.
        """
        try:
            masked = self._mask_payload(payload)
            digest = _digest(masked)
        except Exception as exc:  # noqa: BLE001 - a record must not fail a run
            logger.warning("Could not read the prompt to record it: %s", exc)
            return None, None
        if not self.keep_bodies:
            return digest, None
        return digest, self._write_payload(digest, masked)

    def _mask_payload(self, payload: dict) -> dict:
        """One payload, with every declared value masked at every depth."""
        return {key: self._scrub_value(value) for key, value in payload.items()}

    def _scrub_value(self, value):
        """Masks declared values wherever they sit in a nested value."""
        if isinstance(value, str):
            return self._scrub(value)
        if isinstance(value, dict):
            return {key: self._scrub_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._scrub_value(item) for item in value]
        return value

    def _write_payload(self, digest: str, payload: dict) -> Optional[str]:
        """Writes the body once. A prompt already kept is not written again."""
        name = f"{digest[:NAME_LENGTH]}.json"
        path = self.directory / name
        if path.exists():
            return f"prompts/{name}"
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps({"sha256": digest, **payload}, indent=2, default=str),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            logger.warning("Could not write the prompt to %s: %s", path, exc)
            return None
        return f"prompts/{name}"

    def _serialize_message(self, message) -> dict:
        """One message, with its text scrubbed and its images referenced."""
        sent = message.to_dict()
        content = sent.get("content")
        if isinstance(content, str):
            return {"role": sent.get("role"), "content": self._scrub(content)}
        return {
            "role": sent.get("role"),
            "content": [self._serialize_part(part) for part in content or []],
        }

    def _serialize_part(self, part: dict) -> dict:
        """One part of a multipart message, with any image left as a hash."""
        if part.get("type") != IMAGE_PART:
            return {**part, "text": self._scrub(part.get("text"))}
        url = (part.get("image_url") or {}).get("url") or ""
        return {"type": IMAGE_PART, "image_sha256": _image_digest(url)}


def _image_digest(url: str) -> Optional[str]:
    """The image's own hash, read from the data URL that carried it.

    Over the decoded bytes, so the hash names the picture rather than the way
    it travelled. The decision route hashes the screenshot it was handed, and
    one image has to have one identity whichever route carried it. Base64 of
    the same bytes can differ by padding and by line breaks.

    A URL that is not inline data is left unhashed rather than fetched. The
    record says what was sent, and a fetch would say what is there now.
    """
    if not url.startswith(DATA_URL) or "," not in url:
        return None
    encoded = url.split(",", 1)[1]
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        logger.debug("Could not decode an image to hash it: %s", exc)
        return None
    return hashlib.sha256(raw).hexdigest()
