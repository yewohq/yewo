"""Encode provider messages and durable request evidence."""

from __future__ import annotations

import base64
from collections import OrderedDict
from hashlib import sha256
from io import BytesIO

from PIL import Image

from yewo.devices.screen import png_size
from yewo.models.artifacts import ModelImageReader
from yewo.models.errors import ModelInputError
from yewo.models.profiles import ModelProfile
from yewo.models.records import ImageModelContent, ModelMessage, TextModelContent


class ModelImagePayloadCache:
    """Hold the downscaled form of each screenshot for the roles that share one step.

    One step sends the same screenshot to the analyzer, the planner and both verifiers. Each
    encoder lives for one request, so without a shared cache every role decodes, resizes and
    re-encodes the same frame again. The key is the verified content digest and the long-edge
    bound, because those are what determine the bytes.
    """

    def __init__(self, maximum_entries: int = 8) -> None:
        self._entries: OrderedDict[tuple[str, int | None], bytes] = OrderedDict()
        self._maximum_entries = maximum_entries

    def downscaled(self, digest: str, png: bytes, max_long_edge_px: int | None) -> bytes:
        key = (digest, max_long_edge_px)
        cached = self._entries.get(key)
        if cached is not None:
            self._entries.move_to_end(key)
            return cached
        downscaled = _downscaled_png(png, max_long_edge_px)
        self._entries[key] = downscaled
        while len(self._entries) > self._maximum_entries:
            self._entries.popitem(last=False)
        return downscaled


class ModelMessageEncoder:
    """Encode one logical request while reading each image at most once."""

    def __init__(
        self,
        images: ModelImageReader,
        profile: ModelProfile,
        payload_cache: ModelImagePayloadCache | None = None,
    ) -> None:
        self._images = images
        self._profile = profile
        self._payload_cache = payload_cache or ModelImagePayloadCache()
        self._image_payloads: dict[str, dict[str, object]] = {}
        self._images_sent: dict[str, dict[str, object]] = {}

    def provider_messages(self, messages: tuple[ModelMessage, ...]) -> list[dict[str, object]]:
        """Return LiteLLM-compatible messages."""
        return [self._provider_message(message) for message in messages]

    def artifact_messages(self, messages: tuple[ModelMessage, ...]) -> list[dict[str, object]]:
        """Return the same request with durable image references instead of bytes.

        Call this after provider_messages, so a reference records the image the provider was
        given rather than the stored artifact it was derived from.
        """
        prepared: list[dict[str, object]] = []
        for message in messages:
            parts = [self._artifact_content(part) for part in message.content]
            content: object = (
                parts[0]["text"] if len(parts) == 1 and parts[0].get("type") == "text" else parts
            )
            prepared_message: dict[str, object] = {
                "role": message.role.value,
                "content": content,
            }
            if message.cache_prefix_end:
                prepared_message["cache_prefix_end"] = True
            prepared.append(prepared_message)
        return prepared

    def _provider_message(self, message: ModelMessage) -> dict[str, object]:
        parts = [self._provider_content(content) for content in message.content]
        if len(parts) == 1 and parts[0].get("type") == "text":
            return {"role": message.role.value, "content": parts[0]["text"]}
        return {"role": message.role.value, "content": parts}

    def _provider_content(self, content: TextModelContent | ImageModelContent) -> dict[str, object]:
        if isinstance(content, TextModelContent):
            return {"type": "text", "text": content.text}
        screenshot = content.screenshot
        cached = self._image_payloads.get(screenshot.artifact_id)
        if cached is not None:
            return cached
        if (
            screenshot.width_px > self._profile.images.max_width_px
            or screenshot.height_px > self._profile.images.max_height_px
        ):
            raise ModelInputError("The model image exceeds the profile dimension limit.")
        png = self._payload_cache.downscaled(
            screenshot.sha256,
            self._images.read(screenshot),
            self._profile.images.max_long_edge_px,
        )
        if len(png) > self._profile.images.max_image_bytes:
            raise ModelInputError("The model image exceeds the profile byte limit.")
        payload: dict[str, object] = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"},
        }
        self._image_payloads[screenshot.artifact_id] = payload
        digest = sha256(png).hexdigest()
        dimensions = png_size(png)
        if digest != screenshot.sha256 and dimensions is not None:
            self._images_sent[screenshot.artifact_id] = {
                "sha256": digest,
                "width_px": dimensions[0],
                "height_px": dimensions[1],
                "transform": "downscaled-long-edge",
            }
        return payload

    def _artifact_content(
        self,
        content: TextModelContent | ImageModelContent,
    ) -> dict[str, object]:
        if isinstance(content, TextModelContent):
            return {"type": "text", "text": content.text}
        screenshot = content.screenshot
        reference: dict[str, object] = {
            "type": "image_ref",
            "artifact_id": screenshot.artifact_id,
            "sha256": screenshot.sha256,
            "width_px": screenshot.width_px,
            "height_px": screenshot.height_px,
        }
        sent = self._images_sent.get(screenshot.artifact_id)
        if sent is not None:
            reference["sent"] = sent
        return reference


def _downscaled_png(png: bytes, max_long_edge_px: int | None) -> bytes:
    """Return the image no larger than the profile's long edge, preserving aspect ratio."""
    if max_long_edge_px is None:
        return png
    with Image.open(BytesIO(png)) as image:
        if max(image.size) <= max_long_edge_px:
            return png
        scale = max_long_edge_px / max(image.size)
        resized = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
        stream = BytesIO()
        resized.save(stream, format="PNG")
    return stream.getvalue()
