"""Encode provider messages and durable request evidence."""

from __future__ import annotations

import base64

from yewo.models.artifacts import ModelImageReader
from yewo.models.errors import ModelInputError
from yewo.models.profiles import ModelProfile
from yewo.models.records import ImageModelContent, ModelMessage, TextModelContent


class ModelMessageEncoder:
    """Encode one logical request while reading each image at most once."""

    def __init__(self, images: ModelImageReader, profile: ModelProfile) -> None:
        self._images = images
        self._profile = profile
        self._image_payloads: dict[str, dict[str, object]] = {}

    def provider_messages(self, messages: tuple[ModelMessage, ...]) -> list[dict[str, object]]:
        """Return LiteLLM-compatible messages."""
        return [self._provider_message(message) for message in messages]

    @staticmethod
    def artifact_messages(messages: tuple[ModelMessage, ...]) -> list[dict[str, object]]:
        """Return the same request with durable image references instead of bytes."""
        prepared: list[dict[str, object]] = []
        for message in messages:
            parts = [ModelMessageEncoder._artifact_content(part) for part in message.content]
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
        png = self._images.read(screenshot)
        if len(png) > self._profile.images.max_image_bytes:
            raise ModelInputError("The model image exceeds the profile byte limit.")
        payload: dict[str, object] = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64.b64encode(png).decode('ascii')}"},
        }
        self._image_payloads[screenshot.artifact_id] = payload
        return payload

    @staticmethod
    def _artifact_content(
        content: TextModelContent | ImageModelContent,
    ) -> dict[str, object]:
        if isinstance(content, TextModelContent):
            return {"type": "text", "text": content.text}
        screenshot = content.screenshot
        return {
            "type": "image_ref",
            "artifact_id": screenshot.artifact_id,
            "sha256": screenshot.sha256,
            "width_px": screenshot.width_px,
            "height_px": screenshot.height_px,
        }
