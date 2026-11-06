"""Identify Yewo through provider-supported model request metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import json

from yewo.models.profiles import ModelProfile


YEWO_APP_NAME = "Yewo"
YEWO_APP_SLUG = "yewo"
YEWO_APP_URL = "https://tryyewo.dev"
try:
    _YEWO_VERSION = version("yewo")
except PackageNotFoundError:
    _YEWO_VERSION = "0+unknown"
YEWO_USER_AGENT = f"{YEWO_APP_NAME}/{_YEWO_VERSION} (+{YEWO_APP_URL})"


def build_model_attribution(profile: ModelProfile) -> dict[str, object]:
    """Build transport and provider-native application attribution."""
    extra_headers = {
        "User-Agent": YEWO_USER_AGENT,
    }
    arguments: dict[str, object] = {"extra_headers": extra_headers}

    if profile.provider == "openrouter":
        extra_headers.update(
            {
                "HTTP-Referer": YEWO_APP_URL,
                "X-Title": YEWO_APP_NAME,
            }
        )
    elif profile.provider == "vertex_ai":
        arguments["labels"] = {"application": YEWO_APP_SLUG}
    elif profile.provider == "bedrock":
        metadata = {"application": YEWO_APP_SLUG}
        if "/converse/" in profile.model:
            arguments["requestMetadata"] = metadata
        else:
            extra_headers["X-Amzn-Bedrock-Request-Metadata"] = json.dumps(
                metadata,
                separators=(",", ":"),
            )

    return arguments
