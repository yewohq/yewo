"""Provider-aware Yewo model request attribution."""

from importlib.metadata import version

from yewo.models.attribution import build_model_attribution

from models.support import build_model_profile


def test_direct_provider_receives_yewo_user_agent() -> None:
    attribution = build_model_attribution(
        build_model_profile(provider="anthropic", model="anthropic/example")
    )

    extra_headers = attribution["extra_headers"]
    assert isinstance(extra_headers, dict)
    assert extra_headers["User-Agent"] == f"Yewo/{version('yewo')} (+https://tryyewo.dev)"
    assert attribution.keys() == {"extra_headers"}


def test_vertex_ai_receives_application_billing_label() -> None:
    attribution = build_model_attribution(
        build_model_profile(provider="vertex_ai", model="vertex_ai/example")
    )

    assert attribution["labels"] == {"application": "yewo"}


def test_bedrock_converse_receives_application_request_metadata() -> None:
    attribution = build_model_attribution(
        build_model_profile(
            provider="bedrock",
            model="bedrock/converse/example",
        )
    )

    assert attribution["requestMetadata"] == {"application": "yewo"}


def test_bedrock_invoke_receives_signed_application_metadata_header() -> None:
    attribution = build_model_attribution(
        build_model_profile(provider="bedrock", model="bedrock/invoke/example")
    )

    extra_headers = attribution["extra_headers"]
    assert isinstance(extra_headers, dict)
    assert extra_headers["X-Amzn-Bedrock-Request-Metadata"] == '{"application":"yewo"}'
