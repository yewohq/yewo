"""
LiteLLM wrapper client for communicating with LLMs (e.g. Gemini via OpenRouter).
Supports standard messaging and structured output via Pydantic schemas.
"""

import hashlib
import json
import logging
import time
from typing import Callable, Optional, TypeVar, overload

import litellm
from pydantic import BaseModel, ValidationError

from yewo.errors import ProviderUnavailable

from . import failures, prompt_cache, prompts, usage
from .prompts import PromptStore
from .messages import Message, TextMessage

logger = logging.getLogger(__name__)

STRUCTURED_ATTEMPTS = 2

Answer = TypeVar("Answer", bound=BaseModel)

JSON_INSTRUCTION = (
    "Return only one JSON object matching the requested fields and action instructions. "
    "Do not wrap it in prose."
)


class LLMClient:
    """Client for making provider-neutral calls through the LiteLLM SDK."""

    def __init__(
        self,
        model: str,
        usage_callback=None,
        wait: Callable[[float], None] = time.sleep,
        prompt_store: Optional[PromptStore] = None,
        cache_affinity: Optional[str] = None,
    ):
        """Initializes the LLM client.

        Args:
            model: The API model identifier to use for requests.
            usage_callback: Optional callable receiving a usage.CallRecord per call.
            wait: How to pause between provider attempts. Injected so a test
                exercises the backoff without spending the time.
            prompt_store: Where to record what each call asked. None records
                nothing, which is what a client built outside a run gets.
            cache_affinity: Stable identity for calls in one run. OpenRouter
                uses it for sticky routing to the endpoint holding a prompt
                cache. It does not ask for caching; a provider that caches
                only on request is marked in the message payload instead.
        """
        self._model = model
        self.usage_callback = usage_callback
        self._wait = wait
        self.prompt_store = prompt_store
        self.set_cache_affinity(cache_affinity)

    def set_cache_affinity(self, identity: Optional[str]) -> None:
        """Route this client's related calls to one prompt-cache location.

        A digest keeps filesystem paths, device names, and other caller-owned
        identities out of provider metadata. It is also safely below the
        provider's 256-character session-id limit.
        """
        self._cache_affinity = (
            f"yewo-{hashlib.sha256(identity.encode('utf-8')).hexdigest()}" if identity else None
        )

    @overload
    def generate_response(self, messages: list[Message]) -> str: ...

    @property
    def model(self) -> str:
        """The model this client asks, for a record of what produced a run."""
        return self._model

    @overload
    def generate_response(
        self, messages: list[Message], response_model: None, context: dict | None = ...
    ) -> str: ...

    @overload
    def generate_response(
        self,
        messages: list[Message],
        response_model: type[Answer],
        context: dict | None = ...,
    ) -> Answer: ...

    def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        context: dict | None = None,
    ) -> str | BaseModel:
        """Sends a message sequence to the LLM and returns the completion.

        Overloaded so a caller asking for a model is given that model, rather
        than a union it has to assert its way out of.

        Args:
            messages: A list of Message objects (text, image, multipart) to send.
            response_model: Optional Pydantic BaseModel class for structured output.
            context: What the answer has to be read against, for a rule that
                depends on more than the answer itself. A rule that reaches the
                validator is refused on the same path as any other mismatch.
                So the model is asked again, with the reason.

        Returns:
            The raw text response from the LLM, or a validated instance of response_model
            if specified.
        """
        attempts: list[Message] = list(messages)
        last_error: ValidationError | None = None

        # Schema enforcement is advisory on most providers, so a model may
        # answer in a shape the schema forbids. One correction is far cheaper
        # than losing the scenario the call belongs to.
        for attempt in range(STRUCTURED_ATTEMPTS if response_model else 1):
            try:
                return self._asked(attempts, response_model, attempt + 1, context)
            except ValidationError as exc:
                last_error = exc
                logger.warning(
                    "Model response did not match %s. Attempt %d of %d, with %d validation errors.",
                    response_model.__name__ if response_model else "schema",
                    attempt + 1,
                    STRUCTURED_ATTEMPTS,
                    len(exc.errors()),
                )
                safe_errors = exc.errors(include_url=False, include_input=False)
                attempts = list(messages) + [
                    TextMessage(
                        content=(
                            "Your previous reply did not match the required JSON schema:\n"
                            f"{json.dumps(safe_errors, default=str)}\n"
                            "Answer again as JSON matching the schema exactly."
                        )
                    )
                ]

        raise last_error

    def _asked(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None,
        attempt: int,
        context: dict | None,
    ) -> str | BaseModel:
        """One question, asked again while the provider could still answer it.

        A rate limit, a timeout or a server error is the service rather than
        the request. The same call is made again after a pause. A refusal of
        the request itself is raised as it came. Every later scenario would be
        refused the same way, so the run should stop.

        A schema mismatch is not handled here. It is the caller's loop, which
        asks a different question, with the reason.
        """
        last: BaseException | None = None
        for attempt_number in range(1, failures.ATTEMPTS + 1):
            try:
                return self._complete(messages, response_model, attempt, context)
            except ValidationError:
                raise
            except Exception as exc:  # noqa: BLE001 - provider boundary
                if not failures.is_retryable(exc):
                    raise
                last = exc
                described = failures.describe(exc)
                if attempt_number == failures.ATTEMPTS:
                    break
                pause = failures.BACKOFF_SECONDS[attempt_number - 1]
                logger.warning(
                    "The provider did not answer (%s). Attempt %d of %d, waiting %.0fs.",
                    described,
                    attempt_number,
                    failures.ATTEMPTS,
                    pause,
                )
                self._wait(pause)

        raise ProviderUnavailable(
            f"The provider did not answer after {failures.ATTEMPTS} attempts: "
            f"{failures.describe(last) if last else 'no attempt was made'}"
        ) from last

    def _complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        attempt: int = 1,
        context: dict | None = None,
    ) -> str | BaseModel:
        request_messages = list(messages)
        if response_model:
            # One contract is used through every LiteLLM route. Native schema
            # grammars differ in supported keywords and complexity ceilings;
            # Yewo instead asks every model for JSON and validates the same
            # Pydantic model locally, with one corrective retry on mismatch.
            request_messages.append(TextMessage(content=JSON_INSTRUCTION))

        kwargs: dict = {
            "model": self._model,
            "messages": prompt_cache.mark_cacheable_prefix(
                [m.to_dict() for m in request_messages], self._model
            ),
        }

        # These are OpenRouter request fields, not generic LiteLLM fields.
        # Direct providers receive only the standard request that LiteLLM
        # translates for their API.
        if self._model.startswith("openrouter/"):
            extra_body: dict[str, object] = {
                "plugins": [{"id": "response-healing"}]
            }
            if self._cache_affinity:
                extra_body.update(
                    {
                        "session_id": self._cache_affinity,
                        "prompt_cache_key": self._cache_affinity,
                    }
                )
            kwargs["extra_body"] = extra_body
        # Recorded before the call, so a question that was never answered is
        # still a question this run can be shown to have asked.
        asked = (
            self.prompt_store.store_messages(request_messages) if self.prompt_store else None
        )

        response = litellm.completion(**kwargs)

        if self.usage_callback:
            self.usage_callback(
                usage.record(
                    response,
                    self._model,
                    attempt,
                    asked=asked,
                    schema=prompts.schema_sha256(response_model),
                    answering=response_model.__name__ if response_model else None,
                )
            )

        content = response.choices[0].message.content

        if not response_model:
            return content
        return self._parse(content, response_model, context)

    def _parse(
        self,
        content: str,
        response_model: type[BaseModel],
        context: dict | None = None,
    ) -> BaseModel:
        stripped = (content or "").strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            stripped = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return response_model.model_validate_json(stripped, context=context)
