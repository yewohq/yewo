"""Client for the decision route: a state, typed questions, one answer each.

Sibling to LLMClient rather than a subclass. The two share no request or
response shape, only the `usage.CallRecord` they leave behind.
"""

import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Mapping, Optional

from yewo.llm import failures, usage
from yewo.utils.jev_deps import require_jev

logger = logging.getLogger(__name__)

DEFAULT_DECISION_MODEL = "typesafe/jev-1.14.0"
"""Pinned, not aliased: "latest" resolves to a version that refuses a box
question."""

IMAGES_HEADER = {"TypeSafe-Danger-Zone-Images": "true"}
"""Required on any request carrying an image or a box question."""

MAX_SIDE = 1568
MIN_SIDE = 280
MAX_PIXELS = 2_458_624
"""The API's limits. A phone screenshot exceeds all three."""

ASSET = "q_img"
"""The screenshot's name. Substituted wherever it appears in the request text,
backticks or not, so it must be a word no question will use."""

FALLBACK_INPUT_COST = 0.042 / 1_000_000
"""Used only if LiteLLM's price map has nothing for the model. Output is free."""


@dataclass(frozen=True)
class PreparedImage:
    """A screenshot as the decision model will see it."""

    png: bytes
    original: tuple[int, int]
    sent: tuple[int, int]

    def to_device(self, left: float, top: float, right: float, bottom: float) -> tuple:
        """A fractional box, in the coordinates of the device that produced it."""
        return to_device(self.original, left, top, right, bottom)


def to_device(
    size: tuple[int, int], left: float, top: float, right: float, bottom: float
) -> tuple[int, int, int, int]:
    """A fractional box, in the pixels of a frame of this size."""
    width, height = size
    return (
        round(left * width), round(top * height),
        round(right * width), round(bottom * height),
    )


def device_size(png: bytes) -> tuple[int, int]:
    """The frame's own size, from the header rather than the pixels."""
    from PIL import Image

    with Image.open(BytesIO(png)) as image:
        return image.size


def prepare_image(png: bytes) -> PreparedImage:
    """Shrinks a screenshot to the largest size the API accepts."""
    from PIL import Image

    image = Image.open(BytesIO(png)).convert("RGB")
    width, height = image.size

    factor = min(
        MAX_SIDE / max(width, height),
        (MAX_PIXELS / (width * height)) ** 0.5,
        1.0,
    )
    sent = (max(MIN_SIDE, round(width * factor)), max(MIN_SIDE, round(height * factor)))

    buffer = BytesIO()
    image.resize(sent, Image.Resampling.LANCZOS).save(buffer, format="PNG")
    return PreparedImage(png=buffer.getvalue(), original=(width, height), sent=sent)


def input_price(model: str) -> float:
    """What one input token costs, read from LiteLLM's price map."""
    try:
        import litellm

        price = (litellm.model_cost.get(model) or {}).get("input_cost_per_token")
        if price is not None:
            return float(price)
    except Exception as exc:  # noqa: BLE001 - pricing must never fail a run
        logger.debug("No price for %s from LiteLLM: %s", model, exc)
    return FALLBACK_INPUT_COST


class JevClient:
    """Asks several typed questions of one screen, in one request.

    The screenshot goes in the state, which is billed once however many
    questions ride on it.
    """

    prompt_store = None
    """Class-level, so a client built without __init__ records nothing."""

    def __init__(
        self,
        model: str = DEFAULT_DECISION_MODEL,
        usage_callback=None,
        prompt_store=None,
    ):
        """Initializes the client.

        Args:
            model: The decision model to use. Pinned by version, not aliased.
            usage_callback: Optional callable receiving a usage.CallRecord per call.
            prompt_store: Where to record what each call asked. None records
                nothing.
        """
        require_jev()
        self._model = model
        self._price = input_price(model)
        self.usage_callback = usage_callback
        self.prompt_store = prompt_store
        self._client = None

    @property
    def client(self):
        """The vendor client, built on first use so construction stays cheap.

        Built with its own retries off. The SDK retries twice by default and
        `failures.call_with_retry` retries three times around it. One rate limit
        then became nine requests, and three backoff budgets, while a device
        waited. One owner for the policy, and it is ours.
        """
        if self._client is None:
            from typesafe_sdk import RetryPolicy, TypeSafeClient

            self._client = TypeSafeClient(
                model=self._version, retry=RetryPolicy(max_retries=0)
            )
        return self._client

    @property
    def _version(self) -> str:
        """The model name the vendor knows, without LiteLLM's provider prefix."""
        return self._model.split("/", 1)[-1]

    def ask(self, questions: Mapping[str, Any], screenshot: Optional[bytes] = None):
        """Answers every question, about the screenshot when one is given.

        Args:
            questions: Named questions, as the vendor's question objects or
                their mapping form.
            screenshot: PNG bytes of the screen to ask about, already at device size.

        Returns:
            The vendor's response, carrying one answer per question name.
        """
        state, assets = "", None
        if screenshot is not None:
            from typesafe_sdk import Image

            state = f"`{ASSET}`"
            assets = {ASSET: Image(prepare_image(screenshot).png)}

        prompt_reference = self._store_prompt(state, questions, screenshot)

        started = time.perf_counter()
        response = failures.call_with_retry(
            lambda: self.client.system_one(
                state=state,
                questions=dict(questions),
                assets=assets,
                extra_headers=IMAGES_HEADER,
            ),
            "The decision model",
        )
        latency_ms = (time.perf_counter() - started) * 1000

        if self.usage_callback:
            tokens = response.usage.input_tokens
            self.usage_callback(
                usage.record_decision(
                    response,
                    self._model,
                    latency_ms,
                    None if tokens is None else tokens * self._price,
                    asked=prompt_reference,
                )
            )
        return response

    def _store_prompt(
        self, state: str, questions: Mapping[str, Any], screenshot: Optional[bytes]
    ) -> Optional[tuple]:
        """What this call asked, recorded before it is made.

        The screenshot is referenced by its own hash rather than written
        again. The step that read the screen already keeps the image.
        """
        if self.prompt_store is None:
            return None
        import hashlib

        return self.prompt_store.store_payload(
            {
                "state": state,
                "questions": {
                    name: _serialize_question(question)
                    for name, question in questions.items()
                },
                "image_sha256": (
                    hashlib.sha256(screenshot).hexdigest() if screenshot else None
                ),
            }
        )

    def close(self) -> None:
        """Releases the vendor client's connections."""
        if self._client is not None:
            self._client.close()
            self._client = None


def _serialize_question(question) -> object:
    """One question, in a form a record can hold.

    The vendor's question objects are models. Anything else is kept as it
    arrived, because a record of what was asked must not depend on guessing
    the shape it was asked in.
    """
    dump = getattr(question, "model_dump", None)
    return dump() if callable(dump) else question
