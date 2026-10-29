"""Sending the schema an answer has to match, with the request that asks for it.

Asked for a shape in prose alone, a model infers property names from the
surrounding words. A live Opus 5 run wrote `description` where `ScenarioDraft`
requires `instruction`, then `type` where the discriminator is `kind`, and the
correction renamed one field per attempt without ever converging. Three
attempts produced nothing and the cell was recorded as uncovered.

The schema is the only thing that states those names, so it travels with the
request. No provider is asked to enforce it.
"""

import json
import unittest
from types import SimpleNamespace
from typing import Annotated, Literal
from unittest import mock

from pydantic import BaseModel, Field


class TapAction(BaseModel):
    kind: Literal["tap"]
    instruction: str


class WaitAction(BaseModel):
    kind: Literal["wait"]
    seconds: float


class DraftAnswer(BaseModel):
    scenario_name: str
    action: Annotated[TapAction | WaitAction, Field(discriminator="kind")]


class JsonContractTests(unittest.TestCase):
    """What the request says about the shape it wants back."""

    def _request(self, model="openrouter/test/model", reply=None):
        from yewo.llm.llm_client import LLMClient
        from yewo.llm.messages import TextMessage

        body = reply or '{"scenario_name":"s","action":{"kind":"tap","instruction":"go"}}'
        answer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))]
        )
        with mock.patch("litellm.completion", return_value=answer) as completion:
            LLMClient(model=model).generate_response(
                [TextMessage(content="write one")], DraftAnswer
            )
        return completion.call_args.kwargs

    def _contract(self, **kwargs):
        return self._request(**kwargs)["messages"][-1]["content"]

    def test_the_required_property_names_reach_the_model(self):
        """`instruction` was renamed to `description` because nothing stated it."""
        contract = self._contract()

        self.assertIn("instruction", contract)
        self.assertIn("scenario_name", contract)

    def test_the_discriminator_name_reaches_the_model(self):
        """`kind` was answered as `type`, which no error message names."""
        contract = self._contract()

        self.assertIn("kind", contract)

    def test_the_contract_carries_the_schema_of_the_answer_it_asks_for(self):
        contract = self._contract()
        schema = json.loads(contract.split("\n", 1)[1])

        self.assertEqual(schema, DraftAnswer.model_json_schema())

    def test_one_answer_shape_yields_one_contract_string(self):
        """A reordered mapping reads as a different prompt and loses the cache."""
        self.assertEqual(self._contract(), self._contract())

    def test_every_provider_receives_the_same_contract(self):
        for model in (
            "anthropic/example-model",
            "openai/example-model",
            "openrouter/example-model",
            "gemini/example-model",
        ):
            self.assertEqual(
                self._contract(model=model),
                self._contract(model="openrouter/test/model"),
                model,
            )

    def test_no_provider_is_asked_to_enforce_the_schema(self):
        """Native grammars differ in supported keywords, so none is used."""
        for model in ("anthropic/example-model", "openai/example-model"):
            request = self._request(model=model)

            self.assertNotIn("response_format", request, model)
            self.assertNotIn("output_config", request, model)
            self.assertNotIn("tools", request, model)

    def test_a_plain_call_is_sent_no_contract(self):
        from yewo.llm.llm_client import LLMClient
        from yewo.llm.messages import TextMessage

        answer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        with mock.patch("litellm.completion", return_value=answer) as completion:
            LLMClient(model="openrouter/test/model").generate_response(
                [TextMessage(content="hi")]
            )

        self.assertEqual(
            [message["content"] for message in completion.call_args.kwargs["messages"]],
            ["hi"],
        )

    def test_a_corrected_attempt_still_carries_the_schema(self):
        """The correction names invalid paths. Only the schema names the fields."""
        from yewo.llm.llm_client import LLMClient
        from yewo.llm.messages import TextMessage

        rejected = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"scenario_name":"s","action":{"type":"tap","description":"go"}}'
                    )
                )
            ]
        )
        accepted = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"scenario_name":"s","action":{"kind":"tap","instruction":"go"}}'
                    )
                )
            ]
        )
        with mock.patch(
            "litellm.completion", side_effect=[rejected, accepted]
        ) as completion:
            LLMClient(model="openrouter/test/model").generate_response(
                [TextMessage(content="write one")], DraftAnswer
            )

        retried = completion.call_args_list[1].kwargs["messages"]
        self.assertIn("instruction", retried[-1]["content"])
        self.assertIn("did not match", retried[-2]["content"])


if __name__ == "__main__":
    unittest.main()
