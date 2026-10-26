"""Reading an answer written in a shape the schema did not anticipate.

A rejected assessment takes the good verdicts beside the odd one with it, and
the retry answers in the same shape, so the screen is lost and the step pays
twice for it. Every shape here came back from a live run.
"""

import unittest
from typing import Annotated, Literal
from types import SimpleNamespace
from unittest import mock

from pydantic import BaseModel, Field, ValidationError


class FirstAction(BaseModel):
    type: Literal["first"]


class SecondAction(BaseModel):
    type: Literal["second"]


class StructuredAnswer(BaseModel):
    action: Annotated[FirstAction | SecondAction, Field(discriminator="type")]


class ProviderShapeTests(unittest.TestCase):
    """One unexpected field name used to cost a whole screen's audit.

    A rejected assessment takes the good verdicts beside the odd one with it,
    and the retry answers in the same shape, so the screen is lost and a step
    pays twice for it. Every shape here came back from a live run.
    """

    def _parse(self, **body):
        from yewo.testing.judge.qa_auditor import ScreenAssessment

        return ScreenAssessment.model_validate(body)

    def _verdicts(self, *verdicts):
        return self._parse(screen_name="Checkout", rule_verifications=list(verdicts))

    def test_the_screen_named_as_current_screen_is_read(self):
        """The planner's field name for the same thing."""
        assessment = self._parse(current_screen="Sign-in screen", rule_verifications=[])

        self.assertEqual(assessment.screen_name, "Sign-in screen")

    def test_a_verdict_that_gives_only_an_id_still_names_its_rule(self):
        """An id names a rule as well as its words do."""
        kept = self._verdicts(
            {
                "rule_id": "expected:0",
                "status": "untested",
                "applicable": False,
                "reasoning": "No sign-in UI is rendered here.",
            }
        )

        self.assertEqual(kept.rule_verifications[0].rule_id, "expected:0")
        self.assertEqual(kept.rule_verifications[0].rule, "expected:0")

    def test_a_status_spelled_short_is_the_same_status(self):
        kept = self._verdicts(
            {"rule": "expected:0", "status": "pass", "reasoning": "visible"},
            {"rule": "expected:1", "status": "fail", "reasoning": "not masked"},
        )

        self.assertEqual([v.status for v in kept.rule_verifications], ["passed", "failed"])

    def test_a_status_that_is_not_one_of_the_three_is_refused(self):
        """A verdict is the thing being recorded, so guessing at one invents it."""
        with self.assertRaises(ValidationError):
            self._verdicts({"rule": "expected:0", "status": "maybe", "reasoning": "r"})

    def test_one_odd_field_does_not_lose_the_verdicts_beside_it(self):
        """This is what the rejection cost: eleven answers for one bad name."""
        kept = self._parse(
            current_screen="Checkout",
            rule_verifications=[
                {"rule_id": f"expected:{index}", "status": "untested", "reasoning": "r"}
                for index in range(11)
            ],
        )

        self.assertEqual(len(kept.rule_verifications), 11)


class FindingShapeTests(unittest.TestCase):
    """`type` arrives meaning the name of the value, and meaning its rendering."""

    def _finding(self, **body):
        from yewo.testing.models import SensitiveFinding

        return SensitiveFinding.model_validate(body)

    def test_a_type_that_names_a_rendering_is_the_rendering(self):
        from yewo.testing.models import Rendering

        finding = self._finding(type="masked", location="Card preview", quoted="•••• 4444")

        self.assertIs(finding.rendering, Rendering.MASKED)
        self.assertEqual(finding.field, "Card preview")

    def test_a_type_that_names_no_rendering_is_the_field(self):
        from yewo.testing.models import Rendering

        finding = self._finding(type="CVV", state="label_only")

        self.assertEqual(finding.field, "CVV")
        self.assertIs(finding.rendering, Rendering.LABEL_ONLY)

    def test_a_rendering_under_any_of_its_names_is_the_rendering(self):
        """`state` and `status` both arrive meaning how the value looked."""
        from yewo.testing.models import Rendering

        for alias in ("rendering", "state", "status"):
            finding = self._finding(field="CVV", **{alias: "absent"})

            self.assertIs(finding.rendering, Rendering.ABSENT, alias)

    def test_nothing_quoted_is_nothing_rather_than_missing(self):
        finding = self._finding(field="CVV", rendering="masked", quoted=None)

        self.assertEqual(finding.quoted, "")

    def test_an_exposure_is_still_decided_by_the_rendering(self):
        """The whole point of the enum survives the renaming."""
        self.assertTrue(self._finding(type="value", location="CVV", quoted="321").is_exposure())
        self.assertFalse(self._finding(type="masked", location="CVV").is_exposure())


class ProviderRoutingTests(unittest.TestCase):
    """Provider-specific fields must not leak into another provider's API."""

    def _send(self, model="openrouter/test/model", cache_affinity=None):
        from types import SimpleNamespace
        from unittest import mock

        from yewo.llm.llm_client import LLMClient
        from yewo.llm.messages import TextMessage

        answer = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
        with mock.patch("litellm.completion", return_value=answer) as completion:
            LLMClient(
                model=model, cache_affinity=cache_affinity
            ).generate_response([TextMessage(content="hi")])
        return completion.call_args.kwargs

    def test_openrouter_receives_its_response_healing_plugin(self):
        self.assertEqual(
            self._send()["extra_body"]["plugins"], [{"id": "response-healing"}]
        )

    def test_openrouter_is_not_pinned_to_one_upstream_provider(self):
        self.assertNotIn("provider", self._send()["extra_body"])

    def test_a_run_identity_sets_both_cache_routes_without_exposing_it(self):
        body = self._send(cache_affinity="run-with-private-path")["extra_body"]

        self.assertEqual(body["session_id"], body["prompt_cache_key"])
        self.assertRegex(body["session_id"], r"^yewo-[0-9a-f]{64}$")
        self.assertNotIn("private-path", body["session_id"])

    def test_a_call_outside_a_run_sends_no_cache_route(self):
        body = self._send()["extra_body"]

        self.assertNotIn("session_id", body)
        self.assertNotIn("prompt_cache_key", body)

    def test_direct_providers_receive_no_openrouter_fields(self):
        for model in (
            "openai/example-model",
            "anthropic/example-model",
            "gemini/example-model",
        ):
            request = self._send(model=model, cache_affinity="run")

            self.assertNotIn("extra_body", request, model)

    def _send_structured(self, model):
        from yewo.llm.llm_client import LLMClient
        from yewo.llm.messages import TextMessage

        answer = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":{"type":"first"}}'))]
        )
        with mock.patch("litellm.completion", return_value=answer) as completion:
            LLMClient(model=model).generate_response(
                [TextMessage(content="choose")], StructuredAnswer
            )
        return completion.call_args.kwargs

    def test_every_provider_uses_the_same_locally_validated_json_contract(self):
        for model in (
            "anthropic/example-model",
            "openai/example-model",
            "openrouter/example-model",
        ):
            request = self._send_structured(model)

            self.assertNotIn("response_format", request, model)
            self.assertIn(
                "Return only one JSON object", request["messages"][-1]["content"], model
            )
