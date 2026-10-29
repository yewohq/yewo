"""What a run spent, and what the file has to be able to answer later."""

import json
import tempfile
import unittest
from pathlib import Path


class UsageRecordTests(unittest.TestCase):
    """Every call is written down whole, so a later question still has an answer."""

    def _call(self, **overrides):
        from yewo.llm.usage import CallRecord

        values = {
            "model": "provider/model",
            "served_model": "provider/model-002",
            "generation_id": "gen-1",
            "attempt": 1,
            "finish_reason": "stop",
            "latency_ms": 1200.0,
            "cost": 0.004,
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        }
        values.update(overrides)
        return CallRecord(**values)

    def _recorded(self, *calls):
        from yewo.testing.report.usage_tracker import LLMUsageTracker

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(Path(directory))
            for call in calls:
                tracker.add_call("scenario:TS_1", call)
            return json.loads(tracker.save_report().read_text())

    def test_unknown_cost_is_reported_as_unknown(self):
        payload = self._recorded(self._call(cost=None))

        self.assertEqual(payload["cost_unknown_calls"], 1)
        call = payload["components"]["scenario:TS_1"]["calls"][0]
        self.assertFalse(call["cost_known"])
        self.assertEqual(call["model"], "provider/model")

    def test_a_truncated_answer_is_counted_not_just_recorded(self):
        """ "length" means the answer was cut off, which reads as a bad answer."""
        payload = self._recorded(self._call(finish_reason="length"))

        self.assertEqual(payload["truncated_calls"], 1)

    def test_a_retry_is_counted_as_one(self):
        payload = self._recorded(self._call(), self._call(attempt=2))

        self.assertEqual(payload["total_calls"], 2)
        self.assertEqual(payload["retried_calls"], 1)

    def test_cached_and_reasoning_tokens_are_totalled(self):
        payload = self._recorded(self._call(cached_tokens=900, reasoning_tokens=40))

        self.assertEqual(payload["total_cached_tokens"], 900)
        self.assertEqual(payload["total_reasoning_tokens"], 40)

    def test_the_whole_response_is_kept(self):
        payload = self._recorded(self._call(raw={"id": "gen-1", "choices": []}))

        self.assertEqual(payload["components"]["scenario:TS_1"]["calls"][0]["raw"]["id"], "gen-1")

    def test_declared_secrets_are_masked_inside_the_dumped_response(self):
        from yewo.testing.report.usage_tracker import LLMUsageTracker
        from yewo.utils.secrets import SecretScrubber

        scrubber = SecretScrubber()
        scrubber.learn_secrets({"password": "hunter2"})
        call = self._call(raw={"choices": [{"text": "I typed hunter2 there"}]})

        with tempfile.TemporaryDirectory() as directory:
            tracker = LLMUsageTracker(Path(directory), scrubber=scrubber)
            tracker.add_call("scenario:TS_1", call)
            written = tracker.save_report().read_text()

        self.assertNotIn("hunter2", written)
        self.assertIn("{{password}}", written)


class UsageTotalsTests(unittest.TestCase):
    """The totals are a fixed set of fields, so they are a type rather than a dict."""

    def _call(self, **overrides):
        from yewo.llm.usage import CallRecord

        values = {
            "model": "provider/model",
            "served_model": "provider/model",
            "generation_id": "gen-1",
            "attempt": 1,
            "finish_reason": "stop",
            "latency_ms": 100.0,
            "cost": 0.001,
        }
        values.update(overrides)
        return CallRecord(**values)

    def test_totals_accumulate_across_calls(self):
        from yewo.testing.report.usage_tracker import UsageTotals

        totals = UsageTotals()
        totals.add(self._call(prompt_tokens=10, cost=0.002))
        totals.add(self._call(prompt_tokens=5, cost=0.003))

        self.assertEqual(totals.total_calls, 2)
        self.assertEqual(totals.total_prompt_tokens, 15)
        self.assertAlmostEqual(totals.total_estimated_cost_usd, 0.005)

    def test_a_missing_cost_is_counted_not_treated_as_free(self):
        from yewo.testing.report.usage_tracker import UsageTotals

        totals = UsageTotals()
        totals.add(self._call(cost=None))

        self.assertEqual(totals.cost_unknown_calls, 1)
        self.assertEqual(totals.total_estimated_cost_usd, 0.0)

    def test_the_report_keeps_the_field_names_already_on_disk(self):
        from yewo.testing.report.usage_tracker import UsageTotals
        from dataclasses import asdict

        written = asdict(UsageTotals())

        for name in ("total_calls", "total_estimated_cost_usd", "components"):
            self.assertIn(name, written)


class DecisionRecordTests(unittest.TestCase):
    """A decision call reports less than a completion, and names itself differently."""

    def _response(self, request_id="req_01a1", raises=False):
        from types import SimpleNamespace

        class Response(SimpleNamespace):
            @property
            def request_id(self):
                if raises:
                    raise RuntimeError("The response did not include a request ID.")
                return request_id

        return Response(
            model="jev-1.14.0",
            usage=SimpleNamespace(input_tokens=3216, output_tokens=36),
        )

    def _record(self, response):
        from yewo.llm.usage import record_decision

        return record_decision(response, "typesafe/jev-1.14.0", 264.0, 0.000135)

    def test_the_call_is_named_by_the_id_the_service_gave_it(self):
        """A decision response carries no `id`, so reading one stored a blank."""
        self.assertEqual(self._record(self._response()).generation_id, "req_01a1")

    def test_a_response_that_names_no_call_is_still_recorded(self):
        """The SDK raises when the header is absent, and usage must not raise."""
        record = self._record(self._response(raises=True))

        self.assertEqual(record.generation_id, "")
        self.assertEqual(record.total_tokens, 3252)

    def test_the_tokens_a_decision_reports_are_what_is_counted(self):
        record = self._record(self._response())

        self.assertEqual(record.prompt_tokens, 3216)
        self.assertEqual(record.completion_tokens, 36)
        self.assertEqual(record.total_tokens, 3252)


class ResponseReadingTests(unittest.TestCase):
    """What a provider sent, read into the record a run keeps.

    Every field here is the provider's, so each one is optional in practice. A
    count nothing reported is zero, and a measurement nothing reported is
    unknown. Those are different answers, and writing one as the other puts a
    number in a report that nobody measured.
    """

    def _response(self, usage=None, hidden=None):
        from types import SimpleNamespace

        block = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        block.update(usage or {})
        return SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: block),
            model="provider/model-002",
            id="gen-1",
            choices=[SimpleNamespace(finish_reason="stop")],
            _hidden_params=hidden if hidden is not None else {"_response_ms": 1200.0},
            model_dump=lambda: {"usage": block},
        )

    def _record(self, **kwargs):
        from yewo.llm.usage import record

        return record(self._response(**kwargs), "provider/model", 1)

    def test_a_latency_the_route_did_not_report_is_unknown(self):
        """Zero would read as an instant answer, which nothing observed."""
        self.assertIsNone(self._record(hidden={}).latency_ms)

    def test_a_reported_latency_is_kept_as_a_number(self):
        self.assertEqual(self._record().latency_ms, 1200.0)

    def test_a_count_the_provider_left_out_is_zero(self):
        """A block that breaks out no reasoning tokens spent none."""
        record = self._record()

        self.assertEqual(record.reasoning_tokens, 0)
        self.assertEqual(record.cached_tokens, 0)

    def test_an_image_count_the_provider_left_out_stays_unknown(self):
        """Zero image tokens would say the call sent no image, which it may have."""
        self.assertIsNone(self._record().prompt_image_tokens)

    def test_a_count_that_is_not_a_number_is_not_believed(self):
        """The block is the provider's, so its types are not ours to assume."""
        record = self._record(usage={"prompt_tokens": "lots"})

        self.assertEqual(record.prompt_tokens, 0)

    def test_the_counts_a_provider_did_report_are_kept(self):
        record = self._record(
            usage={
                "prompt_tokens_details": {"cached_tokens": 7, "image_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 11},
            }
        )

        self.assertEqual(record.cached_tokens, 7)
        self.assertEqual(record.prompt_image_tokens, 4)
        self.assertEqual(record.reasoning_tokens, 11)

    def test_a_cache_write_anthropic_spelled_its_own_way_is_counted(self):
        """Reading one spelling recorded zero writes on every cached run.

        The write is the expensive half of a prompt cache, so a zero there
        reads as caching that never engaged.
        """
        record = self._record(usage={"prompt_tokens_details": {"cache_creation_tokens": 3559}})

        self.assertEqual(record.cache_write_tokens, 3559)

    def test_a_provider_reported_cost_is_kept_without_repricing_it(self):
        from unittest.mock import patch

        with patch("yewo.llm.usage.completion_cost") as calculated:
            record = self._record(usage={"cost": 0.0123})

        self.assertEqual(record.cost, 0.0123)
        calculated.assert_not_called()

    def test_litellm_prices_a_call_whose_provider_reports_no_cost(self):
        from unittest.mock import patch

        with patch("yewo.llm.usage.completion_cost", return_value=0.0456):
            record = self._record()

        self.assertEqual(record.cost, 0.0456)

    def test_an_unpriced_model_stays_unknown_without_losing_the_call(self):
        from unittest.mock import patch

        with patch(
            "yewo.llm.usage.completion_cost",
            side_effect=RuntimeError("model has no price"),
        ):
            record = self._record()

        self.assertIsNone(record.cost)


class AnsweredShapeTests(unittest.TestCase):
    """Which component made a call, where two of them share one client.

    The planner and the auditor both answer through the same client whenever
    no fast model is configured. The call site cannot name them apart. What
    each asked for is different, and that is a fact of the call.
    """

    def _response(self):
        from types import SimpleNamespace

        block = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        return SimpleNamespace(
            usage=SimpleNamespace(model_dump=lambda: block),
            model="provider/model-002",
            id="gen-1",
            choices=[SimpleNamespace(finish_reason="stop")],
            _hidden_params={"_response_ms": 10.0},
            model_dump=lambda: {"usage": block},
        )

    def test_the_shape_asked_for_is_recorded_by_name(self):
        from yewo.llm.usage import record

        kept = record(self._response(), "provider/model", 1, answering="QAStepResponse")

        self.assertEqual(kept.answering, "QAStepResponse")

    def test_a_call_that_required_no_shape_names_none(self):
        from yewo.llm.usage import record

        self.assertIsNone(record(self._response(), "provider/model", 1).answering)

    def test_the_client_names_the_model_it_asked_for(self):
        """Read off the response model, so it cannot disagree with the schema hash."""
        from unittest.mock import patch

        from pydantic import BaseModel

        from yewo.llm.llm_client import LLMClient
        from yewo.llm.messages import TextMessage

        class Verdict(BaseModel):
            ok: bool

        kept = []
        client = LLMClient(model="provider/model", usage_callback=kept.append)
        answer = self._response()
        answer.choices = [
            type(
                "Choice",
                (),
                {"finish_reason": "stop", "message": type("M", (), {"content": '{"ok": true}'})()},
            )()
        ]

        with patch("yewo.llm.llm_client.litellm.completion", return_value=answer):
            client.generate_response([TextMessage(content="hi")], Verdict)

        self.assertEqual(kept[0].answering, "Verdict")
        self.assertIsNotNone(kept[0].schema_sha256)


class ScenarioComponentTests(unittest.TestCase):
    def test_answer_shapes_separate_planning_from_fresh_verification(self):
        from yewo.testing.run_usage import (
            PLANNER,
            SCREEN_AUDITOR,
            TRANSITION_JUDGE,
            component_for,
        )

        self.assertEqual(component_for(PLANNER, "QAStepResponse"), PLANNER)
        self.assertEqual(component_for(PLANNER, "ScreenAssessment"), SCREEN_AUDITOR)
        self.assertEqual(component_for(PLANNER, "TransitionVerdicts"), TRANSITION_JUDGE)
