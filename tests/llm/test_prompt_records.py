"""What each model call asked, and the policy for keeping it.

`llm_calls.jsonl` kept every answer and no question, so every claim about a
prompt was a reconstruction from the code that builds it. These cases hold
the record to being readable, content-addressed, free of embedded images, and
free of any declared secret.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel

from yewo.llm.messages import ImageMessage, MultiPartMessage, SystemMessage, TextMessage
from yewo.llm.prompts import PromptStore, schema_sha256
from yewo.utils.secrets import SecretScrubber


class Answer(BaseModel):
    verdict: str


def store(directory, **kwargs) -> PromptStore:
    scrubber = SecretScrubber()
    scrubber.learn_secrets({"password": "password123"})
    return PromptStore(Path(directory), scrub=scrubber.scrub, **kwargs)


def body(reference: str, directory) -> dict:
    return json.loads((Path(directory) / Path(reference).name).read_text())


class HashTests(unittest.TestCase):
    def test_the_same_prompt_is_named_the_same_way(self):
        with tempfile.TemporaryDirectory() as directory:
            keeper = store(directory)
            first, _ = keeper.store_messages([TextMessage(content="hello")])
            second, _ = keeper.store_messages([TextMessage(content="hello")])

        self.assertEqual(first, second)

    def test_a_different_prompt_is_named_differently(self):
        with tempfile.TemporaryDirectory() as directory:
            keeper = store(directory)
            first, _ = keeper.store_messages([TextMessage(content="hello")])
            second, _ = keeper.store_messages([TextMessage(content="goodbye")])

        self.assertNotEqual(first, second)

    def test_the_role_is_part_of_what_was_asked(self):
        """The same words as a system rule and as a user turn are two prompts."""
        with tempfile.TemporaryDirectory() as directory:
            keeper = store(directory)
            first, _ = keeper.store_messages([TextMessage(content="be brief")])
            second, _ = keeper.store_messages([SystemMessage(content="be brief")])

        self.assertNotEqual(first, second)

    def test_a_different_screen_is_a_different_prompt(self):
        """Otherwise every step of a run would appear to have asked one thing."""
        with tempfile.TemporaryDirectory() as directory:
            keeper = store(directory)
            first, _ = keeper.store_messages([ImageMessage(text="look", png_bytes=b"screen-one")])
            second, _ = keeper.store_messages([ImageMessage(text="look", png_bytes=b"screen-two")])

        self.assertNotEqual(first, second)

    def test_one_prompt_is_written_once(self):
        with tempfile.TemporaryDirectory() as directory:
            keeper = store(directory)
            keeper.store_messages([TextMessage(content="hello")])
            keeper.store_messages([TextMessage(content="hello")])

            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_a_required_answer_shape_is_recorded_beside_the_prompt(self):
        """A renamed field changes the answer, and nothing else would record it."""
        self.assertIsNotNone(schema_sha256(Answer))
        self.assertIsNone(schema_sha256(None))


class BodyTests(unittest.TestCase):
    def test_the_text_of_every_message_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            _, reference = store(directory).store_messages(
                [SystemMessage(content="be brief"), TextMessage(content="sign in")]
            )
            kept = body(reference, directory)

        self.assertEqual(
            [message["content"] for message in kept["messages"]],
            ["be brief", "sign in"],
        )

    def test_an_image_is_referenced_and_never_embedded(self):
        """A screenshot reaches a prompt as a megabyte of base64 text."""
        with tempfile.TemporaryDirectory() as directory:
            _, reference = store(directory).store_messages(
                [ImageMessage(text="which button?", png_bytes=b"a-whole-screenshot")]
            )
            written = (Path(directory) / Path(reference).name).read_text()
            kept = body(reference, directory)

        self.assertNotIn("base64", written)
        self.assertNotIn("data:image", written)
        parts = kept["messages"][0]["content"]
        self.assertEqual(parts[0]["text"], "which button?")
        self.assertEqual(parts[1]["type"], "image_url")
        self.assertIsNotNone(parts[1]["image_sha256"])

    def test_a_multipart_prompt_keeps_its_text_and_references_its_images(self):
        with tempfile.TemporaryDirectory() as directory:
            message = MultiPartMessage().add_text("compare these").add_image(b"png")
            _, reference = store(directory).store_messages([message])
            parts = body(reference, directory)["messages"][0]["content"]

        self.assertEqual(parts[0]["text"], "compare these")
        self.assertIn("image_sha256", parts[1])

    def test_a_declared_secret_never_reaches_the_body(self):
        with tempfile.TemporaryDirectory() as directory:
            _, reference = store(directory).store_messages(
                [TextMessage(content="type 'password123' into Password")]
            )
            written = (Path(directory) / Path(reference).name).read_text()

        self.assertNotIn("password123", written)
        self.assertIn("{{password}}", written)


class PolicyTests(unittest.TestCase):
    """The mode that records what was asked without keeping it."""

    def test_nothing_is_written_when_inputs_must_not_be_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            digest, reference = store(directory, keep_bodies=False).store_messages(
                [TextMessage(content="type 'password123' into Password")]
            )

            self.assertIsNotNone(digest)
            self.assertIsNone(reference)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_two_runs_can_still_be_compared_without_a_body(self):
        """Which is the whole reason the hash is written either way."""
        with tempfile.TemporaryDirectory() as directory:
            kept = store(directory, keep_bodies=True)
            quiet = store(directory, keep_bodies=False)

            with_body, _ = kept.store_messages([TextMessage(content="sign in")])
            without_body, _ = quiet.store_messages([TextMessage(content="sign in")])

        self.assertEqual(with_body, without_body)

    def test_a_prompt_that_cannot_be_read_is_recorded_as_nothing(self):
        """A record is a record. A prompt half-read would be a guess."""

        class Unreadable:
            def to_dict(self):
                raise RuntimeError("no")

        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(store(directory).store_messages([Unreadable()]), (None, None))

    def test_a_directory_that_cannot_be_written_still_yields_the_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "file"
            blocked.write_text("not a directory")
            digest, reference = PromptStore(blocked / "prompts").store_messages(
                [TextMessage(content="hello")]
            )

        self.assertIsNotNone(digest)
        self.assertIsNone(reference)


class AgentWiringTests(unittest.TestCase):
    """Every client the agent built records what it asked, or one route is blind."""

    def _agent(self, grounder=None):
        from types import SimpleNamespace

        from yewo.agent.run.agent import DeviceAgent

        agent = DeviceAgent.__new__(DeviceAgent)
        agent.llm = SimpleNamespace(prompt_store=None)
        agent.fast_llm = SimpleNamespace(prompt_store=None)
        agent.grounder = grounder
        return agent

    def test_both_completion_clients_are_told(self):
        with tempfile.TemporaryDirectory() as directory:
            agent = self._agent()
            keeper = store(directory)
            agent.record_prompts(keeper)

        self.assertIs(agent.llm.prompt_store, keeper)
        self.assertIs(agent.fast_llm.prompt_store, keeper)

    def test_the_grounder_is_told_too(self):
        """It asks its own service, and a run that recorded one route is half a record."""
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            grounder = SimpleNamespace(client=SimpleNamespace(prompt_store=None))
            agent = self._agent(grounder)
            keeper = store(directory)
            agent.record_prompts(keeper)

        self.assertIs(grounder.client.prompt_store, keeper)

    def test_an_agent_with_no_grounder_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            self._agent().record_prompts(store(directory))


class ImageIdentityTests(unittest.TestCase):
    """One image has one identity, whichever route carried it."""

    def test_the_hash_names_the_picture_and_not_its_encoding(self):
        """The decision route hashes the bytes it was handed. These have to agree."""
        import hashlib

        from yewo.llm.prompts import _image_digest

        png = b"\x89PNG\r\n\x1a\n" + bytes(range(64))
        message = ImageMessage(text="look", png_bytes=png)
        url = message.to_dict()["content"][1]["image_url"]["url"]

        self.assertEqual(_image_digest(url), hashlib.sha256(png).hexdigest())

    def test_the_stored_reference_matches_the_screenshot_on_disk(self):
        import hashlib

        png = b"\x89PNG\r\n\x1a\n" + bytes(range(32))
        with tempfile.TemporaryDirectory() as directory:
            _, reference = store(directory).store_messages(
                [ImageMessage(text="which button?", png_bytes=png)]
            )
            part = body(reference, directory)["messages"][0]["content"][1]

        self.assertEqual(part["image_sha256"], hashlib.sha256(png).hexdigest())

    def test_a_url_that_is_not_inline_data_is_left_unhashed(self):
        """The record says what was sent. A fetch would say what is there now."""
        from yewo.llm.prompts import _image_digest

        self.assertIsNone(_image_digest("https://example.com/shot.png"))

    def test_data_that_will_not_decode_is_recorded_as_nothing(self):
        from yewo.llm.prompts import _image_digest

        self.assertIsNone(_image_digest("data:image/png;base64,not!valid!base64"))


class PayloadScrubbingTests(unittest.TestCase):
    """The decision route's questions are app content at every depth."""

    def test_a_declared_value_is_masked_however_deeply_it_is_nested(self):
        with tempfile.TemporaryDirectory() as directory:
            _, reference = store(directory).store_payload(
                {
                    "state": "`q_img`",
                    "questions": {
                        "where": {
                            "type": "box",
                            "instructions": "the field holding password123",
                        },
                        "others": ["password123", {"deep": ["password123"]}],
                    },
                }
            )
            written = (Path(directory) / Path(reference).name).read_text()

        self.assertNotIn("password123", written)
        self.assertIn("{{password}}", written)

    def test_the_hash_is_over_the_scrubbed_form(self):
        """Otherwise a rotated secret would change the identity of one question."""
        with tempfile.TemporaryDirectory() as directory:
            keeper = store(directory)
            first, _ = keeper.store_payload({"q": "find password123"})
            second, _ = keeper.store_payload({"q": "find {{password}}"})

        self.assertEqual(first, second)

    def test_nothing_is_written_when_inputs_must_not_be_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            digest, reference = store(directory, keep_bodies=False).store_payload(
                {"q": "find password123"}
            )

            self.assertIsNotNone(digest)
            self.assertIsNone(reference)
            self.assertEqual(list(Path(directory).iterdir()), [])
