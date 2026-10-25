"""What a durable path is allowed to hold, and what it must never hold.

A trace is a file. Everything a run knows passes through the planner, and
the planner is given the resolved value, so the obvious shape of a cached
step carries a password. The rules here are what stops that: a typed step
keeps a name, an anchor keeps a name, and a target keeps no coordinate.

The planner may carry a real declared value or a safe placeholder. Neither
reaches the file. A hash of a six-character password is recovered offline, so
an absent string proves nothing. Changing the secret must change nothing in
the file at all.
"""

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from yewo.agent.actuation.actions import Coordinates, TypeAction
from yewo.agent.memory import replay_store, replay_walk
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo

PASSWORD = "secret-password123"
EMAIL = "qa.tester@example.com"


def journey_values(password=PASSWORD):
    return {"email": EMAIL, "password": password, "city": "Testing City"}


def element(text, role="button", box=(10, 20, 110, 70)):
    x_min, y_min, x_max, y_max = box
    return GUIElement(
        element=text or role,
        bounding_box=BoundingBox(x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max),
        confidence=0.9,
        type=role,
        text=text,
    )


def scan(width=1000, height=2000, count=1):
    return ScanInfo(
        mode="detect", reason="step", phash="p",
        screen_width=width, screen_height=height, element_count=count,
    )


class SecretTests(unittest.TestCase):
    """The file must not change when the value behind a name changes."""

    def _anchors(self, password):
        names = replay_store.ValueNames(journey_values(password))
        # A screen shows a typed value back, which is how one reaches a file.
        screen = [element("Email"), element(EMAIL, role="field"), element(password, role="field")]
        return replay_store.anchors_of(screen, scan(count=3), names)

    def test_changing_a_secret_leaves_the_anchors_identical(self):
        written = [
            json.dumps([a.model_dump() for a in self._anchors(value)], sort_keys=True)
            for value in (PASSWORD, "an-entirely-different-password")
        ]

        self.assertEqual(written[0], written[1])

    def test_the_value_is_not_in_the_file_under_any_spelling(self):
        written = json.dumps(
            [anchor.model_dump() for anchor in self._anchors(PASSWORD)]
        )

        self.assertNotIn(PASSWORD, written)
        self.assertNotIn("{{password}}", written)

    def test_no_words_from_the_screen_reach_the_file(self):
        """Naming the journey's own values is not enough. A screen carries
        an order number, a message, somebody's address. None of it is
        Yewo's to keep, and a path needs only to know two readings agree."""
        written = json.dumps(
            [anchor.model_dump() for anchor in self._anchors(PASSWORD)]
        )

        for words in (PASSWORD, EMAIL, "Email", "Testing City"):
            with self.subTest(words=words):
                self.assertNotIn(words, written)

    def test_one_label_reads_the_same_twice(self):
        """A digest is only useful if it is the same digest next run."""
        names = replay_store.ValueNames(journey_values())

        self.assertEqual(
            replay_store.text_identity("Sign in", names),
            replay_store.text_identity("Sign in", names),
        )

    def test_two_labels_read_differently(self):
        names = replay_store.ValueNames(journey_values())

        self.assertNotEqual(
            replay_store.text_identity("Sign in", names),
            replay_store.text_identity("Continue", names),
        )

    def test_a_field_holding_a_secret_reads_the_same_whatever_it_holds(self):
        """Which is what lets one saved path serve two accounts."""
        first = replay_store.text_identity(
            PASSWORD, replay_store.ValueNames(journey_values(PASSWORD))
        )
        second = replay_store.text_identity(
            "another", replay_store.ValueNames(journey_values("another"))
        )

        self.assertEqual(first, second)

    def test_an_element_with_no_words_keeps_none(self):
        self.assertEqual(replay_store.text_identity("", replay_store.ValueNames()), "")


class DigestTests(unittest.TestCase):
    """Screen text is keyed, because a plain digest of it is reversible."""

    def test_a_four_digit_code_cannot_be_recovered_from_its_digest(self):
        """Ten thousand candidates is not a search. Without the key there
        is nothing to search with."""
        key = os.urandom(32)
        names = replay_store.ValueNames()
        stored = replay_store.text_identity("4721", names, key)

        plain = hashlib.sha256(b"4721").hexdigest()[
            : replay_store.TEXT_DIGEST_LENGTH
        ]

        self.assertNotEqual(stored, plain)

    def test_the_same_text_and_key_read_the_same(self):
        key = os.urandom(32)
        names = replay_store.ValueNames()

        self.assertEqual(
            replay_store.text_identity("Sign in", names, key),
            replay_store.text_identity("Sign in", names, key),
        )

    def test_another_key_reads_differently(self):
        """So a trace carried off this machine matches nothing, which is a
        refusal rather than a fault."""
        names = replay_store.ValueNames()

        self.assertNotEqual(
            replay_store.text_identity("Sign in", names, os.urandom(32)),
            replay_store.text_identity("Sign in", names, os.urandom(32)),
        )

    def test_a_key_is_made_once_and_kept_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "replay-key"

            first = replay_store.digest_key(path)
            second = replay_store.digest_key(path)

            self.assertEqual(first, second)
            self.assertEqual(path.stat().st_mode & 0o077, 0, "readable by others")

    def test_a_key_that_cannot_be_read_still_returns_one(self):
        """No key means no durable identity. It must not mean no run."""
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "a-file"
            blocked.write_text("in the way")

            self.assertEqual(len(replay_store.digest_key(blocked / "key")), 32)

    def test_an_explicit_shared_key_is_portable_without_being_written(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {replay_store.PATH_KEY_ENV: "team-path-key"},
            ),
        ):
            path = Path(directory) / "replay-key"

            first = replay_store.digest_key(path)
            second = replay_store.digest_key(Path(directory) / "another-key")

            self.assertEqual(first, second)
            self.assertFalse(path.exists())

    def test_a_concurrent_creator_uses_the_key_that_won(self):
        """A second process can create the file after the missing read."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay-key"
            winner = b"w" * 32
            with (
                patch.object(
                    Path,
                    "read_bytes",
                    side_effect=[FileNotFoundError, winner],
                ),
                patch(
                    "yewo.agent.memory.replay_store.os.link",
                    side_effect=FileExistsError,
                ),
            ):
                self.assertEqual(replay_store.digest_key(path), winner)


class ScenarioDigestTests(unittest.TestCase):
    """A digest must not change because a declared value changed.

    The parts of a scenario are pydantic models, so they are serialised
    before they are hashed. That is where a value holding a quote stops
    matching its own spelling, which is the case worth a test.
    """

    def _scenario(self, password):
        from yewo.agent.device.launch import SetupMode
        from yewo.testing.evidence import ExpectedOutcome

        return SimpleNamespace(
            setup=SetupMode.CLEAR_DATA,
            steps=[f"Sign in as {EMAIL}"],
            expected_outcomes=[
                ExpectedOutcome(statement=f"The field holds {password}")
            ],
        )

    def _digest(self, password):
        return replay_store.scenario_digest(
            self._scenario(password), journey_values(password)
        )

    def test_a_plain_password_leaves_the_digest_alone(self):
        self.assertEqual(self._digest(PASSWORD), self._digest("another-one"))

    def test_a_password_holding_a_quote_leaves_it_alone_too(self):
        """JSON escapes the quote. A search of the printed form misses it,
        and the path becomes unreadable for every run but the one that
        wrote it."""
        self.assertEqual(self._digest('say "hi" now'), self._digest("plain"))

    def test_a_password_holding_a_backslash_leaves_it_alone(self):
        self.assertEqual(self._digest("back\\slash"), self._digest("plain"))

    def test_a_password_holding_a_newline_leaves_it_alone(self):
        self.assertEqual(self._digest("two\nlines"), self._digest("plain"))

    def test_changing_an_outcome_does_change_the_digest(self):
        """Or a path would be read back for a scenario it never walked."""
        from yewo.testing.evidence import ExpectedOutcome

        changed = SimpleNamespace(
            setup=self._scenario(PASSWORD).setup,
            steps=[f"Sign in as {EMAIL}"],
            expected_outcomes=[ExpectedOutcome(statement="Something else")],
        )

        self.assertNotEqual(
            self._digest(PASSWORD),
            replay_store.scenario_digest(changed, journey_values(PASSWORD)),
        )


class ActionBoundsTests(unittest.TestCase):
    """A file is read before it is trusted, so its bounds are the action's."""

    def test_a_tap_count_beyond_what_an_action_takes_is_refused(self):
        with self.assertRaises(ValidationError):
            replay_store.CachedAction(kind="tap", count=21)

    def test_an_interval_beyond_what_an_action_takes_is_refused(self):
        with self.assertRaises(ValidationError):
            replay_store.CachedAction(kind="tap", interval_ms=6000)


class TypedValueTests(unittest.TestCase):
    """A typed step keeps the name it was declared under, or is not kept."""

    def setUp(self):
        self.names = replay_store.ValueNames(journey_values())

    def test_a_declared_value_resolves_to_its_one_name(self):
        self.assertEqual(self.names.name_of(PASSWORD), "password")

    def test_a_model_placeholder_resolves_to_its_declared_name(self):
        self.assertEqual(self.names.name_of("$PASSWORD"), "password")

    def test_a_last_four_placeholder_resolves_to_its_declared_name(self):
        self.assertEqual(
            self.names.name_of(f"$PASSWORD[…{PASSWORD[-4:]}]"), "password"
        )

    def test_text_the_journey_never_declared_resolves_to_nothing(self):
        """Caching it would mean persisting the literal string."""
        self.assertIsNone(self.names.name_of("something typed by hand"))

    def test_a_value_declared_under_two_names_resolves_to_neither(self):
        """Resolving it later would be a guess between them."""
        names = replay_store.ValueNames({"first": "same", "second": "same"})

        self.assertIsNone(names.name_of("same"))

    def test_two_names_that_normalize_to_one_placeholder_are_ambiguous(self):
        names = replay_store.ValueNames(
            {"test-password": "first", "test_password": "second"}
        )

        self.assertIsNone(names.name_of("$TEST_PASSWORD"))

    def test_a_placeholder_that_is_also_a_literal_value_is_ambiguous(self):
        names = replay_store.ValueNames(
            {"literal": "$PASSWORD", "password": PASSWORD}
        )

        self.assertIsNone(names.name_of("$PASSWORD"))

    def test_a_value_under_two_names_is_not_substituted_either(self):
        names = replay_store.ValueNames({"first": "same", "second": "same"})

        self.assertEqual(names.as_names("same"), "same")

    def test_a_longer_value_is_named_before_a_shorter_one_inside_it(self):
        names = replay_store.ValueNames({"short": "pass", "long": "password123"})

        self.assertEqual(names.as_names("password123"), "{{long}}")

    def test_a_placeholder_type_action_keeps_learning_by_declared_name(self):
        password = element("Password", role="field")
        box = password.bounding_box
        step = TypeAction(
            text="$PASSWORD",
            coordinates=Coordinates(
                x=(box.x_min + box.x_max) // 2,
                y=(box.y_min + box.y_max) // 2,
            ),
            reason="Sign in",
        )
        trace = replay_walk.build_trace(
            replay_store.TraceIdentity(
                scenario_uuid="login",
                scenario_digest="digest",
                platform="android",
                target_profile="Pixel",
                detector="detector@revision",
                grounding="coordinates",
            ),
            [step],
            [(([password], scan()), ([element("Sign in")], scan()))],
            self.names,
        )

        self.assertIsNotNone(trace)
        self.assertEqual(trace.transitions[0].action.value_name, "password")
        written = trace.model_dump_json(by_alias=True)
        self.assertNotIn(PASSWORD, written)
        self.assertNotIn("$PASSWORD", written)


class BoundsTests(unittest.TestCase):
    """Bounds are a fraction of the screen, because pixels do not travel."""

    def test_a_box_becomes_thousandths_of_its_screen(self):
        bounds = replay_store.relative_bounds(
            BoundingBox(x_min=100, y_min=200, x_max=600, y_max=1200), 1000, 2000
        )

        self.assertEqual(
            (bounds.left, bounds.top, bounds.right, bounds.bottom),
            (100, 100, 600, 600),
        )

    def test_the_same_control_on_two_screen_sizes_lands_in_one_place(self):
        """Which is the whole reason the file holds a fraction."""
        small = replay_store.relative_bounds(
            BoundingBox(x_min=50, y_min=100, x_max=150, y_max=200), 500, 1000
        )
        large = replay_store.relative_bounds(
            BoundingBox(x_min=100, y_min=200, x_max=300, y_max=400), 1000, 2000
        )

        self.assertEqual(small, large)

    def test_a_screen_with_no_size_places_nothing(self):
        """Guessing a denominator would put every anchor in one corner,
        where each would match the others."""
        self.assertIsNone(
            replay_store.relative_bounds(
                BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1), 0, 0
            )
        )

    def test_a_screen_with_no_size_yields_no_anchors_at_all(self):
        anchors = replay_store.anchors_of(
            [element("Sign in")], scan(width=0, height=0), replay_store.ValueNames()
        )

        self.assertEqual(anchors, ())

    def test_a_small_move_is_near_and_a_large_one_is_not(self):
        here = replay_store.RelativeBounds(left=100, top=100, right=200, bottom=200)
        nudged = replay_store.RelativeBounds(left=105, top=100, right=205, bottom=200)
        elsewhere = replay_store.RelativeBounds(left=100, top=800, right=200, bottom=900)

        self.assertTrue(here.is_near(nudged, tolerance=10))
        self.assertFalse(here.is_near(elsewhere, tolerance=10))


class TwinTests(unittest.TestCase):
    """How many elements read alike is a fact about the screen."""

    def test_two_controls_reading_alike_both_say_so(self):
        anchors = replay_store.anchors_of(
            [element("Continue"), element("Continue", box=(10, 300, 110, 350))],
            scan(count=2),
            replay_store.ValueNames(),
        )

        self.assertEqual([anchor.twins for anchor in anchors], [2, 2])

    def test_a_control_alone_on_its_screen_has_one(self):
        anchors = replay_store.anchors_of(
            [element("Continue"), element("Cancel", box=(10, 300, 110, 350))],
            scan(count=2),
            replay_store.ValueNames(),
        )

        self.assertEqual([anchor.twins for anchor in anchors], [1, 1])

    def test_the_same_text_in_two_roles_is_not_a_twin(self):
        """A heading reading "Continue" is not the button of that name."""
        anchors = replay_store.anchors_of(
            [element("Continue"), element("Continue", role="text", box=(10, 300, 110, 350))],
            scan(count=2),
            replay_store.ValueNames(),
        )

        self.assertEqual([anchor.twins for anchor in anchors], [1, 1])


class ActionTests(unittest.TestCase):
    """A cached action holds no coordinate, because a control moves."""

    def test_the_model_has_nowhere_to_put_one(self):
        fields = set(replay_store.CachedAction.model_fields)

        self.assertNotIn("coordinates", fields)
        self.assertNotIn("text", fields)
        self.assertNotIn("reason", fields)

    def test_a_trace_names_its_schema_and_its_matcher_separately(self):
        """A stricter matcher must distrust a file an older one wrote, even
        where the shape of that file did not change."""
        written = replay_store.ReplayTrace(
            identity=replay_store.TraceIdentity(
                scenario_uuid="u", scenario_digest="d", platform="android",
                target_profile="Pixel_8",
                detector="d@abc", grounding="marks",
            )
        ).model_dump(by_alias=True)

        self.assertEqual(written["schema"], replay_store.SCHEMA)
        self.assertEqual(written["matcher"], replay_store.MATCHER)


if __name__ == "__main__":
    unittest.main()
