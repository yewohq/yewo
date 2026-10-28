"""Deciding whether a saved path still fits the screen in front of a run.

A saved path is only worth having if it is refused the moment it stops
being right. Acting on the wrong control produces a step nobody planned and
no judge saw, so every case here that is not certain has to end in a
refusal rather than a best guess.
"""

import unittest

from yewo.agent.memory import replay_match, replay_store
from yewo.agent.perception.elements import BoundingBox, GUIElement, ScanInfo

WIDTH, HEIGHT = 1000, 2000


def element(text, role="button", box=(100, 100, 300, 200)):
    left, top, right, bottom = box
    return GUIElement(
        element=text or role,
        bounding_box=BoundingBox(x_min=left, y_min=top, x_max=right, y_max=bottom),
        confidence=0.9,
        type=role,
        text=text,
    )


def scan(count, phash="p"):
    return ScanInfo(
        mode="detect",
        reason="step",
        phash=phash,
        screen_width=WIDTH,
        screen_height=HEIGHT,
        element_count=count,
    )


def anchors(elements, names=None):
    return replay_store.anchors_of(
        elements, scan(len(elements)), names or replay_store.ValueNames()
    )


def signature(elements):
    return replay_store.ScreenSignature(anchors=anchors(elements), element_count=len(elements))


class TargetTests(unittest.TestCase):
    """Finding the one element a transition touched."""

    def _find(self, saved, now):
        wanted = anchors(saved)[0]
        return replay_match.find_target(wanted, now, scan(len(now)), replay_store.ValueNames())

    def test_the_same_control_in_the_same_place_is_found(self):
        screen = [element("Sign in"), element("Email", "field", (100, 400, 300, 450))]

        found, refused = self._find(screen, screen)

        self.assertIsNone(refused)
        self.assertEqual(found.element.text, "Sign in")

    def test_a_control_that_moved_a_little_is_still_found(self):
        """A rebuild nudges a control by padding or a font change."""
        saved = [element("Sign in")]
        now = [element("Sign in", box=(110, 120, 310, 220))]

        found, refused = self._find(saved, now)

        self.assertIsNone(refused)
        self.assertEqual(found.element.bounding_box.x_min, 110)

    def test_the_action_would_use_the_new_place_not_the_old_one(self):
        """Which is the whole reason a coordinate is never stored."""
        saved = [element("Sign in")]
        now = [element("Sign in", box=(110, 120, 310, 220))]

        found, _ = self._find(saved, now)

        self.assertNotEqual(found.element.bounding_box.y_min, saved[0].bounding_box.y_min)

    def test_a_control_that_moved_across_the_screen_is_refused(self):
        saved = [element("Sign in")]
        now = [element("Sign in", box=(100, 1700, 300, 1800))]

        found, refused = self._find(saved, now)

        self.assertIsNone(found)
        self.assertIn("somewhere else", refused.reason)

    def test_a_control_that_is_gone_is_refused(self):
        found, refused = self._find([element("Sign in")], [element("Continue")])

        self.assertIsNone(found)
        self.assertIn("nothing on this screen", refused.reason)

    def test_a_control_that_grew_a_twin_is_refused(self):
        """A build that turned one Continue into two changed what the first
        one meant, and picking either would be a guess."""
        saved = [element("Continue")]
        now = [element("Continue"), element("Continue", box=(100, 900, 300, 1000))]

        found, refused = self._find(saved, now)

        self.assertIsNone(found)
        self.assertIn("one of", refused.reason)

    def test_the_same_words_in_another_role_are_not_the_control(self):
        saved = [element("Continue")]
        now = [element("Continue", role="text")]

        found, refused = self._find(saved, now)

        self.assertIsNone(found)
        self.assertIn("nothing on this screen", refused.reason)

    def test_a_screen_of_no_size_places_nothing(self):
        wanted = anchors([element("Sign in")])[0]

        found, refused = replay_match.find_target(
            wanted,
            [element("Sign in")],
            ScanInfo(
                mode="d", reason="s", phash="p", screen_width=0, screen_height=0, element_count=1
            ),
            replay_store.ValueNames(),
        )

        self.assertIsNone(found)
        self.assertIn("no size", refused.reason)


class ScreenTests(unittest.TestCase):
    """Deciding whether a transition still starts where it started."""

    def _agrees(self, saved, now):
        return replay_match.screen_agrees(
            signature(saved), now, scan(len(now)), replay_store.ValueNames()
        )

    def test_an_unchanged_screen_agrees(self):
        screen = [element("Sign in"), element("Email", "field", (100, 400, 300, 450))]

        self.assertIsNone(self._agrees(screen, screen))

    def test_a_changed_keyboard_state_is_refused(self):
        screen = [element("Email", "field")]
        expected = replay_store.ScreenSignature(
            anchors=anchors(screen),
            keyboard_visible=True,
        )
        current_scan = scan(1)
        current_scan.keyboard_effective_visible = False

        refused = replay_match.screen_agrees(
            expected,
            screen,
            current_scan,
            replay_store.ValueNames(),
        )

        self.assertEqual(refused.reason, "the keyboard visibility changed")

    def test_a_screen_that_grew_a_banner_still_agrees(self):
        """A build adds something. That is not a different screen."""
        saved = [element("Sign in"), element("Email", "field", (100, 400, 300, 450))]
        now = saved + [element("Offer", "text", (100, 50, 300, 90))]

        self.assertIsNone(self._agrees(saved, now))

    def test_a_screen_that_lost_most_of_itself_does_not_agree(self):
        saved = [
            element("Sign in"),
            element("Email", "field", (100, 400, 300, 450)),
            element("Password", "field", (100, 600, 300, 650)),
        ]
        now = [element("Something else", "text")]

        refused = self._agrees(saved, now)

        self.assertIsNotNone(refused)
        self.assertIn("expected elements", refused.reason)

    def test_a_trace_with_no_anchors_is_refused_rather_than_trusted(self):
        """An empty expectation would agree with every screen there is."""
        refused = replay_match.screen_agrees(
            replay_store.ScreenSignature(),
            [element("Anything")],
            scan(1),
            replay_store.ValueNames(),
        )

        self.assertIsNotNone(refused)
        self.assertIn("no anchors", refused.reason)


class DestinationTests(unittest.TestCase):
    """Proving the action arrived where it used to arrive."""

    def test_the_expected_destination_is_accepted(self):
        home = [element("Orders"), element("Account", box=(100, 400, 300, 450))]

        self.assertIsNone(
            replay_match.destination_reached(
                signature(home), home, scan(2), replay_store.ValueNames()
            )
        )

    def test_a_screen_that_lost_one_of_two_lookalikes_is_refused(self):
        """Counted, not a set. Two controls reading alike are two controls,
        and this is the last check a final cached step gets."""
        both = [element("Continue"), element("Continue", box=(100, 900, 300, 1000))]

        refused = replay_match.destination_reached(
            replay_store.ScreenSignature(
                anchors=replay_store.anchors_of(both, scan(2), replay_store.ValueNames()),
                element_count=2,
            ),
            [element("Continue")],
            scan(1),
            replay_store.ValueNames(),
        )

        self.assertIsNotNone(refused)

    def test_a_crowded_screen_losing_one_lookalike_is_still_refused(self):
        """The share would forgive it. The twin count must not: a build
        that removed one of two identical controls changed what the other
        one means."""
        expected = [
            element("Continue"),
            element("Continue", box=(100, 300, 300, 400)),
            element("Cancel", box=(100, 500, 300, 600)),
            element("Help", box=(100, 700, 300, 800)),
        ]
        now = [
            element("Continue"),
            element("Cancel", box=(100, 500, 300, 600)),
            element("Help", box=(100, 700, 300, 800)),
        ]

        refused = replay_match.destination_reached(
            signature(expected), now, scan(3), replay_store.ValueNames()
        )

        self.assertIsNotNone(refused)
        self.assertIn("one of 2", refused.reason)

    def test_a_detector_duplicate_may_drift_on_arrival(self):
        """YOLO can merge overlapping detections between adjacent readings."""
        expected = [
            element("CVV", "yolo_detection"),
            element("CVV", "yolo_detection", (100, 300, 300, 400)),
            element("Pay", "yolo_detection", (100, 500, 300, 600)),
            element("Total", "yolo_detection", (100, 700, 300, 800)),
        ]
        now = [expected[0], expected[2], expected[3]]

        refusal = replay_match.destination_reached(
            signature(expected), now, scan(3), replay_store.ValueNames()
        )

        self.assertIsNone(refusal)

    def test_a_detector_duplicate_still_refuses_a_source_screen(self):
        expected = [
            element("CVV", "yolo_detection"),
            element("CVV", "yolo_detection", (100, 300, 300, 400)),
            element("Pay", "yolo_detection", (100, 500, 300, 600)),
            element("Total", "yolo_detection", (100, 700, 300, 800)),
        ]

        refusal = replay_match.screen_agrees(
            signature(expected),
            [expected[0], expected[2], expected[3]],
            scan(3),
            replay_store.ValueNames(),
        )

        self.assertIsNotNone(refusal)
        self.assertIn("one of 2", refusal.reason)

    def test_a_screen_keeping_both_lookalikes_is_accepted(self):
        both = [element("Continue"), element("Continue", box=(100, 900, 300, 1000))]

        self.assertIsNone(
            replay_match.destination_reached(
                signature(both), both, scan(2), replay_store.ValueNames()
            )
        )

    def test_landing_somewhere_else_is_refused(self):
        """A unique source match only proves something plausible was tapped."""
        home = [element("Orders"), element("Account", box=(100, 400, 300, 450))]
        elsewhere = [element("Error", "text"), element("Retry", box=(100, 400, 300, 450))]

        refused = replay_match.destination_reached(
            signature(home), elsewhere, scan(2), replay_store.ValueNames()
        )

        self.assertIsNotNone(refused)

    def test_one_of_two_destination_markers_may_drift(self):
        basket = element("Basket")
        confirmed = element("Order confirmed", box=(100, 400, 300, 450))
        receipt = element("Receipt", box=(100, 600, 300, 650))
        expected_anchors = anchors([basket, confirmed, receipt])
        expected = replay_store.ScreenSignature(
            anchors=expected_anchors,
            element_count=3,
            distinguishing=tuple((anchor.role, anchor.text) for anchor in expected_anchors[1:]),
        )

        refusal = replay_match.destination_reached(
            expected,
            [basket, confirmed],
            scan(2),
            replay_store.ValueNames(),
        )

        self.assertIsNone(refusal)

    def test_less_than_half_of_destination_markers_is_refused(self):
        stable = [
            element("Basket"),
            element("Account", box=(100, 300, 300, 350)),
        ]
        markers = [
            element("Confirmed", box=(100, 500, 300, 550)),
            element("Receipt", box=(100, 700, 300, 750)),
            element("Ready", box=(100, 900, 300, 950)),
        ]
        expected_anchors = anchors(stable + markers)
        expected = replay_store.ScreenSignature(
            anchors=expected_anchors,
            element_count=5,
            distinguishing=tuple((anchor.role, anchor.text) for anchor in expected_anchors[2:]),
        )

        refusal = replay_match.destination_reached(
            expected,
            stable + markers[:1],
            scan(3),
            replay_store.ValueNames(),
        )

        self.assertIsNotNone(refusal)
        self.assertIn("1 of 3 destination markers", refusal.reason)


class SecretTests(unittest.TestCase):
    """Matching must not depend on a value the trace is forbidden to hold."""

    def test_a_field_showing_a_typed_value_matches_by_name(self):
        values = {"email": "qa.tester@example.com"}
        saved = anchors(
            [element("qa.tester@example.com", "field")],
            replay_store.ValueNames(values),
        )[0]

        found, refused = replay_match.find_target(
            saved,
            [element("qa.tester@example.com", "field")],
            scan(1),
            replay_store.ValueNames(values),
        )

        self.assertIsNone(refused)
        self.assertEqual(
            found.anchor.text,
            replay_store.text_identity("qa.tester@example.com", replay_store.ValueNames(values)),
        )

    def test_the_same_field_matches_after_the_value_changed(self):
        """A run with a different account walks the same path."""
        saved = anchors(
            [element("first@example.com", "field")],
            replay_store.ValueNames({"email": "first@example.com"}),
        )[0]

        found, refused = replay_match.find_target(
            saved,
            [element("second@example.com", "field")],
            scan(1),
            replay_store.ValueNames({"email": "second@example.com"}),
        )

        self.assertIsNone(refused)
        self.assertEqual(found.anchor.text, saved.text)


if __name__ == "__main__":
    unittest.main()
