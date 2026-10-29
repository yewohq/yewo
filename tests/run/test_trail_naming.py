"""Planner-facing trail data is bounded and cannot name a current target."""

import hashlib
import unittest
from dataclasses import dataclass

from yewo.agent.memory.trail import SessionTrail


@dataclass
class _Box:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass
class _Element:
    element: str
    text: str
    bounding_box: _Box


def _at(kind: str, text: str, x: int, y: int) -> _Element:
    return _Element(element=kind, text=text, bounding_box=_Box(x - 20, y - 10, x + 20, y + 10))


DRAWER = [
    _at("text", "Trainer App", 137, 345),
    _at("text", "Edit Schedule", 154, 671),
    _at("icon_button", "", 72, 202),
]

CLOCK_SCREEN = [
    _at("text", "Trainings Schedule", 414, 207),
    _at("text", "6:41", 132, 65),
    _at("icon_button", "三", 72, 202),
]

PHASH = "daa82d505aa82d5058a225545822b04d2d57f2af2d57f2af2d57d2af2d57d2ab"
PHASH_JITTER = "daa82d505aa82d5058a225545823b00d2d57f2af2d57f2af2d57d2af2d57d2ab"
OTHER_PHASH = "0f0f0f0f0f0f0f0ff0f0f0f0f0f0f0f00f0f0f0f0f0f0f0ff0f0f0f0f0f0f0f0"


class TrailPromptTests(unittest.TestCase):
    def _prompt(self, elements, phash: str = PHASH) -> str:
        trail = SessionTrail()
        trail.record_screen(phash, elements)
        return trail.format_for_prompt() or ""

    def test_a_remembered_screen_is_only_counted(self):
        prompt = self._prompt(DRAWER)

        self.assertIn("1 distinct visual screen fingerprint", prompt)
        self.assertIn("1 observation", prompt)

    def test_a_second_screen_changes_only_the_counts(self):
        trail = SessionTrail()
        trail.record_screen(PHASH, DRAWER)
        trail.record_screen(OTHER_PHASH, CLOCK_SCREEN)
        prompt = trail.format_for_prompt() or ""

        self.assertIn("2 distinct visual screen fingerprint", prompt)
        self.assertIn("2 observation", prompt)

    def test_historical_targets_are_not_exposed(self):
        trail = SessionTrail()
        trail.record_screen(PHASH, DRAWER)
        trail.flush_transition(OTHER_PHASH)
        prompt = trail.format_for_prompt() or ""

        self.assertNotIn("Edit Schedule", prompt)
        self.assertNotIn("element 12", prompt)
        self.assertNotIn("154", prompt)
        self.assertNotIn("Navigation", prompt)

    def test_small_hash_jitter_is_counted_as_the_same_screen(self):
        trail = SessionTrail()
        trail.record_screen(PHASH, DRAWER)
        trail.record_screen(PHASH_JITTER, CLOCK_SCREEN)

        prompt = trail.format_for_prompt() or ""
        self.assertIn("1 distinct visual screen fingerprint", prompt)
        self.assertIn("2 observation", prompt)

    def test_ocr_and_model_descriptions_cannot_enter_the_summary(self):
        prompt = self._prompt(CLOCK_SCREEN)

        self.assertNotIn("6:41", prompt)
        self.assertNotIn("三", prompt)
        self.assertNotIn("Trainings Schedule", prompt)

    def test_repeated_reads_are_observations_not_visits(self):
        trail = SessionTrail()
        trail.record_screen(PHASH, DRAWER)
        trail.record_screen(PHASH, DRAWER)

        prompt = trail.format_for_prompt() or ""
        self.assertIn("2 observation", prompt)
        self.assertNotIn("visit", prompt)

    def test_the_summary_stays_bounded_as_the_session_grows(self):
        trail = SessionTrail()
        for number in range(100):
            phash = hashlib.sha256(str(number).encode()).hexdigest()
            trail.record_screen(phash, DRAWER)
            trail.flush_transition(phash)

        prompt = trail.format_for_prompt() or ""
        self.assertLess(len(prompt), 400)
        self.assertIn("not retained", prompt)

    def test_an_empty_trail_says_nothing(self):
        self.assertIsNone(SessionTrail().format_for_prompt())


if __name__ == "__main__":
    unittest.main()
