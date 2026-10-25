"""A loop ends only while its repetition evidence still applies."""

import unittest

from yewo.agent.memory.loop_detector import RollingLoopDetector


class PageStagnationTests(unittest.TestCase):
    def test_a_navigation_clears_page_stagnation_termination(self):
        detector = RollingLoopDetector()

        for _ in range(8):
            detector.check(None, "payment")

        self.assertTrue(detector.terminated())

        detector.check(None, "checkout")

        self.assertFalse(detector.terminated())


if __name__ == "__main__":
    unittest.main()
